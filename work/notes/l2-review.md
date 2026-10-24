---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-13
updated: 2026-10-13
sources:
  - kaggle-l2-review
  - kaggle-l2-preparation
  - kaggle-l1
aliases: []
---

# ДЗ2: независимое ревью L2

Итоги последующего боевого UI v2: [l2-results](l2-results.md), оба worker complete/exits[0,0]. Ниже — история независимого ревью; CPU-разбор результатов не повторяет GPU-smoke аудит.

Ревью исходного `b387c06`, ветка `codex/nlp-hw2-code-generation`. Основной checkout не использован; рабочее дерево перед началом чистое. Цель — сохранение результатов и восстановимых состояний при отказе одного worker, а не проверка качества после 11 часов.

**Статус:** GPU-смоук выполняется; окончательный вердикт и боевой push будут записаны после проверки output.

## Находки и исправления

Номера строк относятся к исправленным файлам; исходное поведение воспроизводят новые регрессионные тесты в `../kaggle/l2_tests.py`.

| Риск | Исправление и проверка |
|---|---|
| Повторяющийся GradScaler `found_inf` мог бесконечно пропускать Adam без эскалации. | `src/l2_runtime.py:11,36–44,142,203`: одиночный overflow остаётся обычным skip/backoff; 32 подряд вызывают full rollback. Два понижения LR, третья эскалация — остановка только этой ветки. `test_persistent_AMP_skips_are_bounded_but_isolated_skips_are_normal`; прежний реальный enabled-scaler тест сохранён. |
| Проверялся только post-Adam model; повреждённые Adam moments могли оставаться до следующего checkpoint. | `src/l2_runtime.py:20–23`: сразу проверить также floating optimizer state; никакой nonfinite payload не заменяет stable receipt. `test_corrupt_Adam_moment_is_rejected_immediately`, прежние corruption/rotation тесты. |
| Время unsafe Adam/OOM попытки выпадало из оси main-времени. | `src/l2_runtime.py:126–135,152–153`: CUDA sync и учёт потраченного времени до отката; курсор/Adam возвращаются, время не уменьшается. `test_unsafe_attempt_time_survives_rollback`. |
| При продолжении L2 ранее записанные marks могли оцениваться повторно и перезаписывать journals/weights. | `src/l2_runtime.py:123`: пропуск уже записанных tags, включая явно неполные точки. `test_resume_does_not_overwrite_evaluated_points`. Новая сессия после полного +8ч требует новых time targets. |
| Malformed summary или exception финального live-sync могли оборвать упаковку. | `src/l2_session.py:51–88`: sanitised supervisor failure, unreadable summary как partial, отдельный catch final sync, затем output manifest. `test_unreadable_summary_and_failed_final_sync_still_package`, `test_supervisor_error_still_records_and_packages_outputs`; исключение/kill одного процесса не создаёт stop-request. |
| После kill оставался незавершённый full `.pt.tmp`, автоматически попадающий в output. | `src/l2_session.py:57–58`: удалить только interrupted checkpoint temporaries после reap всех workers. `test_killed_worker_does_not_stop_sibling_or_packaging` проверяет healthy finish, manifest и очистку. |
| Ротация checkpoint/atomic JSON между `is_file()` и `stat()` в старом disk guard могла остановить оба workers. | `src/l2_session.py:8–17,49`: пропустить исчезнувший файл, сохранить реальные size/free-space limits. `test_disk_guard_tolerates_rotation_between_is_file_and_stat` удаляет файл в середине проверки и отдельно проверяет превышение лимита. |
| Worker доверял SHA файла order, но явно не проверял dtype, границы и число слотов. | `src/l2_sampler.py:42,74–84`, `l2_worker.py:25–31`: SHA/bytes/rows, uint32/1D/nonempty, min/max и weighted slots до первого CUDA update. `test_order_SHA_bounds_integer_dtype_tail_and_large_cursor`: неправильный SHA/ID/dtype, хвост эпохи, большой int64 cursor. |
| Worker наследовал служебные Secrets/IAP JWT, позволяющие обращаться к attached Secrets. | `src/l2_session.py:26–27`, `src/l2_live.py:102–103`: удалить Secrets/IAP tokens из worker env и live-helper env; helpers используют только явно прикреплённые API credentials. `test_platform_and_API_credentials_are_removed_from_workers_and_helpers` с синтетическими значениями проверяет реальные subprocess env. |
| Исходный GPU-smoke сокращал длительности, но не проверял faults и настоящее продолжение. | `src/l2_gpu_smoke.py`, `build_l2.py`: реальный C anchor, CUDA numerical comparison с L1, longest-row probe, GradScaler overflow, NaN после частичного backward, production rollback/LR, две cooldown/eval/restore фазы, K8/greedy256, serialized resume, injected worker exit и ещё 64 healthy updates. Production training flags не заменяются CPU smoke flags. |

Значения пользовательских токенов не инспектируются; регрессия использует только синтетические строки.

## Проверенные инварианты

- **Strict C-main:** `l2_resume.py:9–28`, `l2_worker.py:22–31`. Нужен ровно файл `model-C__checkpoints__main-5746a4a2897e.pt`, SHA `1da28095…`, 47 155 attempted updates, cursor 6 035 840, main 17 569.935878398с. Полный AdamW/scaler/RNG; LR0.0015, warmup не повторяется. Пять semantic SHA L1 не менялись. Torch2.11.0+cu128/CUDA12.8 проверяются до install/cache; fallback на свежую модель отсутствует.
- **Fork:** `l2_resume.py:31–46`. Model/Adam/scaler/history/RNG наследуются; cursor0, отдельный cursor origin, новая identity/order SHA `72bc3dfd…`. Постоянный LR до явно записанной аварийной коррекции; cooldown LR не переносится в main.
- **Weighted order:** `build_l2.py:89–110`, `l2_sampler.py:40–71`. Строится в Kaggle на CPU из verified compact offsets; файл не вложен в notebook. Multiplicity ровно1/2, threshold79, 11 227 811 uint32 slots; bucket4096/batch128. SHA совпадает с подготовленным CPU order; `.npy` и метаданные сохраняются в output/live. Source/target lengths ≤2048 исключают int32 overflow; фактические L2 cursors далеко от int64 предела. `L.take` сохраняет хвост и продолжает цикл.
- **Skip/recovery:** `l2_runtime.py:15–44,144–169`. Rejected loss/нештатный norm → zero_grad, без Adam step, сброс per-optimizer scaler stage. Обычный found_inf → scaler.step skip + scaler.update/backoff. Full rollback восстанавливает model, optimizer, scaler, RNG, committed counters и cursor. Новое LR записывается и в state, и в optimizer groups, и в scheduler; `read_full` проверяет их равенство и Adam steps = updates−AMP skips−nonfinite skips. Recovery history/потраченное время сохраняются вне отката; бесконечного цикла нет.
- **Cooldown/eval:** `l2_runtime.py:184–237`, неизменённый `l1_eval.py:16–155`. Сначала full uncooled anchor, затем отдельные model-only FP16 weights. Eval под no_grad; inference tensors KV-кеша не попадают в training parameters. Restore создаёт новую обычную FP32 model/Adam/scaler вне inference_mode. Ошибка/timeout оценки также проходит finally/restore. CUDA продолжение после оценки входит в смоук.
- **Парность и offline vote:** `l2_compare.py:19–45`, `l2_eval.py:28–113`. Одинаковые marks по добавленному main-времени, cooldown targets, task IDs/question SHA и deterministic tune/eval split. Пары с расхождением actual main >5с исключаются. Greedy per-example содержит correctness и execution/reference; каждый K8 sample содержит code, исполненное значение/статус, restore status, длину, untempered mean LP с EOS. Journals публикуются атомарно после каждого исполненного batch; timeout сохраняет завершённые batches и явный incomplete. Mixed first4+first4 использует C-main greedy для tie, gold только после выбора.
- **Изоляция:** `l2_session.py:32–50,54–88`. Exception/OOM/NaN escalation/kill не создают общий stop-request. Healthy worker идёт до своего завершения. Общая остановка допускается только по deadline, устаревшей live insurance или общему диску; это общие ресурсные ограничения. Финальный report/manifest создаются и при partial; ZIP не дублирует веса.
- **Бюджет:** first cell→workers39600с, финальный sync≤1200с, hard watchdog41040с < backend43200с. Начальный setup≤900с; greedy/CE/test отдельный cap900с, final K8 cap900с внутри общего final eval1800с. Недостаточный резерв даёт incomplete point и сохраняет main anchor; отрезать обучение одной модели ради пары не требуется. Workers прекращают main с резервом120с. Watchdog вооружён до Torch/install/input scan; bounded TERM/KILL, parent-death guard executor, hard os._exit сохраняют доступный output.
- **Диск/live:** full ≈644MB, максимум два на ветку; шесть production cooled weights ≈644MB суммарно. Live хранит latest full обеих веток + latest cooled point/journals +45MB order, существенно ниже2.5GB. Working guard10GiB и free-space guard; live stage/privacy/upload/ready ограничены, retries300с, insurance7200с. Secrets обязательны только production; CLI stdout/stderr discarded; credentials не сериализуются. Race атомарного latest со snapshot обнаруживается по SHA/bytes и вызывает retry, не ложный успешный receipt.
- **Inputs:** metadata private/internet/T4, 26b-inputs Dataset v1, L1 live проверенный v7, compact kernel output `/2`. Mounted anchor/cache/order SHA перепроверяются независимо от metadata. Kaggle pull убирает номер kernel source в отображаемом metadata; request/config pin `/2` и content SHA остаются обязательными.

## Проверки и GPU

Локально: 34 L2 и 16 L1 регрессий прошли; compile/source parity/private metadata/budget checks прошли. CPU tiny не заменяет проверку CUDA. GPU версия, память, тайминги и SHA-аудит будут добавлены после завершения.

## Некритичное — без изменения рецепта

- Weighted long определяется compact длиной79; это ближайшая доля15.444% к heldout original≥90 доле15.283%, а не точное равенство исходных BPE длин.
- K8 tune2048 может не успеть полностью за900с; partial journals пригодны для диагностики, не для full2048 accuracy. Выбор рецепта и K8 eval/mixed single-T4≤180с требуют последующего inference опыта.
- Память micro64 известна из L1, production имеет поздний OOM fallback. Longest-row GPU probe отдельно проверяет её в ревью; автоматически увеличивать micro боевого C-main нельзя, это изменит накопление.
- Потеря работ после последнего30-минутного stable checkpoint при внезапном kill остаётся возможной; исторические points остаются в kernel output/раньше загруженных Dataset versions. Полная потеря GPU/mount/Kaggle storage не воспроизводится смоуком.
- Backend COMPLETE с injected exit[1,0] — ожидаемый результат смоука: только отдельный `cuda-smoke-summary.passed` доказывает, что падение было намеренным и healthy continuation/упаковка проверены.

## Источники

- Исходный `b387c06`, [l2-plan](l2-plan.md), [l1-review](l1-review.md), [l1-long-training](l1-long-training.md); собственные регрессии, pinned Kaggle outputs/source и [runs.json](../kaggle/runs.json).
- [PyTorch2.11 AMP examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html): accumulation/unscale/clip, found_inf skip и scaler backoff.
- [Kaggle CLI kernels](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [UserSecretsClient](https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_secrets.py), [KaggleWebClient](https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_web_client.py): T4×2/API/versioned source, attached Secrets и сетевой timeout.
