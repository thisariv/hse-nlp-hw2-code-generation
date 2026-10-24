---
type: reference
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-12
updated: 2026-10-12
sources:
  - kaggle-l1
  - kaggle-l1-review-smoke
aliases: []
---

# L1: независимое ревью надёжности

Ревью исходного коммита `bba9df5`, ветка `codex/nlp-hw2-code-generation`. Цель — сохранить обучение и результаты C/D в 11-часовой сессии. Боевой L1 не запускался и не отправлялся повторно; опубликованная L1 v1 содержит старый код. Проверяется исправленный локальный `notebooks/nlp-hw02-kaggle-l1.ipynb`.

## Находки и исправления

Ссылки ниже — на исправленный код; исходные строки относятся к `bba9df5`.

| Риск | Место и исправление | Регрессия |
|---|---|---|
| **Критично: D не помещается даже с micro64.** Исходный probe прекращался после 128/96/64 (`l1_runtime.py:63–80`). | [l1_runtime.py](../kaggle/src/l1_runtime.py), строки 16,85–111: цепочка 128→96→64→32→16→8→4→2→1, полный backward/Adam на длиннейших строках, запас 7%; batch128 и нормировка CE сохранены. CUDA v1 подтвердил OOM D на всех трёх старых вариантах. | Реальный T4×2 probe; C64 и D32 успешно. |
| **Поздний OOM обрывал worker.** Частичные scaled grads нельзя использовать повторно; Adam мог изменить лишь часть параметров. | `l1_runtime.py:52–81,266–282`: retry того же batch до шага Adam, очистка grads и внутреннего состояния scaler, восстановление scale/RNG; при OOM в Adam — восстановление committed full state и уменьшение micro, до двух попыток. Unsafe Adam OOM в ветке отбрасывает ветку и возвращает anchor. | `test_late_oom_discards_partial_grad_and_rng`, `test_adam_partial_oom_rolls_back_committed_state`, `test_branch_adam_partial_oom_restores_anchor`; дополнительная CUDA-инъекция во v2. |
| **Ошибка/timeout оценки не выполняли rollback.** Eval имела весь остаток сессии, а execution мог уйти в долгие retries/join. | `l1_runtime.py:236–260`: eval ≤900 с, incomplete при исключении, восстановление полного main anchor и следующий backward. [l1_eval.py](../kaggle/src/l1_eval.py):52–73: весь executor в ограниченном subprocess, kill группы и bounded wait. [l1_execute.py](../kaggle/src/l1_execute.py): смерть родителя завершает группу; Linux PDEATHSIG и отдельный parent guard. | `test_failed_eval_rolls_back_warmup_anchor_then_backward`, `test_executor_wall_timeout_and_actual_restore`, `test_executor_group_dies_with_parent`; реальная CUDA eval→restore→train. |
| **Cooldown повышал LR при anchor внутри warmup.** При resume из pending anchor отсутствовал свежий throughput и длина могла не уложиться в cap. | `l1_runtime.py:214–221`: охлаждение от сохранённого `last_lr` до нуля; без локальных logs использовать сохранённые main seconds / updates для cap. | `test_failed_eval_rolls_back_warmup_anchor_then_backward`, `test_resume_at_anchor_uses_historical_throughput_for_length`. |
| **CE не имела OOM fallback.** | `l1_eval.py:75–97`: batch16→8→4→2→1, lo/сумма/tokens изменяются лишь после успешного forward; FP32 softmax/sum. | `test_ce_late_oom_reduces_batch`. |
| **Resume выбирала самый новый receipt до проверки файла.** Повреждённый output заслонял исправный live. Identity проверяла лишь runtime, хотя вычисления зависят от core/order/train/representation. | [l1_resume.py](../kaggle/src/l1_resume.py): SHA/bytes/path до выбора максимальных updates, независимо для C/D; нет fresh restart при отсутствии состояния. `l1_runtime.py:133–170`: scheduler/LR/scaler-mode checks и semantic SHA пяти модулей. L2 не требует совпадения всего packaging bundle. | `test_output_and_live_resume_preserve_all_state`, `test_graceful_stop_publishes_state_and_identity_rejects_changes`; CUDA 256→288 updates. |
| **При сбое соседа/устаревшей live-страховке supervisor сразу убивал исправный worker.** | [l1_session.py](../kaggle/src/l1_session.py):64–73: stop-request, до120 с на штатное сохранение, затем bounded TERM/KILL. Worker проверяет request в main и cooldown. Final live timeout ограничен также оставшимся hard budget. | `test_graceful_stop_publishes_state_and_identity_rejects_changes`; настоящий watchdog smoke TERM-resistant group/hard exit. |
| **Live retry через300 с фактически обходился due/new каждые30 с.** | [l1_live.py](../kaggle/src/l1_live.py):166–181: после failure выдержать300 с; signature обновляется лишь при успехе. Непроверенная/не ready загрузка не продлевает страховку. | `test_live_failure_retries_after_300_seconds`; исходные private CLI mocks, secret rejection, staging/timeout tests. |
| **Выбор последнего common point по строке ломался при переходе к 7-значным time tags.** | `l1_session.py:35`: сортировать по числовому target main time, сохранить выбор C/D по последней общей точке. | `test_decision_uses_numeric_time_order`: t999000→t1008000. |
| **Длины приводились к uint16 без верхней границы; token ID могли быть отрицательными/вне vocab.** | [l1_cache.py](../kaggle/src/l1_cache.py):92–100: длины проверены до приведения, source≤2048, target+EOS≤2048, ID в допустимом диапазоне; verify=True до workers. | `test_cache_memmap_boundaries_and_overflow`; полный pinned cache принят на Kaggle. |

## Проверенные инварианты

- Training parameters остаются обычными FP32 tensors. Отдельная eval-модель создаётся вне inference_mode, переводится в FP16 под no_grad; inference tensors KV-сэмплера не попадают в training model. После eval создаются новые FP32 model/AdamW/scaler, затем загружается anchor. Ошибка S3 не воспроизвелась на CUDA.
- Loss — сумма CE по токенам всех micro-batches / число непустых target tokens всего effective batch128, включая EOS. Единственные unscale→clip1→scaler.step→scaler.update выполняются после полного накопления. CPU128 против96+32 проверяет loss и gradients; CUDA работает с64+64 и32×4.
- Full main state содержит FP32 model, AdamW moments/step/param groups, scaler/growth tracker, warmup counters/last LR, Python/NumPy/Torch CPU/все visible CUDA RNG, абсолютный cursor, main/branch counters, историю точек и identity. Ветка не увеличивает main cursor/updates/AMP skips. CPU continuous/resume даёт одинаковые tensors, Adam, scaler и RNG; output/live восстановлены через настоящие helpers.
- Warmup1000 attempted updates → постоянный peak LR между сессиями. Неявного main cooldown нет. Ветка использует данные после anchor cursor, возврат восстанавливает cursor и RNG; последний branch LR0. Deferred point не получает более поздний anchor под старой меткой: resume выбирает следующий общий grid. Неполная ветка/оценка не участвует в выборе C/D.
- Cache/order читаются mmap/read-only, offsets int64, token IDs≤8194 помещаются в int16. Доступ на границах shard/index проверен. Circular take сохраняет хвост эпохи; ровно все строки проходят до первого повтора. Фиксированная перестановка повторяется на следующей эпохе — это заявленный рецепт, новый shuffle не добавлялся. Полный order/SHA/биекция и 0/50040 compact round-trip подтверждены предыдущим CPU-аудитом.
- Compact mode tokens декодируются буквально и восстанавливаются тем же неизменённым `s3_representation`; CE считается в compact vocab. Генерируемая некорректная программа даёт restore/exec error, а не exception обучения.
- Боевые сроки сохранены: first cell→40680 с, main stop с резервом120 с, final live≤600 с, hard exit42300 с, backend43200 с. Eval теперь имеет собственный900-секундный cap. Watchdog вооружён до install/inputs, ждёт группы ограниченно и остаётся вооружён при unreaped PID. ZIP не создаётся.
- Диск: два full checkpoints на модель ≈3,78GB суммарно; четыре пары evaluated FP16 ≈1,26GB. Временный новый D full ≈1,25GB, anchor вне working удаляется после restore. Live full C+D + latest weights ≈2,20GB плюс небольшие journals, staging≤2,50GB; flat upload не дублирует весь snapshot ZIP. Cache output2,604GB обычно mounted read-only. Working guard10GiB оставляет запас относительно20GB; физический free-space guard также действует. Ротация протестирована. Массовые downloads/PT ignored.
- Secrets нужны только боевому gate и live helpers, worker env очищен от трёх auth variables; значения не записываются в config/outputs/logs. Audit working/staging охватывает binary; auth только env в пустом KAGGLE_CONFIG_DIR. CLI stdout/stderr discarded, stage/privacy/upload/snapshot/final waits ограничены. Реальная UI Secrets/private live publication этим smoke не проверяется: production gate обязан проверить её до обучения.

## GPU-smoke

Тот же `build_l1.py`, режим `l1_gpu_smoke`; отдельный private slug, InternetOn, NvidiaTeslaT4×2, pinned compact verified cache v2. Full C53,646,403 и D103,902,979, batch128, FP16 GradScaler1024, fused AdamW, wd0.01, clip1, warmup1000, seed42 сохранены. CPU tiny flag выключен: cap512, eval batch256/CE16; Secrets/live выключены только в этом режиме. Проверены production missing-Secrets gate и smoke gate без Secrets.

256 main updates; anchor192; одна19-update cooldown-ветка; heldout64/old16, greedy/restore/bounded exec/CE; test256 с новым таймером от чтения test и загрузки weights/tokenizer до закрытия answers; полный rollback; ещё64 main updates; новый model/Adam/scaler loader и32 resumed updates. Конечный cursor36864 =288×128, branches excluded; AMP skips0 во v1. Метка t009000 синтетическая для smoke, поэтому production common-time decision ожидаемо pending. Эти weights не использовать как L1→L2 input.

[v1](https://www.kaggle.com/code/thisarivson/nlp-hw02-l1-gpu-smoke/versions?versionNumber=1): COMPLETE, internal complete, exits[0,0],336,94 с. Все10 фаз прошли. C: micro64, peak reserved11,631GB=10,83GiB; D: micro32,9,292GB=8,65GiB. Greedy test256 full pipeline: C7,55 с, D12,36 с; batch256, OOM0. Downloaded104 небольших output files + log,6PT пропущены в диагностической v1; финальная v2 скачивается полностью. Peak относится к успешному probe и дальнейшим фазам, не к неуспешным OOM attempts.

[v2](https://www.kaggle.com/code/thisarivson/nlp-hw02-l1-gpu-smoke/versions?versionNumber=2): **COMPLETE/internal complete, exits[0,0],341,95 с**. Все фазы и288 main updates прошли, AMP skips0, cursor36864 обеих моделей. Пики и выбранные micro совпали с v1. Искусственный OOM после scaled backward: C64→32/D32→16 в disposable update; разница weights с чистым update **0**, scaler и CUDA RNG exact (заданный tolerance1e-6). Затем настоящий resume продолжил C64/D32. Test256 full pipeline **C7,51 с / D13,21 с**, greedy batch256, OOM0. Ни token-identity gate, ни Secrets для smoke не требовались. Финальная версия parent-death guard установлена реальным Linux executor на CUDA-сессии; смерть родителя воспроизведена отдельной CPU-регрессией. Две успешные итерации, суммарно678,89 с≈11,31 мин wall T4×2; третья не потребовалась. Полный output v2 скачан: **110 files + log**, 4,099,761,726 байт; **50 SHA/bytes** из manifest прошли. Служебный `__notebook__.ipynb` не выдаётся output API; source отдельно проверен pinned pull, остальные files не исключались. Все4 full PT: main256/288, Adam states/step, FP32 weights, scaler, scheduler/LR, CUDA RNG, cursor и history прошли; обе evaluated точки FP16/model-only. [Аудит](../checks/kaggle-l1-review-gpu-smoke.json), воспроизводимый `../kaggle/l1_review_audit.py`.

CPU: [16 регрессий](../checks/kaggle-l1-review-regressions.json), исходный двухпроцессный smoke и [реальный watchdog](../checks/kaggle-l1-watchdog-smoke.json) прошли. Версии, output и SHA — [runs.json](../kaggle/runs.json).

## Вердикт

**Готов к боевому запуску исправленный локальный ноутбук** после импорта в UI. Published L1 v1 не обновлялась и не должна запускаться со старым source; боевой Secrets/T4/private-ready gate остаётся обязательным. Полные GPU smoke доказали переходы/память/resume, а не качество после11ч.

## Некритичные замечания — без изменения боевого рецепта

- Kaggle service `__notebook__.ipynb` включён в working manifest, но недоступен через output API. Поэтому отдельный versioned source pull обязателен; binary/metrics SHA проверяются полностью.
- Common-time — wall time успешных main updates, включая collate/synchronization; это не счётчик GPU FLOPs. Branch update seconds/CE/test timers записаны, но полный раздельный accounting I/O и неуспешных OOM attempts отсутствует.
- Greedy256 не доказывает full1919 timer/качество после 11 ч. Нет CUDA token-identity gate: допустимые численные различия не должны останавливать сессию, как в S2.
- Межверсионная Torch/CUDA identity намеренно строгая: обновление Kaggle image потребует явного решения о миграции resume. Восстановление OOM в Adam может повторить main work после последнего1800-секундного checkpoint.
- Физическое повреждение mount, полный отказ GPU/CUDA D-state и потеря всех внешних сервисов не воспроизводились. Watchdog/live ограничивают ожидания и сохраняют доступные snapshots; real Secrets/live ready проверяет боевой startup gate.
- Исторические evaluated files остаются в pinned output/прошлых Dataset versions; full checkpoint переносит metrics, а не все предыдущие generations. Результаты live upload во время этой проверки не заявляются.

## Источники

- Исходный `bba9df5`, [L1 протокол](l1-long-training.md), [cache audit](../checks/kaggle-l1-cache-output.json), [S3 dtype failure](s3-representations-results.md); собственное независимое ревью и проверки выше.
- [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html): неизменный scale на effective batch, unscale перед clip и step после накопления.
- [Kaggle CLI kernels](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md): timeout и NvidiaTeslaT4×2; installed2.2.4 source дополнительно проверен для ready status.
- [Linux PR_SET_PDEATHSIG](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html): сигнал при смерти родительского thread; проверка race после установки и process-group cleanup.

Коммит не создан: `git add` и `git commit` заблокированы песочницей при создании shared worktree `index.lock` (`Operation not permitted`). Index не изменён; рабочие изменения и проверки сохранены.
