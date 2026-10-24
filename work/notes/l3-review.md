---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-13
updated: 2026-10-13
sources:
  - kaggle-l2-results
  - kaggle-l2-review
aliases: []
---

# ДЗ2: независимое ревью L3 на аккаунте друга

Ревью коммитов `584320a`, `93ae3bc` в `codex/nlp-hw2-code-generation`. Файлы параллельного final-inference ревью не изменялись. CUDA smoke v1/v3 COMPLETE; v2 остановился до обучения на проверке fallback-mount. V3 прошёл с исправлением и полным независимым SHA-аудитом fallback. Боевой private L3 v1 опубликован, ожидаемый ERROR на отсутствующих Secrets подтверждён по статусу/логу; обучение не запускалось. **Вердикт: готов к UI Save & Run All с attached friend Secrets.**

## Находки и исправления

Номера строк относятся к версии после исправлений.

| Приоритет / риск | Исправление и регрессия |
|---|---|
| Существенный: `l2_runtime.py:212–238` сохраняет stable state после ошибки cooldown/eval, но может вернуть `complete`; неполный K8 тоже возвращает metrics. Такой отчёт ошибочно объявляет L3 завершённым. | `src/l3_resume.py:71–89`, `l2_worker.py:63–67`: L3 проверяет обе новые точки и полноту K8, пишет `partial` в worker summary; supervisor публикует partial-session. Frozen L2 runtime не меняется. `test_real_failed_eval_reports_partial_and_preserves_resumable_state` запускает настоящий tiny runtime с двумя ошибками eval и читает сохранённый full state; `test_incomplete_K8_or_missing_point_cannot_report_complete` проверяет K8/пропущенную точку/stability-stop. |
| Совпадающие cache-slugs делают общий `rglob(slug)` неоднозначным. В v2 ревью я сначала ошибочно предположил owner-qualified путь также для своего dataset; Kaggle подключил его в `/input/<slug>`, а чужой notebook — в `/input/notebooks/<owner>/<slug>`. | `src/l3_resume.py:8–13`, `build_l3.py`: сначала конкретный owner-qualified путь, затем flat mount; SHA проверяется после выбора. Тест `test_owner_qualified_mounts_disambiguate_original_and_fallback_cache` воспроизводит nested/nested, фактический nested/flat и старый single-flat layout. v2 не сделал training updates. |
| Не было отдельной квитанции фактически подключённых входов. | `build_l3.py`: после всех shard SHA записывает `input-preflight.json`; смоук дополнительно подключает private fallback и записывает `fallback-preflight.json`. Статика проверяет расположение квитанции после `verify=True` и до launch. |

Все **34 L2-регрессии** и **10 L3-регрессий** прошли повторно. Новый тест `test_worker_without_L3_kind_keeps_both_L2_routes` вызывает оба реальных маршрута `l2_worker.main` с контролируемыми входами: прежний anchor/fork, order SHA и phase recipe; L3 wrapper не вызывается. Статика/nbformat/compile/source parity/diff-check пройдены. CPU Torch отличается от pinned Kaggle Torch; CPU-тесты проверяют маршруты и инварианты, CUDA-проверка выполняется отдельно.

## Продолжение и сохранность

- `src/l3_resume.py:29–50`, `build_l3.py` startup: latest ↔ recover manifest ↔ pinned SHA/bytes/counters/identity, FP32 model и finite Adam, scheduler/LR, scaler, CUDA RNG, runtime/semantic SHA и Torch/CUDA проверяются **до install/cache**. V1 проверил оба настоящих PT за **24,14 с**, cache/order/setup — за **69,20 с** от первой ячейки.
- `l2_worker.py:22–40`, `src/l2_runtime.py:103–123`: обе ветки получают собственный полный L2 anchor. C-main cursor **16 185 856**, C-long2x **9 815 936**; LR **0,0015**, Adam/scaler/RNG/history сохраняются. Новый fork/сброс cursor/повторный warmup отсутствуют. `phase_recipe` добавляет 16 740/33 480 с к точному накопленному main каждой модели; прежние три точки остаются в history.
- `src/l3_resume.py` restore-order, `src/l2_sampler.py:74–84`: C-main SHA `09cf6db8…`, weighted SHA `72bc3dfd…`; inherited weighted metadata строго совпадает, order не строится заново. Проверяются uint32/размер/границы/slots.
- `src/l2_runtime.py:72–88,145–170,184–242`: stable checkpoint атомарно публикуется до ротации; остаются два full PT на модель. Перед disposable cooldown сохраняется полный uncooled anchor, после eval восстанавливаются FP32 model/Adam/scaler/RNG/cursor, затем продолжается train. NaN/AMP/post-Adam corruption имеют ограниченную эскалацию и отдельный worker-stop.
- `src/l2_session.py:19–88`: падение соседа не создаёт stop-request; summary/output manifest создаются также при ошибках supervisor/final-sync. Незавершённые `.pt.tmp` удаляются после reap. Disk guard терпит ротацию, лимит working 10 GiB, free ≥512 MiB. Обычный объём: до четырёх full PT ≈2,58 GB + два concurrent transient PT ≈1,29 GB + четыре FP16 ≈0,43 GB; cache в read-only input. Live stage ≤2,5 GB, вне working; CLI ZIP/stage требуют дополнительного свободного места.
- Watchdog — **`l1_session.Watchdog` через `l2_session`**, который заменяет старый короткий ceiling на 43 200 с. Armed до Torch/install. Workers 40 260 с, hard **40 800 с =11ч20мин**, server 43 200 с; final-sync получает оставшийся резерв, packaging защищён hard exit. Бюджет от первой ячейки включает main 33 480, cooldown 900/3000, eval 300/1800, setup 480 и I/O/stop reserve. Два hard-wall L3/L4 =22ч40мин; фактические короткие смоуки оставляют запас до пользовательских ≈25 ч.

## Аккаунт, входы, live и Secrets

Все API/CLI операции этого ревью используют `export KAGGLE_CONFIG_DIR=~/.kaggle-friend`; CLI **2.2.4**. Каталоги `~/.kaggle-friend` / `~/.kaggle` и значения credentials не инспектировались. Private friend live найден через **mine** enumeration, v1 ready/private. Ничего на Kaggle не удалялось.

V1 реально подключил `dataset_sources`: `thisarivson/nlp-hw02-26b-inputs`, `thisarivson/nlp-hw02-l2-live`; `kernel_sources`: **`thisarivson/nlp-hw02-compact-cache-verified/2`**. Обе anchors, исходный weighted order, tokenizer/heldout/old/test и **все 360 shard payloads** original cache прошли проверку в ноутбуке аккаунт друга; rows **9 725 737**, shards **60**, manifest SHA **`46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be`**.

Fallback `friend1/nlp-hw02-compact-cache-verified`, private v1: remote manifest скачан с тем же SHA; все 360 required files видны с правильными sizes, 428 remote files всего. Это отдельно **не доказывает SHA payloads**. Whole-ZIP endpoint вернул 404; медленная пофайловая CPU-загрузка остановлена, её temporaries удалены. **V3 независимо проверил SHA всех 360 shard payloads** fallback, ordinary order и tokenizer в реально подключённом flat read-only mount. Manifest точный, 60 shards /9 725 737 rows, roundtrip receipt50 040/0 mismatches; это собственная проверка ревью, а не квитанция автора кода.

Если original notebook output перестанет подключаться, явно собрать обе версии через `python3 build_l3.py [--gpu-smoke] --cache-source friend1/nlp-hw02-compact-cache-verified`. Metadata убирает kernel source и добавляет fallback dataset; runtime требует тот же manifest/order/anchors. При ошибке доступа или SHA обучение не начинается.

Боевой gate требует attached **KAGGLE_USERNAME/KAGGLE_KEY аккаунта друга и две T4; smoke требует T4×2, Secrets/live не нужны (`build_l3.py` startup; `src/l2_live.py:10–34`). Username проверяется перед live helper. Private owner startup upload + ready обязательны; workers удаляют API/Secrets/IAP credentials (`l2_session.py:24–27`), helper использует пустой config и attached key, исключает API token/Secrets/IAP JWT (`s2_live_sync.py:153–158`, `l2_live.py:102–103`). Источники и staging проходят credential scan; env/key не сериализуются.

Live heartbeat 5400 с, retries ≥300 с, stale insurance 7200 с; timeout → штатный stop обоих для сохранения states. Snapshot включает latest обеих C, последний cooled point/journals и weighted order; SHA receipt проверяется после copy. Ротация во время snapshot может отменить upload, но не публикует ложный recover receipt; следующий retry ограничен временем.

## GPU-смоук

[Private notebook](https://www.kaggle.com/code/friend1/nlp-hw02-l3-gpu-smoke), v1 COMPLETE, wall **617,58 с**. Проверены все скачанные JSON/log-файлы по producer output manifest, крупные PT/order не скачивались повторно: full CUDA loader и serialization probe уже проверили их на Kaggle. Output скачивается через `l3_review_api.py` с явным version label и всеми страницами; CLI latest может смениться между страницами при следующем push.

| V1 | C-main | C-long2x |
|---|---:|---:|
| Main updates |819|759 +64 после crash соседа|
| Cooldown → eval → restore |2 ×20 с|2 ×20 с|
| K8 |36 rows /288 samples|36 rows /288 samples|
| K8 full evaluation, с |11,15|26,43|
| Greedy256 full pipeline, с |2,95 /15,88|7,55 /2,98|
| Training/eval allocated peak, GiB |8,28|6,36|
| Training/eval reserved peak, GiB |9,58|9,26|
| Successful longest-row probe reserved, GiB |10,83|10,83|

Finite reference update и serialized restore → следующий CUDA update bit-exact; real FP16/fused Adam/GradScaler, effective128/micro64. Probe128/96 получил ожидаемый OOM, затем64 прошёл. Found-inf backoff не изменил model/Adam; NaN после partial backward не изменил numerical state/не вызвал Adam. Три injected NaN вызвали main rollback и LR0,00075 только у C-main; long сохранил0,0015. Worker0 intentionally exited1 **после** основных диагностик; worker1 сделал ещё64 updates и вышел0; stop-request отсутствует, packaging выполнен, stable cursors корректны. Это проверка устойчивости, не оценка качества L3 и не production LR recipe.

V2 ERROR: за ≈75 с выяснил mixed flat/nested mount; оба anchors и original cache были проверены, train не запускался. V3 **COMPLETE**, wall **602,81 с**, последняя разрешённая итерация. Cache-mount исправлен; original и fallback полные по SHA. C-main **788** main updates; C-long2x **762 +64** после crash. Обе cooldown/eval/restore-фазы, finite/serialized CUDA parity, GradScaler, NaN/rollback/LR, K8 и greedy256 повторно прошли. Training/eval allocated **8,26/6,37 GiB**, reserved **9,48/8,55 GiB**; longest-row successful probe **10,83 GiB** у обеих. [V3 audit](../checks/kaggle-l3-review-gpu-smoke-v3.json), [server private/source metadata](../checks/kaggle-l3-review-smoke-metadata.json).

## Боевой push и вердикт

[Private L3 v1](https://www.kaggle.com/code/friend1/nlp-hw02-kaggle-l3/versions?versionNumber=1) отправлен через API от аккаунта друга с timeout43 200/T4×2. Первый transport оборвался BrokenPipe до появления версии; один обычный повтор создал v1. Статус **ERROR — ожидаемый отказ Secrets**, log и `startup-gate.json`: `Kaggle Secrets unavailable: KAGGLE_USERNAME, KAGGLE_KEY`. Anchor load/train/worker spawn не начинались. Server private=true, ownerfriend1, тот же source port SHA, что в прошедшем v3; production требует Secrets, smoke — нет. [Gate receipt](../checks/kaggle-l3-review-production-gate.json), [metadata](../checks/kaggle-l3-review-production-metadata.json).

**Готов к запуску из UI аккаунта друга:** прикрепить оба friend Secrets, Edit → Save Version → Save & Run All. Полную11-часовую тренировку ревью не запускало; live upload/ready/insurance с реальными Secrets проверит её обязательный startup probe. Три GPU-smoke итерации использованы, дополнительный smoke не отправлять без нового разрешения.

Сохранены только маленькие JSON/log output в ignored `work/results/kaggle-runs/nlp-hw02-l3-gpu-smoke/{1,2,3}` (≈13MB); большие PT/order не скачивались. Временные cache/fallback загрузки и scratch удалены. V3 проверил90 output files по SHA. Краткие receipts и исправления коммитятся, исходные journals — ignored. Сетевые connect/transfer ошибки были временными; авторизация проверялась обычным API, обходов не было.

## Некритичные риски и пределы проверки

- 15-минутный smoke не доказывает отсутствие редкого зависания/исчерпания диска/сети за11 ч. Hard watchdog и stable/live recovery ограничивают последствия. Реальные attached friend Secrets, upload throughput и insurance после старта проверит UI run; smoke сознательно не делает live upload.
- Пины требуют ровно Torch2.11.0+cu128/CUDA12.8 и исходный L2 latest. Обновление Kaggle image/source Dataset должно остановить preflight; не отключать проверки ради запуска.
- Scalar peak в legacy smoke сбрасывается после probes: отдельно показаны training/eval peak и успешный longest-row probe. Failed128/96 OOM не имеют точного peak receipt.
- Setup480/полные greedy+K8 caps — бюджет, не гарантия throughput. При нехватке времени сохраняется stable state и новый partial status. Saved incomplete tags runtime не переоценивает автоматически; следующий протокол обязан задать новые targets.
- L4 требует **новых pinned anchors и накопленных targets/history**, а не повторного build L3 на L2 anchors. Формат stable state остаётся совместимым, но текущий `verify_anchor` намеренно ожидает три L2 points.
- Свежая оценка качества/fresh1024 и финальный инференс относятся к другим задачам.

## Источники

- [l3-plan](l3-plan.md), [l2-review](l2-review.md), [l2-results](l2-results.md); пины [l3-anchors.json](../kaggle/l3-anchors.json), собственные CPU/GPU receipts, [runs.json](../kaggle/runs.json).
- [L2-регрессии](../checks/kaggle-l2-regressions.json), [L3-регрессии](../checks/kaggle-l3-regressions.json), [статика](../checks/kaggle-l3-static.json), [v1 audit](../checks/kaggle-l3-review-gpu-smoke-v1.json), [fallback metadata](../checks/kaggle-l3-review-fallback-access.json).
- [Pinned API downloader](../kaggle/l3_review_api.py), [output audit](../kaggle/l3_review_output.py); `src/l3_resume.py`, `l2_worker.py` и неизменённые frozen L2 runtime/live/session modules. Выводы ревью — собственная интерпретация кода и запусков.

## Коммит

Попытка `git add` явного L3-only списка заблокирована песочницей: shared Git worktree `index.lock`, `Operation not permitted`. Этот вызов не изменил index; чужие final-inference файлы не staging. Ревью/публикация завершены, изменения сохранены; ограничение не обходилось.

Из обычного терминала в этом worktree можно повторить (перед коммитом проверить дифф общих файлов):

```bash
l3_work=work
l3_paths=(
  README.md
  courses/nlp/index.md courses/nlp/progress.md
  "$l3_work/notes/l3-review.md" "$l3_work/notes/l3-plan.md"
  "$l3_work/kaggle/README.md" "$l3_work/kaggle/runs.json"
  "$l3_work/kaggle/build_l3.py" "$l3_work/kaggle/l2_worker.py"
  "$l3_work/kaggle/l3_tests.py" "$l3_work/kaggle/l3_static_check.py"
  "$l3_work/kaggle/l3_review_api.py" "$l3_work/kaggle/l3_review_output.py"
  "$l3_work/kaggle/src/l3_resume.py"
  "$l3_work/kaggle/notebooks/nlp-hw02-kaggle-l3.ipynb"
  "$l3_work/kaggle/notebooks/nlp-hw02-l3-gpu-smoke.ipynb"
  "$l3_work/checks/kaggle-l3-regressions.json" "$l3_work/checks/kaggle-l3-static.json"
  "$l3_work/checks"/kaggle-l3-review-*.json
)
git add -- "${l3_paths[@]}"
git commit --only -m 'nlp: independently review L3 and verify friend CUDA smoke' -- "${l3_paths[@]}"
```
