---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-15
updated: 2026-10-15
sources:
  - kaggle-r1-independent-review
  - kaggle-r1-preparation
  - kaggle-l3-final-results
  - nlp-hw02-d1-results
aliases: []
---

# ДЗ2: независимое ревью R1

**После завершения:** [результаты R1 на CPU](r1-results.md) — COMPLETE/exits[0,0], deep − контроль +0,15 п.п. [−0,86; +1,15], превосходство не установлено. Ниже — история ревью и запуска; её RUNNING-статус относится к раннему наблюдению.

Ревью исходного `0fe726c`, чистое дерево `codex/nlp-hw2-code-generation`.
Решение пользователя: исходный C-long2x weighted order **без D1-удалений**, q10%
стратифицированная у обеих веток. GPU0 — encoder6→12, GPU1 — encoder6;
один L3 latest anchor, одинаковый wall main9ч30мин и linear cooldown20мин.
При измеренном T4 deep/base>1,7 — encoder6→9 с вставкой после old1/3/5.
Private production — аккаунт друга; thisarivson используется только для smoke.

## Находки и регрессии

| Риск | Исправление / проверка |
|---|---|
| Блокирующий: builder разрешал только два прежних аккаунта, startup требовал Secrets; API production без Secrets останавливался до anchors. | `kaggle/build_r1.py:36`, `src/r1_live.py:35`: аккаунт друга, полный owner key включает live, отсутствие/частичный/чужой key оставляет compute и локальные stable checkpoints. `test_optional_secrets_missing_partial_wrong_owner_and_enabled`, `test_dad_build_is_private_optional_live_cross_owner_and_smoke_budget`. |
| Существенный для FP16: вставленные блоки переносились только на device, сохраняя FP32 даже у half-модели. Не было варианта6→9 для throughput gate. | `src/r1_growth.py:9`: dtype старых слоёв,9/12 и точные позиции вставок; старые Parameter objects и moments остаются. `test_nine_layers_insert_after_1_3_5_and_half_dtype`, прежние identity/Adam/next-update tests; CUDA AMP/FP16 gate в `src/r1_gpu_smoke.py`. |
| Существенный пробел GPU-gate:32шага, без реального identity/timing/serialized deep resume/v7. Smoke hard1020/backend1080 превышал требуемые15мин. | `build_r1.py` smoke —256main attempts, deadline780/hard840/backend900; `src/r1_gpu_smoke.py` — identity AMP/FP16, fused Adam/GradScaler overflow,32равных разнесённых батча после4warmup, serialized next-update exact, v7 half loader и cached greedy4. CPU contract сборки плюс реальные CUDA receipts. |
| Расписание не соответствовало новой сессии:50мин cooldown. | Defaults —34200s main/1200s linear/3600s eval, q0,1, старый LR0,0015/новый warmup1000; worker40260/hard40800. Budget/static tests. |

## Проверенные контракты

- `src/r1_growth.py:29–42,50–77`: нулевые out-projection/linear2 делают новый Pre-LN residual блок тождественным; нормы1/0. Старые parameter objects и Adam moments не копируются заново. Новая optimizer group имеет собственные lazy zero moments/step0; pre-hook задаёт warmup после scalar LR frozen P1. Ось warmup — attempted updates, AMP skips отдельно. Dropout0.
- `src/r1_runtime.py:31–77`: full state содержит `model_spec.encoder_layers`, optimizer names по группам, scaler/all RNG/state/cursor, group LRs и runtime SHA. `reload_latest` создаёт архитектуру по model_spec и группы по именам. CPU deep serialized next update exact; новый R1 формата full-v1 несовместим с намеренно L3-only `l3_resume.verify_anchor`: для продолжения использовать R1 loader, не подавать PT в L3 route.
- `src/r1_runtime.py:140–180`: последний full stable публикуется до cooldown, оценка получает отдельный FP16 model-only, после неё reload stable; cooldown cursor/RNG/Adam не попадают в latest. Неполный greedy/CE/test/K8 даёт partial. CPU failure-eval regression сохраняет stable и FP16 без complete receipt.
- `src/final_inference_v6.py:53`, `src/final_inference_v7.py:51`: обе load-функции создают архитектуру по model_spec и strict-load веса. CPU12layer logits проверены обеими; CUDA v7 отдельно. Для нового inference bundle нужны новые SHA весов; прежние test1919 timings на новые веса не переносятся.
- `src/r1_session.py:18–46`: отдельный CUDA_VISIBLE_DEVICES и working/model-GPU0/1, пустой credential config, API/Secrets/IAP vars удалены. Сбой соседа не создаёт stop-request. `test_r1_worker_failure_isolation_and_credential_env` вызывает настоящий supervisor с управляемыми subprocesses, проверяет env, выходы[1,0], manifest и продолжение здоровой ветки.
- `build_r1.py` startup + `src/l3_resume.py:8–13`: чужой notebook cache сначала `/kaggle/input/notebooks/thisarivson/<slug>`, чужой dataset `/datasets/<owner>/<slug>`, затем flat fallback. Anchor SHA/bytes/latest/counters/environment до install/cache; все360cache payload SHA до spawn. L3 mount regression включает nested/flat и два одинаковых slug.
- `src/r1_session.py:47–82`, `src/l1_session.py:10`, `src/session_watchdog.py`: watchdog armed до Torch/install, deadline40260/hard40800, независимые группы, bounded stop/final-sync, atomic latest и два full PT на модель. Checkpoint каждые900s; при kill сохраняется последний опубликованный stable, не обязательно последний update. Disk≤10GiB/free≥512MiB; dataset cache read-only, live stage≤2,5GB вне working. Короткий smoke не гарантирует отсутствие редкой11-часовой ошибки. Атомарность latest сохраняет предшественника при сорванной записи.

## CPU и API

15/15 R1 regressions и6static assemblies (3owners×2modes) прошли. L3 10/10,
P1 17/17 прошли. L2 33/34: один supervisor-error test заблокирован локальным
sandbox `os.killpg(...,0): PermissionError`; frozen runtime не менялся.
Отдельный R1 isolation regression не требует запрещённых OS signals.
[Квитанция](../checks/kaggle-r1-review-inherited-regressions.json).

Auth выбирается только KAGGLE_CONFIG_DIR (friend2/friend) либо default thisarivson.
Credential-каталоги и значения не инспектировались. Private live dataset
`friend2/nlp-hw02-r1-live` создан без удаления других ресурсов.
Квота до smoke: thisarivson5875,175s, friend2108000s; reset17.10 00:00UTC.
Доступ friend2 ко всем3private входам, включая pinned cache notebook outputv2,
подтверждён metadata API08:18:17UTC. Ранний403 кеша исчез после sharing.
Own thisarivson также проверил все3private источника.

## GPU smoke и production

[Private smokev1](https://www.kaggle.com/code/thisarivson/nlp-hw02-r1-gpu-smoke/versions?versionNumber=1)
**COMPLETE**, exits[0,0], session348,651s=5мин48,7с; backend лимит900s.
Одна итерация. Все92малых producer artifacts проверены по bytes/SHA;
94JSON/JSONL/log outputs скачаны, большие PT/cache/order не скачивались.
Anchor preflight27,93s; все360cache SHA и inputs78,56s.
[Независимый audit](../checks/kaggle-r1-review-gpu-smoke-v1.json).

| T4×2 | Base6+6, GPU1 | Deep12+6, GPU0 |
|---|---:|---:|
| Seconds/update,32разнесённых weighted batches после4warmup |0,350900|0,464102|
| Graph OOM fallback /AMP skips в benchmark |0/0|0/0|
| Benchmark allocated peak,GiB |8,689|11,461|
| Benchmark reserved peak,GiB |10,506|13,879|
| Общий session reserved peak,GiB |11,754|14,121|
| Main attempts /cooldown attempts |256/47|256/38|
| Full stable bytes |644209683|871629629|
| K8 rows/samples,mini cap8 |36/288|36/288|

Deep/base **1,322605×**, меньше порога1,7: **выбран encoder12**,9не нужен.
Батчи одинаковы, равномерно разнесены по исходному weighted order; q0,1,
effective128, micro64 у обеих. Это sampled throughput, не гарантия11-часового run.

- Logits до/после growth **bit-exact** в CUDA AMP и pure FP16, max abs0;
  старые Parameter objects/moments и все RNG сохранены.
- Injected GradScaler overflow пропустил Adam/model и снизил scale2097152→1048576;
  лениво созданные moments новых параметров нулевые. Full deep serialization →
  следующий настоящий CUDA q10% update совпал побитно, включая scaler/RNG.
- Обе реальные main→20s cooldown→greedy64/old16/test64→K8-36→stable reload
  завершились; latest хранит ровно256main updates, без cooldown cursor/weights.
- V7 strict-load по model_spec12/6, FP16 forward4 и настоящий cached greedy4cap8
  прошли. Мини-cap8 проверяет pipeline, **не качество** и не полный inference timer.

[Private productionv1](https://www.kaggle.com/code/friend2/nlp-hw02-kaggle-r1/versions?versionNumber=1)
отправлена через API friend208:21:21UTC, подтверждён **RUNNING08:21:43UTC**.
Source port `8d698577…bd053b` в точности совпадает с прошедшим smoke;
проверены private server metadata, обычный weighted order/q10% обеих,
main34200s/cooldown1200s/eval3600s/new warmup1000/hard40800s.
[Production gate](../checks/kaggle-r1-review-production-ready.json),
[metadata](../checks/kaggle-r1-review-production-metadata.json),
[status](../checks/kaggle-r1-review-production-status.json).
Ранний version-pinned API stream подтвердил **anchor-preflight PASSED** обеих
моделей: SHA/counters/cursors/main-time совпали с pins. Secrets отсутствуют,
`live-sync disabled; local stable checkpoints`; после полного cache preflight
оба workers запущены и уже сделали **100/200main updates** (deep/base),
CE0,17293/0,17700 finite. Полнота cache SHA здесь следует из проверенного порядка
кода до launch, live JSON output ещё не опубликован.
[Ранний startup audit](../checks/kaggle-r1-review-production-startup.json).
Ожидаемый конец≈18:20–18:40UTC; hard≈19:42UTC (22:42МСК),
отсчёт40800s от первой code cell. Длинную тренировку до завершения ревью не ждёт.

## Источники

[r1-plan](r1-plan.md), [p1-review](p1-review.md), [l3-review](l3-review.md), [d1-results](d1-results.md), исходный коммит0fe726c;
собственные CPU/API/CUDA receipts в `checks/kaggle-r1-review-*.json`,
малые remote output в ignored `results/kaggle-runs/nlp-hw02-r1-gpu-smoke/`.
Это собственные выводы ревью кода и запусков, не новое исследование качества.


## Коммит

Попытка явного R1-review-only `git add` получила exit128: sandbox запретил
`<vault>/.git/worktrees/study1/index.lock`, `Operation not permitted`.
Индекс не изменён; обходов ограничения не было. Файлы сохранены.
[Точный список файлов и сообщение коммита](../checks/kaggle-r1-review-git.json).
