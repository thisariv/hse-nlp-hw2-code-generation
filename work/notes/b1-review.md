---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-15
updated: 2026-10-15
sources:
  - kaggle-b1-independent-review
  - kaggle-b1-preparation
  - nlp-hw02-architecture-decision
  - kaggle-r1-independent-review
  - nlp-hw02-overfit-check
aliases: []
---

# ДЗ2: независимое ревью и запуск B1

Проверена подготовка `91188a2` на ветке `codex/nlp-hw2-code-generation`, текущий HEAD `076eff3`. Прочитаны [b1-plan](b1-plan.md), [architecture-decision](architecture-decision.md), [r1-review](r1-review.md), [l3-review](l3-review.md), [overfit-check](overfit-check.md). В начале были чужие незакоммиченные изменения R1/навигации/библиографии: их строки сохранены, в коммит ревью не включаются.

Решение пользователя: обе модели fresh; GPU0 **e8-d4-w768/108 694 019**, GPU1 **e6-d6-w768/113 421 827**, heads12/SwiGLU2048. C из L1, peakLR0,001; q0/dropout0, AdamW wd0,01, clip1, AMP, batch128, normalized init, warmup1000→constant. C-long2x threshold79/×2, новый order seed2026101501, bucket4096. Точки main9000/28370/34200с, каждая900с cooldown + greedy heldout4k/old991 + CE, затем полный restore. Ориентиры C62,87%/67,50%; quality gate проверяется по этим receipts между сессиями.

## Находки, исправления и регрессии

Номера строк — после исправлений; пути относительно `work/`.

| Приоритет / риск | Результат и подтверждение |
|---|---|
| Существенный: старая GPU0 6+8/d640, нет ограничений micro по реальным формам. | `kaggle/src/b1_runtime.py:18–39`:8+4/768 и6+6/768, caps64/32. Counts совпали с реальными CPU-моделями и `shape_specs`:108694019/113421827. Normalized token init ровно C; при restore повторного деления нет (`:150–155`). `test_requested_shapes_match_measured_specs_and_micro_caps`, `test_real_parameter_counts_and_normalized_init`, прежний v7 обеих полных форм. |
| Критичный memory gate: inherited probe мог получить AMP skip и принять отсутствие lazy Adam allocation за безопасную память. | `kaggle/src/b1_runtime.py:43–80`: probe scale1, обязательные `AMP_skip==0` и Adam step1, реальные максимальные source/target lengths, headroom93%, возврат исходных model/RNG. Production scaler по-прежнему1024. `test_probe_allocates_adam_and_restores_model_rng` проверяет реальный Adam, reset и отказ при injected skip. CUDA отдельно проверяет overflow без изменения model/moments и уменьшение scale. |
| Существенный: неверные отметки19800/28800с; обычный B1 не оценивал конец main; короткий cooldown был около12,75мин, old991 отсутствовал. | `kaggle/src/b1_runtime.py:22,191–205,226–275,306–317`, `kaggle/src/b1_eval.py:13–31`: ровно9000/28370/конец, target и actual main, timed linear900с как C, q0; оба split с greedy и CE. `test_C_marks_end_main_timed_q0_and_restore`, `test_intermediate_evaluator_runs_both_splits`. Долгий final/K8 не включён в B1. |
| Критичный отчёт: неудачный probe или ранний deadline мог оставить `complete`. | `kaggle/src/b1_runtime.py:191–201,310–312`: проверяются denominators обоих split и finite CE; пропущенные/неполные оценки или недостигнутый main дают partial. `test_incomplete_old991_or_deadline_cannot_report_complete`, прежний failed-eval restore test. |
| Watchdog / продолжение: forced exit может случиться до aggregate output manifest, а transfer требовал именно его. | `kaggle/b1_transfer.py:94–132`: явный `--emergency-receipts` использует pinned version, atomic latest SHA/bytes и order.json, сверяет cache/order обеих моделей; PT и remote Dataset round-trip SHA остаются обязательными. `test_watchdog_output_transfers_without_aggregate_manifest` восстанавливает обе модели без manifest и отклоняет неверный order до upload. |
| Бюджет смоука был20мин, больше разрешённых15мин. | `kaggle/build_b1.py:41–44`: worker780/hard840/backend900с. Статика проверяет точные числа у3owners×2modes. Обычный B1:main34200,worker40260,hard/backend40800; расчёт39420с с3×(900+600), setup480 и save reserve240. |
| CUDA-smoke v1: проверка overflow сравнивала весь optimizer, хотя warmup штатно меняет group LR даже на skipped update. Обе формы остановились на этом assertion после успешных256steps/serialized resume/16peak LR. | `kaggle/src/b1_gpu_smoke.py:75–85`: сравниваются model и optimizer moments/steps, отдельно проверяются skip и scale/2; LR записывается как ожидаемое изменение. `test_warmup_AMP_skip_changes_group_lr_without_changing_weights_moments` воспроизводит настоящий CPU FP16 GradScaler overflow и новый LR. Это исправление теста; production math не меняется. |
| Входы и диагностика API. | `kaggle/build_b1.py:85–106`: owner-qualified notebook/dataset mount → cache manifest + все payload SHA → spawn, реальный input-preflight. `kaggle/b1_review_api.py:21–33`: только friend2 config, inherited auth env удаляется до импорта SDK; private input API receipt. Worker печатает traceback при ошибке. Статика проверяет pinned cache/metadata/Secrets-free gate, CUDA запускает реальный preflight. |

## Resume и выходные файлы

`kaggle/src/b1_runtime.py:113–156`: SHA/bytes, model_spec/identity/data/order/recipe/runtime/Torch/CUDA, finite model+Adam, cursor=128×attempts, optimizer steps=attempts−AMP−nonfinite, LR factor. State хранит scaler и Python/NumPy/Torch CPU/visible CUDA RNG. CPU next-update с dropout0,2 побитно, включая optimizer/RNG; strict loader отклоняет изменения recipe/cursor. CUDA next-update и оценка→restore→12 updates проверяются независимо от no-eval control. Секунды таймера и имена файлов UUID не являются побитно воспроизводимыми; вычислительное состояние — предмет проверки. После обновления Kaggle image resume останавливается по environment mismatch.

`kaggle/src/b1_session.py:18–43,47–82`, `kaggle/src/l1_session.py:10–13`, `kaggle/src/session_watchdog.py:119–161`: CUDA_VISIBLE_DEVICES по одному T4, пустые worker credentials, watchdog armed до Torch/install, periodic saves900с, atomic latest до ротации двух full PT. Один worker не останавливает здорового. На timeout сохраняется последний опубликованный stable, возможна потеря последних≈15мин; forced exit/авария платформы не гарантируют final output upload. Это остаточный инфраструктурный риск без live-sync, а не обещание exactly-once завершения.

Диск: полные состояния≈12N байт; два поколения обеих моделей≈5,33GB, три FP16 точки≈1,33GB, два concurrent transient≈2,67GB; order/journals дают пик около9,4GB, guard10GiB/free512MiB. Кеш и staging вне working, ZIP в output не создаётся. Transfer загружает только последние full обеих моделей+order≈2,7GB, большой PT скачивается потоково; после private ready staging удаляется, remote SHA проверяется по одному файлу. GPU receipt фиксирует фактический объём.

## CPU и API

[20 CPU regressions](../checks/b1-cpu.json) прошли; [6 static assemblies](../checks/b1-static.json) прошли. Frozen C/L1/L2/P1 runtime не менялся. Проверки запускают настоящие tiny forward/backward/Adam/restore, обе полные CPU формы и v7 loads; долгого обучения локально нет.

[Private inputs](../checks/b1-review-inputs.json): API friend2 подтвердил private dataset `thisarivson/nlp-hw02-26b-inputs` и notebook cache v2. [Quota](../checks/b1-review-quota.json):free72175,092с≈20,05ч, reserved0; reset17.10 00:00UTC. Это session quota T4×2, не удвоенная длительность main. Все операции текущего ревью — `KAGGLE_CONFIG_DIR=~/.kaggle-friend2`; каталоги и значения credentials не инспектировались. Ресурсы Kaggle не удалялись; другой checkout не менялся.

## GPU smoke и production

[Private smoke v1](https://www.kaggle.com/code/friend2/nlp-hw02-b1-gpu-smoke/versions?versionNumber=1): ERROR после324,282с, exits[1,1], на ошибочном overflow assertion из таблицы. [Failure audit](../checks/b1-review-smoke-v1-failure.json) проверил все доступные15малых файлов по producer SHA; 256main attempts без AMP/nonfinite, CUDA serialized next-update exact и16peakLR attempts без skips у обеих. Weighted221,01/193,58пар/с, повтор исходных shape batches267,04/234,45 против246,7/229,4 (+8,24%/+2,20%). Это разные length distributions; weighted throughput не обязан равняться synthetic. Реальные caps64/32 совпали с shape-bench.

[Private smoke v2](https://www.kaggle.com/code/friend2/nlp-hw02-b1-gpu-smoke/versions?versionNumber=2) **COMPLETE**, exits[0,0], wall**464,177с=7мин44,2с**, backend900с. [Независимый audit](../checks/b1-review-gpu-smoke-v2.json): **64малых файла по SHA**,76producer entries, output**6 711 342 239байт≈6,71GB**, большие PT не скачивались. Их сериализация/SHA/finite/next-update проверены на T4 обоими workers. Две итерации суммарно788,459с выполнения; quota delta852,356с=14мин12,4с, до15мин. Третьей итерации не было.

| Метрика | GPU0 e8-d4-w768 | GPU1 e6-d6-w768 |
|---|---:|---:|
| Micro /effective |64 /128|32 /128|
| Worst-row probe allocated /reserved,GiB |12,294 /13,395|7,512 /8,199|
| Weighted benchmark allocated /reserved,GiB |9,755 /11,803|6,400 /7,234|
| Weighted order,32батча после4warmup,пар/с |229,10|207,99|
| Исходные8synthetic batches,пар/с |272,38|248,05|
| Отношение к shape-bench246,7/229,4 |1,104×|1,081×|
| Main attempts /peakLR attempts |256 /16|256 /16|
| Обычные AMP /nonfinite skips |0 /0|0 /0|
| Последний full stable,байт |1 304 589 827|1 361 337 747|
| Final stable attempts /cursor |269 /34 432|269 /34 432|

- Serialized следующий CUDA update — bit-exact для model/Adam/scaler/all RNG у обеих.
- Injected overflow — AMP skip, scale1024→512, model/moments/steps не меняются; groupLR штатно0,000256→0,000257 на warmup. После теста возвращён исходный stable.
- Настоящие mini greedy64+old16 и CE обоих split, два3-секундных q0 cooldown: промежуточный и end-main. После первой оценки следующие12updates совпали побитно с no-eval control. Обе complete receipts и последний strict-read stable проверены.
- V7 FP16 load и реальный greedy32/cap128:0,597/0,879с. Это проверка pipeline на ранних весах, **не quality и не cold1919≤180с**.
- Input preflight подтвердил `/kaggle/input/notebooks/thisarivson/nlp-hw02-compact-cache-verified` и `/kaggle/input/datasets/thisarivson/nlp-hw02-26b-inputs`. Order11 227 811slots, SHA`8e9595f655ed2b4456bf5b94e9b5b6b3db040e3f5696a4dc0ef364588048bee8`, seed2026101501.

**Боевой B1 запущен:** [private production v1](https://www.kaggle.com/code/friend2/nlp-hw02-kaggle-b1/versions?versionNumber=1), API push15.10 **20:23:51UTC**, **RUNNING20:24:21UTC**. [Gate](../checks/b1-review-production-ready.json), [status](../checks/b1-review-production-status.json), [server metadata](../checks/b1-review-production-metadata.json). Все87module SHA совпадают со smokev2, port`9c9703751da827b6def744f049be06acf2b87bb655aea99a2d96199ea647c268`. Свободная квота после смоуков71 322,736с≈19,81ч, reserved0, достаточно40800с; [receipt](../checks/b1-review-quota-after-smokes.json).

Production: fresh, dropout0, q0 main и всех коротких веток; main34200, cooldown900, eval reserve600; workers40260/hard/backend40800. Ожидаемый конец **16.10 около10:00МСК (07:00UTC), ориентир09:45–10:20МСК**; hard около**10:45МСК**, фактический watchdog отсчитывает40800с от первой code cell. Полное10-часовое выполнение ревью не ждёт. Находки/запуски внесены в `kaggle/runs.json`; следующий шаг — прочитать completed quality receipts9000/28370/конец и решить, какую форму продолжать.

[Startup audit](../checks/b1-review-production-startup.json): обе модели начали main, по100updates, CE3,8123/3,7303 finite, LR0,0001; setup/cache SHA gate и initial full stable save пройдены до этого по проверенному порядку кода. Secrets/live отсутствуют.

## Коммит

B1-only `git add` получил exit128: sandbox запретил создание shared Git `index.lock` (`Operation not permitted`). Индекс не изменён, обходов ограничения не было; коммит не создан. [Квитанция с33путями и сообщением коммита](../checks/b1-review-git.json). Чужие pre-existing изменения не staging. Отдельный [navigation patch](../checks/b1-review-navigation.patch) содержит только четыре добавленные этим ревью строки, без чужих изменений.

## Источники

Это собственное независимое ревью кода и receipts, не новое исследование качества. [b1-plan](b1-plan.md), [architecture-decision](architecture-decision.md), [r1-review](r1-review.md), [l3-review](l3-review.md), [overfit-check](overfit-check.md); исходный commit91188a2. Исходные работы уже зарегистрированы в bibliography.
