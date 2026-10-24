# Kaggle: Фазы A/B/C для ДЗ2

B4: [план и команды ревьюеру](../notes/b4-plan.md), [next-b4.json](../results/next-b4.json). Один B3 W600 stable → W600/q0 и W600-trunc/q10%; hard11ч20мин, smoke≤12мин. Локальные сборки `build_b4.py [--gpu-smoke]`, CPU checks `b4_tests.py`, `b4_regressions.py`, `b4_static_check.py`. GPU/push в подготовке отсутствуют.

[B3: strict resume обеих W-QK](../notes/b3-plan.md),16.10: `build_b3.py`, `b3_tests.py`, `b3_static_check.py`; [machine handoff](../results/next-b3.json). Private thisarivson/nlp-hw02-b3, hard40800с; отдельный real-state smoke720с. `b1_transfer.py --stream-states` переносит по одному файлу из friend2/nlp-hw02-b2/1 с CPU/SHA/remote SHA. GPU/push здесь0, запуск после reset17.10 00:00UTC — ревьюер.

## B1 — независимое ревью на аккаунте друга

[Ревью и receipts](../notes/b1-review.md), [актуальный план](../notes/b1-plan.md): GPU0 e8-d4-w768, GPU1 e6-d6-w768, fresh C-long2x, LR0,001, оценки9000/28370/конец main. `b1_tests.py` и `b1_static_check.py`; API — `b1_review_api.py`, только `KAGGLE_CONFIG_DIR=~/.kaggle-friend2`, private, без Secrets/live. Смоук≤900с, production hard40800с. Результаты запуска — в `runs.json`.

Перенос только pinned version: `b1_transfer.py`, обе latest full+order, private Dataset и SHA round-trip. После watchdog без aggregate manifest добавить `--emergency-receipts`: atomic latest и order SHA остаются обязательными. Большие outputs не скачивать целиком или ZIP; `b1_output.py` получает один точный файл. Верификация малых GPU outputs — `b1_review_output.py`.

## D1 — независимое ревью и API run

[Ревью/замеры/запуск](../notes/d1-review.md), [протокол](../notes/d1-plan.md).
Две T4 считают C-long2x CE на непересекающихся половинах всех9 725 737 строк;
затем GPU0 C-main / GPU1 C-long2x генерируют один stratified top≤4% (максимум389029).
Frozen v6, cap512, CPU overlap; C-main CE отсутствует. Secrets/live выключены.
Compact outputs/partial receipts — `/kaggle/working`, без полных моделей.
Production watchdog130мин/backend140мин; smoke thisarivson≤12мин, максимум2итерации.
Private friend seed29,6MB содержит1 250 497 pinned train-reference; в smoke
reference исполняются из оригинальных compact targets без четвёртого input.

`build_d1.py [--gpu-smoke]`, `d1_tests.py`, `d1_cpu_check.py`, `d1_static_check.py`.
API — `d1_review_api.py`: push/status/metadata/output/quota, credentials не выводятся.
Friend: `KAGGLE_CONFIG_DIR=~/.kaggle-friend`; smoke — default CLI config.
Точные версии запуска и receipts — [runs.json](runs.json).

## Final v6 — thisarivson, ready без push

[Схема и проверки](../notes/final-inference.md#v6-адаптивный-инференс-с-decoder-all): greedy→K2+K2→K6+K6, global uncertainty priority, decoder all,2 nice10 workers→до4 на хвост с той же очередью/cache. Прогноз≤126,4с; K8≤152,8с не проходит≤150. Private/offline/T4 `thisarivson/nlp-hw02-adaptive-inference-l3-v6`, `bundles/final_inference_l3_v6_mine`; один cold test1919+tune2048,≤720с.

Из `work/kaggle` (локально `/tmp/nlp-s3-env/bin/python`):

```bash
python final_v6_analysis.py
python build_final_inference_v6.py
python final_v6_regressions.py
python final_v6_checks.py
```

Последний шаг публикует `../results/kaggle-runs/nlp-hw02-kaggle-l3/final/next-inference-v6.json` с `bundle`/`ready:true` только после gates. Полный v6 GPU timer/quality pending; push не выполняется. Источник/имена весов: `build_final_inference_v6.py --weights-dataset OWNER/DATASET --main-weight MAIN.pt --long-weight LONG.pt`; custom pair не требует L3 journals, optional `--main-sha256`/`--long-sha256` закрепляют новую пару. Default L3 static gate не объявляет custom weights проверенными. P1-файлы не меняются.

## Профиль decoder — thisarivson, без push

[Статический аудит, флаги, ограничения и план](../notes/decode-speed.md), [ноутбук](notebooks/nlp-hw02-decode-profile.ipynb). Отдельный `bundles/final_profile_mine`, metadata id/title slug `thisarivson/nlp-hw02-decode-profile`, private/offline/T4; входы `thisarivson/nlp-hw02-l3-live` и `thisarivson/nlp-hw02-26b-inputs`. Full1919 baseline/all greedy/K4 с/без execution load, отдельные флаги на192 length-stratified вопросах, четыре коротких profiler trace. `--scope full` переводит все абляции на1919, но при лимите12мин возможен partial.

Из `work/kaggle` в окружении сTorch/tokenizers/nbformat/numpy:

```bash
python final_decode_audit.py
python build_final_profile.py
python final_profile_checks.py
```

Последний шаг публикует `../results/kaggle-runs/nlp-hw02-kaggle-l3/final/next-profile.json` с `bundle`/`ready:true` после CPU gates. Сборщик предварительно сбрасывает ready. GPU gate и≥2× pending; отправка на Kaggle остаётся пользователю. Текущие production final/P1 файлы не меняются. Скачивать `decode-profile/summary.json`, `timings.csv`, `trace-*.summary.json`, `trace-*.json.gz`; requested/effective nice записывается, PermissionError не скрывается.

## Финальный инференс — CPU-ревью, GPU pending

**Обновление final inference v3:** [причины и проверки](../notes/final-inference.md#v3-причины-и-исправления-после-gpu-v2). Текущий notebook — 2cold test1919 +1tune2048, цель≤120с, session900с/watchdog840/870. Greedy1024, sampling256×K4, buckets/OOM; guard сохраняет исполненные голоса. CPU commands: `python build_final_inference.py`, `python final_inference_checks.py`, `python final_v3_checks.py`, `python final_executor_benchmark.py`. API helper также900с, но в этой задаче не вызывался; Kaggle запуск — пользователь. Описание3test/150с/30мин ниже относится к v1–v2.

[Независимое ревью, риски и точные команды](../notes/final-inference-review.md), [протокол](../notes/final-inference.md), [ноутбук](notebooks/nlp-hw02-final-inference.ipynb), [CPU receipt](../checks/final-inference-cpu.json), [frozen source SHA](final-source-identity.json). **API probe остановился: env-авторизация отсутствует; push/run0.** Private `thisarivson/nlp-hw02-final-inference`, internet off, backend1800 с; внутри строго `CUDA_VISIBLE_DEVICES=0`, одна T4. Входы `l2-live` (обе FP16 l2-28800 + журналы) и `26b-inputs`; без Secrets/pip/train, selective ZIP только4 JSON.

Из корня `work/kaggle`, Python с Torch/tokenizers/nbformat:

```bash
python build_final_inference.py
python final_inference_checks.py
```

CPU: 1492/2048=72,8516%, выбор code/execution bit-identical старому mixed; restore32768 exact; переисполнение14799 unique —0 numeric mismatches,8nonnumeric timeout labels изменились. Закрыты SymPy string-eval/format escape, shared state и worker pipe death race, добавлены CPU timer2, fresh winner reexecution, deterministic runtime, fallback count и3answer SHA gate. Исходный/извлечённый physical decoder tokens/LP exact на tiny CPU K1/K4/K8; unit/format/nbformat/supervisor deadline checks пройдены. Новые K4/b64 **не воспроизводят первые4 K8/b32 по seed**: нужен quality gate±1п.п.

Из shell с существующей env-авторизацией (credential-файлы не читать):

```bash
KAGGLE_PYTHON="$(sed -n '1s/^#!//p' "$(command -v kaggle)")"
"$KAGGLE_PYTHON" final_review_api.py probe
"$KAGGLE_PYTHON" final_review_api.py push --iteration 1
```

API helper передаёт environment credentials официальному SDK без автоматического чтения credential files, проверяет private inputs и metadata, ограничивает число push двумя. Старый `kaggle_tools.py push` требует internet-on и здесь не используется. Даже если accelerator физически T4×2, второй GPU не виден процессу. Runtime gate — прежний L2 Torch2.11.0+cu128/CUDA12.8; несовместимость завершает запуск до генерации, установки в пределах квоты не выполняются.

В одном запуске:3 свежих test1919 с холодными моделями, каждый с отдельными `answers.jsonl`/`timing.json`; затем tune2048/reference. Timestamp **до Popen**, чтение/import/load/warmup/generation/restore/execution/vote/reexecution/answers-close внутри; OS page cache между процессами не сбрасывается. Готовый output: `final-inference/answers.jsonl` **из первого test**, `summary.json`, `test-1..3/`, `tune/`. Goal все3≤150 с,0fallback и побайтно одинаковые answers; обязательно<180 с. Quality допуск±1 п.п. к1492/2048; дополнительных GPU replay нет. Guard160 с для работы/172 с для supervisor, session watchdog1740/1770 с. До окончания greedy fatal startup нельзя покрыть настоящим greedy: такой запуск failed и требует ревью. Pinned output проверяет `final_output_check.py`; при успехе копирует первый test в `../submissions/final-mixed-k4k4-l2-28800.jsonl`.

Ожидаемый сценарий≈130 с: load/import5 +warmup1 +greedy18 +main50 +long50 +CPU tail5 +прочее1; это **прогноз**, полного C/T4 замера здесь нет. CPU service time перекрывается с GPU, его нельзя прибавлять второй раз. В summary смотреть все gates и guard/OOM/phase breakdown. После успешного GPU gate `BENCHMARK=False` оставляет один cold test для сдачи. Адаптивного sampling нет до измеренного дефицита времени и отдельной offline-проверки≤0,2 п.п.

## L2 — результаты боевой v2

[Разбор](../notes/l2-results.md), 2026-10-13: UI-запуск пользователя — **серверная версия2**, COMPLETE / worker complete / exits [0,0], wall34 411,55с. Версия1 ERROR, её не использовать для результатов. Greedy main/long 70,02/70,41%; K8 tune1471/1478; mixed4+4+main greedy1492/2048. Long направленный gate не прошла; ensemble пока tune-only. Eval vote, fresh и full mixed single-T4≤180с не выполнялись.

Pinned output v2 + private livev11 + C-only L1livev7 сохранены в `../results/kaggle-runs/nlp-hw02-kaggle-l2/1/` — имя1 задано пользователем и не означает version1. Все JSON/JSONL/logs и шесть FP16 points; **788 MiB**, два final live FP16 — hardlinks. Stable full states и order.npy не скачивались. Все шесть weight SHA/checkpoint/metrics совпали, final CPU finite/format gate прошёл. [Квитанция и аудит](../checks/l2-download-audit.json), [CPU отчёт](../results/l2-results.json). Live compressed receipts проверены воспроизведением gzip; API files отдаёт expanded JSON. Перед L3 проверить restore и фактические имена на mount.

Воспроизведение из корня worktree (интерпретатор с установленным Kaggle SDK; SDK обслуживает авторизацию, ключи вручную не открывать):

```bash
# При обычном CLI latest сейчас v2; фильтр исключает full checkpoints.
kaggle kernels output thisarivson/nlp-hw02-kaggle-l2 --file-pattern '(\.jsonl?|\.log|\.txt|\.csv|/weights/l2-.*-fp16\.pt)$' -p work/results/kaggle-runs/nlp-hw02-kaggle-l2/1/
# Для строгого pinv2, paged livev11, atomic download, dedup и SHA receipt:
python work/kaggle/download-l2-results.py
# CPU, зависимости numpy/matplotlib; нужны также C-only JSON L1livev7.
python work/research/analyze-l2-results.py
```

`download-l2-results.py` жёстко закрепляет kernel v2, выбирает доступную current private live (в этом разборе v11) и пишет её номер в receipt. Для повторения именно этого live-аудита нужна v11: анализ ожидает `live-v11/`. Исторические C-only `model-C__*.json/jsonl` L1 скачаны отдельными file requests из `thisarivson/nlp-hw02-l1-live/7`, квитанция — в `l1-reference-live-v7/`. Utility не запускает kernels и не удаляет remote данные.

Greedy1919 measured19,88/22,86с. K8 tune2048 aggregate156,38/155,72с **без preceding weights/tokenizer load**; separate sampler/exec counters не записаны. Single K8 test сценарий172с,≈198с с запасом; mixed4+4 171…317с,≈197…365с с запасом; mixed8+8≈317/365с. Все mixed estimates требуют двух C на одной T4. Следующий шаг — inference-only full timer до решения о второй C в L3; eval выбранного финального декодера один раз, fresh после фиксации.

## Историческая подготовка L2

[План L2](../notes/l2-plan.md), 2026-10-13: GPU0 C-main строго из L1 C47 155updates/cursor6 035 840/main17 569.935878398с; GPU1 C-long2x из того же полного anchor, меняется sampler. Private L1 live v7 проверена; нужный C SHA `1da28095d960fb0958ff26883bc6dc832e4307c9bab4023a5621b1f9f6ff01dc`. Новый [private l2-live](https://www.kaggle.com/datasets/thisarivson/nlp-hw02-l2-live) заранее создан, v1 ready. D не загружается/не обучается.

Парные +3ч/+6ч/+8ч main; timed cooldown15/15/50мин. Greedy heldout4k/tune2048/eval2048/old991, original length bins, per-example correctness; финальный K8 T0.7 tune≤15мин с каждым executed sample/status/mean logprob. Greedy fulltest1919 timer на всех точках. `l2_compare.py` считает paired gains/losses/CI и mixed K4+K4 офлайн. Workers≤11ч, sync≤20мин, hard wall11.4ч, backend12ч. Quota12.5ч включает смоук.

Nonfinite loss/нештатный gradient пропускаются;3подряд или10/100→rollback/LR÷2, третья эскалация→штатный stop только этого worker. Обычные GradScaler skips отдельно. Ошибка соседа больше не создаёт общий stop-request. Full stable каждые30мин; live1.5ч и после points, weighted order также в live. FP16 weights отдельны от full resume. Secrets обязательны только в боевом gate. [Регрессии](../checks/kaggle-l2-regressions.json), [реальный CPU resume](../checks/kaggle-l2-real-resume.json), [самплер](../checks/kaggle-l2-weighted-order.json), [private inputs](../checks/kaggle-l2-live-input.json).

Сборка создаёт только локальные артефакты:

```bash
python build_l2.py
python build_l2.py --gpu-smoke
python l2_tests.py
python l2_static_check.py
```

Ноутбуки: [боевой L2](notebooks/nlp-hw02-kaggle-l2.ipynb), [смоук ревьюера](notebooks/nlp-hw02-l2-gpu-smoke.ipynb). Bundles `bundles/l2` / `bundles/l2_gpu_smoke` содержат private/internet/T4 metadata и timeout; это генерируемые ignored-папки. Notebook source tracked, weights/order arrays — нет. Реальный C уже находился в ignored results; повторной загрузки644MB не было, чужие файлы сохранены. Собственные временные загрузки — только два JSON v7, удаляются после проверки.

Ревьюер проверяет локальные source/metadata и CPU receipts, выполняет GPU-смоук (Secrets не нужны), затем по отдельной команде private production push/UI Save & Run All с Secrets. Строгий Torch2.11.0+cu128/CUDA12.8 gate — в первой ячейке до installs/cache; автоматически ослаблять нельзя. Боевой startup private upload остаётся обязательным, несмотря на заранее созданный Dataset. Полный mixed GPU timer≤180с и K8 eval2048 — последующий inference-only gate после выбора по tune.

```bash
python l2_compare.py --working PATH_TO_L2_OUTPUT --out paired.json
python l2_compare.py --main-k8 PATH_TO_C_MAIN_K8 --long-k8 PATH_TO_C_LONG_K8 --out mixed.json
```

Ни GPU-smoke, ни боевой L2 kernel не push/run в этой подготовке. Записи L1 ниже — предыдущая подготовка; фактические итоги боевого L1 из сообщения пользователя отражены в плане L2.

**L1 прошёл независимое ревью; CUDA smoke v2 и полный SHA-аудит output прошли.** [Ревью](../notes/l1-review.md). Исправленный [локальный ноутбук](notebooks/nlp-hw02-kaggle-l1.ipynb) нужно импортировать в редактор перед запуском: опубликованный [L1 v1](https://www.kaggle.com/code/thisarivson/nlp-hw02-kaggle-l1) остался старым. Боевой L1 во время ревью не запускался и не repushed. [Протокол](../notes/l1-long-training.md), [runs.json](runs.json).

Кеш [verified CPU v2](https://www.kaggle.com/code/thisarivson/nlp-hw02-compact-cache-verified/versions?versionNumber=2): **9 725 737строк**, все60шардов, int16/int64 mmap, order seed2026101201. SHA manifest `46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be`; round-trip **0/50 040** в Kaggle и локально. Все428output files скачаны,360array SHA/offsets/source/provenance/order проверены: [аудит](../checks/kaggle-l1-cache-output.json). L1 подключает этот pinned kernel output.

C **53 646 403** параметров (6+6/d512/h8/SwiGLU1368); D **103 902 979** (8+8/d640/h10/SwiGLU1712). CPU smoke и независимые регрессии: дваworkers/ветки/eval→backward/точноеresume/микроaccumulation/Secrets+GPU+live-failure gates/privateCLI mocks/staging/timeout/heartbeat/watchdog/выборL2 прошли. CUDA smoke проверяет полные C/D, FP16/fused, memory/branches/eval→backward/resume. Full1919/длинное качество и реальные UI Secrets/live ещё не проверены.

Запуск пользователем:

1. **Edit → File → Import Notebook**: загрузить исправленный `notebooks/nlp-hw02-kaggle-l1.ipynb` из этого worktree; проверить startup и memory-probe source по ревью. Затем **Add-ons → Secrets**: прикрепить и включить `KAGGLE_USERNAME` и `KAGGLE_KEY`; Accelerator **GPU T4×2**, InternetOn.
2. **Save Version → Save & Run All → Save**. Дождаться появления background version в Queued/Running; в логе должны пройти Secrets/T4 и private live-probe gates.
3. В редакторе **Session options → Stop Session**, затем закрыть вкладку. Не отменять background version. Закрытие вкладки без Stop Session не гарантирует освобождение interactive GPU.

Live private `thisarivson/nlp-hw02-l1-live`: начальный probe обязателен, далее full main model/AdamW/scaler/RNG/cursor и latest evaluated FP16 points ≤2.5GB staging. Full snapshot каждые1.5ч и послеpoints; страховка устарела≥2ч → stop. Checkpoints каждые30мин. Warmup1000 → постоянный LR доL2/L3, final main cooldown отсутствует. Общие pure-main-time points и правило C/D записаны до запуска.

Первая conversion CPUv1 завершила все60шардов за≈76мин, но numeric round-trip gate упал; original/restored journals тогда не сохранялись. Все её массивы независимо проверены и переиспользованы. Kaggle запретил ERROR output как kernel source, поэтому промежуточный verifyv1 не получил mount. Verifiedv2 использовала private сжатый recovery-target input481406946байт + исходные26a/26c sources; отдельный Torch/NumPy-free executor wall3с/CPU2с дал50040/50040. Conversion не повторялась. Точная первичная причина memory limit/cold import/timeout не доказана. История в runs; всеCPU безGPU.

Для следующей сессии использовать **новый slug**, ту же implementation/data identity и фактическую завершённую UI-версию (заменить VERSION):

```bash
python3 build_l1.py --kind l1 --slug nlp-hw02-kaggle-l2 --cache-source thisarivson/nlp-hw02-compact-cache-verified/2 --cache-sha 46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be --resume-source thisarivson/nlp-hw02-kaggle-l1/VERSION
```

`--resume-live` дополнительно подключает private live Dataset; выбирается максимальный verified cursor каждой модели. Если L1 осталась ERROR и output недоступен как source, resume возможен только черезlive. Не использовать API gate-only v1 как resume: полных состояний в ней нет. Сборка ничего не запускает; модель/моделиL2 выбираются по зарегистрированному правилу после L1.

**Актуальный S3:** [№30/31/32, протокол до push](../notes/s3-data-representations-and-b-scoring.md),90мин/жёсткий потолок105мин; новый private slug `thisarivson/nlp-hw02-kaggle-s3`. CPU round-trip0/57702 для каждого представления, tiny gates прошли. Версия/статус/URL — [runs.json](runs.json). Исторический вариант продолжения S2 ниже не является рецептом этого S3.

**S2 v2 разобрана: Kaggle COMPLETE, внутренний partial, exits[1,0], 131,29мин.** №27b failed token gate35/4096; M не стартовала; B завершила4точки, t04 2513/4096 противA2538. Fulltest1919 не измерен; A/M/B pending. Live uploads отключены из-за недоступных Secrets. [Разбор и S3](#разбор-s2-v2-и-восстановление-s3), [summary](../results/kaggle-s2-v2-summary.json), [runs.json](runs.json).

**S1 v1: Kaggle COMPLETE, внутренний итог partial (exit codes0/1).** №26b и №27 завершены; №28 остановился на CPU reference-gate до обучения B. Output скачан и проверен. [Разбор S1](../results/kaggle-s1-v1-summary.json) · [26b-final](../notes/26b-final-wsd-data-scaling.md) · [27](../notes/27-execution-voting-s1-results.md) · [28](../notes/28-capacity-b-s1-results.md). Новый протокол S2 (27b/M29/B28, 3ч15мин) записан до push. Версии и статусы — в [runs.json](runs.json).

S1 использовал два независимых процесса на T4×2 с бюджетом3ч. GPU0 завершил26b и27; GPU1 проверял reference и остановился. Три private Dataset v1 остаются входами. [Исходный S1](notebooks/s1-t4x2-26b-27-28.ipynb), [CPU26c](notebooks/26c-cpu-cache-extension.ipynb).

`src/exp26_*` — точные12 модулей исходного26b, исходная identity сохранена в `source-identity.json`. Адаптеры Kaggle и27/28 находятся отдельно. Веса, кеш, staging, push-bundles и скачанные outputs исключены из Git. Ключи читает только штатная библиотека/CLI Kaggle для авторизации; наши скрипты не открывают конфигурацию ключа, не выводят её и не включают в uploads.

Фаза A закоммичена: `78e0df7`. Точные версии загрузок и запусков Фазы B фиксируются в [runs.json](runs.json).

Фаза B сохранена в коммите `35ddb46`; Фаза A — `78e0df7`.

## Метаданные CLI2.2.4

Для T4×2 используется **`"machine_shape": "NvidiaTeslaT4"`**, вместе с `enable_gpu=true`. При push дополнительно передаётся `--accelerator NvidiaTeslaT4`; флаг имеет приоритет над полем. Отдельного поля `accelerator` в JSON нет. Подтверждено официальными [метаданными](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md), [командами](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) и исходником **установленного2.2.4**: `KaggleApi.kernels_push` присваивает `request.machine_shape = acc if acc else ...`. Перехват `save_kernel` подтвердил настоящий сериализуемый запрос без сетевого сохранения GPU-ноутбука. Это проверка клиента, фактическое выделение двухT4 проверяется в первой рабочей ячейке S1.

Оба ноутбука: `is_private=true`, internet on, `dataset_sources` и `kernel_sources` явные. 26c: GPU/TPU false и пустой `machine_shape`. Метаданные генерируются в игнорируемом `bundles/<kind>/kernel-metadata.json`. Все uploads Dataset private по умолчанию; перед version проверяется серверная приватность. Опции public/delete отсутствуют. `other` в license не предоставляет право на распространение исходных материалов. Сервер отверг `copyright-authors`, хотя он перечислен в документации; использован `other` с сохранением прав исходных авторов.

## Фаза B: после окончания скачивания Drive

Из корня worktree выполнить:

```bash
cd work/kaggle
python3 build.py --kind static-inputs
python3 kaggle_tools.py prepare-drive --downloads ~/Downloads/hw02-drive --unpacked-only
python3 kaggle_tools.py upload-drive --owner thisarivson
python3 kaggle_tools.py dataset --folder staging/26b-inputs --id thisarivson/nlp-hw02-26b-inputs --title 'NLP HW02 26b inputs'
python3 build.py --kind s1 --owner thisarivson --budget-hours 3
python3 kaggle_tools.py push --folder bundles/s1 --run-gpu
```

`prepare-drive` безопасно объединяет несколько ZIP и обычные папки в `staging/drive-merged/`. Downloads остаётся неизменным. Конфликтующие копии, symlinks и опасные пути отклоняются. Затем сверяются:

- cache manifest `2c691fef64fea8b4186b8acae54561d5620080286196a4390a23108776e80add`, sealed/7078471/шарды0–41 и SHA/bytes каждого массива;
- order `737dac5a146ad15bb0ec70cb06fea59493277e4332dea440b2bdf2a6316c1637`;
- resume `8400c8e59da764c65fbcb574758acc72e8e034aedeb59b10bf979465f110ab5e`,145323979байт, main46875, phase main; PT JSON должен совпасть с latest;
- в `checkpoints/` разрешён только `session-stop-055625-8feec45530.pt`.

Для уже распакованных входов `--unpacked-only` пропускает все ZIP. Исходные архивы Drive не требуются. `order/order.npy` и его receipt копируются также в корень run-папки только внутри staging — так их читает адаптер Kaggle; SHA и bytes остаются строгими. Датасеты опрашиваются раз в 90с, ожидание `ready` ограничено 60мин.

`upload-drive` сохраняет два private Dataset: `nlp-hw02-26-token-cache` и `nlp-hw02-26-resume`. В cache upload идут только manifest, NPY и shard receipts; raw/pilot/journals не требуются. Для существующего private Dataset выпускается новая версия без удаления старых. Импорт через TAR поддерживает как уже распакованные входы Kaggle, так и архивы; поиск учитывает `/kaggle/input/<slug>` и вложенный `/kaggle/input/datasets/<owner>/<slug>`.

Статусы Dataset должны стать `ready` до S1:

```bash
python3 kaggle_tools.py status thisarivson/nlp-hw02-26-token-cache --dataset
python3 kaggle_tools.py status thisarivson/nlp-hw02-26-resume --dataset
python3 kaggle_tools.py status thisarivson/nlp-hw02-26b-inputs --dataset
python3 kaggle_tools.py status thisarivson/nlp-hw02-kaggle-s1
python3 kaggle_tools.py output thisarivson/nlp-hw02-kaggle-s1 --version 1
```

Output сохраняется в `work/results/kaggle-runs/<slug>/<version>/`. Версию брать из ответа push; номер не угадывать. У установленного2.2.4 есть ошибка: обычный `kernels pull .../N` получает403, а status/output теряют номер и обращаются к latest. `kaggle_versioned.py` передаёт **`version_label=vN`** через официальный SDK, включая каждую страницу output; установка CLI не меняется. Основание — [официальное исправление](https://github.com/Kaggle/kaggle-cli/pull/1201) и проверенный локальный исходник. Live API различил ERRORv1 и RUNNINGv2; pinned pullv2 прошёл. Paged download проверен на двух mock-страницах/файлах; у работающего26c output пока пуст. `api-check` делает только два списка `--mine`.

## Межверсионный resume

Предыдущий output подключается через `kernel_sources`, затем выбранные run-папки копируются в working. Использовать отдельный slug следующей сессии, чтобы не создавать самозависимость:

```bash
python3 build.py --kind s1 --owner thisarivson --slug nlp-hw02-kaggle-s2 --budget-hours 3 --previous-source thisarivson/nlp-hw02-kaggle-s1/1
python3 kaggle_tools.py push --folder bundles/s1 --run-gpu
```

Те же cache/resume/static Datasets остаются подключёнными для order, старых сравнений и контрольных входов. Resume выбирает полный latest из output, не по mtime. Восстанавливаются model/AdamW/GradScaler/scheduler/RNG/cursor. SHA кода/протокола/кеша/order/токенизатора/initial остаются строгими. Torch/CUDA могут отличаться от Colab: пишется migration и актуальный runtime, checkpoint identity сохраняет происхождение. Побитовое равенство CUDA между версиями не обещается; D0 переоценивается с gate±3 matches/CE1e-4.

## Бюджет, диск и CPU

Общий таймер начинается в первой ячейке, включая install/подготовку. По умолчанию3ч;5мин на упаковку, минимум10мин до12ч лимита. Worker0 оставляет30мин резерв финалу/оценкам, worker1 —20мин для оценок/test. Supervisor ждёт **обоих**, печатает heartbeat/log каждые30с, по дедлайну завершает оставшиеся процессы. Последнее незаписанное вычисление может повториться; unsafe live state не сериализуется.

`CUDA_VISIBLE_DEVICES=0/1`, по одному видимому устройству в каждом процессе. Torch/BLAS по1 потоку, исполнение программ по2 spawn-worker на процесс; heavy imports только внутри main, workers песочницы не импортируют Torch. Exec/обучение идут отдельными фазами каждого процесса. T4 есть два, pooled/DDP обучения нет.

Кеш NPY и order читаются через memmap. Сохраняются latest, текущий anchor и checkpoints оценённых точек/финала; старые periodic PT удаляются **только в собственной working-копии**. Источники Dataset и прежние версии не меняются. Guard18GiB плюс проверка перед ZIP с потолком19GiB; это запас до20GB. Общий ZIP содержит результаты и удержанные полные checkpoints, не входной кеш/raw. Итоги: `gpu0-a26/26b-final-summary.json`, `gpu0-exp27/27-summary.json`, `gpu1-exp28/28-summary.json`, `s1-summary.json`, `s1-results.zip`; при остановке summary явно partial.

## 26c CPU

Private [Dataset](https://www.kaggle.com/datasets/thisarivson/nlp-hw02-26c-inputs) содержит прежние tokenizer/protected questions. [CPU-ноутбук](https://www.kaggle.com/code/thisarivson/nlp-hw02-26c-cpu-cache) обрабатывает **только42–59**; cache0–41 не подключён. Те же pinned Parquet SHA, byte/hash/text exclusions, finite solve, AST/alarm/resource sandbox и token schema. Отдельный manifest, per-shard/journal/token-block resume. Outputs: `26c-token-cache/`, `26c-output.json`, `26c-metadata.zip`. GPU-квота не расходуется по согласованному режиму. Версия1 остановилась на поиске mount-пути, её логи сохранены; исправленная версия2 — **COMPLETE**. Output v2 скачан в `work/results/kaggle-runs/nlp-hw02-26c-cpu-cache/2/`: sealed, 18шардов42–59, 2 647 266строк. [Проверка manifest, 108массивов и18Parquet](../checks/kaggle-26c-v2-output-checks.json): SHA/bytes, pinnedidentity и receipts совпали; metadataZIP CRC прошёл, fresh/overlap0. Сервер подтвердил private=true, GPU/TPU=false, internet=true. Ничего на Kaggle не удалялось.

Для нового CPU push:

```bash
python3 build.py --kind 26c --owner thisarivson
python3 kaggle_tools.py push --folder bundles/26c
```

## Проверки до S1: история Фазы A/B

CPU smoke и независимые регрессии: tiny Parquet42/43, исключения/SHA/memmap/shard resume; TAR с вложенным ZIP, несколько Drive ZIP, path guards и вложенный mount; два настоящих CPU subprocess; две короткие ветки/полное state parity при resume; runtime migration; shared crossKV logits; реальные tiny sampling/exec/full timer, greedy fallback/vote/tie fixtures и пороги1.5п.п./120с. Таймер B для детерминированных граничных проверок — явно fixture вокруг настоящих tiny updates, не прогноз T4.

Воспроизведение: `python smoke.py` в CPU-окружении с Torch/numpy/tokenizers/pyarrow/sympy; `cli_checks.py` запускается Python-интерпретатором из shebang установленного Kaggle CLI, проверяет настоящий клиент2.2.4 без сети и с пустым временным каталогом авторизации. Готовые JSON-проверки лежат в `work/checks/`.

Drive/cache/PT проверены: [SHA/bytes 252 массивов, manifest, order и resume](../checks/kaggle-phase-b-input-checks.json). Не проверены: полноценный Colab→Kaggle CUDA restore и D0 gate; T4×2, FP16/fusedAdamW/SDPA5D, B128/OOM96, память, полный heldout/test и120с, полнота S1 за3ч. 5D broadcast может выбрать более медленный backend; это главный риск sampling. Результаты27/28 не выдуманы. Один seed, repeated heldout selection; fresh и официальный формат/score пока открыты.

## Фаза C: разбор S1 v1

`kaggle_tools.py output thisarivson/nlp-hw02-kaggle-s1 --version 1` завершён: **1951 файла**, pinned `version_label=v1`. Для ускорения множества мелких запросов сначала скачан архив этой же версии и безопасно распакован; повторный `output` получил оставшиеся файлы и записал version receipt. Output — ignored `work/results/kaggle-runs/nlp-hw02-kaggle-s1/1/`.

ZIP: **350077134 байт**, SHA256 `e834fe795d2212bedb0b22c17fa4953e26f62f561d829578b6eb9aee01a6ae43`; CRC и SHA/bytes всех1906 manifest members прошли. Четыре summary-файла есть. `s1-summary.status=partial`, exit codes `[0,1]`;26b decision complete,27 complete,28 partial без B state.

| Опыт | Результат | Решение |
|---|---|---|
|26b-final|2538/4096 (61,96%),634/991;CE4k0,314924;test12,547с|Наклон5,290≥2 п.п./удвоение; последний +2,246≥1 п.п. Данные/compute остаются рычагом, capacity hint не сработал.|
|27|Лучший vote T0,7/k32:2726/4096,+4,590 п.п.;493,33с|Все три лучшие альтернативы >120с; выбран greedy13,003с. K4/k8 не timed.|
|28|0 B updates/points/checkpoints; CPU reference finite-count assertion|Pending; S2 начинает B с seed42.|

[CPU-аудит](../checks/kaggle-s1-v1-cpu-audit.json): **531755 program rows /325391 unique**, все status/value,12 greedy scores,50 voting scores и3 test winner sets совпали. [Полные checkpoints и CE](../checks/kaggle-s1-v1-model-cpu-checks.json): AdamW/scaler/scheduler/RNG/cursor/tensors проверены; CPU CE4k0,3149203963 иold0,3310143674, расхождениеGPU≈3,3×10⁻⁶. Новых training updates/GPU runs при разборе0.

| Ресурс | Факт |
|---|---|
|Время сессии|3884,92с=64,75мин до упаковки; последний kernel log3916,44с≈65,27мин с упаковкой/export. Бюджет3ч не израсходован.|
|Квота|ДвеT4 выделены примерно65,3мин. Баланс аккаунта/точное списание квоты не читались.|
|GPU0|A3125 attempted updates/369,31с train, затем27/3368,17с.|
|GPU1|Exit1 уже к heartbeat79,91с; B и D0 не запускались. Большую часть сессии устройство простаивало.|
|AMP|В S1 у A1skip; вся история10, финальная траектория9. У B измерений нет.|
|Память27|Peak reserved14,412GiB; train peak/B memory не записаны.|
|Exec27|Primary3chunk retries/2pool restarts; неизвестно точное число extra program calls.|
|GPU utilisation / SDPA backend|Телеметрии нет; runtime подтверждает TeslaT4×2, CUDA12.8/Torch2.11.0.|

Причина28 по traceback: `exp26_workflow.reference_execution` получил число numeric heldout4k, отличное от4096. Неудачные execution rows не сохранены, поэтому конкретный код и первопричина неизвестны. Cold SymPy import/timeout — гипотезы. На CPU reference полностью воспроизведены. Для S2 подготовлено повторное использование verified references GPU0 из pinned S1; SHA/identity/значения/finite indices проверяются, правила sandbox не меняются.

Greedy-таймер27 не включает execution (`executed_candidates=0`), хотя подпись общая для всех режимов. Таймеры трёх sampling-альтернатив включают execution/voting/I/O. Fresh, официальный score и формат сдачи остаются открыты.

## S2: первоначальная подготовка Фазы C

[Отдельный notebook](notebooks/nlp-hw02-kaggle-s2.ipynb) и ignored `bundles/s1/kernel-metadata.json` готовы: private T4×2, три прежних Dataset, `kernel_sources=["thisarivson/nlp-hw02-kaggle-s1/1"]`. S1 source notebook сохранён. A full final и27 artifacts переиспользуются; B state отсутствует. GPU1 читает reference receipts из предыдущего GPU0 output после строгой проверки. [CPU-проверка исправления](../checks/kaggle-s2-reference-reuse-cpu-checks.json).

Из `work/kaggle/`:

```bash
python3 build.py --kind s1 --owner thisarivson --slug nlp-hw02-kaggle-s2 --budget-hours 3 --previous-source thisarivson/nlp-hw02-kaggle-s1/1
# Команда ниже запускает GPU; в Фазе C она НЕ выполнялась.
python3 kaggle_tools.py push --folder bundles/s1 --run-gpu
python3 kaggle_tools.py status thisarivson/nlp-hw02-kaggle-s2
# Номер VERSION взять из ответа push, затем:
python3 kaggle_tools.py output thisarivson/nlp-hw02-kaggle-s2 --version VERSION
```

B сравнивается с зарегистрированными A при928/2018/4203с чистого cumulative train-time; выбранное27 декодирование — greedy. Состояние stable B и оценённый cooldown checkpoint различаются. Большие ZIP/PT/NPY/candidate outputs в Git не входят.

Проверки воспроизводятся Python с Torch/numpy/tokenizers/sympy:

```bash
python ../checks/kaggle-s1-cpu-audit.py
python ../checks/kaggle-s1-model-cpu-check.py
python ../checks/kaggle-s2-reference-reuse-cpu-check.py
```

## Коммит Фазы C

Попытка `git add` заблокирована песочницей: невозможно создать `index.lock` в общем Git-каталоге worktree (`Operation not permitted`). Index не изменён; правки сохранены. Все изменённые файлы меньше1MiB; ZIP/PT/NPY/массовые outputs ignored. Фаза B уже в `35ddb46`.

Из корня worktree в обычном терминале:

```bash
git diff --check
git add courses/nlp/backlog.md courses/nlp/index.md courses/nlp/progress.md dashboard.md knowledge/bibliography.md
git add courses/nlp/homeworks/hw02-code-generation
git commit -m 'nlp: analyze S1 and prepare pinned S2 for capacity B'
```

Эти команды относятся к сохранённым изменениям Фазы C в текущем чистом до её начала worktree. Git push и Kaggle push не выполнялись.

## S2: №27b → M29 / B28, 3 ч 15 мин

[Протокол до запуска](../notes/s2-fast-sampling-and-capacity.md) · [S2 notebook](notebooks/nlp-hw02-kaggle-s2.ipynb) · [CPU smoke](../checks/kaggle-s2-cpu-checks.json). Пользователь разрешил один private запуск T4×2 на11700с; от первой ячейки до остановки процессов11400с, затем300с на сводки. Повтор v2 отправлен; аварийный watchdog выходит в12000с, серверный timeout —13200с. GPU0 сначала проверяет27b на A6.4M, затем M6+6/d256/15 766 464параметра с LR0.003. GPU1 повторяет B6+6/d384/LR0.002. Точные цели928.116608/2018.225874/4203.384926/7312.883812с читаются из verified26b-final-summary.

27b: физический repeat_interleave cross-K/V, только4D SDPA/FP16; self-KV/output/done заранее выделены, CPU termination-check раз в16шагов. Memory-probe2048 на самых длинных входах, приOOM1024, отдельно для A/M/B; greedy batch≤512. Gate: новый реальный greedy token-identical всем saved4096, k16/T0.7=2707±45. Полные свежие test1919 для greedy/k8/k16 включают **execution даже для greedy**, voting и answers write. Выбирается максимальный k≤120с, без greedy-голоса; при отсутствии допустимого vote явно greedy fallback. Старый S1-ноутбук и frozen12модулей26 не меняются.

Reference берутся из pinned receipts и26b-inputs без исполнения. Generated wall3с/CPU2с, CPU user+system контролируется ITIMER_PROF, по2worker, timeout counts и причины сохраняются. Файловый lock сериализует D0/point/test-оценки, другое GPU может обучаться; wait-time записывается вне fulltest timer. D0gate как28. После четвёртой цели контроллер останавливается; отсутствующие точки явно missing. Stable latest и evaluated cooldown checkpoint различаются.

В output остаются только latest и последний оценённый полный checkpoint с AdamW на M/B; все time-точки хранят model-only FP32 веса. Временный cooldown anchor вне working, а внутри branch-latest есть его payload для полного resume. Кеш/resume читаются из input через memmap; fallback-распаковка и anchor — в `/kaggle/temp` или `/tmp`, с очисткой при финализации. ZIP не создаётся, веса не дублируются. Цель output≤3–4ГБ; tiny CPU output≈66МБ с тестовыми копиями, итоговый объём полного GPU-прогона ещё неизвестен.

Watchdog вооружён в первой ячейке, до установки зависимостей и подготовки входов. Workers/installer — отдельные process group; на дедлайне TERM всей группе, через20с KILL, включая multiprocessing exec-внуков. Wait только с таймаутом; unreaped worker не мешает сохранению сводок. На deadline+10мин watchdog записывает `session-summary.json` и вызывает `os._exit(0)`. Heartbeat: последние3строки логов и время до дедлайна. Метрики и summary точки атомарны и записываются до следующего checkpoint I/O; общий report восстанавливает завершённую точку из отдельных receipts. `session-limits.json` в bundle задаёт push-обёртке серверный `--timeout 13200` (3:40).

Правило выбора: самая поздняя общая time-точка, лидер при≥2п.п. и full selectedtest≤120с; при<2п.п. минимум времени среди близких лидеру допустимых моделей. Метрикиexec4k/991, CE, длины, McNemar по всем общим точкам. В S1 не сохранены tensors ранних A-точек: если общая точка раньшеt04 и для выбора требуется её A-тайминг, итог остаётся pending. A6.4M не подставляется вместо ранней A.

Из `work/kaggle/`:

```bash
python3 build.py --kind s2 --owner thisarivson --slug nlp-hw02-kaggle-s2 --budget-hours 3.25 --previous-source thisarivson/nlp-hw02-kaggle-s1/1 --sample-batch-rows 2048
python3 kaggle_tools.py push --folder bundles/s2 --run-gpu
python3 kaggle_tools.py status thisarivson/nlp-hw02-kaggle-s2
# VERSION брать только из push response:
python3 kaggle_tools.py output thisarivson/nlp-hw02-kaggle-s2 --version VERSION
```

Воспроизведение CPU: `python s2_smoke.py`, `python ../checks/kaggle-s2-reference-reuse-cpu-check.py`; CLI request/pinnedversion checks — `cli_checks.py` из Kaggle CLI Python. Tiny FP32/FP16 logits/greedy, реальные receiptSHA/identity, sampling resume/full timers, два процесса/M/B ветки, complete-state parity после удаления scratch, latest/final retention, model-only/no ZIP и2п.п./120с границы проверены. GPU memory/backends/fusedAdamW/B128→96/full4096gate/test1919скорость и достижимостьt04 **до push не проверены**.

### Устойчивый повтор S2: live-сохранение

Пользователь отменил v1: output и лог пусты, результаты потеряны; API подтвердил `CANCEL_ACKNOWLEDGED`. Причина зависания не установлена; прежний безграничный wait найден в коде. Разрешён полный повтор: новая версия **того же slug**, без `--resume-source`, прежние27b→M/GPU0 и B/GPU1, pinned S1/1. Версию и проверенный статус брать из `runs.json`.

Secrets `KAGGLE_USERNAME`/`KAGGLE_KEY`, прикреплённые к этому ноутбуку, читаются через `UserSecretsClient` только в окружение. В session-config их нет; workers получают окружение без credentials. Вывод CLI и исключений не сохраняется: журнал содержит только тип ошибки/exit code. Перед upload проверяются новые/изменённые файлы working и весь staging, включая бинарные веса. Публичное имя владельца в фиксированном Dataset ID — идентификатор ресурса; оно не берётся из secret и не считается auth-данными.

Daemon-поток создаёт private [`thisarivson/nlp-hw02-s2-live`](https://www.kaggle.com/datasets/thisarivson/nlp-hw02-s2-live), затем выпускает версии: после нового `metrics.json` (опрос5с), обновления test/summary и раз в900с. Маленькая staging-папка вне working: сводки, metrics, heldout/test journals, answers, session-summary/live, хвосты process-log32КБ, веса **только оценённых точек FP16 без AdamW**. Большие JSON сжимаются в `.json.gz`; FP16 recovery не равен исходным FP32 весам. Staging≤450МБ, с запасом на временный CLI ZIP≤1ГБ. Полные latest/final остаются в штатном output.

Каждый subprocess — отдельная группа с ограничением≤600с; upload≤540с, весь snapshot≤590с. Ошибки не останавливают обучение или watchdog. Тяжёлые IO/Torch вынесены в CPU-child. Финальный best-effort upload:30с перед штатным выходом;20с из watchdog, с резервом25с **внутри** потолка12000с. Даже зависшие IO или upload не задерживают `os._exit`.

```bash
kaggle datasets download -d thisarivson/nlp-hw02-s2-live -p ./s2-live --unzip -o
# --dir-mode zip создаёт snapshot.zip внутри скачанного датасета:
unzip -o ./s2-live/snapshot.zip -d ./s2-live/snapshot
```

[CPU live smoke](../checks/kaggle-s2-live-cpu-checks.json), `python s2_live_smoke.py`: mocked create→version/private/zip, FP16/no optimizer, проверка credentials и отказ при внедрённом ключе, периодический/new-metric triggers, реальный hung child+grandchild timeout, bounded final и watchdog exit при зависшем upload. Реальные Secrets/создание Dataset и скорость финальной загрузки проверяются только в Kaggle-сессии.

[Watchdog CPU checks](../checks/kaggle-s2-watchdog-cpu-checks.json): SIGTERM-resistant libc sleep/loop, spawn-pool внуки, TERM→20с→KILL, зависший report, unreapable leader и watchdog до setup. `python s2_watchdog_smoke.py`; новая сборка/входы/лимиты — `python s2_retry_checks.py`. Реальный CUDA D-state, серверное сохранение output после аварийного выхода и все полноразмерные GPU-метрики ещё не проверены.


Коммит повтора заблокирован песочницей на shared `index.lock`; index не изменён. Из обычного терминала в корне worktree:

```bash
git add courses/nlp/homeworks/hw02-code-generation courses/nlp/index.md courses/nlp/progress.md
git commit -m 'nlp: prepare bounded S2 retry with watchdog and compact output'
```


Resume при необходимости делается **новым slug** с неизменным кодом: к указанной build-команде добавить `--resume-source thisarivson/nlp-hw02-kaggle-s2/VERSION` и заменить slug. Это только подготовка; дополнительная GPU-сессия автоматически не запускается. Shared source/kernel/datasets не удаляются; ключ не открывается и не включается в artifacts.


S2 push response: **Kernel version1 successfully pushed**. Pinned pullv1 подтвердил серверные private/GPU/T4 metadata и полное совпадение source cells с локальным notebook. Серверный pull выводит kernel source slug без номера; request attachment сохранён pinned S1/1 (у S1 единственная версия1). Первичный status — QUEUED, GPU-результаты пока отсутствуют. Итоговый status/time сверять с runs.json.

Попытка коммита S2: `git add` получил `Operation not permitted` при создании shared worktree `index.lock`. Index не изменён, рабочие файлы сохранены. Из обычного терминала в текущем worktree:

```bash
git diff --check
git add courses/nlp/homeworks/hw02-code-generation courses/nlp/index.md courses/nlp/progress.md courses/nlp/backlog.md dashboard.md knowledge/bibliography.md
git commit -m 'nlp: launch private S2 with fast sampler and equal-time M/B comparisons'
```

Повтор **v2** принят Kaggle2026-10-11; один разрешённый push, automatic restart отключён. Точный URL/время принятия и проверенный статус — `runs.json`.

Контроль v2 через225с после push: **RUNNING**. Pinned v2 pull: исходники совпадают, server private/internet/GPU/NvidiaTeslaT4 подтверждены. Live Dataset на момент проверки ещё не виден в `--mine`; реальные Secrets/upload пока не подтверждены. Коммит повторно заблокирован песочницей на shared `index.lock`; изменения сохранены.

## Разбор S2 v2 и восстановление S3

[№27b](../notes/27b-fast-sampling-s2-results.md) · [M29/B28](../notes/29m-28b-capacity-s2-results.md) · [summary/SHA/ресурсы](../results/kaggle-s2-v2-summary.json) · [CPU-аудит](../checks/kaggle-s2-v2-cpu-audit.json). Протокол выше и прежние записи запуска сохранены как история.

Pinned `version_label=v2`: **216 output files,1487845904байт**, скачаны в ignored `work/results/kaggle-runs/nlp-hw02-kaggle-s2/2/`. Дополнительно сохранены kernel log, version/list receipts и pinned source. Из-за последовательной задержки скачивание завершено12параллельными read-only downloads той же pinned API-выдачи; signed URL/auth не сохранялись. Все216 local SHA в CPU-аудите,33 producer candidate SHA receipts, SHA/bytes/tensor digests7PT проверены. Manifest166entries:165экспериментальных файлов присутствуют, размеры совпали; служебный `__notebook__.ipynb` не выдаётся API. ZIP отсутствует по зарегистрированному протоколу S2; CRC неприменим. Исходный input ZIP CRC/SHA проверен, summary/partial-флаги сверены.

Live download `thisarivson/nlp-hw02-s2-live` вернул403; paginated `--mine` не нашёл Dataset. В `work/results/kaggle-runs/nlp-hw02-s2-live/latest/download-check.json` сохранён read-only receipt. Kernel16,314с сообщает `LIVE: Kaggle Secrets unavailable; uploads disabled`; live-sync status/log и snapshot отсутствуют. Поэтому latest version скачать невозможно, live-only материалов нет; output содержит полные FP32 веса и AdamW, запланированный live содержал бы только FP16 точки без optimizer. Не читались значения Secrets; какой именно доступ не сработал, лог не сообщает.

|Опыт|Результат|Решение|
|---|---|---|
|27b|Greedy35token mismatches/4096; batch2048 passed,5,844GiB reserved; CPU saved k16=2702/4096|Rejected по обязательному gate; GPU k16 gate/testk8+k16 не выполнены, k не выбран|
|29 M|0updates/points/checkpoints|Missing после отказа27b|
|28 B|1553→1968→2341→2513/4096;373→497→590→618/991; все4timepoints aligned|Training complete; A/M/B pending безM/gates/test|

KaggleCOMPLETE означает окончание kernel. Внутренний итогpartial, GPU0exit1, GPU1exit0. Supervisor7860,524с; последний kernel export7877,358с≈131,29мин. B train7317,474с/28978computeupdates, batch128,3,960updates/s/506,89pairs/s, AMPskips0. Пик greedy evaluation5,109GiB, train peak не записан. Full test greedy/vote отсутствует у всех трёх вS2; heldout timings не подставляются. CPU повторил все сохранённые reference/generated programs, все B execution status/value и10 match sets exact; CE forward/GPU не повторялись.

**Что восстанавливается:** B stable latest main20553/cursor2630784 сAdamW/scaler/scheduler/RNG; отдельно full evaluated t04 (20553+4108/cursor3156608) и FP32 model-onlyt01–t04. M state отсутствует, стартует отseed42 после27b. A и reference остаются в pinnedS1/1. Stable latest не подставляется вместо оценённогоfinal.

Команда сборки S3 из `work/kaggle/`, **не выполнялась**:

```bash
python3 build.py --kind s2 --owner thisarivson --slug nlp-hw02-kaggle-s3 --budget-hours 3.25 --previous-source thisarivson/nlp-hw02-kaggle-s1/1 --resume-source thisarivson/nlp-hw02-kaggle-s2/2 --sample-batch-rows 2048
```

`--previous-source` обязан указыватьS1/1: там A/reference/cache identity. `--resume-source` прикрепляетS2/2 сB/27b. Только `--previous-source S2/2` неверен: код ожидает `gpu0-a26`. Resume требует новыйslug; исходные версии не меняются.

**Эта механическая сборка пока не готова к GPU push.** Неизменённый27b повторно прочитает failed candidate journal и остановится на тех же35 различиях. Изменение sampler source меняет `port_sha`/B `code_sha256`, а strict B resume проверяет identity; нужна проверенная совместимость сохранённого training-кода и новое пространство candidate journals. Следующий шаг — короткий CUDA-аудит35 различий и successful27b/testk8+k16, затем продолжениеM с сохранённымиBточками. Глобальный повтор B снуля не нужен. Live upload проверять коротко после восстановления доступа к прикреплённым Secrets.

Для повторного получения исходных артефактов, без запуска GPU:

```bash
python3 kaggle_tools.py output thisarivson/nlp-hw02-kaggle-s2 --version 2
python3 kaggle_tools.py status thisarivson/nlp-hw02-kaggle-s2/2
```

Аудит offline в CPU-окружении сTorch/tokenizers/numpy/sympy:

```bash
python ../checks/kaggle-s2-cpu-audit.py
```

В этом разборе Git/Kaggle push и новые GPU-сессии не выполнялись; большие outputs/PT/journals остаются ignored.

Коммит разбора S2 снова заблокирован при `git add`: песочница запрещает создать shared `.git/worktrees/study1/index.lock` (`Operation not permitted`). Index не изменён, небольшие правки сохранены. Из обычного терминала в текущем worktree:

```bash
git diff --check
git add courses/nlp/backlog.md courses/nlp/index.md courses/nlp/progress.md dashboard.md knowledge/bibliography.md
git add courses/nlp/homeworks/hw02-code-generation
git commit -m 'nlp: analyze S2 sampler gate and equal-time capacity B results'
```

## S3: №30 / №31 / №32, 90 минут

[Протокол](../notes/s3-data-representations-and-b-scoring.md) · [round-trip](../checks/kaggle-s3-roundtrip.json) · [tiny smoke](../checks/kaggle-s3-smoke.json) · [ноутбук](notebooks/nlp-hw02-kaggle-s3.ipynb).

GPU0: A/compact target снуля, fixedp006250/p012500, затем независимые equal-time branches при бюджете, затем B scoring без обучения. Последние25мин GPU0 резервируются Bscore. GPU1: A/числа на **исходном** таргете. Точный init26b/seed/order0–41/WSD/b128/fusedAdamW/AMP; raw baselines изresumeinput. S1/1 иS2/2 подключены pinnedkernel sources. Веса B — **оценённые t04**, SHAcbbe292c…f720a.

Преобразуется только требуемый prefix **готового**order, без пересортировки; preprocessing внеtrain timer, 2CPU encoderworkers наvariant. 64kblocks ограничивают число открытых mmapfiles. Scratch arrays/anchors/распаковка внеworking. Output безZIP иlive: latestfullstate, FP32model-onlypoints, restoredgenerations, metrics и Bscorebatches. Каждый завершённыйevalsplit/batch пишется сразу; watchdog доinstall, workers5100с, аварийныйexit6000с, backendtimeout6300с.

```bash
python3 build_s3.py
python3 kaggle_tools.py push --folder bundles/s3 --run-gpu
python3 kaggle_tools.py status thisarivson/nlp-hw02-kaggle-s3/VERSION
# После завершения, VERSION из push response:
python3 kaggle_tools.py output thisarivson/nlp-hw02-kaggle-s3 --version VERSION
```

CPU: `s3_roundtrip.py` вresearchvenv; `s3_smoke.py` вTorchCPUenv. Round-trip0/57702 обоих вариантов включает BPEencode/decode, восстановление и новое boundedexecution относительно прежних independentlyverifiedoriginalответов. Smoke: rarelayout/sign/type/repeat/20limit, новыеembeddings/старыеtensors, originalbatchorder, дваcooldown/полныйAdamWrollback, ownCE/exec, realBforward, независимыйsum/meanEOS/PADoracle, end-to-end65candidateJSON/SHA иshuffledIDs. ДоGPU не проверены memory/backend/FP16/fused/quality/timealignment/fullскоростьscoring. ПриERROR — pinnedlog идиагноз, **без повторногоpush**.

## Разбор S3 v1

`kaggle_tools.py output thisarivson/nlp-hw02-kaggle-s3 --version 1` скачал84файла
вignored `../results/kaggle-runs/nlp-hw02-kaggle-s3/1/`; API pinned statusCOMPLETE.
Внутриpartial/exits[1,1],36.85мин. [№30/31 и причина](../notes/s3-representations-results.md),
[summary](../results/kaggle-s3-v1-summary.json), [SHA/CPU audit](../checks/kaggle-s3-v1-cpu-audit.json).
API не выдаёт служебный `__notebook__.ipynb`; остальные manifest bytes совпали.
Полный SHA-инвентарь/producer receipts и original source/port identity прошли.

Оба worker остановились после полногоp006250 при resumed backward: `evaluate`
конвертировала FP32→FP16→FP32 параметры под `inference_mode`. В рабочем source
заменено на `no_grad`; [CPU regression](../checks/kaggle-s3-dtype-regression.json)
воспроизводит старое падение, проверяетactual eval и дваAdamW updates30/31.
CUDA ещё не проверена; frozenv1 notebook и downloadedsource не менялись,
новый bundle/notebook не собирался, push не выполнялся.

Latest каждойA — полныйstable6250 state, cursor800000, optimizerstep6250;
compute-история сохраняет уже потраченный cooldown,960000pairs. Model-only
p006250 — оценённый cooldown, его нельзя подменитьstablelatest. СкриптS3
пока стартует толькосinitial: будущийresume требует отдельного loader спроверкой
identity, model/AdamW/scaler/RNG/cursor и сохранениемистории при новомcodeSHA.
Преобразованныйprefixcache былscratch ивoutput не сохранён.

№30/31p012500/equal-time не получены → frozen gatespending. №32GPU scoring
не стартовал; generic32partial-marker не содержитscores. CPUвосстановление
иexecution exact; taxonomy/bins/McNemar записаны. [CPU-дополнение№32](../notes/32-candidate-ensemble-s3-results.md)
считаетBFP32 отдельно для всех requestedk8/k16/k32+greedy pools; команды
иresumechunks — [research README](../research/README.md#разбор-s3-v1).
CPUscore не равен отсутствующемуT4FP16output. Productionfull1919≤180с/fresh/official
ещё не измерены.

CPU-дополнение№32 завершено:168960scores/129050unique,30.9мин forward+chunkI/O;
winner поk16/tune — B-meanτ0.2. Eval+0.29п.п./p0.392 иold+0.40/p0.454 кvote;
кпрежнему правилу0/−1ответ, убедительного выигрыша нет. Сетки иполныепулы
зафиксированы до eval reporting; B вproduction не добавлять на этих данных.

Коммит разбораS3 заблокирован на `git add`: shared worktree `index.lock`,
`Operation not permitted`. Индекс не изменён, правки оставлены; все новые
report/code файлы меньше1MiB, PT/downloads/массовыеCPU scores ignored.
Новых Kaggle push илиgit push не было.

S3 принят: **Kernel version1 successfully pushed**, 2026-10-12T08:12:56.484030+00:00. [Версия](https://www.kaggle.com/code/thisarivson/nlp-hw02-kaggle-s3/versions?versionNumber=1). Один разрешённый GPU push, automatic restart выключен. Проверенный контрольный статус записывается в runs.json.

Коммит S3 заблокирован: `git add` не смог создать shared `.git/worktrees/study1/index.lock` (`Operation not permitted`). Index не изменён, все правки сохранены. Из обычного терминала в этом worktree:

```bash
git diff --check
git add courses/nlp/homeworks/hw02-code-generation courses/nlp/index.md courses/nlp/progress.md dashboard.md knowledge/bibliography.md
git commit -m 'nlp: launch S3 representation experiments and B candidate scoring'
```

Контроль S3v1 через220.1с послеpush: **RUNNING** (2026-10-12T08:16:36.604609+00:00). Pinnedv1pull подтвердил sourcecells exact, serverprivate/GPU/internet/T4. Repush0; новыеGPUметрики ещё не скачаны. Полный реестр — runs.json.

Коммит L1 заблокирован песочницей при создании shared worktree `index.lock`; index не изменён, рабочие файлы сохранены. Из обычного терминала в этом worktree:

```bash
git add courses/nlp/homeworks/hw02-code-generation courses/nlp/index.md courses/nlp/progress.md courses/nlp/backlog.md dashboard.md knowledge/bibliography.md
git commit -m "nlp: prepare long C/D L1 with verified full compact cache and live resume"
```

## L3 — обе C продолжаются на аккаунте друга

[План](../notes/l3-plan.md), [anchors](l3-anchors.json), [доступ](../checks/kaggle-l3-source-access.json). GPU0: C-main126452/cursor16185856/main46370.1434с; GPU1: **свой** C-long2x123842/cursor9815936/main46370.2458с. L2 runtime/recovery/eval/live/watchdog/sampler переиспользованы без изменения; дополнены только выбор anchors и конфигурация worker. Weighted order скачивается из L2 live с прежним SHA/seed, не перестраивается. LR0.0015, optimizer/scaler/RNG/history/cursors сохраняются, warmup не повторяется.

Аккаунт CLI: `KAGGLE_CONFIG_DIR=~/.kaggle-friend` (аккаунт друга), иначе thisarivson. Оба thisarivson datasets и output кеша доступны после UI-шаринга. **Выбран оригинальный pinned kernel output `/2`**, manifest SHA46ef881b…902be. Запасной private `friend1/nlp-hw02-compact-cache-verified` тоже создан/ready, remote manifest SHA exact; перед upload все shard SHA и свободное место проверены. Private `friend1/nlp-hw02-l3-live` заранее создан/v1 ready. Credentials не открывались и в артефакты не попадают.

Main+9ч18мин на каждую T4, одна mid-оценка на+4ч39мин с cooldown15мин; final cooldown50мин и прежние heldout4k/old991/test1919/K8 tune2048 journals. Workers40260с, hard40800с (**11ч20мин**, запас240с до11.4ч), backend12ч. Два таких hard wall +18мин smoke помещаются в25ч. Финальный live-sync ограничен оставшимся hard резервом (не более420с при полном worker budget).

Из `work/kaggle`, CPU environment с Torch/numpy/tokenizers:

```bash
python build_l3.py
python build_l3.py --gpu-smoke
python l2_tests.py
python l3_tests.py
python l3_static_check.py
# Запасной cache source, только при необходимости:
python build_l3.py --cache-source friend1/nlp-hw02-compact-cache-verified
```

[Два CPU resume receipts](../checks/kaggle-l3-real-resume-C-main.json) / [C-long2x](../checks/kaggle-l3-real-resume-C-long2x.json): реальный full SHA и первые два batch128 шага reference↔production run bit-identical; PT и временные checkpoint copies удалены после проверки. CPU FP32/enabled scaler/Adam без fused; CUDA численная идентичность — GPU-smoke ревьюера. Скрипт воспроизведения `l3_cpu_resume_check.py`; скачивать по одному pinned PT, путь к прежнему compact cache и restored weighted order передавать аргументами.

Bundles `bundles/l3` / `bundles/l3_gpu_smoke`, tracked notebooks [боевой](notebooks/nlp-hw02-kaggle-l3.ipynb) / [смоук](notebooks/nlp-hw02-l3-gpu-smoke.ipynb). **GPU-смоук и боевой push здесь не выполнялись.** Ревьюер проверяет источники/метаданные/CPU и сначала запускает подготовленный private smoke без Secrets. Боевой UI Save & Run All требует attached аккаунт друга KAGGLE_USERNAME/KAGGLE_KEY**, startup проверяет username и owner-private live upload/ready. Ничего на Kaggle не удалять; final-inference файлы относятся к независимому агенту.

Коммит L3 заблокирован песочницей на shared `.git/worktrees/study1/index.lock`: `Operation not permitted` при `git add` явного списка только L3-файлов. Index этой попыткой не изменён, чужие final-inference правки не staging.

Ревью L3: [отчёт](../notes/l3-review.md), [GPU v3](../checks/kaggle-l3-review-gpu-smoke-v3.json). 34 L2/10 L3 регрессий, CUDA-фазы/NaN/crash/fallback SHA пройдены; private боевой аккаунт друга v1 опубликован, ожидаемый ERROR на Secrets подтверждён. Для UI-запуска прикрепить friend KAGGLE_USERNAME/KAGGLE_KEY.

## Чистый cooldown — только локальная CPU-подготовка L4

[Анализ и A/B](../notes/clean-cooldown-analysis.md), [JSON](../results/clean-cooldown-analysis.json), [CPU replay](../checks/clean-cooldown-cpu.json). Из9 725 737 исходных compact rows strict mode≥2/3+singletons оставляет **9 506 623**, −2,2529%; одинаковые protected exclusions усилены нормализацией вопроса, найден один test-overlap и исключён из всех orders. Heldout/old/fresh пересечения0. Новых reference/targets нет.

Ignored `../results/clean-cooldown/` содержит `strict-two-thirds` / `all-modal-ties-kept` / `ordinary-protected-safe` `uint32` orders, SHA, `order.json`, `split.json` и общий manifest; неизменённый `l2_sampler.load_order` и полный replay прошли. Повторно проверены обе L2 per-example:17 конфликтных raw-source proxy, широкие CI, прямого heldout/train пересечения нет. **Kaggle Dataset не создан, push/run/GPU0; текущие L3 файлы и runs.json не изменены.** Предлагаемый L4 — четыре50-мин cooldown из stable anchors, uniform control/R1 для каждой C, frozen tune/eval2048 и paired95/97,5% CI; прогноз2–2,25ч наT4×2. Ожидается небольшой или нулевой эффект; при discordance5–10% eval2048 имеет≈1,1–1,6п.п. полуширины adjusted CI. Это протокол будущего эксперимента.


## L3 final → inference v4 и L4 (подготовка14.10.2026)

Финальные данные уже в `../results/kaggle-runs/nlp-hw02-kaggle-l3/final/`; повторный kernel output download не нужен. [Финальный разбор](../notes/l3-results.md#финал-l3), [L4(a)](../notes/l4-plan.md). Другой seed weighted ветки не создаётся: обе текущие C продолжаются из собственных uncooled latest L3v12.

Локальная сборка и проверки из этого каталога (Python с Torch/numpy/tokenizers/nbformat; анализ — также matplotlib):

```bash
python ../research/analyze-l3-final.py
python build_final_inference_l3.py
python final_l3_checks.py
python build_l4.py
python build_l4.py --gpu-smoke
python l2_tests.py
python l3_tests.py
python l4_tests.py
python l3_static_check.py
python l4_static_check.py
```

Последний `final_l3_checks.py` пишет ready=true только после всех checks: `final/next-inference.json` указывает `bundles/final_inference_l3`. Private **friend1/nlp-hw02-final-inference-l3**, internet off,1T4/GPU0; inputs L3 live+thisarivson26b-inputs; one cold test+tune, backend timeout720. Observer выполняет push отдельно; прежний bundle `final_inference` не менялся. GPU gate pending.

L4: private **friend1/nlp-hw02-kaggle-l4**, internet on/T4×2, `bundles/l4`; smoke `bundles/l4_gpu_smoke`/17мин hard. Main30600с, mid15мин, final50мин, worker37380с, hard37800с. Backend ceiling12ч — внешний предел; runtime Watchdog10ч30мин. Новые tags77580/92880. Private **friend1/nlp-hw02-l4-live v1** создан, listing/private/status ready проверены; ничего не удалялось.

Friend API только с переменной ниже; credential directories не читать:

```bash
KAGGLE_CONFIG_DIR=~/.kaggle-friend python l4_prepare_api.py probe
KAGGLE_CONFIG_DIR=~/.kaggle-friend python l4_prepare_api.py create
```

`create` idempotent: существующий private dataset не перевыгружает. В этой подготовке первый status endpoint временно403, после появления dataset mine listing/private/status ready подтверждены; [receipt](../checks/kaggle-l4-live-created.json). Ни один kernel не pushed.

Реальный resume воспроизводится через `l3_cpu_resume_check.py --session l4 --model C-main`/`C-long2x`, `--anchor-root` с одним скачанным pinned full PT, `--cache` прежний verified compact-cache, `--weighted-order` исходный weighted-order (проверенный SHA72bc3dfd… из L3 live; в локальном L2 snapshot npy нет), `--out` отдельный JSON. Скачивать по одному из l3-livev12 через `l4_prepare_api.py download --file <pinned-file> --path <scratch>`; checker проверяет producer SHA/finite/full counters/history и два actual batch128 шага, затем удаляет PT. Не менять identities/runtime/sampler SHA; CUDA parity — отдельный smoke.

2026-10-14: [Ревью L4](../notes/l4-review.md) — L2 34/34, L3 10/10, L4 5/5; smokev1 612,7с/808 и671+64steps/peak reserved11,06GiB, original+fallback360SHA каждый. Private [боевой L4v1](https://www.kaggle.com/code/friend1/nlp-hw02-kaggle-l4/versions?versionNumber=1) ERROR_expected_Secrets_gate, без training. UI friend Secrets→Save & Run All; live target l4-live, source l3-livev12. Receipts/runs и точные fallback-команды — в ревью.


## P1 — вместо L4

**Решение14.10.2026: L4 не запускать.** [План P1](../notes/p1-plan.md): private `friend1/nlp-hw02-kaggle-p1`, T4×2/internet on, GPU0 A/B cooldown31372updates из одного latest L3 C-long2x, GPU1 reviewer C code→question с нуля. Hard37800с=10ч30мин, worker37380с; отдельные caps A13800/B17400, evaluation1800с после каждой. A≈3ч27мин по L3; B использует graph loss truncation с безопасным двухпроходным OOM fallback. [Независимое ревью](../notes/p1-review.md): smokev1 COMPLETE, B +2,19% времени, productionv1 отправлен; ожидаемый ERROR Secrets gate подтверждён, training0. Live **friend1/nlp-hw02-p1-live v1** создан private, ready через проверку файлов (status endpoint403). L4, L3 и final inference bundles не изменены.

Команды из `work/kaggle/`, Python с Torch/numpy/tokenizers/nbformat:

```bash
python build_p1.py
python build_p1.py --gpu-smoke
python l2_tests.py
python l3_tests.py
python l4_tests.py
python p1_tests.py
python p1_static_check.py
KAGGLE_CONFIG_DIR=~/.kaggle-friend python p1_prepare_api.py probe
# create идемпотентен: live уже создан; новых kernel versions не создаёт.
KAGGLE_CONFIG_DIR=~/.kaggle-friend python p1_prepare_api.py create
```

Scorer CPU smoke, из того же каталога:

```bash
python ../research/score-reviewer.py --smoke --out ../research/outputs/reviewer-smoke
# После P1 скачать только итоговый reviewer FP16 и выполнить:
python ../research/score-reviewer.py --weights ../results/p1/reviewer-final-fp16.pt --out ../research/outputs/reviewer-scores
```

Inputs по умолчанию — существующие S1 `inputs` (heldout/tokenizer); journals — L3 `final`. Другие пути задать `--inputs`/`--journals`. Все32768 saved samples получают sum/mean reviewer logprob по `represented_code`, вопрос+EOS, без PAD/BOS; duplicate forwards дедуплицируются с сохранением голосов. SHA chunks позволяют продолжить CPU scoring. Main8/long8/mixed4+4: voting+reviewer, cluster exp/max/mean; beta fit на другой половине tune, normalized duplicates в одной половине, out-of-fold paired CI против voting. Текущее качество случайного smoke не использовать для выбора recipe.

[CPU resume](../checks/kaggle-p1-real-resume-C-long2x.json): один actual full L3 PT, два batch128 reference↔P1 update bit-exact и повторный exact restore; PT удалён. Воспроизведение: `p1_cpu_resume_check.py --anchor-root <scratch> --cache <verified-compact-cache> --weighted-order <inherited-weighted-order> --out <receipt.json>`. Downloader: friend-only `p1_prepare_api.py download --file model-C-long2x__checkpoints__main-2a27307d7a48.pt --path <scratch>`; не скачивать весь live и не копировать кеш. Historical weighted order SHA72bc3dfd… также совпадает с локальным L2 GPU-smoke1/2. CPU FP32/enabled scaler/fused-off не заменяет CUDA check.

Только независимому ревьюеру, **в этой подготовке команды push не выполнялись**:

```bash
KAGGLE_CONFIG_DIR=~/.kaggle-friend python p1_review_api.py push friend1/nlp-hw02-p1-gpu-smoke --path bundles/p1_gpu_smoke
# Явную VERSION взять из квитанции push, затем status/metadata/output:
KAGGLE_CONFIG_DIR=~/.kaggle-friend python p1_review_api.py status friend1/nlp-hw02-p1-gpu-smoke/VERSION
KAGGLE_CONFIG_DIR=~/.kaggle-friend python p1_review_api.py output friend1/nlp-hw02-p1-gpu-smoke/VERSION --path ../results/kaggle-runs/nlp-hw02-p1-gpu-smoke/VERSION
# После успешного независимого audit:
KAGGLE_CONFIG_DIR=~/.kaggle-friend python p1_review_api.py push friend1/nlp-hw02-kaggle-p1 --path bundles/p1
```

Production requires attached friend `KAGGLE_USERNAME`/`KAGGLE_KEY`; API version без Secrets должна отказать до обучения. UI Save & Run All только **P1**, от аккаунта друга. Smoke без Secrets/live, hard17мин/backend18мин, full C/AMP/b128: A/B64updates, greedy64/old16/test256/K8-36, reviewer warmup20→constant→cooldown. GPU0 intentional exit1 после проверок; primary `p1-smoke-summary.passed`, exits[1,0], healthy reviewer complete и отсутствие stop-request — успешный smoke, session partial ожидаем. Проверить все output SHA, full restore, uniform reviewer protected exclusion, оба journals, NaN/serialized CUDA parity, memory и прогноз cap B; реальный live-auth/startup insurance проверит production gate.

Fallback пересобирается явно; SHA/pin/order сохраняются:

```bash
python build_p1.py --cache-source friend1/nlp-hw02-compact-cache-verified
python build_p1.py --gpu-smoke --cache-source friend1/nlp-hw02-compact-cache-verified
python p1_static_check.py
```

Артефакты: `model-pilot-a/weights/long-fp16.pt`, `model-pilot-b/weights/long-truncation-fp16.pt`, `model-reviewer/weights/final-fp16.pt`; reviewer `latest.json` — full **до cooldown**. Live final snapshot сохраняет все три FP16 и два latest stable. Исходный L3 не перезаписывается. Sampled T4 throughput измерен; полный10,5ч run/live auth ещё не измерены; partial не объявлять complete и T/q/peak не менять по результатам. [17 P1 tests](../checks/kaggle-p1-regressions.json), [статика](../checks/kaggle-p1-static.json), [scorer](../checks/reviewer-score-cpu-smoke.json), [runs](runs.json).

Попытка коммита P1: `git add` явного P1-only списка получила `Operation not permitted` на shared worktree `index.lock`; индекс не изменён. Изменения сохранены, коммит выполнить из обычного терминала после проверки diff. [Квитанция](../checks/kaggle-p1-git.json).

## Адаптивный final inference v5 на L3 (CPU, без push)

[Схема/числа](../notes/final-inference.md#v5-адаптивное-сэмплирование-на-l3): после первого T4 guard140с выбран greedy→K2+K2→K4+K4, numeric unanimity1.0. `sequence_budget=1024`, greedy1024/stage1 512×K2/stage2 512×K2; глобальный приоритет минимальной доли лидера, обе модели подряд на батч. Generation guard160с,20с до потолка180. Frozen vote1511/2048; прогноз test132,1/tune136,7с, осторожно147,2с. V3/V4/P1 не изменены. Бандл `bundles/final_inference_l3_v5`, id `friend1/nlp-hw02-adaptive-inference-l3-v5`, title `NLP HW02 adaptive inference L3 v5`: один согласованный slug существующего kernel. Private, internet off,1T4, один cold test1919+tune2048,720с.

Из `work/kaggle`, Python с Torch/tokenizers/sympy/nbformat (локально `/tmp/nlp-s3-env/bin/python`):

```bash
python final_v5_analysis.py --simulation-script /path/to/selection_analysis.py
python build_final_inference_v5.py
python final_v5_regressions.py
python final_v5_checks.py
```

Сборка и проверки сбрасывают `next-inference-v5.json` в false перед стартом. Последняя команда публикует ready=true только после CPU/static/offline/regression gates; GPU gate остаётся pending. Проверены stop/LP/graph/sequence budget/partial guards/OOM/global priority/scorer interface,2048winners reexecution и статическая one-test suite. Tiny decoder повторно запускается с выбранным K4. Reviewer implementation не входит в bundle. Push/запуск этими командами не выполняются.
