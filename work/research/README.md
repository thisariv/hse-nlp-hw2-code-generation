# CPU-аудит данных ДЗ2

[B7: результаты](../notes/b7-results.md), [анализатор](analyze-b7-results.py), [JSON](../results/b7-results.json), [CPU receipt](../checks/b7-results-cpu.json). Запуск из корня: `python work/research/analyze-b7-results.py` с NumPy/Matplotlib; C++/Torch не нужны. Проверяет184producer SHA и211унаследованныхSHA, greedy/CE/length/old991/парные CI,320K8 и шесть фиксированных составов1/1/1→1/8/8,12288production/independent winner rows;10grouped split-half. Пошаговые B7 training-журналы и tensors отсутствуют: проверка обучения по сводкам/endpoints. API/GPU/forward/sampling/новое исполнение программ не вызываются; frozen B6/E1/E2 и submission не меняются. Критерий v12 не пройден, сохранитьv11.

[B5: результаты](../notes/b5-results.md), [анализ](analyze-b5-results.py), [JSON](../results/b5-results.json), [сетка22025политик](../results/b5-ensemble-grid.csv). Запуск: `python work/research/analyze-b5-results.py` из корня с NumPy/Matplotlib и C++17 compiler `c++`. [Ускоритель](b5-replay.cpp) компилируется во временную библиотеку;399360native/Python bits и18432production/independent winner rows проверяются.207producerSHA/127170main/14631CD,384K8; все недостающие stepJSONL должны лежать в локальном snapshot сисходнымmanifest. Папка `nlp-hw02-b5/1` — download label, пользователь сообщилproductionv6. Сам скрипт не вызываетAPI/GPU/forwards/sampling/execпрограмм; [отдельная read-only квитанция скачивания](../checks/b5-results-download.json). CI/search иtiming описательные; B6N иv9 — планы.

[B4: результаты](../notes/b4-results.md), [скрипт](analyze-b4-results.py), [JSON](../results/b4-results.json), [сетка4438W+C политик](../results/b4-ensemble-grid.csv). Запуск: `python work/research/analyze-b4-results.py` из корня с NumPy/Matplotlib. Проверяются230producer SHA,71087main/10485cooldown updates,32570общих stream prefixes, greedy/CI/длины/CE и320K8journals; график по парам/времени и expanded C-only control. Main/cooldown journals должны быть скачаны из pinned B4 output; full weights/order не нужны. Новые forwards/GPU/sampling/execution/API не вызываются. Сетка и CI исследовательские на повторно использованном tune.

[CPU-проверка переобучения](../notes/overfit-check.md), [JSON](../results/overfit-check.json), [NLL/count/correct](../results/overfit-check-rows.npz). `check-overfit.py` считает FP32 train4096/heldout4k CE и teacher-forcing token accuracy по model_spec; seed2026101503, protected SHA всех9,73M строк, парные CI. Torch/NumPy/tokenizers, без exec/GPU/обучения; `--summarize-only` пересчитывает интервалы без forwards.

[R1: CPU-разбор](../notes/r1-results.md), [JSON](../results/r1-results.json). `analyze-r1-results.py` проверяет manifest/IDs/splits, считает парные bootstrap CI, воспроизводит K8 pools и frozen v7, строит train CE по реальным kernel timestamps. NumPy/Matplotlib, без Torch/сети/GPU/нового sampling; periodic heldout CE отсутствуют. Отдельно `python checks/r1-results-cpu.py` из `work/` проверяет реальные full stable/FP16 и Adam moments на CPU с Torch.

[D1: поиск подозрительных reference](../notes/d1-plan.md). `analyze-d1-labels.py prepare`
сверяет frozen clean-cooldown/noise receipts и строит компактный seed по исходным
row identities. `analyze --input RESTORED --output results/d1-labels` проверяет SHA,
считает согласие двух L3 против reference, длины/дубли/конфликты и создаёт
детерминированные L2 orders control/A/B/C. Ошибки/неисполненные/обрезанные строки
сохраняются; `--allow-partial` явно разрешает неполный проход. Модельные программы
никогда не становятся таргетами. Анализу нужен NumPy; GPU/Torch/сеть ему не нужны.

[L3 mid: CPU-разбор](../notes/l3-results.md), [JSON](../results/l3-results.json), [кривая](../results/l3-learning-curve.png). `analyze-l3-results.py` использует helpers L2 и 15 локальных mid JSON/TXT: сверяет10174строки/IDs/reference/bins, парные CI, продолжение counters и четыре коротких точки. Запуск в CPU-окружении с NumPy/Matplotlib: `python work/research/analyze-l3-results.py`. Сеть/GPU/forwards/новое исполнение программ не нужны; main stability/logs ещё не получены.

## Чистый cooldown: полная перепись compact train

[Разбор и дизайн L4](../notes/clean-cooldown-analysis.md),
[числа](../results/clean-cooldown-analysis.json). `clean-cooldown.py` использует
все 60 локальных compact/raw-шардов, original BPE offsets и уже исполненные
reference нормализованных дублей. Группы пересчитываются после пересечения с
compact row IDs; heldout/model answers не участвуют в отборе. Требуются NumPy
и PyArrow. Только CPU, без Torch, сети и новых исполнений программ.

```bash
python work/research/clean-cooldown.py
python work/checks/clean-cooldown-cpu.py
```

Создаются три ignored order в `results/clean-cooldown/`: строгая мода ≥2/3,
все модальные кластеры с сохранением ties и обычный uniform control. В каждом
одинаково усилены protected exclusions до lower+collapse whitespace. Формат
`uint32`, исходные global compact indices, `order.json`/`split.json`/SHA;
совместимость проверяет неизменённый `l2_sampler.load_order`. Временные массивы
только в RAM; постоянные orders занимают около 110 MiB. Kaggle Dataset не создаётся.
Диагностика heldout отдельно помечает отсутствующее прямое пересечение с train
и proxy по исключённым raw-reference, с удалением собственной heldout-пары.
Повторный CPU replay также сверяет обе L2 по IDs/SHA/numeric correctness,
воспроизводит proxy counts/CI для heldout/tune/eval и сохраняет сценарии
чувствительности eval2048 в `L4_sensitivity` [receipt](../checks/clean-cooldown-cpu.json).

## L2: CPU-разбор сохранённых журналов

[Итоги](../notes/l2-results.md), [метрики/CI](../results/l2-results.json), [кривая](../results/l2-learning-curve.png). `analyze-l2-results.py` читает ignored pinned L2v2 output/livev11 и C-only L1livev7, сверяет producer SHA/source/IDs, воспроизводит `exp27_decode.choose` без gold в выборе и считает paired CI. Model forward/GPU/новый sampling/fresh не нужны. Зависимости — NumPy и Matplotlib; последнего в историческом requirements ниже нет, установить отдельно в своё CPU-окружение.

```bash
python -m pip install numpy matplotlib
python work/research/analyze-l2-results.py
```

Исходные файлы нужны в `../results/kaggle-runs/nlp-hw02-kaggle-l2/1/`; имя1 — локальное, серверная версия2. [Компактная выгрузка и версии](../kaggle/README.md#l2--результаты-боевой-v2). Четыре source-файла протокола побайтно совпали с pinned producer. Fits/time ranges описательные; полного mixed таймера одной T4 здесь нет.

## Исторический аудит данных

Результат: [заметка](../notes/data-research-ceiling.md), [числа](../results/data-research-ceiling.json).
Обучения, GPU-инференса и Kaggle push здесь нет. Подставленные программы — только
диагностические кандидаты; они не становятся обучающими данными или таргетами.

Запуск из корня репозитория, Python 3.14. Локальная `.venv/` и `outputs/`
игнорируются Git. Установить зависимости из `requirements.txt`.

```sh
python3 -m venv work/research/.venv
work/research/.venv/bin/pip install -r work/research/requirements.txt
```

Запускать скрипты этим Python, с `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4`:

1. `census.py` — проверка SHA всех 60 raw Parquet, полная перепись точных и
   нормализованных дублей. Нужны уже сохранённые raw в ignored staging.
2. `targets.py` — случайная выборка по 1000 eligible строк из каждого шарда,
   исполнение оригиналов и трёх нормализаций, BPE8k и числа.
3. `noise.py` — все reference нормализованных дублей, проверка равенства
   текстов внутри хеш-групп, bounded CPU execution и модальные ответы.
4. `retrieve.py` — вложенные индексы 200k/1M/весь finite-кеш 26a+26c; точный
   sparse cosine top20 в фиксированном train-only TF–IDF пространстве.
5. `retrieval.py` — подстановка чисел по позиции/локальной роли,
   исполнение, similarity, AST-скелеты, saved greedy A/6.4M.
6. `sampling.py` — только чтение сохранённых кандидатов и результатов S1.
7. `routing.py` — пороговые inference-правила, выбранные по old991; primary4k
   используется для проверки. Gold не участвует в самом правиле.
8. `alignment.py` — дополнительный baseline с выравниванием слов и чисел:
   допускает лишние числа в одном из вопросов, проверяет неоднозначности.
9. `finalize.py` — компактный JSON, QA и CPU-only диагностические дополнения.
10. `write-note.py` — заметка по агрегатам и ранжированные гипотезы/бюджеты вJSON.

`common.py` использует прежний bounded executor №26: AST-фильтр, разрешённые
math/sympy, wall timeout 1 с, CPU 2 с, четыре spawned worker. Изменённый каркас
восстанавливается до исполнения. Удаление пустых строк защищено равенством AST;
наивный вариант отдельно выявил одну newline-sensitive программу.

Индексы допускают повторы исходных train-строк. На границе равных cosine внутри
шарда библиотека выбирает произвольный набор; объединение сохранённых top20
сортируется по score, затем `(shard,row)`. Каждая точка использует ту же
TF–IDF/IDF-выборку; отличие —
только доступные строки. Числа test используются только для retrieval-сравнения;
test gold отсутствует. Fresh1024 не читается и не оценивается, кроме списка
защищённых вопросов №26a.

Тяжёлые кэши сохраняют завершённые этапы. После изменения правил предобработки
нужно удалить только соответствующие новые исследовательские кэши из `outputs/`
или выбрать новый каталог; исходные артефакты домашки скрипты не изменяют.

## Выбор кандидата на CPU

[Протокол и результат](../notes/candidate-selection-cpu.md),
[JSON](../results/candidate-selection-cpu.json). Протокол записан до измерений:
SHA256 ID → tune2048/eval2048, old991 только контроль. Все параметры pure/mixed
выбираются на k16/tune и переносятся на k8/k32; all65 отдельно.
Сохранённые candidate/execution SHA и code identity проверяются; недостающие
execution исполняются песочницей №26. Fresh/test не читаются, обучения нет.

Первичная фиксация протокола в ignored `outputs/candidate-selection/preregistration.json`
включает UTC, SHA и текст до `## Результаты`. В новом checkout скрипт восстановит
её из `protocol` сохранённого итогового JSON с исходной датой регистрации.
Выполнить:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 work/research/.venv/bin/python work/research/candidate-selection.py
work/research/.venv/bin/python work/research/write-candidate-selection-note.py
```

JSON содержит все tune-сетки, выбранные параметры, eval/old McNemar, парные CI,
длины каждого правила/пула, oracle≥2 и ранги правильных кластеров. Локальный
`selected-clusters.npz` хранит индексы выбранных кластеров. Суммарное время
исследования не является production inference timer; 180 с на T4 не проверены.
Исторический контроль уточнён: k16/T0.7 без greedy2707/4096, с greedy2703/4096.

## Разбор S3 v1

[№30/31](../notes/s3-representations-results.md),
[№32 и CPU-scoring B](../notes/32-candidate-ensemble-s3-results.md),
[summary](../results/kaggle-s3-v1-summary.json).
Downloaded v1 остаётся неизменным; большие outputs, повторное execution и scores B
живут в ignored `results/kaggle-runs/` и `research/outputs/`.

Запуск из корня. `audit`, `s3-score-b-cpu.py` и dtype regression требуют Python с
Torch/numpy/tokenizers/sympy/pyarrow; `selection` и writer — прежнюю researchvenv.

```sh
python work/research/s3-analysis.py audit
python work/checks/kaggle-s3-dtype-regression.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 python work/research/s3-score-b-cpu.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 work/research/.venv/bin/python work/research/s3-analysis.py selection
work/research/.venv/bin/python work/research/write-s3-notes.py
```

`audit`: pinned manifest bytes, локальный SHA-инвентарь/producer receipts, frozen
source/port identity, full latest/AdamW/scaler/RNG/cursor, восстановления,
generation/control alignment, exec/McNemar/bins и прежняя taxonomy. CPU replay
всех восстановленных программ, обоих control points и reference кешируется по
SHA списка code; wall3с/CPU2с/2workers. Большие int вне float range сохраняются
точно и считаются outside-number; остальные правила taxonomy прежние.

`s3-score-b-cpu.py`: **CPU FP32**, pinned S2/2 B t04, decoded code+EOS/PAD policy
изS3. Только greedy и первые32 T0.7 — все168960 кандидатов5120 вопросов,
достаточно для полных пуловk8/k16/k32; unused T0.5/T1.0 не считаются.
Дедуп внутри вопроса, глобальный length-bucketing, атомарные chunks/SHA и resume.
Source SHA и ID/code alignment не позволяют переиспользовать чужой scoring.
Это новый CPU score, а не отсутствующий FP16 output KaggleS3.

`selection`: точный прежний hash tune2048/eval2048,old991; параметры наk16/tune,
после freeze применяются кk8/k32. Если CPU scoring ещё partial/отсутствует,
отчёт включает A-only и явно pending B/A+B; подвыборка не оценивается.
Сетки A/B mean — прежниеτ; A+B — одинλ∈{0,0.25,0.5,1,2,4}, mean-combination
сфиксированнымτ1. С/без exact-code dedup; vote-dedup использует tie послеdedup.
JSON хранит все tune grids, parameters, eval/old Δ/McNemar/CI/длины и закрытый
oracle gap относительноvote и прежнего tune-победителя. Ни scorer, ни выбор
не получают gold/reference length как признаки. Full T4 timer≤180с не измеряется.


## P1: reviewer scoring на CPU

[score-reviewer.py](score-reviewer.py): frozen L3 final `represented_code` → log P(question|code), sum/mean с question EOS. Все32768samples обеих C +2048greedy C-main; duplicate forwards с сохранением multiplicity, SHA-resume chunks, voting+reviewer/cluster exp/max/mean. β и отдельно семейство/normalization/β подбираются на другой половине tune; normalized questions остаются вместе. Main8/long8/mixed4+4/8+8 с/безgreedy, paired CI против собственного voting и число исправленных/сломанных oracle-misses. [Протокол](../notes/p1-plan.md), [полный результат](../notes/p1-results.md), [CPU checks](../checks/p1-results-cpu.json).

```bash
python score-reviewer.py --smoke --out outputs/reviewer-smoke-v2
python score-reviewer.py --weights ../results/kaggle-runs/nlp-hw02-kaggle-p1/final/model-reviewer__weights__final-fp16.pt --out ../results/p1-reviewer --threads 4 --batch 16
```

Полный trained reviewer CPU pass выполнен:34816candidates/26291unique pairs/all2048tasks,≈9мин вместе с анализом; jointCV full17 Δ0 [−0,39;+0,39]п.п. Summary (`../results/p1-reviewer/summary.json`, нет в репозитории), identity и benchmark сохранены; score chunks/progress ignored локально. `--benchmark-only` измеряет пять length-quantile batches; `--max-tasks 1024 --subset-seed 2026101402` даёт случайную подвыборку задач при слишком долгомCPU. Для P1 она не понадобилась. Веса должны иметь identity model=reviewer; `--inputs`/`--journals` задают pinned local snapshots. Current tiny smoke136candidates, исторический smoke128 — [receipt](../checks/reviewer-score-cpu-smoke.json); оба не измеряют качество random reviewer.

[analyze-p1-results.py](analyze-p1-results.py) проверяет COMPLETE/weights/start+restore+stream receipts, greedy4096/991, fixed split/bins, frozen K8/mixed и adaptive replay,31372строки truncation, reverse CE и scoring summary; пишет [p1-results.json](../results/p1-results.json) и3графика. Нужны NumPy/Matplotlib, без Torch-forward/нового исполнения программ. Запуск из корня: `python work/research/analyze-p1-results.py`. Все immutable score chunk SHA, sum/mean/count, реальные FP16 finite и отсутствие gold в selection проверяет [p1-results-cpu.py](../checks/p1-results-cpu.py).
## Архитектура и бюджет инференса

[Решение](../notes/architecture-decision.md): `prepare-shape-lengths.py` проверяет реальные cache offsets/order и сохраняет компактные joint lengths; `analyze-architecture-decision.py` воспроизводит384K8 журнала/кривые0…16, adaptive v6/v7,18hybrid pools и сценарии40GPU-ч. SingleK9–16 не наблюдались. `write-architecture-note.py` обновляет таблицы заметки из JSON. После пользовательского T4: `python research/analyze-architecture-decision.py --bench /path/to/shape-bench/summary.json`, затем writer. Новый sampling/execution/GPU/push не вызываются; [shape bench](../kaggle/notebooks/nlp-hw02-shape-bench.ipynb) имеет отдельные CPU/static checks.
