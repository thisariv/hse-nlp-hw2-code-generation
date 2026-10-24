# Чистая сдача ДЗ2 NLP: v2

Финальный метод: **W B7 (greedy-якорь) → C-long2x B7 → P1-B**,
адаптивно **1/1/1 → 1/8/8**, T0.7, seed270042, sequence budget **640**.
Это выбранный пользователем вариант v2: public **80,10%**, прежний свежий
tune **1557/2048**, cold test **111,421 с**. Финальная упаковка: test **122,66 с**, tune **1560/2048**, пик reserved
**11,12 GB**.

**Разброс между прогонами.** Метод сэмплирует кандидатов (T = 0,7), а порядок досэмплирования и уплотнения батчей на GPU зависит от тайминга, поэтому два запуска ноутбука дают немного разные ответы (≈150–200 различающихся численных ответов из 1919). Качество при этом одинаковое: два прогона v2 дали на public 80,10% и 79,37%, на свежем tune — 1557 и 1560 из 2048. Ожидаемый результат одного запуска на public — около 79,4–80,1%.

## Что где лежит

- [inference.ipynb](inference.ipynb) - читаемый Run All, полный cold timer,
  test1919 → `answers.jsonl`, затем отдельный tune2048; сохранённые outputs.
- [code/hw2_codegen/](code/hw2_codegen/) - обычные модули. `model.py` описывает
  C6+6/d512 и W6+6/d768/QK-LayerNorm; `decode.py` - KV, row compaction,
  async EOS, cross-KV sharing. `pipeline.py`, `execute.py`, `vote.py` -
  адаптивный ансамбль, ограниченное исполнение и независимая проверка winners.
  `runner.py` запускает свежий процесс. Политика/guards/SHA - в одном
  [config.json](code/hw2_codegen/config.json).
- [TRAINING.md](TRAINING.md) - как обучались три финальные модели: цепочки запусков, рецепт, подготовка данных.
- [tests/](tests/) - frozen v11 CPU parity, QK/cache, voting/execution,
  OOM/deadline-related regressions, обучение и notebook dry-run.
- [report/latex/report.pdf](report/latex/report.pdf) - отчёт; исходник
  [report.tex](report/latex/report.tex), графики - [make_figs.py](report/latex/make_figs.py).
- [../work/](../work/) - весь исторически использованный код обучения,
  экспериментов и анализов, заметки и метрики. Ignored кеши и full checkpoints
  в Git не входят.

## Run All на Kaggle

1. В Kaggle открыть **Add Input**, найти и добавить dataset
   [thisarivson/nlp-hw02-final-weights](https://www.kaggle.com/datasets/thisarivson/nlp-hw02-final-weights).
   Используйте **версию 4**. В нём три model-only FP16 веса, `tokenizer.json`, обычный `code/`,
   `package-manifest.json` и фиксированные входы test/tune. Датасет публичный.
2. Импортировать notebook, выбрать NVIDIA T4, Internet off. Если Kaggle выделил
   пару T4, используется только GPU0: `CUDA_VISIBLE_DEVICES=0` устанавливается
   до Torch import. Пакет проверяет, что видна одна T4.
3. Runtime финального GPU-кода: `torch==2.11.0+cu128`; нужны `tokenizers`,
   `sympy`, `numpy`. Установка библиотек/монтаж dataset - до таймера;
   imports, чтение/SHA весов и tokenizer - внутри полного timer.
4. Выполнить **Run All** в новом kernel. Пути находятся по manifest автоматически.
   Перед импортом проверяются SHA всех Python/JSON файлов пакета; worker проверяет
   SHA весов/tokenizer. Все три checkpoint strict-loaded с `weights_only=True`.
5. Скачать `/kaggle/working/final-package-v2/answers.jsonl`: 1919строк,
   каждая только `{"code": "..."}`, исходный порядок вопросов.
   Для аудита сохранить `summary.json`, `test-1/timing.json`, `test-1/summary.json`,
   tune answers/quality/journal и исполненный notebook.

По умолчанию Run All выполняет **ровно один cold test и один свежий tune**.
`RUN_TUNE=False` отключает только диагностический tune. Gold tune используется
после записи answers и не передаётся selector. Повторный запуск требует нового
`HW2_OUTPUT`: завершённые результаты не перезаписываются.

Timer начинается перед созданием свежего Python-процесса и заканчивается
после закрытия test answers: imports, SHA/read/load, warmup, GPU decode,
restore/CPU execution/vote, независимое переисполнение и write включены.
Guard160/166/170/172с кооперативный, поэтому сам по себе не доказывает180с.
Guard/partial/fallback, ошибка переисполнения или OOM отмечаются в проверках.
Память: точные Torch peak allocated/reserved и NVML device-wide used каждые20мс.

## Запуск в Colab

1. Открыть `inference.ipynb` в Colab и выбрать runtime с T4. Измеренный runtime —
   `torch==2.11.0+cu128`. Установка библиотек выполняется до cold timer.
2. Датасет публичный, токен Kaggle не нужен. Перед Run All выполнить отдельную
   setup-ячейку (скачивание ≈440 МБ не входит в лимит):

```python
!pip install -q kagglehub tokenizers sympy numpy
import os, kagglehub
root = kagglehub.dataset_download("thisarivson/nlp-hw02-final-weights/versions/4")
os.environ["HW2_WEIGHTS_ROOT"] = root
os.environ["HW2_CODE_ROOT"] = os.path.join(root, "code")
os.environ["HW2_OUTPUT"] = "/content/hw2-output"
```

3. Выполнить **Run All** финального notebook. Скачать
   `/content/hw2-output/answers.jsonl`. Для повторного прогона задать новый
   `HW2_OUTPUT`; готовые ответы не перезаписываются.

Локально используются те же переменные; `HW2_WEIGHTS_ROOT` указывает на каталог
manifest, `HW2_CODE_ROOT` — родитель hw2_codegen. `HW2_TEST_PATH` позволяет
задать JSON-список с `question`; строки не пересортировывать. SHA привязаны
к финальным весам; смена runtime/batching может изменить FP16 sampling.

## Веса и происхождение

| Модель (порядок ансамбля) | Файл | SHA256 |
|---|---|---|
| W B7 — greedy-якорь | w-b7-fp16.pt | `3378c368f601c7cd046d40de22cf5cc6f5081deff190819bb4f616f0eb2761ac` |
| C-long2x B7 | c-long2x-b7-fp16.pt | `bcb41c54d031b70523d357dd7d14dc24ec7d1393a6cb8714d8eacfe761cca32b` |
| P1-B | p1-b-fp16.pt | `8439037720dda16dda6b73fe676860d5498bbcba4cacebc0d709bbb22db464a5` |

Tokenizer SHA `261f25b5d4be006a6a00e41e0eafbe522580741da627ba670f6b9805a2659b7d`.
B7: output Kaggle-запуска `nlp-hw02-b7` v1 (аккаунт друга), W соответствует
`model-GPU0/points/b6-final`, C — `model-GPU1/points/b6-final`.
Веса экспортированы в `model-GPU{0,1}/weights/b6-final-fp16.pt`;
SHA/bytes сверены с `checkpoint.json` этих points. P1-B — прежний
запуск `nlp-hw02-kaggle-p1` v3 (аккаунт друга), `model-pilot-b/weights/long-truncation-fp16.pt`.
Файлы переименованы без изменения байтов. Все три — model-only FP16,
без Adam и обучающих данных. В новой версии dataset оставлены только эти
три веса: B6 не нужны финалу; прежние версии dataset сохранены.

## Проверки

Из submission, Python с зависимостями [requirements.txt](requirements.txt):

```sh
PYTHONPATH=code python -m unittest discover -s tests -v
python tests/check_notebook.py
```

21 CPU tests: tokens/logprobs/restores/winners frozen v11 exact на8строках,
C/P1-B/W-QK и одинаковые budget14/21; 49152restores/2048L3 winners exact;
cache/full-prefix, QK, исполнение/нетранзитивное vote, сохранение кандидатов
при guard, AdamW/truncation/resume. Полный CPU notebook dry-run8 прошёл.
Исторические parity tests требуют `../work/kaggle` и ignored журналы;
`HW2_WORK` меняет их путь. Сам inference-пакет от work не зависит.

`HW2_DRY_RUN=1` запускает tiny random модели на8synthetic вопросах, без GPU
и без оценки качества; при локальном запуске `HW2_CODE_ROOT=code`.
Квитанция: [final-package-v2-cpu.json](../work/checks/final-package-v2-cpu.json).
CPU parity не гарантирует FP16 тождественность разных форм батча.
Финальные W B7/C B7/P1-B проверены отдельно: SHA, FP16 finite, архитектура,
strict-load. Неизменность 16 исполняемых модулей относительно прогона пяти
вариантов подтверждена SHA, config семантически равен сохранённой v2.
Optional real-weight parity с frozen v11 decoder: `python tests/v11_real_weights_check.py`
(три вопроса, FP32 CPU, cap64/budget14; не оценка качества).

GPU-проверка финального v2: **один private push**, [thisarivson v1](https://www.kaggle.com/code/thisarivson/nlp-hw02-final-package-v2-budget640/versions?versionNumber=1),
21.10.2026, dataset final-weights **v4** (сам Kaggle-ноутбук приватный, его outputs сохранены в inference.ipynb). Серверные code cells exact,
версия подтверждена до и после скачивания. Notebook содержит настоящие
outputs этого прогона; повторов/отмен/смены видимости нет.

| Split | Полный cold, с | NVML peak used, GB | Torch allocated / reserved, GB |
|---|---:|---:|---:|
| Test1919 | **122,66** | **11,76** | 11,01 / **11,12** |
| Tune2048 | **107,99** | **9,12** | 8,41 / **8,49** |

GB=10⁹bytes. Tune **1560/2048 (76,172%)**, +3 к прежней v2.
Все gates passed: test≤150с (официальный лимит180с), reserved≤14GB,
OOM/guards/partial/fallback/non-numeric/reexecution mismatch —0.
1919 test и2048 tune ответов независимо переисполнены; replay tune2048 exact.
Runtime `torch==2.11.0+cu128`, CUDA12.8, TeslaT4; используется GPU0.
[Протокол](../work/notes/final-package-v2.md),
[GPU-аудит](../work/results/final-package-v2.json).

Готовый [final-package-v2.jsonl](../work/submissions/final-package-v2.jsonl):
1919 code-only строк, SHA `f9c772e41fae010c2fe9e50723cfda524dc4346c653f0319a86561ca0e7329ed`.
С `variants-2026-10-20/v2.jsonl` **нет побайтного совпадения**:
**279 code /159 numeric** различий при независимом исполнении обоих файлов.
Public 80,10% относится к прежнему v2.jsonl; JSONL этой финальной упаковки дал 79,37% — это разброс между прогонами, см. выше.
[Сравнение](../work/checks/final-package-v2-comparison.json).

Config, SHA весов, tokenizer/входы, seed, budget и16 исполняемых модулей exact
относительно прежней v2. Различаются уже stage1 samples C/P1-B до selector;
на tune все2048 первых W samples и greedy exact. Вероятный механизм —
async EOS: `event.query()` определяет момент compaction, что меняет форму
Exp(1) noise и расход RNG. Поэтапные CUDA-трассы не записывались; точный
момент расхождения и возможный вклад FP16 kernels не изолированы.
[Диагностика](../work/checks/final-package-v2-reproducibility.json).
Политика по результату не менялась; нового GPU-прогона нет.

## Воспроизведение обучения

Полная цепочка обучения трёх финальных моделей по Kaggle-запускам, с ноутбуками и модулями каждого шага, — в [TRAINING.md](TRAINING.md).

[train.py](code/hw2_codegen/train.py) реализует базовую математику C/P1:
normalized token init, Pre-LN/SwiGLU, batch128, AdamW wd=.01, clip1,
AMP FP16 + GradScaler1024, warmup → constant → cooldown. Основной trajectory
использует q=0. Для pilot-B cooldown используется `1-sqrt` и length-stratified
loss truncation q=.1: global mask на effective batch, квоты по compact длинам
53/79/105, средняя CE для отбора, token-weighted CE для backward.
L3 C-main/C-long2x использовали линейный cooldown. Финальные B7 C/W: q0, фиксированные 65800/33000 attempted updates по `1-sqrt`, из тех же uncooled B5, что B6. Для W обучение включает QK-norm и z-loss: исходные `b2_optim.py`, `b4_runtime.py`, `b6_runtime.py` в `work/kaggle/src/`; чистый train.py не подменяет их исторический recovery/resume протокол.

1. Подготовить исходный sealed compact-кеш (9 725 737 пар) по коду
   [26a/26c](../work/notes/wsd-data-scaling-and-sharded-cache.md) и
   [L1 cache](../work/kaggle/src/l1_cache.py). Это исходные пары
   `SlavaYus/tinysms_qa_12m`: ограничения UTF-8 question≤512/code<512 байт,
   исключение protected вопросов и hash-heldout seed42, finite numeric references.
   Текстовые дубли остаются. Compact меняет представление исходного кода,
   новых семантических targets нет.
2. `data.CompactCache` читает `compact-manifest.json` и shard-массивы
   `source/target.npy`, `source/target-offsets.npy` через mmap. Подготовку корпуса
   и исторический отбор не упрощали повторной фильтрацией: используйте тот же
   sealed кеш для сохранения состава данных.
3. C-main: `data.make_order` — seeded permutation с локальной сортировкой в 4096
   и перемешиванием batches128. C-long2x: сохранённый L2 `order.npy`, где длинные
   строки встречаются дважды; функция `data.long2x_order` воспроизводит этот выбор
   при тех же исходных длинах. Для точного продолжения нужны тот же order и cursor.
4. Скопировать `training-example.json`, задать кеш, order, output, model_spec и
   фактические main/cooldown updates выбранной траектории. Выполнить:

```bash
PYTHONPATH=code python -m hw2_codegen.train training-example.json
```

`stable.pt` сохраняется **перед cooldown**; `latest.pt` содержит model, AdamW,
scaler, RNG, cursor; `weights-fp16.pt` совместим с inference. `resume` принимает
full checkpoint этого чистого скрипта; model-only файл для resume не подходит.
Боевые full L2/L3 имеют другой служебный формат, их восстановление — в `work/`.

В боевых сессиях main/cooldown имели wall-time budgets, аварийные откаты и
продолжались между сессиями. Здесь длительность явно задана updates, а при OOM
или nonfinite update выполнение останавливается для восстановления checkpoint.
Для полного исторического replay с timed cooldown и recovery используйте исходные
[l1_runtime.py](../work/kaggle/src/l1_runtime.py),
[l2_runtime.py](../work/kaggle/src/l2_runtime.py),
[p1_math.py](../work/kaggle/src/p1_math.py) и сохранённые `training.json`.
CPU-тесты подтверждают математику шагов, не воспроизведение многочасового обучения.

## PDF и история экспериментов

```sh
cd report/latex
python make_figs.py          # графики: matplotlib
latexmk -xelatex report.tex  # нужен TeX Live/MacTeX со шрифтами CMU (cm-unicode)
```

[Архитектуры](../work/notes/architecture-decision.md),
[B7](../work/notes/b7-results.md), [варианты](../work/notes/final-variants.md), [E1](../work/notes/e1-ensembles.md),
[история v7–v11](../work/notes/final-inference.md),
[скорость decoder](../work/notes/decode-speed.md).
Историческая Kaggle-оркестрация находится в work/kaggle; credentials для
инференса не нужны. Full caches/checkpoints/journals не дублируются в Git.
