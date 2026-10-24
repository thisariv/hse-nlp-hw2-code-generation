---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-13
updated: 2026-10-13
sources:
  - nlp-hw02-final-inference-v3
  - nlp-hw02-final-inference-review
  - nlp-hw02-final-inference-preparation
  - kaggle-l2-results
  - sympy-string-parsing-security
  - pytorch-reproducibility
aliases: []
---

# Независимое ревью финального инференса

**Дополнение v3:** GPU v2 дал полный greedy из-за двух deadline-дефектов. Исправления, новый executor benchmark и CPU receipts — в [разделе v3](final-inference.md#v3-причины-и-исправления-после-gpu-v2). Текущий notebook: 2cold test+tune, ≤15мин, цель≤120с; старые требования3test/150с/30мин ниже относятся к ревью v1–v2. Защиты этого ревью повторно проверены; Linux/CUDA и полный T4 timer v3 ещё не измерены, push здесь0.

**Вердикт: CPU-ревью прошло после исправлений; GPU gate заблокирован отсутствием разрешённой env-авторизации Kaggle. Для бота mixed-файл ещё не получен.** Ревью исходного коммита `0fd23b9` в ветке `codex/nlp-hw2-code-generation`. Сначала проверены [условие](../../task/hw2-code-generation.md), [final-inference](final-inference.md) и [l2-results](l2-results.md). Правки прежних экспериментов и существующие greedy submissions сохранены.

## Критичные риски и исправления

| Риск исходного кода | Место после исправления | Исправление и регрессия |
|---|---|---|
| **SymPy обходил запрет строкового eval**: `safe_sympy()` возвращает настоящий Symbol; `(x+1).subs(x, строка)` вызывает sympify/parse_expr за пределами wrapper. Возможны произвольный импорт, I/O и процессы | `kaggle/src/final_numeric.py:62–73`; прежний путь `exp26_exec.py:58–94`, `final_execute.py:27–29` в `0fd23b9` | Финальная отдельная numeric policy отключает parse_expr во всём worker. Регрессия пытается создать только временный marker через строку `.subs`; marker отсутствует, результат execution_error. Обычная Symbol/Eq/solve задача остаётся numeric |
| **Общее изменяемое состояние между программами**: старый `_MODULES` переживает задания; `math.pi = 42` загрязняет следующие результаты и дедуп-кеш | `final_numeric.py:40–42, 94–99, 113–114` | Запрет Attribute Store/Del, новые namespaces на программу, очистка SymPy cache. Один worker исполняет mutation → независимую math-программу: mutation blocked, pi правильное |
| **Worker death мог обрушить сервис**: между is_alive и pipe.send дочерний процесс может исчезнуть, BrokenPipe не перехватывался | `final_execute.py:73–110`, `final_executor.py:50–62` | Send/recv pipe errors заменяют только один slot; непринятое задание возвращается в очередь ровно один раз. Регрессия инъецирует смерть после is_alive и получает единственный правильный результат17. Drain теперь сообщает код аварийного выхода |
| **CPU2 был округлённым cumulative soft RLIMIT**, фактическая отсечка могла быть почти CPU3; SIGALRM можно поймать | `final_numeric.py:117–145`, `final_execute.py:67–74` | Добавлен ITIMER_PROF2; RLIMIT_CPU остаётся вторым ограничителем, внешний wall+2=5 с убивает и заменяет worker. Регрессия ловит Exception в бесконечном цикле; исполнение ограничено, следующий ответ 23 получен |
| **Проверка выбранного кода отсутствовала в новом GPU-пайплайне**; старый CPU replay проверял только исторические samples | `final_vote.py:54–73`, `final_inference.py:219–228` | Числовые winners повторно исполняются в свежем сервисе до закрытия answers. Сравнение именно значения выбранного кода с его голосовавшим execution, isclose1e-6. Расхождение/незавершённая проверка даёт greedy fallback и явный event. Регрессия подменяет кеш 7→8 и обнаруживает строку 0 |
| **Seed не фиксировал весь runtime**, не было проверки совпадения трёх test-файлов | `final_runner.py:29–31`, `final_inference.py:93–97`, `final_numeric.py:25, 71–73, 114`, `final_session.py:208–217` | deterministic algorithms, CUBLAS workspace, фиксированный Python hash seed; внутренние Python/SymPy RNG сбрасываются, Symbol.random запрещён. Suite сравнивает SHA всех трёх answers; регрессии отклоняют разные файлы и random method. Tiny sampler K1/K4/K8 сохраняет tokens/LP исходной реализации |
| **Fallback-аналитика неоднородна**: список IDs либо строка all; счётчика нет | `final_runner.py:68–70`, `final_inference.py:235–238` | `fallback_count` всегда число, включая hard stop. Регрессии: sampling deadline → 3/3 и точный greedy; supervisor hard kill → 2/2; до готовности greedy → явный failed без выдуманного кода |

Закрыт также обход фильтра атрибутов через поля `str.format`: строки формата умеют обращаться к скрытым атрибутам без соответствующего Attribute в AST. `format` запрещён в финальной policy; регрессия использует только безвредный `math.pi`, без чтения окружения. В14799 исторических unique нет этого метода.

Дополнительно worker перенаправляет файловые дескрипторы stdout/stderr в devnull (`final_execute.py:22–26`): миллион символов print не ломает JSON-протокол. Fork/import os блокируется до исполнения. Новая финальная numeric policy встроена в ноутбук; исторический `exp26_exec.py` не изменён.

## Проверка требований

- **Честный таймер:** `final_session.py:167–171` ставит monotonic timestamp до Popen supervisor; `final_runner.py:4,29–34` передаёт его worker до Torch import. `final_inference.py:74–120,169–253` включает импорты, input read, SHA/обе модели/tokenizer/encode, warmup, greedy, обе K4, restore/submit, execution tail, shutdown, vote, независимое переисполнение и answers close. Restore уже внутри generation phases; service wall перекрывается с GPU и отдельно не суммируется. Сумма неперекрывающихся phases проверена. Reference scoring, candidate journal и диагностические SHA после answers close не влияют на выбор.
- **Ровно один используемый GPU:** supervisor устанавливает CUDA_VISIBLE_DEVICES=0; worker требует count1, cuda:0 и T4 (`final_inference.py:98–101`). Веса обеих C одновременно на GPU0, генерация последовательная. Kaggle может выделить физическую T4×2, второй GPU процессам не виден; фактическую стоимость квоты без запуска не измеряли.
- **Deadline:** work/reexecution deadline160 с, greedy deadline150 с (`final_inference.py:79–80`); независимый supervisor прерывает worker172 с и закрывает greedy answers в оставшемся резерве (`final_runner.py:39–74`). Все записи guard/fallback проверяются. Запуск с full timer≥180 отклоняется. До готовности полного greedy гарантировать настоящий fallback невозможно; такой запуск failed. Абсолютной гарантии при зависшем OS/драйвере/диске нет, CPU mock её не доказывает.
- **Выбор кода:** связные компоненты isclose, count→greedy→mean LP→order, повторные samples сохраняют голоса. Greedy выбирается только в победившей компоненте; графовая близость нетранзитивна, поэтому у компоненты нет единственного канонического числа. Выбранный код сохраняет собственное числовое значение, повторное исполнение проверяет его непосредственно. Исторические code/execution winners1492/2048 exact (`final_vote.py:10–35`, `final_choose.py:4–29`).
- **Изоляция:** spawn без Torch; wall3/CPU2/RLIMIT_AS768MiB на Linux; дополнительный внешний wall5, worker kill/restart; закрытие executor killpg + bounded waits (`final_execute.py:17–29,48–115`, `final_executor.py:68–84`, `final_numeric.py:101–145`). Linux parent-death guards каскадно убивают отдельные process groups. Запрет imports/fork, string parser и мутаций проверен. Это ограниченный numeric executor, не OS-контейнер для произвольного недоверенного Python. Linux RLIMIT_AS и реальный parent-death cascade на текущем macOS-хосте **не проверены**.
- **Gold:** test-конфиг не содержит reference; worker читает только question, модели и tokenizer. Вызов select получает greedy/sample code, execution и LP, без correctness/reference. Tune reference читается только после answers close (`final_inference.py:254–266`). Исторический baseline сверяется отдельно и не выбирает test-файл. Никакого нового обучения, eval2048 или fresh1024.
- **Порядок/формат:** записи сохраняются по исходному индексу selected, итог идёт по records, включая сортированный greedy (`final_inference.py:133–148,169–178,209–231`). Валидатор требует ровно1919 строк с единственным ключом code:str. Добавлены input SHA, question-order SHA и answers SHA. [Проверка output](../kaggle/final_output_check.py) сверяет pinned test.json, фазовые суммы, один GPU, fallback0, reexecution, три SHA, отсутствие test reference; копирует только test-1. Fixtures отклоняют неверный порядок, время>150, fallback, несколько GPU и разные ответы.
- **RNG:** seed270042 отдельно на модель; warmup greedy не потребляет sampler RNG. K4/b64 отличается от первых4 K8/b32 по lane consumption, это ожидается. CUDA OOM retry восстанавливает CPU/CUDA RNG, но меньший batch меняет последовательность lanes; детерминизм подтверждается только тремя реальными одинаковыми файлами. CUDA-проверки ещё отсутствуют.
- **Память:** обе FP16 модели ≈204,6MiB параметров; self-KV при256 lanes/cap512/6layers/d512 ≈1,5GiB, cross-KV дополнительно зависит от source padding и повторяется для K4. Это расчёт, не peak T4. OOM уменьшают batch и логируются; записываются allocated/reserved peaks. Фактический peak и отсутствие OOM pending.
- **Без интернета:** notebook и push metadata internet=false; только два mounted private datasets, selective ZIP4JSON, локальный tokenizer/weights, никаких pip/download/API/secrets внутри GPU notebook. Требуется Torch2.11.0+cu128/CUDA12.8 как в L2; неподходящий runtime завершится ошибкой до greedy.

## CPU-проверки и некритичное

[Receipt](../checks/final-inference-cpu.json), [основные проверки](../kaggle/final_inference_checks.py), [регрессии ревью](../kaggle/final_review_checks.py).

- Frozen replay: **1492/2048 =72,8515625%**, совпали код и execution всех2048 winners; восстановлены32768 samples.
- Повторно исполнены **14799 unique**, numeric mismatches0, duplicate conflicts0. 8 нечисловых `timeout_or_worker_exit` стали `timeout`; результат голосования не меняется. CPU8workers8,237с — не тайминг Kaggle.
- Tiny cold CPU: **2,388с**, фазовая сумма и новый reexecution этап проверены; принудительный guard1,970с с fallback3. GPU не использован.
- Notebook/source AST/nbformat/private/offline и mocked3test→tune прошли. Реальные Linux memory/kill-tree и CUDA ещё не подтверждены.

Один дополнительный CPU-аудит вернул неполный набор результатов; подробного stderr тогда не было, причина именно того случая не установлена. Диагностический повтор завершил14799/14799. Отдельно выявлена и исправлена воспроизводимая гонка is_alive→send; после её исправления полный набор regression/sampler/execution/cold-process/static/suite проверок прошёл. Ошибка исходного дополнительного прохода не скрыта успешными повторами.

Некритичное: string-based SymPy выражения теперь намеренно отклоняются; у исторических numeric programs изменений нет. Строгий runtime pin затрудняет перенос на новое окружение. Файлы test одинакового seed проверяются по байтам, но воспроизводимость между другими GPU, версиями Torch или OOM batching не обещается. Возможны естественные нечисловые модельные ответы — они отдельно перечисляются и не маскируются счётчиком deadline fallback.

Допуск качества приведён к запросу пользователя: **±1п.п.**, то есть1472…1512 matches из2048. Убран автоматический дополнительный K8 replay: ровно3cold test +1tune, без обучения; сессия timeout1800/watchdog1740/1770. Для корректных числовых samples sampler/vote/restore не изменены. Независимая reexecution добавляет CPU время, поэтому прежние≈130с остаются непроверенной гипотезой.

## Kaggle: остановка и точные команды

Попытка разрешённого API probe вернула exit3: `environment_authorization_unavailable`. Не было обращений к credential-файлам, HTTP-запросов, push, GPU-сессии или download. Наличие авторизации в запрещённых файлах не исследовалось. Изменять/обходить авторизацию и переходить к UI не стали.

В shell с уже доступной env-авторизацией (`KAGGLE_API_TOKEN` либо `KAGGLE_USERNAME`+`KAGGLE_KEY`), из корня этого worktree:

```bash
cd work/kaggle
KAGGLE_PYTHON="$(sed -n '1s/^#!//p' "$(command -v kaggle)")"
python3 build_final_inference.py
"$KAGGLE_PYTHON" final_review_api.py probe
"$KAGGLE_PYTHON" final_review_api.py push --iteration 1
```

API helper использует официальный установленный SDK без автозагрузки credential-файлов. Перед push проверяет private входы, private/offline/T4 metadata и1800с timeout; регистрирует попытку до отправки, максимум2. Номер версии взять **из ответа push**, затем:

```bash
VERSION=1  # заменить номером из ответа push
"$KAGGLE_PYTHON" final_review_api.py status --version "$VERSION"
# Повторять status с интервалом30–60с до COMPLETE/ERROR; не делать повторный push.
"$KAGGLE_PYTHON" final_review_api.py output --version "$VERSION"
python3 final_output_check.py "../results/kaggle-runs/nlp-hw02-final-inference/$VERSION/final-inference"
```

Output — JSON/JSONL/logs/source и gzip tune journal, без весов/checkpoints; download pinned vN, без удаления на Kaggle. При доступной авторизации и провале первого тайминга сначала исправить выявленное узкое место, пересобрать и сделать только `push --iteration 2`; третьего запуска нет. Не отправлять второго запуска при отказе авторизации/сети или неоднозначном первом API ответе.

После успешной проверки результат копируется в `work/submissions/final-mixed-k4k4-l2-28800.jsonl` из **test-1**, существующий иной файл не перезаписывается. Сейчас такого файла нет; прежние greedy submissions не заменены.

| Метрика | Результат этой сессии |
|---|---|
| Три полных test1919 / фазы | Не запускались |
| ≤150 и строго<180 каждый | Не проверено |
| Новые K4 tune2048 | Не измерено;72,85% только старые журналы K8 |
| Fallback / детерминизм GPU | Не измерены |
| API push / GPU iteration | 0 / 0 |
| Файл для бота mixed | Отсутствует |

Коммит заблокирован на `git add`: exit128, shared Git `index.lock`, `Operation not permitted`. Индекс не изменён, изменения ревью сохранены; параллельные L3 файлы не включались. Обход ограничений не выполнялся.

## Источники

Содержание исходников, сохранённых L2 журналов и CPU receipts — наблюдения. Расчёт KV, оценка рисков и вердикт — выводы ревью; тайминги GPU не подменены ими.

- **nlp-hw02-final-inference-review**: пользовательский протокол13.10.2026, review `0fd23b9`, текущие patches/regressions; запись в bibliography. [Реестр](../kaggle/runs.json).
- [final-inference](final-inference.md), [l2-results](l2-results.md), [условие](../../task/hw2-code-generation.md).
- **sympy-string-parsing-security**: [SymPy Basic Operations](https://docs.sympy.org/latest/tutorial/basic_operations.html), [Parsing](https://docs.sympy.org/latest/modules/parsing.html): string sympify/parse_expr используют eval и не предназначены для непроверенных строк.
- **pytorch-reproducibility**: [официальные PyTorch notes](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/randomness.md), [deterministic algorithms](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html): один seed не гарантирует воспроизводимость между runtime/платформами.
