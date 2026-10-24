---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-13
updated: 2026-10-19
sources:
  - nlp-hw02-e1-ensembles
  - nlp-hw02-ensemble-inference-v10
  - nlp-hw02-ensemble-inference-v9
  - kaggle-b5-results
  - nlp-hw02-ensemble-inference-v8
  - kaggle-cli-t4-pool
  - kaggle-b4-results
  - nlp-hw02-ensemble-inference-v7
  - nlp-hw02-final-inference-l3-v6
  - nlp-hw02-final-inference-l3-v5
  - nlp-hw02-final-inference-l3-v4
  - nlp-hw02-final-inference-v3
  - nlp-hw02-final-inference-preparation
  - kaggle-l2-results
aliases: []
---

# Финальный быстрый mixed-инференс

**Актуально: [v8](#v8-c-long2x--p1-b--w-plain-b4) прошла cold single-T4 gate: test1919 **91,924с**, tune **1532/2048=74,8047%**, guard/fallback/partial0. Offline replay1541, winner parity2048/2048. Ответ — `work/submissions/final-ensemble-v8.jsonl`,1919code-only строк;1216отличий от v7. Ниже сохранена история v3–v7.**

**[V9](#v9-c-long2x-b5--p1-b--w-b5) прошла интерактивный T4 gate: cold test1919 **93,897с**, tune **1532/2048=74,8047%**, guard/fallback/partial0. Ответ — `work/submissions/final-ensemble-v9.jsonl`,1919code-only строк; отv8 отличаются1166code и231исполненный ответ.**

Отдельная подготовка14.10.2026: [Профиль decoder на T4](decode-speed.md) — cap512/неактивные lanes/KV-трафик, opt-in compaction/EOS/CPU/sampler. CPU-ready private thisarivson profile bundle, full1919 baseline/all +192 абляции,≤12мин; GPU/push0. Рабочий v5 и P1 не изменены.

**V3 готов к следующему GPU-прогону; CPU-проверки пройдены.** V2 на T4 завершился guard и полным greedy, поэтому mixed gate не прошёл. Текущая цель ≤120 с при потолке <180 с. [Ноутбук](../kaggle/notebooks/nlp-hw02-final-inference.ipynb) делает ровно **2 cold test1919 +1 tune2048**, сессия ≤15 мин. На Kaggle в этой сессии ничего не отправлено; запуск выполняет пользователь.

Первоначальная попытка коммита v1 была заблокирована на `git add`: shared Git `index.lock`, `Operation not permitted` в песочнице.

Повторная попытка коммита v3 также остановилась на `git add` явного списка только своих файлов: `.git/worktrees/study1/index.lock`, `Operation not permitted`. Индекс остался пустым; правки сохранены в worktree, чужие L3/README дописки не staging.

Пользователь сообщил результат бота: greedy C-long2x l2-28800 **681/960 = 70,94% public pass@1**; порядок `test.json` правильный, ответ — JSONL с единственным ключом `code`. Это внешний подтверждённый пользователем score, не результат текущей CPU-сессии.

## V3: причины и исправления после GPU v2

### Что показал v2

[Компактная диагностика](../checks/final-inference-v2-diagnostic.json) сохранена из предоставленных пользователем timing/summary/quality. Test-1: greedy29,010 с, main K4 78,214 с, обрезанный long33,824 с; полный timer160,304 с. Test SHA `a59feda8…` соответствует greedy C-main по подтверждению пользователя; tune1425/2048=69,5801% также greedy. Это **не измерение качества mixed**.

Два дефекта guard: для голосования требовались все4+4, а reexecution использовал уже истёкший160-секундный deadline. В test-1 повторная проверка не завершилась ни для одного из1918 numeric winners: все оказались в mismatch/fallback, включая строки с полным набором кандидатов.

**Вывод о четырёхкратном замедлении executor из service wall некорректен.** Это время жизни сервиса, включающее ожидание очередного GPU-батча. Все10578/10578 уникальных программ test-1 были исполнены, execution tail7 мкс; повторный сервис работал только0,107 с до закрытия из-за истёкшего срока. Spawn уже был долгоживущим, импорт не повторялся в отдельном процессе на каждую программу. В K8-корпусе25402 unique лишь одна программа содержит SymPy: очистка его кеша не объясняет основной GPU-прогон. Теперь отдельно измеряются сумма worker task wall/CPU и интервал первого submit → последнего result.

GPU обрабатывал только256 lanes; peak около3GiB оставлял большой запас. Большинство стоимости декодирования — много шагов и мелких GPU-операций. Увеличение batch уменьшает число таких шагов для всего набора; точный выигрыш на T4 пока не измерен.

### Изменения

- **Guard:** голосуют greedy и все уже исполненные samples каждой строки. Неисполненные кандидаты исключаются; дубликаты сохраняют число голосов. Без исполненных samples остаётся greedy. В timing записываются `partial_vote_rows/count`, `no_executed_sample_count`, `executed_sample_counts`, fallback и reexecution failures. Generation140 с → execution tail152 с → независимая проверка170 с; внешний hard stop172 с. Для малых CPU-тестовых бюджетов резервы уменьшаются пропорционально. Аварийный supervisor при зависании процесса сохраняет прежний greedy checkpoint.
- **GPU:** greedy b1024; sampling b256 вопросов×K4=1024 lanes. Обе фазы сортируются по длине вопроса, результаты записываются по исходным IDs. OOM уменьшает batch вдвое с восстановлением CPU/CUDA RNG. Математика Context/Decoder осталась AST-identical; отдельный `final_decode.py` сохраняет EOS/LP/RNG sampling, использует предвыделенные KV/output/sum buffers и contiguous PAD/BOS mask. В greedy убраны ненужные log-softmax и FP32-копия logits: greedy LP всё равно равна0 в vote. Все EOS проверяются раз в16 шагов. CUDA graphs не добавлены.
- **Executor:** Torch-free spawned workers живут весь сервис. Обычная арифметика исполняется с новым env/builtins/math namespace и сброшенными RNG/timers; SymPy-код — в новом fork-процессе от worker с неизменным прогретым SymPy. Его состояние и кеши не возвращаются родителю. AST проверяется один раз, `multiprocessing.connection.wait` заменяет обязательный sleep1мс на программу. Сохранены wall3/CPU2, Linux AS768MiB, CPU rlimit, внешние wall limits, kill/reap группы программы, parent-death guards, recovery/race protections, SymPy string-parser deny и format-обходы. Winners повторно исполняются в свежем сервисе внутри timer.
- **Квота:** ровно2 cold test +tune, одинаковый seed; сравнение двух answer SHA, качество±1п.п. от1492, цель обоих test≤120 с. Session budget900 с, watchdog840/870 с; `find_input` и вложенные Kaggle mounts сохранены. Автоматического K8 replay нет. L3-код и runs.json не изменены.

### Измерения и проверки

[Микробенчмарк](../checks/final-executor-v3-benchmark.json), [скрипт](../kaggle/final_executor_benchmark.py): одни и те же **2000 unique** из обеих K8 tune journals, shuffle seed270042, редкий SymPy включён;4 workers, три холодных сервиса каждой версии с чередованием порядка. Baseline берётся из commit2cd5208. Медиана v2 **1,281 с /1561 программ/с**, v3 **0,505 с /3963 программ/с**, ускорение **2,54×**, **0 расхождений побайтно сериализованных execution results**. Это macOS CPU-хост;≥150 программ/с на Kaggle4vCPU одновременно с GPU ещё требует замера.

[Прежние регрессии](../checks/final-inference-cpu.json): SymPy escape, namespace mutation, format traversal, bounded caught timeout/recovery, pipe-death race, выбранный код, порядок/формат/SHA и supervisor. Offline mixed **1492/2048=72,8516%**, code/execution mismatches0; restore32768 exact; полный повтор14799 unique с numeric mismatches0. Восемь нечисловых timeout/error labels зависят от host scheduling, как в предыдущем ревью; строгий2000-program benchmark имеет0 status/value mismatches. Sampler K1/K4/K8 tokens и LP побайтно совпадают с исходным при одинаковом batch; LP-free greedy также сохраняет tokens.

[V3 fixtures](../checks/final-inference-v3.json), [скрипт](../kaggle/final_v3_checks.py): budget8 с, main-батч выполнен, long отсутствует → **2 строки выбрали sample,1 осталась greedy**, независимая reexecution завершилась; полного отката нет. OOM после потребления RNG → retry b2→b1 с тем же первым random draw;1 sample winner сохранён,2 greedy-only. Генерация в fixtures подменена, executor/vote/files/timer настоящие. Static notebook/embedded sources/nbformat/private/offline/900-секундный bundle и2test+tune suite прошли.

CPU FP32 tiny decoder, width32/2 decoder layers/vocab8195,1024 total lanes/cap48: greedy **2,999→1,761 с (1,70×)**; sampling **12,512→12,949 с (0,97×)**. MPS недоступен. Это относительный CPU-тест; он не доказывает GPU-выигрыш sampling. Batching/bucketing меняют RNG lanes относительно v2 и первых4 L2 K8; новые K4 обязательно проходят tune gate. Linux RLIMIT_AS/prctl и CUDA OOM/RNG проверяются на следующем T4, локально хост macOS.

**Гипотеза T4, не измерение:** imports/CUDA setup/load/warmup15–20 с; greedy8–12 с; main K4≈30–40 с; long K4≈30–40 с; execution tail +reexecution≈4–7 с; vote/write≈1–2 с. Номинально около100–110 с, верх диапазона≈120 с. CPU service wall перекрывает GPU и к этим фазам не прибавляется. Только следующий2test+tune подтверждает полный timer, peak memory, deterministic SHA и качество; при guard статус остаётся review_required.

> [!info] История v1–v2
> Следующие разделы сохраняют первоначальную схему, проверки и прогнозы до первого T4. Актуальные batches, deadlines, executor и протокол запуска описаны в разделе v3 выше.

## Схема и неизменные правила

Один свежий supervisor-процесс → GPU-процесс с `CUDA_VISIBLE_DEVICES=0` → чтение test → SHA/загрузка **обеих** FP16 l2-28800 и compact-tokenizer → небольшой greedy warmup обеих → greedy C-main → C-main K4 и C-long2x K4/T0,7 → restore полного кода → bounded execution → mixed vote → закрытие answers. Все этапы, включая Python/Torch imports, учитываются: notebook ставит monotonic timestamp **до создания процесса**, supervisor передаёт его GPU-процессу. Итоговый timer заканчивается после закрытия answers; hashes, tune correctness и диагностика пишутся после него.

Сэмплер — неизменные AST-функции `Context`, `Decoder`, `decode` из `exp27b_fast.py`: FP16, физический `repeat_interleave` cross-KV, SDPA4D, cap512, PAD/BOS запрещены, EOS учитывается в **untempered mean logprob**, проверка завершения раз в16 шагов. Seed270042 сбрасывается отдельно перед каждой моделью. Greedy: сортировка по source length, b256. Sampling: естественный порядок test / возрастающие fixed-split IDs tune, b64 вопросов ×K4 =256 lanes. При OOM batch уменьшается с восстановлением CPU/CUDA RNG; попытка записывается, очистка allocator только после OOM.

Голосуют 4+4 samples и ровно один greedy C-main с LP=0 как в L2. `exp27_decode.choose` скопирована генератором **без изменения AST**. Numeric-кластеры — связные компоненты попарного isclose rel/abs1e-6. Выбор: count → наличие greedy в компоненте → средняя LP → исходный order. Код — greedy, если он в победившей компоненте; иначе максимальная mean LP с устойчивым order. Для журналов L2 эта явная политика кода полностью совпадает с прежним `choose` (greedy LP0 выше sample LP).

Reference и correctness не передаются в выбор. Новых таргетов, обучения и обращения к train нет; heldout используется только для проверки зафиксированного декодера. Eval2048/fresh1024 не оцениваются.

## Ускорения и эквивалентность

| Изменение | Проверка |
|---|---|
| Постоянный Torch-free исполнитель вместо процесса/пула на каждом батче | Реальные процессы; весь mixed tune переисполнен, 0 расхождений статуса/значения |
| `os.cpu_count()` workers; spawn, memory768MiB, прежние wall3/CPU2 и AST/import/numeric policy | 14 799 unique на8 CPU за8,498 с; цикл ограничен и worker восстановлен. Это CPU-хост, не замер Kaggle4 CPU |
| Общая дедупликация полного кода, включая greedy и обе модели | 18 432 кандидата →14 799 программ (−19,71% вызовов); голоса и order сохранены |
| CPU исполняет предыдущие батчи одновременно со следующей GPU-генерацией | Incremental submit/attach, результаты сопоставляются по коду; CPU повтор всех программ exact |
| Нет обычного gc/empty_cache и candidate JSON I/O после каждого батча | Математика decoder извлечена без изменений; исходный и новый CPU-сэмплер дают bit-identical tokens/LP для K1/K4/K8 |
| K4/b64 вместо сохранённых K8/b32 | **Не token-equivalent**: меняются lanes, RNG consumption и batching. Отдельный GPU quality gate обязателен |

Точная offline-проверка **сохранённых первых4 K8 каждой модели**: **1492/2048 =72,8515625%**, 0/2048 расхождений выбранного execution и кода с `analyze-l2-results.py`. Execution digest `62104d458161d42ce5bd64cc4ee58f1a8a9a2e883b28decfdee0489eba5833dc`; code digest `8a572da313e85b4de96af01aad03efe800534ea67d59e9e3e7a3159d960dc12e`. Restore повторён для **32 768** сохранённых samples; 0 расхождений. Все14 799 unique mixed программ исполнены заново; numeric/status mismatches0, противоречий duplicate execution0.

Unit-проверки: count/greedy/LP/order, нет numeric, non-transitive graph, выбор greedy-кода, duplicate votes, три compact-mode и invalid restore, точный формат `{"code":...}`/UTF8/порядок. Tiny CPU cold-process smoke:27 кандидатов, полный таймер1,734 с; это тест склейки, не прогноз C/T4. Принудительный sampling deadline дал точный greedy JSONL, все3 строки fallback; повтор со старыми outputs отклонён. Внешний hard-stop и отказ до greedy проверены с mock process/clock: готовый greedy публикуется точно, до готовности код не фабрикуется. Статика: `nbformat.validate`, AST всех ячеек/12 embedded modules, private/GPU/internet-off metadata, SHA/AST source equivalence. Suite с mock GPU launches проверила3 test→tune, cold timestamps, selective ZIP, неизменный первый answer, один bounded diagnostic и режим сдачи.

## Deadline guard и бюджет

GPU-generation и ожидание execution заканчиваются на160-й секунде общего180-секундного бюджета. Любая незавершённая строка (не все4+4 samples или нет execution) получает **уже сгенерированный greedy C-main**; завершённые строки голосуют полноценно. События, elapsed и fallback IDs сохраняются. После greedy готов полный `greedy-fallback.jsonl` и атомарный ready receipt. Внешний supervisor на172-й секунде убивает зависший GPU-процесс и публикует весь greedy за оставшийся резерв. Обработаны также ошибки после greedy; они не объявляются успешным mixed gate.

Linux parent-death guards вооружены до Torch import; каскад notebook watchdog → supervisor → GPU → executor убивает отдельные process groups и spawn workers. Все waits ограничены. Если инициализация/GPU зависли **до окончания greedy**, настоящий greedy fallback ещё отсутствует: такой запуск явно failed, фиктивных ответов нет. Абсолютный срок при зависшем драйвере/OS не доказуем CPU-тестом; внешний guard и реальный тройной T4 timer — обязательная проверка.

Notebook watchdog: kill на1740 с, аварийное завершение1770 с; API timeout1800 с. Нормально3×test +tune ≈9–11 мин. Автоматического K8 replay больше нет: ровно3test+tune. Watchdog ограничивает сессию; повторный GPU запуск возможен только как вторая и последняя исправленная итерация. Output — маленькие JSON/JSONL/logs и gzip tune candidates; веса остаются mounted, ZIP раскрывает только4 нужных JSON, initial/control/train не копируются. Интернет и Secrets не нужны.

## Ожидаемое время: гипотеза до T4

Номинальный сценарий, **не измерение** и не гарантированный диапазон: импорты/чтение/обе загрузки/encode≈5 с; warmup≈1 с; greedy≈18 с; main K4≈50 с; long K4≈50 с; CPU tail≈5 с; vote/write/shutdown/bookkeeping≈1 с. Итого **≈130 с**, рабочее ожидание120–150 с. Основание — прежний greedy main19,884 с и aggregate single K8 tune≈156 с, из которого предполагается убрать многократный executor startup/gc/I/O и скрыть CPU execution. В L2 отдельный GPU/CPU sample timer не сохранялся; main/long sampling и test-length distribution могут опровергнуть этот прогноз.

CPU service wall перекрывается с sampling и **не складывается** с критическими фазами. `timing.json` хранит неперекрывающиеся phases (сумма равна full timer), отдельный restore subtotal и overlapping execution service wall. При fallback <180 с не означает, что полный mixed уложился; gate требует также **0 fallback** во всех3 test.

## Единственный запуск ревьюера

Private локальный bundle: `thisarivson/nlp-hw02-final-inference`, T4, internet off, inputs `l2-live` + `26b-inputs`. Runtime строго прежний Torch2.11.0+cu128/CUDA12.8; при несовместимости остановиться, не расходовать квоту на установки. Физический accelerator Kaggle может выдавать T4×2, но процесс видит и использует **только GPU0**.

1. Независимое ревью исходников и CPU receipt; повторить build/checks.
2. **Один** API push/run с timeout1800, по [команде](../kaggle/README.md#финальный-инференс--готов-к-ревью).
3. Проверить3 cold test: каждый≤150 с (обязательно<180), 0 guard/OOM drift, phase breakdown. Корректный формат1919 строк.
4. Tune2048: ожидается около1492, допуск±1 п.п. Новые K4 не могут обещать точные1492 только на основании seed. При провале смотреть first4/greedy mismatch diagnostics и original K8/b32 replay; status остаётся `review_required`, даже если replay прошёл. Первый test answer не заменяется по tune score.
5. Итог `final-inference/answers.jsonl` побайтно первый test, рядом `summary.json`. После успешного gate `BENCHMARK=False` даёт один свежий test для будущей сдачи. Адаптивное досэмплирование не добавлено: сначала измерить полный фиксированный mixed; при нехватке времени отдельно проверить адаптивное правило на сохранённых журналах с потерей≤0,2 п.п.

## Источники

- **nlp-hw02-final-inference-v3**: пользовательский протокол13.10.2026 и предоставленная выгрузка GPU v2; [диагностика](../checks/final-inference-v2-diagnostic.json), [CPU benchmark](../checks/final-executor-v3-benchmark.json), [v3 checks](../checks/final-inference-v3.json), текущие исходники; bibliography.
- **nlp-hw02-final-inference-preparation**: пользовательский протокол13.10.2026, текущие исходники/CPU checks/source identity и локальный notebook; bibliography.
- **kaggle-l2-results**: [l2-results](l2-results.md), saved K8 journals/private livev11, [offline анализ](../research/analyze-l2-results.py); измеренные19,884/156 с и первый4+4 baseline.
- [execution-voting-decoding](execution-voting-decoding.md), [s2-fast-sampling-and-capacity](s2-fast-sampling-and-capacity.md): исходные vote/LP/execution/physical4D правила. [Условие](../../task/hw2-code-generation.md): одна T4/180 с, inference notebook, запрет дополнительных обучающих targets.

## V4 на финальных весах L3

**Локальная подготовка и CPU/статические gates пройдены; GPU gate ещё не запускался.** Private `friend1/nlp-hw02-final-inference-l3`, Internet off, одна T4/GPU0. [Ноутбук](../kaggle/notebooks/nlp-hw02-final-inference-l3.ipynb), отдельный ignored `bundles/final_inference_l3`; v3-бандл и `src/final_*.py` не менялись. [Сборщик](../kaggle/build_final_inference_l3.py), [source identity](../kaggle/final-l3-source-identity.json).

Входы через прежний `find_input`: **friend1/nlp-hw02-l3-live** и **thisarivson/nlp-hw02-26b-inputs**. В livev12 remote listing подтвердил оба FP16 l2-62280; manifest SHA/bytes совпали с локальными весами и metrics/checkpoint. C-main SHA8c11b103…/C-long2x d10bf001…. В inference worker меняются только две pinned SHA и metadata quality baseline1511; decoder/executor/restore/vote и timings каждого холодного test остаются v3-identical. Константы tag/baseline в session заменены на L3; настройка suite — по новому заданию пользователя.

**Ровно один cold test1919 +tune2048**, benchmark=True; цель test≤120с, обязательный потолок<180с, fallback0. Сессия720с, watchdog stop660/exit690, следующий процесс не стартует после480с. Ответ берётся из единственного test, остаётся исходный порядок и только `code`. Один test не проверяет repeatability; deterministic gate v3 из двух запусков здесь не применяется. Дополнительного K8 replay/адаптивного GPU run нет.

Выбран прежний K4+K4+main greedy: saved L3 tune **1511/2048=73,7793%**, ±1п.п. для новых K4. Против best single longK8: +0,20 [−0,84;1,23]п.п.; фиксированный pipeline оставлен по заданию пользователя, production superiority смеси не объявляется. Bucketing/b256 меняет RNG lanes относительно сохранённых first4/b32; GPU quality и полный single-T4 timing требуют нового прогона.

[Проверки](../checks/final-inference-l3-cpu.json), [скрипт](../kaggle/final_l3_checks.py): nbformat/compile/embedded parity/private/offline/T4/budgets/one-test suite; strict загрузка обеих настоящих L3 FP16,212finite FP16 tensors на модель,2вопроса greedy/K4/cap16 на CPU; весь saved first4 mixed1511 воспроизведён,16выбранных кодов повторно исполнены в новом executor без mismatches, JSONL формат проверен. CPU FP32 после загрузки не измеряет CUDA/FP16 throughput или качество новых полных K4.

## V5: адаптивное сэмплирование на L3

**Исправленная версия прошла CPU/офлайн/регрессии/статику; бандл готов (`ready=true`). Новый GPU gate ещё не измерен, push не выполнялся.** [Ноутбук](../kaggle/notebooks/nlp-hw02-final-inference-l3-v5.ipynb), [сборщик](../kaggle/build_final_inference_v5.py), ignored `bundles/final_inference_l3_v5`. V3/V4 и P1 не изменяются. Ниже разделены результаты предыдущего T4-прогона и прогноз исправленной версии.

### Почему предыдущий прогон не дал прироста

Пользователь предоставил output `friend1/nlp-hw02-adaptive-inference-l3-v5`: test timing (`../results/kaggle-runs/nlp-hw02-kaggle-l3/final/inference-v5/final-inference-l3-v5/test-1/timing.json`, нет в репозитории), tune timing (`../results/kaggle-runs/nlp-hw02-kaggle-l3/final/inference-v5/final-inference-l3-v5/tune/timing.json`, нет в репозитории). Test **141,107с**, tune **140,985с**; generation guard140с оборвал stage2 у обоих. Tune **1493/2048=72,9004%** вместо ориентира1533: многие неединогласные строки получили неполный набор голосов. Сравнивать этот результат с полным K6 как доказательство бесполезности адаптивности нельзя.

Проверка `generation_batches` уточнила исходную гипотезу: **батчи уже задавались числом последовательностей**. Stage1 —512вопросов×K2=1024последовательности, stage2 —256×дополнительныеK4=1024, greedy —1024×K1. OOM0; test peak reserved≈12,85GiB. Greedy1024 уже проверен T4; оснований обещать ускорение переходом на512 нет, он увеличивает число запусков. Оставлен1024 с уменьшением при OOM.

Основная проблема — хвост декодирования и недостаточный бюджет stage2. Test: greedy22,731с; main/long stage1 **29,423/29,325с**, по4батча; stage2 **38,359с**,4завершённых батча плюс оборванный. Tune: greedy27,888с; main/long stage1 **22,647/44,680с**; stage2 **37,883с**,5завершённых плюс оборванный. Несколько батчей доходили до cap512; decoder продолжает все lanes до завершения самого длинного. Поэтому время K2 не обязано быть вдвое меньше K4. CPU `sum_worker_task_wall`17–31с уже перекрыт GPU и не прибавляется к critical path.

### Исправления и выбранные параметры

Greedy C-main → **K2 C-main + K2 C-long2x**, T0.7 → execution. Если все5 кандидатов численные, конечные и `isclose(rel_tol=abs_tol=1e-6)` с greedy — стоп. Остальные добираются **до K4+K4 всего**, ещё2+2. `K1=2`, `K2=4` означают samples на модель; `unanimity_threshold=1.0`.

- Единый явный `sequence_budget=1024`: вопросов в батче `sequence_budget // K`. Greedy1024; stage1 512×K2; stage2 512×дополнительныеK2. K6/K8 поддерживаются: budget делится именно на число **добавляемых** samples. При OOM батч уменьшается, CPU/CUDA RNG откатываются, уменьшенный предел сохраняется для следующих батчей этой модели/K. Старые `sample_lanes`/`greedy_batch` принимаются для совместимости.
- Перед stage2 дождаться всех stage1 executions, чтобы составить **глобальный** приоритет. Сначала минимальная доля лидирующей численной компоненты среди всех5кандидатов; нечисленные программы входят в знаменатель, голосов не дают. Компоненты строятся по прежнему graph-isclose. Длина и row ID разрешают только равные доли. CPU execution всё ещё перекрывает greedy и обе stage1 GPU-фазы; короткий барьер учтён в прогнозе.
- Каждому приоритетному батчу дать main и long подряд, затем перейти к следующему. При guard самые неопределённые строки уже имеют обе модели; возможный оборванный текущий батч сохраняет доступные голоса. Seed и отдельные состояния RNG моделей сохранены. Смена порядка/формы батча меняет новые stochastic samples относительно старых журналов.
- Generation guard **160с**, execution166, verification170, supervisor172, потолок180. После планового окончания генерации20с на CPU-хвост, голосование, независимое переисполнение победителей и запись. Проверка deadline внутри decoder каждые16шагов может дать небольшое превышение160; новый T4 gate остаётся обязательным.
- В `generation_batches` добавлены row IDs и wall seconds каждого завершённого батча; в adaptive timing — глобальный порядок и доли лидера. Это позволит проверить реальную стоимость батчей и приоритет при следующем GPU-прогоне.

Голос неизменен: numeric graph connected components → **count / наличие greedy / средняя mean LP / порядок**. Повторные программы сохраняют голоса. Код — greedy в победившей компоненте, иначе максимальная mean LP. При deadline учитываются все исполненные samples; greedy-only и partial считаются отдельно. Опциональный scorer по умолчанию выключен; P1-рецензент в бандл не входит.

### Офлайн-эквивалентность и выбор K

[Replay](../checks/final-inference-v5-offline.json), [скрипт](../kaggle/final_v5_analysis.py): те же production `stage1_decision`, `replay_record`, `select_available`,128проверенных SHA журналов L3. Early stop **1133/2048=55,3223%** для всех K. Перестановка по приоритету при фиксированных кандидатах сохраняет все2048winner code/execution для каждого K4–K8. K4 совпадает с v3/v4 в2048/2048 кодах и исполнениях. Это эквивалентность **сохранённых кандидатов**, а не обещание одинакового CUDA-сэмплирования при другом порядке.

| K всего на модель | Frozen vote | Samples/задачу | Stage2 батчей на модель | Test, номинал | Осторожная оценка test/tune, максимум |
|---|---:|---:|---:|---:|---:|
| **4** | **1511 (73,7793%)** | **5,7871** | **2** | **132,1с** | **147,2с** |
| 5 | 1520 (74,2188%) | 6,6807 | 3 | 146,8с | 168,3с |
| 6 | 1533 (74,8535%) | 7,5742 | 4 | 161,4с | 189,4с |
| 7 | 1528 (74,6094%) | 8,4678 | 5 | 176,1с | 210,5с |
| 8 | 1531 (74,7559%) | 9,3613 | 6 | 190,8с | 231,6с |

**Выбран K4** — лучший по tune среди вариантов, проходящих generation≤160с и полный прогноз≤165с (≥15с до лимита). K5 даёт ещё9верных задач, но осторожная оценка168,3с не проходит; K6–K8 тем более. Это tune-only выбор, без eval/fresh и без гарантии прироста на новых samples. Прежнее расхождение пользовательского count/greedy/order с frozen count/greedy/LP/order документировано: K8 —1530 против1531. Frozen vote сохраняется; отдельного подтверждения для готовности текущая задача не требует.

### Новая оценка времени на T4

Расчёт опирается на **реально выполненные v5 батчи**, а не линейное масштабирование объёма samples или прежний v4-прогноз147,8с. Stage1 занимает8батчей суммарно: среднее **7,344с** на test, **8,416с** на tune. Для stage2 осторожная опора — **9,590с/батч**:38,359/4; в числителе есть время оборванного батча, поэтому это верхняя оценка по агрегату, не измерение отдельного батча. Новый logger даст точные batch timings.

Трудных строк860test/916tune. K4: **4stage1 +2stage2 батча на модель**, суммарно8+4, greedy ещё2. Greedy≈22,7/27,9с; stage1≈58,7/67,3с; stage2 по среднему≈29,4/33,7с. Измеренные imports/load/warmup/launcher и короткие финальные фазы уже включены. Получается **test132,1с, tune136,7с**.

Осторожный расчёт: сохранить все измеренные fixed phases, stage2 округлить до целых батчей и умножить на9,590×1,10; добавить2с на глобальный CPU barrier. Максимум по test/tune **147,246с**, generation≈146,300с, запас до180≈32,754с. K6 требует8stage2 батчей суммарно, K8 —12. Перестановка по неопределённости меняет padding и длинные хвосты;10% — допущение, не статистическая граница и не гарантия. Новый T4 прогон не выполнен.

### Проверки и готовность

[CPU/статика](../checks/final-inference-v5-cpu.json): sequence-budget и leading-fraction unit checks; subprocess complete/guard-stage1/guard-stage2/OOM/global-priority/priority-guard. Последняя регрессия проверяет, что при обрыве самый неопределённый длинный вопрос уже получил обе модели, более уверенный короткий — только stage1. OOM проверяет RNG rollback. [Tiny real decoder](../checks/final-inference-v5-real-decoder.json) использует настоящий tokenizer/pack/decoder/executor с выбранным K4, cap16, толькоCPU. Все2048выбранных программ независимо переисполнены; mismatches0. [V3](../checks/final-inference-v5-regressions.json) и [V4](../checks/final-inference-v5-v4-regression.json) регрессии, реальные FP16 loads, nbformat/compile/embedded source parity/веса SHA/private/offline/T4/one-test suite повторно прошли.

Kaggle **id и slug title согласованы с уже созданным kernel**: `friend1/nlp-hw02-adaptive-inference-l3-v5`, title `NLP HW02 adaptive inference L3 v5`. Файл notebook сохраняет имя `nlp-hw02-final-inference-l3-v5.ipynb`. Прежние L3-live/26b-inputs, SHA/tag l2-62280, Internet off, одна T4/GPU0. **Ровно один cold test1919 + tune2048**,720с максимум, watchdog660/690; никаких дополнительных GPU runs. GPU gate: test≤155с, <180с, fallback0/complete; quality1511±1п.п. и отсутствие guard. Ответ — единственный test-1 в исходном порядке, только `code`.

Сборка и проверки сначала сбрасывают `results/kaggle-runs/nlp-hw02-kaggle-l3/final/next-inference-v5.json` в `ready:false`; **`ready:true` и `bundle` публикуются только после всех локальных gates**. Это готовность бандла к следующему отдельно разрешённому GPU-прогону; новый GPU gate pending. В этой работе push/GPU0. Попытка коммита: `git add` явного списка18собственных файлов вернул exit128, `<vault>/.git/worktrees/study1/index.lock: Operation not permitted`. Индекс не изменён; правки сохранены в worktree, обход ограничений не выполнялся.

### Источники v5

- **nlp-hw02-final-inference-l3-v5**: пользовательский протокол14.10.2026,128 L3 journals, frozen v3/v4 code, выгрузка первого v5 T4 test/tune и `selection_analysis.py`; bibliography. Измеренные фазы и replay — наблюдения; новая оценка времени — собственный расчёт.
- [CPU/статика](../kaggle/final_v5_checks.py), [регрессии](../kaggle/final_v5_regressions.py), [identity](../kaggle/final-v5-source-identity.json), [final-inference-review](final-inference-review.md), [l3-results](l3-results.md).

`results/kaggle-runs/nlp-hw02-kaggle-l3/final/next-inference.json`: **ready=true**, абсолютный путь к отдельному бандлу и receipt. Это разрешает наблюдателю выполнить ранее авторизованный push; в этой работе push=0/GPU=0. Перед отправкой посылки проверить runtime Torch2.11.0+cu128/CUDA12.8, summary/timing/quality и отсутствие partial/fallback. L3 greedy посылки и полный разбор — [l3-results — Финал L3](l3-results.md#финал-l3); L4 — [l4-plan](l4-plan.md).

Источник **nlp-hw02-final-inference-l3-v4**: пользовательский протокол14.10.2026, saved L3 final/private livev12, точные source hashes и CPU receipt; bibliography.

## V6: адаптивный инференс с decoder all

**CPU/офлайн/регрессии/статика пройдены, ready:true; push не выполнялся.** [Ноутбук](../kaggle/notebooks/nlp-hw02-adaptive-inference-l3-v6.ipynb), [сборщик](../kaggle/build_final_inference_v6.py), `bundles/final_inference_l3_v6_mine`. Схема v5 сохранена: greedy C-main → K2+K2/T0.7 → для неединогласных **до K6+K6 всего на модель**. Глобальный приоритет — минимальная доля лидирующей численной компоненты среди пяти кандидатов; затем длина и ID. Каждый priority batch получает main и long подряд. Ровно один cold test1919 + tune2048; guard160 / execution166 / verification170 / supervisor172, сессия≤720с, watchdog660/690.

### Что перенесено

Production [final_inference_v6.py](../kaggle/src/final_inference_v6.py) вызывает тот же `final_speed.decode` и ровно те же `Options`, что профильный `all`: compaction при удалении≥25% lanes, async EOS32 с pinned snapshot/event, sharing cross-KV и mask для greedy K1, exponential-race sampler, lean greedy без LP/length updates. Self-KV остаётся предвыделенным, копируется только записанный prefix при compaction. PAD/BOS/EOS, untempered mean LP, capacity512 и восстановление исходных row/lane IDs сохранены. CUDA graphs остаются выключены. Compaction/RNG меняют новые stochastic samples; эквивалентность журналов не означает token identity новых сэмплов.

[Новый executor](../kaggle/src/final_executor_v6.py) начинает с **2 workers**; отдельный Torch-free [service](../kaggle/src/final_execute_v6.py) применяет `nice(+10)`, приоритет наследуется workers. CUDA dispatcher не меняет nice. После генерации команда расширяет сервис **до4 workers** без перезапуска: текущие задачи, очередь, code→ID cache и готовые результаты сохраняются. `drain` ждёт также подтверждения нового числа workers. Повторная независимая проверка winners использует свежий сервис с≤4 workers. Записываются `worker_history`, фактические nice/workers/error и decode compactions. Сохраняются прежний isolated numeric worker, AST/SymPy policy, wall3/CPU2 и parent-death guards. Локальная macOS sandbox может запретить nice; отказ фиксируется, обхода нет. На предоставленном T4-профиле nice применился.

Vote — frozen count → greedy → mean LP → order, graph-isclose1e-6. При deadline голосуют все готовые исполненные кандидаты; дубликаты дают отдельные голоса, код greedy берётся внутри победившей компоненты. OOM retry восстанавливает CPU/CUDA RNG и сохраняет уменьшенный batch; sequence_budget1024 делится на число добавляемых samples. Для K6 stage1 —512×K2, stage2 —256×дополнительныеK4.

### Профиль T4 и известное FP16-отличие

Источник измерений: summary profile v2 (`../results/kaggle-runs/nlp-hw02-decode-profile/2/decode-profile/summary.json`, нет в репозитории), только C-main/test1919/generation; это не полный v6 pipeline timer.

| Режим | Baseline→all без нагрузки | Baseline→all с нагрузкой |
|---|---:|---:|
| Greedy |22,321→11,670с,1,91×|27,629→16,679с,1,66×|
| K4 |42,817→32,350с,1,32×|62,010→37,796с,1,64×|

Нагрузка в all —2 workers/nice10, baseline —4 workers. Основной выигрыш даёт сочетание flags; отдельные абляции не заменяют измеренный all. **Известное допустимое отличие по решению пользователя:** greedy all отличается в1/1919 строк без нагрузки (ID274) и3/1919 с нагрузкой (IDs358,1563,274). Compaction меняет FP16 GEMM/SDPA batch shapes и округление при той же математике. Поэтому `GPU_gate_passed:false` исходного профиля из-за строгого bit-exact не переименовывается в успех; v6 явно допускает эти расхождения. Полный v6 quality/timer ещё требует запуска.

### Выбор K_max по времени и L3 replay

[Расчёт/replay](../checks/final-inference-v6-offline.json), [скрипт](../kaggle/final_v6_analysis.py). Измеренные fixed phases каждого v5 test/tune сохраняются целиком: холодные imports/load/warmup, executor startup, restore/tail/vote/reexecution/write. Greedy делится на measured loaded speedup1,6565; stage1/stage2 — на осторожные1,6 против measured1,64065. Число stage2 батчей округляется вверх; используется худшая старая proxy9,590с/батч×1,10, добавляются2с на глобальный CPU barrier/expansion. Это собственная оценка, не статистическая граница: C-main K4 synthetic load экстраполируется на K2/K4/K6, C-long и другую сортировку.

| K всего на модель | L3 frozen matches | Номинал test/tune | Осторожно test/tune |
|---|---:|---:|---:|
| **6** |**1533/2048,74,8535%**|110,4/110,8с|**126,4/121,5с**|
| 8 |1531/2048,74,7559%|128,7/131,9с|152,8/147,9с|

**K_max=6:** K8 не проходит пользовательский порог≤150с с запасом по худшему холодному test; K6 оставляет≈33,6с до guard160 и≈53,6с до180. Выбор сделан по времени, без оптимизации по tune accuracy. User-simulation K8 даёт1530: прежняя разница count/greedy/order против frozen count/greedy/LP/order сохранена. Early stop1133/2048 для обоих K. Все128 SHA L3 journals подтверждены; production replay/глобальная priority schedule дают ровно прежние v5 winner code и execution для всех2048 строк при K4/K6/K8. K4 также совпадает с v3/v4.

### Сборка и подмена весов

Metadata id `thisarivson/nlp-hw02-adaptive-inference-l3-v6`, title `NLP HW02 adaptive inference L3 v6` имеет тот же slug. Private, Internet off, T4/GPU0, datasets `thisarivson/nlp-hw02-l3-live` + `thisarivson/nlp-hw02-26b-inputs`. По умолчанию обе FP16 L3 l2-62280 с pinned SHA8c11b103…/d10bf001…. Из `work/kaggle`, Python с Torch/tokenizers/sympy/nbformat/numpy (локально `/tmp/nlp-s3-env/bin/python`):

```bash
python final_v6_analysis.py
python build_final_inference_v6.py
python final_v6_regressions.py
python final_v6_checks.py
```

Другой dataset/пара (например, готовые P1 A/B) задаются **только параметрами сборки**, без редактирования worker/session:

```bash
python build_final_inference_v6.py \
  --weights-dataset OWNER/DATASET \
  --main-weight MAIN-FP16.pt \
  --long-weight LONG-FP16.pt
```

Названия — точные basenames внутри dataset. Model-only payload format/spec и strict state_dict остаются обязательными; фактические SHA пишутся в timing. Для custom pair L3 journals не требуются, L3 accuracy1533 остаётся сравнительным ориентиром quality±1п.п. Можно дополнительно закрепить **оба** `--main-sha256` и `--long-sha256`; без них custom weights проверяются форматом/загрузкой, не pinned L3 digest. Пересборка сначала сбрасывает ready; static/readiness checker намеренно рассчитан на default L3 bundle, для замены весов нужны отдельные проверки новой пары. Сами P1-файлы не изменяются.

### Проверки и готовность

Локальные gates прошли; результаты в [CPU/static receipt](../checks/final-inference-v6-cpu.json) и [v3–v5 regressions](../checks/final-inference-v6-regressions.json). Проверяются реальные L3 FP32 greedy4 разных длин/cap512 против baseline all, cache prefix/reorder/continue, staggered EOS/lanes/LP, categorical race distribution и K6 при вырожденном распределении. CPU не проверяет CUDA async ветку и FP16 speed; она уже исполнялась в предоставленном профиле, полный v6 timer/memory/quality pending.

V6 fixtures: complete/guard-stage1/guard-stage2/OOM/global-priority/priority-guard с прежними v5 assertions, настоящий executor/vote/reexecution; tiny real tokenizer/decoder/pack pipeline на выбранном K6. Отдельно live2→4 с незавершённой очередью/дедупликацией и неизменным parent nice; mock-suite ровно test→tune, отсутствие gold в test, первый answer, а также alternate dataset/A-B filenames без L3 journals. NBFormat, compile всех embedded modules/cells, SHA/source parity, frozen v3–v5 и metadata/budgets проверяются; все2048 K6 winners независимо переисполнены:2041 unique, mismatches0. Реальные L3 FP32 baseline/all greedy совпали во всех4 строках, compaction4→3→1.

`results/kaggle-runs/nlp-hw02-kaggle-l3/final/next-inference-v6.json` получает **`bundle` и `ready:true` только после всех локальных gates**. Это готовность артефакта к запуску; GPU v6 ещё не измерен. Push запрещён текущим заданием и не выполняется. Ответ будущего прогона — единственный test-1, исходный порядок, только `code`.

Попытка коммита v6: `git add` явного списка18 собственных файлов завершился exit128 — `<vault>/.git/worktrees/study1/index.lock: Operation not permitted`. Песочница запретила staging; индекс не изменён, коммит не создан. Бандл и ready receipt сохранены.

### Источники v6

- **nlp-hw02-final-inference-l3-v6**: пользовательский протокол14.10.2026, предоставленный T4 profilev2,128 L3 journals, прежние v5 T4 phases и собственные CPU/regression/static receipts; bibliography. Профиль/replay — наблюдения, прогноз времени и выбор K6 — собственный расчёт.
- [decode-speed](decode-speed.md), [l3-results](l3-results.md), frozen v3–v5 и `final_speed.py`; новые исходники/сборщик связаны выше.

## V7: ансамбль из трёх моделей

**Локальные CPU/replay/v3–v6/memory-budget/static gates прошли; ready:true, kernel push=0.** [Ноутбук](../kaggle/notebooks/nlp-hw02-ensemble-inference-v7.ipynb), [сборщик](../kaggle/build_final_inference_v7.py), [worker](../kaggle/src/final_inference_v7.py), [решения и vote](../kaggle/src/final_ensemble.py), [session](../kaggle/src/final_session_v7.py). Бандл — `work/kaggle/bundles/final_inference_v7_mine`, указатель — `results/kaggle-runs/nlp-hw02-kaggle-l3/final/next-inference-v7.json`, ключи `bundle` и `ready:true`. Это готовность к холодному GPU-прогону; фактические N3 peak memory, новые samples и полный timer ещё не измерены.

### Конфигурация и исполнение

Список из **2–4 моделей** задаёт порядок голосов и содержит `name`, `filename`, `sha256`, `dataset`, `K1`, `K2`. `K2` — **общее**, а не добавочное число сэмплов модели. Greedy создаёт только первая модель. Выбранный [конфиг](../kaggle/final-v7-config.json):

| Модель, в порядке голосования | Файл FP16 | K1 | K2 всего | SHA256 |
|---|---|---:|---:|---|
| C-main L3 | `model-C-main__weights__l2-62280-fp16.pt` | 1 | 3 | `8c11b1038855422b6912e243049a554317978f848edb412ed48aebbe4cb6b886` |
| C-long2x L3 | `model-C-long2x__weights__l2-62280-fp16.pt` | 1 | 5 | `d10bf0018d0b14fd6cc3c8b9c7121f127b5cec97057544fd63e5824253a55057` |
| pilot-B | `model-pilot-b__weights__long-truncation-fp16.pt` | 1 | 4 | `8439037720dda16dda6b73fe676860d5498bbcba4cacebc0d709bbb22db464a5` |

Этап1: greedy + по одному T0.7 sample каждой модели. Только единогласные finite numeric результаты останавливаются; ошибки остаются в знаменателе неопределённости. Остальные задачи сортируются по доле крупнейшей graph-isclose компоненты, затем длине tokenized question и ID. Этап2 добавляет 2/4/3 sample в таком же порядке моделей; общий priority wave256 задач выбран по максимальному extra4 при sequence_budget1024. Все модели одной волны обрабатываются до следующей волны. При OOM batch уменьшается с восстановлением CPU/CUDA RNG; каждая модель продолжает собственный RNG stream.

Кандидаты уплощаются в порядке greedy→все C-main→все C-long2x→все pilot-B. Frozen v3–v6 vote — count→greedy→meanLP→order, graph-isclose1e-6; внутри победившей компоненты код greedy сохраняется, иначе берётся code кандидата с максимальным meanLP. Неисполненные кандидаты исключаются, повторные программы сохраняют отдельные голоса. Reviewer отключён. Decoder `all`, executor2 nice10→до4, guards160/166/170/172 и полный timer перенесены из v6 без изменения. Модели resident одновременно, encoder/KV/decoder активны только для одной модели за раз.

### Офлайн-выбор K и парное сравнение

[Replay/расчёт](../kaggle/final_v7_analysis.py), [полный receipt19 вариантов](../checks/final-inference-v7-offline.json). Проверены SHA192 K8 tune-журналов, совпадение IDs, question SHA и references трёх моделей. Пользовательская full-проверка воспроизведена: greedy+6/5/5 даёт1538/2048 при count/greedy/order; frozen vote с meanLP даёт1539. Full L3 greedy+8/8 —1530/1531 соответственно. Эти16 samples превышают бюджет v7 и не стали конфигом.

Для выбора требуются одновременно максимум≤12 samples, среднее≤v6 и осторожный прогноз≤140с. Среди прошедших вариантов выбирается максимальное tune matches; ничья — меньший прогноз/среднее, затем порядок объявленного grid. Порядок K ниже: C-main/C-long2x/pilot-B.

| K1 → K2 | Matches /2048 | Точность | Среднее samples без greedy | Прогноз test/tune, с | Осторожно test/tune, с |
|---|---:|---:|---:|---:|---:|
| 1/1/1 → 3/3/3 |1520|74,2188%|5,3906|95,3 /91,7|107,3 /99,7|
| 1/1/1 → 4/3/4 |1529|74,6582%|6,1875|109,1 /107,5|127,1 /119,5|
| 1/1/1 → 4/4/4 |1528|74,6094%|6,5859|109,1 /107,5|127,1 /119,5|
| **1/1/1 → 3/5/4** |**1535**|**74,9512%**|**6,5859**|**109,1 /123,3**|**127,1 /139,3**|
| 1/1/1 → 3/4/5 |1535|74,9512%|6,5859|109,1 /123,3|127,1 /139,3|
| 1/1/1 → 6/3/3 |1528|74,6094%|6,5859|122,8 /123,3|146,9 /139,3|
| 2/2/2 → 4/4/4 |1528|74,6094%|8,8799|122,8 /123,3|134,9 /131,3|
| v6: 2/2 → 6/6 |1533|74,8535%|7,5742|110,4 /110,8|126,4 /121,5|

Выбрано **1/1/1→3/5/4**:1232 early-stop,816 escalations, максимум12 samples, среднее на13,05% меньше v6. Парно к v6: **33 исправлено,31 сломано**, Δ**+0,0977п.п.**, **95% CI[−0,6681;+0,8635]п.п.** CI — нормальный интервал среднего парных correctness differences. K выбран на том же tune; интервал pointwise, без поправки за выбор конфигурации. Подтверждённого прироста качества нет; frozen eval/fresh этим выбором не затронуты.

Время — собственная экстраполяция тех же v5 cold phases/T4 profile, что v6. Сохранён fixed overhead, добавлены половина measured load+warmup на третью модель:3,284с для test/1,920с tune. Greedy speedup1,6565, sampling1,6, худший old batch9,590с×1,10 и2с на barrier/expansion. Считаются округлённые этап1 batches6 и этап2 batches9/12. Даже неполные lanes маленьких extra оплачиваются как полные батчи. Осторожный максимум139,267с оставляет20,73с до guard160.

> [!warning]- Ограничение прогноза
> Это не статистическая верхняя граница и не новый GPU timer. Test escalations спроецированы из tune с measured test:v6-tune отношением:767 задач; при переходе за768 появляется ещё одна волна. Другая доля несогласия, длинные программы pilot-B, padding, compaction и CPU scheduling могут изменить время. Guard и частичное голосование сохраняются; окончательный gate — холодный запуск.

### Private P1 weights dataset и запуск

[thisarivson/nlp-hw02-p1-weights](https://www.kaggle.com/datasets/thisarivson/nlp-hw02-p1-weights) **создан через Kaggle CLI** с default `~/.kaggle`, без чтения/вывода credential contents и без auth env overrides. CLI сообщил private creation; status ready, owner thisarivson, шесть remote filenames/bytes проверены. Состав: FP16 pilot-a, pilot-b, reviewer и три исходных checkpoint JSON. SHA pilot-a `ef0fb6de…`, pilot-b `84390377…`, reviewer `aeb7d645…`; полные digests/bytes — в [dataset receipt](../checks/final-inference-v7-dataset.json). Исходные P1-файлы и checkpoint JSON не изменены.

[Stager](../kaggle/prepare_p1_weights_dataset.py) воспроизводит папку `work/kaggle/staging/p1_weights_mine` с `dataset-metadata.json`, проверяя каждый SHA. Команда для нового создания из `work/kaggle` при отсутствии созданного dataset:

```bash
env -u KAGGLE_USERNAME -u KAGGLE_KEY -u KAGGLE_API_TOKEN -u KAGGLE_CONFIG_DIR \
  kaggle datasets create -p staging/p1_weights_mine -r skip
```

Бандл: id **`thisarivson/nlp-hw02-ensemble-inference-v7`**, title **`NLP HW02 ensemble inference v7`**, совпадающий slug; private, Internet off, одна T4/GPU0. Входы: `thisarivson/nlp-hw02-l3-live`, `thisarivson/nlp-hw02-p1-weights`, `thisarivson/nlp-hw02-26b-inputs`. Ровно один cold test1919→tune2048, session≤720с, watchdog660/690. Ответ всегда первый test-1, исходный порядок, только code. Test не получает gold; tune reference используется после закрытия answer timer. Live gates: test≤140с, оба запуска complete/<180с, quality±1п.п., фактические N/model SHA и peak memory.

Воспроизводимые локальные команды (Python с Torch/tokenizers/sympy/nbformat):

```bash
python prepare_p1_weights_dataset.py
python final_v7_analysis.py /path/to/selection_analysis.py
python build_final_inference_v7.py
python final_v7_regressions.py
python final_v7_checks.py
```

Custom ordered2–4 ensemble: `python build_final_inference_v7.py --config ensemble.json`; формат — как `final-v7-config.json`. Пересборка сбрасывает ready. Default readiness checker сверяет именно выбранный N3 конфиг; custom набор требует собственных weight/memory/quality checks. Kernel push в этой работе не выполнялся.

### Проверки памяти и эквивалентности

[CPU/static receipt](../checks/final-inference-v7-cpu.json), [v3–v6 regressions](../checks/final-inference-v7-regressions.json), [source identity](../kaggle/final-v7-source-identity.json), [проверки](../kaggle/final_v7_checks.py). Все frozen receipts сохранены. Historical L2 weights локально отсутствуют: только mock legacy session получил temporary unused weight paths; реальные L2 journals, L3 FP16 loads/decoder и v3–v6 guard/OOM/priority/executor checks исполнялись без этой подмены.

N2 helper точно повторяет v6 stop/count/code/execution на всех2048 задачах. N3 replay19 конфигов совпадает с независимым union-find oracle и полным priority wave schedule, включая реальные tokenized lengths. Все2048 выбранных winners независимо переисполнены без mismatch. Девять v7 fixtures покрывают N2/N3/N4, обе стадии guard, OOM/RNG, глобальный приоритет, priority guard и модель с zero extra. Tiny real tokenizer/decoder/pack/executor и отдельный cold subprocess supervisor→worker с pinned SHA проходят; повторный stale output отклонён. Mock suite подтверждает test→tune, три входа, первый answer и отсутствие gold в test. NBFormat/compile/SHA/frozen decoder/executor паритет и custom2/4 builder прошли.

Каждая модель:53 646 403 FP16 элемента,107 292 806байт; три resident weights **306,967MiB**. Реальные payload SHA/dtype/spec/strict loads проверены. Self-KV при1024 lanes/cap512 —6GiB, не умножается на N. Исторические all(one-model) peak reserved13 778 288 640байт и v5(two-model)13 795 065 856байт дают projected N3 **13,032GiB**; плюс512MiB allocator reserve —**13,532GiB**, ещё≈479,36MiB до консервативного14GiB budget. Это memory-budget check, **не фактическое измерение трёх моделей на T4**; live timing пишет resident bytes, actual allocated/reserved и device total, а OOM retry уменьшает batch.

Файлы D1 не изменялись. Попытка коммита: собственные16 файлов выбраны явно, отдельный patch только v7-строк index/progress/библиографии прошёл `git apply --check --cached`; D1-строки исключены. `git add` завершился exit128: `<vault>/.git/worktrees/study1/index.lock: Operation not permitted`. Песочница запретила staging; индекс остался пустым и неизменным, коммит не создан. Работа и ready receipt сохранены.

### Источники v7

- **nlp-hw02-ensemble-inference-v7**: пользовательский протокол14.10.2026,192 K8 tune-журнала L3/P1 с recover manifests, checkpoint SHA, `selection_analysis.py`, T4 profilev2/v5 phases и собственные CPU/regression/static receipts; bibliography. Replay — наблюдение; выбор K и прогноз времени/памяти — собственный расчёт.
- [p1-results](p1-results.md), [l3-results](l3-results.md), [decode-speed](decode-speed.md), [v6](#v6-адаптивный-инференс-с-decoder-all).


## V8: C-long2x / P1-B / W plain B4

По сообщению пользователя19.10: **78,54% public**, v7 —77,92% (Δ+0,62п.п.). Это пользовательский факт с лидерборда, без независимого API-аудита и парных CI. После [b5-results](b5-results.md) v8 остаётся основным вариантом; условный v9 — ниже.

Выбрана фиксированная политика из [b4-results — K8 tune2048 и ансамбли](b4-results.md#k8-tune2048-и-ансамбли): **greedy C-long2x; K1=1/1/1 → K2=1/8/1**, T0,7, seed270042. Максимум10 и среднее5,8027 stochastic samples по сохранённым журналам;1228 early stops. Reviewer выключен. Новый GPU tune использует новые samples: воспроизведение1541 не предполагает их побитного совпадения.

| Модель / источник | Охлаждённые FP16-веса | SHA256 |
|---|---|---|
| C-long2x L3 / `thisarivson/nlp-hw02-l3-live` | `model-C-long2x__weights__l2-62280-fp16.pt` | `d10bf0018d0b14fd6cc3c8b9c7121f127b5cec97057544fd63e5824253a55057` |
| P1-B / `thisarivson/nlp-hw02-p1-weights` | `model-pilot-b__weights__long-truncation-fp16.pt` | `8439037720dda16dda6b73fe676860d5498bbcba4cacebc0d709bbb22db464a5` |
| W plain B4 / kernel output `thisarivson/nlp-hw02-b4/1` | `model-GPU0/weights/b4-final-091801-fp16.pt` | `777b21996352df7c489af4efc17f6b8922ff6e005775f0f601d12143c8c1496c` |

W соответствует точке `model-GPU0/points/b4-final-091801`; `checkpoint.json` ссылается на файл в общем `model-GPU0/weights/`. Это cooled model-only payload без optimizer. W:6+6/d768/12heads/SwiGLU2048/vocab8195,113426435параметров; headwise Q/K LayerNorm с FP32 statistics и обучающим z-loss1e−4. Z-loss при инференсе не вычисляется. Базовый B4 test timer из `l1_eval.evaluate`: read→повторная загрузка модели/tokenizer→FP16 greedy batch256→restore→execute→answers close; imports и предшествующая heldout-оценка вне него. Его30,88с не являются холодным временем v8. Peak stats B4 не сбрасываются перед test и могут включать прежнюю heldout/CE-оценку; это не изолированный peak mixed decoder.

[Сборщик](../kaggle/build_final_inference_v8.py), [ноутбук](../kaggle/notebooks/nlp-hw02-ensemble-inference-v8.ipynb), [конфиг](../kaggle/final-v8-config.json). W подключена через `kernel_sources`, C — прежними dataset sources v7; пути разрешают `find_input` / `resolve_cache_mount`, для W закреплён относительный путь GPU0. [Session v8](../kaggle/src/final_session_v8.py) отличается от v7 только разрешением этого пути. **Worker, fast decoder, QK attention, selector, executor и supervisor побайтно совпадают с embedded v7.** QK-norm уже поддерживалась в v7: norm q перед SDPA, norm k при записи self/cross KV; fused encoder path выключен. Сохранены sequence budget1024, compaction≥25%, async EOS32, cross-KV sharing при K1, per-model RNG/OOM rollback, executor2 nice10→4 и guards160/166/170/172. Модели resident одновременно, кеш активен для одной; FP16 параметры всех трёх441438482bytes. Только self-KV W при1024×512 занимает9GiB; фактический T4 peak проверяется отдельно, OOM уменьшает batch прежним механизмом.

[CPU-проверки](../kaggle/final_v8_checks.py):

- [Replay](../checks/final-inference-v8-offline.json): **1541/2048=75,2441%**,1228 early stops; winner code/execution совпали во всех2048строках с `B.Replay.evaluate` из анализа B4. Проверены SHA192 C/P1-журналов и64W-журналов, IDs/question SHA/references. Число1541 — exploratory tune, не независимая оценка.
- [Веса](../checks/final-inference-v8-weights.json): SHA, все FP16 tensors finite, strict load, ключи/формы/значения после FP16→FP32; W284ключа, включая72QK norm tensors.
- [Greedy W](../checks/final-inference-v8-greedy.json): heldout IDs0/1/100/1000, cap512 — **4/4 побитно совпали с B4 GPU FP16 generations**; CPU baseline/all также равны. Cached/full-prefix logits: max abs difference1,91e−5. Это проверка четырёх строк; FP16/FP32 и формы батчей не обещают тождество всего heldout.
- [Регрессии v7](../checks/final-inference-v8-regressions.json):9fixtures complete/N2/N4/guard1/guard2/OOM/priority/priority-guard/zero-extra, настоящий pipeline, embedded cold supervisor/load/SHA и полный2048offline/reexecution — passed. Frozen v7 receipts не перезаписывались.
- [V8 mixed pipeline](../checks/final-inference-v8-pipeline.json): настоящий embedded cold pipeline с разной шириной C/W и QK W, strict SHA load; только P1-B получает stage2, всего10samples; формат/таймер/переисполнение passed. [Статика](../checks/final-inference-v8-static.json): nbformat/compile/SHA/source parity, private/offline/single-T4/mounts/guards passed. Локально Torch2.11.0 CPU; GPU worker требует2.11.0+cu128/CUDA12.8.

### Измерение T4 и ответ v8

Публичный pool `NvidiaTeslaT4` выделяет пару T4 ([подтверждение Kaggle](https://github.com/Kaggle/kaggle-cli/issues/1196)); бенчмарк использует **только одну T4**, GPU0. `CUDA_VISIBLE_DEVICES=0` устанавливается до Torch import, worker требует `device_count()==1`; второй GPU не участвует. bibliography — kaggle-cli-t4-pool.

Private [Kaggle v8, версия3](https://www.kaggle.com/code/thisarivson/nlp-hw02-ensemble-inference-v8/versions?versionNumber=3) — **COMPLETE / все gates passed**. [Компактный результат](../results/final-inference-v8.json), [валидация output](../checks/final-inference-v8-output.json), ignored raw `results/kaggle-runs/nlp-hw02-ensemble-inference-v8/3/ensemble-inference-v8`. Все20embedded modules в скачанном output совпали по SHA с проверенным source identity; source cells текущего private notebook совпали с локальными. Одна активная T4/GPU0, модели resident одновременно.

| Полный pipeline | Время, с | Early stops | Mean samples | Peak allocated / reserved, GiB |
|---|---:|---:|---:|---:|
| Cold test1919 | **91,924169** |1138|5,848880|**12,9871 /14,2227**|
| Tune2048, отдельный свежий child | **65,101563** |1266|5,672852|**13,8958 /14,4043**|

Timer — от запуска interpreter до закрытия answers: imports, model/tokenizer load, warmup, decode/restore, execution, vote, независимое переисполнение и write включены. Общая notebook session161,788с. На test imports5,469с/load7,677с/warmup1,286с; greedy8,626с, stage1 long/B/W10,211/7,221/18,049с, stage2 P1-B25,012с. До цели140 остаётся48,076с, до generation guard160 —68,076с.

Guard/fallback/partial и mismatch переисполнения — **0** на обоих splits; все голоса получены. На test обработан **один OOM W**, stage1 offset1024, question_batch1024/K1: прежний retry откатил RNG и уменьшил batch до512. Это не ответный fallback и не изменение K. Tune OOM0. Device total14,5625GiB; tune reserved оставляет162MiB, allocated —≈682MiB. Поэтому OOM retry остаётся необходимой частью пути; новую настройку по памяти не подбирали.

Новый GPU tune: **1532/2048=74,8046875%**, против сохранённого offline1541 — **−9 /−0,439453п.п.** Runtime samples отличаются от старых K8 batch32: другие формы/compaction, exponential-race и потребление RNG. Это не расхождение селектора: offline2048winner parity exact; production replay скачанного runtime-журнала снова дал1532 и все выбранные code совпали с tune answers. Политику по новому tune не меняли; greedy anchor/K/веса прежние.

**Ответ:** [final-ensemble-v8.jsonl](../submissions/final-ensemble-v8.jsonl), **1919строк**, в каждой только строковый `code`, порядок исходного test подтверждён input/question-order SHA и frozen row mapping. SHA256 `0b23fd5d8e639dc5da43af5b6752374d720e8b3f614f42465f058c3bd81fff60`. С [final-ensemble3-v7.jsonl](../submissions/final-ensemble3-v7.jsonl) отличаются **1216строк code**. В бота ничего не отправлялось.

API: первая push-попытка≈30мин оставалась QUEUED без execution output; наблюдались краткий CANCEL_ACKNOWLEDGED и сетевые ошибки, затем QUEUED. Один повтор того же notebook/SHA завершился успешно. CLI вернул версии1 и3; происхождение промежуточной2не подтверждено, дополнительных push не было. Статусы и metadata сохранены в [launch](../checks/final-inference-v8-launch.json), [history](../checks/final-inference-v8-status.json), [remote parity](../checks/final-inference-v8-remote-versions.json). Лимит каждой из двух наших push-попыток720с, notebook watchdog660/690, суммарный запрошенный предел24мин<40мин. Снимки API квоты перед повтором/после COMPLETE:21,92→21,97used GPU-ч, **≈3мин** наблюдённого расхода (округление счётчиков даёт±36с); осталось8,03ч. Credentials contents не читались; всё private, на Kaggle ничего не удалялось.

Попытки коммита кода/CPU gates и полного финального набора: `git add` собственного списка оба раза завершился128 — песочница запрещает создание `.git/worktrees/study1/index.lock` (`Operation not permitted`). Staged index пуст и не изменился; коммит не создан. [Квитанция и точный финальный список](../checks/final-inference-v8-git.json). B5 не включалась; наши две записи общей библиографии вынесены в [отдельный patch](../checks/final-inference-v8-bibliography.patch), чтобы не включить чужую B5-запись.

### Источники v8

- **nlp-hw02-ensemble-inference-v8** — собственная сборка/CPU-аудит и private T4 run18.10.2026; bibliography — nlp-hw02-ensemble-inference-v8. [Source identity](../kaggle/final-v8-source-identity.json), [V7: ансамбль из трёх моделей](#v7-ансамбль-из-трёх-моделей), [decode-speed](decode-speed.md).
- **kaggle-b4-results** — cooled W checkpoint, producer manifest и K8-журналы, [b4-results — K8 tune2048 и ансамбли](b4-results.md#k8-tune2048-и-ансамбли); bibliography — kaggle-b4-results. Новых внешних научных источников нет.

## V9: C-long2x B5 / P1-B / W B5

**Лидерборд (сообщения пользователя):** v7 — 77,92%, v8 — **78,54%** (лучший, первое место), v9 — 78,44%. На свежем GPU-tune v8 и v9 одинаковы (1532/2048), хотя offline replay K8-журналов давал 1541 и 1547: выбор политики на тех же журналах завышает выигрыш. Вывод: сравнивать кандидатов по свежему GPU-прогону tune; финальный кандидат на данный момент — v8.

**Прогон прошёл; ответ готов.** После отменённых batch-версий внешний оркестратор предоставил исполненный интерактивный notebook и output. Обе code cells побайтно совпадают с подготовленнымv9; проверены20embedded module SHA, реальные model SHA, порядок строк и SHA ответов. [Итог](../results/final-inference-v9.json), [валидация](../checks/final-inference-v9-output.json), [сравнениеv8](../checks/final-inference-v9-comparison.json).

### Политика и источники весов

По явной просьбе пользователя собрана B5-версия для реального leaderboard-замера; B6 не является условием запуска. Политика заморожена до GPU: **greedy C-long2x B5; K1=1/1/1 → K2=1/6/5**, T0,7/seed270042, reviewer off. По [b5-results — K8 tune2048 и ансамбли](b5-results.md#k8-tune2048-и-ансамбли) это максимум matches при осторожном proxy≤140с; при равенстве выбраны меньший proxy, затем меньшая средняя вычислительная цена.

**1547/2048=75,5371% против v8 offline1541=75,2441%**: +6 ответов,50исправлено/44сломано, Δ+0,293п.п., парный bootstrap95% CI **[−0,635;+1,221]п.п.** CI описательный, без поправки за22025политик. Weighted mean cost7,480 против6,149. Новые GPU samples не обязаны повторить1547; v8 GPU1532 не является согласованным offline comparator. Осторожный proxy136,84с, nominal test118,03с, stress152,79с — **прогнозы**, не измеренияv9. V8 public78,54% остаётся пользовательским фактом; приростv9 на public не измерялся.

| Модель / pinned private output | Payload | SHA256 |
|---|---|---|
|C-long2x B5 / `friend1/nlp-hw02-b5/6`|`model-GPU1/weights/b5-final-113690-fp16.pt`|`32475040a951f74869663b740d0047ddfdc233309d55d5eb0dfb28c2068860c7`|
|P1-B / `friend1/nlp-hw02-kaggle-p1/3`|`model-pilot-b/weights/long-truncation-fp16.pt`|`8439037720dda16dda6b73fe676860d5498bbcba4cacebc0d709bbb22db464a5`|
|W B5 / `friend1/nlp-hw02-b5/6`|`model-GPU0/weights/b5-final-125641-fp16.pt`|`30ff358be26dc3b0e9093fd73f3f56539ac00c33b8eaf00f9963c015ba14755a`|

B5 payloads скачаны через явный version_label=v6; manifest полностью совпадает с snapshot анализа B5. P1 dataset thisarivson закрыт для аккаунта друга, поэтому взят исходный private output P1v3: payload побайтно равен v8. Inputs `thisarivson/nlp-hw02-26b-inputs` доступны через CLI. UI-шаринг не нужен; [доступы](../checks/final-inference-v9-access-friend1.json), [скачивание/SHA](../checks/final-inference-v9-download.json). Содержимое Kaggle-конфигов не читалось и не выводилось.

### Сборка и CPU gates

[Сборщик](../kaggle/build_final_inference_v9.py), [конфиг](../kaggle/final-v9-config.json), [ноутбук](../kaggle/notebooks/nlp-hw02-ensemble-inference-v9.ipynb), [source identity](../kaggle/final-v9-source-identity.json). Все20embedded modules побайтно совпадают с v8, включая session/mounts/decoder/QK/selector/executor/supervisor. Guards160/166/170/172, sequence budget1024, RNG/OOM rollback сохранены.

- [Production replay](../checks/final-inference-v9-offline.json):1547matches,1233early stops,2048/2048winner code/execution exact с B.Replay и независимым oracle. Проверены journal SHA/IDs/question SHA/references и максимум/дешевизна по сохранённой полной сетке.
- [Strict-load](../checks/final-inference-v9-weights.json): FP16 finite, формат model-only/без optimizer, ключи/формы/значения после FP16→FP32, pinned payload/manifest passed. Resident441438482bytes; W284ключа/72QK norm tensors, C/P1по212ключей.
- [Cold mixed fixture](../checks/final-inference-v9-pipeline.json):12stochastic samples и stage2 P1+W, QK/разная ширина/strict SHA/формат/reexecution passed. [Регрессии v8](../checks/final-inference-v9-regressions.json):offline/реальные веса/greedy/static/pipeline и12 прежних сценариев passed; frozen receipts не перезаписывались.
- [Статика](../checks/final-inference-v9-static.json):nbformat/compile/SHA/private/offline/version-pinned mounts/GPU0/guards passed; финальная проверка исходников/JSON также прошла. Torch2.11.0 CPU. V8 JSONL остался побайтно неизменным.

### Измерение T4 и ответ

Исполненный notebook и исходные отчёты: ignored `results/kaggle-runs/nlp-hw02-ensemble-inference-v9/interactive/`. Пакет поступил от внешнего оркестратора; сам интерактивный запуск этим агентом не создавался. Code cells exact, execution counts1/2, один cold test и отдельный свежий tune child, `GPU_runs=2`, training false, одна видимая GPU. Неизменённый worker проверяет T4/CUDA и `CUDA_VISIBLE_DEVICES=0`. Отдельная session metadata/ID в полученном пакете отсутствует; private/offline T4 конфигурация notebook независимо проверена через API для batch-версий.

| Полный pipeline | Время, с | Early stops | Mean stochastic samples | Peak allocated /reserved, GiB |
|---|---:|---:|---:|---:|
|Cold test1919|**93,896795**|1171|6,508077|**12,9757 /14,3145**|
|Tune2048, свежий child|**79,100761**|1248|6,515625|**13,9085 /14,2520**|

Общая session179,443674с, все шесть gates passed. Timer какv8: от запуска дочернего interpreter до закрытия answers, включая imports/load/warmup/generation/restore/execution/vote/независимое переисполнение/write. Подготовка Kaggle VM и mounts вне этого timer. Test быстрее цели140 на46,103с; generation guard160 не сработал. Guard/fallback/partial0 на обоих splits, все голоса получены. Переисполнение выбранных программ в GPU-пайплайне:1919test и2048tune, mismatches0.

Device total14,5625GiB; peak gates passed. Test: один OOM W-B5, stage1/offset1024/question_batch1024/K1; прежний механизм уменьшил batch с восстановлением RNG. Tune OOM0. Это не ответный fallback и не изменение K.

GPU tune **1532/2048=74,8046875%**, на15ответов/0,732422п.п. ниже сохранённого offline1547. Уv8 GPU тоже1532, но samples отличаются от прежних K8 batch32 и отv8 из-за моделей/форм/compaction/RNG consumption. По новому tune политику не меняли; согласованное сравнение исходных журналов остаётся1547−1541=6, CI включает0.

> [!info]- Граница независимой runtime-проверки
> Полный raw tune answers/candidate-journal интерактивной session не передан. Поэтому дополнительный replay новых GPU candidates не выполнен (`runtime_replay_verified:false`). Accuracy взята из исполненного notebook/summary; проверены число/уникальность1532match indices, n2048, арифметика accuracy, tune input/question-order SHA и исходный код вычисления метрики. Встроенное независимое переисполнение2048tune winners завершилось без mismatches. Это ограничение относится к дополнительной проверке нового GPU sampling; требуемый CPU replay сохранённых B5-журналов1547и2048/2048winners полностью пройден.

**Ответ:** [final-ensemble-v9.jsonl](../submissions/final-ensemble-v9.jsonl), **1919строк**, в каждой только строковый `code`. Порядок подтверждён исходным test SHA/question-order SHA. Файл побайтно совпадает с внешним `final-ensemble-v9-interactive.jsonl` и `answers_sha256` GPU-отчёта: `51fe169f8e9801a2962c62a3ff1569f25521c709b0f95f4232958a420fbb3179`. [Валидатор](../kaggle/final_v9_output_check.py), режим `--interactive`.

С [v8](../submissions/final-ensemble-v8.jsonl) отличаются **1166строк code** и **231исполненный численный ответ**. Обе посылки независимо исполнены заново прежним ограниченным production executor:3838строк/3073уникальные программы, все numeric, завершены полностью. Ответы сравнивались черезisclose(rel_tol=abs_tol=1e−6); row IDs сохранены в [квитанции](../checks/final-inference-v9-comparison.json). V8 JSONL не изменялся. Чужой интерактивный файл и B6-правки не изменялись.

> [!info]- API-попытки и квота
> Собственный **единственный push**: friend1/v1,19.10 07:52:43UTC. До него target отсутствовал в полном owned inventory34notebooks, активной версии не было. Backend timeout720с, watchdog660/690. Агент больше не push-ил и ничего не отменял. V2–v5 появились вне агента; отдельно проверены всеv1–v5: private/offline T4, code cells exact, CANCEL_ACKNOWLEDGED, failure message отсутствует, output0. [Launch](../checks/final-inference-v9-launch.json), [version audit](../checks/final-inference-v9-remote-versions.json), [status history](../checks/final-inference-v9-status.json), [наблюдения/квота](../checks/final-inference-v9-observations.json). НаблюдаемогоQUEUED>15мин не было, самовольных перезапусков не было. API last_run_time содержит дату16.10 при локальной19.10; хронология использует локальные UTC-квитанции.
> После ранних отмен GPUused возвращался к baseline39202,861с. Стабильный снимок08:50UTC —39751,655с: **548,794с≈9,15мин<40мин** прироста account-level GPUused, включая интерактивное исполнение/подготовку. Это отдельный счётчик, не cold test timer. В бота ничего не отправлялось, на Kaggle ничего не удалялось, все созданные notebook private.

Две попытки коммита — CPU-набора и финального набора с JSONL — остановились на `git add` с кодом128: песочница запрещает создание `.git/worktrees/study1/index.lock` (`Operation not permitted`). Staged index оба раза был пустым и не изменился; commit не создан. [Квитанция/финальный список для повтора](../checks/final-inference-v9-git.json). Чужие `kaggle/b6_prepare.py` и `final-ensemble-v9-interactive.jsonl` в список не включались. `progress.md`, `index.md`, dashboard и backlog этим агентом не редактировались.

### Источники v9

- [b5-results](b5-results.md), [V8: C-long2x / P1-B / W plain B4](#v8-c-long2x--p1-b--w-plain-b4), bibliography — nlp-hw02-ensemble-inference-v9. CPU replay/CI, timing-proxy и фактические GPU-данные различаются явно; новых внешних научных источников нет.

## V10: C-long2x B6 / P1-B / W B6

Состав и политика **заданы оркестратором**, нового поиска нет: **greedy C-long2x B6; K1=1/1/1 → K2=1/5/6**, порядок C/P1-B/W, T0,7, seed270042, reviewer off. B6 — длинный q0 cooldown $1-\sqrt{j/N}$ от uncooled B5; [b6-plan](b6-plan.md), [b6-results](b6-results.md). V8 public78,54% и v9 78,44% — сообщения пользователя; свежий GPU-tune обеих1532/2048 — основной comparator для нового запуска, а не offline1541.

**Offline1553/2048=75,8301% против v8 1541:** +12ответов, Δ+0,58594п.п., заданный описательный CI[−0,34;+1,51]п.п. Ранее найденный максимум сетки: оценка выигрыша смещена вверх, CI не скорректирован за выбор. Осторожный proxy136,8с — прогноз, не timer v10. Новый sampling не обязан воспроизводить1553.

| Модель / pinned private kernel output | FP16-веса | SHA256 |
|---|---|---|
|C-long2x B6 / `friend1/nlp-hw02-b6/1`|`model-GPU1/weights/b6-final-fp16.pt`|`ee30794048bb6200cf97ac4d7e5c1606f0e97893ea04f6567c9487f6c04345fa`|
|P1-B / `friend1/nlp-hw02-kaggle-p1/3`|`model-pilot-b/weights/long-truncation-fp16.pt`|`8439037720dda16dda6b73fe676860d5498bbcba4cacebc0d709bbb22db464a5`|
|W B6 / `friend1/nlp-hw02-b6/1`|`model-GPU0/weights/b6-final-fp16.pt`|`cb34c81e6cda83c73bf4662c5c425446be8cdb48c30dd5e9e217fd6cbc1cd496`|

Обе B6 — точки `model-GPU*/points/b6-final`, их `checkpoint.json` указывает на веса в `weights/`. B6 manifest скачан с явным version_label=v1 и побайтно совпал с локальным audited snapshot (SHA `dd00395d133cd5c005261bf39b41c9a967dbd36e9523b838ecfa865aae218dfe`). P1-B — тот же payload v8/v9. [Download receipt](../checks/final-inference-v10-download.json); креды читает только SDK, содержимое `~/.kaggle*` не читалось и не выводилось.

[Сборщик](../kaggle/build_final_inference_v10.py), [конфиг](../kaggle/final-v10-config.json), [ноутбук](../kaggle/notebooks/nlp-hw02-ensemble-inference-v10.ipynb), [source identity](../kaggle/final-v10-source-identity.json). **Все20runtime-модулей побайтно совпадают с v8/v9.** Decoder all, QK W, selector/executor/session/mounts/supervisor, guards160/166/170/172, sequence_budget1024, per-model RNG/OOM rollback сохранены. Одновременно resident441438482bytes FP16 weights; GPU0 — единственная видимая T4. `NvidiaTeslaT4` у Kaggle может резервировать пару, вторая не используется.

### CPU gates v10

- [Production replay](../checks/final-inference-v10-offline.json): **1553matches,1266early stops;2048/2048winner code/execution exact** с B.Replay и отдельным union-find oracle. SHA128B6-журналов сверены с producer manifest, P1/C control journals с прежними manifests; IDs/question SHA/references/K8 проверены. Winner SHA `5441b9d451bfcdead3d4757e002dfaff8a950fd34e0294cf8742b6d1020a36e3`. Подбора политики нет.
- [State dict](../checks/final-inference-v10-weights.json): SHA, model-only/optimizer false, FP16 finite, strict keys/shapes/values после FP16→FP32. C/P1по212ключей, W284/72QK norm tensors.
- [Cold mixed fixture](../checks/final-inference-v10-pipeline.json): настоящий embedded supervisor/pipeline, разная ширина/QK, strict SHA load,12samples и stage2P1+W, code-only JSONL и reexecution passed.
- [Регрессии](../checks/final-inference-v10-regressions.json): v8 offline/real weights/greedy/static/pipeline и все исторические v7 fixtures/guard/OOM/priority/reexecution passed, frozen receipts не перезаписаны. [Статика](../checks/final-inference-v10-static.json): nbformat/compile/parity, private/offline/GPU0/version-pinned kernel_sources и timeout720 passed. CPU Torch2.11.0.

### Интерактивный T4-прогон и public v10

**V10 закрыт. По сообщению пользователя: public78,96%**, против v8 78,54% (Δ+0,42п.п.) и v9 78,44% (Δ+0,52п.п.). Это факт со слов пользователя, без независимого leaderboard-аудита. Свежий tune **1530/2048=74,7070%**, на2ответа ниже GPU v8/v9 и на23ниже offline1553 (−1,1230п.п.). `status=review_required`: единственный непройденный gate — `quality_within_1pp`; остальные пять gates true. По tune политику не меняли.

Оркестратор передал исполненный notebook и три test-файла в ignored `results/kaggle-runs/nlp-hw02-ensemble-inference-v10/interactive/`. [Проверка](../kaggle/final_interactive_check.py), [итог](../results/final-inference-v10.json), [сравнение](../checks/final-inference-v10-comparison.json). **Обе code cells exact с notebook и бандломv10**, execution counts1/2, все20embedded source SHA совпадают. Реальные SHA весов и порядок C/P1-B/W подтверждены test-отчётом. Private/offline/GPU0 конфигурация ранее проверена API; отдельные interactive session metadata в пакете отсутствуют.

|Измерение|Результат|
|---|---|
|Cold test1919, полный pipeline|**115,631357с**|
|До generation guard160 / цели140|44,369 /24,369с|
|Early stops / mean stochastic samples|1184 /6,447108|
|Peak allocated /reserved|12,9871 /14,2246GiB|
|Session test+tune|206,519127с|
|Guard /fallback /partial /reexecution mismatch|0 /0 /0 /0|

Timer включает запуск interpreter/imports/load/warmup/decode/restore/execution/vote/независимое переисполнение/write; подготовка VM/mounts вне него. Test reexecution1919/1919. Два OOM W-B6: stage1 batch1024 при offset1024 и stage2 batch204/K5 при offset0; прежний retry с откатом RNG уменьшил batch, ответного fallback нет. Точное время tune и его отдельные memory/journal/answers в полученном пакете отсутствуют:1530 взято из вывода исполненного notebook; count/match indices/accuracy согласованы. Независимый runtime replay tune не заявляется.

**Ответ:** [final-ensemble-v10.jsonl](../submissions/final-ensemble-v10.jsonl),1919строк, только строковый `code`, порядок input/question SHA подтверждён. SHA256 `0628359bec615a905693a45d37ef359dd192c9ef97cc323793905c8886d6361c`. Отv8 отличаются **1176строк code и255исполненных численных ответов**; оба JSONL независимо переисполнены bounded production executor, сравнение `isclose(rel_tol=abs_tol=1e-6)`, нечисловых результатов0. В программном бандле относительноv8 изменены веса C/W и политика1/8/1→1/5/6; runtime-модули совпадают побайтно. Новый sampling не обязан совпадать со старыми журналами.

> [!info]- История batch-запуска
> Единственный собственный push19.10 14:36:08UTC: private [v10/v1](https://www.kaggle.com/code/friend1/nlp-hw02-ensemble-inference-v10/versions?versionNumber=1), timeout720/watchdog660/690. V1 и внешняяv2 дошли до CANCEL_ACKNOWLEDGED, output за60мин не появился. Обе code parity exact. [Launch](../checks/final-inference-v10-launch.json), [remote versions](../checks/final-inference-v10-remote-versions.json), [последняя batch-download receipt](../checks/final-inference-v10-output-download.json). Позднейший interactive output выше имеет отдельное происхождение. Повторного push/отмены/удаления агентом не было.

Прежняя попытка коммитаv10 блокировалась запретом `index.lock`; [квитанция](../checks/final-inference-v10-git.json). Итоговая попытка v10/v11 записана в разделеv11. В бота ничего не отправлено.

### Источники v10

- [b6-plan](b6-plan.md), [b6-results](b6-results.md), [V8: C-long2x / P1-B / W plain B4](#v8-c-long2x--p1-b--w-plain-b4), [V9: C-long2x B5 / P1-B / W B5](#v9-c-long2x-b5--p1-b--w-b5), bibliography — nlp-hw02-ensemble-inference-v10. Собственные CPU receipts и последующий private GPU run; новых внешних научных источников нет. Offline/CI/proxy и свежие GPU-измерения различаются явно.


## V11: та же v10, K2=1/8/8

**Лидерборд (сообщение пользователя):** v11 — **79,37% public, первое место** (v10 — 78,96%, v8 — 78,54%). Порядок совпал со свежим GPU-tune: v11 1547 против v10 1530. Текущий финальный кандидат — v11. Пиковая память GPU 15,30 из 15,64 ГБ — для финального ноутбука уменьшить батч ради запаса.

**Единственное изменение инференса:** K1=1/1/1 → **K2=1/8/8**, greedy C-long2x B6, порядок C/P1-B/W B6, T0,7/seed270042. При численном единогласии greedy+stage1 остановиться, иначе добавить7samples P1-B и7W; максимум17stochastic+greedy. Веса/SHA, decoder all, голосование/count/greedy/meanLP/order/выбор кода, per-model RNG/OOM retry, resident models, sequence_budget1024, guards160/166/170/172 и watchdog660/690 **сохранены изv10**. Все20embedded modules exact. Diagnostic `expected_matches=1553` также сохранён: это прежний reference v10 для quality gate, а не ожидаемый score v11; на решения он не влияет.

**Основание:** [E1 frozen primary budget-1-8-8](e1-ensembles.md#результаты-e1), выбрана на tune **1550/2048** до открытия eval. На том же свежем E1-пуле eval **1574/2048 против1559** у политикиv10,29gains/14losses, Δ+0,7324п.п., paired95%CI **[+0,0977;+1,3672]п.п.** Это один sampling pool; CI не оценивает разброс повторной генерации. Frozen прогноз cold test1919 **137,322с**, Результат отдельного coldT4 приведён ниже. [Selection](../results/e1-selection.json), [confirmation](../results/e1-confirm.json), [политика](../results/e1-v11-policy.json). E1-файлы этим агентом не изменялись.

[Сборщик](../kaggle/build_final_inference_v11.py), [ноутбук](../kaggle/notebooks/nlp-hw02-ensemble-inference-v11.ipynb), [конфиг](../kaggle/final-v11-config.json), [source identity](../kaggle/final-v11-source-identity.json). Builder читает frozen notebookv10; текущие src не подменяют его модули. Run All запускает один cold test1919, затем отдельный свежий child tune2048, без API/credentials/обучения внутри notebook.

### CPU gates v11

- [K8 replay](../checks/final-inference-v11-offline.json): **1551/2048**,1266early stops, winner code/execution2048/2048 exact с B.Replay и независимым oracle. Старые K8-журналы B6/P1 проверены по producer SHA/IDs/question SHA/references; они отличаются от E1-пула. Эти1551 не являются свежим GPU-score и не отменяют подтверждениеE1.
- [Свежий E1 replay](../checks/final-inference-v11-e1.json): production selector на первых8samples каждого C/P/W совпал с frozen E1 policy во всех4096строках: tune1550/eval1574, winner code/execution/early-stop/count exact. E1 прочитан без изменений.
- [Веса](../checks/final-inference-v11-weights.json): прежние pinned SHA, FP16 finite/model-only, strict load/keys/shapes/values; resident441438482bytes.
- [Cold mixed CPU pipeline](../checks/final-inference-v11-pipeline.json): разная ширина/QK,17samples, stage2P1+W/K7, bounded execution/reexecution и code-only JSONL; passed.
- [Статика](../checks/final-inference-v11-static.json): только две config-разницы, P1-B K2 5→8 и W K2 6→8; все20runtime sources exactv10, nbformat/compile, private/offline/T4/GPU0, pinned mounts и720с проверены. [Регрессииv10](../checks/final-inference-v11-regressions.json) выполняются в памяти; passed, frozen receipts не перезаписаны.

### Запуск и получение output v11

Private [friend1/v11](https://www.kaggle.com/code/friend1/nlp-hw02-ensemble-inference-v11). Один разрешённый push: `final_v11_api.py push`, timeout720, `NvidiaTeslaT4`, используется толькоGPU0. SDK/CLI получает `KAGGLE_CONFIG_DIR=~/.kaggle-friend`; содержимое конфигов не читается и не выводится агентом. [Launch](../checks/final-inference-v11-launch.json), [remote parity](../checks/final-inference-v11-remote.json), [observations](../checks/final-inference-v11-observations.json). Пороги QUEUED>10мин или RUNNING>10мин без output: зафиксировать статус, продолжать наблюдение до45мин от push, без повторного push/отмены/удаления. Внешние версии отслеживаются только после exact code parity.

Для интерактивного запуска открыть notebook и **Run All**. Из `/kaggle/working/ensemble-inference-v11/` скачать:

- `test-1/answers.jsonl`;
- `test-1/timing.json`;
- `test-1/summary.json`;
- `summary.json`.

Сохранить также исполненный `.ipynb` для exact code parity. Для независимого replay tune полезны `tune/answers.jsonl`, `tune/candidate-journal.json.gz`, `tune/summary.json`, `tune/quality.json`. Оркестратор кладёт пакет в ignored `results/kaggle-runs/nlp-hw02-ensemble-inference-v11/interactive/`; допустимы исходные подпапки либо имена `test-1-answers.jsonl`, `test-1-timing.json`, `test-1-summary.json`. Проверка: `python kaggle/final_interactive_check.py 11 <packet-dir>`. `summary.json` содержит свежий tune/gates; test-1 всегда единственный источник submission.

### Интерактивный T4-прогон v11 — passed

Оркестратор предоставил полный session `summary.json`, test answers/timing/summary и исполненный notebook в ignored `results/kaggle-runs/nlp-hw02-ensemble-inference-v11/interactive/`. **Обе code cells exact с локальным notebook и push-бандлом; все20embedded modules/SHA exactv10, execution counts1/2.** Runtime model SHA, порядок C/P1-B/W, K1/K2, seed, input/question-order SHA, строгий code-only формат и переисполнение подтверждены. Отдельные interactive session metadata не предоставлены; private/offline/T4 конфигурация проверена сервером для опубликованного кода. [Итог](../results/final-inference-v11.json), [output audit](../checks/final-inference-v11-output.json).

|Полный pipeline|Время, с|Early stops|Mean stochastic samples|Peak allocated /reserved, GiB|
|---|---:|---:|---:|---:|
|Cold test1919|**126,846266**|1193|8,296509|12,9871 /14,2578|
|Свежий tune2048|**110,177838**|1288|8,195313|13,5688 /14,2520|

Session243,360346с, **все6gates passed**. Полный timer включает interpreter/imports/load/warmup/decode/restore/execution/vote/независимое переисполнение/write. Cold test быстрее цели140 на13,154с и guard160 на33,154с; forecast137,322с был консервативнее на10,476с. Guard/fallback/partial/reexecution mismatch —0на обоих splits, все запланированные голоса исполнены. Test переисполнены1919winners, tune2048. Два OOM W наtest: stage1 batch1024/offset1024 и stage2 batch146/K7/offset0; один OOM W наtune вstage1. Прежний RNG rollback/batch reduction сработал, K не менялся; повторного полного test не было.

Свежий tune **1547/2048=75,5371% против v10 1530/2048=74,7070%**: **+17ответов /+0,8301п.п.**,44gains/27losses; описательный paired bootstrap95%CI **[+0,0488;+1,6602]п.п.** (20000, seed2026101904). Это одна генерация каждой версии на общих задачах; CI не покрывает вариативность повторного sampling. Политика не менялась после результата. Raw tune answers/candidate journal не переданы: score/match indices/accuracy и tune-order проверены по полному session report; независимый runtime replay tune не заявляется.

**Ответ:** [final-ensemble-v11.jsonl](../submissions/final-ensemble-v11.jsonl), **1919строк, только строковый code**, исходный порядок подтверждён. SHA256 `d9091efb8a72b18f01f75c1e9ab5acb5f45ffa94b09dcd36fc437b9b09d7cd59`. [Сравнениеv10](../checks/final-inference-v11-comparison.json): **290строк code /187численных ответов отличаются**, оба JSONL независимо исполнены bounded production executor, `isclose(rel_tol=abs_tol=1e-6)`, нечисловых результатов0. Cold test дорожеv10 на11,214910с. Public v10 **78,96%**; **public v11 ещё не измерен**. В бота ничего не отправлено.

> [!info]- История единственного batch push
> 19.10.2026 17:15:17UTC — private [версия1](https://www.kaggle.com/code/friend1/nlp-hw02-ensemble-inference-v11/versions?versionNumber=1), timeout720, T4/GPU0, exact server code parity. В17:25:41UTC QUEUED превысил10мин без output; порог записан в ledger. В17:27:58UTC API сменил статус на CANCEL_ACKNOWLEDGED; отмену агент не выполнял. Последняя batch-версия1 оставалась exact-code/без output, затем≈17:36UTC получен внешний interactive пакет выше, монитор завершён. Собственный push ровно1, повторных push/отмен/удалений0. Интерактивный прогон не приписывается отменённой batch-сессии.

Коммит v10/v11: `git add` собственного списка завершился128, `index.lock: Operation not permitted`; индекс пуст. [Квитанция/точный список](../checks/final-inference-v11-git.json). Чужие E1-файлы и общая библиография не включались.

### Источники v11

- [e1-ensembles](e1-ensembles.md), [V10: C-long2x B6 / P1-B / W B6](#v10-c-long2x-b6--p1-b--w-b6), bibliography — nlp-hw02-e1-ensembles, bibliography — nlp-hw02-ensemble-inference-v10. Собственная фиксированная политика и CPU/GPU receipts; новых внешних источников нет.
