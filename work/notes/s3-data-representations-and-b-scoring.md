---
type: experiment
course:
  - nlp
status: draft
understanding: 3
created: 2026-10-12
updated: 2026-10-12
sources:
  - kaggle-s1-v1
  - kaggle-s2-v2
aliases: []
---

# ДЗ2: S3 — представление данных и B как оценщик

Протокол зафиксирован **до push**. Пользователь одобрил одну private-сессию T4×2, `thisarivson/nlp-hw02-kaggle-s3`, internet on. Бюджет от первой ячейки — 5400с; workers останавливаются в5100с, аварийный watchdog выходит в6000с, серверный потолок6300с. Перепушей после ERROR нет. Live upload не используется. [Реестр версий](../kaggle/runs.json).

## Контроль и единственный фактор

Бесплатный контроль — сохранённые №26b A p006250 / p012500 из `nlp-hw02-26-resume`, те же4096/1024 вопросов и исходный порядок. Primary:1888/4096 и2210/4096; secondary465/991 и537/991. Чистая compute-история:928.116608451с и2018.225873704с. Новая генерация control не нужна; проверяются alignment и сохранённые execution/match sets.

Обе A:4+4/d256/h4/SwiGLU680, dropout0, LR0.003, batch128, AMP FP16/scaler1024, fused AdamW/wd0.01, clip1.0. Исходные initial tensors загружаются и token embedding делится на√256; все старые embedding/head/прочие tensors сохраняются точно. Новые special embedding — независимая нормированная init с seed42; новые head rows — стандартная init. Seed42, кеш0–41 и **готовый исходный order.npy**, без пересортировки под новые длины. Warmup300, затемstable; cooldown от6250/12500 на1250/2500updates линейно до0.0003. После первой ветки восстанавливаются model/AdamW/scaler/RNG/cursor; стоимость ветки остаётся в compute-истории. Предъявления точек960000/1920000; вся compute-история ко второй точке2080000.

## №30, GPU0: компактный обратимый таргет

Гипотеза: исключение восстановимого текста уменьшит target length и улучшит качество при тех же предъявлениях/времени. Вариантc из [data-research-ceiling](data-research-ceiling.md): комментарии удаляются токенизаторомPython, пустые физические строки — сAST guard, строки внутри multiline strings сохраняются. Убирается только точный header `def solve() -> int:`. При стандартном tail снимаются `result = ` и `return result`, выражение остаётся. Один special задаёт `header_tail`/`header`/`raw`; восстановление детерминировано. AST equality после восстановления защищает необычные многострочные хвосты. Нестандартный каркас используетraw fallback. BPE8k№22 расширяется3specials, без переобучения BPE.

После двух фиксированных точек — дополнительные **независимые** ветки от сохранённых stable anchors для времени control. Stable продолжение и cooldown20% оцениваются по measured step-time. Эти ветки не входят в историю фиксированных точек. Tolerance: max(5с,2%target), как в S2; actual time и exposures сохраняются. Misaligned/missing явно означают pending; качество по времени не интерполируется. Если фиксированная точка сама совпала по времени, она переиспользуется. Если она медленнее control, временная точка missing. На дополнительные ветки действует общий бюджет; они могут остатьсяpartial.

## №31, GPU1: числа на исходном таргете

Гипотеза: явная привязка литералов к цифровым значениям вопроса уменьшит ошибки копирования. Каждое уникальное значение получает `<|N1|>`…`<|N20|>` по первому появлению; метка стоит **перед сохранённой исходной величиной**. Повторное значение получает ту же метку; новые значения после20 остаются без метки. Поддерживаются знак/decimal/comma-group/fraction/exponent. Словесные числа не добавляются.

В исходном коде заменяются толькоAST numeric literal spans, равные помеченным значениям; комментарии/header/tail/непомеченные константы остаются. Unary знак учитывается целиком, strings/comments не заменяются. При несовпадении типов `int`/`float` добавляется детерминированныйcast вокруг метки, чтобы `range(5)` не превратился в `range(5.0)`, а тип исходного5.0 сохранился. Подстановка обрамляет значение скобками для сохранения приоритетов; неизвестная сгенерированная метка считается ошибкой восстановления. BPE№22 +20specials. Сжатия таргета в этом опыте нет.

## №32, GPU0 после №30: B teacher forcing

Гипотеза: вероятности независимо обученной B дают информацию для выбора среди A-кандидатов сверх уверенности A. Обучения нет. Подключён **S2/2**, B оценённые FP32model-only t04 (7312.883812с target), SHA`cbbe292c…f720a`; stable latest не используется. Подключён **S1/1**, `gpu0-exp27/{heldout4k,old1024}/candidates`:greedy +32T0.7 +16T0.5 +16T1.0, все65 на каждом из4096+1024 вопросов.

Score — untempered logpB по канонической BPE-токенизации сохранённого decoded code, сEOS, безPAD. FP16forward, FP32cross entropy иsum/mean; для cap-stopped code добавляетсяEOS, это явно записанный scoring convention. Exact-code дедупликация внутри вопроса сокращаетforward, но scores записываются для **всех** исходных индексов. JSON: questionid, candidateindex0=greedy/1–64=samples, исходная temperature/sampleindex, codeSHA, sum/mean, lengthwithEOS. CandidateSHA иalignment проверяются. Сохранение каждого завершённогоbatch атомарное. Tune/eval и ансамбль A/B — последующийCPU-анализ по [candidate-selection-cpu](candidate-selection-cpu.md), здесь выбора поgold нет.

## Оценка и правило решения

На обеих фиксированных точках: greedy после восстановления, exec-match heldout4096 (primary) иold991 (secondary), CE FP32/b32 **в собственном представлении, несопоставима с26b**, bins по длинам **исходного**reference/tokenizer22, McNemar против того же26bpoint. Execution:wall3с/CPU2с,2workers сITIMER_PROF; reference receipts переиспользуются без нового фильтра. Generated text/восстановленныйcode/restoreerrors сохраняются. Свежийtest/fresh не оценивается.

№30/31 принимаются, если на1.92M heldout≥control+1.0п.п. иold≥control−1.0п.п.; №30 требует также≥+1.0п.п. при равном времени. Иначе — нейтрально или отказ. Missing/misaligned → pending, не успех. Порог primary:≥2251/4096, secondary:≥528/991. Одинseed и повторное использованиеheldout ограничивают вывод; это неofficialscore.

Чистое train-время иpairs/s считаются по той же `train_chunk`№26b, включаяcollate/H2D/backward/update/sync; preprocessing/transformation/cacheI/O preparation, eval иcheckpointI/O исключены. Преобразованныеprefixarrays, anchors и распаковка находятся внеworking. Output: толькоlatestполныйstate каждойA, FP32model-only фиксированныеpoints, compactJSONmetrics/generations иBscorebatches; ZIP не создаётся. Partial summaries пишутся до обучения, послеtrainpoint и каждогоevalsplit/scorebatch. Watchdog вооружён доinstall/preparation.

## CPU gate и ограничения

[Round-trip](../checks/kaggle-s3-roundtrip.json): **№30 0/57702; №31 0/57702** execution расхождений после BPE encode/decode и восстановления. Та же stratifiedtrainвыборка сseed20261012; original ответы — прежний независимо проверенныйexecution, все новые restored программы исполнены заново. Это не доказательство для всех11.8M строк. [Tiny smoke](../checks/kaggle-s3-smoke.json): результаты записываются доpush.

Не проверены доGPU: T4peakmemory/фактическийSDPAbackend, новыеCUDAметрики/численнаяидентичность control междуruntime, стоимостьpreprocessing/scoring и полнотаS3 внутрибюджета. ЕслиBscore не успеет, завершённыеbatches сохранятся, пропуски будут явными. Полный production inference≤180с иCPUанализ ансамбля не входят вS3.

## Источники

- [data-research-ceiling](data-research-ceiling.md) — собственный CPU-аудит и вариантc.
- [candidate-selection-cpu](candidate-selection-cpu.md) — гипотеза независимогоscoreB и frozenCPUsplit.
- [26b-final-wsd-data-scaling](26b-final-wsd-data-scaling.md) — recipe/controltime/points, `kaggle-s1-v1`.
- [29m-28b-capacity-s2-results](29m-28b-capacity-s2-results.md) — оценённыеBweights, `kaggle-s2-v2`.
