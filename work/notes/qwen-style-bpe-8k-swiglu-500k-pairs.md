---
type: experiment
course:
  - nlp
status: learning
understanding: 1
created: 2026-10-09
updated: 2026-10-09
sources:
  - https://raw.githubusercontent.com/huggingface/transformers/v4.43.1/src/transformers/models/qwen2/tokenization_qwen2.py
  - https://huggingface.co/docs/tokenizers/v0.23.2/api/pre-tokenizers
  - https://huggingface.co/docs/tokenizers/v0.23.2/api/trainers
  - https://arxiv.org/html/2402.14905v2
  - https://huggingface.co/Salesforce/codet5-small/blob/main/config.json
  - https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/tree/b2724689ff079e9c22efd548fe134bfb29036dbb/data/train
aliases:
  - Опыт 19 — Qwen-style BPE8k на 500k парах
---
# Опыт 19: свой BPE8k по правилам Qwen на 500k парах

[Notebook для Colab](../notebooks/19-qwen-style-bpe-8k-swiglu-500k-pairs-100000-steps.ipynb). Получен и разобран100k T4-прогон: выигрыша по правильности пока нет. Рабочий контроль — [18/100k](swiglu-500k-continue-to-100k.md), а не случайная модель CPU-проверки.

## Гипотеза и выбранный фактор

Отдельные цифры до BPE могут облегчить перенос чисел между вопросом и программой:1500→1|5|0|0, пробел перед числом не меняет digit IDs. Нечисловой текст продолжает сжиматься BPE. Это гипотеза о представлении; неверную операцию, единицу или число tokenizer автоматически не исправляет.

Меняем **tokenizer как компонент**: regex Qwen2, словарь4096→8192, корпус обучения BPE20k→500k. Пользователь выбрал8k и затем предложил больше пар для BPE. Вклад правил, размера, корпуса и добавочных embedding/head параметров отдельно этот опыт не установит. Рецепт адаптирован: vocabulary/merges свои, normalizer=None, текст сохраняется байт в байт; качество готового Qwen сюда не переносим.

## Протокол

| Параметр | Значение |
| --- | --- |
| Корпус BPE | Все прежние500k разрешённых train-пар, вопрос+код,1M текстов; heldout/test исключены |
| BPE | Tokenizers0.23.2,8189 content IDs+PAD0/BOS1/EOS2, byte-alphabet256, min_frequency2 |
| Pretokenization | Qwen2 regex → ByteLevel(use_regex=False/add_prefix_space=False), без нормализации |
| LM train | Те же500k пар/485373 вопроса, pinned shards0–2, прежние byte-фильтры и split/hash |
| Модель | Encoder2/decoder2, d256,4 головы, Pre-LN с final LayerNorm, SwiGLU680, позиции2048 |
| Параметры | 8 407 360 против6 306 112 у18; shared source/target embedding, отдельный head |
| Обучение | С нуля, seed42, FP32/cache/batch8,100k updates/800k предъявлений; AdamW wd.01/dropout.1/clip1 |
| LR | Warmup1000 до.001, cosine до.0001 на50k, затем floor до100k; как цепочка17→18 |
| Контроли | Ранние train64/validation64;50k/75k/100k — прежние heldout1024/888; fresh1024 только резерв |
| Инференс | Прежний KV-кеш/greedy/batch32/cap512; полный1919 test с таймером≤180с |

Старые weights18 не загружаются: размеры embedding/head изменились. Recipe и sampling-index budget прежние; одинаковую инициализацию остальных весов между4k/8k не обещаем — больший vocab меняет потребление RNG. Равные updates/пары не означают равные tokens/FLOPs/time.

## Хватит ли20k для BPE

Сейчас отдельно проверено: прежние20k содержат7 983 574 UTF8-байта и дают ровно8189 content-токенов, то есть запрошенный model vocab8192. [Проверка](../checks/19-qwen-bpe8k-20k-vocab-capacity-check.json). Это техническая достаточность, не доказательство достаточного покрытия или качества LM.

В notebook19 BPE обучается на всех500k:198 974 144 UTF8-байта. Мотивация — более представительные оценки частот и редких конструкций; преимущество по правильности программ нужно измерить. На локальном CPU fitting занял99.56с против4.02с для20k; это новая локальная проверка, не время Colab/T4.

## Проверки при подготовке notebook

[Статика](../checks/19-qwen-bpe8k-static-checks.json), [CPU-проверки](../checks/19-qwen-bpe8k-cpu-checks.json).21 ячейка/11 кодовых, outputs очищены. Hashes данных/heldout совпали с18; пересечений train с old/fresh heldout/test нет. Обучен новый BPE на500k, весь train и старый heldout прошли byte round-trip, диапазоны IDs/лимиты позиций;9 digit probes и17 online/cache batch-проверок прошли.

| Длины train, content tokens | Сохранённый BPE4k18 | Новый BPE8k19, проверено сейчас |
| --- | ---: | ---: |
| Question mean / p95 / max |44.370 /81 /233 |46.375 /81 /232 |
| Code mean / p95 / max |64.813 /121 /294 |64.977 /113 /291 |

Среднее число токенов не сократилось: регулярное представление цифр имеет цену. Tokenizer SHA256: `261f25b5d4be006a6a00e41e0eafbe522580741da627ba670f6b9805a2659b7d`; фиксировать фактический hash каждого запуска, tokenizer сохраняется в ZIP/checkpoints.

Полная модель прошла11 CPU-updates суммарно:1 при контролируемом прерывании,4 в notebook-smoke,4 online-контрольных,2 из checkpoint. Проверены finite loss/gradients, точные weights/AdamW/LR/RNG/coverage online↔cache и resume2→4, full-prefix↔KV logits/greedy, early64/full1024, weighted byte-NLL/парные сравнения, rerun guard и частичный ZIP. В финальной ячейке исправлена старая ссылка на comparison15; затем затронутые финальные ячейки проверены с checkpoint4 без новых updates: validation64/byte512,2 test-строки и полный ZIP/CRC/manifest/SHA. Все остальные ячейки между фазами идентичны. Подробности обеих фаз в отчёте.

> [!note]- Границы проверки
> При подготовке model smoke использовал cap4 и искусственный control-step4, привязанный к сохранённому50k только для проверки plumbing. Это не сравнение качества при равном бюджете. Длительное100k LM-обучение, CUDA/T4, полный1919 test и браузерная загрузка Colab на этапе подготовки не выполнялись. Fitting BPE на500k выполнялся реально. Fresh1024 не оценивался.

## Как сравнить результат

Для50k/75k/100k встроены **сохранённые** метрики18: numeric-reference110/127/142 из888, EOS994/1003/1010. При100k старый CE.856009 и test46.64с — результаты пользователя, не новые замеры. Проверяем парные приобретённые/потерянные задачи, EOS/undefined names, программы и расширенный ручной аудит; reference содержат ошибки, proxy не официальный pass@1.

Raw token CE разных tokenizer напрямую не сравнивать. Дополнительная likelihood-диагностика:

$$\ell_{\mathrm{byte}}=\frac{\sum_r C_r T_r}{\sum_r B_r}.$$

$C_r$ — token-weighted teacher-forcing CE в группе$r$, $T_r$ — число gold target-токенов с EOS, $B_r$ — число исходных UTF8-байт gold-кода. Суммируем NLL, затем делим на общий byte count; EOS входит в NLL, но не добавляет байтов. У18 этот показатель восстановлен из сохранённых CE/counts:50k/75k/100k=.276035/.261707/.251605 nats/byte. Это likelihood канонической токенизации, не функциональная accuracy. При объединении групп CE взвешивается по gold-токенам, не по числу строк.

Считать опыт полезным по правильности программ и числовых решений, а byte NLL использовать дополнительно. Один seed/многократно использованный heldout не доказывают устойчивый выигрыш; близких кандидатов повторить, fresh1024 оставить финальному сравнению.

## Сохранённый T4-результат и новый анализ

[Пользовательский IPYNB с outputs](../results/19-qwen-style-bpe-8k-swiglu-500k-pairs-100000-steps-results.ipynb), [сводка/manifest/веса](../results/19-qwen-style-bpe-8k-swiglu-500k-pairs-100000-steps-summary.json), [статика](../checks/19-qwen-bpe8k-results-static-checks.json), [парный/ручной аудит](../checks/19-qwen-bpe8k-functional-quality-audit.json), [новый CPU-forward](../checks/19-qwen-bpe8k-conditioning-cpu-checks.json), [работа декодера/бюджет глубины](../checks/19-qwen-bpe8k-inference-work-and-depth-budget.json).

**Выигрыш по правильности решений не подтверждён.** Регулярное представление цифр работает, но136/888 против142/888 у18; прежний ручной23 даёт3 правильных/19 неправильных/1 unsupported против6/16/1. Разницы малы и один seed не доказывает устойчивое ухудшение, но объявлять19 победителем оснований нет.18 сохраняем контрольной базой по смысловому качеству.

| Показатель |18/100k, сохранено ранее |19/100k, сохранено пользователем |
| --- | ---: | ---: |
| Reference numeric agreements /888 |142 (15.99%) |136 (15.32%) |
| NLL на исходный UTF8-байт ↓ |.251605 |.254696 (+1.23%) |
| EOS /1024 |1010 |1016 |
| Undefined name /1024 |71 |54 |
| AST /1024 |1015 |1014 |
| Ручной набор23, верно |6 |3 |
| Полный test1919 |46.635с |40.611с (−12.92% времени) |
| Test AST /1919 |1908 |1890 |
| Чистый train100k |35.20мин, цепочка17→18 |37.30мин (+5.96% времени) |

Это saved T4/PyTorch2.11.0+cu130, не новый GPU benchmark агента. Token CE4k/8k напрямую не сравнивается; byte NLL вычислен по общей мере исходного gold-кода, не является функциональной accuracy. Reference известны ошибками;136/888 не official pass@1.

| Шаг |19 token CE1024 внутри одного tokenizer |19 byte NLL |18 /19 reference agreements из888 |
| --- | ---: | ---: | ---: |
|50k |.955534 |.279718 |110 /113 |
|75k |.905240 |.264996 |127 /127 |
|100k |.870055 |.254696 |142 /136 |

Внутри19 обучение продолжается:50k→100k byte NLL−8.95%, reference113→136. Парные изменения внутри19:43 новых/20 потерянных совпадений,93 сохранённых. Сравнение18/100k→19/100k:89 сохранены,47 приобретены,53 потеряны, итог−6. Это смена решений, а не равномерное улучшение. На fixed888 у19:136 совпадений,660 числовых расхождений,64 определённых сбоя и28 неподдержанных программ; unsupported не автоматически неверны. Синтаксис/EOS/числовая вычислимость отдельно от правильности.

Ручной23 внутри19:4→5→3 правильных на50k/75k/100k. Относительно18 потеряны случаи4/18/26, новых верных в этом наборе нет. Не выбирать75k по маленькому неслучайному набору вместо всего контроля. Пример4: две рубашки по20 со скидкой30% должны дать28; у19 числа20/2/.3 записаны, затем `total_cost = total_cost + total_cost` даёт80. У18 было28. Это ошибка вычислительной структуры при наличии нужных чисел; нехватка глубины этим не доказана.

Teacher forcing19 на validation64:correct CE.847484, подмена вопроса2.273865. Source используется; это не доказательство правильного greedy-решения. Train64 CE.812070/10 numeric-reference agreements из59, heldout1024 .870055/136 из888: простой collapse и сильное запоминание всех train-пар здесь не видны; малого train64 недостаточно для диагноза ёмкости.

### Скорость и бюджет

100k updates/800k предъявлений, увидены399353/500000 строк (79.87%); Python RNG и per-row seen flags на50k/75k/100k совпадают с18. Полезные target tokens52 789 363, padding37.98% против41.54% у18. Tokens и padded slots различаются при прежних индексах пар — tokenizer изменился.

Train2237.72с/37.30мин/44.69updates/s, активный train-цикл с диагностикой/checkpoints2553.47с/42.56мин. Setup280.04с/4.67мин, включая BPE fitting84.04с и cache build137.05с. Это разные области таймеров, их сумма не выдаётся за точный wall time всего notebook. Tokenizer hash совпал с подготовительным CPU fitting.

Полный test40.611с/47.25ответа в секунду, peak allocated278.21MiB; текущий лимит180с пройден с запасом139.39с. Test EOS1891 против1887, cap28 против32, undefined122 против145, numeric1680 у обеих; синтаксических ошибок29 против11, поэтому форма тоже улучшилась не по всем показателям.

Средняя длина test68.62 против68.27 токена почти та же. В коде finished rows остаются до окончания последнего ответа батча. Из60 батчей до512 дошли21 против26; сумма batch-max decoder steps16523 против19130 (−13.63%), row-step work528599 против611648 (−13.58%), proxy накопленной длины attention-префиксов−19.12%. Это согласуется с меньшим таймером, но не profiler и не точное причинное разложение. Размер head вырос; разные T4-сессии не заменяют парный benchmark.

### Что проверено сейчас

21/11 source совпал с подготовленным notebook; counts1–11/no errors. Прочитаны все ZIP members с CRC/manifest/SHA,5887 saved examples с token→byte/UTF8/AST/numeric/агрегатами, полный test-order1919, weighted NLL/reference denominator888/223 и история18. Четыре checkpoints loaded with weights_only=True: finite FP32/AdamW, steps/LR/RNG/data/tokenizer/coverage; latest state==100k. Свежие1024 совпали по hash и остаются неоценёнными.

Новый CPU/PyTorch2.14.1 forward на saved100k воспроизвёл correct/rotated CE64 в пределах1e−5;3 полных greedy-ответа совпали между full-prefix, KV и savedT4. Веса не менялись, optimizer updates0. GPU, длительное обучение и полный1919 inference сейчас не повторялись. Ручные expected answers23 переиспользованы только для оценки, train-таргеты не менялись.

Полные JSON/test/fresh и4 checkpoints/tokenizer сохранены локально в ignored `work/results/generations/` и `work/results/checkpoints/`, опись/hashes в сводке. Для переноса нужны эти ignored файлы или исходный ZIP; Downloads сохранены.

## Следующая гипотеза: глубина decoder

Предлагается отдельный decoder2→4 при прежнем encoder2/d256/BPE8k/train500k/Pre-LN/SwiGLU/FP32/cache/batch8/LR/100k updates. По формам checkpoint один decoder-layer содержит1 051 728 параметров; два дополнительных слоя дают10 510 816 вместо8 407 360. Получен [20: decoder 2→4 на BPE8k](deeper-decoder-4-layers-qwen-bpe-8k.md): CE1024 .7490/byteNLL.21925/reference170/888, ручной74=13 против19/18 12/14, test51.74с. Улучшение loss/proxy подтверждено, смысловой выигрыш пока нет. Следующий кандидат encoder2→4 приdecoder4 (12.086912M), LR/batch отложены.

Мотивация: дополнительные Transformer-преобразования для генерации кода; параметры19 выросли преимущественно в embedding/head, число блоков осталось2+2. В [MobileLLM §2.2.2](https://arxiv.org/html/2402.14905v2#S2.SS2.SSS2) глубина выиграла у ширины в их125M/350M decoder-only протоколе; перенос на наш encoder-decoder8M/математический код — гипотеза. CodeT5-small имеет6 encoder/6 decoder слоёв по [config](https://huggingface.co/Salesforce/codet5-small/blob/main/config.json), но pretrained data/objectives/масштаб отличаются. Две дополнительные decoder-стадии не гарантируют устранения ошибок чисел/операций; encoder-capacity тоже пока не измерена.

BPE8k сохраняется **для этого сравнения**, чтобы проверить предложенную пользователем связь нового представления с глубиной, и сравнивается с19 при тех же updates.18 остаётся контрольной базой качества. Без одновременного изменения encoder/ширины/LR/данных; checkpoint19 не является полной инициализацией новой глубины, нужен запуск с нуля. Контроли50k/75k/100k по888 proxy/парным изменениям/byte NLL, ручной аудит старого heldout расширить примерно до100 с фиксированными критериями; fresh1024 финальному сравнению. Нужен новый полный test-таймер, имеющийся запас не гарантирует скорость глубокой модели. Если выигрыша нет, гипотезу пересмотреть, а не объявлять tokenizer исправленным ёмкостью.

## Протокол выполненного запуска

Новая Colab/T4-сессия → выполнить notebook по порядку; старые ZIP/PT не нужны. Последняя ячейка создаёт `hw02-qwen-bpe8k-swiglu-500k-100000steps-results.zip`; вернуть его и выполненный ipynb с outputs. Latest сохраняется каждые1000 updates, отдельные checkpoints на50k/75k/100k включают tokenizer/AdamW/LR/RNG/coverage. После прерывания train последняя ячейка упаковывает текущие результаты; после перезапуска runtime нужна отдельная подготовка continuation.

## Источники

- [Qwen2Tokenizer v4.43.1](https://raw.githubusercontent.com/huggingface/transformers/v4.43.1/src/transformers/models/qwen2/tokenization_qwen2.py): точный regex и byte-level BPE; переносится рецепт границ, не готовый vocabulary/normalizer/веса.
- [Tokenizers0.23.2 pre-tokenizers](https://huggingface.co/docs/tokenizers/v0.23.2/api/pre-tokenizers), [BpeTrainer](https://huggingface.co/docs/tokenizers/v0.23.2/api/trainers): Split/Sequence/ByteLevel и собственный training corpus.
- [MobileLLM §2.2.2](https://arxiv.org/html/2402.14905v2#S2.SS2.SSS2), [CodeT5-small config](https://huggingface.co/Salesforce/codet5-small/blob/main/config.json): источники гипотезы о глубине, различия в масштабе/данных/архитектуре не позволяют переносить результаты напрямую.
- [Pinned разрешённый train](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/tree/b2724689ff079e9c22efd548fe134bfb29036dbb/data/train), [неизменное условие](../../task/hw2-code-generation.md), [выбор tokenizer](tokenizer-choice-for-numbers.md), [архитектурный ресёрч](small-code-transformer-architectures.md), библиография. Выводы о качестве/масштабировании — собственные проверяемые гипотезы.
