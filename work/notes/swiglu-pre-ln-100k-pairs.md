---
type: experiment
course:
  - nlp
status: complete
understanding: 1
created: 2026-10-08
updated: 2026-10-08
sources:
  - https://arxiv.org/abs/2002.05202
  - https://docs.pytorch.org/docs/2.11/notes/amp_examples.html
  - https://docs.nvidia.com/cuda/archive/11.8.0/ampere-tuning-guide/index.html
  - https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/torch/cuda/__init__.py
  - https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/modules/transformer.py
aliases:
  - ДЗ 2 опыт 15 SwiGLU
---
# Опыт 15: SwiGLU вместо ReLU в Pre-LN Transformer

**Получен полный T4-прогон: 100000 шагов.** SwiGLU снижает CE и ошибки имён, численные reference-совпадения растут умеренно:117→123/888 относительно14. Принимаем15/100k как рабочую базу. [IPYNB с outputs](../results/15-swiglu-pre-ln-bpe-4k-100k-pairs-100000-steps-heldout-1024-results.ipynb), [сводка/manifest/опись весов](../results/15-swiglu-pre-ln-bpe-4k-100k-pairs-100000-steps-heldout-1024-summary.json), [новая статика](../checks/15-swiglu-results-static-checks.json), [CPU-forward без обучения](../checks/15-swiglu-conditioning-cpu-checks.json), [численный и ручной аудит](../checks/15-swiglu-functional-quality-audit.json). Official pass@1 неизвестен.

## Зачем

[14: продолжение до 100k](pre-ln-continue-to-100k-expanded-heldout.md) улучшил CE и дал умеренный прирост reference-совпадений: 104→117/888. Источник уже влияет на модель, но в коде остаются ошибки чисел, операций и имён. **Гипотеза:** взаимодействие двух зависящих от входа ветвей FFN позволит лучше преобразовывать признаки после attention при почти том же размере модели. Это не гарантия правильной арифметики и не доказательство недостаточной ёмкости ReLU.

После обсуждения решили **не делать overfit16 обязательным условием** этого опыта. Loss падает, conditioning сильный, есть новые верные решения; проверки masks/decoder/KV на обученных весах прошли. Старый byte-overfit действительно относится к другой конфигурации. Повторный overfit текущей модели остаётся диагностикой на случай подозрительного плато или сбоя, а не обязательной ступенью перед каждым изменением.

## Формально и интуитивно

Прежний FFN для представления одного токена $x\in\mathbb R^{256}$:

$$
f_{\mathrm{ReLU}}(x)=\mathrm{ReLU}(xW_1+b_1)W_2+b_2.
$$

$W_1\in\mathbb R^{256\times1024}$ расширяет признаки, $W_2\in\mathbb R^{1024\times256}$ возвращает ширину модели; ReLU обнуляет отрицательные координаты.

Новый FFN:

$$
g=\mathrm{SiLU}(xW_g+b_g),\qquad v=xW_v+b_v,\qquad
f_{\mathrm{SwiGLU}}(x)=(g\odot v)W_o+b_o,
$$

$$\mathrm{SiLU}(z)=z\sigma(z).$$

$W_g,W_v\in\mathbb R^{256\times680}$ — gate и value, $W_o\in\mathbb R^{680\times256}$ — выходная проекция; $b$ — bias, $\sigma$ — sigmoid, $\odot$ — поэлементное произведение. Один и тот же value-признак получает разный вклад при разных входах: координата $g_j(x)$ модулирует $v_j(x)$. Gate в SwiGLU не является вероятностью: возможны отрицательные значения и значения больше единицы. Это преобразование каждого токена; обмен информацией между позициями по-прежнему выполняет attention.

В коде dropout0.1 остаётся после произведения и после выходной проекции, residual и Pre-LN сохранены. В формулах dropout опущен. Bias остаются, как в baseline; в статье Shazeer сравнивались варианты без bias, поэтому её точная конфигурация не копируется.

## Один фактор и бюджет

Три матрицы вместо двух: без bias $2dm\approx3dg$, поэтому новая внутренняя ширина $g\approx2m/3$. При $d=256$, $m=1024$ выбираем $g=680$ — близкое округление до кратного8.

| | ReLU14 | SwiGLU15 |
| --- | ---: | ---: |
| Внутренняя ширина | 1024 | 680 |
| Параметров FFN одного слоя, с bias | 525 568 | 523 856 |
| Параметров всей модели | 6 312 960 | 6 306 112 |

Разница −6848 (−0.1085%). Близкое число матричных операций не гарантирует одинаковое реальное время: добавлены SiLU/произведение, а fused encoder для ReLU/GELU не поддерживает gated-проекцию.

Сохраняем: encoder2+decoder2, d256/4heads, Pre-LN и конечные нормы; shared src/tgt embedding и отдельный head; learned absolute positions2048; BPE4096 из08, PAD/BOS/EOS0/1/2; dropout0.1, FP32, batch8, AdamW/weight_decay0.01/clip1/seed42. Маски, teacher forcing и KV/greedy/cap512/batch32 прежние.

**Обучение с нуля 100000 updates, не продолжение14.** LR полностью повторяет13+14: шаг1 $10^{-6}$; warmup1000 до0.001, cosine до0.0001 на50000; далее floor0.0001 до100000. Горизонт scheduler50000, общий бюджет100000 — разные настройки. Повторно обучать BPE или загружать старые веса не нужно.

Для уменьшения случайных различий сначала создаётся прежняя модель13, затем заменяются только FFN под изолированным CPU RNG seed42. Xavier для новых матриц, bias стандартных Linear. Начальные embeddings/positions/attention/norms/head и RNG до первого batch точно сохранены на CPU. Размер скрытого слоя меняет dropout-маски и дальнейшее обучение; один seed не оценивает вариативность результатов.

## Данные и оценка

Те же100000 train-пар/99212 разных вопросов из фиксированного shard, BPE JSON SHA `73614c921b43aa6ef715bb23bba583b96785a82304c1964f74ab11d8deef7f86`. Heldout прежний1024=64+192+768, old256 prefix; train-overlap0. Хеш всех1024 question/code `e187bc297218365f1a1452a15376e0ff2fceaf8a5a5517820549854b90564743`.

- Ранние2k/5k/10k/20k/30k: train64/validation64, CE, greedy/AST/EOS/имена, question conditioning.
- 50k/75k/100k: дополнительно все1024 heldout; token-weighted CE, fixed888 numeric-reference proxy и старые fixed223. Диагностика сохраняет RNG/режим, cache/full parity проверяется до обучения и на каждом контроле.
- Финал: прежний byte512-validation и весь1919 test, исходный порядок, тот же таймер. Сравнить скорость/память, EOS, длины, численную и ручную оценку, затем решить, сохранять ли SwiGLU.
- Latest каждые1000 updates; отдельные50k/75k/100k checkpoints с AdamW/scheduler/RNG/BPE/data hashes, ZIP/manifest и partial archive. После прерывания последняя ячейка только упаковывает доступные файлы.

Сохранённые пользовательские T4-показатели14, встроенные в15 для сравнения по одинаковому числу обновлений:

| Updates | Heldout1024 CE | Reference-совпадения из888 |
| --- | ---: | ---: |
| 50 000 | 1.03159 | 104 |
| 75 000 | 0.98946 | 97 |
| 100 000 | 0.96195 | 117 |

Старый test1919:41.19 с; EOS1884, undefined237. Это исторические измерения14, **не результат15**. Numeric agreement не официальный pass@1: reference иногда неверен,136 неподдержанных reference отдельно; ошибки генерации остаются в фиксированном знаменателе888. Просмотреть индивидуальные приобретения/потери и тот же ручной23, если разница небольшая.

## Проверки при подготовке, до пользовательского запуска

CPU/PyTorch2.14.1: полный класс6.306M, реальные100k train/1024 heldout и встроенный BPE, 21 ячейка/11 code/синтаксис/пустые outputs. Все4 SwiGLU соответствуют формуле; обе ветви и выходные проекции/attention/cross-attention получают конечные ненулевые градиенты. Target causal и source padding invariance прошли. Исходные non-FFN веса и CPU RNG совпали с13.

Обычный decoder и KV совпали до512 позиций, в том числе с padding завершённых строк; max error1.32e-6 до/после smoke, greedy64 tokens совпали. **11 optimizer updates всего**:4 notebook/control,4 без диагностики,2 resume и1 до контролируемого прерывания. Weights/AdamW/scheduler/RNG точно совпали с/без оценок и после resume. Проверены early64/full1024 ветви, fixed223/888 и canonical hashes, rerun guard, latest на шаге1, полные/частичные ZIP/CRC/manifest. В smoke generation cap4/test2; teacher-forced targets полные. После исполнения исправлены только номера/пробелы Markdown-заголовков; идентичность всех code cells подтверждена.

**При подготовке не выполнялись:** полное100k обучение, T4-тайминги, полная1919 генерация, CUDA RNG resume, скачивание через браузер Colab и официальная оценка. Последующий пользовательский T4-прогон разобран ниже; эти11 CPU updates относятся к подготовке, а не к новому разбору.

## Полученные результаты: SwiGLU15 против ReLU14

Одна и та же модель GPU T4/PyTorch2.11.0+cu130, прежние данные/BPE/Pre-LN/FP32/batch8 и LR. Checkpoints показывают ровно100k AdamW updates; warmup1000/cosine до50k/floor до100k применены. Размер модели6 306 112 параметров против6 312 960; обе архитектуры обучены с нуля на суммарном бюджете100k. Сохранённое14 складывается из13/первых50k и14/добавочных50k.

| Метрика на фиксированном heldout | ReLU50k | SwiGLU50k | ReLU75k | SwiGLU75k | ReLU100k | SwiGLU100k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CE1024, token-weighted |1.03159|0.97612|0.98946|0.93525|0.96195|0.91067|
| Reference-совпадения из888 |104|120|97|118|117|123|
| Прежние reference223 |24|31|18|32|30|32|
| EOS из1024 |999|1009|1000|1011|1008|1013|
| Undefined names из1024 |102|103|120|78|105|82|

На100k train64 CE0.7774→0.7196, validation64 CE0.9332→0.8648, CE1024 −5.33%. Вопрос продолжает влиять на teacher forcing: подмена correct→rotated повышает CE15 на1.479. По этому разрыву15 не сильнее14 (+1.540); меньше CE не доказывает улучшения каждого аспекта conditioning.

Численный proxy117/888→123/888 (13.18%→13.85%, +0.68п.п.):80 совпадений сохранены,43 появились,37 потерялись. Прирост небольшой; одна seed-пара и накопленная настройка на heldout не устанавливают статистически устойчивое превосходство. Reference иногда неверен,136 unsupported reference отдельно; это **не official pass@1 и не подтверждённая точность13.85%**.

Ручной прежний23:14/100k —3верных/19неверных/1unknown,15/100k —4/17/2. Просмотрены все80 proxy-изменений:40 новых и36 потерянных корректных чисел при указанных чтениях/единицах;4 случая не дают прямой оценки. Например, по530 шарикам нельзя получить точные2/3 целых красных; «10cards each» даёт10каждому, reference5 ошибочен;41 конфета на5 друзей неоднозначны между8+остаток и8.2;12×9 разрезов дают130кусков, reference108 неверен. Единицы/допущения отдельно записаны в аудите. Это просмотр изменившихся случаев, а не оценка всех1024.

Реальные приобретения:480miles/20gallons теперь24 вместо400;2cups/12cookies для30cookies теперь5 вместо undefined; налог10% на200 теперь220. Есть и потери:7×6feet стало36 вместо42,8hours×5days×12/hour стало96 вместо480. Имена улучшились, перенос чисел/выбор операций ещё ненадёжен.

## Скорость и запас

Сохранённый T4: чистое обучение15 **2161.90с=36.03мин**,46.26updates/s,21.62ms/update. С диагностиками/логами/checkpoints **2451.53с=40.86мин**; последние byte-validation/test сюда не входят. Накладная часть289.64с=4.83мин/11.8% активного времени. Прежний полный ReLU100k =1044.38с(13)+1039.50с(14)=2083.88с=34.73мин: SwiGLU примерно на3.74% медленнее, не вдвое. Сравнивать весь15 только с добавочными17.32мин14 некорректно.

| Полный test1919, FP32/KV/batch32 | ReLU14/100k | SwiGLU15/100k |
| --- | ---: | ---: |
| Время, с |41.1931|36.6384|
| Ответов/с |46.59|52.38|
| AST Python |1901|1906|
| EOS |1884|1898|
| Undefined names |237|175|
| Численно поддержанных программ |1587|1657|
| Достигнут cap512 |35|21|
| Пик памяти инференса, MiB |223.48|223.74|

Test быстрее на11.06%, лимит текущего таймера180с с большим запасом. Это не замер полного неизвестного официального таймера. Объяснение по сохранённым длинам: средний output67.48→64.25tokens, batched decode steps17898→15072, batches with cap26→18 из60. Более короткие хвосты, вероятно, объясняют часть ускорения; это не доказательство более быстрых kernels SwiGLU. Активных token slots25.6%, но это не GPU utilization. Память223.7MiB — инференс, **не пиковая память обучения**.

Внутри15 добавочные50k снизили CE1024 0.9761→0.9107, но proxy120→123 (38gains/35losses). Для первичного отбора будущих гипотез разумно сравнение на50k;100k оставить перспективным вариантам. Это выбор следующего протокола, не утверждение, что модель перестала учиться.

## Предложение после15: ускорение обучения (история)

**Обновление после16:** [16: скорость обучения](training-speed-bpe-cache-amp-fp16.md) получен. FP32+cache выбран:19.794мс противonline21.082; AMP24.600мс медленнее. База качества15 сохраняется, следующий предложенный фактор — train500k при прежнем бюджете, контроль50k/100k. Ниже исходный план до получения16; доли времени и nativeBF16 теперь проверены в сохранённом T4-прогоне.

1. Одинаковые начальные weights/AdamW/RNG и последовательность batch8. Сравнить A:FP32+onlineBPE, B:FP32+cachedBPE, C:AMP FP16+cachedBPE. A→B измеряет подготовку данных, B→C — точность вычислений. Отдельно первый setup/pretokenization, прогрев, steady-state step, полезные target tokens/s, CPU preparation/transfer, GPU и память. Повторить короткие блоки200–500updates, восстановив исходные состояния; финальное качество ими не оценивать.
2. AMP через autocast FP16 и GradScaler, веса/AdamW сохранять FP32, unscale до clipping; учитывать skipped optimizer steps и двигать LR по успешным updates, сохранять scaler. Проверить finite loss/gradients и стабильность. Оценку/инференс первоначально оставить FP32 для сопоставимости. Численная траектория изменится, точное равенство weights/ответов с FP32 не ожидается.
3. Если выигрыш заметный и реализация стабильна — свежий50k обучающий контроль против15/50k на прежних данных/модели/расписании, затем решение о100k. Bucketing и batch8→16/32 измерять отдельно: batch меняет предъявления и оптимизацию, сравнивать tokens/examples/time/quality, а не только steps/s. torch.compile рассматривать после профиля и стабилизации форм; выгоду и цену компиляции сейчас не знаем.

В исходном коде BPE вызывается заново для каждой выбранной пары и tensors собираются на CPU каждый update; предварительный cache IDs должен сохранить точные входы/labels, masks и выбор пар. Но **доля времени этой работы на Colab не измерена**. Малые batch8/model6.3M могут ограничиваться CPU/мелкими kernel launches; AMP не гарантирует ускорение2×. Сокращение только диагностик даёт ограниченный выигрыш, поскольку88.2% активного времени уже приходится на train.

**BF16 на нашей T4 не выбирать первым.** T4 — Turing/compute7.5, Tensor Cores поддерживают FP16; native BF16 Tensor Core math появляется в Ampere. BF16 имеет диапазон экспоненты какFP32 и полезен на совместимом GPU, но программная возможность хранить BF16-тензор не доказывает аппаратного ускорения. В PyTorch2.11 проверять `torch.cuda.is_bf16_supported(including_emulation=False)`; defaultTrue допускает эмуляцию. Здесь это вывод из официальных характеристик и кода API; нового аппаратного probe/FP16-замера не было.

## Что проверено при разборе15

Только новые проверки на CPU:21/11 ячеек/source/counts/outputs/no errors, ZIP CRC/manifest,4 safe `weights_only` checkpoints/finite FP32 weights+AdamW+scheduler+RNG/LR/BPE/data/latest==100k. Проверены5887 savedrows:token decoding/AST/numeric/агрегаты, все fixed question/reference pairs888/223, test1919 порядок, byte512 prefix, напечатанные сводки и сохранённые исторические14.

Новый CPU/PyTorch2.14.1 forward64:correct0.864756/rotated2.343722/reversed2.426948, первые2 воспроизвели T4 в1e-5. Три полных greedy-ответа(indices13/2/1:верный/undefined/неверный numeric) совпали full/KV/savedT4, веса неизменны. **Optimizer updates0.** Полное обучение/GPU-тайминги/1919 генераций/официальный checker/AMP не повторялись, supplied программы не исполнялись.

В Git — notebook с outputs, сводка, компактные отчёты/заметки. Локально ignored: `results/generations/15-swiglu-pre-ln-bpe-4k-100k-pairs-100000-steps-heldout-1024-{full-results,test-generations}.json`, `results/checkpoints/15-swiglu-pre-ln-bpe-4k-tokenizer.json`, `15-swiglu-pre-ln-100k-pairs-heldout1024-step-{50000,75000,100000}.pt` и latest-step100000. Точные hashes/размеры в сводке; для переноса копировать отдельно либо восстановить из пользовательского ZIP. Текущий checkpoint100k (`../results/checkpoints/15-swiglu-pre-ln-100k-pairs-heldout1024-step-100000.pt`, нет в репозитории).

## Источники

- Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202),2020,§2–3: SwiGLU/SiLU, три матрицы и выравнивание бюджета2/3. Источник уже в библиографии. Улучшение нашей генерации — собственная проверяемая гипотеза.
- PyTorch [Transformer2.11 source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/nn/modules/transformer.py), [EncoderLayer docs](https://docs.pytorch.org/docs/2.11/generated/torch.nn.TransformerEncoderLayer.html): callable activation, FFN/dropout/Pre-LN и ограничения fused fast path; проверены при подготовке. CPU runtime2.14.1 указан отдельно.
- GLU/SwiGLU, лекция4, слайды33–36; [14 и сохранённые результаты](pre-ln-continue-to-100k-expanded-heldout.md) — база сравнения, данные/оценка.

- Пользовательский IPYNB/ZIP15 — сохранённые T4-результаты, происхождение/SHA/manifest в сводке. Численный/ручной аудит и CPU-forward выполнены при разборе отдельно.
- NVIDIA [T4 specs](https://www.nvidia.com/en-us/data-center/tesla-t4/), [compute capability table](https://developer.nvidia.com/cuda/gpus), [Ampere Tensor Core formats](https://docs.nvidia.com/cuda/archive/11.8.0/ampere-tuning-guide/index.html#improved-tensor-core-operations):FP16 уTuring/T4, nativeBF16 уAmpere; не benchmark нашей модели.
- PyTorch [AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html), [AMP2.11 examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html), [is_bf16_supported2.11 source](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/cuda/__init__.py):autocast/scaler/clipping/skips, условия выигрыша и проверка эмуляции. Источники прочитаны сейчас; AMP на T4 не запускался.
