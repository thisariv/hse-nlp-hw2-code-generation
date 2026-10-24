---
type: experiment
course:
  - nlp
status: complete
understanding: 1
created: 2026-10-08
updated: 2026-10-08
sources:
  - https://docs.pytorch.org/docs/2.11/notes/amp_examples.html
  - https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
  - https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html
  - https://docs.pytorch.org/docs/2.11/generated/torch.optim.Optimizer.register_step_post_hook.html
  - https://docs.pytorch.org/docs/2.11/notes/randomness.html
aliases:
  - ДЗ 2 опыт 16 скорость обучения
---
# Опыт 16: BPE-кеш и AMP FP16

**Получен полный пользовательский T4-benchmark. Выбран FP32 с кешем BPE при batch 8.** Кеш сократил время шага примерно на 6%; AMP FP16 увеличил его примерно на 24% относительно кеша/FP32. Это результат скорости коротких блоков. Качественная база остаётся [15/100k](swiglu-pre-ln-100k-pairs.md), новых итоговых весов нет.

[Исходный ноутбук](../notebooks/16-training-speed-bpe-cache-amp-fp16-swiglu-bpe-4k-100k-pairs.ipynb) · [прогон с outputs](../results/16-training-speed-bpe-cache-amp-fp16-swiglu-bpe-4k-100k-pairs-results.ipynb) · [сводка](../results/16-training-speed-bpe-cache-amp-fp16-swiglu-bpe-4k-100k-pairs-summary.json) · [полный JSON](../results/16-training-speed-bpe-cache-amp-fp16-swiglu-bpe-4k-100k-pairs-results.json) · [CSV](../results/16-training-speed-bpe-cache-amp-fp16-swiglu-bpe-4k-100k-pairs-summary.csv) · [проверки результата](../checks/16-training-speed-results-static-checks.json).

## Гипотезы и протокол

A→B проверяет устранение повторной токенизации. B→C проверяет смешанную точность при прежней подготовке данных.

| Режим | Подготовка данных | Вычисления |
| --- | --- | --- |
| A | BPE на каждом шаге, как в 15 | FP32 |
| B | Заранее посчитанные ID; прежние CPU tensors/padding/transfers | FP32 |
| C | Тот же кеш ID | autocast FP16 + GradScaler, веса/AdamW FP32 |

Прежняя модель: encoder 2 / decoder 2, ширина 256, 4 heads, Pre-LN/SwiGLU680, 6 306 112 параметров, dropout 0.1. Те же 100k пар / validation64 из pinned shard0, BPE08 без переобучения, batch 8, AdamW wd 0.01, clip 1 и LR из 15: warmup 1000, cosine до 50k, затем floor. Каждый режим начинает с одинаковых весов и пустого AdamW; checkpoint 15 не загружается.

Три повтора, порядок ABC/BCA/CAB; внутри повтора одинаковые batch indices и начальный RNG. Каждый режим: 100 warmup attempts + 500 измеряемых. В первом повторе ещё 20 отдельных profile attempts на режим. Завершены все **5460 attempts / 5460 optimizer updates**; весь benchmark с прогревом, отдельным профилем и оценками занял 133.05 с, setup вне этого таймера. Runtime: Tesla T4, PyTorch 2.11.0+cu130, CUDA 13.0, Tokenizers 0.23.2, один CPU thread. Native BF16 probe вернул false.

Главный таймер охватывает prepare/transfer/forward/loss/backward/unscale/clip/step/scaler/scheduler; CUDA sync только на границах. Setup, warmup, logging, validation и отдельный сериализованный профиль исключены. **В таблице медиана трёх средних времён по 500 шагов**, не медиана отдельных step latencies. Все попытки успешны, поэтому ms/attempt здесь равны ms/update.

## Сохранённые результаты T4

| Режим | мс/шаг, медиана | Диапазон средних трёх блоков | Полезные target tokens/s, медиана | Максимум peak allocated, MiB |
| --- | ---: | ---: | ---: | ---: |
| A: FP32 + online BPE | 21.082 | 21.077–21.336 | 24 798 | 366.44 |
| B: FP32 + кеш BPE | 19.794 | 18.725–19.984 | 26 475 | 366.44 |
| C: AMP FP16 + кеш BPE | 24.600 | 22.671–25.020 | 21 507 | 281.40 |

A→B: время шага −6.11%, updates/s +6.51% по отношению медиан. Кеш быстрее во всех трёх парных повторах: ускорение 1.065–1.126×. Построение кеша всех 100k пар заняло 27.43 с, 11 120 481 ID; окупаемость по разности медиан примерно 21.3k updates. Это экстраполяция, а не измеренный полный 50k/100k train. При расширении до500k стоимость подготовки вырастет: окупаемость21.3k относится только к кешу100k, полный wall time нужно измерять заново. Повторное использование готового кеша в следующих запусках может амортизировать setup.

B→C: время шага +24.28%, peak allocated −23.21% (−85.04 MiB). Во всех трёх парных повторах C медленнее B. Измеряемых и warmup skips нет, loss / unscaled gradient norm / параметры конечны; AMP scale остался 65536 после 600 updates в каждом повторе. **Численная стабильность не даёт ускорения автоматически.** Peak allocated описывает память тензоров PyTorch в обучении; это не весь расход GPU по nvidia-smi.

Отдельный профиль, медиана 20 samples:

| Режим | CPU preparation, мс | Transfer stream span, мс | Compute stream span, мс |
| --- | ---: | ---: | ---: |
| A | 2.315 | 0.094 | 17.959 |
| B | 0.489 | 0.092 | 17.948 |
| C | 0.508 | 0.092 | 20.663 |

Кеш ускорил подготовку примерно в 4.73 раза, но она занимала небольшую часть цикла. FP16-замедление наблюдается в пути вычисления и обновления, а не в BPE. **Причина пока не измерена по kernels.** Рабочая гипотеза: у небольшой модели и batch 8 накладные расходы autocast/scaler и мелких операций перевешивают выигрыш матричных умножений. PyTorch AMP recipe предупреждает о слабом ускорении малых workload. Эти spans включают ожидания/запуски с CPU и получены с дополнительными sync; их нельзя складывать с главным таймером или выдавать за чистое kernel time / GPU utilization.

## Короткий численный контроль

Online/cache inputs/labels точно совпали на 17 контрольных батчах. Полезные token counts и padding одинаковы во всех парных измеряемых блоках; target padding около 40.8–41.3%. A/B validation64 CE после 600 updates отличается максимум на 0.00000431. Конечные FP32-веса **не побитово равны**: max abs difference 0.000364–0.000501. CUDA-недетерминизм — возможное объяснение, не установленная причина; deterministic rerun и сами веса не предоставлены.

CE около 3.2 после 600 updates со старта не сопоставлять с CE1024 0.9107 после 100k в 15. В 16 нет полного обучения, greedy-программ, test1919 или inference timings. Его показатели не заменяют функциональную оценку и не меняют результат 15: reference proxy 123/888, официальный pass@1 неизвестен.

## Что проверено сейчас

Сверены 19 ячеек / 9 code, точное совпадение всех source cells с подготовленным 16, execution counts 1–9, отсутствие error outputs и синтаксис. ZIP CRC, manifest sizes/SHA256, printed runtime/data/tokenizer/model/summary/pairs согласованы с JSON/CSV. Повторно посчитаны времена/throughput, медианы и парные отношения девяти блоков, медианы 60 profile samples, число updates и цена кеша. План batch indices восстановлен только из seed и размеров, hash совпал; data/BPE hashes прежние.

**Сейчас optimizer updates 0, model forwards 0, GPU-запусков 0.** Метрики CUDA, finite flags и scaler state взяты из сохранённого пользовательского прогона. Сохранены исходные notebook/JSON/CSV без изменения; provenance/hashes и ограничения — в [отчёте](../checks/16-training-speed-results-static-checks.json).

При подготовке ранее выполнялись [CPU-проверки](../checks/16-training-speed-cpu-checks.json): полный отбор/кеш/паритет, 27 успешных updates полной модели (18 FP32 / 9 FP16), partial ZIP/rerun и 2 attempts маленького Transformer (1 successful / 1 injected overflow). Всего 29 attempts / 28 successful CPU updates. Это прежние проверки реализации, не новые измерения скорости; [отчёт подготовки](../checks/16-training-speed-static-checks.json) сохранён как исторический.

## Решение и следующий опыт

Использовать **B: кеш ID + FP32 + batch 8**. AMP при данной конфигурации не принимать: его единственный выигрыш здесь — память. Полный 50k quality-control для AMP был условным следующим шагом при ускорении; это условие не выполнено. Отдельный длинный прогон только ради кеша сейчас не приоритетен: модель/правило обновления прежние, входы проверены, короткий A/B CE практически совпал. Это решение о бюджете, не доказательство побитовой идентичности длинных траекторий.

**Предложен следующий обучающий фактор: train 100k → 500k исходных разрешённых пар**, со старта, та же SwiGLU/Pre-LN/BPE/LR/batch 8/FP32 и кеш. Первый рубеж 50k сравнить с 15/50k (CE1024 0.9761, proxy 120/888); при полезной динамике продолжить до 100k и сравнить с 15/100k (0.9107, 123/888). Не менять LR/глубину/batch одновременно. Восемьсот тысяч предъявлений на 100k updates дают в среднем 8 на пару старого корпуса или 1.6 на пару нового; гипотеза — полезнее разнообразие задач, а не многократные повторы.

**Обновление после получения 17:** [17:500k пар](more-data-500k-pairs-swiglu.md) выполнен на T4:500k/50k, cache+FP32,46.53 updates/s/17.91 мин чистого train, setup195.57 с(cache137.18 с). CE1024 лучше 15/50k, proxy хуже(110 против 120/888); база 15/100k сохраняется. Это не парный benchmark кеша, эффект−6.1% из 16 автоматически не переносить. Предложено продолжение 17→100k без других факторов; fresh1024 не оценивался. [Очередь](experiment-plan.md).

## Источники

- Пользовательский notebook/ZIP 16 — фактические результаты T4; исходные имена и hashes в отчёте. [15](swiglu-pre-ln-100k-pairs.md) — база качества и параметры модели.
- PyTorch [AMP examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html), [AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html): autocast/scaler/clipping и условия ускорения. Recipe прочитан повторно 2026-10-08; это объяснение возможной причины, не профиль нашей модели.
- PyTorch2.11 [CUDA Event](https://docs.pytorch.org/docs/2.11/generated/torch.cuda.Event.html), [optimizer post-hook](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Optimizer.register_step_post_hook.html): timing и реальные optimizer updates.
- PyTorch2.11 [Reproducibility](https://docs.pytorch.org/docs/2.11/notes/randomness.html), прочитано 2026-10-08: CUDA и межверсионная воспроизводимость не гарантируются одним seed. Конкретная причина A/B-различий не установлена.
