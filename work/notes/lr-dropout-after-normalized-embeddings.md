---
type: experiment
course:
  - nlp
status: complete
understanding: 2
created: 2026-10-10
updated: 2026-10-10
sources:
  - https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/tree/b2724689ff079e9c22efd548fe134bfb29036dbb/data/train
aliases: []
---

# ДЗ2, опыт25: LR и dropout после нормировки эмбеддингов

[Ноутбук25](../notebooks/25-lr-dropout-after-normalized-embeddings.ipynb) · [реестр](../../JOURNAL.md) · [CPU-проверки](../checks/25-lr-dropout-normalized-embeddings-cpu-checks.json) · [разбор24](fast-recipe-equal-presentations-lr-and-ablations.md).

## Получен результат25: LR и dropout

**Новая рабочая база — D0:** нормированная инициализация токенов, peak LR0.003, dropout0. Exec-match **1677/4096 (40.94%)** и **422/991 (42.58%)**, CE4k0.4764. Контроль R5 переоценён точно; повтор его обучения не потребовался.

| Запуск | Peak LR | Dropout | Exec4096 | Exec991 | CE4096 | CEold1024 | Train, мин |
|---|---:|---:|---:|---:|---:|---:|---:|
| R5 | 0.002 | 0.1 | 1602/4096 (39.11%) | 406/991 (40.97%) | 0.4914 | 0.4997 | 12.95 (сохранён24) |
| L1 | 0.001 | 0.1 | 1432/4096 (34.96%) | 360/991 (36.33%) | 0.5425 | 0.5561 | 12.49 |
| L2 | 0.003 | 0.1 | 1585/4096 (38.70%) | 414/991 (41.78%) | 0.4868 | 0.4991 | 12.51 |
| L3 | 0.0045 | 0.1 | 1581/4096 (38.60%) | 398/991 (40.16%) | 0.4832 | 0.4952 | 12.50 |
| D0 | 0.003 | 0.0 | 1677/4096 (40.94%) | 422/991 (42.58%) | 0.4764 | 0.4863 | 11.72 |

Это сохранённый прогон Colab/T4. Четыре новых запуска обучены с нуля: одна модель A4+4/d256/h4/Pre-LN/SwiGLU680 (12.086912M), одинаковые initial/план/500k/BPE8k,6250×128=800000 предъявлений и52.797M полезных target-токенов. Train без reference-фильтрации. AMP FP16/GradScaler, bucketing, fused AdamW, wd0.01, clip1 после unscale; warmup313→cosine до10% peak. Token init std $d^{-1/2}$ при множителе $\sqrt d$; позиции прежние.

Основная метрика — совпадение числового результата исполнения с reference, math.isclose(rel=abs=1e-6). Heldout4k:4096 числовых reference; old1024:991 числовой reference и33 исключённых из exec. CEold использует все1024. Генерация — greedy FP16 sorted b512,cap512; CE — FP32,b32, по полезным токенам с EOS, без PAD. Выборка fresh1024 не оценивалась.

### Что получилось

**LR выбран по зафиксированному правилу.** Максимум среди R5/L1/L2/L3 —1602 уR5. L2 отстаёт на17 задач (−0.4150п.п.), входит в допуск <0.5п.п. и выбран по меньшей CE. L3 отстаёт на21 (−0.5127п.п.) и исключён, хотя CE0.4832 ниже, чем уL2. L3 действительно запущен с0.0045 после первоначального выбора L2.

LR0.001 явно хуже. R5/L2/L3 не разделены парным McNemar: L2vsR5 p0.4666, L3vsR5 p0.3735. **Собственный вывод:** нет подтверждения уникального оптимума0.003 или снижения лучшего LR после нормировки. Выбор по допуску20/21 чувствителен к одной задаче; это свойство заранее заданного правила, которое сохранено.

**Dropout0 принят:** D0 против L2 дал +92 задачи/+2.2461п.п.;334 gains/242 losses, exact paired McNemar p0.0001452. Против R5: +75/+1.8311п.п.;325/250,p0.002003. На old991: +16/+1.6145п.п. против R5;83/67,p0.2205 — отдельного подтверждения на этом split нет.

Здесь сравнение D0 сL2 меняет только dropout. Сравнение сR5 меняет также LR. Отрицательный R4 №24 относился к прежнему большому масштабу инициализации. Dropout0 при нормированной инициализации иLR0.002/0.0045 не проверялся; взаимодействие этих факторов полностью не изолировано. Все p приведены без поправки на выбор и несколько сравнений; выполнен один seed.

### Длинные программы

| Reference BPE без EOS | Количество | R5 | L2 | D0 |
|---|---:|---:|---:|---:|
| <60 | 2013 | 1195 (59.36%) | 1182 (58.72%) | 1224 (60.80%) |
| 60–90 | 1457 | 355 (24.37%) | 341 (23.40%) | 387 (26.56%) |
| 90–120 | 488 | 49 (10.04%) | 58 (11.89%) | 62 (12.70%) |
| ≥120 | 138 | 3 (2.17%) | 4 (2.90%) | 4 (2.90%) |

D0 улучшил результат во всех группах против R5, но при reference≥120 токенов это лишь **4/138**. Длина связана и со сложностью задачи, поэтому таблица не устанавливает причину ошибок или архитектурное ограничение. Основной exec не требует совпадения текста программы.

### Сохранённые замеры T4

Окружение: Tesla T4, Torch2.11.0+cu130/CUDA13.0, fused AdamW. Preflight инференса наблюдал efficient SDPA; это не проверяет backend всех форм обучения и backward.

Все четыре запуска: конечные loss/grad, **AMP-skips0**,6250 optimizer updates, source38706084/target52797347 полезных токенов; padding30.22%. L1/L2/L3 обучались12.49/12.51/12.50мин:1066–1068пар/с,≈70.4k target-токенов/с. **D0:702.9482с=11.72мин,1138.1пар/с,75.11k target и55.06k source-токенов/с.** Его throughput выше L2 на6.78%, время меньше на6.35%. Пиковая выделенная память:7.64GB вL1–L3 и7.34GB вD0; зарезервированная13.25–13.30GB, единицы десятичные.

Сумма четырёх train49.22мин, активная сессия57.78мин, от первой ячейки58.63мин; бюджет≤75мин выполнен. Итоговое скачивание/копия ZIP вне таймера. Вход через Drive со стартовым ZIP100.08MB занял **27.44с** по runtime.json (mount/copy/verify/import); это не гарантированная скорость сети для других файлов.

D0 test1919: **полное время10.3348с, генерация9.9065с**;1456 шагов декодирования суммарно по четырём batch, зарезервировано1.797GB. Полный таймер включает загрузку/восстановление весов, tokenizer,FP16/H2D,warmup,сортировку,greedy,восстановление порядка и запись answers. Data download/cache/exec/CE/JSON dump/download исключены. EOS1915/1919,cap4,AST-valid1911. В этой области таймера выполнен лимит180с; официальный score и формат бота остаются открытыми. JSONL — кандидат для посылки.

### Проверено при разборе сейчас

- CRC/manifest/SHA всех **86файлов**; модули совпали с подготовленным25. Единственная правка кода notebook — INPUT_MODE auto→drive.
- Exact initial/tokenizer/план/heldout24, уникальность и исключения test/old/fresh. Train заново не скачивался: совпадение подтверждено сохранёнными selection hashes/counts. Fresh прочитан только для hash/exclusion; encoding/evaluation0.
- Все четыре полных checkpoint прочитаны с weights_only=True: finite FP32 model/AdamW moments, strict spec/12.086912M, optimizer steps6250, scheduler horizon/last update6250, GradScaler8192/growth tracker250, RNG и provenance SHA. Состояния нового optimizer к старым весам R5 не приписывались.
- Порядок генераций, decode/EOS/cap/answers SHA, bins/McNemar и решения LR/D0 пересчитаны по сохранённым JSON.
- CPU повторно исполнил **30720программ/25475уникальных за33.03с**: все reference-числа и10exec-scores/наборы правильных строк воспроизведены точно. Это macOS CPU; RLIMIT_AS действует только наLinux. Обучение, CE, greedy и T4 повторно не запускались; новых LM-updates0.

Снимок notebook имеет null execution_count уcell12/14 и не содержит output14; история сохранена без исправлений. ZIP содержит полные training/evaluation receipts, decision, test и manifest. Его результаты проверены, поэтому неполные outputs notebook не означают незавершённый train.

**Ограничения:** один seed, heldout4k использован для выбора LR/dropout, old991 многократно переиспользован. Fresh не оценён; test gold/leaderboard неизвестны. CE измеряет teacher forcing и не заменяет exec-match. Новый опыт26 не подготовлен; следующий протокол обсуждается с наставником.

### Файлы

[Notebook с outputs](../results/25-lr-dropout-after-normalized-embeddings-results.ipynb) · [сводка](../results/25-lr-dropout-after-normalized-embeddings-summary.json) · [статические проверки](../checks/25-lr-dropout-after-normalized-embeddings-results-static-checks.json) · [CPU exec](../checks/25-lr-dropout-after-normalized-embeddings-results-exec-cpu-checks.json).

Raw/test JSONL: `work/results/generations/25-lr-dropout-after-normalized-embeddings/`. Полный ZIP: `work/results/checkpoints/25-lr-dropout-after-normalized-embeddings-results.zip` (**691182608байт**). Лучшая база отдельно: `work/results/checkpoints/25-d0-normalized-embeddings-lr003-dropout0/` (`checkpoint.pt`, `tokenizer.json`, `source.json`). Outputs/ZIP/PT локально, вне Git. Checkpoint145223339байт содержит model/AdamW/GradScaler/scheduler/RNG; пригоден для дальнейшего продолжения. Последний LR0.0003; старый scheduler закончился, новый horizon нужно задать отдельным протоколом.

SHA checkpoint `597451d05c0130b4aef09eb0d504180b8056ca9f1368a0961454310e99ffb4c6`; tokenizer `261f25b5d4be006a6a00e41e0eafbe522580741da627ba670f6b9805a2659b7d`. Для следующего входа достаточно нужного checkpoint/tokenizer через Drive; стартовый ZIP26 не создавался.

Последующие разделы сохраняют **протокол до запуска25** и подготовительные CPU-smoke. Формулировки «доtrain»/«не проверено» относятся к тому этапу.



## Гипотеза

LR0.002 выбран в24 при token init $N(0,1)$ и множителе $\sqrt d$. После уменьшения initial token-table в $\sqrt{256}=16$ раз (R5) лучший LR и роль dropout могли сместиться. Предположение, что старым большим случайным векторам нужен больший шаг, — пользовательская гипотеза о механизме; отдельного измерения движения эмбеддингов нет. Сильное улучшение R5 мотивирует новый LR-поиск при этой инициализации.

## Неизменные условия

- A: 4 encoder +4 decoder, d256/h4/SwiGLU680, Pre-LN, 12086912 параметров. Token init std $d^{-1/2}$, multiplier $\sqrt d$, позиции прежние. Загружен точный initial-state24; меняется только token-table /16. Seed42 перед каждым train и один точный NPY-план24; новые случайные initial tensors не генерируются.
- Train500k, BPE8k22, pinned HF `b2724689ff079e9c22efd548fe134bfb29036dbb`, прежние лимиты/выбор/порядок, **без reference-фильтрации**. 6250×128=800000 предъявлений; source38706084/target52797347 полезных токенов, target включает EOS, PAD исключён. Bucketing в исходном плане: сортировка внутри чанков4096, shuffle batches, plan seed230042.
- AMP FP16+GradScaler(init_scale1024), fused AdamW при доступности (режим фиксирован across resume), betas0.9/0.999, wd0.01, unscale→clip1, dropout0.1. Warmup313=ceil(5%6250)→cosine до10% peak. Точная арифметика каждого LR совпадает с24; scheduler сериализует параметры и последний update. Warmup входит в6250, дополнительных предъявлений нет.
- Используются точные heldout4k/old1024 из24, новая выборка не создаётся. Уникальность/validation-rule/исключения test/old/fresh проверяются. Fresh1024 только exclusion/hash, без encoding/generation/evaluation; строки fresh не сохраняются в25.

<details>
<summary>Зафиксированные SHA256 входов</summary>

- R5/model-state: `0459432561dc0b19605d7516062e70ef9c45a4dcbb21b8cd8ee2a59687f7de09`.
- Initial24/file: `daedd90fe51fef95dc151bc28e2f1ed0c66de80e5aa50a9ad8723ec50204c647`.
- Normalized initial/tensors: `974f9b6efca5e95e8b4e9269858984d6c70bdf62e354f490931d1b984005e380`.
- Tokenizer: `261f25b5d4be006a6a00e41e0eafbe522580741da627ba670f6b9805a2659b7d`.
- Batch NPY: `48734ce4d9621e0adf464e5fad25bfc625772d7a222baeada107ea92c0a13628`.
- Heldout4k/pairs: `6b0d0d559a56921faac1842e3508889225af1fe1ad8e04de378840bc4df8b047`.
- Heldout4k/file: `28a3c0c9088c5b4a17d75d216e4abc4262cc9ca62884966c43e7ec09b6f040c2`.

</details>

## Запуски и правила до обучения

| Запуск | Peak LR | Dropout | Условие |
| --- | ---: | ---: | --- |
| R5/control24 | 0.002 | 0.1 | Загрузить и переоценить |
| L1 | 0.001 | 0.1 | С нуля |
| L2 | 0.003 | 0.1 | С нуля |
| L3 | 0.0005 / 0.0045 | 0.1 | Если лучший{L1,R5,L2} —0.001 /0.003; при0.002 пропустить |
| D0 | Лучший LR после L3 | 0 | С нуля, только dropout |

**Лучший LR:** exec-match4k; при разнице <0.5п.п. — CE4k. Чтобы сравнение нескольких моделей не зависело от порядка, берётся максимум exec, кандидаты с отставанием строго <0.005×4096=20.48 совпадения, затем минимум CE и стабильный ID. Поэтому отставание20 разрешает выбор поCE,21 — уже нет. L3 определяется этим же правилом; после него заново выбирается LR среди{R5,L1,L2,L3}.

**D0 принимается:** потеря≤0.5п.п. против выбранного LR, максимум20 совпадений на4k. Из выбранного LR и принятогоD0 итоговый запуск выбирается тем же exec/CE-правилом. McNemar выводится отдельно и не является порогом принятия. Finite loss/grad и ноль AMP-skips обязательны для полного сравнения; неуспешный train останавливает этап, без компенсации дополнительными парами или выбора по неполной LR-сетке.

## Оценка

Greedy FP16 sorted batch512, cap512, прежний KV/preallocated путь с проверкой EOS каждые16 шагов; исходный порядок восстанавливается. Exec-match на4096 — основной, на old991 — дополнительный. CEFP32/b32, token-weighted с EOS, безPAD; CEold использует1024 строки. Reference-length bins по BPE-content безEOS: <60 /60–90 /90–120 /≥120. Все новые запуски получают exact paired McNemar против **переоценённогоR5** на обоих splits.

Песочница и метрика из24 без изменения: AST-filter names/imports, limited process pool, CPU/wall/per-task timeout, restart при killedworker; RLIMIT_CPU и Linux RLIMIT_AS. Bool→int; конечное число, math.isclose(rel=abs=1e-6). Reference не исправляется вручную. Reference-программы повторно исполняются в Colab; их значения обязаны совпасть с сохранёнными24, знаменатели4096/991 фиксированы.

До train контроль должен дать1602±3/4096 и406±3/991; обеCE в abs tolerance1e-4 от сохранённых24. Несовпадение останавливает обучение для разбора окружения/оценки. Веса/hash/protocol контролируются и при reuse; исходные метрики24 хранятся отдельными `source-*24.json` и не подменяют новую оценку25. GPU/Torch/CUDA/SDPA flags и наблюдаемые SDPA-операторы preflight печатаются; флаг не доказывает фактический backend всех train/inference shapes.

## Checkpoint, resume и передача файлов

Каждый новый полный train сохраняет self-contained `runs/<id>/checkpoint.pt`: FP32model, AdamW, GradScaler, scheduler, RNG, recipe/initial/plan/tokenizer hashes. Формат читается `weights_only=True`. `best-checkpoint.json` указывает выбранный PT в ZIP без дублирования большого файла. Train-time заканчивается до сохранения checkpoint; wall-time включает сохранение. Логируются пары и useful source/target tokens/s, peak memory, padding, finite/AMP-skips.

**У исходного R5 нет optimizer/scaler.** Пользователь разрешил условный повтор его обучения **только при необходимости**. По умолчанию `CONTROL_STATE_POLICY='replay'`: если после LR/D0 выбран исходныйR5, повторить тот же рецепт6250×128 с нуля, сохранить собственный полный checkpoint, оценить и проверить1602±3/CE±1e-4. Это не новый LR-кандидат; результат помечен `R5_replay`, original/replay tensor hashes и фактические метрики печатаются. Новые состояния не приписываются старым весам. При невоспроизведении автоматический экспорт останавливается; исходныеR5 и новый checkpoint остаются вZIP для разбора. Если победил L/D, повтораR5 нет. `weights_only` — явное ручное отклонение от полногоcheckpoint, не default.

Для выбранного полного checkpoint формируется test1919 FP16/b512 с fulltimer: load/restore/tokenizer/H2D/warmup/sort/greedy/order/answers write; download/cache/exec/CE/JSON dump снаружи. JSONL — кандидат длябота, официальный формат ещё открыт.

ZIP обновляется послеtrain/evaluation, содержит входы/генерации/метрики/checkpoint/progress; train-corpus и freshrows исключены. Resume — на границе запусков: законченный train/eval не повторяется; прерванный train начинается с общегоinitial, step-resume не включён. Полный state сохранён для дальнейшего продолжения в отдельном опыте. Manifest/CRC/каждыйSHA/код/протокол/имена и конфликты локальных файлов проверяются. Последняя упаковка самостоятельная, работает доsetup и послеinterrupt; раннийZIP безprotocol не является завершённой границейresume.

Стартовый ZIP25 (`../results/checkpoints/25-start-from-24-r5-and-fixed-plan.zip`, нет в репозитории): **100083340 байт**, SHA `9cb672cb3535aa4d37482da157f60cd3986bffe3127fea6e5a92cc00d73f21b0`; R5+exactinitial24+tokenizer/план/heldout вместо полногоZIP24 (353675202 байт). Хранится локально, вне Git. ДоGPU положить его в `MyDrive/nlp-hw02/`; `INPUT_MODE='auto'` делает reuse или Drive→локальный диск. Для resume в новомruntime указатьZIP25; для уже загруженного файла есть `paths`, браузерный `upload` — опция. Полный исходныйZIP24 также принимается. SHA проверяются до тяжёлой подготовки.

`BACKUP_EACH_BOUNDARY_TO_DRIVE=True` включает проверенную SHA копию ZIP после каждой train/eval-границы; это выбранный пользователем backup, по умолчанию локальный snapshot. Последняя ячейка поддерживает `SAVE_TO_DRIVE=True`, `DOWNLOAD_ZIP=False`. Checkpoint25 с optimizer больше прежнего model-only; дляresume передавать его черезDrive, не повторять браузерную загрузку. Бюджет60–75мин ожидаемый, не измеренный; условный повторR5 добавляет около13мин к текущему пути. Принимаются только полные6250×128, фактический бюджет печатается и не сокращает обучение. Само скачивание/финальный Drive-transfer вне budgettimer.

## Проверки при подготовке25 до T4

**CPU при подготовке:** exact initial/token scaling/R5/tokenizer/heldout/NPY SHA; direct компактный и originalZIP24; реальные tiny train FP32 и FP16+GradScaler; полные model/AdamW/scaler/scheduler/RNG readback и побитовое совпадение следующего update с непрерывным выполнением; все31250 LR-значений пяти возможных peaks совпадают с24. Пороги20/21, три L3-ветки фактических controller-cells (mock безT4), D0-accept/reject, control-CE gate, реальные greedy sorted/order и wrapper/answersSHA/CE, train/eval boundary resume, interrupt, CRC/SHA/неверныйкод/path traversal и самостоятельная упаковка прошли. Финальный smoke: 18 завершённых tiny updates +2 interrupt; полных A-updates и T4-прогонов0. [Протокол проверок](../checks/25-lr-dropout-normalized-embeddings-cpu-checks.json). Colab/Drive API замоканы, локальная копия реальных файлов проверена.

**Оставалось непроверенным до Colab-прогона:** полные6250×128 L1/L2/L3/D0/условныйR5_replay, качество/контрольgreedy/CE, CUDAFP16/fusedAdamW/SDPA на длинном горизонте, пиковая память и60–75мин, реальныеDrive/network/downloadUI. Сохранённые24-метрики и CPUtiny не заменяют эти проверки. Один seed и многократный выбор по4k не оценивают разброс или итоговое качество наfresh; official leaderboard score неизвестен.

## Источники

- Пользовательский протокол25 и разрешение условногоR5 replay только при его победе.
- [Разбор24](fast-recipe-equal-presentations-lr-and-ablations.md), [сводка24](../results/24-fast-recipe-equal-presentations-lr-and-ablations-summary.json), [неизменённый notebook24](../notebooks/24-fast-recipe-equal-presentations-lr-and-ablations.ipynb): параметры, точные входы, модель/песочница/оценка и сохранённыеT4-результаты.
- Публичный датасет `SlavaYus/tinysms_qa_12m`, pinned revision выше; его библиография и AMP/fairseq источники уже учтены в24.

- Пользовательские результаты25: [notebook](../results/25-lr-dropout-after-normalized-embeddings-results.ipynb), [сводка](../results/25-lr-dropout-after-normalized-embeddings-summary.json); исходныйZIP SHA `b0f0cdd628966024e7634169340f989948604cee8c715f0d867eaa24bcd5e65d`. CPU-reexecution сейчас отдельно отT4 training/inference.
