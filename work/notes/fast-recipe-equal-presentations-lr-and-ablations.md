---
type: experiment
course:
  - nlp
status: complete
understanding: 3
created: 2026-10-10
updated: 2026-10-10
sources:
  - https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/tree/b2724689ff079e9c22efd548fe134bfb29036dbb/data/train
  - https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_base.py
  - https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_encoder.py
  - https://docs.pytorch.org/docs/2.11/notes/amp_examples.html
aliases: []
---

# ДЗ2, опыт24: быстрый рецепт при равных предъявлениях

[Colab notebook24](../notebooks/24-fast-recipe-equal-presentations-lr-and-ablations.ipynb) · [реестр](../../JOURNAL.md) · [CPU data](../checks/24-data-selection-cpu-checks.json) · [CPU smoke/resume](../checks/24-fast-recipe-cpu-smoke-checks.json) · [сохранённые результаты23](throughput-batch-amp-bucketing-and-batched-inference.md).

**Получен полный T4-прогон:** ниже — результат24 и его CPU-проверки; последующие разделы сохраняют протокол до запуска. R5 принят, R4 отклонён по заранее заданному порогу.

## Получен результат24: быстрый рецепт и масштаб эмбеддингов

**Вывод:** R5 — текущая лучшая база: exec1602/4096 (39.11%) и 406/991 (40.97%). Контроль №22 тем же FP16/b512 — 820/4096 (20.02%) и 190/991 (19.17%). Лучший LR до абляций — 0.002 (R2); уже R2 проходит правила быстрого рецепта. R6 не требовался и не запускался. R1–R5 завершены на одной T4; каждый получал 800000 пар / 52797347 полезных target-токенов, 6250 updates × batch128 с нуля. Train — прежние 500k/BPE8k без фильтрации; общий initial-state и план. Warmup313 → cosine до 10% пика, AMP FP16/GradScaler, unscale → clip1, fused AdamW.

| Запуск | Настройка | Exec4096 | Exec991 | CE4096 | Train, мин |
|---|---|---:|---:|---:|---:|
| control22 | Контроль22/100k | 820/4096 (20.02%) | 190/991 (19.17%) | 0.7751 | 66.79 |
| R1 | Peak LR0.001 | 739/4096 (18.04%) | 178/991 (17.96%) | 0.8654 | 12.83 |
| R2 | Peak LR0.002 | 916/4096 (22.36%) | 220/991 (22.20%) | 0.7302 | 12.93 |
| R3 | Peak LR0.004 | 535/4096 (13.06%) | 132/991 (13.32%) | 0.7698 | 12.83 |
| R4 | LR0.002, dropout0 | 880/4096 (21.48%) | 207/991 (20.89%) | 0.7793 | 12.16 |
| R5 | LR0.002, token init std1/√d | 1602/4096 (39.11%) | 406/991 (40.97%) | 0.4914 | 12.95 |

В строке контроля время — сохранённое суммарное обучение цепочки21/22 (66.79 мин). Метрика: исполнить reference/generated solve, сравнить конечные числа math.isclose1e-6/1e-6. Старый знаменатель — 991, CEold на 1024. На новых 4096 все reference возвращают число. Оценка всех моделей одинаковая: greedy FP16 sorted batch512/cap512. Вопросы нового heldout не пересекаются с old/test/fresh; fresh1024 только исключён, не оценивался.

- **R2:** +96 совпадений / +2.34 п.п. против №22 на 4k, paired McNemar $p=2.60\cdot10^{-6}$. LR0.001 (R1) не проходит пороги: 17.96% на 991, потеря 1.98 п.п. на 4k. LR0.004 (R3) хуже по exec, хотя CE лучше R1. Все loss/grad finite, AMP skips — 0; признаков взрыва или пропущенных updates нет.
- **R4:** dropout0 даёт 880 против 916 у R2: −0.88 п.п., хуже разрешённых −0.5 п.п.; отклонён по протоколу. McNemar $p=0.0943$: доказательства вреда на уровне 0.05 нет. Порог принятия и статистическая значимость — разные выводы.
- **R5:** изменён только initial token-table: те же Gaussian draws делятся на $\sqrt{256}=16$, множитель embedding $\sqrt d$ и positions прежние; dropout0.1, LR0.002. Token std $d^{-1/2}$ при scaling $\sqrt d$ даёт начальный элементный масштаб ≈1 вместо ≈16. +686 совпадений / +16.75 п.п. против R2: 782 gains / 96 losses, McNemar $p=1.95\cdot10^{-134}$. Против №22: +782 совпадения / +19.09 п.п., 856 gains / 74 losses, $p=1.72\cdot10^{-169}$. Изменение масштаба входа помогает в этой конфигурации; конкретный механизм — гипотеза, отдельно не измерялся. Общие tensors и изменение только token scaling подтверждены локально. LR-сетка при новой инициализации R5 ещё не проверялась.

**Скорость, сохранённая T4:** R5 train776.91 с / 12.95 мин, 1029.72 пар/с, 67958 полезных target-токенов/с; 5.16× быстрее сохранённого №22 при равных предъявлениях. Peak allocated7.64GB / reserved13.30GB; во всех R1–R5 skips0 и finite loss/grads. Padding30.22%. Reserved выше краткого benchmark23 (~8.98GB); полного долгого запуска B/C нет. Основной прогон №24 до финальной упаковки/скачивания: 73.93 мин от первой ячейки, 73.07 мин после ввода; бюджет90мин соблюдён. Вход компактного ZIP через Drive — 24.33 с; разовая загрузка в Drive вне этого таймера.

**Инференс R5:** test1919 full12.16 с (чистая генерация11.40 с), FP16 sorted batch512/cap512. Таймер включает model-state load/restore/tokenizer/H2D/warmup/order restore/answer write; corpus download/cache/exec/CE/JSON dump и скачивание ZIP снаружи. Ответы в исходном порядке; EOS1903/cap16, AST1907, 1910 разных программ. Официальный score неизвестен; JSONL — внутренний формат, схему бота нужно подтвердить.

| Reference content BPE | Количество | Контроль22 | R2 | R5 |
|---|---:|---:|---:|---:|
| <60 | 2013 | 34.77% | 38.70% | 59.36% |
| 60–90 | 1457 | 7.28% | 8.72% | 24.37% |
| 90–120 | 488 | 2.25% | 1.64% | 10.04% |
| ≥120 | 138 | 2.17% | 1.45% | 2.17% |

Длинные reference≥120 остаются слабым местом: 3/138 (2.17%) у R5. Из 2494 несовпадений на 4k, 2314 дают конечное число, отличающееся от reference; остальные 180 — нечисловые/ошибочные программы. Следующее улучшение должно оцениваться по корректности результата. Один seed и выбор по 4k не оценивают разброс и качество на fresh; old991 уже многократно использован. Против №21 на новом 4k контроля нет: пользователь предоставил только №22. Batch/AMP/bucketing/LR вместе отличаются от старой базы; отдельного причинного эффекта каждого ускорения №24 не выделяет. R5 vs R2 изолирует token init при этом LR.

**Проверки сейчас на CPU:** 95 файлов CRC/manifest/SHA, исходный код и recipe, plan/data hashes/unique/overlap, 6250 updates/800k counts/LR, FP32 model-state strict/finite (control/R1–R5/initial), изменение только token scale в R5, decoded tokens/EOS/order/JSONL/SHA, bins/McNemar/decisions прошли. Переисполнены 35840 reference/generated программ (32404 уникальные): все численные reference и все 12 scores (6 моделей × 2 splits) совпали точно за 42.12 с. Новых LM-updates/GPU/greedy/CE-прогонов — 0; CE и времена — сохранённые Colab outputs. На macOS RLIMIT_AS не применяется, CPU/wall limits применяются.

Файлы: [ноутбук с outputs](../results/24-fast-recipe-equal-presentations-lr-and-ablations-results.ipynb), [сводка](../results/24-fast-recipe-equal-presentations-lr-and-ablations-summary.json), [статические проверки](../checks/24-fast-recipe-equal-presentations-lr-and-ablations-results-static-checks.json), [CPU exec](../checks/24-fast-recipe-equal-presentations-lr-and-ablations-results-exec-cpu-checks.json). Raw JSON в исключённом из Git `work/results/generations/24-fast-recipe-equal-presentations-lr-and-ablations/`; полный исходный ZIP в `work/results/checkpoints/24-fast-recipe-equal-presentations-lr-and-ablations-results.zip`. Лучшая модель и tokenizer отдельно: `work/results/checkpoints/24-r5-scaled-token-init/`, также вне Git. PT содержат только model-state, AdamW/scaler отсутствуют; для продолжения обучения потребуется заново создать optimizer. Следующий опыт обсуждается с наставником, нового notebook нет.



## Гипотеза и мотивация

A4+4/d256/h4/SwiGLU680 (12086912 параметров) с AMP FP16/batch128/bucketing при800k предъявлениях достигает ≥18% exec-match на старых991 и тратит около1/5 прежнего времени. Переход batch8→128 уменьшает число обновлений100000→6250, поэтому качество нужно проверить, а не переносить вывод по throughput.

Сохранённый23: A/AMP/b12865289.57 полезных target-ток/с,988.31пар/с,5.65× контроля; peakreserved8.98GB. При нашем объёме это ориентир около13.5 мин train на запуск; фактический T4-замер24 ещё не выполнен. Сохранённая цепочка21→22 до100k:4007.49с=66.79мин train. В таблице24 печатается отношение текущего train-time к этим сохранённым4007.49с; это межзапусковое сравнение, а не новая парная проверка скорости.

Train остаётся500k/BPE8k безreference-фильтрации. B/C, новые train-данные, beam иsampling отложены до25+.

## Фиксированные данные и оценка

- Revision`b2724689ff079e9c22efd548fe134bfb29036dbb`, прежние hashes500k/locations/old1024 и tokenizer22 проверяются до обучения. ЗагрузкаZIP22 илиPT22/100k+tokenizer, необязательноZIP21/PT21/100k. Legacy PT читается только после проверки закреплённого SHA; в рабочемZIP сохраняется толькоmodel-state, безoptimizer/RNG.
- Новыйheldout4k: физический порядокshards3→4→5 и строк; прежнее`is_validation(q)` сSEED42 (SHA256(`42:q`), первые8 bytes big-endian,mod10==0), question≤512/code<512 UTF8 bytes. Вопросы уникальны, нетtest/old1024/fresh1024, reference даёт конечное число. Отбираются первые4096 подходящих строк. Если первый reference вопроса невалиден, поздний валидный дубликат может быть принят. Исполнитель/AST/timeout/RLIMIT/pool-restart тот же, что23; это ограниченная числовая песочница.
- Fresh1024 используется только как read-only набор исключаемых вопросов. Проверен прежний pairsSHA`eaf5f7c00559d6b1e5650a0dd879e5b4a214931ee803f8c1f6efd956c54971b2`; fresh не кодируется, не исполняется, не сохраняется заново и не участвует в выборе.
- Новый CPU-отбор: достаточноshard3;4096 numeric/4096 distinct,0 пересечений. PairsSHA`6b0d0d559a56921faac1842e3508889225af1fe1ad8e04de378840bc4df8b047`, JSON-fileSHA`28a3c0c9088c5b4a17d75d216e4abc4262cc9ca62884966c43e7ec09b6f040c2`, locationsSHA`f8c41a5c601f4710cdf9bccd769adb61052e9a28fa87086aaf1d2678b13dccb1`. Colab обязан воспроизвести pairsSHA; reference исполняется заново, знаменатель4096 обязателен. JSON, execution-values и план сохранены локально в исключённом изGit `work/results/generations/24-fast-recipe-equal-presentations-lr-and-ablations-*`.

| Новый shard | Размер bytes | SHA256 |
|---|---:|---|
|3|26167481|`b2b63db46b005e229aa5e91f1010adc763ca9f4b806561b328432fbc9e5c11e9`|
|4|26185069|`272ac87a686134a3e3ba6488985879bd8ea567dcaba1e1145aae0cb724a0afe5`|
|5|26168730|`89687321a1df7338e543a7d8c46d7bc0bd6596d855f9d3e43995e2a9cff858b7`|

Контроль22/100k и, если предоставлен,21/100k оцениваются тем же способом: FP16 sortedbatch512/preallocated/KV/EOS-check16/greedycap512. Контроль22 наold991 должен дать190±3 передtrain;189/991 из23 относится кFP32, FP16/b512 дал190. До запуска24 новый4k контроль ещё не был измерен; теперь его820/4096 приведены выше.

Для каждого запуска: exec4k (основной), execold991, CE на4k/old1024, точность по длине reference и парный exactMcNemar против текущего22 на обоихsplits. Exec вызывает`solve()`, bool→int, только конечное число;`math.isclose(rel=abs=1e-6)`. Gold — результатreference, без математического исправления. CE — FP32/token-weighted, PAD исключён, EOS включён; oldCE на всех1024, oldexec только991. Bins — BPE content безEOS:0,60),[60,90),[90,120),[120,∞).

## Равные предъявления и запуски

Один план на800000 пар: chunk4096 случайных train-indices, сортировка внутри по source+target length, затем shuffle батчей; seed230042, как23. R1–R5 получают один6250×128 план. R6 — reshape того же плоского списка в25000×32. Различия порядка/объёма междуR1–R6 исключены; с историческими21/22 совпадает число предъявлений, их прежний random.sample план не копируется.

**Проверено на CPU:** source38706084,target52797347 полезных токенов (sourceBOS/EOS,targetEOS,PAD исключён). NPY SHA`48734ce4d9621e0adf464e5fad25bfc625772d7a222baeada107ea92c0a13628`; проверяется вColab. Не добавляются дополнительные timing-warmup батчи.

| Запуск | Peak LR | Dropout | Batch×updates | Изменение |
|---|---:|---:|---|---|
|R1|.001|.1|128×6250|Базовый быстрый рецепт|
|R2|.002|.1|128×6250|ТолькоLR|
|R3|.004|.1|128×6250|ТолькоLR|
|R4|ЛучшийR1–R3|0|128×6250|Независимая абляцияdropout|
|R5|ЛучшийR1–R3|.1|128×6250|Независимая абляцияtoken-init-scale|
|R6 условно|ЛучшийLR/2|.1|32×25000|Если лучшийR1–R3<18% на991|

Общие: Pre-LN/finalLN, learnedpositions2048, vocabulary8192/sharedtokenembedding/untiedhead, SwiGLU680; AdamW wd.01/betas(.9,.999),fused если доступен (одинаковый режим во всех запусках),clip1. AMP FP16+GradScaler init_scale1024, unscale доclip. Seed42 и сохранённый общийinitial-model-state; передtrain каждого запуска RNG сбрасывается. Пропущенный AMP update или nonfinite loss/norm делает запуск непригодным для решения; дополнительные предъявления не компенсируют skips. OOM/interrupt не создаёт completion-receipt.

R5 сохраняет исходныеGaussian token draws и делит только token-table на$\sqrt d$ ($d=256$); это std$d^{-1/2}$ при прежнем множителе$\sqrt d$, то есть элементный масштаб≈1 вместо≈16. Positions и все прочие начальные tensors прежние; изменяется и соотношение token/position сигналов. У fairseq заимствовано это правило token-init/scale, остальные блоки остаются нашими. R4/R5 не объединяются.

LR для update$s=1\ldots T$, peak$p$, warmup$W=\lceil .05T\rceil$:

$$\mathrm{LR}(s)=\begin{cases}ps/W,&s\le W,\\.1p+.45p\left(1+\cos\frac{\pi(s-W)}{T-W}\right),&s>W.\end{cases}$$

$T=6250,W=313$ для128; $T=25000,W=1250$ для32. Первый update имеет$p/W$, update$W$ —peak, последний —$.1p$. Warmup входит в фиксированное число предъявлений; рестартов нет. Scheduler реализован явнымLR каждого update.

## Правила решения до запуска

1. ЛучшийLR: максимумexec4k среди полных численно корректныхR1–R3; при равенстве меньшийCE4k, затемID. УсловиеR6 смотрит именно этот лучшийR1–R3, независимо от абляций.18% на991 означает минимум179 совпадений.
2. Лучший запускR1–R6 выбирается тем жеexec/CE правилом. Рецепт принят приold991≥18% и потере на4k против22≤1.5п.п. (максимум61 совпадение). Дополнительно отдельно печатается принятие лучшегоbatch128: успех толькоR6 не подтверждает гипотезу быстрогоbatch128.
3. R4/R5 сравниваются с лучшимLR: допускается потеря≤.5п.п. на4k (максимум20 совпадений), явный выигрыш≥1п.п. (минимум41). McNemar против лучшегоLR также печатается. Пороговое принятие не равнозначно статистической значимости; одинseed/выборпо4k не оценивает разброс качества. Fresh сохраняется для будущего финального сравнения.
4. Таблица: пары/useful source/target/с, counts, train/wall-time, peak memory, AMP-skips, exec/CE/McNemar. Дляbest —test1919 в исходном порядке, FP16/b512, full timer отmodel-state load доanswers write; вне него download/cache/exec/CE/полныйJSON dump. JSONL — внутренний формат, схема бота ещё неизвестна.

## Resume, ZIP и бюджет

Послеtrain — толькоmodel-state PT +training-receipt/sha; послеevaluation —генерации, execution, метрики, receipts. ZIP обновляется на обеих границах, содержит tokenizer/controls/initial-state/план/heldout/JSON/модельные веса, безoptimizer/scaler/train-data. PT читаются сweights_only=True, кроме исходных строгоhash-verifiedlegacy21/22.

В той же сессии повтор ячеек пропускает завершённые этапы. В новой — загрузитьZIP24 (ZIP22 заново не нужен). Проверяются manifestCRC/каждыйSHA, имена файлов/пути/код/протокол иизначальныйSHA весов. Незавершённыйtrain повторяется сinitial-state; завершённыйtrain с незавершённойevaluation продолжается без переобучения. Генерации/таймеры привязаны кSHA модели/tokenizer. Последняяячейка упаковки самостоятельная, работает послеinterrupt идоsetup; такой раннийZIP ещё не содержит обученных границ. Автоматических многократных download-диалогов нет, ZIP скачать последнейячейкой.

Ориентир90мин, сR6~115; это ожидаемый бюджет, не сокращение запусков дедлайном. Логируются elapsed отfirstcell и activewall послеupload отдельно; фактический укладываемый бюджетT4 неизвестен. GPU/Torch/CUDA/SDPA-флаги иobservedcontrolSDPA печатаются; включённыйflash-флаг не доказываетFlashAttention. МежверсийнаяCUDA-побитовая воспроизводимость не обещается; runtime sessions сохраняются.

## Передача файлов: подготовка до T4

Проверено 2026-10-10: исходный ZIP22 занимает 390,68 МБ и содержит три полных checkpoint примерно 146,27 МБ каждый. Для входа24 подготовлен [24-start-from-22-model-only.zip (`../results/checkpoints/24-start-from-22-model-only.zip`, нет в репозитории), 48,51 МБ: control22/model-state, токенизатор, source.json и protocol/manifest. Архив хранится локально и исключён из Git. Он содержит исходные веса для контроля; результатов R1–R6 ещё нет. SHA256 компактного ZIP: `5b70c8adebc8affaa20b2aa7254fbfc5eb048c0371b0ad286130fbed2cd85de8`. Каждый tensor точно совпадает с исходным22/100k; SHA исходного PT и токенизатора проверены до чтения legacy PT.

В notebook24 `INPUT_MODE='auto'` проверяет и использует уже подготовленные входы runtime; при отсутствии открывает загрузку. `drive` монтирует Google Drive по выбору пользователя и копирует только `INPUT_PATHS` на локальный диск; `paths` принимает уже загруженные файлы. По умолчанию ожидаемый путь Drive — `/content/drive/MyDrive/nlp-hw02/24-start-from-22-model-only.zip`. Исходные ZIP21/22, PT+tokenizer и последний resume ZIP24 по-прежнему принимаются. В новой сессии для продолжения выбрать ZIP24 с результатами; стартовый компактный ZIP содержит control22/tokenizer. Общий SHA кода модулей `35532789452eed2ff8e60943c291870bcf606261fd9fbeb8dae9045d6d190779` прежний, поэтому ранее сохранённые ZIP24 совместимы.

Последняя ячейка поддерживает `SAVE_TO_DRIVE=True` и `DOWNLOAD_ZIP=False`: готовый ZIP копируется в выбранную папку Drive с проверкой SHA. Резервная копия создаётся при выполнении последней ячейки; автоматические snapshot между запусками по-прежнему локальные. Train/generation работают на локальном диске. Google рекомендует сократить частые операции с монтированным Drive и копировать архивы на VM; скорость Drive зависит от размещения и квот, моментального копирования не обещаем.

[Новые CPU-проверки](../checks/24-colab-input-transfer-cpu-checks.json): manifest/SHA и тождественность tensors, paths, autoreuse без диалога, legacy PT/tokenizer, сохранение прежних resume-файлов, отклонение повреждённого ZIP и модели, самостоятельная упаковка до setup и копирование в Drive без download. Вызовы Colab/Drive замоканы; реальное подключение Drive, сеть, браузер и T4 не проверены. MODULE_SOURCES и ячейки обучения/оценки не изменены, новых updates и GPU-прогонов — 0.

## Проверки при подготовке24 до T4

**CPU при подготовке:** прежние500k/old/fresh hashes иновый4k полностью воспроизведены, numeric4k/дедуп/overlap/NPY-counts/SHA проверены. Реальныеhashed22 PT+tokenizer прошли strict-load иконвертациюmodel-state безupdates. Tiny1+1/d16/h2/FFN24/vocab37:12 завершённых updates (FP32 иCPU FP16/GradScaler),2 шага искусственногоinterrupt; full/KV maxerror$2.38\cdot10^{-7}$. R4 общиеweights иR5 толькоtoken-scaling, LRendpoint/целочисленныепороги/McNemar/bins, реальныеsort/restore/individualgreedy, resume послеtrain ипослеevaluation безновыхupdates прошли. Mockcontroller провёлR1–R6 с правильнымиpeak/dropout/scale/reshape. ZIP CRC/SHA/modeltensors, повреждённыйSHA/неверныйкод/pathtraversal исамостоятельнаяпоследняяячейка доsetup/послеinterrupt проверены сmockdownload.9code cells parse, outputs пустые.

**Оставалось непроверенным до Colab-прогона:** полные6250×128/25000×32, стабильностьLR.001/.002/.004 иR4/R5, качество4k/991/CE, контроль22/21 на новом4k, FP16/CUDA/fused на длинном горизонте/редкие длинныебатчи, реальные времена/90–115мин/ZIP-downloadUI, новый4k вLinuxAS-limit. Скорости23 не заменяют эти проверки; полногоA-обучения24/новогоGPU-замера локально0.

## Источники

- Пользовательский протокол24; [сводка23](../results/23-throughput-batch-amp-bucketing-and-batched-inference-summary.json), [разбор23](throughput-batch-amp-bucketing-and-batched-inference.md), [сводка22](../results/22-cosine-tail-from-75000-to-100000-steps-encoder4-decoder4-qwen-bpe-8k-swiglu-500k-pairs-summary.json). Числа23/21/22 — сохранённые T4-результаты.
- [Pinned public corpus](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/tree/b2724689ff079e9c22efd548fe134bfb29036dbb/data/train): публичныеHub/LFS SHA3–5 сверены2026-10-10, реальныйshard3 скачан/проверен, новыеtrain-данные не добавлены.
- Fairseq [Embedding](https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_base.py), [encoder scale](https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_encoder.py), [decoder scale](https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/transformer/transformer_decoder.py): первичныйкод прочитан2026-10-10;normal std$d^{-1/2}$/scale$\sqrt d$, позиции внашемR5 прежние.
- PyTorch2.11[AMP/GradScaler/unscale/clip](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html). AMP/bucketing, Pre-LN, Абсолютные позиции, KV-кеш. Отдельной связанной заметки DL1/knowledge об этом token-init при проверке индексов не найдено.

- Google Colab [FAQ: Drive I/O](https://research.google.com/colaboratory/faq.html), [drive.py](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py), [files.py](https://github.com/googlecolab/colabtools/blob/main/google/colab/files.py). Проверено 2026-10-10: mount, загрузка файлов, локальное копирование из Drive.

- Пользовательские результаты24: [ноутбук](../results/24-fast-recipe-equal-presentations-lr-and-ablations-results.ipynb), [сводка](../results/24-fast-recipe-equal-presentations-lr-and-ablations-summary.json), оригинальный ZIP SHA`0a184d0a86a34098d2313926a2f83448e21c5972abd55fecc96a28a8cfebaae1`. Текущие CPU-проверки перечислены в разделе результата.
