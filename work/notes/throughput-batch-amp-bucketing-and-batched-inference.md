---
type: experiment
course:
  - nlp
status: complete
understanding: 3
created: 2026-10-10
updated: 2026-10-10
sources:
  - https://huggingface.co/datasets/SlavaYus/tinysms_test_prompts
  - https://huggingface.co/datasets/SlavaYus/tinysms_train_qa
  - https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/tree/b2724689ff079e9c22efd548fe134bfb29036dbb/data
  - https://docs.pytorch.org/docs/2.11/notes/amp_examples.html
  - https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html
  - https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html
  - https://docs.python.org/3/library/multiprocessing.html
  - https://docs.python.org/3/library/resource.html
aliases: []
---

# ДЗ 2, опыт 23: batch, AMP, bucketing и batched greedy

[Colab-notebook23](../notebooks/23-throughput-batch-amp-bucketing-and-batched-inference.ipynb) · [реестр](../../JOURNAL.md) · [CPU exec-match](../checks/23-exec-match-cpu-checks.json) · [CPU-аудит 50k](../checks/23-train50k-reference-exec-audit.json) · [CPU smoke/static](../checks/23-throughput-cpu-smoke-checks.json).

**До запуска:** это замер скорости, не поиск качества обучения. Случайные A/B/C получают короткие обновления и удаляются.22/100k используется только для greedy; beam, sampling и длинного обучения здесь нет. Старые notebooks не меняются.

## Метрика и исходный контроль

**Источник:** публичные карточки test_prompts/train_qa описывают gold как исполненный reference, выбрасывают строки без числового ответа; test2000→1919. Это подтверждает принцип метрики, но не даёт точный код официального scorer/формат посылки. Карточки не заменяют прежний pinned training dataset.

**Принятое правило:** reference и генерация вызывают `solve()`; допустим конечный числовой scalar, bool→int. Сравнение $|g-r|\le\max(10^{-6}|r|,10^{-6}|g|,10^{-6})$ соответствует `math.isclose(g,r,rel_tol=1e-6,abs_tol=1e-6)`: $g$ — ответ генерации, $r$ — исполненный reference. Число знаменателя фиксировано 991 из 1024. Ручной 74 помогает разобрать ошибки, не заменяет leaderboard-метрику; математически неверный reference всё равно является gold.

**Сохранённые пользовательские T4-генерации, вновь исполненные сейчас на CPU; новых LM-updates0:**

| Опыт/шаг | Exec-match /991 | Доля |
| --- | --- | --- |
|18/100k|151|15.24%|
|19/100k|145|14.63%|
|20/100k|184|18.57%|
|21/50k|156|15.74%|
|21/75k|176|17.76%|
|21/100k|193|19.48%|
|22/100k|189|19.07%|

Все счёты воспроизведены точно,991 finite/33 non_number reference.21 добавил 37/991≈3.73 п.п. между 50k и 100k; плато не доказано. Парный exact McNemar20–21/20–22/21–22: $p=0.448/0.688/0.716$. Малые разницы не позволяют выбрать надёжного победителя. Числа /888 в прежних разборах — исторический ограниченный AST-proxy, не текущая метрика.

Прежний train: batch8, FP32, около 206 пар/с и 38%padding (ориентир 22). Batch никогда не менялся; AMP16 проверяли только на batch8, оттуда нельзя вывести поведение больших batch.

## Гипотеза и фиксированный протокол

**Гипотеза:** большие батчи лучше загрузят T4; bucketing уменьшит padding; AMP может дать выигрыш при больших размерах/моделях. Это предположения. Изменения в основной матрице совместные (batch+bucketing+fused), поэтому отдельную причину ускорения она не выделяет. FP32/AMP получают один и тот же список батчей; A/AMP/batch8 отдельно сравнивается с контролем.

Одна T4, текущий Colab Torch/CUDA записывается, tokenizers0.23.2; pyarrow/sympy также записываются. Печатаются GPU/driver/память/temperature/clocks, enabled SDPA-флаги и фактически наблюдаемые profiler-операторы. Backend не форсируется; наблюдение отдельных shapes не гарантирует один backend для всех батчей.

Вход: исходный ZIP22 или PT22/100k+tokenizer JSON. Фиксированные SHA из [сводки 22](../results/22-cosine-tail-from-75000-to-100000-steps-encoder4-decoder4-qwen-bpe-8k-swiglu-500k-pairs-summary.json):

| Файл | SHA256 |
| --- | --- |
|Original ZIP22|`4db40c0313f10afbf11d4a76beb2de59a729e1af9c6d40f08240c6f55697c615`|
|step100000.pt|`5d115cf06c55f8d6245b5244d4cd622e38480fd5ef7dfe8806ba8bbcd70d4833`|
|latest.pt22/100k|`13f8ae4ae2ee2c15fc9d034cf8773de420226e073550c9774adb1846f3e27b23`|
|tokenizer.json|`261f25b5d4be006a6a00e41e0eafbe522580741da627ba670f6b9805a2659b7d`|

SHA проверяются до `torch.load` и данных. Архив не распаковывается целиком: выбираются проверенные PT/JSON; содержимое не определяет доверенные hashes. Данные — прежняя revision`b2724689ff079e9c22efd548fe134bfb29036dbb`, shards0–2. Воспроизводятся hashes train500k/locations/heldout1024 и отсутствие overlap с test. BPE-кеш строится заранее; filtering reference в 23 не применяется. Fresh1024 не оценивается.

### Исполнение программ и аудит данных

Импортируемый worker-модуль +spawnPool, не notebook-функция. AST запрещает опасные имена/атрибуты/импорты; числовые builtins, math и ограниченный числовой SymPy. RLIMIT_CPU2 с на программу (soft limit с учётом накопленного CPU worker), wall1 с, родительский timeout3 с; Linux RLIMIT_AS768MiB. Без чтения/записи файлов программой. Это ограниченный числовой исполнитель, не изолированный OS-контейнер.

Не `Pool.map`: максимум по одному inflight-заданию на worker, per-task deadline. Убитый worker может быть автоматически заменён Pool, а его async-result потерян; на deadline весь пул уничтожается/создаётся заново, завершённые результаты сохраняются, соседние незавершённые задания повторяются. Порядок/дубликаты восстанавливаются. Заблокированный или timeout reference не доказывает, что исходный reference не имеет числа.

**Новый CPU-аудит:** равномерно 50k сырых строк по 16667/16667/16666 из shards0/1/2, seed230042+shard; это выборка исходных train-шардов, не только отфильтрованного train500k. Hash locations`e817d6d4aeca88e6dd04f964bb0edea81640beaf730ecf320f814f740d82ffd4`.

- 48210 finite (96.42%);
- 1727 non_number +3 runtime-error (3.46%);
- 60 AST-blocked (54 imports/6 names,0.12%), истинный результат неизвестен;
- совместно 1790 не дали число:3.58%; CPU-exec36.58 с/49561unique, pool restarts0.

Пользователь разрешил будущую фильтрацию строк без конечного reference; неизвестные AST-случаи учитывать отдельно. Новые таргеты не создаются. В notebook записан сделанный CPU-аудит; `REPEAT_TRAIN50K_AUDIT=True` повторяет ту же выборку на CPU Colab в общем бюджете. Heldout-reference исполняется заново всегда и обязан дать 991.

### Обучение: скорость

| Модель | encoder+decoder | d/head/FFN | Параметры, CPU meta-count |
| --- | --- | --- | --- |
|A|4+4|256/4/SwiGLU680|12086912|
|B|6+6|384/6/SwiGLU1024|31940096|
|C|6+6|512/8/SwiGLU1368|53643328|

C hidden≈1365 округлён до 1368, кратности 8. Остальное общее: Pre-LN/finalLN/learned positions2048/vocab8192/sharedinputembedding/untiedhead/dropout.1.

20 строк: A/FP32/batch8/random.sample/ordinaryAdamW контроль; A/AMP/batch8 на тех же батчах;18 сочетаний A/B/C×FP32/AMP×batch32/128/256 с bucketing/fusedAdamW если проверка доступности прошла. AdamW LR1e-4/WD.01/clip1, seed230042, свежие веса/optimizer на каждую строку; это не continuation22. Чанки 4096 случайных пар сортируются по source+target длине, разбиваются на батчи, затем батчи перемешиваются. Планы общие по моделям/precision и сохраняются через SHA.

50 прогревочных (прогрев таймера, не LR) +200 измеренных шагов. AMP: FP16autocast/GradScaler, unscale перед clip. Измерение CUDA-sync по 20-шаговым чанкам включает collate/H2D/loss/backward/clip/update/scaler; первый SDPA-profile находится в прогреве, вне 200. Warmup sync по 10 шагов для контроля дедлайна. Нет scheduler/accumulation.

Регистрируются мс/шаг, пары/с, полезные source/target/суммарные токены/с, source/target/общий padding, peakallocated/reserved, конечность loss/norm и AMP-skips (one-device found_inf у GradScaler, записьна GPU/чтениепослетаймера; поддержкадиагностического API проверяется). Полезный source содержит BOS/EOS, targetEOS; PAD исключён. Память учитывает весь запуск от создания модели; OOM и неполный замер получают статус и не прерывают очередь. CPU-таймер не выдаётся за T4.

### Инференс: скорость и сохранность

Фиксированные 22/100k, greedy/cap512. Контроль: исходный порядок/batch32/FP32/прежний grow-output+KV-path; heldout1024 и test1919. Обязан дать 189±3/991, иначе остановка до train-сетки.8 вариантов: сортировка по длине вопроса +batch64/128/256/512×FP32/FP16; тот же KVdecoder, предаллоцированный output вместо cat, CPU done-check раз в 16 шагов. Уже завершённые строки продолжаются PAD и обрезаются по EOS. Исходный порядок восстановлен; FP16 — halfweights, без autocast-training/GradScaler.

Для каждого split веса/токенизатор загружаются заново. Два отдельных таймера:

1. Старый: encode/sort/H2D/encoder/KVdecode/CPUdecode/AST/ref-token diagnostics; вне него load/warmup/exec/файл.
2. Полный wall: начало`torch.load`→restore/tokenizer/cast/H2D→warmup→тот же generation→answersfile. Данные/кеш заранее,exec scoring позже. Отдельный SDPA-preflight вне обоих таймеров.

Плюс сумма decoder-итераций по батчам, batch×iterations, token-identical fraction к контролю для обоих splits, exec-match heldout; timing не сравнивать с общим временем Colab-подготовки. Файл ответов — внутренний JSONL; формат посылки бота остаётся открытым. Все heldout-генерации и execution-details сохраняются; test-generations — контроль и выбранный вариант, partialZIP сохраняет уже завершённое.

## Правила решения, закреплённые до T4

- Train: максимум полезных **target-токенов/с** среди 200 полных measured, finiteloss/grads, AMP measured-skips0, оба peak<14e9 байт. Печатаются глобальный и отдельные A/B/C победители. Суммарные source+target/с также видны; парам/с не позволено подменить useful-throughput.
- Inference: минимум testfullwall среди завершённых обоих splits, denominator991, падение execaccuracy≤0.005 от контроля (максимум 4 потерянных matches). Совпадение tokencount не обязательное условие; оно диагностическое. Полный wall<180 с выводится отдельно.
- Бюджет 60 мин от первойячейки, последний 1 мин на ZIP; после раннего greedy-контроля train имеет phase-deadline с 20 мин резервоминференсу. Дедлайн/проекция не уменьшают 50+200: configs OOM/budget/partial явно не кандидаты. Все 20/9 строк регистрируются при штатном проходе, и вывод показывает полноту; скорость T4 до запуска неизвестна, полнота сетки зачас не обещана.

После каждого split/config записываются JSON; упаковка независима от успешного прохождения ноутбука. Один ZIP с CSV/JSON/control-besttest/heldout/manifestCRC+SHA, без весов/данных/исходного ZIP. При прерывании выполнить последнююячейку; `partial_archive` и фаза сохранятся.

## Проверки до запуска T4 (история подготовки)

**Новые CPU-проверки:** exact151/145/184/193/189 и 156/176, аудит 50k; реальные SHA/PT+JSON/ZIP и отклонение плохого SHA; параметры metaA/B/C; строгая загрузка A22 и 3 сохранённых greedy-префикса 64; сортировка/restore на 3 реальныхстроках; tiny1+1/d16/heads2/FFN24/vocab37 logitsfull/KV(maxerror2.38e-7) и control/preallocatedgreedy.5tinyFP32-updates, отдельно 13tinyFP32+13tinyAMP-updates проверили optimizer/GradScaler/finite/clip. Это smoke, не оценка обучаемости/качества A/B/C; реальные 22-веса 0updates.

Искусственное убийство worker →1 restart/восстановление остальных/дубликатов. Fixtures bool/non-finite/import/dunder/timeout; deadline/OOM-nextconfig; selection14GB/200steps/AMPskips и 4vs5 потерянных matches; реальная последняя ZIP-ячейка до setup/после interrupt/после finish с mockdownload, CRC/SHA/noPT. Управляющие notebook-ячейки 9/11/13/15 прошли 20/9 строк и interruption с mockGPU/model/metriccalls: это проверка проводки, не GPU-замер.

**Не проверено на T4:** фактические скорости/CUDA-sync/SDPA-backends/FP16CUDA+GradScaler/fusedAdamW; OOM/padding/memory больших batchB/C; полный новый cap512greedy1024/1919 и 189±3gate; вся 20/9 сетка в 60 мин; LinuxRLIMIT_AS и реальные Colabupload/downloadUI. Текущий локальный Torch2.14.1CPU отличается от Colab, поэтому контроль обязателен. Неизмеренные числа не подставлены.


## Полученные результаты T4 и текущие проверки

**Сохранённый запуск пользователя:** Tesla T4, Torch2.11.0+cu130/CUDA13.0; 45.49 мин от первой ячейки до ZIP, бюджет60 мин соблюдён. Все9 вариантов инференса завершены. Train:9 успешных,5 OOM,1 остановлен в прогреве,5 не начаты по бюджету. Полной сетки20 нет. Фактический SDPA-оператор в профилях — `aten::_scaled_dot_product_efficient_attention`, в train также backward; включённый flash-флаг не означает использование FlashAttention.

[Присланный notebook](../results/23-throughput-batch-amp-bucketing-and-batched-inference-results.ipynb) · [сводка с параметрами/всеми строками](../results/23-throughput-batch-amp-bucketing-and-batched-inference-summary.json) · [train CSV](../results/23-training-benchmark.csv) · [inference CSV](../results/23-inference-benchmark.csv) · [новая CPU-проверка](../checks/23-throughput-results-static-checks.json). Полные JSON/генерации/execution/answers/manifest сохранены локально в `work/results/generations/23-throughput-batch-amp-bucketing-and-batched-inference/`, исключены из Git. Веса23 не создавались для дальнейшего обучения.

SHA notebook `4e17849e9a9897c640c300a6aa8fdc6d3a5e61059fd60ac2d7cd3c4bf39b4f97`; ZIP `f12714c49cc56298e1888b480bfb77a3268b13f0d452c0b2e22bca256d2123ba`. Вход latest22/100k и tokenizer совпали с закреплёнными SHA до подготовки данных; pinned revision и hashes500k/1024 также прежние. Reference-фильтрация не применена, fresh1024 не оценивается.

### Обучение: завершённые замеры

Случайная инициализация каждой модели; 50 прогревочных +200 измеренных updates. A=4+4/d256/h4/SwiGLU680,12.087M; B=6+6/d384/h6/SwiGLU1024,31.940M; C=6+6/d512/h8/SwiGLU1368,53.643M. LR1e-4/clip1; AMP с GradScaler и unscale до clip. Random8 использует обычный AdamW, bucket32/128/256 — fused. Все9 успешных строк: конечные loss/norm,0 AMP-skips.

|Модель/режим|Batch пар|мс/шаг|пар/с|Полезный target, kток/с|Полезный source, kток/с|Общий padding|Peak allocated / reserved, GB|
|---|---:|---:|---:|---:|---:|---:|---:|
|A FP32 random|8|45.50|175.8|11.55|8.49|37.86%|0.62 / 0.98|
|A AMP FP16 random|8|53.61|149.2|9.80|7.21|37.86%|0.49 / 0.69|
|A FP32 bucket|32|82.11|389.7|25.73|18.80|23.79%|2.34 / 2.81|
|A AMP FP16 bucket|32|53.39|599.3|39.57|28.91|23.79%|1.61 / 1.84|
|A FP32 bucket|128|324.95|393.9|26.02|19.08|30.11%|9.19 / 14.37|
|A AMP FP16 bucket|128|129.51|988.3|65.29|47.87|30.11%|6.10 / 8.98|
|B FP32 bucket|32|199.31|160.6|10.60|7.75|23.79%|4.22 / 5.36|
|B AMP FP16 bucket|32|89.05|359.3|23.72|17.34|23.79%|2.87 / 3.31|
|B AMP FP16 bucket|128|289.07|442.8|29.25|21.45|30.11%|10.39 / 13.50|

**Выбор по исходному правилу: A/AMP/batch128.** Target65289.57 ток/с против контроля11546.95 — **5.65×**; пары988.31 против175.83 —5.62×. Source47871.09 ток/с; сумма113160.66 ток/с. Peakreserved8.98GB. B/AMP/batch128 завершён:29251.98 target/с,442.80 пар/с,peakreserved13.50GB, запас до14GB невелик. A/FP32/batch128 исключён правилом:reserved14.37GB≥14GB, несмотря на завершение и allocated9.19GB. GB здесь десятичные, оба пика должны быть<14e9 байт.

AMP на batch8 замедлил throughput на15.1%:175.83→149.24 пар/с. При том же bucket/fused и плане batch32 AMP ускорил A в1.54×, B в2.24×; при batch128 A в2.51×. Предыдущий вывод16 по batch8 нельзя переносить на большие батчи. Контроль23 сам по себе175.8 пар/с отличается от прежних~206; причину межсессионной разницы не измеряли.

Bucketing снизил padding37.86% до23.79% при32 и30.11% при128. В основной сетке batch/bucketing/fused меняются вместе: эффект каждого отдельно не выделен. Больший batch128 содержит в16 раз больше пар на update, чем8. Скорость токенов не доказывает ускорение достижения заданного качества; LR/schedule/число предъявлений данных требуют отдельного quality-контроля.

**Незавершённые строки:** A/FP32/256 OOM после11 warmup; A/AMP/256 после50 warmup+180 measured; B/FP32/128 после17 warmup, B/FP32/256 на первом шаге, B/AMP/256 после8 warmup. Частичные180 шагов не допустимы как результат200. C/FP32/32 дошла только до12 warmup (`budget_warmup`); остальные5 C —`budget_not_started`. Про C нет измеренных rates и вывода о пригодности/качестве. Train-phase заканчивается раньше общего60-мин дедлайна, оставляя20 мин для инференса; завершение всего ноутбука за45.49 мин не означает доступность оставшегося резерва для C в исходном протоколе.

### Инференс: фиксированные22/100k

Greedy cap512 с KV-кешем, только способ исполнения/precision/batch изменяются. Контроль: исходный порядок,FP32,batch32. Sorted-варианты: сортировка вопросов по длине, предаллоцированный буфер, проверка завершения на CPU каждые16 шагов, затем restore исходного порядка.

|Путь/precision|Batch|Test старый / полный таймер, с|Heldout полный, с|Exec-match|Те же tokens heldout|Те же tokens test|
|---|---:|---:|---:|---:|---:|---:|
|контроль, FP32|32|70.88 / 71.22|39.19|189 /991|1024/1024|1919/1919|
|sorted/prealloc, FP32|64|38.20 / 38.59|19.20|189 /991|1024/1024|1919/1919|
|sorted/prealloc, FP16|64|40.07 / 40.46|19.06|191 /991|997/1024|1859/1919|
|sorted/prealloc, FP32|128|32.45 / 32.82|15.75|189 /991|1024/1024|1919/1919|
|sorted/prealloc, FP16|128|26.40 / 26.78|12.81|191 /991|990/1024|1854/1919|
|sorted/prealloc, FP32|256|38.24 / 38.90|15.64|189 /991|1024/1024|1919/1919|
|sorted/prealloc, FP16|256|20.68 / 21.38|7.00|190 /991|994/1024|1850/1919|
|sorted/prealloc, FP32|512|39.59 / 39.96|16.34|189 /991|1024/1024|1919/1919|
|sorted/prealloc, FP16|512|15.19 / 15.57|5.89|190 /991|998/1024|1860/1919|

**Выбор по исходному правилу: sorted FP16/batch512.** Test полный15.5667 с против71.2167 — **4.57×**; старый таймер15.1869 против70.8765 —4.67×. Heldout полный5.8855 с. Peaktest allocated1.274GB/reserved1.818GB. Exec190/991=19.17% против189/991=19.07% контроля; улучшение на1 задачу незначимо (приобретений1/потерь0, exact McNemar$p=1$). FP16/64 и128 дают191/991, приобретений2/потерь0,$p=.5$: это также не доказательство улучшения качества.

FP16/batch512 изменил26/1024 heldout-программ и59/1919 test-программ: совпадение tokens97.46%/96.93%. Нельзя считать генерации полностью неизменными. У всех sortedFP32 tokens совпали на100% в обоих splits. Если нужен именно прежний вывод, быстрый вариант FP32/batch128:full32.82 с,2.17×,exec189/991. В FP32 batch256/512 медленнее128 — максимум batch не всегда максимум скорости.

Полный таймер включает загрузку весов/токенизатора, восстановление/cast/H2D, прогрев, generation/диагностики и запись answers. Download HF/BPE-кеш/exec-scoring вне него.15.57<180 с имеет большой запас в этой области таймера, но это не подтверждённый end-to-end запуск официального бота: формат посылки и его таймер ещё неизвестны. Все timings — по одному сохранённому проходу, интервалы вариации не измерены.

Сумма decoder-итераций на test снизилась13333→2048, но batch×iterations вырос426557→982528 (2.30×): завершённые строки не удаляются из активного батча, проверки EOS реже. Ускорение wall не доказывает уменьшение всей вычислительной работы. Неуплотняемые завершённые строки остаются возможным источником лишней работы.

### Что подтверждено сейчас

**Новые проверки на CPU, без LM-updates/GPU:** 18cells/9code,execution1…9,ошибок outputs нет; sourcecells совпадают с подготовленным23. ZIP CRC и SHA всех78 файлов прошли. Reference снова дал991 finite/33 non_number. Все9 наборов heldout-программ исполнены заново:189/189/191/189/191/189/190/189/190, execution values/statuses совпали с Colab. Контроль23 на всех1024 строках token-identical сохранённому22/100k. Восстановление порядка1919 test/1024 heldout, token-identical, rates/padding/counts, финальный выбор и файлы best проверены.

CPU50k-аудит внутри ZIP — копия проверки перед23:48210finite/1790не дали число (3.58%;60AST-blocked неизвестны), из них1730явно non_number/runtime-error (3.46%). В Colab этот50k-аудит не повторялся; текущий анализ его также не запускал. Это не новый GPU-результат. Пользователь разрешил фильтрацию будущего train, в23 её нет.

**Открыто:** длительное качество batch128/AMP и масштабированных B/C; отдельный throughput C, если понадобится; поведение памяти на будущих редких длинных батчах; вариация таймеров; official score/посылка. Тестовые1919 не имеют доступного gold, поэтому выигрыш скорости не даёт оценки test-качества. Нового quality-обучения, beam/sampling и следующего notebook при этом разборе не готовили.

## Источники

- HF[ test_prompts](https://huggingface.co/datasets/SlavaYus/tinysms_test_prompts), [train_qa](https://huggingface.co/datasets/SlavaYus/tinysms_train_qa): gold/referenceexecution/filter; публичные cards прочитаны 2026-10-09.
- [Pinned корпус](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/tree/b2724689ff079e9c22efd548fe134bfb29036dbb/data), [сводка 22](../results/22-cosine-tail-from-75000-to-100000-steps-encoder4-decoder4-qwen-bpe-8k-swiglu-500k-pairs-summary.json), [21](../results/21-deeper-encoder-4-layers-decoder-4-qwen-bpe-8k-swiglu-500k-pairs-100000-steps-summary.json), [16](../results/16-training-speed-bpe-cache-amp-fp16-swiglu-bpe-4k-100k-pairs-summary.json).
- PyTorch[AMP/unscale/clip](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html), [AdamW/fused](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html), [SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html).
- Python[multiprocessing](https://docs.python.org/3/library/multiprocessing.html), [resource/RLIMIT](https://docs.python.org/3/library/resource.html).
- Лекции: Bucketing и AMP, KV-кеш, SwiGLU. B/C — наши масштабированные конфигурации, не перенос обученных внешних моделей.

- Присланные [notebook23](../results/23-throughput-batch-amp-bucketing-and-batched-inference-results.ipynb) и [сводка ZIP23](../results/23-throughput-batch-amp-bucketing-and-batched-inference-summary.json); сохранённый пользовательский T4-запуск, получен 2026-10-10. Новые CPU-проверки отделены от сохранённых GPU-замеров.
