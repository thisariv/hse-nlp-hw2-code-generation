---
type: experiment
course:
  - nlp
status: learning
understanding: 3
created: 2026-10-11
updated: 2026-10-11
sources:
  - kaggle-s2-v2
  - kaggle-s1-v1
aliases: []
---

# ДЗ2, M29 и B28: S2 — четыре точки B, M не стартовала

**B завершила четыре допустимые train-time точки; последняя2513/4096 противA2538/4096 (−0,610п.п.). M не инициализирована. Общий выбор A/M/B — pending.** [Протокол](s2-fast-sampling-and-capacity.md), [сводка](../results/kaggle-s2-v2-summary.json), [CPU-аудит](../checks/kaggle-s2-v2-cpu-audit.json), [отказ №27b](27b-fast-sampling-s2-results.md).

## Равное накопленное время обучения

Время включает все main/cooldown compute, включая откатываемые ветки; setup/eval/lock/checkpoint I/O исключены. Exec — reference numeric совпадения, не официальный score. Primary4096; secondary991 eligible из1024. CE сохранена FP32/b32 с одинаковым tokenizer. McNemar exact: gains/losses B относительно A, затемp.

|Точка /target,с|A exec4k /991|B exec4k /991|A CE4k /1024|B CE4k /1024|B−A,п.п.|McNemar4k gains/losses;p|
|---|---:|---:|---:|---:|---:|---|
|t01 /928.12|1888 /465|1553 /373|0.441133 /0.467310|0.524011 /0.546319|-8.179|201/536;5.15e-36|
|t02 /2018.23|2210 /537|1968 /497|0.379172 /0.400452|0.416560 /0.438181|-5.908|201/443;7.1e-22|
|t03 /4203.38|2446 /601|2341 /590|0.336627 /0.354381|0.350418 /0.373344|-2.563|249/354;2.18e-05|
|t04 /7312.88|2538 /634|2513 /618|0.314924 /0.331018|0.315256 /0.331087|-0.610|241/266;0.286|

Для M на всех четырёх точках exec/CE/длины/McNemar/тайминги **missing**:0updates,0points,0checkpoints. Причина — GPU0 exit1 после greedy token gate; это не ошибка обучения M и не OOM.

|Точка|B actual train,с|Ошибка времени,с|Main+cooldown updates|Вся compute-история updates|Предъявления финальной траектории|AMP skips истории|
|---|---:|---:|---:|---:|---:|---:|
|t01|928.483|+0.366|3077+616|3693|472 704|0|
|t02|2018.728|+0.502|6171+1231|8018|947 456|0|
|t03|4204.875|+1.490|12364+2470|16681|1 898 752|0|
|t04|7317.474|+4.591|20553+4108|28978|3 156 608|0|

Все errors укладываются в `max(2%target,5с)`, включаяt04 +4,591с. D0gate passed:1677/4096,422/991, CE0,476417/0,486333. Reference reused изS1 после SHA/identity/finite/value проверки; при разборе независимо исполнены и воспроизведены4096+991 чисел.

## Длины и парные изменения

Средняя длина — число generated token IDs, включаяEOS; secondary — все1024 генерации. B bins показывают matches primary при reference BPE длинах<60 /60–90 /90–120 /≥120; denominators2013/1457/488/138. Полные A/B bins и обе secondary/primary McNemar-таблицы — в JSON.

|Точка|A средняя длина4k/1024|B средняя длина4k/1024|B matches по4 length bins|McNemar991 B vsA gains/losses;p|
|---|---:|---:|---|---|
|t01|64.286/62.604|63.132/63.830|1150/353/45/5|34/126;1.25e-13|
|t02|61.287/60.572|64.378/63.680|1355/509/96/8|62/102;0.00222|
|t03|62.993/63.530|62.875/62.699|1492/688/134/27|64/75;0.396|
|t04|61.459/60.961|62.461/61.514|1558/755/170/30|53/69;0.174|

|Точка|McNemar4k vsD0 gains/losses;p|Vs предыдущей B gains/losses;p|
|---|---|---|
|t01|248/372;7.21e-07|—|
|t02|480/189;4.22e-30|569/154;9.36e-57|
|t03|774/110;1.23e-123|522/149;2.12e-49|
|t04|938/102;6.45e-170|364/192;2.67e-13|

Ранние B существенно хуже A при равном compute-time; наt04 разница незначима (primaryp0,286;secondaryp0,174). Это не доказательство эквивалентности. Последний прирост B+4,199п.п. противt03,364gains/192losses,p2,669e-13: плато не установлено. B наt04 увидела3,157M разных строк противA6,4M; сравнение фиксирует время, а не число данных илиupdates.

## Ресурсы и test

|Ресурс|B|M|
|---|---:|---|
|Параметры|31 940 096|Не создана; план15 766 464|
|Train batch|128; OOM fallback96 не понадобился|Не выбран|
|Чистая compute-история|7317.474с;28 978updates|0|
|Средний throughput, вся история|3.960updates/s;506.89pairs/s|Не измерен|
|Полезные source/target tokens/s|24712.8 /33955.0|Не измерены|
|AMP skips, вся история /final trajectory|0 /0; finite loss/grad|Не измерены|
|Peak reserved greedy evaluation|5.109GiB|Не измерен|
|Peak train memory /sample batch probe|Не записан /не выполнялся|Не измерены|
|Полный test1919 greedy /selected vote|Не измерен /не измерен|Не измерен /не измерен|

Greedy heldout evaluation времена42,848с/4096 и10,686с/1024 наt04 исключают execution; это **не fulltest1919**, экстраполяция для≤120с запрещена. Test-ветка B пропущена, поскольку27b summary неcomplete. A тоже не имеет нового fulltest с execution вS2; S1 greedy13,003с был без execution. На всех saved B execution есть7timeouts (D0+4точки),0chunk retries/0pool restarts; CPU status/value и все10 match sets воспроизведены полностью.

## Решение и сохранённые состояния

По старому отдельному порогу №28 **B≥A+2п.п. не выполнен**. По общему протоколу S2 итог **pending**: нет успешного27b gate, M time-точек и соответствующих fulltest timings. Общих A/B точек4, общих A/M/B0; M↔B McNemar не существует. При отрыве A отB<2п.п. один только heldout не выбирает модель: требуются тайминги. Рабочая база остаётся A/6,4M + прежнийgreedy.

Сохранены B FP32 model-only весаt01–t04, полный оценённыйfinalt04 сAdamW/scaler/scheduler/RNG/cursor и отдельный stable latest main20553. **Stable latest не является оценёнными t04 весами**: final20553+4108updates, cursor3156608; stable20553/cursor2630784. CPU hashes/tensor digests, optimizer steps и состояния проверены; final tensors точно совпадают сmodel-onlyt04. Полный forward CE/обучение/CUDA при разборе не повторялись.

Сессия: supervisor7860.524с, последний kernel export7877.358с=**131.29мин≈2ч11мин17с**. KaggleCOMPLETE, внутренний statuspartial, exits[1,0], unreaped0; watchdog/hard timeout не понадобились. GPU0 exit1 виден уже к450,156с; остальное время устройство не выполняло M. Utilisation, фактический SDPA backend и точное списание квоты не записаны.

Live Dataset не скачан:403, отсутствует вpaginated `--mine`. Kernel16,314с: `LIVE: Kaggle Secrets unavailable; uploads disabled`; синхронизация не включилась. Дополнительных live-only артефактов нет; конкретная причина доступа к Secrets не сохранена. [Результаты download/manifest и S3](../kaggle/README.md#разбор-s2-v2-и-восстановление-s3).

## Источники

- `kaggle-s2-v2`: pinned outputv2, B28 summary/metrics/generations/execution/training receipts,7PT и kernel/process logs. SHA/bytes — [CPU-аудит](../checks/kaggle-s2-v2-cpu-audit.json).
- `kaggle-s1-v1`: четыре A baselines и reference receipts; [A/6,4M](26b-final-wsd-data-scaling.md), [S1/B до обучения](28-capacity-b-s1-results.md).
