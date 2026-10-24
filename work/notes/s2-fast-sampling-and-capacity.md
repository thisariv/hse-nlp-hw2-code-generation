---
type: experiment
course:
  - nlp
status: learning
understanding: 3
created: 2026-10-11
updated: 2026-10-11
sources:
  - kaggle-s1-v1
  - https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html
  - https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md
  - https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_secrets.py
aliases: []
---

# ДЗ2, S2: №27b, №29 M и повтор №28 B

**Протокол записан до запуска.** Private Kaggle T4×2, два независимых процесса, бюджет **11700 с = 3 ч 15 мин** от первой ячейки, включая подготовку; последние 300 с зарезервированы сводкам. Для повторной S2 добавлены watchdog и серверный потолок 3 ч 40 мин; параметры опытов и правила выбора прежние. [Реестр](../kaggle/runs.json), [код и запуск](../kaggle/README.md). Источник A и reference — pinned `thisarivson/nlp-hw02-kaggle-s1/1`. Старые результаты и Kaggle-сущности не удаляются.

## GPU0: №27b перед №29

Гипотеза №27b: узкое место S1 — 5D broadcast cross-KV, а не исполнение программ. Encoder один на вопрос; cross-K/V физически `repeat_interleave(k)` в `[candidate, head, source, dim]`, обычный 4D SDPA FP16. Self-KV, tokens/output/valid/done/logprob заранее выделены, проверка завершения на CPU раз в 16 шагов. Проверить 2048 строк-кандидатов, при OOM 1024; сохранить попытки, пик памяти и выбранный batch. Для больших M/B выбор памяти повторяется.

На финальных весах [A/6,4M предъявлений](26b-final-wsd-data-scaling.md): новый greedy должен **token-identical** старому сохранённому greedy на всех heldout4096; принудительные токены не используются. Vote k16/T0.7 без greedy-голоса должен дать **2707±45** совпадений. При провале gate сэмплер не допускается к финальному выбору; диагноз и partial сохраняются.

Полный свежий test1919 измерить отдельно для k8 и k16/T0.7, без greedy-голоса: чтение test, загрузка checkpoint/tokenizer, генерация, исполнение, голосование, запись answers. Greedy также измеряется с исполнением. Выбрать **наибольший k из {8,16} с полным временем ≤120 с**. При отсутствии допустимого k явно записать отсутствие допустимого голосования и greedy fallback. Tie-break как S1: размер numeric-кластера → наличие ответа greedy → средний untempered logprob; greedy не добавляет голос. При отсутствии numeric sample fallback greedy. Fresh не кодируется и не оценивается.

№29 M: **6 encoder + 6 decoder / d256 / h4 / SwiGLU680**, с нуля, normalized token init, dropout0, peak LR0.003/floor0.0003, AMP FP16/GradScaler1024, fused AdamW/WD0.01/clip1, bucketing, batch128. Тот же cache0–41, order230042 и seed42. M ≈15.7M параметров; точное число записывается из модели.

## GPU1: повтор №28

[B](capacity-b-equal-training-time.md): 6+6/d384/h6/SwiGLU1024, 31 940 096 параметров, peak LR0.002/floor0.0002, batch128. Единственный OOM fallback96 в первых500 updates с безопасным рестартом, событие сохраняется. Остальное D0 как M.

Reference4096/991 брать из уже сохранённых receipts и `26b-inputs`, сверяя SHA, row identity, finite indices и значения; **не переисполнять под конкуренцией**. Optional повторная сверка может только записать расхождения, без assert числа новых numeric. D0 на Kaggle повторяется с gate ±3 matches / CE1e-4 как №28; при провале обучение останавливается с partial и диагнозом.

## Общие train-time точки и оценка

WSD warmup300 → stable; обратимые линейные cooldown-ветки в накопленном чистом времени **всей compute-истории**, включая боковые ветки. Setup/eval/ожидание lock/checkpoint I/O исключены. Fork уточняется по live throughput, cooldown около20% main-prefix. Возврат восстанавливает model/AdamW/scaler/RNG/cursor, затраты остаются в истории. После последней цели обучение останавливается, stable latest не выдаётся за оценённые веса.

| A point | Train-time, с | A exec4096 | A exec991 |
|---|---:|---:|---:|
|p006250|928.116608|1888|465|
|p012500|2018.225874|2210|537|
|p025000|4203.384926|2446|601|
|p050000|7312.883812|2538|634|

Точные времена читаются из `26b-final-summary`. Допуск совпадения времени: `abs(error) <= max(2% target, 5 s)`, как №28. Последняя точка запускается только если позволяет остаток; иначе явно `missing`, без экстраполяции.

На каждой точке M/B: exec4096 и991, CE FP32/b32, reference BPE length bins, средняя длина генерации, McNemar против D0/предыдущей точки/соответствующей A; попарный M↔B добавляется в общую сводку. Greedy FP16/b512/cap512 как №28; полный финальный test greedy и selected k используют №27b.

Generated execution: **wall3 с, CPU2 с, 2 spawn-worker на процесс**, AST/numeric policy прежняя, timeouts и pool recovery записываются. Оценки процессов сериализованы файловым lock; другое GPU в это время может обучаться. Ожидание lock не входит в измерение test pipeline, его длительность записывается отдельно.

## Правило выбора до результатов

Сравнить A/M/B на каждой общей допустимой time-точке (exec4096, McNemar, длины). Финальный кандидат — лучший по exec4096 на **самой поздней общей допустимой точке** всех трёх моделей. При отрыве **≥2 п.п.** от следующего выбрать его только при полном test selected vote≤120 с. При отрыве **<2 п.п.** выбрать меньший полный inference-time среди моделей, отстающих от лидера менее чем на2 п.п. и укладывающихся≤120 с. При отсутствии необходимых таймингов/общей точки/успешного gate — pending. Для A на более ранней общей точке финальный таймер повторяется на её соответствующем checkpoint, если сохранён; иначе pending, без подстановки A6.4M.

M/B сохраняют финальные test-тайминги своей последней допустимой точки. Если самая поздняя общая точка более ранняя, нужные тайминги на ней измеряются отдельно при наличии времени. Это позволяет применять правило к одним и тем же весам.

## Надёжность и CPU smoke

Checkpoint примерно каждые 600 с. В output остаются **latest и последний оценённый полный checkpoint с AdamW** на модель; каждая точка отдельно хранит FP32 model-only weights. Stable latest и оценённый final различаются. Anchor обратимой cooldown-ветки живёт во временной папке вне `/kaggle/working`; latest во время ветки содержит его полный payload, поэтому resume восстанавливается после потери временной папки. Устаревшие полные состояния удаляются после атомарной смены receipt.

Кеш и исходный resume читаются из `/kaggle/input` через memmap. Если Dataset остался упакованным, распаковка — только в `/kaggle/temp` или `/tmp`; временная папка удаляется при штатной финализации. **ZIP отсутствует**: JSON, генерации и веса скачиваются отдельными Kaggle outputs, опись — `s2-output-manifest.json`. Цель — ≤3–4 ГБ; tiny CPU output около66 МБ, включая копии для проверки resume. Расчёт тензоров полноразмерных M/B выполняется отдельно; размер всех будущих генераций до GPU-прогона неизвестен.

Независимый watchdog вооружён **в первой ячейке до pip/Torch/распаковки**. Workers и installer запускаются с `start_new_session=True`; exec-пулы наследуют группу. На11400-й секунде — SIGTERM группе, через20 с — SIGKILL группе; все process waits ограничены, unreaped PID не задерживают report. Heartbeat печатает последние три строки логов и остаток времени. Аварийный выход —11400+600=**12000 с (3:20)**: KILL дочерним группам, атомарный `session-summary.json`, затем `os._exit(0)`. Push-обёртка передаёт серверный `session_timeout_seconds=13200` (**3:40**) через `session-limits.json` в bundle.

`metrics.json`, `training.json` и summary каждой точки пишутся атомарно сразу после оценки, до финального checkpoint I/O. При смерти между metrics и summary общий report восстанавливает точку из отдельных receipts. CPU smoke: tiny FP32/FP16 sampler и реальные reference receipts; два M/B-контроллера, model/AdamW/scheduler/RNG parity после удаления scratch; bounded retention/model-only/no ZIP; TERM-resistant libc sleep/loop с multiprocessing-внуками; зависший report/подготовка и имитация unreapable leader. [Обычный smoke](../checks/kaggle-s2-cpu-checks.json), [watchdog](../checks/kaggle-s2-watchdog-cpu-checks.json).

CPU не проверяет настоящий CUDA/driver D-state, сохранение output сервером Kaggle после `os._exit(0)`, T4 memory/backend, full heldout gates, test≤120с и достижимостьt04. Один seed, repeated heldout selection; official score и формат остаются открыты.

## S2 v1: наблюдение и гипотеза

По сообщению пользователя, v1 продолжала RUNNING около4,5 ч при бюджете3:15, output0 B и пустом UI-логе. Пользователь отменил v1, output/лог остались пустыми: результаты потеряны; новая API-проверка подтверждает `CANCEL_ACKNOWLEDGED`. **Гипотеза**: прежний `session_runner` после `kill()` вызывал `wait()` без таймаута; CUDA/driver D-state либо дочерние процессы могли удерживать ноутбук. Безграничный wait подтверждён кодом, причина v1 без лога не установлена.

Разрешён полный повтор как **новая версия того же slug**, с pinned S1/1 и запуском M/B с нуля,4 точки/11700с/потолок13200с. Версия/URL/статус — в [runs.json](../kaggle/runs.json).

Живое сохранение — private Dataset `thisarivson/nlp-hw02-s2-live`: новая версия после metrics/test/summary и раз в≈15мин. Staging вне output≤450МБ с запасом на ZIP; сводки, heldout/test journals, FP16 model-only веса оценённых точек, session-summary и log tails. Credentials из `UserSecretsClient` только в окружении; workers их не наследуют, файлы проверяются перед upload, stdout/stderr CLI подавлены. Upload≤540с, errors nonfatal; финальный upload≤30с, watchdog≤20с в пределах12000с. CPU smoke с моками и реальными hung subprocess trees прошёл; скорость/успех real upload неизвестны до Kaggle. Подробности и команда скачивания — [README](../kaggle/README.md#устойчивый-повтор-s2-live-сохранение).

## Источники

- Пользовательский протокол S2 и запрос устойчивого повтора от2026-10-11 — параметры, правила решения и наблюдение о v1; не новый GPU-результат.
- `kaggle-s1-v1`: [26b-final](26b-final-wsd-data-scaling.md), [27](27-execution-voting-s1-results.md), [28](28-capacity-b-s1-results.md) — исходные результаты и артефакты.
- PyTorch2.11, [SDPA](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.scaled_dot_product_attention.html): 4D формы и boolean mask; фактическая скорость/backend T4 проверяются в S2.
- Kaggle, [Datasets CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md), [UserSecretsClient](https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_secrets.py): private create по умолчанию, version/dir-mode zip и runtime secrets; дополнительно проверен установленный CLI2.2.4.

## Запуск

S2 **v1** принят Kaggle2026-10-11: [ноутбук](https://www.kaggle.com/code/thisarivson/nlp-hw02-kaggle-s2). Первичный statusQUEUED; serverprivate/GPU/T4 и source совпали с локальной сборкой. [Runs](../kaggle/runs.json), [CPU smoke](../checks/kaggle-s2-cpu-checks.json), [static build](../checks/kaggle-s2-static-checks.json). Протокол выше был встроен в source доpush; новые GPU-результаты отсутствуют.

Полный повтор **v2** принят2026-10-11; контроль через225с — **RUNNING**, pinned source/private/internet/T4 подтверждены. Live Dataset ещё не виден; первая реальная синхронизация не подтверждена. Дополнительных push нет. Коммит заблокирован shared `index.lock`, изменения сохранены.

## Получен результат S2 v2

Kaggle **COMPLETE**, внутренний итог **partial**, exits `[1,0]`, сессия с экспортом **7877,358 с = 131,29 мин**. [№27b](27b-fast-sampling-s2-results.md): 35/4096 greedy token mismatches, обязательный gate failed; batch2048 прошёл, testk8/k16 не измерены. CPU vote2702/4096 внутри2707±45 — только post-hoc диагностика. [M29/B28](29m-28b-capacity-s2-results.md): M не стартовала; B завершила все4aligned точки, t04 **2513/4096,618/991,CE4k0,315256** противA2538/634/0,314924; общий выбор **pending** по исходным правилам.

[Pinned output/SHA и сводка](../results/kaggle-s2-v2-summary.json), [CPU-аудит](../checks/kaggle-s2-v2-cpu-audit.json): все сохранённые программы, reference, B execution/match sets и состояния весов проверены; полного CPU CE forward и новых GPU updates нет. Live uploads отключены: `Kaggle Secrets unavailable`; Dataset не найден, дополнительных материалов нет. [S3: сохранённые состояния, команды и ограничения resume](../kaggle/README.md#разбор-s2-v2-и-восстановление-s3). Правила протокола выше не изменены.
