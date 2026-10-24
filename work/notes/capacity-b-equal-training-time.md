---
type: experiment
course:
  - nlp
status: draft
understanding: 3
created: 2026-10-11
updated: 2026-10-11
sources:
  - https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html
  - https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
aliases: []
---

# ДЗ2, опыт28: ёмкость B при равном времени обучения

[Ноутбук S1](../kaggle/notebooks/s1-t4x2-26b-27-28.ipynb) · [команды](../kaggle/README.md) · [A/26b](wsd-data-scaling-and-sharded-cache.md) · [27](execution-voting-decoding.md). GPU1 независимо от GPU0. **S1 v1 остановился на CPU reference-gate до обучения B; решение pending.**

[Результат S1 v1](28-capacity-b-s1-results.md) · [Сводка сессии](../results/kaggle-s1-v1-summary.json). Пороги ниже сохранены без изменений.

## Гипотеза

**Наша гипотеза:** большая B улучшит exec-match при том же доступном чистом времени, несмотря на меньшую скорость. Из23 известно лишь442,8 пар/с B/AMP128 против988,3 уA; это короткий throughput benchmark случайных весов, не качество. A на3,84M дала59,72%/4096, но плато ещё не доказано.

B:6encoder+6decoder/d384/h6/SwiGLU1024, **31 940 096 параметров**; A4+4/d256/h4/SwiGLU680,12 086 912. Остальное D0: normalized token init std1/√d при прежнем√d scaling, dropout0, AMP/GradScaler, fused AdamW/WD.01/clip1, seed42, прежние cache0–41/order230042. Один peak LR0.002 по эвристике0.003×256/384, floor0.0002; LR-search нет. Batch128; при OOM в первых500 шагов единственный fallback96 с полным рестартом небезопасного prefix и записью события. Поздний OOM сохраняет последний безопасный checkpoint и требует следующей сессии.

## Равный бюджет и WSD

Сравнивается **накопленное чистое train-время всей вычислительной истории**, включая уже оплаченные боковые ветки. Setup/eval/checkpoint I/O исключены, как в сохранённой `train_seconds_total` A. Это не одинаковыеupdates и не одинаковое число строк.

| Точка A | Фактические предъявления | Целевой train-time, с |
|---|---:|---:|
|p006250|960000|928.116608|
|p012500|1920000|2018.225874|
|p025000|3840000|4203.384926|

Первые500 настоящих обновлений B входят в обучение. По их throughput строится план числаmain/cooldownupdates. Если средняя стоимость шага $\tau$, накоплено $c$ боковых шагов, целевой бюджет $t$, первичный прогноз fork — $u\approx(t/\tau-c)/1.2$, длина cooldown≈0.2u. $u$ — mainupdates,1.2 учитывает main+20% cooldown. Прогноз сохраняется отдельно.

Перед fork число оставшихсяupdates уточняется по накопленному live train-time и среднему throughput. При запуске длина ветки фиксируется; linearLR снижается доfloor за этиsteps. Фактическое отклонение времени от цели записывается. Общая точка допустима при |ошибка|≤max(2%целевого времени,5с): это заранее выбранный технический допуск к дискретным GPU-шагам, не экспериментальное качество. При большем отклонении точка остаётся в отчёте, но не в решении «равное время».

Основная ветка: warmup300→stable0.002→linear cooldown40000–50000steps, как26b; продолжать, сколько позволяет S1. Полные model/AdamW/scaler/RNG/cursor восстанавливаются после каждой боковой ветки; её стоимость остаётся в истории. При batch96 используется тот же плоский order128, порезанный на96; это сохраняет row order, но меняет границы батчей и немного ухудшает bucketing. Записывать batch/предъявления/AMP skips.

## Оценка и решение

В каждой time-точке: greedy FP16 sortedb512/cap512; exec4096/991, CE FP32/b32, длиныreferenceBPE, McNemar противD0/предыдущейB и соответствующейA, время/память/предъявления. Reference/D0 gate проверяются на Kaggle; fresh не оценивается.

На самой поздней **общей допустимой time-точке** сохранить полный checkpoint B и greedytest1919 с полным таймером. После готовности27 измерить test1919 на этом checkpoint B с выбранным тамT/k/voting (илиgreedy), включая весь pipeline. B идёт в финал только при≥A+2п.п. на4096 (минимум82 matches) и этом времени≤120с. Иначе A с большим числом предъявлений. При отсутствии сопоставимойA или полного выбранного27 тайминга решение pending; основной stable latest B не выдаётся за оценённый cooldown checkpoint.

## Проверки и ограничения

CPU: настоящие tiny optimizer updates с калибровкой/ветками/resume из подключённого output; время для граничных controller-тестов фиксировано fixture, не прогноз GPU. Branch/fullstate parity подтверждена отдельно в двухпроцессном smoke. Пороги2п.п./120с и exclusion несопоставимых точек проверены. Полная B128/96, позднийOOM, rareAMPskips, CUDA restore, память/SDPA и качество/скорость не измерялись. СменаTorch/CUDA фиксируется, побитовое CUDA равенство не обещается. Одинseed/LR и repeatedheldout не позволяют доказать общий оптимум ёмкости.

## Источники

- Пользовательский протокол S1 от2026-10-11; [сводка26b](../results/26b-wsd-data-scaling-46875-of-50000-main-updates-summary.json) — targettrain-time/счётыA; [23](throughput-batch-amp-bucketing-and-batched-inference.md) — прежние параметры/скоростьB.
- PyTorch2.11, [AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html): fusedoptimizer; поведение на новых KaggleT4 не проверено.
- Kaggle, [метаданные](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md): два независимых процесса внутри одного T4×2 ноутбука — наш протокол, не рекомендация документации.

## Продолжение S2

[Новый протокол](s2-fast-sampling-and-capacity.md) зарегистрировано до запуска: №27b → M29 наGPU0, повторB28 наGPU1,3ч15мин; четыре временные точки и новое общее правило A/M/B. Описание S1 выше является историческим; текущая версия и статус — в [runs.json](../kaggle/runs.json).
