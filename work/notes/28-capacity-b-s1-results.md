---
type: experiment
course:
  - nlp
status: draft
understanding: 3
created: 2026-10-11
updated: 2026-10-11
sources:
  - kaggle-s1-v1
aliases: []
---

# ДЗ2, №28: S1 остановился до обучения B

**Решение pending. B не инициализирована: 0 updates, 0 time-точек, 0 B checkpoints.** [Сводка](../results/28-capacity-b-equal-training-time-summary.json) · [Протокол №28](capacity-b-equal-training-time.md).

## Причина по логам

GPU1 завершился с exit1 в `exp26_workflow.reference_execution`: assertion числа finite-reference heldout4k ==4096. Это CPU-gate до D0 и `initial_state` B. Ошибка обучения или OOM в B отсутствуют, поскольку модель ещё не создавалась.

В output GPU1 есть только partial-summary, созданная supervisor. Statuses и не прошедшие строки reference **не сохранены**. Логи устанавливают несовпадение числа finite-reference, но не позволяют определить конкретную программу или причину. Cold SymPy import, timeout и worker recovery — **гипотезы**. На CPU те же 4096 reference и 991 дополнительных чисел воспроизвелись точно.

| Целевая точка | Train-time A,с | A exec4096 | A exec991 | A CE4k | B exec/CE/длины/McNemar |
|---|---:|---:|---:|---:|---|
| p006250 | 928.12 | 1888 | 465 | 0.441133 | Нет точки |
| p012500 | 2018.23 | 2210 | 537 | 0.379172 | Нет точки |
| p025000 | 4203.38 | 2446 | 601 | 0.336627 | Нет точки |

Batch128 и OOM fallback96 ещё не проверены; фактически выбранного batch нет. B memory, throughput, AMP skips, greedy test и selected27 test не измерены.

Правило B≥A+2 п.п. на самой поздней общей допустимой time-точке и ≤120 с с выбранным декодированием **неприменимо**: итог pending. Текущая база — A/6,4M + greedy. Stable latest B отсутствует.

## Что сохранено и S2

- **A:** полный финальный model/AdamW/scaler/scheduler/RNG/cursor и все оценки.
- **№27:** весь grid, receipts, execution chunks и test timings; решение complete.
- **B:** только partial-summary без state. Требуется старт от seed42, 6+6/d384/31,94M параметров, LR0,002, прежний order.

Подготовленный S2 с `--previous-source thisarivson/nlp-hw02-kaggle-s1/1` копирует A/27 и использует их verified reference-receipts для GPU1 после проверки SHA, identity, finite indices и значений. Это предусмотренное протоколом повторное использование однажды проверенных reference; timeout и policy не менялись.

[CPU-проверка](../checks/kaggle-s2-reference-reuse-cpu-checks.json) прошла на реальных S1 receipts; проверен отказ при изменённом вопросе. Исходные 12 модулей №26 не менялись. S1-ноутбук сохранён; [отдельный S2-ноутбук](../kaggle/notebooks/nlp-hw02-kaggle-s2.ipynb) готов. **Ничего не pushed.** [Команды продолжения](../kaggle/README.md#s2-продолжение-подготовлено-не-запущено).

B в S2 сравнивается с зарегистрированными A0,96/1,92/3,84M при времени 928/2018/4203 с. A6,4M не является target №28 в текущем протоколе.

## Источники

- `kaggle-s1-v1`: `gpu1-process.log`, `gpu1-failure.json`, `s1-summary.json`, `gpu1-exp28/28-summary.json`; [SHA и ресурсы](../results/kaggle-s1-v1-summary.json).
- [Протокол №28](capacity-b-equal-training-time.md); [CPU reference gate и replay](../checks/kaggle-s1-v1-cpu-audit.json).

## Продолжение S2

[Новый протокол](s2-fast-sampling-and-capacity.md) зарегистрировано до запуска: №27b → M29 наGPU0, повторB28 наGPU1,3ч15мин; четыре временные точки и новое общее правило A/M/B. Описание S1 выше является историческим; текущая версия и статус — в [runs.json](../kaggle/runs.json).
