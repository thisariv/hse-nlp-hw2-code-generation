---
type: experiment
course:
  - nlp
status: learning
understanding: 4
created: 2026-10-15
updated: 2026-10-15
sources:
  - kaggle-b1-independent-review
  - kaggle-b1-preparation
  - kaggle-r1-independent-review
  - nlp-hw02-architecture-decision
  - nlp-hw02-overfit-check
aliases: []
---

# B1: две широкие модели с нуля

План уточнён пользователем после независимого ревью подготовки `91188a2`. Результаты проверок и запуска — [b1-review](b1-review.md). Старый вариант GPU0 6+8/d640 и точки 19 800/28 800с отменены. [architecture-decision](architecture-decision.md) содержит исходные реальные T4 замеры; выбор пользователя здесь имеет приоритет над условным исследовательским планом той заметки.

## Архитектуры и рецепт

| Worker | Код | Форма | SwiGLU / heads | Параметров | Начальный предел micro | Shape-bench, пар/с |
|---|---|---|---|---:|---:|---:|
| GPU0 | E | e8-d4-w768 | 2048 / 12 | 108 694 019 | 64 | 246,7 |
| GPU1 | W | e6-d6-w768 | 2048 / 12 | 113 421 827 | 32 | 229,4 |

Обе с нуля, прежний Pre-LN `SimpleTransformer`: shared token embedding, отдельный untied output head, learned positions2048, biased SwiGLU, final encoder/decoder norms. Реальное число параметров проверяется на CPU и независимой формулой. Micro выбирается отдельным forward/backward/**настоящим Adam step** на самых длинных cached rows; безопасный fallback может быть ниже табличного предела.

Рецепт C из L1: normalized token init (embedding делится на$\sqrt d$, forward умножает обратно), model seed42, dropout0, FP32 параметры, AMP FP16 + GradScaler(init1024), fused AdamW wd0,01, clip1. Effective batch128, token-weighted accumulation с полным хвостом. Отличие LR — **peak0,001**, warmup1000 attempted updates → constant. AMP skips записываются отдельно; emergency rollback/halving ограничен тремя эскалациями.

Verified compact cache: 9 725 737 пар, BPE8k/compact30. Weighted order C-long2x: compact target length≥79 предъявляется2×, остальные1×; **новый order seed2026101501**, bucket4096, batch128, циклический cursor. Без D1-фильтра, новых targets и main loss truncation: **q0**. Seed и порядок сохраняются при переносе.

## Оценки и критерий продолжения

| Отметка чистого main | C heldout4k | Ветка B1 |
|---:|---:|---|
| 9 000с | 62,87% | 900с linear cooldown → greedy heldout4k + old991 + CE обоих |
| 28 370с | 67,50% | то же |
| Конец main, цель34 200с | — | то же |

Каждая ветка начинается после полного uncooled stable save; затем восстанавливаются model/Adam/scaler/Python/NumPy/Torch CPU/CUDA RNG/cursor/warmup/main-time. Cooldown updates, RNG и данные не входят в дальнейший main. Последняя оценка B1 тоже короткая, **q0**, после неё latest остаётся uncooled. Test/K8 в B1 не требуются. Долгая финальная сессия с `--final-session` — отдельный будущий режим и в боевой B1 не включается.

Время setup, I/O, cooldown и eval исключено из оси main. Отметка достигается с точностью одного update; отчёт хранит target и фактически достигнутое время. Пропущенные/неполные оценки дают `partial`, их нельзя считать положительным gate. Порог продолжения — сопоставление completed receipts с C, в первую очередь28 370с/67,50%; модель ниже этой точки не переносить автоматически. Окончательное решение между сессиями, без автоматической смены архитектуры по ходу B1.

## Бюджет и сохранность

Main34 200с≈9,5ч; workers40 260с, watchdog/backend40 800с=11ч20мин **от первой code cell**, включая setup. Резерв:34 200+3×(900+600)+480setup+240I/O=39 420с; запас840с до worker deadline и ещё540с до hard wall. Периодические saves тоже потребляют этот запас; при нехватке времени main/оценка получают partial.

`/kaggle/working/model-GPU{0,1}`: atomic latest.json, SHA/bytes и два последних FP32 full PT. Сохранение каждые900с wall, до/после веток и при штатном выходе. Payload: параметры, Adam moments/steps/groups, scaler, все RNG, cursor, LR/warmup/emergency factor, main-time и история. Strict resume проверяет data/order/model_spec/recipe/runtime SHA/Torch/CUDA, counters и finite tensors. Fresh fallback и повторная нормировка при resume запрещены.

Без Secrets и live-sync. Watchdog armed до Torch/install, убивает отдельные process groups к deadline/hard wall; сохраняет последний опубликованный stable. Это ограничивает потерю main последними≈15мин, но не гарантирует выгрузку output при аварии инфраструктуры Kaggle. Сбой одного worker не останавливает здорового.

Оценка диска: full Adam≈12N байт. Два поколения обеих моделей≈5,33GB, три evaluated FP16 точки≈1,33GB; два одновременных transient full≈2,67GB. Вместе с order/journals≈9,4GB в пике, меньше guard10GiB. Не делать ZIP/копии кеша в working. Transfer берёт только latest обеих моделей и order≈2,7GB; runtime mount через symlinks, без копий больших PT.

## Private API и перенос

Аккаунт B1 — аккаунт друга только через `KAGGLE_CONFIG_DIR=~/.kaggle-friend2`**. Каталоги/значения credentials не инспектируются; inherited auth env очищается. Входы: private `thisarivson/nlp-hw02-compact-cache-verified/2` (notebook, путь `notebooks/owner/slug`) и `thisarivson/nlp-hw02-26b-inputs` (dataset). Owner-qualified mount имеет приоритет, затем flat fallback; cache manifest и все payload SHA проверяются до spawn, input-preflight.json фиксирует реальные пути.

Из `work/kaggle/`:

```sh
python build_b1.py --owner friend2 --dropout 0.0
python build_b1.py --owner friend2 --gpu-smoke --dropout 0.0
python b1_tests.py
python b1_static_check.py
```

`b1_review_api.py` выполняет inputs/quota/push/status/metadata/logs/output через friend2 config, без операций удаления. GPU smoke backend≤900с, worker780/hard840; одна итерация, максимум две при исправлении конкретного сбоя. Обе формы: реальный memory probe, исходные8 length-батчей shape-bench,256 main attempts,16 peak-LR attempts, AMP overflow, serialized exact next update, greedy/CE мини-наборы + restore и следующие12 main updates bit-exact, end-main оценка, stable states.

Перенос в следующую сессию — `b1_transfer.py` с **pinned kernel version**: потоково только latest/full обеих моделей и order; новый private Dataset следующего владельца, remote sizes + SHA round-trip каждого файла, receipt с transfer manifest SHA. После forced exit общий output manifest может отсутствовать: `--emergency-receipts` восстанавливает список из atomic latest и order.json, сверяет общий cache/order SHA и каждый PT. Этот маршрут покрыт регрессией без aggregate manifest. Никаких удалений ресурсов Kaggle или перезаписи существующего slug. При продолжении использовать тот же вычислительный runtime и dropout0.

## Источники

- [architecture-decision](architecture-decision.md) — реальные T4 замеры форм и исходные length-батчи, не обещание качества.
- [l1-long-training](l1-long-training.md), [r1-review](r1-review.md), [l3-review](l3-review.md), [overfit-check](overfit-check.md) — рецепт C, stable/restore, история нестабильности и ограничения вывода о переобучении.
- [b1-review](b1-review.md) — независимые CPU/API/CUDA receipts текущего решения. Библиография исходных работ — bibliography.
