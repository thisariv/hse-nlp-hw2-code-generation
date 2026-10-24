---
type: experiment
course:
  - nlp
status: draft
understanding: 3
created: 2026-10-12
updated: 2026-10-12
sources:
  - kaggle-s3
  - kaggle-s1-v1
  - kaggle-s2-v2
aliases: []
---

# ДЗ2: S3 — результаты компактного таргета и плейсхолдеров

**Решение по заранее записанному правилу: №30 и №31 pending.** Точек 1.92M нет; №30 также не имеет сравнения при равном времени. На 0.96M compact быстрее при близком качестве, placeholders ухудшают качество. Это ранняя диагностика, не окончательное принятие или отказ.

## Полнота, причина остановки и восстановление

Kaggle v1 COMPLETE, внутренний partial, exits `[1,1]`, wall **2211.14 с / 36.85 мин**. Скачаны все **84 output-файла** в ignored `results/kaggle-runs/nlp-hw02-kaggle-s3/1/`; с локальным receipt и API-логом — 86 файлов / 404314068 байт. Все manifest bytes совпали; служебный `__notebook__.ipynb` API не выдаёт. Полный локальный SHA-инвентарь, producer SHA весов/latest/tokenizer, original frozen26 modules и pushed port SHA проверены. Manifest сам содержит bytes, а не SHA.

Оба worker упали при первом forward после оценки и rollback к stable6250: `Inference tensors cannot be saved for backward`. `evaluate` меняла FP32→FP16→FP32 модель внутри `torch.inference_mode`; созданные при конверсии parameter storages стали inference tensors. Это ошибка режима выполнения, а не deadline, OOM или ошибка восстановления программ.

Локальный `s3_runtime.evaluate` исправлен на `torch.no_grad`. [Регрессия](../checks/kaggle-s3-dtype-regression.json) воспроизводит старую ошибку на CPU после half/float, затем проверяет настоящий patched evaluate и два AdamW updates для обоих вариантов; parameter identity сохранена. Старый tiny smoke не ловил ошибку, поскольку CPU выполнял FP32→FP32. Реальный CUDA-путь исправления ещё не проверен. Frozen v1 notebook и downloaded source сохранены; новая сборка и push не выполнялись.

Сохранены для каждого варианта: полные поколения/exec/CE обоих split на0.96M, model-only FP32 cooldown weights и latest full stable checkpoint с AdamW/scaler/scheduler/RNG. Latest: main6250, branch0, cursor800000; optimizer step6250. История включает стоимость уже оценённого cooldown — 960000 compute pairs; она не равна stable cursor. SHA и finite FP32 tensors проверены; после загрузки tensors обычные. Продолжение должно восстановить stable state и сохранить эту compute-историю; текущий worker пока всегда стартует с initial и требует отдельной resume-подготовки.

№32 **не стартовал**: ошибка GPU0 вышла до `B.run`. Generic `32-summary.json` — только partial-маркер, scores0/332800, score batches отсутствуют. Существующие B t04 weights SHA подтверждены. GPU scoring не повторялось; отдельно наCPU/FP32 заполнены168960 scores всех requestedk8/k16/k32+greedy pools, и [ансамблевый выбор завершён](32-candidate-ensemble-s3-results.md). Неиспользуемые T0.5/T1.0 остаются вне CPU-повтора.

## Равные предъявления

Δ в п.п.; McNemar — exact two-sided, gains/losses относительно того же 26b point. Heldout denominator4096, old только991 finite reference из1024.

| Точка / вариант | Exec4k | Δ / p | Exec991 | Δ / p | Чистый train, с | Пар/с |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.96M / 26b | 1888 | — | 465 | — | 928.12 | 1034.4 |
| 0.96M / №30 | 1876 | -0.29 / 0.638 | 450 | -1.51 / 0.228 | 802.84 | 1195.8 |
| 0.96M / №31 | 1733 | -3.78 / 5.25e-10 | 436 | -2.93 / 0.0215 | 915.86 | 1048.2 |
| 1.92M / 26b | 2210 | — | 537 | — | 2018.23 | 951.3 |
| 1.92M / №30 | missing | — | missing | — | — | — |
| 1.92M / №31 | missing | — | missing | — | — | — |

№30: gains/losses **267/279** на4k, **60/75** на991. №31: gains/losses **233/388** на4k, **60/89** на991. 

№30: +15.60% пар/с и −13.50% чистого времени; полезных target tokens51.245M против64.316M (−20.32%). Преобразование prefix1.92M заняло1299.38с и исключено из train timer; для №31 —829.94с. Скорость обучения не означает такой же выигрыш end-to-end при первом запуске. AMP skips0. CE измерена в собственных представлениях и численно с26b не сравнивается.

## Равное время №30

На контрольных928.12/2018.23с независимых compact-веток нет. Единственная compact-точка802.84с находится **ниже первой положительной точки control928.12с**: интерполяция по log-time не имеет нижней опоры. D0 с time0 не позволяет брать logarithm. Экстраполяция не выполнена; equal-time quality и обязательный gate остаются pending.

## Длины исходного reference

BPE22, без specials; bins и denominator совпали с control. Значения — matches/denominator на0.96M; lengths не пересчитаны в новом представлении.

| Split / вариант | <60 | 60–90 | 90–120 | ≥120 |
| --- | ---: | ---: | ---: | ---: |
| 4k / 26b | 1307/2013 | 483/1457 | 86/488 | 12/138 |
| 4k / 30 | 1319/2013 | 462/1457 | 85/488 | 10/138 |
| 4k / 31 | 1240/2013 | 421/1457 | 64/488 | 8/138 |
| 991 / 26b | 335/495 | 113/345 | 12/115 | 5/36 |
| 991 / 30 | 321/495 | 112/345 | 11/115 | 6/36 |
| 991 / 31 | 324/495 | 93/345 | 14/115 | 5/36 |

## Ошибки

Пересекающиеся метки из [data-research-ceiling](data-research-ceiling.md), только среди exec ошибок; строка может иметь несколько меток. «Missing» — только цифровое число вопроса, реально использованное reference. Из outside-number исключены фиксированные, словесные и процентные эвристические константы. Same-set означает совпавший набор литералов при неверном ответе, а не доказательство конкретной структурной причины.

| Split / вариант | Все ошибки | Число вне Q/констант | Missing | Same-set | Runtime/non-number |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4k / 26b | 2208 | 208 | 579 | 717 | 139 |
| 4k / 30 | 2220 | 214 | 589 | 741 | 97 |
| 4k / 31 | 2363 | 206 | 467 | 876 | 141 |
| 991 / 26b | 526 | 59 | 146 | 181 | 31 |
| 991 / 30 | 541 | 55 | 142 | 184 | 26 |
| 991 / 31 | 555 | 46 | 110 | 205 | 32 |

Собственный вывод: placeholders уменьшили missing на4k579→467, но увеличили same-set717→876; общий результат упал. Compact уменьшил runtime139→97, однако не дал выигрыша exec. У обоих вариантов **restore errors0/5120**. Syntax/runtime после успешного восстановления учтены отдельно. В ранних control обнаружены экстремальные int вне float range:4k ID597/866 и old ID971; taxonomy сохраняет точные int и относит их к outside-number, без падения анализа. Это узкое расширение прежней эвристики.

## CPU-проверка и решение

Заново исполнены **23113 уникальных программ**: все10240 восстановленных S3 generations,10240 control generations двух точек и5120 reference rows до дедупликации. Wall3с/CPU2с,2workers; statuses/численные ответы и полные exec-match sets совпали во всех8 evaluations, reference finite identity совпала. Повторная генерация моделью и CE forward не выполнялись.

По frozen gate требуются ≥2251/4096 и≥528/991 **на1.92M**; compact дополнительно должен пройти equal-time gate. Все необходимые финальные точки отсутствуют → **pending**, ни один вариант не принят. Early placeholders имеют отрицательный сигнал, compact — только выигрыш скорости. Рабочая модель остаётся прежней A6.4M. Fresh/test не читались; CUDA fix, FP16 score и production timer≤180с не проверены; CPU A/B выбор выполнен отдельно.

Следующий шаг: короткая CUDA eval→rollback→backward проверка, затем подготовка resume stable6250 для недостающих точек; новый push требует отдельной команды пользователя.

## Источники

- [s3-data-representations-and-b-scoring](s3-data-representations-and-b-scoring.md) — неизменный протокол до push; [26b-final-wsd-data-scaling](26b-final-wsd-data-scaling.md) — контроль и frozen данные.
- `kaggle-s3` v1: downloaded source/process logs, model-only и stable latest; библиография bibliography — kaggle-s3.
- [Summary](../results/kaggle-s3-v1-summary.json), [SHA/CPU audit](../checks/kaggle-s3-v1-cpu-audit.json), [dtype regression](../checks/kaggle-s3-dtype-regression.py), [анализ](../research/s3-analysis.py).
- [32-candidate-ensemble-s3-results](32-candidate-ensemble-s3-results.md) — отдельный CPU-выбор кандидатов; [data-research-ceiling](data-research-ceiling.md) — taxonomy.
