---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-14
updated: 2026-10-14
sources:
  - kaggle-l3-final-results
  - kaggle-l2-preparation
aliases: []
---

# L4: продолжение обеих C из финального stable L3

Выбран **вариант (a): продолжить C-main и C-long2x**. [l3-results — Финал L3](l3-results.md#финал-l3): mixed K4+K4+greedy1511/2048 против single long K81507, +0,20п.п. [−0,84;1,23]. Вклад main при равном K мал по точечной оценке, но неизвестно, даст ли ответвление второй long с новым seed больше. Mixed16 выигрывает у long8 на1,42п.п. [0,49;2,35], однако K удвоен. Убедительного основания заменять main нет; для неочевидного вывода пользователь задал (a). Новый seed/sampler, clean cooldown и новые данные в этот L4 не входят.

## Строгое продолжение

Источник: **private friend1/nlp-hw02-l3-live v12**, оба uncooled latest, полный model/AdamW/scaler/scheduler/RNG/history. FP16 после cooldown — только для инференса. [Pins](../kaggle/l4-anchors.json):

| Модель | Full-state SHA256 | updates | cursor | main_s | l2_main_s |
|---|---|---:|---:|---:|---:|
| C-main | c6ad6dda249f5cd87e598a29fd7714549fc8e25b5f00a45b92bbf3027e2e53e3 | 213386 | 27313408 | 79850,520 | 62280,584 |
| C-long2x | 48f2ba3fa4ffe6a722b45a084d3ee33a27cc4ed02b72433605c9cb1740f5a80b | 209144 | 20734592 | 79850,429 | 62280,493 |

Сохраняются LR0,0015/effective128/micro64, identities и обе перестановки: ordinary09cf6db8…/weighted72bc3dfd…; seed weighted2026101302, compact threshold79, short/long multiplicity1/2. Cursor/history не сбрасываются, warmup не повторяется. Пять inherited points должны быть complete с точными tags10800/21600/28800/45540/62280; проверяется latest↔manifest↔pin, SHA/bytes, finite FP32+Adam и recipe/runtime SHA до обучения.

Точки L4: **l2-77580**, **l2-92880** (точный inherited l2_main_s каждой C +15300/+30600). Старые tags не повторяются. До и после disposable cooldown сохраняется/восстанавливается полный stable state; ошибки eval дают partial, main остаётся восстанавливаемым.

## Квота от первой code cell

| Фаза, параллельно на двух T4 | Cap, с |
|---|---:|
| Gates, обе anchors, private live probe, install, cache/order | 480 |
| Main:15300+15300 | 30600 (8ч30мин) |
| Один mid cooldown / mid eval | 900 /300 |
| Final cooldown / greedy+CE+old+test+K8 tune | 3000 /1800 |
| Checkpoint/restore/I/O /worker stop reserve | 180 /120 |
| **Worker deadline** | **37380 (10ч23мин)** |
| Final live sync /packaging watchdog reserve | 300 /120 |
| **Hard wall** | **37800 (10ч30мин)** |

Backend ceiling12ч остаётся аварийным внешним пределом; **watchdog37800с** завершает локальную сессию. Budget включает обе точки; при исчерпании времени stable checkpoint сохраняется, неполная оценка не объявляется complete. По L3 фактическому wall38000с−2880с main≈35120с (9ч45мин); это эмпирический сценарий, не гарантия throughput или upload. При остатке11–11,5ч hard wall оставляет30–60мин для смоука/резерва; расходы двух T4 учитываются как session clock Kaggle.

## Бандлы и проверки

[build_l4.py](../kaggle/build_l4.py) вызывает параметризованный [build_l3.py](../kaggle/build_l3.py); L2 runtime/recovery/eval/live/sampler не изменены. Обёртка l3_resume принимает expected_points5; worker выбирает continuation для l3/l4. Локальные бандлы `bundles/l4` и `bundles/l4_gpu_smoke`; **push=0, GPU=0**, ревью и смоук отдельно.

Private notebook **friend1/nlp-hw02-kaggle-l4**, internet on, T4×2; inputs l3-live, thisarivson/26b-inputs и pinned compact-cache-verified/2, при необходимости friend cache fallback. Target live **friend1/nlp-hw02-l4-live**; выбор аккаунта только `KAGGLE_CONFIG_DIR=~/.kaggle-friend`. Credential directories/значения не инспектируются, ничего на Kaggle не удаляется. Для production потребуется ранее заданный attached friend Secrets gate.

CPU проверки и remote live receipt будут записаны в `checks/kaggle-l4-*.json`. Реальные full states скачиваются по одному, проверяются двумя batch128 шагами reference↔production bit-exact и удаляются вместе со scratch checkpoints. CPU FP32/enabled scaler/Adam без fused не доказывает CUDA FP16/fused parity. Финальный inference v4 — отдельный бандл, [final-inference — V4 на финальных весах L3](final-inference.md#v4-на-финальных-весах-l3).

## Источники

- **kaggle-l3-final-results**: [l3-results — Финал L3](l3-results.md#финал-l3), локальный snapshot и private livev12, [CPU-анализ](../research/analyze-l3-final.py), [результаты](../results/l3-final-results.json); bibliography.
- [l3-plan](l3-plan.md), [l3-review](l3-review.md), [l2-plan](l2-plan.md); пользовательский предел10ч30мин и требование выбора(a), если преимущество замены неясно.

Private l4-livev1 создан от аккаунта друга и ready; [receipt](../checks/kaggle-l4-live-created.json). Первый status endpoint временно403; повторная read-only проверка подтвердила private mine listing и ready. Создание не повторялось. Remote sourcev12 и оба FP16 подтверждены [listing](../checks/kaggle-l4-source-access.json).

Проверки завершены: [L2 34/34](../checks/kaggle-l2-regressions.json), [L3 10/10](../checks/kaggle-l3-regressions.json), [L4 3/3](../checks/kaggle-l4-regressions.json) (неполный/неверный history/K8 отказ, новые tags, реальный worker routing обеих C); [C-main resume](../checks/kaggle-l4-real-resume-C-main.json), [C-long2x resume](../checks/kaggle-l4-real-resume-C-long2x.json) — по2actual batch128 шага reference↔production бит-в-бит, точное восстановление model/Adam/scaler/RNG/state и indices. [Статика](../checks/kaggle-l4-static.json): nbformat/empty outputs/compile/source parity, frozen L2 modules, latest↔producer manifest↔pins, private metadata, exact budgets. GPU/push=0.

Оба full PT удалены после проверки; temporary checkpoint copies очищены. Для long исходный order.npy отсутствовал в локальном L2 snapshot: скачан45MB из L3 live, проверен SHA72bc3dfd… и прежняя metadata; после проверки temporary order также удалён. Existing FP16/results/cache не удалялись. CPU Torch2.14.1 отличается от pinned GPU2.11.0+cu128; resume проверяет одну и ту же CPU-математику с восстановленными реальными states, CUDA/fused/AMP gate остаётся отдельному ревьюеру.
