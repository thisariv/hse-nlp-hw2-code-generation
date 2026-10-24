---
type: experiment
course:
  - nlp
status: learning
understanding: 4
created: 2026-10-16
updated: 2026-10-16
sources:
  - kaggle-b1-independent-review
  - kaggle-b1-ev-b2
  - dehghani2023-vit22b
  - course-methods-instability
  - course-methods-olmo
  - chowdhery2022-palm
aliases: []
---

# B1 EV и B2: оценки и устойчивость W

Независимое ревью и фактический запуск B2 — [b2-review](b2-review.md): smoke аккаунт друга COMPLETE, production canonical `friend2/nlp-hw02-b2/1` RUNNING. Числа подготовки ниже сохранены как исходный план; измеренная скорость и исправления — в ревью.

Решение пользователя от16.10: **вся квота аккаунт друга остаётся для B2**. EV — thisarivson; B2 GPU-smoke — независимому ревьюеру на аккаунте друга. B2 здесь не запускать и не push. [b1-plan](b1-plan.md) и [b1-review](b1-review.md) — исходный протокол.

## Что осталось от B1

`friend2/nlp-hw02-kaggle-b1/1` завершился ERROR на итоговом assertion; main обеих моделей достиг34 200с. По данным пользователя E@9000=49,83%, W@9000=49,98%, W@34200=60,74%; отсутствуют оценки E@28370/E@34200/W@28370. У обеих было по две NaN-эскалации, peak0,001→0,0005→0,00025; у E около22k updates, у W около32k. Эти значения подтверждены [producer-SHA проверенными summary](../checks/b1-final-summary.json); это ещё не результат EV.

[Реальный inventory](../checks/b1-ev-source-inventory.json) и producer manifest подтверждают:

| Артефакт | E / GPU0 | W / GPU1 |
|---|---|---|
| Готовый FP16, cooldown900с, q0 @28370 | `weights/probe-028370-fp16.pt`,217457211байт | тот же путь,226916891байт |
| Последний uncooled full @34200 | `checkpoints/main-63bf1a6d8aa7.pt`,1304613251байт | `checkpoints/main-7bb6a1232f35.pt`,1361386515байт |
| Main / attempted updates / cursor |34200,3718с /51472 /6588416 |34200,4395с /53789 /6884992 |

Пути относительно `model-GPU{0,1}/`. В output также есть `main-33e1c16da2fa.pt` и `main-5c5fee482c41.pt`; прежние `ce826…/af922…/8cc98…/df337…` ротация удалила. FP16 `end-main-034200` обеих тоже есть, но по решению пользователя final EV начинается от latest stable. Full SHA — в inventory и transfer receipt.

## EV: private thisarivson, ≤45мин

Новый Dataset `thisarivson/nlp-hw02-b1-ev-inputs`: две готовые FP16@28370, две latest/full@34200, order, atomic latest и checkpoint receipts. `b1_transfer.py --ev-inputs` последовательно скачивает **один** payload pinned B1v1, сверяет producer SHA/bytes, загружает blob от thisarivson и удаляет локальный payload. Затем создаёт private Dataset, проверяет ready/private/размеры и скачивает каждый remote файл по одному для полного SHA round-trip. Upload descriptors не выводятся. [Receipt](../checks/b1-ev-transfer.json).

EV notebook: `kaggle/bundles/b1_ev/`, `notebooks/nlp-hw02-b1-ev.ipynb`. Входы — новый Dataset, `thisarivson/nlp-hw02-26b-inputs` и notebook cache `thisarivson/nlp-hw02-compact-cache-verified/2`. Прямой mount output другого аккаунта не нужен. Два отдельных процесса/T4, E наGPU0, W наGPU1; Secrets/live-sync отсутствуют.

В каждом worker:

1. Strict load final stable: SHA, model/Adam/scaler/RNG, recipe/Torch/CUDA/order, finite tensors. **Исходные реализации B1 встроены из опубликованного локального B1 notebook**, без изменений QK/B2; [пины runtime](../checks/b1-ev-runtime-pins.json).
2. C timed linear cooldown900с из last LR0,00025: $\eta(t)=\eta_0(1-t/900)$, где $t$ — сумма времени cooldown updates. q0; cursor продолжает anchor, main не продвигается. Как в B1, последний update использует LR по времени **до** него; дополнительный нулевой update не добавляется.
3. Сохранить FP16 `weights/ev-034200-fp16.pt`; оценить greedy heldout4096 + old991 и token CE обоих. Затем оценить готовую FP16@28370. Final E и W-control получают первый приоритет.

На оценку точки cap700с; setup≤240с, worker2640с, watchdog2680с, server2700с от первой code cell. Процессы здоровой GPU продолжаются независимо при сбое другой. Результаты: `model-GPU*/ev-results.json`, `summary.json`, `eval-{34200,28370}/metrics.json`, greedy per-example journals/generations/executions; общий session summary и SHA manifest. Main training updates=0; только явно запрошенный cooldown.

**EVv1 запущен16.10 в07:51:59UTC; RUNNING подтверждён повторно в08:10UTC. ETA≈11:37МСК.** Запуск через API thisarivson после CPU/static и transfer SHA; [push](../checks/b1-ev-push.json), [status](../checks/b1-ev-status.json). ETA и фактическая версия занесены в `kaggle/runs.json`. GPU-smoke EV не нужен.

## B2: fresh W, сравнение двух LR

Обе модели e6-d6/d768,12heads/SwiGLU2048, **113426435 параметров**, общий seed42 и одинаковые начальные веса/weighted order. GPU0 `W-lr1000`:peak0,001; GPU1 `W-lr600`:peak0,0006. Batch128, cap micro32 с настоящим Adam allocation probe/fallback; FP32 параметры + AMP FP16/GradScaler1024, fused AdamW wd0,01, clip1, dropout0, q0, normalized token init. Кеш и long2x order/seed2026101501 сохранены из B1.

Изменения относительно B1:

- **QK LayerNorm** по последней размерности каждой головы ($d_h=64$, affine scale/bias, eps1e-5), во всех encoder self / decoder self / cross attention; статистики вFP32. $Q'=\operatorname{LN}(Q)$, $K'=\operatorname{LN}(K)$, $A=\operatorname{softmax}(Q'K'^T/\sqrt{d_h}+M)V$, где $M$ — causal/padding mask. Нормировочные параметры общие для голов одного attention, нормировка каждой головы независима. Это выбранный headwise вариант, а не полное воспроизведение архитектуры OLMo2.
- **z-loss**: $\mathcal L=\mathcal L_{CE}+10^{-4}\,N^{-1}\sum_{i:y_i\ne PAD}[\log\sum_v\exp z_{iv}]^2$. $z_{iv}$ — выходной logit токена $v$ на позиции $i$; $N$ — число непустых target tokens всего effective batch. FP32 logsumexp, PAD исключён; CE и z-loss логируются раздельно. Обе части token-weighted через microbatches; cooldown использует тот же loss.
- **Warmup2000 attempted updates**, затем constant; emergency LR factor продолжает действовать после restore. Остальной рецепт C/B1 сохранён.

Мотивация из источников: QK-norm ограничивает рост масштаба attention logits, z-loss штрафует дрейф log-partition выхода. **Гипотеза B2**: вместе с более длинным warmup этого достаточно против наблюдавшейся нестабильности W. Устойчивость и качество этим ещё не доказаны; сравнение двух LR не разделяет вклады трёх изменений.

## Отметка3,1M и wall budget

$3\,100\,000/229=13\,537{,}12$с, то есть **3ч45мин37с чистого main**. Из-за batch128 реальный gate — **3100032 pairs /24219 attempted main updates**. `next_point_pairs` проверяет **cursor**, без cooldown exposures. Wall отметки равен измеренному main +setup+probe+save/recovery pauses; оценочно3ч50мин–4ч при229пар/с. Реальная QK/z-loss скорость будет измерена ревьюером и может сдвинуть wall. В B1 smoke weighted W было207,99пар/с, поэтому229 — ориентир исходного shape-bench, а не обещание нового throughput.

На gate и в конце: отдельная disposable ветка900с C timed linear cooldown/q0 → greedy heldout4k, old991 и CE → полный model/Adam/scaler/RNG/cursor restore. Ориентир пользователя: C62,87% после≈3,1M пар/9000с. Сравнивать именно фактические main pairs в training receipt.

[Квота аккаунт друга](../checks/b2-quota.json): **33568,076с=9,3245ч**, reserved0; reset17.10 00:00UTC. Hard31500с=**8ч45мин**, остаётся2068,076с≈34мин28с; B2-smoke тратит аккаунт друга, EV — thisarivson. Worker31200с, watchdog31460с; setup cap480с, main cap27000с (7ч30мин чистых updates), две ветки по900+600с, I/O reserve240с. Расчёт30720с оставляет480с до worker deadline для checkpoint/I/O overhead. Каждые900с wall — atomic full stable.

**Финальная ветка зарезервирована явно**: main прекращается заранее при остатке900+600+240+60с, даже если27000с не достигнуты. Достигнутый main cap записывается отдельно; успешная бюджетная остановка с обеими completed оценками считается complete. Неполная оценка остаётся partial; аварийный stable сохраняется. Это устраняет ошибку фиксированного main cap, оставившую B1 без финальных оценок.

## Логи, resume и inference

`updates.jsonl`: каждый attempted main update — CE/z-loss, grad-norm до clip, AMP/nonfinite skips, LR, main pairs/time. Каждые100updates — **sampled** max absolute scaled attention logit: первая строка batch, все головы, первые8queries, все keys; без полного attention matrix. Это диагностическая выборка, не глобальный максимум. `recovery-events.jsonl` и `stability-history.json` фиксируют rollback/halving; журналы сохраняются при откате counters. История normal AMP skips отделена от nonfinite rejection. Cooldown instability даёт incomplete ветку.

Output: `model-GPU{0,1}/latest.json`, два full поколения, points/FP16, per-example eval. Payload сохраняет spec с `qk_norm: layernorm`, model/Adam/scaler/all RNG, warmup/cursor/LR-factor, счётчики и историю. B1 transfer формат сохранён, разрешены имена `W-lr1000`/`W-lr600`; обычный и emergency путь подходят для продолжения после reset. Strict runtime/Torch/CUDA SHA gate сохранён; после несовместимой смены image нужен отдельный разбор, fresh fallback отсутствует.

```sh
# Только подготовка; команды не запускают GPU.
python build_b2.py --owner friend2
python build_b2.py --owner friend1 --gpu-smoke
python b1_ev_b2_tests.py
python b1_ev_b2_static.py

# После завершения B2, pinned source; новый Dataset нужного владельца.
python b1_transfer.py --source friend2/nlp-hw02-kaggle-b2/1 \
  --to thisarivson --dataset nlp-hw02-b2-resume-v1 --receipt b2-transfer.json
# Если общий manifest отсутствует: --emergency-receipts.
python build_b2.py --owner thisarivson --resume-source thisarivson/nlp-hw02-b2-resume-v1 \
  --transfer-sha <SHA-из-receipt>
```

[Готовность к независимому ревью](../checks/b2-prepared.json). Бандлы: `kaggle/bundles/b2_friend2/` и `kaggle/bundles/b2_gpu_smoke_friend1/`. [API подтвердил доступ аккаунт друга](../checks/b2-smoke-inputs.json) к private cache v2 и thisarivson inputs. Reviewer smoke≤900с: реальный memory probe с z-loss,128 representative steps/throughput,16peak LR, injected AMP overflow, serialized exact next update, mini greedy/CE, короткий cooldown с restore/12следующими steps и настоящий v7 QK-aware FP16 greedy32/cap128. Боевой push и этот smoke выполняет независимый ревьюер.

QK-aware training, обычный greedy evaluator, v7 `final_core/final_fast`, `CachedDecoder` и чистый `submission/code/hw2_codegen` используют один порядок нормировки: K нормируется до помещения в cache, Q перед SDPA. Старые spec без флага сохраняют исходную архитектуру; strict load отвергает QK tensors при выключенном флаге. V7 builder явно добавляет QK module; notebook/bundle пересобраны. Состав прежнего ансамбля не менялся.

## Проверки и следующий шаг

[CPU](../checks/b1-ev-b2-cpu.json):9tests — headwise norm/masks/backward, cached logits=full prefix (training eval/v7/clean), z-loss/mask/FP32/gradient accumulation, настоящий CPU AMP overflow skip, strict QK/legacy loading, точный Adam/RNG resume и halving, pair-gate/end-budget restore, перенос B2 через B1 transfer, frozen B1 cooldown-only. Дополнительно20B1 regressions и14clean submission tests (3QK standalone) прошли; [полный v7 CPU/static](../checks/final-inference-v7-cpu.json) тоже прошёл на пересобранном bundle. [Статика](../checks/b1-ev-b2-static.json) проверяет metadata, все embedded SHA, frozen B1 pins, quota reserve, двух owners B2 и EV. GPU устойчивость/throughput B2 ещё не проверены.

Следующий шаг: результаты EV и независимое ревью B2 → smoke наfriend1 → отдельно разрешённый production push отfriend2 с повторной проверкой квоты. B2 smoke/push в этой работе0.

Попытка коммита: `git add` завершился128 — sandbox запретил создание shared Git `index.lock` (`Operation not permitted`). Индекс не изменён, обход ограничения не выполнялся. [Квитанция](../checks/b1-ev-b2-git.json).

## Источники

- [b1-plan](b1-plan.md), [b1-review](b1-review.md), [architecture-decision](architecture-decision.md) — локальный протокол и замеры.
- Dehghani et al. (2023), [Scaling Vision Transformers to22 Billion Parameters](https://arxiv.org/abs/2302.05442) — QK normalization; bibliography — dehghani2023-vit22b.
- Wortsman et al. (2023), [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322) — attention/output logit instability; bibliography — course-methods-instability.
- Team OLMo et al., [2 OLMo2 Furious](https://arxiv.org/abs/2501.00656), декабрь2024/arXiv2025; [анонс Ai2, ноябрь2024](https://allenai.org/blog/olmo2) — QK-norm и z-loss как части более широкого рецепта; bibliography — course-methods-olmo.
- Chowdhery et al. (2022), [PaLM](https://arxiv.org/abs/2204.02311) — z-loss; bibliography — chowdhery2022-palm.
