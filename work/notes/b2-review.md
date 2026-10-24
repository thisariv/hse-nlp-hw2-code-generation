---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-16
updated: 2026-10-16
sources:
  - kaggle-b2-independent-review
  - kaggle-b1-ev-b2
  - kaggle-b1-independent-review
  - dehghani2023-vit22b
  - chowdhery2022-palm
aliases: []
---

# ДЗ2: независимое ревью B2

Проверен commit `80ed3d8` в `codex/nlp-hw2-code-generation`. Рабочее дерево в начале чистое. Прочитаны [b1-ev-b2-plan](b1-ev-b2-plan.md), [b1-review](b1-review.md), [b1-plan](b1-plan.md). EV на thisarivson не изменялся; его API-запуски и статус в этом ревью не трогались. Credentials не инспектировались; SDK использовал только явно выбранный config directory.

Обе модели fresh W: encoder6/decoder6, d768, heads12, SwiGLU2048, 113426435 параметров; seed42, dropout0, batch128, normalized embedding init, AdamW wd0,01, clip1. GPU0 peak0,001, GPU1 peak0,0006. Данные и weighted long2x order/seed прежние; main/cooldown q0.

## Находки и проверка рисков

Пути относительно `work/`, строки после исправления.

| Приоритет / область | Вывод, файлы и регрессия |
|---|---|
| **P1: промежуточная оценка могла израсходовать время финальной** | `kaggle/src/b2_runtime.py:278–287`: раньше второй резерв добавлялся только при `final_session=True`, хотя боевой B2 использует обычную end-main ветку. Теперь резервируются обе оценки, включая 60с между ними. При нехватке времени точка явно пропускается с `partial`; финальная сохраняет свой бюджет. `b2_review_tests.py::test_pair_branch_cannot_spend_end_main_reserve`: настоящий tiny Adam/cooldown/save/restore при63с остатка, две ветки по2с и margin60; оценивается только финал. Старый код делал обе. |
| **P1: production id расходился с каноническим slug** | `kaggle/build_b2.py:54–56`: title`NLP HW02 B2` создаёт ресурс`nlp-hw02-b2`, хотя подготовка ожидала`nlp-hw02-kaggle-b2`. Первый push принят, запросы старого id давали403. Собственный kernels_list нашёл реальный ресурс; pinned metadata подтвердили те же92module SHA. Нового push не было. Builder теперь использует существующий canonical id, `b2_review_api.py:19–24,58` извлекает URL из push response. `test_builder_and_push_receipt_use_canonical_title_slug`: сборка в temp, title/id equality, canonical URL parsing и отказ без URL/с чужим owner. |
| QK-norm / SDPA | `kaggle/src/qk_attention.py:17–56,59–69`: projection → reshape `[batch,heads,tokens,64]` → отдельные Q/K LayerNorm по последней оси → SDPA. Affine shared между головами одного attention, статистики каждой головы отдельные; V не нормируется. FP32 mean/variance/affine с возвратом исходного dtype, eps1e-5. Масштаб $1/\sqrt{64}$ применяется ровно один раз: внутри SDPA, либо явно в ветке `need_weights`. Bool MHA masks переведены в additive masks. Fused encoder отключён. `test_attention_matches_explicit_headwise_math_and_single_scale` сравнивает обе ветки с независимой явной формулой, causal/padding/backward — прежняя регрессия. |
| z-loss / CE | `kaggle/src/b2_optim.py:13–17,22–29,41`: $L=L_{CE}+10^{-4}\sum_{i:y_i\ne PAD}(\log\sum_v\exp z_{iv})^2/N$, где $z_{iv}$ — logit слова$v$ в позиции$i$, $N$ — число target tokens с EOS во всём effective batch. CE/logsumexp вFP32, PAD исключён, каждая microbatch делится на общий$N$. `CE` в логе — чистая CE, `z_loss` — уже взвешенная добавка; evaluation CE также без неё (`l1_eval.py:84–106`). Прежний тест проверяет формулу, shift-инвариантность CE, влияние сдвига на z-loss, PAD gradient0, FP16 logits1000 и одинаковые градиенты micro128/32. |
| Warmup | `kaggle/src/b2_runtime.py:85–105,160–166`: первые2000 **attempted** updates, LR первого0,001/2000 или0,0006/2000, затем constant; skips продвигают warmup. `lr_factor` сохраняется и действует после recovery, нового warmup при restore нет. Точный Adam/RNG next-update и halving наCPU проверены. |
| NaN recovery | `kaggle/src/l2_runtime.py:12–13,37–44`, `b2_optim.py:67–87`, `b2_runtime.py:169–189,299–313`:3 rejected подряд,10/100 или32 AMP skips подряд; обычные отдельные AMP skips LR не уменьшают. Nonfinite model/Adam после шага и Adam OOM немедленно требуют rollback. Первые две эскалации halve, третья останавливает worker с последним good stable. Время/history сохраняются, cursor/model/Adam/scaler/RNG откатываются. Независимый тест проверяет200 изолированных AMP skips и точные три порога. |
| Точки оценки | `kaggle/src/b2_runtime.py:23,223–273,275–287,321–333`: cursor3100032=24219×128 attempted main pairs; cooldown exposures исключены, после него полный restore. Вторая точка — конец main, включая штатную бюджетную остановку. В каждой900с timed linear cooldown + greedy4096/991 и CE. Неполные denominators/nonfinite CE не дают complete. Final-session/K8 выключен. Измеренный throughput определяет wall точки;229пар/с из плана не является гарантией. |
| Stable states | `kaggle/src/b2_runtime.py:108–157`: full FP32 model+Adam/scaler/all RNG/counters/LR/warmup/spec/recipe/runtime; finite проверка до записи, temp→rename, SHA/bytes и atomic latest, два поколения. Strict read проверяет SHA/environment/cursor/Adam steps; `b1_session.py:18–28,47–82` изолирует workers, Secrets удаляются из env. Checkpoint каждые900с wall, до/после оценки и при выходе; авария платформы может потерять последний интервал/выгрузку outputs. |
| KV-cache / v7 / чистая сдача | `exp26_core.py:166–195`, `exp27b_fast.py:23–51`, `final_fast.py:18–46`, `submission/code/hw2_codegen/decode.py:47–48,79–84,106`: K нормируется **до** cache insertion, Q перед SDPA, cross-K один раз. V7 `final_inference_v7.py:51–59` и clean `pipeline.py:44` строят по `model_spec`, strict load. Отсутствующий/false флаг сохраняет legacy; QK tensors с false отвергаются. CPU проверены full-prefix/cached FP32 иFP16, обе реализации; GPU-smoke отдельно проверяет v7 FP16 greedy32/cap128. |
| Смоук: недостаточное покрытие | `kaggle/src/b2_gpu_smoke.py:22–60`: расширен с128 до256 main steps,16peakLR; добавлены sampled max logits, z-loss range, точный training memory peak и peak-LR skips. GPU gate проверяет реальные receipts обеих LR. Пик logits выборочный: первая строка, все головы, первые8 queries, все keys; это не глобальный максимум. |

QK LayerNorm не задаёт строгий постоянный bound после обучения affine. Короткий smoke не доказывает устойчивость многочасового обучения или качество. Вклад QK/z-loss/warmup отдельно этим сравнением не разделяется.

## Квота, входы и ограничения

[Свежая квота аккаунт друга](../checks/b2-review-friend2-quota-final.json),08:26UTC:33568,076с свободно, reserved0, reset17.10 00:00UTC; официальный [`kaggle quota`](../checks/b2-review-friend2-quota-cli.txt) также показал9,32ч. Hard/backend31500с=8ч45мин, watchdog31460с, worker31200с от первой code cell; запас **2068,076с=34мин28с**, больше требуемых20мин. Main cap27000с, setup≤480с, две ветки по1500с, I/O reserve240с; main прекращается заранее для финальной ветки. Перед production push API gate повторно проверил свободную квоту и окончание hard wall минимум за20мин до reset.

[аккаунт друга](../checks/b2-review-friend-quota.json):5975,88с свободно. Smoke backend900с, worker780с/watchdog840с. Account config задаётся через SDK, значения и файлы `~/.kaggle*` не читаются напрямую. Credentials/Secrets в notebook не передаются.

[Friend2 inputs](../checks/b2-review-friend2-inputs.json), [friend inputs](../checks/b2-review-friend-inputs.json): оба видят private notebook cache`thisarivson/nlp-hw02-compact-cache-verified/2` и dataset`thisarivson/nlp-hw02-26b-inputs`. `build_b1.py:80–106` → `l3_resume.py:8`: owner-qualified `/kaggle/input/notebooks/thisarivson/nlp-hw02-compact-cache-verified` и `/kaggle/input/datasets/thisarivson/nlp-hw02-26b-inputs`, затем flat fallback. Перед workers проверяются manifest/payload SHA,9725737rows и weighted order. Реальные пути подтверждаются runtime preflight.

## Проверки и запуск

[5 независимых CPU regressions](../checks/b2-review-regressions.json),9B2 tests,20B1 tests,14clean-package tests прошли;[48всего](../checks/b2-review-cpu.json). [Static inspection](../checks/b2-review-static.json) проверила3private bundles, одинаковые tested runtime sources в smoke/production, QK module вclean и frozen EV pins. GPU-проверка находится ниже; long quality/180с cold inference в задачу не входят.

Smoke v1: [private аккаунт друга](https://www.kaggle.com/code/friend1/nlp-hw02-b2-gpu-smoke/versions?versionNumber=1), API push16.10 **08:16:26UTC**, **COMPLETE**. [Независимый GPU audit](../checks/b2-review-gpu-smoke-v1.json): exits[0,0], wall427,753с=**7мин07,8с**, quota delta459,945с=**7мин39,9с**, одна итерация. Проверены38малых файлов по SHA; суммарный output5945916175байт≈5,95GB, full PT локально не скачивались. Их запись/strict SHA reload/finite/serialized next-update проверены GPU workers. [Серверные92module SHA](../checks/b2-review-smoke-metadata-v1.json) совпадают со smoke и production.

| Метрика | GPU0 /LR0,001 | GPU1 /LR0,0006 |
|---|---:|---:|
| Micro/effective |32/128|32/128|
| Worst-row Adam probe allocated/reserved,GiB |9,083/9,650|9,083/9,650|
| Representative training allocated/reserved peak,GiB |7,631/12,045|7,631/12,045|
| Weighted order,32representative batches после4warmup,пар/с |168,21|165,53|
| Main attempts /peak-LR attempts |256/16|256/16|
| Обычные AMP /nonfinite skips /NaN escalations |0/0/0|0/0/0|
| Sampled max absolute scaled attention logit |6,093|5,770|
| Взвешенный z-loss,диапазон |0,00839–0,01266|0,00838–0,01168|
| Full stable,байт |1361473725|1361473725|
| V7 FP16 greedy32/cap128,с |0,907|0,935|

Обе GPU: serialized next CUDA update bit-exact; injected overflow пропускает Adam, scale1024→512, model/moments не меняются. Настоящие mini greedy64+old16 и CE обоих split,3-секундный cooldown; после restore следующие12main updates побитно совпадают с no-eval control. Latest остаётся **uncooled**256updates/cursor32768. Peak-memory включает representative mixed lengths; worst-row probe имеет отдельный пик. Reserved memory — allocator cache, не только живые tensors.

Грубый прогноз отметки3,1M по измеренным representative batches: **5ч07мин /5ч12мин чистого main** +setup/I/O. Это оценка из короткого smoke, не измеренное время всей траектории; вместо прежних3ч46мин по229пар/с.

Production [аккаунт друга B2v1](https://www.kaggle.com/code/friend2/nlp-hw02-b2/versions?versionNumber=1) отправлен через API **16.10 08:26:47UTC**, **RUNNING08:29:20UTC**. [Push](../checks/b2-review-production-push.json), [status](../checks/b2-review-production-status.json), [private server metadata](../checks/b2-review-production-metadata.json), [gate](../checks/b2-review-production-ready.json). Перед push gate проверил все92module SHA и заново33568,076с свободной квоты; запас2068,076с. ETA≈**20:10–20:20МСК (17:10–17:20UTC)** с учётом старта VM; hard31500с от первой code cell. До17.10 00:00UTC остаётся более6ч, существенно больше20мин.

Реестр — `kaggle/runs.json:b2_review`; ровно один smoke и один production push. Следующий шаг после завершения — скачать complete receipts3100032pairs/end-main обеих моделей, проверить recovery history и сравнить качество с C по фактическим main pairs. EV не затронут.

## Коммит

Попытка `git add` завершилась128: песочница запретила создание `<vault>/.git/worktrees/study1/index.lock` (`Operation not permitted`). Индекс не изменён, `git commit` не запускался, обход ограничений не выполнялся. Все изменения и receipts сохранены в рабочем дереве; [квитанция](../checks/b2-review-git.json).

## Источники

- [b1-ev-b2-plan](b1-ev-b2-plan.md), [b1-review](b1-review.md), [b1-plan](b1-plan.md) — проверяемый протокол; результаты выше — собственное независимое ревью.
- [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) — стандартное масштабирование и семантика mask.
- bibliography — dehghani2023-vit22b, bibliography — chowdhery2022-palm — источники рецепта; новых выводов об их результатах это ревью не делает.
