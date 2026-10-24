---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-14
updated: 2026-10-14
sources:
  - kaggle-p1-preparation
  - kaggle-l3-final-results
  - hagele2024-cooldown
  - kang2020-loss-truncation
  - zhang2022-coder-reviewer
aliases: []
---

# P1: длинный cooldown и модель-рецензент

**Решение пользователя: подготовленный L4 не запускать.** Вместо него один private P1 на аккаунте друга, T4×2, Internet on. Hard wall **37 800 с = 10ч30мин**, от первой code cell; GPU-smoke и боевой push выполняет независимый ревьюер. Качество P1 ещё не измерено. [Финал L3](l3-results.md#финал-l3) — контроль; [консенсусная фильтрация](clean-cooldown-analysis.md) здесь не используется.

## Гипотезы и контроль

1. **A / long:** более длинный `1−sqrt` cooldown может улучшить execution accuracy C-long2x относительно финального линейного 50-минутного cooldown L3. Сравнение A−L3 меняет одновременно длительность и форму; их отдельные эффекты не идентифицируются.
2. **B / long + loss truncation:** исключение верхних 10% per-example CE внутри каждого effective batch может уменьшить вклад шумных пар. Сравнение B−A меняет только loss; высокий loss также бывает у трудных правильных задач. Улучшение не обещается.
3. **Reviewer:** $\log P(q\mid c)$ может помочь выбрать согласованный с вопросом код среди уже сохранённых K8-кандидатов. Это исследовательская адаптация идеи Coder–Reviewer, а не повтор эксперимента статьи: отдельная модель обучается с нуля на обращённых исходных парах.

Новых данных/сгенерированных обучающих таргетов нет. Heldout/reference применяются только для оценки и офлайн-подбора selector; выполнение кандидатов не становится supervised-таргетом reviewer. Результат ещё не выбран для сдачи; дополнительный reviewer-inference должен отдельно пройти ограничение задания **180 с на одной T4**.

## Одна stable-точка для A и B

Private источник **friend1/nlp-hw02-l3-live v12**, `C-long2x latest`:

- `model-C-long2x__checkpoints__main-2a27307d7a48.pt`, **644 197 075 байт**;
- SHA256 `48f2ba3fa4ffe6a722b45a084d3ee33a27cc4ed02b72433605c9cb1740f5a80b`;
- updates **209 144**, cursor **20 734 592**, main_s **79 850,428875**, l2_main_s **62 280,492996**;
- full FP32 model/AdamW/GradScaler/RNG/history, LR **0,0015**, effective128, micro64;
- inherited weighted order SHA `72bc3dfd3a3e738979c489c0cf8f41cde092c41d4ae90f2de8378a034603a4db`, 11 227 811 slots.

Это restored stable latest после финального cooldown L3: его model/Adam/RNG/cursor относятся к исходной точке до затухания; история уже включает финальную оценку. Producer `training.json` финала L3 указывает те же **209144 /20734592**. SHA предыдущего full PT до добавления истории отличается — не подменять latest тем PT или cooled FP16. [Пин](../kaggle/p1-anchors.json) наследуется из проверенного L4 pin; startup сверяет latest↔recover-manifest↔pin и исходный Torch2.11.0+cu128/CUDA12.8 до install/cache.

GPU0 не продолжает constant-LR main. Ревью добавляет сравнение SHA tensor-bytes model/Adam/scaler, всех RNG и cursor/history перед A/B и после каждого restore. Перед **каждой** веткой и после неё читается тот же неизменяемый full PT и полностью восстанавливается состояние. A не влияет на B. Обе ветки получают тот же circular weighted order с того же cursor; итоговый SHA потока индексов обязан совпасть. Stable source не перезаписывается.

## Длина и форма cooldown

$$
T=\operatorname{round}(0{,}15\cdot209144)=31372,
\qquad \eta_j=0{,}0015\left(1-\sqrt{j/T}\right),\quad j=1,\ldots,T.
$$

$T$ — число attempted effective-batch updates; $j$ — шаг ветки; $\eta_j$ — LR. В начале непрерывного расписания $\eta_0=0,0015$, первый дискретный update немного ниже; последний имеет **LR0**. Нулевой floor соответствует terminal cooldown в Hägele et al.; дополнительный положительный floor здесь не вводится. Доля от полного пути с новой веткой — **13,04%**, а не 15%: пользователь задал долю от уже накопленных updates. Disposable cooldowns L1/L2/L3 не прибавляются к stable update counter. AMP/nonfinite skips логируются отдельно, не выдаются за успешные updates.

Ось — updates, а не время: обе ветки предъявляют **4 015 616 пар** и имеют одинаковый LR на соответствующих батчах. Это контролирует поток данных, хотя B работает дольше. Фактический финал L3: **7572 updates /3000,102 с update time**, 3007,723 с wall; перенос даёт **12 430 с ≈3ч27мин** для A. Ориентир 2,5–3ч пользователя оказался ниже этого локального расчёта. Первоначальный B делал дополнительный selection-forward (+20–35% ожидаемого времени). Независимое ревью заменяет его одним forward с графами всех128 до глобального отбора; при OOM — точный двухпроходный fallback. Скорость и cap проверяются на T4 до production push.

Если cap/deadline не позволяет закончить все31372, ветка записывается **incomplete** и не получает фиктивный полный финал. После восстановления следующая ветка всё равно может работать; падение GPU0 не останавливает GPU1.

## Loss truncation

$$
\ell_i=\frac{\sum_{k=1}^{n_i}\operatorname{CE}_{ik}}{n_i},
\qquad L=\frac{\sum_{i\in K}\sum_{k=1}^{n_i}\operatorname{CE}_{ik}}{\sum_{i\in K}n_i}.
$$

$i$ — позиция примера в effective batch, $n_i$ — число target-токенов с EOS, без PAD; $\ell_i$ — loss для ранжирования; $K$ — оставленные позиции. Сначала считается $\ell_i$ **для всех128** с сохранением microbatch-графов, затем выбирается общая маска, потом backward с token normalization по оставшимся. При OOM графы/градиенты удаляются, RNG/scaler восстанавливаются; fallback делает no-grad selection и повторный forward/backward. Dropout0. Отбор не делается отдельно в microbatches, не меняет order/cursor и не заменяет исходные code targets.

При128 ближайшее целое к10% — **13 исключений /10,15625%**, остаётся115. Квоты пропорциональны размерам bins, остатки распределяются детерминированно; в каждом непустом bin остаётся хотя бы один пример. Equal loss разрешается по позиции батча, включая повторные row IDs weighted order. Если маленький synthetic batch не позволяет сохранить bins, число исключений уменьшается и отражается в журнале.

**Compact proxy bins:** `<53`, `53–79`, `79–105`, `≥105` content-токенов, включая mode и без EOS. Граница79 — прежний L2 compact-аналог90; остальные округлены по масштабу79/90. Это приближение, не исходные BPE60/90/120: raw offsets во входах P1 нет. Для quality evaluation остаются **точные original BPE bins L2**. На каждый update B сохраняет `truncation.jsonl`: позиции/row IDs исключений, source/target lengths, долю, CE retained/excluded, квоты и токены; итоговые totals — в training.json.

Из L2 переиспользованы finite checks, OOM retry со снижением microbatch, scaler rollback, NaN/AMP threshold и full restore. В disposable A/B threshold или небезопасный Adam/OOM **отбрасывает ветку и восстанавливает L3**, сохраняя protocol LR: изменение peak ради получения результата нарушило бы чистоту A/B. Reviewer использует bounded stable rollback/emergency LR halvings; такие случаи остаются явной сменой его recipe.

## Оценка GPU0 и артефакты

После **каждой** полной ветки — тот же `l2_eval.evaluate`, что у L3: greedy heldout4096/old991, compact token CE, полный test1919 pipeline timer без gold; K8 на fixed tune2048, **T0,7 /seed270042 /batch32 /cap512**. Сохраняются все execution/represented_code/LP журналы. Complete требует K8 `complete=true`, rows=target_rows=2048. Test не используется для выбора модели.

FP16 model-only обеих веток находятся в `model-pilot-a/weights/long-fp16.pt` и `model-pilot-b/weights/long-truncation-fp16.pt`. Отдельные model roots позволяют прежнему live staging сохранить **обе** точки в последней версии. `model-pilot/latest.json` — неизменённый stable L3, main-added0. Сравнения A−L3 и B−A выполняются позже офлайн, с paired CI на heldout/eval и отдельными длинами; tune и old — диагностика. До просмотра результатов критерий убедительного улучшения: положительная нижняя граница97,5% paired CI для двух сравнений на eval2048; иначе результат неопределённый, L3 остаётся контрольной моделью. Повторное использование eval не даёт независимого финального test.

## GPU1: reviewer с нуля

Architecture C **6+6 /d512 /heads8 /SwiGLU1368 /vocab8195**, тот же normalized token init и seed42, dropout0, AdamW LR0,0015/WD0,01/clip1, AMP FP16+GradScaler/fused, effective128. Encoder получает `[BOS] + cached compact target с mode + [EOS]`. Decoder получает BOS + исходный вопрос без его source BOS/EOS; labels — вопрос +EOS. PAD не входит в loss. Проверяется новый longest-row memory probe для обращённого направления.

Обычная исходная bucketed permutation кеша сохраняется; без long2x weighting. Все sealed-cache protected exclusions остаются; дополнительно исключён выявленный [normalized test overlap](clean-cooldown-analysis.md#protected-exclusions), global row **8545870**. GPU0 сохраняет ровно historical L3 stream, чтобы не добавлять смену data policy в A/B. Reviewer получает **9 725 736** исходных пар; никаких labels correctness или новых текстов.

Warmup1000 updates → constant LR → final `1−sqrt` cooldown на последних **10% доступного training wall**. Из сессии заранее вычтены CE900с и I/O180с; точные границы фиксируются после probe. Stable checkpoints каждые1800с и **перед cooldown**; CE на обращённых heldout4k/old991 каждые3ч и в финале. `model-reviewer/latest.json` остаётся **uncooled full-state** для продолжения; финальные FP16 — `model-reviewer/weights/final-fp16.pt`. При нормальном setup/probe доступно≈9ч55мин training wall, cooldown≈59мин. Setup/probe/периодические CE/I/O могут уменьшить фактическую долю updates; логировать её, не объявлять ровно10% updates.

## Бюджет от первой code cell

| Фаза | Cap / резерв, с |
|---|---:|
| Secrets/T4 gate, anchor/cache/all-shard SHA, install, private live probe |480|
| GPU0 A,31372updates |13 800 (3ч50мин)|
| GPU0 A greedy/CE/test/K8 |1800|
| GPU0 B,31372updates |17 400 (4ч50мин)|
| GPU0 B greedy/CE/test/K8 |1800|
| GPU0 checkpoint/restore I/O |180|
| Штатный stop |120|
| Нераспределённый worker запас |1800|
| **Worker deadline** |**37 380 (10ч23мин)**|
| Final live sync, в пределах оставшегося времени |до300|
| Hard watchdog/packaging reserve |120|
| **Hard wall** |**37 800 (10ч30мин)**|
| Backend ceiling, дополнительный внешний предел |43 200|

GPU1 параллелен этому бюджету: всё оставшееся до worker deadline минус CE900/I/O180; последний10% training wall — cooldown. При≈13ч friend quota после боевого hard wall остаётся≈2ч30мин до smoke/прочих расходов; smoke hard1020с =17мин оставляет≈2ч13мин. Kaggle квота задана пользователем, её реальное остаточное значение перед запуском проверяет ревьюер. Смоук: те же C/full L3/AMP/effective128, обе ветки64updates, greedy64/old16/test256/K8 на36 tune IDs; reviewer warmup20, короткая WSD, worker900/hard1020/backend1080. Expected GPU0 crash после успешных проверок подтверждает независимость reviewer; общий summary partial ожидаем, первичен `p1-smoke-summary.json`.

## Входы, live и ограничение диска

- Private notebook **friend1/nlp-hw02-kaggle-p1**; новый private live **friend1/nlp-hw02-p1-live**, заранее создан от аккаунта друга.
- Datasets: `friend1/nlp-hw02-l3-live`, `thisarivson/nlp-hw02-26b-inputs`; original kernel cache `thisarivson/nlp-hw02-compact-cache-verified/2`.
- Явный fallback: `friend1/nlp-hw02-compact-cache-verified`; пересобрать с `--cache-source`, не обходить403. Cache manifest SHA `46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be`.
- Пути: owner-qualified `input/datasets/<owner>/<slug>`, `input/notebooks/<owner>/<slug>`, затем собственный flat mount; `resolve_dataset`/`resolve_cache_mount` и SHA, по [l3-review](l3-review.md).
- Friend CLI **только** `KAGGLE_CONFIG_DIR=~/.kaggle-friend`; credential directories/значения не читать. Production требует attached friend Secrets и owner-private upload+ready startup probe; workers очищены от API/Secrets/IAP env. Live heartbeat5400с, insurance7200с, nonfatal bounded sync, final reserve внутри hard wall.
- Cache read-only memmap; GPU0 одна stable full копия, GPU1 два stable full с ротацией и один transient, три FP16≈322MB, order≈45MB. Обычный output≈2,3–3GB плюс журналы; live последний snapshot≈1,7GB, ниже прежних2,5GB. Stage/CLI ZIP временные вне working. Не скачивать весь kernel output/кеш заново; один644MB PT для CPU, затем удалить. На Kaggle ничего не удалять.

## CPU-проверки и независимое ревью

Квитанции: [L2](../checks/kaggle-l2-regressions.json), [L3](../checks/kaggle-l3-regressions.json), [L4](../checks/kaggle-l4-regressions.json), [P1](../checks/kaggle-p1-regressions.json), [real resume](../checks/kaggle-p1-real-resume-C-long2x.json), [статика](../checks/kaggle-p1-static.json), [scorer smoke](../checks/reviewer-score-cpu-smoke.json). CPU FP32 с восстановленным enabled scaler и Adam без fused проверяет два batch128 updates reference↔P1 bit-for-bit и два полных restore; CUDA/FP16/fused numerical identity, память/throughput и real live-auth проверяет ревьюер.

Scorer [research/score-reviewer.py](../research/score-reviewer.py) читает **все32768 K8 samples L3 final** (обе модели), именно `represented_code`, считает sum/mean log P(question|code), включая question EOS. Дедупликация forwards не меняет голоса. Селекторы: log cluster count+β·max reviewer score; logsumexp β·score по кластеру; max/mean score. Numeric clusters — прежние isclose1e-6 connected components; tie count/greedy/coder LP/order. β-grid фиксирован заранее; две половины tune с группировкой одинаковых normalized questions, fit на другой половине, out-of-fold paired bootstrap95% CI против собственного K8 voting. Main8/long8/mixed4+4 анализируются отдельно; sum/mean variants диагностические, лучший из них не становится независимым победителем.

Scored chunks имеют identity/SHA и допускают CPU resume. Random tiny smoke использует128 реальных candidates, проверяет все32768 baseline vote decisions без model forwards и не интерпретирует качество случайной модели. Офлайн-scoring полного reviewer на CPU может быть долгим; в P1 GPU-бюджет не входит.

Ревьюер: source/CPU receipts → private GPU-smoke → audit `p1-smoke-summary`, A/B restore/поток/CE/K8, reviewer reverse CE/stable state, memory/projection wall → затем private боевой push/UI Save & Run All с friend Secrets. **L4 Save & Run All не выполнять.** Если smoke показывает, что B не помещается в cap, остановиться и согласовать сокращённый *новый* paired протокол; не молча менять T/q/peak внутри P1.

## Источники

- **Содержание исследований:** [Hägele et al.2024](https://arxiv.org/abs/2405.18392), constant LR и terminal cooldown, форма1−sqrt; [Kang & Hashimoto2020](https://arxiv.org/abs/2004.14589), robust loss truncation; [Zhang et al.2022](https://arxiv.org/abs/2211.16490), вероятность инструкции по коду и reranking. Все три добавлены в bibliography.
- **Локальные измерения:** [l3-results — Финал L3](l3-results.md#финал-l3), [l3-review](l3-review.md), [l4-review](l4-review.md), L3 livev12/latest, сохранённые training/K8 receipts; [final-inference](final-inference.md) — прежний timer/format ограничения сдачи.
- **Собственный протокол:** доля15% от stable updates, длиновые квоты/proxy bins, отдельный scratch-trained reviewer, бюджет/caps и критерии CI — решение пользователя и эта подготовка; статьи не доказывают пользу этих вариантов на ДЗ2.
