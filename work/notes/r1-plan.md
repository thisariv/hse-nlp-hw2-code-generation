---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-14
updated: 2026-10-15
sources:
  - nlp-hw02-d1-results
  - kaggle-r1-preparation
  - kaggle-l3-final-results
  - kaggle-p1-final-results
  - chen2016-net2net
  - gong2019-progressive-stacking
  - kim2023-solar
  - kang2020-loss-truncation
  - hagele2024-cooldown
aliases: []
---

# R1: продолжение L3 с ростом encoder

**R1 завершена:** [парный CPU-разбор](r1-results.md) — deep − контроль +0,15 п.п. [−0,86; +1,15] на heldout4k; одинаковые9,5ч main/20мин cooldown. Следующий рекомендуемый тест — одинаковый50-минутный cooldown из обеих stable. Ниже сохранён исходный план.

**Текущий протокол ревью,15.10:** первая боевая сессия — private аккаунт друга, GPU0 C-long2x encoder6→12 (либо9 при T4 ratio>1,7), GPU1 тот же L3 anchor без роста. Исходный weighted order без D1-удалений; q10% у обеих. Main9ч30мин, одинаковый linear cooldown20мин, полная оценка; hard11ч20мин. Secrets необязательны: без них stable/output/watchdog, с Secrets владельца — private live. См. [r1-review](r1-review.md); исторические оценки стоимости ниже не заменяют T4 gate.

**Результаты D1,15.10:** [d1-results](d1-results.md) — A/B отклонены:121/886=13,66%
согласия с модой, увеличение CE ухудшает proxy. Рекомендация основного R1 —
protected-safe train без D1-удалений; готовые ordinary/long2x orders в
`results/d1-labels/r1-control/`. Строгий C удаляет121 строку и остаётся резервом
(`r1-C/`), улучшение обучения не измерено. Оба типа order прошли CPU R1 load;
новые private Dataset/order_source ещё не созданы. Исходные планы ниже —
гипотезы и варианты конфигурации, а не результат проверки фильтра.

## Гипотезы и сравнение

**Собственный эксперимент:** увеличить ёмкость encoder без скачка начальной функции, продолжить на исходных отфильтрованных парах и проверить качество при равном wall-бюджете. Результатов R1 на GPU пока нет. [P1](p1-results.md) не установил преимущество длинного cooldown или truncation по заранее заданным gates; $q=0{,}1$ здесь — новое решение пользователя, не доказанный лучший рецепт.

| Параметр типового запуска | GPU0 | GPU1 — контроль |
|---|---|---|
| Full stable anchor | L3 latest C-long2x | тот же C-long2x |
| Growth | encoder 6→12, decoder6 | none, encoder6/decoder6 |
| Данные | protected-safe control; C-фильтр как резерв | тот же order и sampling |
| Новый seed порядка | 2026101701 | 2026101701 |
| Loss truncation | стратифицированная, $q=0{,}1$ | то же |
| LR старых параметров | inherited0,0015 | inherited0,0015 |
| Новые параметры | warmup1000attempted updates | отсутствуют |
| Main / cooldown / eval | 9ч30мин / linear20мин / ≤60мин | то же |

Такой контроль выделяет **рост при равном времени**, включая цену меньшего числа updates. Это не сравнение при одинаковом количестве данных и не абляция фильтра или truncation. Из-за разного throughput один seed даёт общий порядок, но не одинаковое число предъявлений. Для сравнения разных anchors параметр `anchor` отдельно принимает C-main; тогда вывод уже смешивает рост и происхождение модели.

## Encoder-deepen и AdamW

Вместо копирования обученных блоков вставляется независимый новый Pre-LN блок **после каждого** старого: old0/new0/…/old5/new5. В new обнулены `self_attn.out_proj.weight/bias` и `linear2.weight/bias`; LayerNorm имеет weight1/bias0. Остальные параметры создаёт исходный `SimpleTransformer`: PyTorch MHA/linear и Xavier для SwiGLU. Это адаптация идеи function-preserving growth из Net2Net; progressive stacking и SOLAR служат мотивацией роста глубины. Она не повторяет их конкретные процедуры.

Для входа блока $x$:

$$u=x+W_o\operatorname{Attn}(\operatorname{LN}_1(x))+b_o,
\qquad y=u+W_2\operatorname{SwiGLU}(\operatorname{LN}_2(u))+b_2=x.$$

$W_o,b_o$ — выход attention; $W_2,b_2$ — выход FFN. Все четыре тензора нулевые, поэтому обе residual-ветви сначала нулевые. Encoder final norm не дублируется. Pre-LN необходим: Post-LN не был бы тождественным. После первых updates новые выходы перестают быть нулевыми; выигрыш качества из тождества не следует.

Старые `Parameter` остаются теми же объектами: их AdamW moments и step не пересоздаются. Новая param group содержит только вставленные параметры, с ленивыми нулевыми moments и step0. На $j$-м attempted update после ответвления:

$$\eta_{\rm old}=\eta_{\rm anchor},\qquad
\eta_{\rm new}(j)=\eta_{\rm old}\min(1,j/1000).$$

$\eta$ — LR соответствующей группы. При cooldown общий LR убывает; множитель warmup сохраняется. Отдельный pre-hook задаёт LR непосредственно перед Adam.step, после scalar-LR кода L1/P1. Scaler и Python/NumPy/Torch CPU/CUDA RNG берутся из anchor; инициализация вставленных блоков не сдвигает training RNG. AMP skips считаются attempted updates, как в P1. При unsafe Adam/OOM или nonfinite threshold worker останавливается с последним finite stable, без продолжения повреждённого состояния.

`nlp-hw02-r1-full-v1` сохраняет имена параметров по optimizer groups, все moments/scaler/RNG/cursor/history, `model_spec` и scheduler state. Stable сохраняется **до** terminal cooldown; после оценки проверяется его полный reload. Cooled FP16 — отдельный `nlp-hw02-model-only-v1`. Новое количество encoder layers загрузчики v6/v7 получают из `model_spec`; менять их не потребовалось. При подключении новых весов надо задать новые SHA в конфиге инференса; старые quality/timing receipts на них не распространяются.

## Данные и truncation

- `data=ordinary`: C-main продолжает исходный cache order, C-long2x — исходный weighted order. SHA, курсор, история и RNG из anchor; ни shuffle, ни повторного warmup старой группы.
- `data=filtered`: параметр `order_path` указывает `order.npy` с соседним `order.json` формата clean-cooldown/D1. Проверяются SHA/bytes/cache manifest/uint32/границы/protected row8545870. Uniform order требует уникальности. Объявленный `nlp-hw02-filtered-long2x-v1` сохраняет1× при compact target<79 и2× при≥79: проверяется каждая retained кратность и fingerprint membership; необъявленные/частичные повторы запрещены. Это индексы исходного compact cache, без новых targets. Multiset заново перемешивается `order_seed`, bucketing4096 и batch128; готовый order и provenance сохраняются в output/live. **Курсор и cursor_origin сохраняются из anchor**, как Adam/scaler/RNG; первый слот нового circular order равен `anchor_cursor % slots`. Оба значения записываются в R1 state. Для сравнения deepen/none на C-long2x обеим веткам нужен один и тот же weighted order.
- `q=0` выключает truncation; $0<q<1$ включает P1 objective. Внутри compact-length bins53/79/105 удаляется верхний per-example mean CE с пропорциональными квотами; хотя бы один пример/bin остаётся. Loss нормируется по **retained target tokens**, включая EOS. Это собственная стратификация метода Kang–Hashimoto, не точное воспроизведение статьи. Высокий loss может означать трудный правильный пример.

Graph path P1 использует один forward на пример. При OOM до Adam он откатывает RNG/scaler и переключается на two-pass с меньшим microbatch; Adam OOM не повторяется. На deepen может вырасти частота fallback и реальная цена шага. В `truncation.jsonl` сохраняются q/квоты/excluded IDs/длины/CE/fallback отдельно для main/cooldown.

## Бюджет и расписания

Отсчёт от первой code cell; GPU работают одновременно. Main ограничен **wall**, включая checkpoint overhead, а не суммой только update timers. Mid cooldown/eval в R1 отсутствует.

| Окно | Длительность | Конец от старта |
|---|---:|---:|
| Watchdog, owner/Secrets, full SHA/версии, private live probe, cache SHA | ≤8мин | 00:08 |
| Main | ≤9ч30мин | 09:38 |
| Final cooldown | ≤20мин | 09:58 |
| Greedy4k/old991 + CE + test1919 + K8 tune2048 | ≤60мин | 10:58 |
| Save/reload/I/O резерв | 3мин | 11:01 |
| Reap, bounded final live-sync, manifest, hard shutdown | 9мин | **11:20** |

Worker deadline40260с, независимый watchdog40800с, server ceiling43200с. Каждая ветка резервирует cooldown/eval/I/O **до** main; setup>480с прерывает запуск. Неуспевшие cooldown/eval помечаются partial. Hard wall — ограничение watchdog, а не обещание полного K8. Исчерпание квоты или остановка Kaggle могут прервать раньше.

`cooldown=linear`: $\eta(t)=\eta_0\max(0,1-t/1200)$, где $t$ — wall-секунды фазы; последний update приближается к0, математический endpoint0 фиксируется отдельно.

`cooldown=1-sqrt`: после фактически выполненных $M$ main updates задать долю $f$ cooldown среди **новых R1 updates**:

$$T=\max(1,\operatorname{round}(Mf/(1-f))),\qquad
\eta_j=\eta_0(1-\sqrt{j/T}),\quad j=1,\ldots,T.$$

$T$ — число attempted cooldown updates; endpoint на последнем шаге точно0. `cooldown_seconds` в этом режиме — **потолок**, а не подмена шаговой оси временем. При постоянной скорости $f=0{,}1$ и main570мин требуют63,3мин cooldown: это не помещается в50мин. Для прежнего бюджета разумный запас — **f=0,075**, оценка46,2мин; фактическая скорость/fallback решает, завершатся ли все $T$. Другой вариант: main9ч + cooldown60мин + eval40мин, общий расчёт `480+32400+3600+2400+180=39060с`, остаётся дополнительный резерв20мин до worker deadline. Формы и длительности не выбирать по test.

### Стоимость шага

CPU benchmark и прогноз T4 — в [receipt](../checks/kaggle-r1-cpu-benchmark.json). Реальные next16 train indices C-long2x, C512/6+6→12+6, q0,1, одинаковые microbatch/длины, **три warmup и пять измерений**. Параметры53 646 403→72 589 411 (+35,3%). Функция полной C512 после роста **побитно совпала**, max abs0.

| CPU FP32,1thread,batch16 | C6+6 | C12+6 |
|---|---:|---:|
| Median полного update, с |1,6551|3,2310|
| Диапазон пяти замеров, с |1,5722–2,0205|2,9996–3,3522|
| Относительная стоимость |1|**1,952× (+95,2%)**|

Первый pilot с одним warmup имел сильный дрейф6,22→3,21с у deepen; прогрев увеличен до фиксации финальной оценки. CPU FP32/batch16 не воспроизводит T4 FP16/batch128/fused AdamW: перенос отношения — **плановый сценарий**, не GPU-измерение и не аппаратный коэффициент. База L3 long2x2,573 attempted updates/с даёт0,3887с/шаг; умножение на1,952 даёт **≈0,759с/шаг deepen**. Небольшой P1 truncation overhead≈1,4% повышает сценарий до≈0,769с против≈0,394с контроля; фильтр и graph→two-pass fallback способны изменить оба числа.

За570мин main это примерно**44,5тыс. deepen /86,8тыс. control updates** (5,7M/11,1M предъявлений), ещё≈3900/7600updates за50мин cooldown; checkpoint/I/O уменьшат доступные main updates. Это оценка масштаба для планирования, не обещание throughput. CPU-отношение следует заменить измерением на T4 при ревью.

## Оценка и критерий

Оценка переиспользует P1/L2: полные greedy heldout4096 и old991, token CE, исходные target-length bins60/90/120, frozen tune/eval2048, свежий full test1919 timer без gold и K8 T0,7/seed270042/batch32/cap512. Все K8 samples с represented code/LP/execution сохраняются для offline equal-K mixed voting. На K8 зарезервировано≤20мин, общее evaluation≤40мин. Fresh1024 не используется.

Основное сравнение — deepen против контроля на eval2048, paired gains/losses и95% bootstrap CI, плюс длинные bins и old991 как диагностика. Положительная нижняя граница CI — свидетельство преимущества в этом повторно использованном eval, не независимый новый test. При неопределённости оставлять обе модели кандидатами до проверки ensemble, без объявления победителя по CE. Test используется только для полного таймера **≤180с/T4**; рост encoder дешевле decoder для autoregressive инференса, но overhead и полный v6/v7 timer ещё надо измерить.

## Сборка и передача ревьюеру

Из `work/`, Python с Torch/tokenizers/NumPy/nbformat:

```bash
python kaggle/build_r1.py --owner friend1
python kaggle/build_r1.py --owner thisarivson
python kaggle/build_r1.py --owner friend1 --gpu-smoke
python kaggle/r1_tests.py
python kaggle/r1_static_check.py
```

Без `--config` обе ветки используют **ordinary** и q0,1: это самостоятельный рабочий вариант. Для типового опыта заполнить две записи [r1-filtered.example.json](../kaggle/r1-filtered.example.json) реальными D1 `order_source/order_path/order_sha256`, затем `build_r1.py --owner ACCOUNT --config CONFIG.json` и отдельно тот же config с `--gpu-smoke`. Пример намеренно содержит невалидный placeholder SHA: сборка не превращает отсутствующий фильтр в ordinary молча. `--cache-source` позволяет взять friend fallback; все артефакты private. Текущие канонические bundles: production аккаунт друга, smoke thisarivson; builder переключает owner за один вызов.

Для thisarivson подготовлен потоковый `r1_prepare_api.py copy-states`: friend subprocess скачивает L3v12 full PT, CPU проверяет две точные next128 updates, default SDK загружает blob, PT удаляется, затем берётся следующий. SDK собирает обе blob-ссылки в **одну private версию** `thisarivson/nlp-hw02-l3-states`; одновременно двух full PT на диске нет. Скопированы также producer recover-manifest/latest и weighted order, поэтому действуют прежние SHA-контракты. Локальные credential-каталоги скрипты не открывают; значения в отчёты не попадают. Live datasets `ACCOUNT/nlp-hw02-r1-live` создаются отдельно. Квитанции — `checks/kaggle-r1-states-copy.json`, `kaggle-r1-real-resume-*.json`, `kaggle-r1-live-*.json`.

CPU suite проверяет тождественность, моменты/разные Adam steps, warmup/custom q, R1 full resume и следующий update, filtered membership/tamper/seed, обе короткие main→cooldown→evaluate→stable-reload ветки и реальные load-функции v6/v7. Статика проверяет оба аккаунта × production/smoke, frozen embedded sources, watchdog до Torch, private metadata, hard budgets и notebook schema. L2/L3/P1:34/10/17регрессий прошли, исходники не изменены. CPU resume не доказывает CUDA FP16/fused эквивалентность.

**Фактические квитанции подготовки:** обе full states644 195 795/644 197 075bytes созданы в private `thisarivson/nlp-hw02-l3-states/1`, server names/bytes сверены; локальных PT0. Обе CPU none-continuations — по2batch128updates побитно, включая model/AdamW/scaler/RNG и логические state counters. Private live ready у обоих аккаунтов. Реальный clean-cooldown order9 506 623из9 725 737строк прошёл SHA/уникальность/protected/membership и новый seed, временный order удалён. Это проверка **формата**, не выбор будущего D1-фильтра. Всего10 R1 CPU tests, включая отказ на неполной оценке и live snapshot/restore обеих optimizer groups с новым order; smoke notebook собран, на GPU не исполнялся.

> [!info]- Воспроизведение тяжёлых CPU/API проверок
> `python kaggle/r1_benchmark.py --cache CACHE --order WEIGHTED_ORDER` — full C512,3warmup/5timings. `python kaggle/r1_data_check.py --cache CACHE --order CLEAN_ORDER_DIR` — реальный subset. `python kaggle/r1_cpu_resume_check.py --anchor-root SCRATCH --cache CACHE --weighted-order WEIGHTED_ORDER --model C-long2x` —2exact updates и удаление загруженного PT по умолчанию; отдельно C-main.
>
> Copy использует **default** SDK: `python kaggle/r1_prepare_api.py copy-states --cpu-python CPU_PYTHON --cache CACHE --weighted-order WEIGHTED_ORDER`. Внутренний downloader запускается только с `KAGGLE_CONFIG_DIR=~/.kaggle-friend`; повторный upload готового target запрещён. При задержке server readiness использовать `python kaggle/r1_prepare_api.py verify-states`, без скачивания/загрузки. Pins — [r1-anchors.json](../kaggle/r1-anchors.json).

**Ревьюеру остаётся:** закончить D1 и выбрать фильтр по его диагностике, заполнить SHA/путь и источник; независимо проверить R1, запустить короткий GPU-smoke обеих веток, измерить T4 memory/throughput/fallback и полноту eval. Затем отдельно production push/UI Secrets после сброса17.10 и новый final-inference timer. Здесь GPU-smoke и kernel push не выполняются; файлы D1 не меняются.

Попытка содержательного коммита: `git add` только R1 и навигации отклонён песочницей на `.git/worktrees/study1/index.lock`: `Operation not permitted`. Индекс не изменён, чужие D1-файлы не добавлялись. Квитанция — [kaggle-r1-git.json](../checks/kaggle-r1-git.json); изменения сохранены в рабочем дереве.

## Источники

- Chen T., Goodfellow I., Shlens J. **Net2Net: Accelerating Learning via Knowledge Transfer**, ICLR2016. [Статья](https://arxiv.org/abs/1511.05641). Function-preserving transfer; конкретный Pre-LN zero-output рецепт выше — собственная адаптация.
- Gong L. et al. **Efficient Training of BERT by Progressively Stacking**, ICML2019. [PMLR](https://proceedings.mlr.press/v97/gong19a.html). Постепенное увеличение глубины BERT; не прямой результат для нашей encoder–decoder модели.
- Kim D. et al. **SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling**,2023. [Статья](https://arxiv.org/abs/2312.15166). Depth up-scaling и continued pretraining; SOLAR не обещает тождественности нашего роста.
- Kang D., Hashimoto T. **Improved Natural Language Generation via Loss Truncation**, ACL2020. [ACL](https://aclanthology.org/2020.acl-main.66/). Стратифицированные квоты/retained-token objective — адаптация P1/R1.
- Hägele A. et al. **Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations**,2024. [Статья, §3 и приложение B.1](https://arxiv.org/html/2405.18392v3). Constant+cooldown/WSD, linear/square-root; перенос на этот encoder–decoder проверяется экспериментом.
- [l3-results](l3-results.md), [p1-results](p1-results.md), [d1-plan](d1-plan.md), [clean-cooldown-analysis](clean-cooldown-analysis.md); [условие](../../task/hw2-code-generation.md) неизменно. Код: [builder](../kaggle/build_r1.py), [growth](../kaggle/src/r1_growth.py), [runtime](../kaggle/src/r1_runtime.py), [статика](../checks/kaggle-r1-static.json). Все источники внесены в библиографию.
