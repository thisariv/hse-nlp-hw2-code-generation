---
type: experiment
course:
  - nlp
status: learning
understanding: 4
created: 2026-10-17
updated: 2026-10-18
sources:
  - kaggle-b4-independent-review
  - kaggle-b4-preparation
  - kaggle-b3-independent-review
  - kaggle-p1-independent-review
aliases: []
---

# B4: независимое ревью и запуск

18.10: сессия завершилась COMPLETE; [CPU-разбор B4](b4-results.md) содержит фактические main/probes/final/K8 и рекомендацию о B5. Ниже — история ревью и запуска17.10.

Проверен исходный commit `31872e7`, ветка `codex/nlp-hw2-code-generation`; рабочее дерево в начале чистое. Прочитаны [b3-review](b3-review.md), [p1-review](p1-review.md), [b4-plan](b4-plan.md), код и CPU receipts. Пользователь разрешил private GPU-smoke ≤12мин и, после успешного gate, боевой B4. Содержимое `~/.kaggle*` не инспектировалось: аутентификацией занимается SDK/CLI. Все входы и запуски private.

## Актуализация: двухпроходная truncation (17.10, smoke v4)

Работа продолжена из commit `8e91521`, дерево исходно чистое. История ниже описывает прежний graph/offload-вариант. Он заменён после v3: `save_on_cpu` нарушил layout saved attention bias, backward mem-efficient SDPA завершился `attn_bias.stride(2)=98`, требуется кратность4. CPU-копирование не сохраняет необходимое выравнивание; custom CE+z/offload полностью удалены из B4.

Текущий `src/b4_math.py`: **два прохода**. Первый — `p1_math.per_example_ce` под `torch.no_grad`, training mode и тем же FP16 autocast; по всем128 предъявлениям получаем CE/token, EOS включён, PAD исключён. RNG Python/NumPy/Torch CPU/CUDA возвращается в `finally`, включая ошибку selection. P1-mask/q10%/compact bins53/79/105 не изменены. Второй — неизменённый `b2_optim.update` только по115 retained: стандартный autograd, CE + FP32 z-loss1e-4, единый знаменатель retained target tokens; один unscale/clip/step/scaler.update. Граф живёт только до backward текущего microbatch; память ограничивается как у GPU0, без retained128 graphs и offload.

Pinned B3 имеет dropout0; дополнительные scoring forwards не меняют RNG обеих GPU. Курсор в main обеих веток по-прежнему +128; stream SHA учитывает все предъявления. OOM до Adam повторяет оба прохода с меньшим micro после B2 RNG/scaler/gradient reset; Adam OOM остаётся unsafe. В журналах сохранены исключённые ID/позиции/длины/strata/CE/tokens и `forward_mode=two-pass`, `extra_selection_forward=true`, `selection_rng_restored=true`.

По разрешению пользователя ожидается **≈+30% времени шага** из-за дополнительного forward. Прежний порог≤15% отменён до получения v4 output; обе фактические time ratios GPU1/GPU0 записываются отдельно, равный wall-budget допускает разные конечные курсоры. Gate: COMPLETE, exits[0,0],200+ main updates/GPU, nonfinite0, оба полных restore/probe/final/K8, одинаковые старт и stream, отсутствие OOM. В этой сессии разрешены v4 и не более одного исправляющего повтора; после успеха production уже разрешена.

CPU: независимый full-batch oracle считает CE всех примеров, P1-mask и CE+z по retained с общим token denominator; сравниваются loss, gradient norm, **сырые и clipped gradients** с microbatched two-pass. Forward hook проверяет128 no-grad и115 grad примеров/micro≤32, spies — ровно один optimizer/scaler cycle. Также реальные NaN в обоих проходах, AMP overflow, OOM retry, RNG при успешном/failed selection, три одинаковых батча GPU0/GPU1, стратификация/нормировка и прежние B4/B3/B2/B1/P1 регрессии.

<!-- B4_TWO_PASS_VALIDATION -->
18 B4+67 inherited tests и обе сборки/статика прошли. **v4 passed**: COMPLETE, session447,732с≤720, exits[0,0],200 main updates/GPU, nonfinite0/AMP0/OOM0; обе probe/final full restore-аудита на каждой GPU точные, final main RNG одинаков. Mini greedy64/old16/CE/test64, K8 на36 frozen tune IDs,288 samples/GPU полны. Одинаковы start fingerprints и все200 stream prefixes. Проверены122 small producer SHA; full PT не скачивались, output6,406GB. [GPU receipt](../checks/b4-review-gpu-smoke-v4.json), [status](../checks/b4-review-smoke-status-v4.json), [server source](../checks/b4-review-smoke-metadata-v4.json), [CPU/static receipt](../checks/b4-two-pass-local.json).

Main step GPU0 **0,77115с**, GPU1 **1,03147с**: ratio1,33757, **+33,76%**. Девять sampled batches:0,83840/1,04318с, ratio1,24425, **+24,43%**; совпадают sampled cursor, включая worst aligned batch. Sampled allocated peak7,95/6,16GiB, reserved8,37/7,29GiB; run/eval reserved peak12,42/9,20GiB. Дополнительный no-grad forward дал ожидаемый порядок замедления≈30%; память GPU1 ниже GPU0, retained graphs/offload отсутствуют.

API push v4 **20:42:03UTC**, первый RUNNING20:42:44, COMPLETE подтверждён20:55:41; это платформенный интервал, **измеренный session wall447,73с**. Квота до/после67914,630/67431,430с: delta**483,200с≤720**. Прежний15% gate отменён по пользовательскому протоколу до output. Повтор v5 не потребовался.
<!-- /B4_TWO_PASS_VALIDATION -->

## История: находки до двухпроходного исправления

Пути относительно `work/kaggle/`, строки после исправлений.

| Приоритет | Находка и исправление | Регрессия |
| --- | --- | --- |
| **P1: GPU1 не помещается на T4** | Smoke v1 выявил retained-graph OOM даже при micro1: уменьшение micro не ограничивает память всех128 graphs. `src/b4_math.py:15–68,79–96`: custom CE+z сохраняет logits и малый log-normalizer; backward вычисляет общую аналитическую производную. Surplus saved activations после10GiB allocation переносятся стандартными pinned `save_on_cpu` hooks; FP32 параметры остаются на GPU. Число forward/маска/цель прежние, recompute модели/two-pass нет. `src/b4_runtime.py:21–24` фиксирует реализацию в GPU1 recipe/SHA. | `b4_tests.py:58`, `test_b4_analytic_CE_z_derivative_PAD_and_compact_saved_logits`: CE/z, weighted gradients, zero PAD и единственный saved vocab tensor против обычного autograd. `b4_tests.py:73`, `test_b4_saved_activation_offload_preserves_gradient_and_never_copies_parameters`: реальные saved hooks с копированием, точные gradients, исключение parameter storage. Полный retained-batch oracle/один forward на128 и все прежние OOM/AMP/restore tests повторены. |
| **P1: недостаточный GPU gate** | Подготовленный smoke ограничивался64 main updates. `build_b4.py:23–24`, `src/b4_gpu_smoke.py:14–20,45–55`: теперь200 на каждой GPU; меньший cap отклоняется. Также gate требует нулевую дельту nonfinite skips: один пропущенный NaN update раньше мог остаться ниже recovery threshold и дать successful smoke. | `b4_tests.py:133–155`: полный controller с200 настоящими tiny updates, enabled CPU GradScaler, probe/final и restore;64 отклоняется. `b4_tests.py:156`: один injected nonfinite selection в первом main update, продолжение остальных updates и сохранение stable; controller обязан отклонить этот результат. |
| **P2: потеря результата ревью/запуска** | `b4_static_check.py` безусловно переписывал handoff в CPU-ready/GPU-pending с нулевыми runs. `b4_static_check.py:8–21,96–103`: `ready/smoke_passed` теперь требуют successful receipt с теми же SHA всех модулей; отсутствие/чужой/failed receipt не проходит. Сохраняются ссылка, статус, время и ETA уже запущенного B4 при совпадении исходников. Static receipt отдельно сообщает число GPU-операций самой проверки. | `b4_tests.py:168`: отсутствие/failed/чужие SHA, successful same-source, сохранение RUNNING/ссылки/ETA и сброс чужого launch. |

GPU0 math, orchestration, inherited B2 implementation SHA и data policy не менялись. GPU1 получила эквивалентную производную прежней цели и управление памятью; побитного совпадения двух способов FP32 суммирования не обещается. Для воспроизводимости добавлены [API helper](../kaggle/b4_review_api.py) и [аудитор output](../kaggle/b4_review_output.py). Допуск замедления GPU1 задан **до получения output: ≤15%** и на одинаковых sampled batches, и на200 main updates.

До результата v2 уточнён gate: безопасные pre-Adam OOM retries допустимы по уже проверенной политике B2; вся их стоимость входит в `seconds` и тот же допуск15%. Unsafe Adam, nonfinite и two-pass по-прежнему не проходят. OOM K8 не допускается, поскольку меняет sampling batch shape.

## Проверенные риски

| Область | Проверка и границы |
| --- | --- |
| Общий старт | `b4_worker.py:23–25`: обе GPU читают именно `resume_root/model-GPU1` B3 W-lr600. `src/b4_runtime.py:45–70` сверяет exact latest, full bytes/SHA, architecture/recipe, Adam steps, scheduler, scale и CUDA RNG, cache/order/seed; `b2_runtime.py:136–164`, `l1_runtime.py:139–143` восстанавливают model/Adam/scaler/Python/NumPy/Torch CPU/CUDA RNG/state. Исходник `e32b8649…19c0`, updates73416/cursor9397248/main59400,861588с; LR0,0006/factor1, warmup2000 уже пройден. Fresh fallback отсутствует. |
| Поток | `src/b4_runtime.py:28–34,166–180`: GPU0 вызывает frozen B2 main; GPU1 использует тот же `take(order,cursor,128)` и продвигает cursor на все128 предъявленных, включая исключённые/AMP skip. `src/b4_session.py:8–23` сравнивает все стартовые fingerprints и SHA каждого общего префикса. GPU1 identity меняет только описание нового objective/его SHA, не численный старт. Равный wall даёт разные конечные cursor; matched-pair выводы требуют общего префикса, не только точек +3/+6ч. |
| Маска и нормировка | `src/b4_math.py:77–96`: один forward каждого примера, detached mean CE с EOS/без PAD; одна маска по всей effective128 **до первого backward**. `p1_math.py:30–56`: top-loss внутри compact bins53/79/105, proportional largest remainder, ≥1 retained/bin, ties по позиции;13/128=10,15625%, не ровно10%. Все microbatches используют один denominator: retained target tokens. Сохранённые graphs не делают дополнительный selection forward. |
| z-loss / accumulation / AMP | `src/b4_math.py:22–37,89–105`: FP32 CE и z-loss1e-4; одна retained-mask для обоих, выбор только по CE. Backward накоплен до единственных unscale/clip1/Adam step/scaler.update. `b2_optim.py:49–100`: OOM до Adam очищает graphs/grad, восстанавливает scaler/RNG и уменьшает micro; Adam OOM unsafe. Rejected nonfinite не вызывает Adam; overflow уменьшает scale и пропускает Adam. CPU oracle сравнивает CE/z и gradients с retained batch; CUDA overflow проверяется отдельно. |
| NaN recovery | `src/b4_runtime.py:182–195`: unsafe model/Adam или recovery threshold восстанавливают полный последний stable и останавливают ветку с partial; LR не уменьшается, q/order не меняются. Одиночные nonfinite/AMP skips учитываются прежней политикой B2. `b4_tests.py:97–132,232` покрывает rejected loss/backward, OOM/RNG/scaler reset, unsafe Adam, overflow и fixed-LR stop. |
| Точки / restore | `build_b4.py:16–19`, `src/b4_runtime.py:103–108,113–153,155–163`: probes origin+10800/+21600 main-секунд; каждая — disposable900с timed linear. Final3000с timed linear. В `finally` возвращаются model/Adam/scaler/RNG/cursor/state из uncooled stable, затем добавляется только branch history/time. Прямые before/restored hashes через inherited `BranchAudit` проверяют12 полей; сравнение независимых CUDA траекторий не используется как restore-инвариант. |
| GPU1 final | `src/b4_runtime.py:97,120–133`, `update:37–41`: одна переменная q0,1 проходит в main, оба probe и final. Final GPU0 q0; GPU1 q0,1. LR вычисляется до update как peak×(1−elapsed/cap); последний фактический LR записан. Искусственного нулевого шага нет. |
| K8 / FP16 / stable | `b4_worker.py:33–39`, `b1_eval.py:13–16`, `l2_eval.py:32–119`: final greedy4096/old991, CE, test1919 и frozen tune2048/K8/T0,7/seed270042; сохраняются ID/question SHA, represented code/logprob/execution. `b4_runtime.py:78–87`: mini-final требует36 фактических tune IDs. `b2_runtime.py:112–134`: finite full FP32 model/Adam/scaler/RNG, atomic latest, два поколения. `l1_runtime.py:145–153`: отдельные model-only FP16 points, strict-loaded evaluator. Final cooldown не попадает в B5 uncooled latest. |
| Мягкий финал | `b4_worker.py:41–44`, generated notebook final statements из `build_b4.py:38`: partial печатается и публикуется без финального assert. `src/b4_runtime.py:139–152,202–218`: incomplete evaluation сохраняет FP16/неполный receipt и восстанавливает stable, successful metrics не подделываются. `b4_session.py:26–52` держит watchdog до завершения A/B publication и обновляет manifest SHA после изменения summary. |
| Бюджет / quota | `build_b4.py:18–24`: production worker40500/watchdog40740/backend40800с=11ч20мин от first code cell; smoke660/690/720с. `src/b4_runtime.py:74–75,157–162`: main всегда оставляет5100с final reserve; probe дополнительно900+450+120с. Номинально32400+2×1350+3000+1800+480=40380с; checkpoint I/O сокращает main. `l1_session.Watchdog:12–15` переопределяет старый короткий потолок. Перед smoke exact quota68798,525с; reset24.10 00:00UTC. API повторно проверяет≥42000с перед production. |
| Входы | `build_b4.py:31–34,51–54`, `b4_resume.py:7–21`: kernel_sources B3 и cache/2, **без b2-states**. B3 manifest SHA `fa5d3d8d…2453b8`, exact latest+full SHA и inherited order проверяются на mount; cache проверяется по всем payload SHA. `l3_resume.py:8–13` поддерживает `/kaggle/input/notebooks/thisarivson/nlp-hw02-b3` и flat mount. [API preflight](../checks/b4-review-inputs.json) подтвердил private B3/1/cache/2 и ready private starter. |

CPU: **18 B4 tests passed**, [receipt](../checks/b4-cpu.json), [67 inherited tests](../checks/b4-regressions.json), [статика/nbformat/source parity](../checks/b4-static.json). Подготовкой уже проверен реальный next batch128 CPU bit-exact для GPU0: [receipt](../checks/b4-real-resume.json); повторно full PT не скачивался. GPU1 objective/gradients проверены отдельным retained-batch oracle и analytic derivative regression.

Собственный вывод для custom backward: если $a_i,b_i$ — входящие gradients по per-example CE и z-loss, $s_{it}=\log\sum_v e^{x_{itv}}$, $p_{itv}=e^{x_{itv}-s_{it}}$, то

$$\frac{\partial L}{\partial x_{itv}}=p_{itv}\bigl(a_i+2\cdot10^{-4}b_i s_{it}\bigr)-a_i\mathbf1[v=y_{it}].$$

PAD обнуляется. $a_i,b_i$ уже содержат retained-mask, общий token denominator и GradScaler scale. Производная вычисляется в FP32, затем приводится к dtype исходных logits. Пересчитывается softmax из сохранённых logits, не forward модели.

## GPU-smoke

**v1:** private [thisarivson/nlp-hw02-b4-gpu-smoke/1](https://www.kaggle.com/code/thisarivson/nlp-hw02-b4-gpu-smoke/versions?versionNumber=1), API push17.10.2026 **18:45:31UTC**; backend720с. COMPLETE, session **partial/exits[0,1]**: GPU0 прошла200 main/probe/final/overflow, GPU1 — OOM в disposable benchmark, даже micro1. Wall **411,759с**, окончательная quota delta **436,953с**, обе≤720. Проверены67 small producer SHA. [Неуспешный gate](../checks/b4-review-gpu-smoke-v1.json), [push](../checks/b4-review-smoke-push-v/1.json), [server source](../checks/b4-review-smoke-metadata-v1.json). Боевой запуск после этого не выполнялся.

**v2 — второй и последний разрешённый smoke:** private [версия2](https://www.kaggle.com/code/thisarivson/nlp-hw02-b4-gpu-smoke/versions?versionNumber=2), API push **19:01:57UTC**, backend720с. Перед push68361,572с свободной квоты. [Push](../checks/b4-review-smoke-push-v/2.json), [server source](../checks/b4-review-smoke-metadata-v2.json). Логи во время запуска не использовались; gate определяется COMPLETE и output.

<!-- B4_SMOKE_RESULT -->
Актуальный результат — **v4 passed**, подробности и доказательства в актуализации выше. v3: COMPLETE с GPU1 SDPA strideM failure; лог (`../results/kaggle-runs/nlp-hw02-b4-gpu-smoke/3/gpu1-process.log`, нет в репозитории). v4 заменяет прежний ожидающий v2 gate; successful same-source receipt выдан.
<!-- /B4_SMOKE_RESULT -->

## Боевой B4

<!-- B4_PRODUCTION_RESULT -->
Private [B4 v1](https://www.kaggle.com/code/thisarivson/nlp-hw02-b4/versions?versionNumber=1) отправлен **17.10.2026 20:57:00UTC (23:57МСК)**; **RUNNING** подтверждён20:57:06UTC. GPU0 W600/q0, GPU1 W600-trunc/q10%: тот же uncooled B3 W600/Adam/scaler/RNG/order/cursor. Все server/module SHA production совпали со successful v4; backend40800с, worker40500, watchdog40740. Перед push**67431,430с≈18,73ч** свободной квоты; после hard-бюджета40800с остаётся26631,430с≈7ч24мин без платформенного overhead.

Ориентир завершения **18.10 ≈11:17МСК /08:17UTC**: первый наблюдённый RUNNING +11ч20мин; фактический first-code-cell start не измерен, VM/output overhead может сдвинуть платформенный COMPLETE. [Push](../checks/b4-review-production-push-v/1.json), [RUNNING](../checks/b4-review-production-status-v1.json), [server source](../checks/b4-review-production-metadata-v1.json), [handoff/ETA](../results/next-b4.json). Дополнительное разрешение не запрашивалось: запуск уже был разрешён пользователем после successful smoke.
<!-- /B4_PRODUCTION_RESULT -->

## Ограничения

Короткий smoke проверяет исполнение тех же модулей, не качество после9ч/50мин cooldown и не cold ensemble≤180с. Его generation cap8 и малые splits намеренно сокращены. Sampled memory/throughput не гарантируют все будущие формы. GPU1 выполняет дополнительный no-grad forward; память ограничена текущим microbatch, фактические peak и замедление записываются в GPU receipt. При равном wall-бюджете число updates обеих GPU может различаться.

Output штатно около6,8GB; concurrent checkpoint peak около9,6GB при guard10GiB. Это небольшой запас, не гарантия против полного диска. Без live-sync сбой VM может потерять локальный output; atomic latest защищает обычную запись, не VM. Backend timeout/first-cell watchdog не доказывают потолок времени очереди, VM setup, upload и окончательного списания Kaggle.

## Коммит

<!-- B4_GIT_RESULT -->
Текущая попытка `git add` явного B4-only списка завершилась128: песочница запрещает создать `<vault>/.git/worktrees/study1/index.lock` (`Operation not permitted`). До попытки staged index пуст; `git commit` не выполнялся. Ограничение не обходилось. Код, заметки и receipts остаются в рабочем дереве; [квитанция](../checks/b4-two-pass-git.json). Исходный commit `8e91521`, ветка `codex/nlp-hw2-code-generation`. Боевой запуск уже RUNNING, результат попытки Git на него не влияет.
<!-- /B4_GIT_RESULT -->

## Источники

- **kaggle-b4-independent-review** — собственное независимое ревью17.10.2026, код/CPU/API/GPU receipts выше; bibliography — kaggle-b4-independent-review.
- [b4-plan](b4-plan.md), [b3-review](b3-review.md), [p1-review](p1-review.md) и исходники. Протокол/q/бюджет заданы пользователем; допуск скорости и выводы о рисках — reviewer judgment. Новых внешних научных источников не добавлено.
