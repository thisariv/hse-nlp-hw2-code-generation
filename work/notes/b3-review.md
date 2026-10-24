---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-16
updated: 2026-10-16
sources:
  - kaggle-b3-independent-review
  - kaggle-b3-preparation
  - kaggle-b2-independent-review
  - kaggle-b2-results
  - pytorch-reproducibility
aliases: []
---

# ДЗ2: независимое ревью B3

Проверен commit `356c562` в `codex/nlp-hw2-code-generation`; исходное рабочее дерево чистое. Прочитаны [b3-plan](b3-plan.md), [b2-review](b2-review.md), [b2-results](b2-results.md). **Боевой B3 не запускался.** Единственный разрешённый GPU-запуск — private smoke на thisarivson до reset; это уточнение пользователя имеет приоритет над прежним планом «смоук после reset». Содержимое `~/.kaggle*` не инспектировалось; аутентификацией занимался SDK с выбранным config directory. Secrets в notebook не нужны.

## Дополнение: диагностика сбоя v1 и единственный повтор v2

Пользователь отдельно разрешил **один повтор** smoke, production B3 запретил. Ниже — актуальное дополнение; последующие разделы сохраняют результаты первичного ревью и прежнее состояние RUNNING как историю.

**Полученный output v1:** Kaggle `COMPLETE`, session `partial`, exits `[1,0]`, first-cell wall600,318с. GPU0 W-lr1000 остановился на сравнении полного digest после200 main с независимым control200; GPU1 W-lr600 прошёл также overflow/recovery. Обе ветки прошли probe после **одного** main update и final после200; обе сохранили complete metrics. Новых AMP/nonfinite skips нет, LR-factor1, micro32. Отличаются исходные model/Adam/RNG/cursor и LR, а не условие restore. Логирование attention включается при `updates%100==0`: GPU0 —34200/34300, GPU1 —32500/32600 (это counters **до** шага); старый control его не включал. Сам diagnostic — `no_grad` FP32 matmul, RNG/параметры не меняет; теперь control повторяет и этот путь.

После окончательного завершения v1 API показал3297,747с свободной квоты против3936,950с до push: дельта639,203с. Прежнее временное показание расхода>720с **не равно окончательному списанию**; вывод первичного ревью о нарушенном лимите по нему не подтверждается. First-cell wall600,318с укладывается в720с; [117 small SHA проверены](../checks/b3-fix-v1-diagnosis.json). Это наблюдение одной сессии, не гарантия платформенного billing cap.

Старый assert смешивал два вопроса: точность восстановления anchor и воспроизводимость200 независимых CUDA updates. Без deterministic algorithms второе не гарантировано даже при одинаковом RNG ([PyTorch2.11](https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html)). Успех GPU1 не доказывает детерминизм GPU0: начальные веса, данные и формы промежуточных тензоров различаются. Конкретное CUDA-ядро по v1 output не локализуется; приписывать сбой именно embedding `atomicAdd` оснований нет.

**Исправление:** `b3_runtime.run(..., branch_audit=None)` вызывает необязательный smoke callback непосредственно перед disposable branch и сразу после полного restore, **до** добавления branch history. `b3_gpu_smoke.BranchAudit` сравнивает отдельные SHA model (включая buffers), полного Adam, GradScaler и enabled-флага, Python/NumPy/Torch CPU/CUDA RNG, **всего state** (cursor/LR/stability/history), train/eval каждого module и grad/inference flags. Любое расхождение по-прежнему падает. Production callback не передаёт: хеширование/дополнительные копии/deterministic kernels в нём отсутствуют, frozen B2 implementation SHA не менялись.

Control200 и continuation200 сохранены. Их per-field сравнение стало диагностикой, без ложного утверждения `CUDA_updates_bit_exact`; отдельно replay первых8 updates из того же B2 anchor без каких-либо оценок (`noeval-replay.json`). Прямые before/restored SHA двух веток сохранены в `restore-audits.json`; output-аудитор требует их точного совпадения и соответствия двум фактическим точкам. AMP overflow, recovery, constant LR, mini greedy/CE/test/K8 и B4 latest проверяются как раньше.

**CPU:**16 B3 +9 B2 +5 B2-review +20 B1 = **50 tests passed**, [сводка](../checks/b3-fix-cpu.json). Tiny QK с dropout0,2: no-eval9 vs две cumulative пробы+final — bit-exact; evaluator расходует Python/NumPy/Torch RNG. Отдельно проверены ошибки model/Adam/scaler/RNG/cursor/train-mode/buffer. Полный tiny controller выполняет control200 + replay8 + continuation200 с enabled CPU GradScaler; искусственное отличие независимого control допускается только при точном прямом restore. [B3 CPU receipt](../checks/b3-cpu.json).

**Настоящий B2 GPU0 на CPU:** source full PT проверен по pinned SHA, все frozen implementation SHA прежние. Через [b3_cpu_branch_check.py](../kaggle/b3_cpu_branch_check.py) выполнены **3 main update × effective batch128 на каждом пути**: control против двух probes+final; model/Adam/GradScaler/RNG совпали побитно, counters/stability/LR прежние, все три before/restored SHA равны. Evaluator отдельно загружал FP16 point и делал один настоящий forward в eval, затем расходовал RNG. [Receipt](../checks/b3-fix-real-cpu.json),264,847с. Ограничения: короткие **синтетические** строки, FP32/no-autocast/fused=False, отдельная временная CPU identity, metrics-fixtures вместо измерения качества; это не повтор настоящих следующих строк cache и не CUDA-проверка. Production environment gate не ослаблялся. Единственный скачанный full PT и временные checkpoints удалены после проверки.

**Повтор v2:** отправлен16.10.2026 19:23:27UTC, private [thisarivson/nlp-hw02-b3-gpu-smoke/2](https://www.kaggle.com/code/thisarivson/nlp-hw02-b3-gpu-smoke/versions?versionNumber=2), backend720с. Перед push свободно3297,747с. [Push](../checks/b3-fix-smoke-push-v2.json), [server SHA](../checks/b3-fix-smoke-metadata-v2.json), [квота](../checks/b3-fix-quota-before.json). **COMPLETE, session complete, exits `[0,0]`**. First-cell wall **611,704с**, окончательная дельта квоты **645,747с**, обе величины≤720с; свободно2652с. [Статус](../checks/b3-fix-smoke-status-v2.json), [финальная квота](../checks/b3-fix-quota-after.json), [полный output-аудит](../checks/b3-review-gpu-smoke-v2.json). Проверены **123 SHA малых файлов**, output6 402 614 786байт (<10GiB); full PT из smoke не скачивались. Production push0, дополнительных GPU-запусков после v2 нет.

| Проверка v2 | GPU0 W-lr1000 | GPU1 W-lr600 |
| --- | --- | --- |
| Probe: SHA непосредственно до/после restore | Все12 полей равны | Все12 полей равны |
| Final: SHA непосредственно до/после restore | Все12 полей равны | Все12 полей равны |
| Независимые200 main: control vs continuation | Отличаются model/Adam | Отличаются model/Adam |
| Те же scaler/RNG/modes в200-step comparison | Равны | Равны |
| Отдельный replay8 без оценки | Побитно равен | Побитно равен |
| Overflow skip, recovery, LR, история, mini greedy/CE/test/K8 | Passed | Passed |

**Причина — (b), неверный инвариант smoke:** equality200 независимых CUDA updates выдавал численное расхождение model/Adam за дефект restore. В v2 это повторилось и на GPU1, которая в v1 прошла: отдельной ошибочной ветки кода GPU0 нет. Прямой полный restore на обеих GPU точный, в том числе optimizer/scaler/все RNG/cursor/modes, поэтому вариант(a) для проверенных точек исключён. CUDA training без deterministic mode не обязан давать побитно одинаковые траектории; результаты не позволяют приписать разницу конкретному SDPA/embedding/atomic ядру или одному LR. **Ограничение диагностики:** чистый no-eval replay8 совпал; два полных no-eval200 отдельно не выполнялись. Не утверждается, что на восьми шагах воспроизведён CUDA-недетерминизм. Успех gate основан на четырёх прямых restore-аудитах, не на допуске/tolerance к ошибке восстановления.

Обе сборки пересобраны, все95 source SHA совпадают с серверным v2. `smoke_passed:true`, `ready:true`, команда/ETA обновлены в [next-b3.json](../results/next-b3.json). Запуск production остаётся только пользователю после17.10 00:00UTC и проверки≥42000с свободной квоты.

## Находки и исправления

Пути относительно `work/`, строки после исправления.

| Приоритет | Находка и исправление | Регрессия |
| --- | --- | --- |
| **P1** | Два итоговых assert превращали сохранённый `partial` в worker exit1 и затем ошибку notebook — повтор риска B1. `kaggle/b3_worker.py:40–43` теперь печатает статус; `kaggle/build_b3.py:63–64` публикует `b3-publication.json`, не выбрасывая исключение после supervisor. `session-summary.status` остаётся `partial`, успешность оценки не подделывается. | `b3_tests.py:108`: реальная tiny-оценка возвращает неполные metrics; выполняются настоящие завершающие statements worker и notebook, исключения нет, full stable/Adam/scaler/RNG побитно прежние. |
| **P1: полнота smoke** | Было только два update, что не покрывает требование сотен шагов. `kaggle/src/b3_gpu_smoke.py:25–50`: **200 main + независимый control200** на каждой настоящей B2-модели, сравнение model/Adam/scaler/RNG и stability/cursor/LR после probe+final. В smoke cooldown2/3с, greedy64/old16/CE, test64 и K8 на36 frozen tune IDs. Deadline660/watchdog690/backend720с (`build_b3.py:29–30`). | `b3_tests.py:169`: весь smoke controller на tiny QK/Adam с enabled CPU GradScaler, control200 + main200, restore, injected overflow и recovery. Hardware APIs/evaluator заменены только в этом CPU-тесте. Настоящее CUDA-подтверждение — ниже. |
| P2 | Preflight ошибочно называл mount resume-датасета `cache_mount` из-за перезаписи `source`. `build_b3.py:60–62`: отдельно сохраняются cache и resume mount. | `b3_tests.py:129`: исполняет строки с разными mount и сверяет оба поля receipt. |
| P2 | `b3_static_check.py` каждый раз переписывал handoff в `ready:true/GPU_smoke_pending:true`, теряя результат smoke. `kaggle/b3_static_check.py:7–9,71–85`: ready только по успешному GPU receipt **с теми же SHA модулей**; сохраняются `smoke_passed`, команда и ETA. | `b3_tests.py:142`: отсутствие receipt, провал и чужие SHA не проходят; совпадающий successful receipt проходит. |

## Проверенные риски

- **Строгое продолжение.** `b3_runtime.py:15–39`, `b2_runtime.py:136–164`, `l1_runtime.py:112–148`: исходные pinned latest/identity/spec/cache/order, model, полный AdamW со step/моментами, GradScaler, Python/NumPy/CPU/CUDA RNG, cursor и history. Torch2.11.0+cu128/CUDA12.8 и все10 implementation SHA B2 проверяются до обучения; fresh fallback отсутствует. QK LayerNorm и z-loss1e-4 **не менялись**. LR0,001/0,0006 constant; updates34153/32499 уже после warmup2000. Реальный next-batch128 CPU проверен подготовкой; CUDA replay проверяется smoke.
- **Перенос.** Private `thisarivson/nlp-hw02-b2-states/1`: API повторно подтвердил ready/private, точный inventory **7 файлов / 2767972202байта**. Два full PT1,3615GB, два latest, order.npy/order.json и manifest; model-only FP16 не подставляется. [Исходный source+remote SHA round-trip](../checks/b3-transfer.json) и [повторный API inventory](../checks/b3-review-inputs.json). SHA full GPU0 `c1fb5913…a1c795f0`, GPU1 `a3895df2…bf215bb3`, manifest `f42252b2…2996bc`. На mount manifest fingerprint проверяет notebook, каждый payload — `b1_resume.py:9–29`; B2 reader снова сверяет PT bytes/SHA.
- **Точки и restore.** `b3_runtime.py:66–69,77–114,119–124`: абсолютные main_s37800/48600, каждая с отдельным900с linear cooldown; main_pairs берутся из cursor. После любой оценки/исключения восстанавливается полный uncooled stable; добавляются только история и счётчики branch compute. CPU проверяет три anchors и совпадение с no-eval control. Final tag и receipt используют фактический main_s; недостигнутый cap не выдаётся за достигнутый.
- **Final.** `build_b3.py:23–25`, `b3_runtime.py:82–105,154–168`, `b3_worker.py:33–39`: **3000с timed linear**, q0, затем полный greedy4096/old991 + teacher-forced compact30 CE, новый test1919 timer, K8/T0,7/seed270042 на frozen tune2048. `b1_eval.py:13–16`, `l2_eval.py:32–100,102–120` сохраняют поштучные samples/logprob/execution/ID/SHA. Полнота проверяется отдельно, частичный K8 не считается complete. Финальные FP16 отделены от двух последних **uncooled** full PT для B4.
- **Бюджет.** Backend40800с=11ч20мин; worker40500; watchdog40740. Setup≤480с; main cap32400с. `final_reserve=3000+1800+240+60=5100с` всегда ограничивает main; probe требует ещё свои1350с +120с. I/O уменьшает main; ровно9ч не гарантируются. `b3_tests.py:150` проверяет пропуск probe при нехватке резерва с сохранённым final; статус при этом partial. Штатный reserve оставляет240с после worker и60с до backend. `b3_tests.py:158` проверяет настоящие deadlines в обеих сборках, включая наследуемый12ч Watchdog. Реальные TERM/KILL и emergency partial-output также прошли на CPU.
- **Output.** `b2_runtime.py:112–134` — atomic latest после записи полного PT, ротация до двух поколений. Штатно четыре full PT≈5,446GB + шесть evaluated FP16≈1,362GB + order44,9MB/журналы. При одновременной записи двух новых PT временный пик≈9,6GB, ниже guard10GiB. Исходные states/cache живут в read-only input или temp вне `/kaggle/working`; ZIP не создаётся. GPU-smoke дополнительно даёт измеренный размер output ниже.
- **Пути входов.** Owner-qualified `/kaggle/input/datasets/thisarivson/nlp-hw02-b2-states`, `/kaggle/input/datasets/thisarivson/nlp-hw02-26b-inputs`, `/kaggle/input/notebooks/thisarivson/nlp-hw02-compact-cache-verified`; flat fallback. Dataset version дополнительно защищён manifest SHA, cache pinned `/2` и всеми payload SHA. Точные реальные mount — в GPU preflight.

Без live-sync остаётся платформенный риск потери output при сбое VM; завершение Python без исключения устраняет повтор B1 на штатном partial, но не гарантирует сохранность при сбое Kaggle. Full production3000с cooldown/2048 K8/11ч20мин не выполнялись; smoke проверяет те же модули на сокращённых данных/времени с generation cap8, не качество и не холодный180с inference.

## Проверки и GPU-smoke

15 B3 CPU tests +9 B2 +5 B2-review +20 B1 = **49 passed**; отдельно реальные watchdog TERM/KILL/hard exit. [B3 CPU](../checks/b3-cpu.json), [статическая сборка](../checks/b3-static.json). Все10 B2 implementation SHA неизменны; production и серверный smoke содержат одинаковые runtime sources.

Smoke v1 отправлен через API **16.10.2026 18:46:23UTC**, private [thisarivson/nlp-hw02-b3-gpu-smoke/1](https://www.kaggle.com/code/thisarivson/nlp-hw02-b3-gpu-smoke/versions?versionNumber=1), backend720с. Квота перед push3936,950с; reset17.10 00:00UTC, fallback не понадобился. [Push](../checks/b3-review-smoke-push-v1.json), [серверные SHA](../checks/b3-review-smoke-metadata-v1.json). **На19:07UTC статус RUNNING, files0, logs недоступны; функциональный результат не подтверждён.** `smoke_passed:false`, `ready:false`. Повторного push не было. На18:59:42UTC квота уже уменьшилась на798,606508с (>720с), затем продолжала уменьшаться: **потолок списания12мин не выдержан**. Причину/разделение setup, code и upload без output установить нельзя; утверждать успешный CUDA resume/restore/K8 нельзя. [Состояние и блокер](../checks/b3-review-smoke-pending.json), [квота во время выполнения](../checks/b3-review-quota-during.json).

SDK отмены требует `kernel_session_id`; официальные status/metadata его не вернули. Live stream по owner/slug даёт ReadTimeout; SDK stream без session ID —400. [Issue Kaggle CLI#1172](https://github.com/Kaggle/kaggle-cli/issues/1172) предлагает использовать versionNumber, но read-only probe ID1 вернул404; отменять с угаданным ID небезопасно, cancel к нему не отправлялся. Доступного браузера нет. Пользователю отправлен запрос отменить **только smoke** либо передать точный session ID. **До получения output и успешного gate боевой B3 не рекомендован.**

Это также ограничивает вывод о бюджете B3: проверены first-cell deadlines и request timeout, но фактический потолок списания всей платформенной сессии11ч20мин не доказан. Наблюдение smoke не позволяет считать backend timeout гарантией такого потолка.

## Боевой бандл и запуск пользователем

`kaggle/bundles/b3_thisarivson/`; private `thisarivson/nlp-hw02-b3`, T4×2/internet, Secrets-free. **Production push=0.** После успешного smoke gate и reset17.10 00:00UTC =03:00МСК проверить свободную квоту≥42000с. Из `work/kaggle`:

```sh
KAGGLE_CONFIG_DIR=~/.kaggle kaggle kernels push -p bundles/b3_thisarivson --accelerator NvidiaTeslaT4 --timeout 40800
```

ETA кода: фактический старт первой code cell + **до11ч20мин**. Очередь/подготовка VM и публикация output добавляются к этому времени; platform `COMPLETE` может появиться позже. Например, старт17.10 в08:00МСК → бюджет кода до19:20МСК (16:20UTC), затем возможна выгрузка. Машинная передача — [next-b3.json](../results/next-b3.json); ready требует successful smoke с совпадающими SHA.

## Коммит

**Дополнение после успешного v2:** `git add` и `git commit` повторно попробованы; оба завершились128: sandbox запрещает создание shared Git `index.lock` (`Operation not permitted`). Индекс пуст, коммит не создан, обход ограничений не выполнялся. [Новая квитанция](../checks/b3-fix-git.json). Код, обе сборки, successful GPU gate и ready handoff сохранены.

`git add` завершился128: sandbox запретил создание `<vault>/.git/worktrees/study1/index.lock` (`Operation not permitted`). Индекс не изменён, `git commit` не запускался; обхода ограничений не было. [Квитанция](../checks/b3-review-git.json). Все исправления/отчёт/бандлы сохранены; GPU gate пока заблокирован внешней сессией Kaggle.

## Источники

- **kaggle-b3-independent-review** — собственное независимое ревью16.10.2026, код и receipts выше; bibliography — kaggle-b3-independent-review.
- **kaggle-cli-cancel-api** — [Kaggle CLI#1172](https://github.com/Kaggle/kaggle-cli/issues/1172), предложение API отмены; утверждение issue о соответствии versionNumber/session ID не принято без проверки.
- [b3-plan](b3-plan.md), [b2-review](b2-review.md), [b2-results](b2-results.md) — подготовка, исходный рецепт и измеренные результаты. Новых научных работ нет; контракт воспроизводимости PyTorch указан ниже.
- **pytorch-reproducibility** — [PyTorch2.11: deterministic algorithms](https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html); bibliography — pytorch-reproducibility. Общий контракт CUDA-воспроизводимости, не доказательство конкретного виновного ядра.
