---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-16
updated: 2026-10-16
sources:
  - kaggle-b3-preparation
  - kaggle-b2-results
  - kaggle-b2-independent-review
  - kaggle-b1-ev-b2
aliases: []
---

# B3: продолжение обеих W-QK после B2

**Пользователь выбрал продолжение обеих веток**, несмотря на более узкую рекомендацию [b2-results](b2-results.md). Private `thisarivson/nlp-hw02-b3`, title `nlp-hw02-b3`, T4×2, internet on. Сборка не запускает GPU. Независимое [ревью B3](b3-review.md) выполняет разрешённый пользователем смоук **до reset**, на сгорающей квоте thisarivson. Production запускает **только пользователь утром после17.10.2026 00:00 UTC =03:00МСК**, с повторной проверкой квоты.

Гипотеза: ещё около 9ч main и первый 50-минутный cooldown дадут полезный прирост W и материал для проверки ансамбля W+C. Это план измерения, а не прогноз качества. B3 не меняет C или текущий final inference v7.

## Исходные состояния и строгий resume

Pinned output `friend2/nlp-hw02-b2/1`; [полные latest и SHA](../kaggle/b3-anchors.json).

| GPU | Ветка | Updates / main-пары | Накопленный main_s | Uncooled checkpoint |
| --- | --- | --- | --- | --- |
| 0 | W-lr1000 | 34153 / 4371584 | 27000,558216 | `main-4b21b0d306c4.pt` |
| 1 | W-lr600 | 32499 / 4159872 | 27000,345398 | `main-612d2a094343.pt` |

Обе W e6/d6/d768, QK LayerNorm, z-loss1e-4, batch128/micro32, dropout0/q0. LR **constant 0,001 / 0,0006**, с прежним emergency factor; warmup2000 уже пройден и заново не начинается. Полностью восстанавливаются model, AdamW с моментами/step, GradScaler, Python/NumPy/Torch CPU/CUDA RNG, cursor, stability/points/sessions. Исходная история B2 остаётся в full states; новые журналы `updates.jsonl` продолжают прежние updates/main_s/main_pairs.

Weighted order не строится заново: переносится исходный `order.npy`, SHA `8e9595f655ed2b4456bf5b94e9b5b6b3db040e3f5696a4dc0ef364588048bee8`, seed2026101501, threshold79, long×2. Cache manifest SHA `46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be`.

Новый [B3 runner](../kaggle/src/b3_runtime.py) меняет только оркестрацию. `b2_runtime.py`, `b2_optim.py`, QK и остальные 10 implementation pins B2 **не изменяются**. Первая ячейка сразу проверяет runtime; `B.read_full` повторно проверяет Torch **2.11.0+cu128**, CUDA **12.8**, все implementation SHA и рецепт. При несовместимом Kaggle image сессия останавливается; fresh fallback отсутствует. Прежняя NaN-recovery: rollback полного stable, сохранение затраченного времени/истории, LR-factor÷2; нормальные AMP skips учитываются отдельно.

## Перенос без накопления full PT

[b1_transfer.py --stream-states](../kaggle/b1_transfer.py) читает output только через `KAGGLE_CONFIG_DIR=~/.kaggle-friend2`; создаёт **новый private** `thisarivson/nlp-hw02-b2-states` через default `~/.kaggle`. Файлы конфигурации и значения credentials напрямую не читаются и не печатаются. Env credentials исключаются, SDK diagnostics скрыты.

Порядок для каждого файла: pinned download → source SHA/bytes → CPU-проверка, если нужна → upload blob → unlink. В staging не бывает двух payload одновременно. Dataset создаётся один раз, затем private/ready/inventory проверяются API; каждый удалённый файл скачивается отдельно для SHA round-trip и удаляется. При CLI ZIP-обёртке временно существуют архив и один распакованный файл, максимум около 2,72GB; две модели локально вместе не хранятся. Повторно создавать существующий slug скрипт запрещает.

Переносятся только обе latest, два full stable, order.json/order.npy и transfer manifest. [Квитанция переноса](../checks/b3-transfer.json); fingerprint transfer manifest зашит в обе сборки. CPU-проверка выполняется **до загрузки** каждого полного состояния. Выбранные реальные 128 следующих индексов фиксируются при проверке исходного order; [order/cache audit](../checks/b3-order.json).

Входы notebook:

- `thisarivson/nlp-hw02-b2-states` (конфигурация resume дополнительно фиксирует версию Dataset и SHA manifest);
- `thisarivson/nlp-hw02-compact-cache-verified/2`;
- `thisarivson/nlp-hw02-26b-inputs`.

## Main, оценки и hard budget

Бюджет на **всю двух-GPU сессию**, от первой code cell: backend **40800с = 11ч20мин**, workers40500с, watchdog40740с. Setup cap480с; atomic stable каждые900с wall, перед/после каждой ветки и в конце. Без Secrets/live-sync: outputs и checkpoints находятся в `/kaggle/working`; watchdog завершает процессы и сохраняет summary, уже опубликованный stable остаётся доступен. Сбой VM может потерять последний интервал main; это не гарантия платформенной выгрузки outputs.

| Точка | Накопленный main_s | Обучение disposable ветки | Оценка |
| --- | --- | --- | --- |
| Исходная B2 | ≈27000 | Уже измерена в B2 | История сохранена |
| B3 probe1 | 37800 (≈+3ч) | 900с linear | greedy heldout4096 / old991 / teacher-forced compact30 CE обоих |
| B3 probe2 | 48600 (≈+6ч) | 900с linear | То же |
| B3 final | cap≈59400 (≈+9ч), фактический anchor в receipt | **3000с linear** | greedy4k / old991 / CE + test1919 timing + K8 tune2048 |

Время main — сумма времени updates, без setup/I/O/cooldown/eval. Метки 37800/48600 абсолютные, продолжают кривую B2; фактические main_s, updates и **main_pairs** записаны в training/complete/summary каждого anchor. Из-за разной скорости LR-веток пары будут различаться. Отдельных дополнительных pair-gates нет: график по парам строится из фактических counters этих же точек, включая точки B2.

На промежуточную оценку cap450с, на финальную1800с. Cap main32400с — верхняя граница, **примерно 8ч40мин–9ч** практически: до окончательного cooldown всегда резервируются3000+1800+240+60=5100с. Промежуточная проба допускается только вместе с резервом финальной ветки; нехватка main-времени записывается как budget stop, пропущенная/незавершённая оценка — partial. Бюджет важнее достижения ровно9ч main. Полный план без I/O:32400+2×(900+450)+3000+1800+480=40380с; часть main при необходимости уступает I/O и запасу. Жёсткая граница40800с не меняется.

Все cooldown — timed **linear** из текущего LR: $\eta(t)=\eta_0(1-t/T)$, где $t$ — сумма времени updates данной ветки, $T=900$ или3000с. Как в B2, LR вычисляется до update; искусственного дополнительного нулевого шага нет. Используются QK/z-loss и следующий префикс того же order. После оценки восстанавливается **полный uncooled anchor**, включая RNG и cursor. Сохраняются охлаждённые FP16 `weights/b3-*-fp16.pt`, полный cooldown journal и per-example generations/executions/reference/question SHA.

В конце `latest.json` и два поколения full PT остаются **uncooled для B4**, summary содержит `B4_uncooled_latest`. Финальный FP16 — отдельный файл. K8/T0,7/seed270042 сохраняет все samples, logprob, executions и исходные tune IDs; можно офлайн смешивать W+C. Это сбор журналов, политика ансамбля здесь не выбирается. Test timing — реальный reload/tokenizer/generation/restore/execute/answers-close на1919; cold T4≤180с для production ансамбля остаётся отдельным gate.

## Проверки и передача ревьюеру

- Реальные full B2: [GPU0 CPU-resume](../checks/b3-real-resume-gpu0.json), [GPU1 CPU-resume](../checks/b3-real-resume-gpu1.json). Все model/Adam/scaler/state/scheduler/RNG загружены; следующий настоящий effective batch128 через B3/B2 route побитно сравнивается с прямым full-state control. На CPU используются FP32/no-autocast, enabled CPU GradScaler из B2, fused выключается **после** проверки загруженных optimizer tensors/settings. CUDA RNG bytes проверяются в payload; их исполнение проверит GPU-smoke, CPU не доказывает CUDA bit-exactness.
- [Восемь B3 tests](../checks/b3-cpu.json): две cumulative пробы и final с точным restore, linear cooldown, reserve, partial eval, recovery, неверный anchor, CPU AMP-адаптер и точные frozen tune IDs в смоуке. [34 B2/B1 regressions](../checks/b3-b2-regressions.json).
- [Статика](../checks/b3-static.json): compile/nbformat, все embedded SHA, неизменные B2 pins, одинаковые модули smoke/production, private/title/inputs/limits/Secrets-free.
- Смоук-бандл **≤720с=12мин** использует реальные обе B2 full states, 200 настоящих CUDA main updates + control200 на каждой ветке, probe+final mini greedy/CE/test/K8 и прямой before/after restore-аудит, AMP overflow и recovery. После v1 сравнение независимых200-step траекторий оставлено диагностикой, добавлен отдельный no-eval replay8; [v2 COMPLETE и четыре точных restore](b3-review.md). Никакой fresh модели. Full test timing/качество не утверждает. **В этой подготовке GPU и kernel push =0.**

**CPU/transfer/static готовы**: обе реальные проверки, 8 B3 tests и 34 B2/B1 regressions прошли. Dataset v1; transfer manifest SHA `f42252b28725a998945f8e086bb3042395fa7460b83b645c225ecd25de2996bc`.

Машинная передача: [results/next-b3.json](../results/next-b3.json), `ready:true` только после CPU/static gates и успешного GPU-smoke с теми же module SHA. Production `kaggle/bundles/b3_thisarivson/`; smoke `kaggle/bundles/b3_gpu_smoke_thisarivson/`. Сборки локальные/ignored, notebook и builder версионируются.

```sh
# Из work/kaggle, только воспроизведение сборки и проверок, без запуска GPU:
python build_b3.py
python build_b3.py --gpu-smoke
python b3_tests.py
python b3_static_check.py

# Ревьюер: smoke разрешён до reset на сгорающей квоте:
KAGGLE_CONFIG_DIR=~/.kaggle kaggle kernels push -p bundles/b3_gpu_smoke_thisarivson \
  --accelerator NvidiaTeslaT4 --timeout 720
# Только пользователь утром после reset, successful smoke и проверки квоты:
KAGGLE_CONFIG_DIR=~/.kaggle kaggle kernels push -p bundles/b3_thisarivson \
  --accelerator NvidiaTeslaT4 --timeout 40800
```

После независимого ревью итоговые assert worker/notebook заменены сохранением статуса partial; исправлен preflight mount и добавлен smoke/ready gate. Текущие результаты — [b3-review](b3-review.md), первоначальные CPU/transfer receipts выше описывают подготовку.

## Коммит

`git add` завершился128: песочница запретила создание shared Git `index.lock` (`Operation not permitted`). Индекс не изменён, `git commit` не запускался; обхода ограничений не было. [Квитанция](../checks/b3-git.json). Изменения и готовые бандлы сохранены.

## Источники

- **kaggle-b3-preparation** — собственная подготовка16.10.2026, [B3 runner](../kaggle/src/b3_runtime.py), [anchors](../kaggle/b3-anchors.json), CPU/transfer/static receipts выше; bibliography — kaggle-b3-preparation.
- [b2-results](b2-results.md), [b2-review](b2-review.md), [b1-ev-b2-plan](b1-ev-b2-plan.md) — измеренные исходные результаты и прежний рецепт. Новых внешних работ в B3 нет.
