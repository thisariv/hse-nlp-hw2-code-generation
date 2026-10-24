---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-12
updated: 2026-10-13
sources:
  - kaggle-l1
  - kaggle-l2-preparation
  - course-methods-review
aliases: []
---

# L2: продолжение C и удвоение длинных строк

Результаты боевой UI v2 и CPU-вердикт: [l2-results](l2-results.md). Протокол ниже сохранён; mixed eval и полный таймер одной T4 остаются отдельным шагом.

**Гипотеза:** дополнительные предъявления длинных исходных train-пар улучшат слабый длинный срез без заметной потери короткого. GPU0 — C-main, GPU1 — C-long2x; один общий anchor, одинаковое накопленное main-время L2 и одинаковые cooldown. D больше не обучается. GPU-смоук и боевой push выполняет независимый ревьюер.

## Основание и контроль

Из сообщения пользователя: L1 `357694322` остановилась около18 881с. D получила nonfinite loss около17 450с main; общий stop-request прекратил здоровую C. C после9 000с main + cooldown:2575/4096=62.87%; длины79.2/53.0/36.9/20.3%; test1919 greedy33.7с. Это сообщения о боевом L1, а не результат новой L2.

Проверяемый anchor: private `thisarivson/nlp-hw02-l1-live`, текущая v7, `model-C__checkpoints__main-5746a4a2897e.pt`. SHA `1da28095d960fb0958ff26883bc6dc832e4307c9bab4023a5621b1f9f6ff01dc`,644 049 235байт;47 155attempted updates, cursor6 035 840, main17 569.935878398с. AdamW, FP32 weights, GradScaler scale4194304/growth1079, Python/NumPy/Torch CPU/visible CUDA RNG, history сохранены. Torch2.11.0+cu128/CUDA12.8 и semantic SHA L1 проверяются строго до install/cache scan. Обновлённый Kaggle image должен отказать сразу; не мигрировать состояние неявно.

C-main сохраняет **исходную recipe identity, order SHA и cursor**, LR0.0015 без warmup. Неизменённый `l1_runtime.py` остаётся эталоном finite update. Новая политика аварийного восстановления одинакова у обеих веток и записывается отдельно в full-state format L2, включая hash runtime. C-long2x получает те же tensors/Adam/scaler/RNG/history, собственную identity, новый order и cursor0. Абсолютные updates наследуются; cursor origin записан явно, поэтому cursor не обязан равняться updates×128. L2 full state и `read_full` поддерживают дальнейшее продолжение; evaluated FP16 — самостоятельный inference artifact.

## Самплер C-long2x

Вес строки равен2 при длине target≥порог, иначе1. Weighted epoch содержит ровно одну копию каждой train-строки и одну дополнительную копию каждой длинной; дополнительного множителя loss нет. Effective batch128, micro64 из C anchor, bucketing4096, token CE/EOS, AdamW/wd0.01, AMP/clip1 прежние. При позднем OOM micro уменьшается с накоплением до128.

Целевая доля —626/4096=15.283203125% heldout reference длиной≥90 original BPE, без специальных tokens. В заданных mounts нет полного исходного BPE-кеша тех же9 725 737строк; локально исходные offsets есть только для18/60shards. Поэтому используется **длина compact target из verified offsets**, единообразно для всего train. Код может принять полный original cache только после SHA source offsets/provenance/question identity каждого shard.

Ближайший достижимый целочисленный порог —**79compact BPE**:1 502 074строки=15.4443205692%; отклонение+0.1611174442п.п. Точное равенство долей невозможно при целочисленном пороге и одинаковом весе всех равных длин. Weighted epoch11 227 811слотов, доля long среди предъявлений26.7563107359%. Seed2026101302; order SHA `72bc3dfd3a3e738979c489c0cf8f41cde092c41d4ae90f2de8378a034603a4db`. Детерминированное построение/проверка multiplicity выполняются на CPU до workers; статистика и SHA записываются в `weighted-order/order.json`. Order включён в live, чтобы L3 не зависела от повторного построения.

## Парный протокол и бюджет

Все time targets — **добавленное main-время L2**, с синхронизацией CUDA и неуспешными попытками; setup/checkpoint I/O/оценка/cooldown исключены. После rollback уже потраченное main-время не исчезает; committed updates/cursor возвращаются к checkpoint, потерянная работа видна в истории. Расхождение anchors≤5с; фактические времена публикуются. Периодические marks+3ч/+6ч, финальная+8ч. Если target/cooldown не достигнуты, точка incomplete и не считается парной.

| Фаза, одновременно на обеих T4 | Максимум wall |
|---|---:|
| Secrets/T4 gate, C preflight, install, private live probe, cache/order |15мин|
| Main:3ч →3ч →2ч |8ч|
| Два промежуточных timed cooldown,15мин каждый |30мин|
| Две промежуточные greedy/CE/test оценки,≤15мин каждая |30мин|
| Финальный timed cooldown |50мин|
| Финальная greedy/CE/test≤15мин + K8 tune≤15мин |30мин|
| Checkpoint/restore/bookkeeping reserve |25мин|
| **Workers от первой ячейки** |**11ч**|
| Финальные live/report sync |до20мин|
| Hard watchdog reserve |4мин|
| **Абсолютный hard wall** |**11.4ч**|

Kaggle server timeout12ч; quota12.5ч оставляет≤1.1ч на смоук/прочие расходы при максимальном wall. Подготовленный GPU-smoke artifact:15мин worker/17мин hard/18мин backend, без Secrets/live. Само обучение L2 не сокращается асимметрично ради зависшей оценки. При недостатке резерва точка incomplete, stable main сохранён.

Cooldown — disposable linear decay от фактического anchor LR до почти0 за одинаковое **main-update wall время**15/50мин; разрешён overshoot одного update. Counts могут различаться из-за sampler throughput. Сохранить full stable anchor → cooldown FP16 → отдельная eval-модель → точное восстановление model/Adam/scaler/RNG/cursor и продолжение main. Final stable state остаётся постоянного LR для L3.

## Оценка и критерии решения

Greedy exec всех heldout4096 и old991, original BPE bins<60/60–90/90–120/≥120; unsmoothed compact CE. Fixed split точно из [candidate-selection-cpu](candidate-selection-cpu.md): SHA256 `candidate-selection-v1:heldout4k:{id}`, первые2048ID tune, вторые2048eval. Сохранённые eval2048 при промежуточных marks — описательная диагностика; выбирать sampler только по tune, не менять решения по eval. Fresh не оценивается.

В финале K8,T0.7 **на tune2048**, независимый cap900с на каждой GPU. Каждый завершённый batch сразу публикует per-example greedy correctness, sample code, execution status/value, length и untempered mean logprob с EOS. Timeout оставляет полноценные batch journals и явный incomplete, без пересчёта частичной точности как full2048. CPU `l2_compare.py` считает C-main K8, C-long K8 и C-main firstK4+C-long firstK4, с greedy C-main как фиксированным tie anchor; gold используется только после выбора для оценки. Это не доказательство полного mixed GPU timer.

Критерии из [course-methods-review — Протоколы трёх направлений GPU1](course-methods-review.md#протоколы-трёх-направлений-gpu1) перенесены на matched +3/+6ч: tune overallΔ≥+0.5п.п., ≥90BPEΔ≥+3п.п., <60BPE не хуже−0.5п.п. Нейтральный overall−0.3…+0.5 при положительном long — specialist только если frozen mixed vote≥+0.5п.п. при том жеK. Финально eval одиночной модели≥+1п.п. либо mixed vote≥+0.7п.п., paired CI/gains/losses, без сильного отрицательного old991 и full pipeline≤180с. Финальный K8 eval/mixed timer не входят в текущую задачу: K8 tune gate даёт основание для следующего отдельного inference-only опыта, а не готовую production победу.

Fulltest1919 greedy заново на каждой точке: input read + weights/tokenizer load + generate/restore + bounded exec + answers close; без gold. C-main/C-long test timers отдельно; mixed K4+K4 на одной T4≤180с пока не установлен.

## Устойчивость и live

Сбой одного worker записывается, второй продолжает до своего бюджета. У каждого атомарный latest full checkpoint и предшественник, каждые1800с wall, до/после оценки и в конце. Nonfinite loss/нештатный gradient:zero_grad, без optimizer.step, счётчик; обычный GradScaler found_inf skip отдельно и без эскалации. При3подряд или10/100attempts откат full stable и LR÷2; на третьей эскалации штатная остановка только этого worker, good state/history сохранены.3отсекает устойчивую серию;10/100 допускает редкие события, выявляет10%отказов; две попытки понижения LR ограничивают потерю времени. Post-Adam nonfinite также откатывается; nonfinite checkpoint никогда не заменяет good receipt.

Private `thisarivson/nlp-hw02-l2-live` создаётся заранее; server owner-private/ready startup probe обязателен в боевом режиме. Secrets только в parent/helper env; worker env без credentials. Полный latest обеих C + последняя cooled point со всеми complete/partial per-example journals + weighted order; flat staging≤2.5GB. Heartbeat5400с, retries≥300с; страховка≥7200с требует штатной остановки. Прежние точки сохраняются в immutable Dataset versions и полном kernel output. Ротация full states≤2/worker. Optional averaging snapshots **не включены**: приоритет — восстановимый полный Adam и journals; расширение не добавлено перед ревью.

## Проверки и артефакты

- [24 L2 регрессии](../checks/kaggle-l2-regressions.json), `../kaggle/l2_tests.py`; также прошли16 прежних L1 регрессий. Реальные enabled CPU GradScaler overflow/reject→следующий update и восстановление после nonfinite cooldown проверены.
- [Реальный CPU resume](../checks/kaggle-l2-real-resume.json), `../kaggle/l2_cpu_smoke.py`: тот же C и фактические train batch128; CPU FP32 forward/Adam со restored enabled scaler, finite update L1/C-main сравнивается бит-в-бит. CUDA kernels/FP16/fused численная идентичность остаётся ревьюеру.
- [Current private L1 input](../checks/kaggle-l2-live-input.json), [private L2 precreation](../checks/kaggle-l2-live-created.json), [реестр](../kaggle/runs.json).
- [Статическая сборка](../checks/kaggle-l2-static.json): все code cells compile, embedded modules совпадают с файлами, private/T4/inputs/pins/budgets проверены, outputs пустые. `../kaggle/l2_static_check.py` не выполняет notebook.
- `../kaggle/build_l2.py`, `src/l2_*.py`, `l2_worker.py`, `l2_compare.py`, `l2-anchor.json`, notebooks и bundles L2/smoke. Ничего не push/run автоматически.

## Источники

- [l1-long-training](l1-long-training.md), [l1-review](l1-review.md), [course-methods-review](course-methods-review.md), [candidate-selection-cpu](candidate-selection-cpu.md); сообщение пользователя и проверенный saved C. Самплер, пороги recovery и бюджет — решения этого опыта, не обещания качества из литературы.
- [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html): accumulation, unscale, clipping и штатный пропуск found_inf.
- [Kaggle Datasets CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md): private create, versions и отдельные файловые downloads. Авторизацию обслуживает SDK; значения credentials не инспектируются и не попадают в отчёты.
