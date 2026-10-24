---
type: experiment
course:
  - nlp
status: learning
understanding: 4
created: 2026-10-20
updated: 2026-10-20
sources:
  - kaggle-b6-results
  - kaggle-b5-preparation
  - hagele2024-cooldown
aliases: []
---

# B7: удвоенный cooldown из тех же uncooled состояний B5

**W N=33 000, C-long2x N=65 800; сокращение N не требуется.** Private `friend1/nlp-hw02-b7`, T4×2, internet on. [Handoff](../results/next-b7.json) содержит готовность, source/config SHA, бюджет, бандлы и команды. Пользователь уже разрешил production после независимого ревью и успешного smoke; **GPU запускает только ревьюер**. Подготовка делает только CPU-проверки и read-only quota. E2, `submission/`, предметные progress/index/dashboard/backlog не меняются.

## Гипотеза и единственное изменение обучения

[B6](b6-results.md) дала greedy W72,41% / C72,61%, к B5 final50 +1,46/+1,49п.п.; улучшение ансамбля не установлено. B7 проверяет дальнейшее увеличение длины того же $1-\sqrt{\cdot}$ cooldown. Результат заранее неизвестен.

В [Hägele et al., 2024, §3 и Fig.5–6](https://arxiv.org/html/2405.18392v2#S3) cooldown порядка20% **полного** обучения — опорный режим; достаточная длина зависит от общей длительности, для длинных запусков бывает достаточно меньшей доли. Это мотивация, а не универсальная гарантия оптимума для нашей seq2seq-задачи. Заданные N составляют21,35%/22,41% от main updates и17,59%/18,30% от main+новый cooldown. N не подгоняется под буквальные20%: выполняется выбранное пользователем удвоение B6.

$$\eta_j=\eta_0\left(1-\sqrt{j/N}\right),\quad j=1,\ldots,N.$$

$\eta_0$ — унаследованный stable LR: W0,0006, C0,0015. $j$ — номер attempted update; AMP/nonfinite skip также расходует слот и128пар. Ровно N попыток, последний update с $\eta_N=0$, без дополнительного шага. $q=0$, прежние AdamW/scaler, effective batch128, weighted order/cursor, FP32 модель и все RNG. Нового main/warmup нет.

Все embedded modules и исполняемые ячейки **побайтно совпадают с B6**, кроме CONFIG; разрешённые отличия CONFIG — N, имя сессии/сравнения и временные пределы. Это проверяют [CPU-тесты](../kaggle/b7_tests.py) и [статика](../kaggle/b7_static_check.py). `b6_pins`, порт/math, C overrides и остальные поля CONFIG неизменны. Файлы runtime, формат checkpoint `nlp-hw02-b6-full-v1`, внутренние имена `b6-final`, `b6-mid`, `b6-publication.json` и подписи B6 в worker logs намеренно сохранены: B7 определяется kernel/session-config и N.

## Исходные состояния и resume

Прямой mount **`friend1/nlp-hw02-b5/6`**, исходный [b6-anchors.json](../kaggle/b6-anchors.json) без изменений. Manifest SHA `f662a966b5031a93bb46d25530a2710753a26f69e5262af80eca05ac6c6aa525`. Не используются охлаждённые веса B5/B6.

| Поле | GPU0 W600 | GPU1 C-long2x |
|---|---|---|
| Full PT |`model-GPU0/checkpoints/main-789d436a3fb0.pt`|`model-GPU1/checkpoints/main-6b2c636e5f7f.pt`|
| SHA256 |`3b11360d5242e0e8b3fc1ac38254f0d4d544fba8bb62ee4c7bb034943e6cdc32`|`69dff4405d2236097f48733adbfc9e3bfd5d20b501a3fcfd1dae62ec2ce63074`|
| Main updates / начальный cursor |154559 /19783552|293688 /31556224|
| Новые предъявления / конечный cursor |4224000 /24007552|8422400 /39978624|
| Order SHA |`8e9595f655ed2b4456bf5b94e9b5b6b3db040e3f5696a4dc0ef364588048bee8`|`72bc3dfd3a3e738979c489c0cf8f41cde092c41d4ae90f2de8378a034603a4db`|

Сохраняется C cursor offset и изолированный `src-C`. Mount проверяет bytes/SHA latest PT, manifest/final anchors и orders; training проверяет Torch2.11.0+cu128/CUDA12.8 и frozen implementation SHA. CPU-подготовка B7 **переиспользует** реальные batch128/disk-resume receipts B6 при совпадении исходников и anchors, не скачивает заново многогигабайтные состояния. Новые N дополнительно проверяются через численный fixture replay обеих веток с disk save/restore между шагами; полное расписание и cursor — на всех98 800слотах. CUDA остаётся проверкой нового smoke.

Сохраняется ограничение B6: B5 удалила pre-final PT; проверены metadata этих anchors и численное равенство двух сохранённых post-restore поколений, а не tensors удалённых файлов.

## Бюджет по фактической B6

[Расчёт](../kaggle/b7_budget.py) заново читает **все16500/32900** строки B6 `cooldown-updates.jsonl`, сверяет bytes/SHA с producer manifest и суммы с complete summaries. Бюджет всей сессии — максимум двух одновременно работающих веток.

| Расчёт | W | C-long2x |
|---|---:|---:|
| B6 pure update time, с |12896,692|13876,533|
| B6 updates/с |1,279398|2,370909|
| B7 N |33000|65800|
| B7 pure update time, с |25793,384|27753,066|
| Update time×1,10 +480setup +1800eval/K8 +600I/O, с |31252,722|33408,373|
| Прогноз всей ветки |8ч40мин53с|9ч16мин48с|
| Запас до hard wall, с |9547,278|7391,627|

**Hard backend timeout40 800с=11ч20мин от первой code cell; worker deadline40 500с, watchdog40 740с.** Перед финалом runtime резервирует1800с на полную оценку/K8 и600с I/O; periodic full-save каждые900с, два поколения/GPU. +10% покрывает дополнительный checkpoint/log/midpoint I/O и замедление относительно B6, но не гарантирует скорость другой T4. При deadline/unsafe state — finite full checkpoint и partial, N не перенастраивается.

Read-only `KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle quota --csv`20.10 показал14,11ч; [receipt](../checks/b7-quota.json). Консервативно используем заданные14,1ч=50 760с. Production hard40 800 + smoke hard720 оставляют **9240с=2ч34мин**; отдельно предусмотрен резерв600с на platform startup/publication, остаётся2ч24мин. До production ревьюер обновляет quota (guard требует минимум40 920с). Очередь и platform publication вне первой code cell могут изменить время COMPLETE и расход квоты. Содержимое `~/.kaggle-friend` не читается: authentication выполняет CLI/SDK.

## FP16 и финальная оценка

- Midpoint на N/2: W16500 / C32900, **только FP16**, без оценки. Условие B6: initial full-save≤30с и запас до train deadline>120с. В B6 FP16 midpoint стоил0,90/0,55с; если I/O дороже — пропуск с receipt. Это половина нового расписания, не воспроизведение финальной точки B6.
- Финал: FP16, полные heldout4096 иold991 greedy execution + CE с EOS/без PAD/BOS; original target-BPE length bins. Свежий test1919 full-pipeline greedy timer. K8 по fixed tune2048, T0,7/seed270042/batch32/cap512, прежние journals ID/question SHA/code/mean LP/execution. Fresh не оценивается.
- Два retained full cooldown checkpoints/GPU, midpoint/final FP16, журналы и manifest; без ZIP, копий cache и исходных full PT. Полный финал, восстановление и ошибки оценки обрабатывает неизменный B6 runtime.

## Проверки и передача ревьюеру

**CPU/static прошли, handoff `ready:true`; GPU runs/pushes=0.** 157тестов:4новых B7 +12B6 +141исторический; **156passed,1прежний macOS sandbox-skip `killpg`**. Дополнительный CPU-проход4B7 после финализации budget helper прошёл. Проверены compile, nbformat, local/embedded SHA и точный config-only diff, а также B7 reviewer gates без сетевой мутации: отказ production без smoke, отказ stale receipt, single-push и эксклюзивный intent. CPU-проверки и результаты хранятся только в B7 receipts: [регрессии](../checks/b7-regressions.json), [CPU](../checks/b7-cpu.json), [статика](../checks/b7-static.json). Проверяются LR для всех N, ровно N вызовов/нулевой endpoint, AMP skip/cursor, численный disk-resume обеих веток и прежние B6/B5/P1/B4/B3/B2/B1/L2/L3 тесты. Повторные статики сохраняют историю только при совпадении module/config SHA и успешном B7 smoke receipt.

Smoke-бандл: private **`friend1/nlp-hw02-b7-gpu-smoke`**, тот же source port, обе GPU поN200, mini heldout64/old16/CE/test64/K8 на36fixed tune IDs, full model/Adam/scaler/RNG restore и FP16 comparison. Backend720с, worker660с/watchdog690с. Предыдущий аналогичный B6 smoke завершился за360,526с; результат нового B7 smoke пока отсутствует.

Из `work/kaggle` (Python с CPU Torch/nbformat; reviewer API также требует Kaggle SDK):

```sh
PYTHONPATH=src python3 b7_tests.py --regressions
python3 b7_static_check.py
# Только независимый ревьюер:
KAGGLE_CONFIG_DIR=~/.kaggle-friend python3 b7_review_api.py push friend1/nlp-hw02-b7-gpu-smoke --path bundles/b7_gpu_smoke_friend1
# Скачать compact output pinned version; затем:
python3 b7_review_output.py <smoke-output> --smoke --out ../checks/b7-review-gpu-smoke.json
# После review + successful same-source smoke + fresh quota:
KAGGLE_CONFIG_DIR=~/.kaggle-friend python3 b7_review_api.py push friend1/nlp-hw02-b7 --path bundles/b7_friend1
```

[Guarded API](../kaggle/b7_review_api.py) наследует B6 single-push/intent/version/source/config/private/quota gates, использует отдельные B7 targets и receipts. [Output auditor](../kaggle/b7_review_output.py) знает сохранённые B6 имена артефактов, проверяет новый N и40 800с. Прямые CLI-команды и относительные пути бандлов также записаны в handoff. Подготовка ничего не отправляет на Kaggle.

## Источники

- [b6-plan](b6-plan.md), [b6-review](b6-review.md), [b6-results](b6-results.md) и B6v1 producer logs — код, фактический throughput, evaluation contract и ограничения provenance; `kaggle-b6-results` и `kaggle-b5-preparation` уже есть в bibliography.
- **hagele2024-cooldown**: Hägele et al. «Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations»,2024, [§3/Fig.5–6](https://arxiv.org/html/2405.18392v2#S3); запись уже есть в bibliography — hagele2024-cooldown. Формулировка про диапазон cooldown — содержание статьи; ожидаемая польза B7 и бюджет — собственная гипотеза/инженерный расчёт.
- [B7 handoff](../results/next-b7.json), [CPU](../checks/b7-regressions.json), [static/budget](../checks/b7-static.json), [quota](../checks/b7-quota.json) — воспроизводимые локальные проверки; новых GPU-результатов нет.

## Коммит

Попытка `git add` явного списка14B7-файлов получила exit128: создание `<vault>/.git/worktrees/study1/index.lock` запрещено (`Operation not permitted`). Индекс не изменён, commit не создан; ограничение не обходилось. [Квитанция и список](../checks/b7-git.json). Чужие E2 иsubmission файлы не включались.
