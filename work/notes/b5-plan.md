---
type: experiment
course:
  - nlp
status: learning
understanding: 4
created: 2026-10-18
updated: 2026-10-18
sources:
  - kaggle-b5-preparation
  - kaggle-b4-preparation
  - kaggle-l3-final-results
  - kaggle-p1-independent-review
aliases: []
---

# B5: независимое продолжение W600 и C-long2x

Итог независимого ревью18.10 — [b5-review](b5-review.md): same-source smoke v2 passed, обе200 main updates/restore/K8; активный private B5v3 RUNNING с13:13UTC, ETA19.10≈03:33МСК. V3 появилась из параллельной сессии; полная notebook parity проверена, собственный API v1 был отменён до наблюдаемого GPU-расхода. Точные квитанции и ограничения — в ревью и [handoff](../results/next-b5.json).

**Протокол выбран пользователем; боевой запуск уже разрешён после независимого ревью и успешного смоука.** Рекомендация отложить B5 в [b4-results](b4-results.md) остаётся историей анализа. Здесь готовится продолжение двух разных состояний; стартовые веса, RNG и weighted orders между GPU не уравниваются. Гипотеза — ещё около9,4ч main улучшит обе ветки и оставит anchors для длинного $1-\sqrt{\cdot}$ cooldown следующей сессии. Качество B5 пока не измерено.

**CPU/API/static завершены; B5 готова к независимому ревью.** [Handoff](../results/next-b5.json): `ready:true`, `GPU_smoke_pending:true`. [10 B5 проверок](../checks/b5-cpu.json), [128 успешных исторических тестов и1 sandbox-skip](../checks/b5-regressions.json), реальные [W](../checks/b5-real-resume-gpu0.json)/[C](../checks/b5-real-resume-gpu1.json) по2 batch128 с production full-save/read, [статика обоих бандлов](../checks/b5-static.json) прошли. [Private transfer v1](../checks/b5-transfer.json) проверен по всем remote bytes/SHA; собственные временные PT/order/staging удалены.

Private kernel `friend1/nlp-hw02-b5`, title `nlp-hw02-b5`, T4×2, internet on. CLI: `KAGGLE_CONFIG_DIR=~/.kaggle-friend`. Содержимое credential directories не инспектируется; аутентификация только через CLI/SDK. Secrets и live-sync не требуются. **GPU/push в подготовке —0**, оба GPU-запуска выполняет независимый ревьюер.

## Строгие исходные состояния

| Поле | GPU0 W600 | GPU1 C-long2x |
|---|---|---|
| Источник | `thisarivson/nlp-hw02-b4/1`, plain GPU0 | `friend1/nlp-hw02-l3-live/12`, L3 `l2-62280` |
| Full stable | `model-GPU0/checkpoints/main-9bdf4998eb30.pt` | `model-C-long2x__checkpoints__main-2a27307d7a48.pt` |
| SHA256 | `f23fe1e36135e23b1ba1d393537ea501f25165a084b8462c16d8bc24843abc7a` | `48f2ba3fa4ffe6a722b45a084d3ee33a27cc4ed02b72433605c9cb1740f5a80b` |
| Updates |111933|209144|
| Всего main-предъявлений |14327424|26770432|
| Order cursor |14327424|20734592|
| Main_s |91801,75519380573|79850,42887455746|
| LR main |0,0006 constant|0,0015 constant|
| Архитектура |6+6/d768, QK LayerNorm, z-loss1e-4|исходная C6+6/d512, обычная CE|
| Weighted order SHA |`8e9595f655ed2b4456bf5b94e9b5b6b3db040e3f5696a4dc0ef364588048bee8`|`72bc3dfd3a3e738979c489c0cf8f41cde092c41d4ae90f2de8378a034603a4db`|

C — **тот же uncooled anchor, из которого начиналась P1-B**, до50-минутного финального cooldown L3. P1-B FP16 не используется. Разница между total pairs и order cursor у C унаследована от перехода L1→L2: порядок тогда был заменён и курсор сброшен; сейчас ничего не сбрасывается. `l2_main_seconds=62280,4929961593` сохраняется и продолжает увеличиваться.

[Полные pins](../kaggle/b5-anchors.json). FP32 model, AdamW moments/step/param groups, enabled GradScaler, Python/NumPy/Torch CPU/CUDA RNG, scheduler, stability/history и cursor восстанавливаются полностью. Сохраняются оба исторических формата; frozen B2 и L3 training math не меняется. У C отдельный `src-C`: точные `exp26_core.py` и `exp27b_fast.py` из L3 сохранены в `kaggle/legacy-c/`, остальные модули имеют те же проверенные bytes. После L3 общий core был изменён для W/QK; его нельзя подставлять C или ослаблять проверку SHA. Боевые gates требуют Torch2.11.0+cu128/CUDA12.8 и все SHA из обеих исходных identities. Нет нового warmup, truncation, фильтра строк или перестроения order. Long2x означает multiplicity1/2 при compact target length≥79, а не loss reweighting.

Серьёзная нестабильность или unsafe Adam OOM: возврат последнего full stable и partial stop при прежнем LR. Здоровая соседняя GPU продолжает. Это сохраняет заданный constant-LR протокол; автоматического LR-halving нет.

## Переносы и доступ

Новый private dataset — `friend1/nlp-hw02-b5-inputs`: **только W full stable, его latest и два файла weighted order**. Источник привязан к архивному B4v1 manifest SHA, exact latest и заданному SHA полного stable. Kaggle CLI latest endpoint не фиксирует версию; такое совпадение содержимого обязательно, другой latest не принимается.

L3 live v12, расшаренный `thisarivson/nlp-hw02-compact-cache-verified/2` и `thisarivson/nlp-hw02-26b-inputs` монтируются напрямую. Доступ друга проверяется реальным скачиванием через CLI; manifest кеша должен иметь SHA `46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be`. Source W переносится с аккаунта владельца, загружается на друга private и повторно скачивается для проверки bytes/SHA. Kaggle-ресурсы не удаляются.

Runtime использует symlinks к mounted states, без локальной копии full PT. Оба порядка сохраняются отдельно в output: `weighted-order-GPU0/` и `weighted-order-GPU1/`. Все собственные временные локальные PT/order/download/staging удаляются после проверки. Маленькие квитанции остаются в `checks/b5-*.json`.

## Расписание и бюджет

Квота проверяется через `kaggle quota`: **30,00ч**, reset **24.10.2026 00:00 UTC**. Параллельные T4 используют одно расписание с разными абсолютными main_s.

| Фаза | Cap |
|---|---:|
| Setup: gates/install/полный cache SHA/обе пары state+order |480с|
| Main, сумма времени updates |33840с =9ч24мин|
| **Одна** проба на +16200с main |900с linear +до300с greedy4k/old991/CE|
| Финал |3000с linear +до1800с полная оценка|
| Периодический stable |каждые900с и перед/после каждой ветки оценки|
| Worker deadline / watchdog / hard/backend timeout |40500 /40740 /40800с|

Hard **11ч20мин от первой code cell**. Main cap приблизительный: перед финалом всегда остаются5100с на cooldown/eval/I/O/stop; затраты setup/checkpoint/restore могут сократить main. Номинальные setup+compute+eval caps40320с; полный набор максимальных caps с I/O может потребовать уменьшить main. Пропуск пробы, недостигнутый target или неполная оценка дают partial, сохраняют stable/output и не превращают ноутбук в ERROR финальным assert.

Cooldown обеих веток disposable, timed linear от собственного LR до конца900/3000с update time. Последний LR записывается фактически; дополнительного искусственного нулевого update нет. После оценки возвращаются полный uncooled state/RNG/cursor. Исторические `points`, `sessions`, `l2_points` остаются; новая история хранится в `b5_points`/`b5_sessions`.

Финал: greedy heldout4096 +old991, unsmoothed token CE с EOS без PAD/BOS, test1919 pipeline timer, **K8 tune2048** с прежними seed270042/T0,7/batch32/cap512, ID/question SHA, represented code, untempered mean logprob и execution journals. Fresh не читается. Охлаждённые FP16 сохраняются отдельно. **Full uncooled stable обеих GPU остаются в `latest.json`**, summary содержит `B6_uncooled_latest` для следующего длинного cooldown.

Прогноз по измеренным B4 plain/L3 long скоростям: W **≈5,15M новых пар**, total≈19,48M; C **≈11,04M**, total≈37,81M. Это расчёт, не гарантия скорости новой T4. Main-пары не включают disposable cooldown. Без live-sync сбой всей VM может потерять ещё не опубликованный working output. Output ожидается≈4,8GB, пик атомарных checkpoint≈6,8GB; сохраняются два full stable поколения/GPU, два FP16/GPU и компактные журналы, ZIP не создаётся.

Production11ч20мин +один smoke≤12мин оставляют от30ч **≈18ч28мин** до reset. Это бюджет session wall; platform startup/publication overhead в остаток не включён. Перед production ревьюер повторяет quota и проверяет свободно≥42000с.

## Проверки и передача ревьюеру

Подготовлены production и smoke с одинаковыми module SHA. Smoke hard720с, workers660/watchdog690: обе реальные ветки по200 main updates, sampled throughput/память на9 batches включая самый длинный aligned batch, probe с полным restore, mini-final/K8, сохранение stable/FP16 и injected AMP overflow. Сам смоук GPU здесь не запускается; короткие decode caps не являются измерением полного качества/test1919.

CPU: каждый исходный stable скачивается отдельно, SHA проверяется; два следующих реальных batch128 — historical uninterrupted control против B5 adapter со штатными `save_full`/`read_full` и полным restore между шагами. Проверяются exact model/Adam/scaler/state/CPU RNG, CUDA RNG bytes, cursor/поток/constant LR. CPU forward — FP32 без autocast, fused отключается **после** точной проверки загруженного optimizer; При CPU full-save исходные CUDA RNG bytes сохраняются как непрозрачный payload и проверяются после чтения; CUDA/fused execution остаётся GPU-smoke.

Малые runtime-тесты обеих веток проверяют probe/final restore, историю, периодические checkpoints, FP16, поток, reserve/partial и stop без смены LR. Полный контроллер смоука прогнан на CPU с200 шагами каждой ветки; evaluation contract подставлен fixture. Frozen B4/B3/L3 и дополнительные B2/B1/P1/L2 регрессии проверяются отдельно. Один дополнительный L2 тест process-group cleanup пропущен: macOS sandbox запрещает `killpg`; производственный Linux supervisor не менялся.

Статика проверяет compile/nbformat, локальный/embedded source parity, frozen implementation SHA обеих identities, private/owner/title/mounts, deadlines/watchdog до Torch/install, отсутствие Secrets и мягкую публикацию partial. Машинная готовность и точные результаты — [next-b5.json](../results/next-b5.json), `ready:true` выставляется **только после успешных обязательных CPU/API/static проверок**. Это готовность к ревью; `GPU_smoke_pending:true` остаётся до независимого запуска.

> [!info]- Воспроизведение и команды ревьюера
> Из `work/kaggle`, без GPU:
> ```sh
> python b5_prepare.py --cpu-python <CPU-python> --cache <verified-compact-cache>
> python b5_tests.py
> python b5_regressions.py
> python b5_static_check.py
> ```
> Только ревьюер, после проверки diff:
> ```sh
> KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle kernels push -p bundles/b5_gpu_smoke_friend1 --accelerator NvidiaTeslaT4 --timeout 720
> # После успешного same-source smoke и повторной quota:
> KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle kernels push -p bundles/b5_friend1 --accelerator NvidiaTeslaT4 --timeout 40800
> ```

## Коммит

Попытка `git add` только B5-файлов и собственной записи библиографии завершилась128: `Unable to create .../index.lock: Operation not permitted`. Индекс не изменён, commit не создан; [точная квитанция и список файлов](../checks/b5-git.json). Ограничение песочницы не обходилось.

## Источники

- **kaggle-b5-preparation** — протокол пользователя и собственные CPU/CLI/static квитанции18.10.2026; [код и pins](../kaggle/b5-anchors.json), [handoff](../results/next-b5.json), bibliography — kaggle-b5-preparation. Скорость/пары/объём output — собственные прогнозы.
- [b4-plan](b4-plan.md), [b4-review](b4-review.md), [b4-results](b4-results.md) — W600 stable, frozen math, K8 format, throughput и уроки восстановления.
- [l3-plan](l3-plan.md), [l3-results](l3-results.md), [p1-review](p1-review.md) — C-long2x anchor/order/история и подтверждение происхождения P1-B.
- [b1_transfer.py](../kaggle/b1_transfer.py), [l3_resume.py](../kaggle/src/l3_resume.py) — private transfer/remote SHA и mounted live verification. Новых внешних научных источников нет.
