---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-14
updated: 2026-10-14
sources:
  - kaggle-l3-final-results
  - kaggle-l2-review
aliases: []
---

# ДЗ2: независимое ревью L4 на аккаунте друга

Ревью `089a70c`, ветка `codex/nlp-hw2-code-generation`; исходное дерево чистое. Файлы финального инференса не изменялись. **GPU-smoke v1 COMPLETE, независимый audit passed**, одна из двух разрешённых итераций. Боевой private L4v1 опубликован; ожидаемый ERROR на Secrets подтверждён status/log. **Вердикт: готов к UI Save & Run All от аккаунта друга с attached friend Secrets**, hard10ч30мин. Полный длинный запуск ревью не выполняло.

## Находки и исправления

| Приоритет / исходное поведение | Исправление и регрессия |
|---|---|
| Существенный пробел в защите от ложного complete: `src/l3_resume.py:83` принимал финальные metrics без K8 и доверял одному флагу complete; пустой K8 dictionary вызывал KeyError после сохранения full state. | `src/l3_resume.py:75–99`: боевой L4 требует final K8 complete=true, rows=target_rows=2048; отсутствие/неполнота дают partial в summary, без потери stable state. Проверка `.get('complete',False)` устраняет KeyError. `l4_tests.py:44–75`: отсутствующий/пустой K8, ложный complete, неверный target и успешный полный receipt; настоящий tiny runtime делает две train→cooldown→eval→restore фазы, получает metrics без K8, сохраняет full state и возвращает partial. |

L2 **34/34**, L3 **10/10**, L4 **5/5**; nbformat/compile/source parity/diff-check прошли. Численная математика, frozen L2 runtime/resume/sampler/eval/live/session не менялись. Новое требование 2048 действует только в production L4; короткий CUDA smoke проверяется по своему target_rows. `l3_resume.py` не входит в inherited semantic/runtime SHA. Исторические L3 notebooks сохраняют прежний embedded snapshot; новые L4 notebooks включают исправленную обёртку.

## Продолжение и ресурсы: проверка кода

- `build_l3.py:69–84`, `src/l3_resume.py:29–54`: до install/cache сверяются latest↔recover-manifest↔pin, bytes/SHA, identity/spec/counters, LR0,0015, scaler/CUDA RNG, finite model/Adam, runtime/semantic SHA и Torch2.11.0+cu128/CUDA12.8. L4 требует пять complete tags10800/21600/28800/45540/62280 и complete inherited final K8. Стартовые full states **L3 latest**, не cooled FP16 и не L2 latest.
- `l2_worker.py:22–41`, `src/l2_runtime.py:102–123`: C-main213386updates/cursor27313408, C-long2x209144/cursor20734592, собственные identities/order/history. `L.restore_full` восстанавливает FP32 model, Adam, GradScaler и RNG. Постоянный LR передаётся в safe update; warmup не повторяется. `src/l3_resume.py:70–73` прибавляет15300/30600с к точному inherited l2_main_seconds: новые tags77580/92880.
- `build_l3.py:99–102`, `src/l3_resume.py:56–68`, `src/l2_sampler.py:74–84`: original order SHA09cf6db8… и inherited weighted SHA72bc3dfd…; weighted order копируется, не строится. Metadata строго совпадает с inherited sampler, uint32/slots11227811/границы проверяются; cursor не сбрасывается.
- `src/l2_runtime.py:184–241`: full uncooled stable перед cooldown; eval и исключения проходят restore full state; новый checkpoint после записи истории. `:73–85`: атомарные PT→latest→ротация, два full PT на модель. NaN/AMP/post-Adam corruption имеют ограниченную эскалацию; ошибка одной ветки не останавливает вторую (`src/l2_session.py:32–50`).
- `build_l3.py:36–44,49–66`: watchdog вооружён до Torch/install. Production worker37380с, hard37800с (**10ч30мин**) от первой code cell, backend43200с. Main30600с=8ч30мин; cooldown900/3000с, eval300/1800с, setup480с, I/O180/stop120с. Final sync получает остаток до hard−120с, номинально300с; force-exit защищает packaging. Short smoke worker900/hard1020/backend1080с. [Сам объект watchdog](../checks/kaggle-l4-review-watchdog.json) создан из обоих configs: старый3ч40мин ceiling действительно переопределён.
- `src/l2_session.py:8–17,57–58,84–88`: working guard10GiB/free512MiB, race ротации допустим, interrupted PT temporaries удаляются после reap. Обычный максимум checkpoints: четыре full≈2,58GB + два concurrent transient≈1,29GB + четыре cooled≈0,43GB; cache остаётся read-only input. Live stage≤2,5GB и CLI ZIP вне working, но занимают общий диск. `src/l2_live.py:35–91` проверяет SHA latest↔snapshot и не публикует ложный recovery receipt при race.

## Аккаунт, входы, fallback, Secrets

Все Kaggle CLI/SDK команды этого ревью: `export KAGGLE_CONFIG_DIR=~/.kaggle-friend`, CLI2.2.4; credential directories и значения не читались. Private friend l3-live v12 доступен через mine/list-files. Сервер smokev1 подтвердил private/T4/internet и все четыре sources ([metadata](../checks/kaggle-l4-review-smoke-metadata.json)). `/2` зафиксирован в upload metadata; server pull отображает kernel source без version, поэтому content SHA обязателен.

Источники: thisarivson/26b-inputs, friend1/l3-live; kernel thisarivson/compact-cache-verified/2. Smoke также подключает fallback friend1/compact-cache-verified. **V1 реально подключил все входы**: оба datasets в `/kaggle/input/datasets/<owner>/…`, notebook-cache в `/kaggle/input/notebooks/thisarivson/…`. Обе full anchors проверены за19,30с. `build_l3.py:93–125` проверил manifest SHA46ef881b…/9725737rows/60shards и SHA каждого из360payloads **обоих кешей**, ordinary order и weighted order. Fallback tokenizer SHA также exact, roundtrip50040/0 mismatches. Это runtime audit в ноутбуке аккаунт друга, а не вывод из remote names/sizes.

Если original notebook output недоступен, остановиться до training и явно пересобрать:

```bash
export KAGGLE_CONFIG_DIR=~/.kaggle-friend
python build_l4.py --cache-source friend1/nlp-hw02-compact-cache-verified
python build_l4.py --gpu-smoke --cache-source friend1/nlp-hw02-compact-cache-verified
```

Fallback не меняет manifest/order/anchors; metadata заменяет kernel source на friend dataset. Нельзя отключать SHA или обходить403.

`build_l3.py:64–68`, `src/l2_live.py:10–33`: production требует attached KAGGLE_USERNAME/KAGGLE_KEY аккаунт друга, smoke Secrets/live не требует. Значения только в env; worker env очищается от API/Secrets/IAP credentials, helper использует attached key и empty config. Startup owner-private upload+ready обязателен; heartbeat5400с, retry≥300с, stale insurance7200с останавливает workers для сохранения. Назначение **friend1/nlp-hw02-l4-live**, источник **l3-live**. В одном пункте задания указан sync в l3-live; сохранён явный L4 target из начала задания и [l4-plan](l4-plan.md), чтобы не перезаписывать источник L3.

## GPU-smoke и боевой push

[Private smokev1](https://www.kaggle.com/code/friend1/nlp-hw02-l4-gpu-smoke/versions?versionNumber=1) **COMPLETE**, wall612,695с=10мин12,7с. [Независимый audit](../checks/kaggle-l4-review-gpu-smoke-v1.json) прошёл. Одна итерация, второй push не нужен. Из93 малых outputs скачаны JSON/JSONL/logs,90 файлов producer manifest проверены по bytes/SHA; full PT/order не скачивались. Два настоящих L3 states в формате L2 full-v1 загружены на CUDA; это продолжение L3, не старт из L2.

| Проверка | C-main | C-long2x |
|---|---:|---:|
| Main attempted updates |808|671 +64 после crash соседа|
| Train→cooldown→eval→restore→train |две20-с cooldown фазы|две20-с cooldown фазы|
| K8 rows/samples |36/288|36/288|
| K8 full pipeline, с |2,708|3,036|
| Greedy256 full pipeline, с |2,601/2,710|7,912/8,226|
| Train/eval allocated peak, GiB |7,07|6,10|
| Train/eval reserved peak, GiB |10,00|11,06|
| Longest-row successful probe reserved, GiB |10,83|10,83|
| Максимальный наблюдённый reserved, GiB |10,83|11,06|

Finite reference update и serialized restore→next CUDA update bit-exact: настоящий FP16/fusedAdamW/GradScaler, effective128/micro64. Longest-row128/96 дали ожидаемый OOM,64 прошёл. Found-inf backoff сохранил model/Adam; NaN после partial backward сохранил numerical state и очистил gradients. Три injected NaN вызвали full rollback и LR0,00075 только у main; production LR остаётся0,0015. Five inherited points побайтно на уровне JSON равны исходной L3 history; own identity/cursor сохранены, weighted order SHA прежний.

Worker0 intentionally exited1 после всех primary checks; healthy worker1 продолжил ещё64 updates, общий stop-request отсутствует, exits[1,0], packaging passed. Stable updates/cursors main214194/27416832, long209879/20828672. K8 journals имеют36unique IDs, все288samples каждой модели исполнены с конечным mean_logprob; обе новые точки complete и full greedy/test256 сохранены. Это проверка continuation/faults/outputs, не оценка качества L4.

[Боевой private L4v1](https://www.kaggle.com/code/friend1/nlp-hw02-kaggle-l4/versions?versionNumber=1) отправлен через API от аккаунта друга, timeout43200/T4×2. Source port SHA21c9354a… совпадает с прошедшим smoke; production timings/gates восстановлены. Server private/owner/sources подтверждены [metadata](../checks/kaggle-l4-review-production-metadata.json).

Status **ERROR — ожидаемый Secrets gate**. [Gate receipt](../checks/kaggle-l4-review-production-gate.json), `startup-gate.json` и скачанный log: `Kaggle Secrets unavailable: KAGGLE_USERNAME, KAGGLE_KEY`. Нет anchor-preflight, worker logs и model directories: anchor load/worker spawn/train не начинались. Пользователь: в аккаунте друга прикрепить оба friend Secrets, Edit→Save Version→Save & Run All. Startup ещё раз проверит inputs/anchors и private live upload+ready перед первым update.

Сохранено≈8MiB JSON/JSONL/logs smoke и малый production gate output в ignored `results/kaggle-runs`. Большие PT/order и временный кеш не скачивались; собственные temporary test logs удалены. На Kaggle ничего не удалялось и не публиковалось публично. Credential values/directories и основной checkout `<vault>` не инспектировались.

## Некритичные риски и пределы проверки

- Короткий smoke не доказывает отсутствие редкого отказа за10,5ч. Внезапный kill может потерять updates после последнего30-мин stable checkpoint; source L3 и live recovery остаются доступными.
- Guard512MiB меньше одного full PT≈644MB; checkpoint I/O при внезапном исчерпании общего диска может упасть. Атомарный latest сохраняет предыдущий state; рассчитанный обычный объём checkpoints≈4,3GB. Stage и ZIP требуют дополнительного свободного места.
- Setup/eval/upload caps — пределы, не гарантия скорости. При плохой сети/throughput — partial со stable state, а не обещание всех targets. Friend quota≈13ч задана пользователем; hard10,5ч + один18-мин ceiling оставляет≈2,2ч без учёта прочих параллельных сессий.
- Torch/CUDA/source pins должны отказать при изменении image/latest; не обновлять identities автоматически. Repeated incomplete tags не переоцениваются; для следующей сессии нужны новые targets.
- Attached Secrets/private upload/ready/insurance будут реально проверены обязательным startup UI run; GPU smoke не использует Secrets. Backend COMPLETE с injected exits[1,0] сам по себе не доказательство успеха — обязательны cuda-smoke-summary и audit всех фаз.

## Источники

- [l4-plan](l4-plan.md), [l3-plan](l3-plan.md), [l3-review](l3-review.md), [l3-results — Финал L3](l3-results.md#финал-l3), [l2-review](l2-review.md), [l2-results](l2-results.md); соответствующие работы уже в bibliography.
- [L4 pins](../kaggle/l4-anchors.json), [static receipt](../checks/kaggle-l4-static.json), [L4 regressions](../checks/kaggle-l4-regressions.json), [runs.json](../kaggle/runs.json).
- Независимые инструменты [l4_review_api.py](../kaggle/l4_review_api.py), [l4_review_output.py](../kaggle/l4_review_output.py); выводы этой заметки — собственная проверка кода/outputs, не гарантия11-часовой безотказности.

## Коммит

Попытка `git add` явного L4-review списка заблокирована sandbox: `Unable to create '<vault>/.git/worktrees/study1/index.lock': Operation not permitted`. Индекс не изменён, ограничения не обходились; изменения и ревью сохранены. Повтор из обычного терминала в этом worktree (перед staging проверить дифф общих файлов):

```bash
l4_work=work
l4_paths=(
  README.md
  courses/nlp/index.md courses/nlp/progress.md
  "$l4_work/kaggle/README.md" "$l4_work/kaggle/runs.json"
  "$l4_work/kaggle/src/l3_resume.py" "$l4_work/kaggle/l4_tests.py"
  "$l4_work/kaggle/l4_review_api.py" "$l4_work/kaggle/l4_review_output.py"
  "$l4_work/kaggle/notebooks/nlp-hw02-kaggle-l4.ipynb"
  "$l4_work/kaggle/notebooks/nlp-hw02-l4-gpu-smoke.ipynb"
  "$l4_work/notes/l4-review.md"
  "$l4_work/checks/kaggle-l4-regressions.json"
  "$l4_work/checks/kaggle-l4-static.json"
  "$l4_work/checks"/kaggle-l4-review-*.json
)
git diff -- "${l4_paths[@]}"
git add -- "${l4_paths[@]}"
git commit --only -m 'nlp: review L4 continuation and verify friend CUDA smoke' -- "${l4_paths[@]}"
```
