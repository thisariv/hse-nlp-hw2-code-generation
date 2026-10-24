---
type: experiment
course:
  - nlp
status: learning
understanding: 4
created: 2026-10-19
updated: 2026-10-19
sources:
  - kaggle-b5-results
  - kaggle-p1-final-results
  - kaggle-b5-preparation
aliases: []
---

# B6: длинный cooldown W600 и C-long2x

**CPU/API/static завершены, [handoff](../results/next-b6.json) `ready:true`; independent GPU review/smoke pending.** 10 B6 tests, 140passed исторических тестов и1 sandbox-skip; реальные W/C по2 batch128 с production disk resume побитно совпали с uninterrupted q0 control (139,37/54,40с). Проверены SHA всех четырёх сохранённых full PT, обеих orders и полного cache; временные PT/orders удалены. Compile/nbformat/local–embedded parity/frozen SHA/metadata обоих бандлов прошли. GPU/push в подготовке —0.

**Только q0 cooldown из uncooled end-main B5v6, две независимые ветки T4×2.** Private `friend1/nlp-hw02-b6`, title `nlp-hw02-b6`, internet on. Пользователь уже разрешил production после независимого ревью и успешного same-source smoke. Оба GPU-запуска выполняет ревьюер; подготовка не делает push и не запускает GPU.

Мотивация пользователя: v8 public78,54%, v9 с B5-моделями78,44%; свежий GPU-tune у обеих1532/2048. [b5-results](b5-results.md) показывает плато main. B6 проверяет более длинный cooldown при сохранённом v8 fallback. P1-B дала +0,71п.п. greedy4k к L3, но включала truncation; чистая q0 P1-A дала +0,27п.п., CI включает0. Эффект B6 заранее не установлен. B6 меняет одновременно длину и форму cooldown относительно final50 B5; main/truncation/архитектура не меняются.

## Источники и продолжение

Все исходники/метаданные закреплены в [b6-anchors.json](../kaggle/b6-anchors.json). Mount напрямую из `friend1/nlp-hw02-b5/6`; archived manifest SHA `f662a966b5031a93bb46d25530a2710753a26f69e5262af80eca05ac6c6aa525`. Отдельный transfer dataset не нужен. Остальные mounts: verified compact cachev2 и прежний starter dataset.

| Поле | GPU0 W600 | GPU1 C-long2x |
|---|---|---|
| Full stable |`model-GPU0/checkpoints/main-789d436a3fb0.pt`|`model-GPU1/checkpoints/main-6b2c636e5f7f.pt`|
| SHA256 |`3b11360d5242e0e8b3fc1ac38254f0d4d544fba8bb62ee4c7bb034943e6cdc32`|`69dff4405d2236097f48733adbfc9e3bfd5d20b501a3fcfd1dae62ec2ce63074`|
| Main_s |125641,96956177584|113690,72837862006|
| Main updates /cursor |154559 /19783552|293688 /31556224|
| Stable LR |0,0006|0,0015|
| Weighted order SHA |`8e9595f655ed2b4456bf5b94e9b5b6b3db040e3f5696a4dc0ef364588048bee8`|`72bc3dfd3a3e738979c489c0cf8f41cde092c41d4ae90f2de8378a034603a4db`|
| B5 final |`b5-final-125641`|`b5-final-113690`|
| Его pre-cooldown anchor SHA |`894413495ceba5cd840a6ed73f3d41bb4896ccb5925134b6c5752672b86d9695`|`cd26644d4961c953410504510d8f5a7eaddbda8b31da51ad8cf14b7841096cad`|

**Ограничение provenance:** B5 удалила оба pre-final PT при ротации двух stable-поколений; самих этих файлов в final manifest нет. Их полные anchor receipts точно совпадают по updates/cursor/main_s/identity/model_spec с выбранными latest. Полные bytes/SHA latest и сохранённого предшественника проверяются отдельно; model/Adam/scaler/все RNG/scheduler и весь main state между двумя post-restore поколениями сравниваются побитно. Отличается только добавленная B5 session history. Это проверка сохранённых состояний после штатного restore B5, а не прямое tensor-сравнение с удалённым pre-final PT. Попытка обратной сериализации отчётных полей не воспроизвела его SHA; такой результат не используется как доказательство.

Восстанавливаются FP32 model, полные AdamW moments/steps/param groups, enabled scaler, Python/NumPy/Torch CPU/CUDA RNG, исторический main state и собственный weighted order/cursor. У C сохраняется унаследованный cursor offset. W использует frozen QK/z-loss path; C — прежний `legacy-c` core/decoder через отдельный `src-C`. Все implementation SHA и Torch2.11.0+cu128/CUDA12.8 проверяются до training. Нового warmup нет.

## Расписание и бюджет

$$\eta_j=\eta_0\left(1-\sqrt{j/N}\right),\qquad j=1,\ldots,N.$$

$\eta_0$ — stable LR ветки; $j$ — номер attempted cooldown update; $N$ — фиксированный предел. Используется та же `p1_math.sqrt_lr`, что в P1. Первый update чуть ниже stable LR, последний имеет **LR0**; дополнительного update нет. AMP/nonfinite skips расходуют один слот и128пар, учитываются отдельно. $q=0$: все128пар участвуют в обычном loss, truncation и дополнительного forward для отбора нет.

Скорость пересчитана по всем42626/84544 main и3764/7481 final50 update-логам B5. Берётся большее время среднего update, затем $N=100\lfloor13200/(100s)\rfloor$, где $s$ — это время в секундах. N закреплён в бандле, GPU-smoke его не перенастраивает; при новой скорости меняется только статус complete/partial, не расписание.

| Расчёт | W | C-long2x |
|---|---:|---:|
| Main updates/с |1,259626|2,498323|
| Final50 updates/с, опорные |1,254551|2,493328|
| **N** |**16500**|**32900**|
| Cooldown-предъявления |2112000|4211200|
| Pure update time, с |13152,12|13195,22|
| Update time×1,10 +480setup +1800eval/K8 +600I/O, с |17347,33|17394,74|
| Прогноз всей сессии |4ч49мин7с|4ч49мин55с|

Hard backend ceiling **19200с =5ч20мин от первой code cell**; worker deadline18900, watchdog19140. Cooldown останавливается до `worker deadline −1800eval −600I/O`; оставшиеся оценки/публикация укладываются в общий предел. N остаётся прежним даже при досрочном останове. ≥10% — инженерный запас скорости, не гарантия другой T4. Платформенный startup/publication вне первого code cell требует отдельного квотного остатка.

Квота проверяется через `KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle quota`: исходно **18,96ч**, reset24.10.2026 00:00UTC. Один production≤5ч20мин и один smoke≤12мин оставляют около **13ч26мин**, до platform overhead. Точная последняя квитанция и бюджет — [next-b6.json](../results/next-b6.json). Содержимое `~/.kaggle*` не читается; CLI/SDK самостоятельно выполняют authentication. Secrets, upload/live-sync и API-credentials в workers не нужны.

## Checkpoints, оценка и частичный результат

- Начальный и периодические full cooldown checkpoints каждые900с; два поколения на GPU, atomic latest. Это новый формат `nlp-hw02-b6-full-v1`, не uncooled main. Исходный main state остаётся неизменным; cooldown имеет собственные cursor, attempted updates, LR, scheduler, counters и stability. Fixed N хранится вместе с checkpoint.
- Один midpoint: только model-only FP16 без оценки. Admission — initial full-save≤30с и запас≥120с; более короткая FP16-запись ожидается≤60с, её фактическое время сохраняется. При дорогом I/O точка пропускается. Midpoint не преобразует training model и не расходует RNG.
- Финал после ровно N: FP16, полная greedy heldout4096 иold991, CE с EOS/без PAD/BOS, original BPE length bins, свежий test1919 greedy full-pipeline timer. K8 — fixed tune2048, seed270042/T0,7/batch32/cap512; B4/B5 journals с ID/question SHA, represented code, mean LP и execution. Fresh не читается.
- Unsafe Adam OOM/nonfinite state: восстановить последний finite full checkpoint и закончить partial без LR-halving и продолжения main. Здоровый сосед заканчивает свою ветку. Неполная оценка сохраняет готовый FP16/checkpoints и даёт partial; финальный assert не превращает её в notebook ERROR.
- Output без ZIP, dataset-копий, cache и full исходных PT. Два поколения full/GPU, midpoint/final FP16 и журналы: ориентир≈4,8GB плюс небольшие logs. Supervisor/watchdog унаследованы от B5, Secrets-free worker env и независимое завершение веток сохранены.

## CPU и передача ревьюеру

В [b6-cpu.json](../checks/b6-cpu.json) — проверки полного LR-расписания/endpoint, ровно N, обоих math paths, cursor/потока, serializer/RNG, AMP skip, deadline/stop, unsafe rollback, partial evaluation и того же smoke-контроллера по200updates каждой ветки. [b6-regressions.json](../checks/b6-regressions.json): 141 исторический тест, 140passed и1 прежний macOS sandbox-skip `killpg`; B5 и17P1 прошли.

Реальный CPU replay скачивает stable обеих веток и сохранённые predecessor states по SHA, проверяет полный cache/order и SHA всего planned потока. Два реальных batch128: uninterrupted P1-style q0 control против B6 adapter с production full-save/read между шагами. CPU FP32 без autocast, fused выключается после точного восстановления optimizer; CUDA RNG проверяется как opaque bytes. Actual CUDA execution остаётся задачей смоука. Все временные PT/orders удаляются; файлы весов в Git не добавляются.

Готовность выставляет только [b6_static_check.py](../kaggle/b6_static_check.py) после CPU/input/static gates. Handoff отдельно сохраняет `GPU_smoke_pending`; повторная статика не стирает историю ревью при тех же module/config SHA. GPU smoke — тот же production runtime с **N200 на каждой GPU**, mini greedy64/old16/CE/test64/K8 на36 frozen tune IDs, full restore и FP16 comparison, hard720с. Это функциональный смоук, не полный timer/quality gate.

[b6_review_api.py](../kaggle/b6_review_api.py) предоставляет account-scoped quota/status/metadata/compact output и guarded push: свежая статика, quota, private/T4/timeout, совпадение source/config и successful smoke перед production. Неизвестная server version и повторный production push запрещены; используется проверенный guard B5. Альтернативные `guarded_*_command` сохранены в handoff. [Output auditor](../kaggle/b6_review_output.py) проверяет producer SHA, source/config, ровно N, все LR и полноту K8; PT/orders для него не скачиваются.

> [!info]- Воспроизведение и команды независимого ревьюера
> Из `work/kaggle`, CPU:
> ```sh
> python b6_prepare.py --cpu-python <torch-CPU-python> --cache <verified-compact-cache>
> python b6_tests.py
> python b6_regressions.py
> python b6_static_check.py
> ```
> `b6_prepare.py` запускать Python с установленным Kaggle SDK. GPU — только ревьюер, после diff и проверки текущих server versions:
> ```sh
> KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle quota
> KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle kernels push -p bundles/b6_gpu_smoke_friend1 --accelerator NvidiaTeslaT4 --timeout 720
> # Скачать небольшие outputs; b6_review_output.py <folder> --smoke --out ../checks/b6-review-gpu-smoke.json
> # После successful same-source smoke, независимого ревью и повторной quota:
> KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle kernels push -p bundles/b6_friend1 --accelerator NvidiaTeslaT4 --timeout 19200
> ```

## Коммит

Попытка `git add` только29 B6-файлов получила exit128: `Unable to create .../index.lock: Operation not permitted`. Индекс не изменён, commit не создан; ограничение sandbox не обходилось. [Квитанция и точный список файлов](../checks/b6-git.json). Чужие незавершённые изменения не включались; `progress.md`, index, dashboard и backlog не редактировались.

## Источники

- [b5-results — B6 и последняя квота](b5-results.md#b6-и-последняя-квота), [b5-plan](b5-plan.md), [b5-review](b5-review.md) — uncooled sources, actual update logs, frozen W/C math и evaluation contract; `kaggle-b5-results`/`kaggle-b5-preparation` уже внесены в bibliography.
- [p1-review](p1-review.md), [p1-results](p1-results.md) — attempted-update $1-\sqrt{j/N}$, q0 control, restore всех RNG и ограничения эффекта cooldown; `kaggle-p1-final-results` уже в bibliography.
- Собственные проверки B6: [pins](../kaggle/b6-anchors.json), [CPU](../checks/b6-cpu.json), [регрессии](../checks/b6-regressions.json), [статика](../checks/b6-static.json), [handoff](../results/next-b6.json). Расчёт бюджета и гипотеза улучшения — собственные выводы, не результаты GPU-эксперимента.
