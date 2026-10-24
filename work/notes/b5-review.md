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
  - kaggle-b4-independent-review
  - kaggle-b3-independent-review
aliases: []
---

# B5: независимое ревью и запуск

Проверен commit `a33319b`, ветка `codex/nlp-hw2-code-generation`. Исходный handoff `ready:true`. Прочитаны [b4-review](b4-review.md), [b3-review](b3-review.md), [l3-plan](l3-plan.md), [b5-plan](b5-plan.md), исходники и квитанции подготовки. В начале B5-файлы были untracked; отдельная незавершённая правка `knowledge/bibliography.md` оставлена без изменения. `progress.md`, индексы, dashboard и backlog не редактировались. Пользователь разрешил private smoke и production после успешного gate. Аутентификация только SDK/CLI; содержимое `~/.kaggle*` не открывалось. Kaggle-ресурсы не удалялись.

## Находка и исправление

**P2 — статическая проверка стирала результат ревью и запуска.** Ранее `b5_static_check.main` каждый раз записывал `GPU_runs=0`, `pushes=0`, `GPU_smoke_pending:true`, теряя ссылку/старт/ETA. `kaggle/b5_static_check.py:9` теперь сохраняет историю; GPU gate требует successful receipt с совпадающими **modules и C_overrides**. Изменение C core также инвалидирует старое подтверждение. `kaggle/b5_tests.py:203` проверяет отсутствие/failed/stale receipts в обеих namespaces, successful receipt, сохранение RUNNING/URL/UTC/ETA и счётчиков, инвалидирование старого запуска при смене исходников. Training math и production modules не менялись.

Остальных критичных дефектов до GPU-проверки не найдено. Добавлены account-scoped [API helper](../kaggle/b5_review_api.py) с production gate/quota/private/T4/timeout и [output auditor](../kaggle/b5_review_output.py), который проверяет сохранённые данные вместо текущих логов UI.

После обнаружения неизвестной v2 добавлен **guard параллельных push**: `kaggle/b5_review_api.py:11` требует совпадения текущей серверной версии с последней собственной квитанцией, не допускает третьего reviewer smoke push, а обычный production допускает только при отсутствии предыдущего push. Неизвестная версия останавливает helper до мутации. `kaggle/b5_tests.py:test_b5_push_rejects_untracked_version_and_duplicate_production` проверяет новый kernel, известный smoke, неизвестный concurrent version, исчезнувшую версию, повторный production и исчерпание двух smoke attempts. Это проверка orchestration, training sources не меняются.

После отмены production до выделения GPU предусмотрено одно исключение `--retry-cancelled`: свежий terminal CANCEL_ACKNOWLEDGED, output0, полное равенство code cells, used_GPU точно прежний и reserved0; не больше одного такого повтора. RUNNING/расход_GPU/резерв/outputs/смена кода не допускают retry. Все отрицательные случаи покрыты той же регрессией. **При реальной попытке helper обнаружил concurrent v3 RUNNING и рост used_GPU; guard сработал, новый push не отправлен.** [Read-only preflight](../checks/b5-review-production-retry-preflight.json).

Перед production GET нового kernel вернул403. Он не трактуется как отсутствие: `b5_review_api.require_absent_target` требует пустую собственную push-history и полностью перечисленный owned inventory без этой цели. Эта проверка подтвердила отсутствие B5 перед единственным фактическим production push. Та же регрессия проверяет существующую цель, неполный список и предыдущий push: все отклоняются. Первая failed preflight-попытка ничего не отправила на Kaggle.

После последних дополнений guard отдельно повторно проверен: [регрессия с SHA файлов](../checks/b5-review-extra-tests.json),1 test passed. Остальные B5/historical tests повторно не запускались: training modules не менялись.

**P2, не блокирует обучение — неоднозначный `main_pairs` у C.** `src/b5_runtime.py:54,67,120` записывает туда order cursor, тогда как total main pairs = `state.updates*128`. Из-за унаследованного cursor_origin у C разность6035840 предъявлений; у W эти величины совпадают. Для анализа B5 брать total из updates, добавленные пары из delta updates, cursor использовать только для порядка. Runtime не менялся ради переименования поля; reviewer receipt отдельно содержит обе величины. Прогноз в handoff уже правильно различает total_pairs и order_cursor.

## Проверенные риски

Пути ниже относительно `work/kaggle/`.

| Область | Доказательство и границы |
| --- | --- |
| W600: строгое продолжение | `src/b5_state.py:18,46,58,67`; `src/b4_runtime.py:47`; `src/b2_runtime.py:130,151`. Exact latest/full bytes/SHA/identity/spec/cache/order; model, AdamW groups/moments/step, enabled scaler, Python/NumPy/CPU/CUDA RNG, state/history. B4 plain SHA `f23fe1e3…abc7a`, updates111933/cursor14327424/main91801,755194с. LR0,0006/factor1; warmup2000 уже пройден. |
| C-long2x: строгое продолжение | `src/b5_state.py:33,67,85`; `src/l2_runtime.py:57`; `src/l1_runtime.py:145`. SHA `48f2ba3f…f5a80b`, updates209144/cursor20734592/main79850,428875с/l2_main62280,492996с. Сохраняется cursor_origin; total pairs26770432 отличается от cursor по историческому переходу L1→L2. LR0,0015 constant, новый warmup отсутствует. Старые l2_points/history не сбрасываются. |
| Реализации и RNG | `build_b5.py:16,36`; `b5_worker.py:9`; `b5_sources.py:7`. C запускается в отдельном `src-C` с исходными L3 core/decoder; W — QK LayerNorm/z-loss1e-4. Проверяются frozen SHA и Torch2.11.0+cu128/CUDA12.8. Общий старт/поток/RNG между разными моделями не требуется. Побитная идентичность двух независимых200 CUDA trajectories не используется как restore-инвариант — урок B3. |
| Входы/доступ/SHA | `build_b5.py:46,76`; `src/b5_resume.py:10,26,36`; `src/l3_resume.py:8,29,56`. W transfer v1 manifest/payloads, exact latest; C live v12 latest↔manifest↔pin и full SHA. Cache manifest `46ef881b…02be` и все shards проверяются до workers. Каждый унаследованный order проходит SHA/cache/seed/multiplicity check, C metadata exact. Mounted full PT — symlinks, без дополнительных копий. [Повторный API preflight](../checks/b5-review-inputs.json): private transfer v1 с exact inventory, private live v12, shared private cache/2 и friend-access starter. Ранее remote bytes/SHA проверены [подготовкой](../checks/b5-transfer.json); GPU повторит на реальных mounts. |
| Проба/restore | `src/b5_runtime.py:58,71,85,90,104`. Одна проба origin+16200 main-секунд; disposable900с timed linear; eval cap300с. Полный uncooled state восстанавливается в finally и после ошибки; затем добавляются только b5 history/time. Smoke callbacks до ветки/после restore сверяют12 полей, включая весь state и режимы модулей. |
| Final50мин/K8/FP16 | `build_b5.py:23`; `src/b5_runtime.py:64,75,83`; `src/b1_eval.py:13`; `src/l2_eval.py:38,98`. Final3000с update-time linear, q0 обеих веток; сохранён фактический endpoint LR без дополнительного нулевого шага. Greedy4096/old991, CE, test1919, K8 tune2048/T0,7/seed270042, batch32/cap512, ID/question SHA/code/LP/execution journals. Fresh не читается. `src/l1_runtime.py:150`: отдельные FP16 weights. Короткий smoke сокращает splits и generation cap, не измеряет качество финала. |
| Uncooled full/checkpoints | `src/b5_runtime.py:55,62,98,135,137,147`; `src/b5_state.py:77`; `src/b2_runtime.py:115`; `src/l2_runtime.py:73`. Stable на старте, каждые900с, перед/после branches и в конце main; atomic file→latest, два поколения/GPU. Final summary `B6_uncooled_latest` указывает на полный main model/Adam/scaler/RNG/cursor, не охлаждённые веса. |
| Безопасность обновления | `src/b5_runtime.py:122`; frozen B2/L2 safe_update. Nonfinite и AMP skips учитываются; unsafe Adam OOM/escalation восстанавливает последний stable и останавливает только эту ветку с partial. LR сохраняется. Здоровая соседняя GPU продолжает. |
| Бюджет | `build_b5.py:23`; `src/b5_runtime.py:20,71,101`. Worker40500/watchdog40740/backend40800с=11ч20мин от first code cell. Номинальные caps40320с до checkpoint I/O. Main cap33840с; reserve5100с автоматически сокращает main; probe требует ещё собственный reserve. Полный9ч24мин main не гарантируется. Setup480с; smoke660/690/720с. Queue/VM startup/output publication вне first-cell бюджета и могут сдвинуть COMPLETE/финальное списание Kaggle. |
| Мягкий финал/watchdog | `b5_worker.py:42`; `src/b5_runtime.py:88,139`; `src/b1_session.py:27,49,65`; `src/l1_session.py:8`; `src/session_watchdog.py:92,117`. Incomplete evaluation даёт partial и stable, финального throwing assert нет. Watchdog включён до Torch/install; дочерние process groups завершаются TERM/KILL, hard exit пишет partial. Наследованный короткий watchdog cap переопределён до40800. |
| Private/квота/объём | `build_b5.py:76`; API helper перед push проверяет owner friend1/private/T4/входы/timeout/квоту, перед production ещё successful same-source smoke. `KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle quota`:30ч, reset24.10 00:00UTC; exact API108000с перед smoke. Планируемый output≈4,8GB, temporary checkpoint peak≈6,8GB, guard10GiB/free512MiB. Live-sync отсутствует: сбой VM до публикации может потерять output. |

## Локальные проверки

**12 B5 tests +128 historical tests passed**, один дополнительный L2 process-group test пропущен из-за macOS sandbox (`killpg`); Linux supervisor не менялся. [B5 receipt](../checks/b5-cpu.json), [historical receipt](../checks/b5-regressions.json), [обе сборки/статика](../checks/b5-static.json), [review receipt](../checks/b5-review-local.json). Production и server smoke совпадают по106 module SHA и двум C overrides.

Подготовленные реальные CPU next-batch128 проверки обеих веток приняты после проверки SHA исходников: [W](../checks/b5-real-resume-gpu0.json), [C](../checks/b5-real-resume-gpu1.json), по2 update с production save/read/restore и bit-exact historical control. Повторно многогигабайтные PT не скачивались. Это FP32 CPU/no autocast/fused=False; GPU gate обязателен.

## GPU-smoke

Private [v1](https://www.kaggle.com/code/friend1/nlp-hw02-b5-gpu-smoke/versions?versionNumber=1), API push18.10.2026 **11:39:53UTC**, timeout720с. [Push](../checks/b5-review-smoke-push-v/1.json), [server source/private](../checks/b5-review-smoke-metadata-v1.json), [status](../checks/b5-review-smoke-status-v1.json). После длительного QUEUED в11:55:40UTC получен CANCEL_ACKNOWLEDGED, failure_message пустой; reviewer отмену не отправлял. Output0 в11:55 и11:57. Успешное CUDA-покрытие не установлено. Квота после отмены сначала42,240361с used, перед повтором132,524226с; это наблюдения, не доказанная окончательная цена v1. [Failed gate](../checks/b5-review-gpu-smoke-v1.json).

Второй и последний reviewer push — [v3](https://www.kaggle.com/code/friend1/nlp-hw02-b5-gpu-smoke/versions?versionNumber=3), **11:57:26UTC**, тот же source port и timeout720с. [Push](../checks/b5-review-smoke-push-v/3.json), [server source](../checks/b5-review-smoke-metadata-v3.json). Отдельно обнаружена **v2 RUNNING** с теми же module SHA; этот reviewer её не отправлял, происхождение уточняется у пользователя. Версии не смешиваются; третьего reviewer smoke push не будет. Production ещё не запускалась. Gate требует COMPLETE/output, обе200+ main updates без nonfinite, четыре точных restore-аудита, stable/FP16 и mini-K8.

В12:02UTC v3 получила CANCEL_ACKNOWLEDGED, output0; reviewer cancel не отправлял. V2 продолжает RUNNING. [Failed v3 gate](../checks/b5-review-gpu-smoke-v3.json), [v2 metadata](../checks/b5-review-smoke-metadata-v2.json), [v2 status](../checks/b5-review-smoke-status-v2.json). Любое принятие v2 потребует полного output-аудита и подтверждённого COMPLETE; одного совпадения исходников недостаточно.

**V2 gate passed:** COMPLETE подтверждён12:04:59UTC; session complete, exits[0,0], wall **466,796932с≤720**. [Полный аудит](../checks/b5-review-gpu-smoke-v2.json). Обе200 main updates, nonfinite/AMP skips/OOM0; четыре before/restored-аудита по12 полям точные. Probe2с +final3с, mini greedy64/old16/CE/test64, K8 на36 frozen tune IDs/288 samples на каждой GPU полны. Uncooled latest/два full поколения на GPU и четыре FP16 points согласованы с producer SHA; latest отдельно загружен/проверен самим GPU controller до injected overflow. Overflow изменил scaler, сохранил model/Adam; после него stable не перезаписывался.

Скачано126 small outputs через **version-label**, проверены123 producer SHA и все106 `src-C` module hashes; PT не скачивались. Output **4 779 614 294байта**. Production, server v2 и runtime inventory совпадают по106 modules +двум overrides. Сравнение независимых W/C RNG или stream отсутствует по протоколу. Source anchors/orders, различающиеся cursor_origin и constant LR проверены отдельно. Периодическое сохранение по900с проверено кодом/CPU; короткий GPU-run до этого интервала не доходит, но использует те же production serializers перед/после branches и в конце main.

Дополнительно через SDK получены полные notebook sources **v2 и production v1**: все три code cells каждого побайтно равны своему локальному bundle, включая startup/CONFIG/install/setup/launch, а не только SOURCES. [Полная notebook parity](../checks/b5-review-notebook-parity.json). Это исключает незамеченную смену orchestration в параллельной v2 при неизменённых module SHA.

Все server attachments production v1 совпали с successful smoke v2: starter, W transfer, L3 live и compact-cache; [квитанция](../checks/b5-review-server-mounts.json). Этот metadata endpoint возвращает refs без номера версии, поэтому exact содержимое по-прежнему защищено runtime SHA/latest/manifest pins, реально пройденными v2. При длительном QUEUED отдельно проверен latest production: существует только v1, не неизвестная параллельная v2.

От первой quota108000с до production preflight107502,56с наблюдалась суммарная дельта **497,44с≤720**; reserved0. Это весь аккаунт между snapshots, не индивидуальная цена отменённых версий. Первые42,24/132,52с не объявляются финальной ценой v1. Собственных smoke push два — v1/v3; successful v2 наблюдалась отдельно, её origin не установлен. Качество полного final50мин и холодный ensemble180с коротким smoke не измерялись.

## Боевой B5

Private [B5 v1](https://www.kaggle.com/code/friend1/nlp-hw02-b5/versions?versionNumber=1), единственный фактический production push этого reviewer **18.10.2026 12:07:58UTC**, backend40800с, `--accelerator NvidiaTeslaT4`, owner аккаунт друга. Перед push свободно **107502,56с≈29,86ч≥42000с**, reset24.10 00:00UTC; после полного40800с бюджета останется66702,56с≈18ч32мин без дополнительного платформенного overhead. [Push](../checks/b5-review-production-push-v/1.json), [server private/T4/source](../checks/b5-review-production-metadata-v1.json), [status](../checks/b5-review-production-status-v1.json). Все production modules/overrides совпали с successful v2.

V1 после QUEUED получила CANCEL_ACKNOWLEDGED, output0, reviewer cancel не отправлял. Затем обнаружена внешняя v2, также отменённая; её полные code cells/mounts совпали с v1, [проверка](../checks/b5-review-production-v2-parity.json). До этого used_GPU оставался497,44с: расход production не наблюдался.

**Активный B5 — private [v3](https://www.kaggle.com/code/friend1/nlp-hw02-b5/versions?versionNumber=3), RUNNING подтверждён18.10.2026 13:13:24UTC.** Это concurrent version, которую данный reviewer не отправлял; происхождение внешних v2/v3 не установлено. Её полный code совпал с проверенной v1, modules/C overrides — с successful smoke, private/T4 и runtime40500/40740/40800с подтверждены. [Активная версия](../checks/b5-review-production-active-v3.json), [metadata](../checks/b5-review-production-metadata-v3.json), [RUNNING](../checks/b5-review-production-status-v3.json). Второй reviewer production push guard не допустил; третьего smoke push не было.

**ETA≈19.10 00:33UTC /03:33МСК** = первое формальное наблюдение RUNNING +11ч20мин. Это ориентир, не фактический first-code-cell start; queue/VM/output overhead могут сдвинуть COMPLETE. Указанный40800с backend request проверен у собственного v1; backend timeout concurrent v3 metadata API не раскрывает. Её **runtime/watchdog ceiling40800с** точно тот же по полному коду. [Машинный handoff](../results/next-b5.json) хранит активную ссылку v3, UTC/ETA, происхождение версии и эту границу проверки; собственный push v1 сохранён отдельными полями.

## Коммит

Финальный `git add` явного списка67 B5-файлов завершился128: песочница запрещает создать `<vault>/.git/worktrees/study1/index.lock` (`Operation not permitted`). Индекс не изменён; commit не создан, ограничение не обходилось. [Квитанция](../checks/b5-review-git.json). Чужая правка библиографии не staging.

## Источники

- [b5-plan](b5-plan.md), [b4-review](b4-review.md), [b3-review](b3-review.md), [l3-plan](l3-plan.md) и локальные исходники/CPU/API/GPU receipts, ссылки выше. Библиографические записи подготовки и прошлых ревью уже существуют в bibliography. Новых внешних работ не использовано. Оценка рисков и выводы — собственное независимое ревью.
