---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-20
updated: 2026-10-20
sources:
  - kaggle-b6-results
  - kaggle-b5-preparation
aliases: []
---

# B7: независимое ревью и запуск

Ветка `codex/nlp-hw2-code-generation`, HEAD `5b5bb39`. Прочитаны [b6-review](b6-review.md), [b6-results](b6-results.md), [b7-plan](b7-plan.md), handoff и фактические исходники. Исходный `next-b7.json` имеет `ready:true`. От прерванной попытки остались quota receipt и незавершённый CPU log; ни intent, ни push receipt не было. [Серверная проверка](../checks/b7-review-server-before.json) подтвердила отсутствие обеих целей полным owned inventory после HTTP403. `submission/`, E2, progress/index/dashboard/backlog не меняются.

## Находки и проверенные риски

Критичных дефектов training/resume/расписания не обнаружено; runtime не изменён. **Недостаток проверки:** тесты подготовки проверяли single-push через B6 helper, хотя запуск выполняет B7 helper. Добавлены две независимые проверки именно B7: запрет повторов/неизвестных версий для обеих целей, эксклюзивность intent, production gate с обязательным same-source/config smoke и отказ при stale modules/C-overrides/config. [Тесты](../kaggle/b7_review_tests.py), [результат](../checks/b7-review-guard-tests.log). Это дополнение валидации, не изменение GPU-кода.

Пути и строки в таблице — относительно `work/kaggle`.

| Риск | Проверка, результат и границы |
|---|---|
| Исходные состояния B5 | `src/b6_resume.py:8–27`, `src/b5_state.py:18–64`, `b7_static_check.py:37–49`: mount B5v6; exact manifest `f662a966b5031a93bb46d25530a2710753a26f69e5262af80eca05ac6c6aa525`, bytes/SHA/latest/final-anchor receipts. W PT `3b11360d5242e0e8b3fc1ac38254f0d4d544fba8bb62ee4c7bb034943e6cdc32`, updates154559/cursor19783552; C PT `69dff4405d2236097f48733adbfc9e3bfd5d20b501a3fcfd1dae62ec2ce63074`, updates293688/cursor31556224. Это uncooled latest main B5, не cooled B5/B6. |
| Provenance-ограничение | Pre-final B5 PT удалены самой B5; их tensors непосредственно не сверены. Переиспользуются SHA-bound B6 receipts: exact final-anchor metadata, равенство двух сохранённых post-restore поколений и реальные CPU2×batch128/disk-resume. Скрипт/implementation SHA сверяются заново (`b7_static_check.py:75–88`). Нельзя объявлять это прямым сравнением с удалёнными PT. |
| Strict resume | `src/b5_state.py:67–82`, `src/l1_runtime.py:145–148`, `src/b6_state.py:55–80`: FP32 model/Adam moments, steps и groups/scaler/Python+NumPy+CPU+CUDA RNG; неизменный historical main state. `src/b5_resume.py:36–44`: SHA/cache/seed/order/multiplicity short1,long2. C cursor offset сохранён. Torch2.11.0+cu128/CUDA12.8, frozen W/QK и C overrides закреплены. |
| LR и ровно N | `src/p1_math.py:24–27`, `src/b6_state.py:15–34`, `src/b6_runtime.py:34–66`: $\eta_j=\eta_0(1-\sqrt{j/N})$, $j=1,\ldots,N$; $\eta_0$=0,0006/0,0015 из stable state, N33000/65800, $\eta_N=0$. N означает attempted updates: AMP/nonfinite skip расходует слот/128пар. Полный N проверен controller-тестом на98800 слотах, численный disk-resume — отдельно для обеих math paths. |
| q0 без truncation | `src/b5_state.py:81–82`, `src/b6_state.py:29–34`: frozen обычный safe_update, все128пар без loss selection. `build_b7.py:10–22`: единственные изменения B6 — N и session envelope/название; embedded modules и все исполняемые ячейки вне CONFIG совпадают. |
| Бюджет ≤11ч20мин | `b7_budget.py:17–61`: SHA-verified полные B6 logs16500/32900, 0,781618/0,421779с на update. N×2 +10% +480setup +1800eval/K8 +600I/O →31252,72/33408,37с; max=9ч16мин48с, не сумма GPU. `src/b6_runtime.py:23–36` резервирует2400с до worker deadline40500; watchdog40740/backend40800. При замедлении partial и fixed N, без автоматического уменьшения N. |
| Квота | [Свежая API quota](../checks/b7-review-quota-current.json):50806,306с свободно, reserved0, reset24.10 00:00UTC. Production40800+smoke720+600platform оставляют8686,306с≈2ч25мин. Перед каждым push helper заново требует hard+120с (`b7_review_api.py:81–86`). Очередь/startup/publication вне первой code cell могут сдвинуть COMPLETE и расход квоты. |
| Финал/K8/FP16 | `b6_worker.py:25–40`, `src/b1_eval.py:13–16`, `src/l2_eval.py:38–112`, `src/l1_runtime.py:150–158`: full heldout4096/old991 greedy execution+CE, original length bins, test1919 pipeline timer; fixed tune2048/K8/T0,7/seed270042/batch32/cap512. Midpoint только FP16; final FP16 до оценки. Smoke мини64/16/test64/K8 на36IDs/cap8/batch2. Fresh не оценивается. Smoke подтверждает механику, не качество N×2 и не full K8 timer. |
| Watchdog/мягкие assert | `src/b1_session.py:12–26,30–81`, `src/session_watchdog.py:92–127`: watchdog до Torch/install, отдельные process groups, TERM/KILL/partial receipts. `src/b6_runtime.py:41–47,67–89`: unsafe rollback к finite full checkpoint без LR-halving; ошибка final eval оставляет FP16/latest и partial. `b6_worker.py:42–46`: нет throwing final assert; ошибка одной GPU не останавливает здоровую соседнюю. Atomic latest и два full поколения, disk guard10GiB. Нет live backup при аварии VM до publication. |
| Private/дубли | `build_b7.py:35–41`, `b7_review_api.py:11–31,64–86`: private friend1/T4×2, pinned inputs, source/config и fresh quota, server-version+полный owned inventory при403, exclusive intent перед сетью. Ровно один push на цель; неоднозначный ответ не разрешает повтор. `src/b1_session.py:19–22`: credentials удалены из worker env; нет Secrets/live uploads. Содержимое `~/.kaggle*` не читалось, authentication только SDK/CLI; на Kaggle ничего не удаляется. |

## Проверки

Независимый повтор157 CPU-регрессий завершён: **156passed,1прежний macOS sandbox-skip killpg**; дополнительно2/2B7 guard tests passed. Compile и `git diff --check` прошли. CPU использует Torch env, API — отдельный Kaggle SDK env. Секреты не инспектируются. [CPU log](../checks/b7-review-tests.log), [B7 guard tests](../checks/b7-review-guard-tests.log), [независимый anchor/budget audit](../checks/b7-review-provenance-budget.json). Read-only observer дополнительно сверяет все server/local code cells, private/T4/internet и текущую последнюю версию, не полагаясь на запаздывающие UI logs.

## GPU-smoke

[Private smoke v1](https://www.kaggle.com/code/friend1/nlp-hw02-b7-gpu-smoke/versions?versionNumber=1) отправлен **20.10.2026 07:57:03,847UTC**, ровно один push точной CLI-командой из handoff через guarded helper. Перед сетевой мутацией повторно проверены server absence, source/config/private/quota; [intent](../checks/b7-review-smoke-push-v.intent.json), [push receipt](../checks/b7-review-smoke-push-v/1.json). Backend timeout720с. Первое наблюдение07:57:14UTC — QUEUED, private/T4/internet/server code exact; [история](../checks/b7-review-smoke-observations.json).

**RUNNING подтверждён08:10:51,619UTC**, после очереди≈13мин48с. Server code/private/T4/internet совпали. Output пока отсутствует.

При QUEUED>15мин или RUNNING>15мин без output — запись здесь и ожидание до45мин без повторного push; при внешнем перезапуске отслеживается последняя версия. Признак успеха — COMPLETE и проверенный producer output.

## Результат GPU-smoke

**COMPLETE подтверждён08:19:04UTC. Wall380,125с≤720, exits[0,0], ровно один push.** [Основной аудит](../checks/b7-review-gpu-smoke.json):98 producer files проверены по bytes/SHA, скачаны99компактных файлов с manifest; PT/order/cache не скачивались. Config/111modules/C overrides совпадают с production; server code cells полностью равны локальному smoke. В producer manifest все111 `src-C` SHA совпали с frozen namespace и двумя overrides.

[Независимый дополнительный аудит](../checks/b7-review-extra-gpu-smoke.json), воспроизводится `PYTHONPATH=src <CPU-python> b7_review_extra_output.py ../results/kaggle-runs/nlp-hw02-b7-gpu-smoke/1`: initial model/Adam/scaler/main state совпали с real CPU B5 receipts, исходный CUDA RNG точен; historical main state не изменён. Проверены все400 LR/cursor/pairs128/q0, finite CE и FP16 generation, все K8 IDs/question SHA/represented code/mean LP/execution. Все serialized RNG и финальная FP16-проекция full model точны.

| Проверка | GPU0 W | GPU1 C-long2x |
|---|---:|---:|
| Attempted updates |200|200|
| Nonfinite /AMP skips /OOM |0/0/0|0/0/0|
| Первый LR |0,0005575735931288071|0,001393933982822018|
| Последний LR |0|0|
| Pure update time, с |157,665941|77,441342|
| K8 rows /samples |36/288|36/288|
| Model/Adam/scaler/RNG/cursor/FP16 |passed|passed|

W0,788330с/update лишь≈0,86%медленнее полного среднего B6; C0,387207 быстрее B6. Оснований менять N нет, сохраняется консервативный B6-прогноз9ч16мин48с. Smoke не доказывает качество длинного cooldown или полный K8 timer.

[Квота после smoke](../checks/b7-review-quota-after-smoke.json):50391,674с свободно, reserved0; расход414,632с≤720. После полного production cap40800с останется9591,674с≈2ч40мин, отдельно600сplatform reserve. До production helper снова проверяет fresh quota и отсутствие цели/истории.

## Боевой запуск

[Private B7 v1](https://www.kaggle.com/code/friend1/nlp-hw02-b7/versions?versionNumber=1) отправлен **20.10.2026 08:20:58,557UTC**, ровно один production push. Guarded helper выполнил точную `launch_command` из handoff: `KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle kernels push -p bundles/b7_friend1 --accelerator NvidiaTeslaT4 --timeout 40800`. Перед push серверный HTTP403 дополнен полным owned inventory/отсутствием цели, собственная история пуста; эксклюзивный [intent](../checks/b7-review-production-push-v.intent.json), [push receipt](../checks/b7-review-production-push-v/1.json). Fresh quota50391,674с, reserved0.

**RUNNING подтверждён08:21:24,469UTC**, private/T4/internet и точное равенство всех server/local code cells — [наблюдение](../checks/b7-review-production-observations.json). N33000/65800, q0, нового main нет. Один smoke push и один production push, других запусков/повторов нет.

**ETA20.10≈17:38UTC /20:38МСК** (B6-based conservative forecast9ч16мин48с); ориентир hard cap≈19:41UTC /22:41МСК. Старт — первое наблюдение RUNNING, не известное время первой code cell; startup/publication могут сдвинуть COMPLETE. Ссылка, version1, UTC-start/basis, ETA UTC/МСК, cap и quota сохранены в [next-b7.json](../results/next-b7.json). Задача запуска выполнена; завершение и качество длинного run ещё не проверены.

## Источники

- [b6-review](b6-review.md), [b6-results](b6-results.md), [b7-plan](b7-plan.md), [handoff](../results/next-b7.json); собственное независимое ревью локальных исходников и API/CPU/GPU receipts. Использованные источники `kaggle-b6-results`/`kaggle-b5-preparation` уже есть в bibliography; новых научных источников нет.

## Коммит

20.10 в08:09:05UTC попытка `git add` явного B7-only списка получила exit128: создание `<vault>/.git/worktrees/study1/index.lock` запрещено (`Operation not permitted`). Индекс остался неизменным/пустым, коммит не создан; ограничение не обходилось. [Квитанция и список](../checks/b7-review-git.json). Все материалы сохраняются в worktree; чужие submission/E2 не затронуты.
