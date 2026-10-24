---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-19
updated: 2026-10-19
sources:
  - kaggle-b5-preparation
  - kaggle-b5-results
  - kaggle-p1-final-results
aliases: []
---

# B6: независимое ревью и запуск

Проверена ветка `codex/nlp-hw2-code-generation`, HEAD `c350d6a`; исходный handoff `ready:true`. B6-файлы подготовки были untracked. Прочитаны [b5-review](b5-review.md), [p1-review](p1-review.md), [b6-plan](b6-plan.md), код и receipts. `progress.md`, index, dashboard и backlog не редактируются. Пользователь разрешил один private smoke и один production после успешного smoke. Содержимое `~/.kaggle*` не открывается; authentication выполняет SDK/CLI с `KAGGLE_CONFIG_DIR=~/.kaggle-friend`. Удалений на Kaggle нет.

## Находки

**P2, исправлено: разрешался второй smoke push.** `kaggle/b6_review_api.py:11–22` вводит отдельное правило B6: только пустая собственная история и server version0. Наследованный B5 guard разрешал два smoke-push. До сетевой мутации теперь эксклюзивно создаётся intent; timeout/неоднозначный ответ не открывает путь к повтору. Регрессия `kaggle/b6_tests.py:test_b6_exactly_one_push_per_target_no_retry_or_unknown_version` проверяет обе цели, повтор, неизвестную/исчезнувшую серверную версию и сохранность первого intent. Training modules не менялись.

**P2, исправлено: успешный GPU-аудит не записывался в handoff при относительном `--out`.** `kaggle/b6_review_output.py:10–11,59` теперь нормализует receipt path перед `relative_to`; ранее рекомендованный `../checks/...` давал ValueError уже после успешной проверки output. Регрессия `kaggle/b6_tests.py:test_b6_output_receipt_accepts_relative_and_absolute_paths` проверяет оба вида пути и отклонение пути вне work. GPU-код не менялся; новый push не нужен.

**Ограничение provenance, не скрывается:** исходные pre-final PT B5 удалены самой B5. Их SHA `89441349…d9695` / `cd26644d…96cad` нельзя непосредственно сверить с tensors. Совпадают полные anchor metadata (updates/cursor/main_s/identity/spec), а у двух сохранённых post-restore поколений побитно совпадают model/Adam/scaler/RNG/scheduler и main state без четырёх B5 reporting-полей. Это свидетельство штатного восстановления B5, а не прямое сравнение с удалённым PT. Проверены исходники этой проверки и SHA её CPU receipts; повторного скачивания многогигабайтных PT не было.

## Проверенные риски

Пути относительно `work/kaggle`; строки относятся к reviewed source.

| Область | Проверка и границы |
| --- | --- |
| Исходные uncooled состояния | `src/b6_resume.py:8–27`, `src/b5_state.py:18–64`: B5v6 manifest SHA `f662a966…aa525`, exact latest/bytes/SHA и final receipt. W SHA `3b11360d…cdc32`, updates154559/cursor19783552/main125641,969562с; C SHA `69dff440…63074`, updates293688/cursor31556224/main113690,728379с. Не B5 cooled FP16. Независимо пересчитаны metadata и B5 update-log SHA: [receipt](../checks/b6-review-provenance-budget.json). |
| Строгость продолжения | `src/b6_runtime.py:16–20`, `src/b6_state.py:15–34,55–80`, `src/b5_state.py:67–82`, `src/l1_runtime.py:145–148`: FP32 model, Adam moments/steps/groups, enabled scaler, Python/NumPy/CPU/CUDA RNG; historical main state неизменен, собственный CD cursor. `src/b5_resume.py:45–53` проверяет weighted order SHA/cache/seed/multiplicity; C cursor offset сохранён. Torch2.11.0+cu128/CUDA12.8 и frozen W/QK/z-loss/C overrides закреплены. Реальные CPU2×batch128/disk restore receipts подготовки совпадают с текущими implementation SHA. |
| LR и q0 | `src/p1_math.py:24–27`, `src/b6_state.py:15–34`, `src/b6_runtime.py:34–66`: peak из stable LR0,0006/0,0015; $\eta_j=\eta_0(1-\sqrt{j/N})$, $j=1,…,N$, N16500/32900. Ровно N attempted updates, последний LR0 без добавочного шага; AMP/rejected-loss skip расходует слот/128пар отдельно. `S.update_for` вызывает обычный frozen q0 safe_update; отбора/truncation нет. |
| Бюджет и квота | `build_b6.py:20–32`, `src/b6_runtime.py:23–36,62–77`: main0; max(mean B5 main, mean final50) =0,797098/0,401070с на update; +10% update-time +480setup +1800eval +600I/O →17347,33/17394,74с (<4ч50мин). Worker18900/watchdog19140/backend19200с=5ч20мин; CD останавливается до deadline−2400, N не меняется. Fresh quota перед smoke:68248,345с, reserved0, reset24.10 00:00UTC; [receipt](../checks/b6-review-quota-before.json). |
| Final/K8/FP16 | `b6_worker.py:25–40`, `src/b1_eval.py:13–16`, `src/l2_eval.py:38–112`, `src/l1_runtime.py:150–158`: heldout4096/old991 greedy и CE, original length bins, test1919; fixed tune2048/K8/T0,7/seed270042/batch32/cap512, journals ID/question SHA/code/LP/execution. Fresh не оценивается. Midpoint только FP16 при запасе I/O; final FP16 до оценки. Smoke мини64/16/test64/K8 на36 IDs с cap8/batch2. |
| Watchdog и мягкий финал | `src/b1_session.py:12–79`, `src/l1_session.py:8–11`, `src/session_watchdog.py:92–127`: watchdog до Torch/install, TERM/KILL process groups, B6 override короткого исторического cap. `src/b6_runtime.py:41–47,67–89`: unsafe Adam/nonfinite rollback к finite checkpoint и partial без LR-halving; ошибка оценки оставляет FP16/latest. `b6_worker.py:42–46`: throwing final assert отсутствует. Здоровая соседняя GPU продолжает. |
| Private/сохранность | `build_b6.py:61–69`, `b6_review_api.py`: owner аккаунт друга, private/T4x2/internet/timeout/source/config/quota и server-version gates. Нет Secrets/live/upload в worker. Два full поколения/GPU, atomic latest; около4,8GB output, temporary checkpoint peak около6GB, guard10GiB. Нет локальных копий входных full PT. |

Hard budget начинается с первой code cell; очередь, VM startup и output publication могут сдвинуть COMPLETE и квотную стоимость. Нет live backup: авария VM до публикации может потерять output. Полный N или финальная оценка при замедлении не гарантируются; partial не выдаётся за complete. Smoke не оценивает качество длинного cooldown.

## Проверки и GPU

12/12 B6 CPU tests passed, включая новую регрессию; [CPU receipt](../checks/b6-cpu.json). Повторная статика обеих сборок, embedded/local source parity и frozen SHA passed. 141 исторический тест:140 passed,1 прежний macOS sandbox-skip killpg. [Независимый local receipt](../checks/b6-review-local.json). GPU-smoke v1 отправлен одним push в09:55:09UTC; RUNNING подтверждён09:56:08UTC, COMPLETE —10:03:16UTC. Серверный notebook полностью совпал по code cells; private/T4x2/internet подтверждены.

## Источники

- [b6-plan](b6-plan.md), [b5-review](b5-review.md), [b5-results](b5-results.md), [p1-review](p1-review.md); локальные исходники и CPU/API/GPU receipts. Использованные работы уже есть в bibliography; новых внешних источников нет. Выводы — собственное независимое ревью.


[Private smoke v1](https://www.kaggle.com/code/friend1/nlp-hw02-b6-gpu-smoke/versions?versionNumber=1) **COMPLETE**, ровно один push; session wall **360,526с≤720**, exits[0,0]. [Основной audit](../checks/b6-review-gpu-smoke.json) проверил98 producer files по bytes/SHA; скачано99 малых файлов, включая manifest. Все111 `src-C` module hashes совпали с проверенной namespace; оба C overrides совпадают со smoke/production. Полный server notebook совпадает по всем code cells. Output≈4,78GB, PT/order/cache локально не скачивались.

| Проверка | GPU0 W600 | GPU1 C-long2x |
| --- | ---: | ---: |
| Cooldown updates |200|200|
| Nonfinite /AMP skips /OOM retries |0/0/0|0/0/0|
| Первый LR |0,0005575735931288071|0,001393933982822018|
| Последний LR |0|0|
| Pure update time, с |162,277625|79,472554|
| K8 rows /samples |36/288|36/288|
| Full restore /все RNG /cursor /FP16 |passed|passed|

[Дополнительный независимый аудит](../checks/b6-review-extra-gpu-smoke.json), воспроизведение из `kaggle/` через `b6_review_extra_output.py <output-folder>`: initial model/Adam/scaler/main state точно совпали с real CPU B5 receipts, opaque CUDA RNG совпал; historical main state остался неизменным. Проверены все200 LR на каждой GPU, pairs128/q0, finite CE/FP16 greedy и все K8 IDs/question SHA/represented code/finite LP/execution. Обе финальные FP16-проекции точно равны full model, serialized restore всех RNG точен. Фактические0,811388/0,397363с на update близки к опорным0,797098/0,401070; W медленнее на1,79%, внутри10% резерва. N не менялся. Smoke не измеряет качество и полный production eval timer.

## Боевой B6

[Private B6 v1](https://www.kaggle.com/code/friend1/nlp-hw02-b6/versions?versionNumber=1) отправлен **19.10.2026 10:05:00,654UTC** одним push через `guarded_launch_command` из handoff. Helper выполнил точную CLI-команду `kaggle kernels push -p bundles/b6_friend1 --accelerator NvidiaTeslaT4 --timeout 19200`, соответствующую `launch_command`. Перед push сервер ответил403; полный owned inventory подтвердил отсутствие цели, собственная история была пуста. [Эксклюзивный intent/preflight](../checks/b6-review-production-push-v.intent.json), [push receipt](../checks/b6-review-production-push-v/1.json).

**RUNNING подтверждён10:05:10,161UTC**, private/T4/internet и полное равенство server/local code cells — [receipt](../checks/b6-review-production-observations.json). Version1, N16500/32900, q0; smoke/production используют одинаковые111 modules и C overrides. Последняя free quota **67856,554с**, reserved0, reset24.10 00:00UTC. Smoke-account delta391,791с≤720; после полного production cap останется48656,554с≈13ч31мин до platform overhead.

**ETA19.10≈14:55UTC /17:55МСК**, ориентир по hard cap≈15:25UTC /18:25МСК. Старт — первое наблюдение API RUNNING, не известное время первой code cell; startup/publication могут сдвинуть COMPLETE. Ссылка, версия, UTC start/basis, ETA и quota записаны в [next-b6.json](../results/next-b6.json). Длинный run и его качество ещё не завершены; ждать его окончания в задачу запуска не входит.


## Коммит

Попытка `git add` явного списка45 B6 paths завершилась exit128: sandbox запретил создать `<vault>/.git/worktrees/study1/index.lock` (`Operation not permitted`). Индекс не изменён; commit не создан, ограничение не обходилось. [Receipt с полным списком и commit message](../checks/b6-review-git.json). После этого в receipt/report добавлена только итоговая запись; содержательные результаты сохранены в worktree.
