---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-21
updated: 2026-10-21
sources:
  - final-variants.md
  - final-package-v11.md
  - ../checks/final-package-v2-cpu.json
aliases: []
---

# Финальная сдача v2

Пользователь выбрал **W B7 (greedy-якорь) → C-long2x B7 → P1-B**,
**1/1/1 → 1/8/8**, budget640, seed270042/T0.7. Исходный вариант v2:
public80,10% (сообщение пользователя), tune1557/2048, test111,421с.
Код/веса/selector по результатам нового прогона не подбираются.

## Пакет и веса

[Config](../../submission/code/hw2_codegen/config.json) семантически совпадает
со всей сохранённой конфигурацией v2 из `final-variants-build.json`.
SHA16 исполняемых модулей, включая model/decode/pipeline/runner/execute/vote,
совпали с тем прогоном. В config.py изменена только устаревшая docstring.
Исторический сборщик пяти вариантов отвязан от текущего submission/config:
его база сохранена в `work/kaggle/frozen-v11-budget640.json`.

B7 взяты из локального pinned output `friend1/nlp-hw02-b7/1`:
W — `model-GPU0/points/b6-final`, C — `model-GPU1/points/b6-final`.
Каждый point/checkpoint.json указывает на `weights/b6-final-fp16.pt`;
SHA/bytes/FP16/no-optimizer подтверждены. P1-B — прежний output
`friend1/nlp-hw02-kaggle-p1/3`. Таблица SHA — в [README](../../submission/README.md).

Новая private версия `thisarivson/nlp-hw02-final-weights` — **v4**.
В неё включены только три финальных checkpoint: B6 опущены, чтобы проверяющий
не скачивал ненужные веса и не путал состав. Все прежние версии сохранены;
`delete_old_versions=False`. Новая версия создана одним version-call; все26 файлов скачаны обратно с явным
суффиксом `/4` и проверены по bytes/SHA, включая веса/code/manifest.
Содержимое `~/.kaggle*` не просматривалось; default SDK auth thisarivson.
Удалений, изменения видимости и действий от аккаунта друга нет.

CPU: **21/21 tests,17,292с**, fresh-kernel notebook Run All на8synthetic rows,
AST/JSON и git diff checks passed. Три финальных веса: SHA, FP16 finite,
правильные архитектуры, strict-load CPU. Все четыре входных файла byte-exact
относительно прогона вариантов. [CPU receipt](../checks/final-package-v2-cpu.json).

## GPU и аудит

21.10.2026 **06:55:27 UTC** выполнен ровно один private push, сервер принял v1.
Фактический [kernel](https://www.kaggle.com/code/thisarivson/nlp-hw02-final-package-v2-budget640/versions?versionNumber=1):
`thisarivson/nlp-hw02-final-package-v2-budget640`. Kaggle сформировал slug из
title, поэтому первое чтение запрошенного `...-v2` дало403. Owned inventory
показал фактический slug; все серверные code cells v1 exact, T4, Internet off,
единственный источник final-weights (в push указан `/4`). Это исправление
локальной привязки, без второго push. [Launch](../checks/final-package-v2-launch.json),
[server parity](../checks/final-package-v2-remote.json).

Один cold test1919, отдельный свежий tune2048. Таймер включает imports/read/SHA/load/warmup/decode/execute/vote/
переисполнение/закрытие answers. Память: exact Torch allocated/reserved,
NVML device-wide used каждые20мс; GB=10⁹bytes.

При QUEUED>10мин или RUNNING без output>12мин повторного push/отмены нет:
ожидание до60мин. Авторизованный пользователем interactive output можно
аудировать из `work/results/kaggle-runs/nlp-hw02-final-package-v2/interactive/`.
COMPLETE впервые подтверждён **07:01:02 UTC**, пороги ожидания не пересечены.
Скачаны19 output-файлов и отдельно исполненный notebook v1; code cells exact
до и после скачивания, execution counts1–4, ошибок0. Реальные outputs и
измерения перенесены в `submission/inference.ipynb`; GPU повторно не запускался.

| Split | Cold, с | Allocated, GB | Reserved, GB | NVML used, GB |
|---|---:|---:|---:|---:|
| Test1919 | **122,657770** | 11,008209 | **11,123294** | **11,759976** |
| Tune2048 | **107,986684** | 8,412472 | **8,487174** | **9,123856** |

Runtime Torch2.11.0+cu128/CUDA12.8/TeslaT4, одна видимая GPU0.
**Tune1560/2048=76,171875%**, +3/+0,146484п.п. к прежней v2.
Все gates passed: test≤150с/официальные180с, reserved≤14GB;
OOM/guards/partial/fallback/no-numeric/verification mismatch0.
1919 test,1919 прежних v2 и2048 tune независимо исполнены в трёх свежих
executor-сервисах; все numeric. Полный tune replay2048 winners/early stops
и match_indices exact. [Результат](../results/final-package-v2.json).

## Сравнение с прежней v2

Новый [final-package-v2.jsonl](../submissions/final-package-v2.jsonl):
1919 code-only строк, SHA `f9c772e41fae010c2fe9e50723cfda524dc4346c653f0319a86561ca0e7329ed`.
С прежней `variants-2026-10-20/v2.jsonl` **побайтного совпадения нет**:
**279 code /159 numeric** различий; численные сравнения используют
`math.isclose(rel_tol=abs_tol=1e-6)`. Все индексы и SHA — в
[comparison](../checks/final-package-v2-comparison.json).
Public80,10% относится к прежнему файлу; новый JSONL в бота не отправлялся.

Подтверждено: config семантически exact; веса, tokenizer, четыре входных
файла, seed/budget/decoder и16 исполняемых модулей exact; runtime версии одинаковы.
В runtime config различаются только пути и явное `dry_run:false`.
На test отличается1 greedy, на tune greedy2048/2048 exact. Все2048 первых
stochastic W samples совпали, включая logprob/execution. У C/P1-B первые
stage1 code отличаются1370/1382строками, logprob1600/1611; уже в первом
640-row batch различаются65/70программ. Значит, расхождение начинается в
**генерации до голосования**, а затем меняет early-stop и stage2 priority.
Early-stop symmetric difference: test131/tune162строки.

**Вероятный механизм (вывод по коду):** `decode()` обрабатывает async EOS,
когда `event.query()` сообщил готовность pinned snapshot. Момент обработки
зависит от CPU/GPU timing; compaction меняет число активных lanes и форму/
число вызовов `noise.exponential_()`. Фиксированный seed и deterministic
CUDA algorithms не фиксируют эту асинхронную ветку управления. Это объясняет,
почему одинаковые начальные batches/seed не гарантируют тот же поток samples.
Поэтапные CUDA event/compaction traces не сохранялись: точный шаг готовности
и возможный вклад FP16 kernels **не изолированы**. Новых forward/sampling/GPU
для выяснения причины нет, selector/веса/K не менялись.
[Воспроизводимая диагностика](../checks/final-package-v2-reproducibility.json),
[код](../kaggle/final_v2_reproducibility.py), [decoder](../../submission/code/hw2_codegen/decode.py).

## Воспроизведение аудита

Из корня репозитория (API — Python с Kaggle SDK, CPU — с зависимостями сдачи):

```sh
python work/kaggle/final_v2_api.py observe
python work/kaggle/final_v2_api.py output
python work/kaggle/final_v2_results.py
```

`observe`/`output` read-only для Kaggle. Повторный push блокируется ledger.
Аудит проверяет версию/код/SHA/порядок и конфигурацию, replay2048 tune-строк,
независимо исполняет test, прежний v2 и tune, сохраняет
`work/submissions/final-package-v2.jsonl` и реальные outputs notebook.
Для локально полученного интерактивного output поддерживается
`final_v2_results.py --root <каталог>` при наличии квитанций скачивания/версии.

Отчёт `submission/report/`, progress/index/dashboard/backlog не изменяются.
Dataset остаётся private; пользователь сделает его публичным при сдаче.

## Git

Попытка `git add` явного списка28 собственных файлов завершилась128:
песочница запретила создание shared worktree `index.lock` (`Operation not permitted`).
Коммит не создан; обход ограничений и повторный push не выполнялись.
[Квитанция](../checks/final-package-v2-git.json). Изменения остаются в worktree.

## Что осталось

Пользователь публикует dataset при сдаче и передаёт notebook/code/answers.
Отчёт будет обновлён отдельной задачей. Public новой посылки неизвестен.
Коммит нужно создать локально: текущая попытка заблокирована песочницей.

## Источники

- [final-variants](final-variants.md), [final-package-v11](final-package-v11.md), [b7-results](b7-results.md), [p1-results](p1-results.md);
  ранее учтённые в knowledge/bibliography.md эксперименты, новых работ нет.
- [Финальный пакет](../../submission/README.md), [CPU](../checks/final-package-v2-cpu.json).
