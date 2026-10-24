---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-20
updated: 2026-10-20
sources:
  - final-inference.md
  - pytorch211-cuda-memory
  - ../checks/final-package-v11-cpu.json
  - ../checks/final-package-v11-dataset.json
  - ../checks/final-package-v11-launch.json
aliases: []
---

# Чистая сдача v11 с запасом памяти

> [!success] Текущее решение
> Budget640 принят: test137,36с, tune1551/2048, max reserved10,99GB.
> Точная конфигурация и GPU-результаты — в разделе «Budget640» ниже.

## Исходная упаковка budget768

Пакет [submission](../../submission/README.md) продолжает v11:
greedy C-long2x B6, C/P1-B/W B6, T0.7/seed270042,
**1/1/1 → 1/8/8**. Изменены только формы батчей: sequence budget
**768 вместо 1024**, включая greedy. Политика и guards находятся в одном
`submission/code/hw2_codegen/config.json`; архитектура C/W — в `model.py`
и `model_spec` checkpoints. Исполняемый пакет не содержит SOURCES-блобов.

768 уменьшает предвыделенный self-KV W с 9 до 6,75 GiB при cap512.
Это расчёт размера одного кеша, не прогноз полного пика: cross-KV,
временные буферы compaction и CUDA allocator требуют отдельного измерения.
Новые формы меняют FP16/sampling; public79,37% исходного v11 нельзя
автоматически приписывать новой посылке. K и веса по новому tune не меняются.

## CPU и данные

Повторная проверка: **18 tests /18,771 с**, fresh-kernel notebook Run All
на восьми synthetic rows /3,118 с. Frozen notebook v11: C/P1/W-QK,
8 строк, одинаковые budget14/21 — все candidates/logprobs/restores/answers
exact. Replay L3: 49152 restores, 2048 winners,1535 matches — exact.
Дополнительные регрессии: synthetic OOM после расходования RNG возвращает
те же candidates, что прямой меньший batch; незаконченный greedy вызывает
ошибку и не создаёт answers. CPU не проверяет CUDA FP16 тождественность.

Private dataset [thisarivson/nlp-hw02-final-weights](https://www.kaggle.com/datasets/thisarivson/nlp-hw02-final-weights),
**v2**: три model-only FP16, tokenizer, читаемый code/, test1919,
фиксированные tune2048/reference и manifest. Все **26 файлов** скачаны
обратно с явным номером версии и сверены по bytes/SHA.
B6/P1 взяты из уже локально закреплённых исходных output, без новых API
действий в аккаунте друга. V1/V2 сохраняются; ничего не удалялось.

Дополнительно настоящий FP16-export C/P1-B/W загружен в FP32 CPU: три test-вопроса, cap64/budget14, все candidates/logprobs/answers exact с frozen v11. SHA каждого checkpoint проверен. Это отдельный optional `v11_real_weights_check.py`, не GPU-оценка качества.

## Единственный GPU-push

20.10.2026 **08:07:46 UTC** — private
[thisarivson/nlp-hw02-final-package-v11, v1](https://www.kaggle.com/code/thisarivson/nlp-hw02-final-package-v11/versions?versionNumber=1).
Default SDK authentication, содержимое credentials не читалось.
Только thisarivson; аккаунт друга не использовался.
Сервер подтвердил все code cells exact, Internet off, T4 и единственный
dataset final-weights. До push свободно 28591,49 с квоты. Backend cap720 с;
GPU0 видна как единственная карта, вторая выделенная T4 не вычисляет.

Run All: **один cold test1919 + отдельный свежий tune2048**, без обучения.
Cold timer включает subprocess/imports/SHA/read/load/warmup/decode,
restore/exec/vote/reexecution и закрытие answers. NVML device-wide used
опрашивается каждые20 мс; Torch peak allocated/reserved записываются отдельно.
Критерии: test≤150 с, peak≈≤14 GB, tune1547±1 п.п., полный ансамбль без
OOM/guards/partial/fallback и без mismatch переисполнения.

Статус хранится в [observations](../checks/final-package-v11-observations.json).
При QUEUED>10 мин либо RUNNING без output>12 мин повторного push нет:
ожидание до60 мин. Статус COMPLETE сам по себе не считается проверкой результата.

## Результат T4 и аудит

Version1 COMPLETE в08:26:14UTC, перед этим QUEUED>10мин был записан.
Own push1, повторов/отмен0. **Test134,378853с; tune103,109895с,
1542/2048=75,292969%.** Runtime Torch2.11.0+cu128/CUDA12.8/TeslaT4.
Обе стадии complete, OOM/guards/partial/fallback/mismatch0, все1919/2048
winners независимо переисполнены. Полный tune replay2048/2048 exact.

**Память не прошла цель:** test NVML15,184626GB /tune15,325135GB из15,636038GB.
Torch allocated12,104001/11,035489GB; reserved14,547943/14,688453GB.
Отличие reserved от allocated совместимо с caching/fragmentation, но конкретная
причина не изолирована. Одного уменьшения sequence budget до768 недостаточно.
`audit_passed:true` означает целостность результата; `targets_passed:false` —
провал целевого memory gate. Оба поля сохраняются раздельно.

Ответ [final-package-v11.jsonl](../submissions/final-package-v11.jsonl),1919строк,
SHA2202c645b5ee94c0a6aad4d866e8de6c49ce87eda6b78dc60d886b921714df16.
С прежним v11: **298 code /185 numeric** различий, оба файла независимо
исполнены, нечисловых0. [Сравнение](../checks/final-package-v11-comparison.json).
Tune −5 к1547 (−0,244п.п.), политика после результата не подбиралась.
[Полный результат](../results/final-package-v11.json).

Исполненный `__notebook__.ipynb` скачан через отдельный version1 endpoint:
code cells exact, execution counts1–4, ошибок0. Настоящие outputs скопированы
в `submission/inference.ipynb`. Обычный CLI pull возвращает source без outputs;
это нельзя путать с исполненным notebook.

## Исправление памяти — отдельный кандидат

[Patch](../checks/final-package-v11-memory-candidate.patch),
[CPU receipt](../checks/final-package-v11-memory-candidate.json): budget640,
`expandable_segments:True,garbage_collection_threshold:0.8` перед Torch import,
освобождение свободного CUDA cache между batches.18 CPU-регрессий passed.
Кандидат лежит отдельно в ignored staging; проверенный submission/dataset v2
не заменён. **GPU0/push0** у кандидата: время≤150с, память≤14GB и новый tune
не установлены. Новый прогон требует явного разрешения сверх исходного
пользовательского лимита «ровно один push»; вопрос направлен пользователю.
Мотивация allocator — [официальная PyTorch2.11 документация](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-memory-management).

## Аудит после output

Из корня репозитория, Python с Kaggle SDK, затем Python с зависимостями сдачи:

```sh
python work/kaggle/final_package_api.py observe
python work/kaggle/final_package_api.py output
python work/kaggle/final_package_results.py
```

Аудитор сверяет источник/версию/порядок/SHA, replay tune-journal и quality,
независимо исполняет обе посылки и сравнивает с `final-ensemble-v11.jsonl`.
Ответ сохраняется как `work/submissions/final-package-v11.jsonl`; реальные
outputs notebook копируются только из exact-code исполненного notebook.
В бота ничего автоматически не отправляется.

## Отчёт и границы

Отчёт (`../../submission/report/report-draft.md`, нет в репозитории): задача/данные, C/W/QK/z-loss,
WSD/B6, отрицательные эксперименты, decoder/adaptive/E1, publicv7–v11,
обязательные роли Claude/Codex/reviewers и ошибки. Четыре графика из results;
`submission/report/build_pdf.py` собирает PDF через Pandoc/XeLaTeX.
Многочасовое обучение не перезапускалось. Для проверки владелец должен выдать
доступ к private weights либо передать те же файлы отдельно. Публикации нет.
E2/B7 и progress/index/dashboard/backlog не изменялись.

Попытка коммита: `git add` явного списка 32 собственных файлов завершился128:
песочница запретила shared `index.lock` (`Operation not permitted`).
Коммит не создан, обход ограничений не выполнялся; [квитанция](../checks/final-package-v11-git.json).
Финальная попытка после GPU-аудита: явный список43 собственных файлов,
та же ошибка index.lock; staged index пуст.

## Budget640: разрешённый проверочный прогон

Пользователь отдельно разрешил один GPU-прогон budget640. Применён сохранённый
patch: оба sequence budget640, allocator
`expandable_segments:True,garbage_collection_threshold:0.8` до Torch import,
`empty_cache()` между batches. Веса, K1/K2, seed/temperature, guards и
голосование не изменены. В notebook memory gate проверяет **reserved≤14e9 bytes**.

CPU:18 tests/17,510с, notebook fresh-kernel dry-run8/2,178с, AST/JSON/diff checks
passed; frozen/L3 parity сохранена. [Квитанция](../checks/final-package-v11-budget640-cpu.json).
Private final-weights **v3**, все26 файлов bytes/SHA roundtrip exact.
20.10.2026 **08:49:27 UTC**: единственный private
[budget640 push, v1](https://www.kaggle.com/code/thisarivson/nlp-hw02-final-package-v11-budget640/versions?versionNumber=1).
Default SDK auth thisarivson, credentials не читались/не выводились.
Серверные code cells exact, T4/Internet off, источник только final-weights;
[launch](../checks/final-package-v11-budget640-launch.json),
[server parity](../checks/final-package-v11-budget640-remote.json).
Повторного push/отмен/удалений/смены видимости0; B7 не изменяется.
### Результат и решение: принять budget640

COMPLETE впервые подтверждён **08:54:51 UTC**. Никаких повторных push/отмен;
QUEUED>10мин/RUNNING без output>12мин не потребовались.19 output-файлов и
отдельный исполненный notebook скачаны с привязкой к серверной v1;
code cells exact до и после скачивания. Torch2.11.0+cu128/CUDA12.8/TeslaT4.

| Split | Cold, с | Allocated, GB | Reserved, GB | NVML used, GB |
|---|---:|---:|---:|---:|
| Test1919 | 137,361258 | 10,856472 | **10,986979** | 11,623662 |
| Tune2048 | 106,998326 | 9,533960 | **9,621733** | 10,258416 |

GB=10⁹ bytes; device total15,636038GB. **Tune1551/2048=75,732422%**,
Δ+4/+0,1953125п.п. к1547. Все критерии пройдены: max reserved≤14GB,
test≤150с, tune±1п.п. OOM/guards/partial/fallback/no-numeric/mismatch0;
1919/2048 winners независимо переисполнены. Полный tune-journal replay2048 exact,
SHA трёх моделей/входной порядок/параметры exact. Веса и selector не подбирались.
Память существенно ниже прежнего budget768, но вклад budget/allocator/cache
по отдельности этим прогоном не изолирован.

[GPU-аудит](../results/final-package-v11-budget640.json),
[ответы](../submissions/final-package-v11-b640.jsonl):1919 code-only строк,
SHA `6310b5892395e9f4c076dd7c559815a934ea42dec2c713a7973693608bf540cd`.
От исходного `final-ensemble-v11.jsonl` отличаются **302 code /193 numeric**
ответа: обе посылки независимо исполнены, нечисловых0;
[индексы и SHA](../checks/final-package-v11-budget640-comparison.json).
В бота ничего не отправлялось.

**Budget640 принят**, откат к1024 не нужен. В submission остаётся точный
проверенный config/runner/pipeline; в `inference.ipynb` перенесены реальные
outputs этого прогона. README содержит время/память, Kaggle Add Input→dataset→
Run All и Colab/Kaggle API. Dataset остаётся private; пользователь сделает
его публичным при сдаче. Старые версии и результаты сохранены, B7 не изменён.

Воспроизведение аудита после скачивания:

```sh
python work/kaggle/final_budget640_api.py observe
python work/kaggle/final_budget640_api.py output
python work/kaggle/final_budget640_results.py
```

Команды read-only для Kaggle; `output` скачивает файлы локально.
Новый push заблокирован существующей квитанцией launch.

Попытка коммита25 собственных файлов: `git add` завершился128,
`Operation not permitted` на shared worktree `index.lock`. Коммит не создан,
staged index пуст, обхода ограничений нет;
[квитанция](../checks/final-package-v11-budget640-git.json).

## Источники

- [final-inference — V11](final-inference.md#v11) — исходные веса, политика и прежние измерения.
- [CPU](../checks/final-package-v11-cpu.json), [private dataset/SHA](../checks/final-package-v11-dataset.json),
  [push](../checks/final-package-v11-launch.json), [server parity](../checks/final-package-v11-remote.json).
- [b6-results](b6-results.md), [e1-ensembles](e1-ensembles.md), [p1-results](p1-results.md) — происхождение финала.
