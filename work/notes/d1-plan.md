---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-14
updated: 2026-10-15
sources:
  - nlp-hw02-d1-preparation
  - kaggle-p1-final-results
  - kaggle-l3-final-results
  - nlp-clean-cooldown
aliases: []
---

# D1: вероятно неверные train-reference

**Результаты15.10:** [d1-results](d1-results.md) — COMPLETE, полный CE9 725 737,
парный greedy135 324; A38 644, но совпадение с модой лишь121/886=13,66%.
Ручная проверка30 и оба типа R1 orders готовы. План ниже сохранён как протокол.

**Пересмотрено независимым ревьюером под квоту 14.10.** Боевой D1 на аккаунте друга:
C-long2x CE на всех9 725 737 строках двумя GPU, затем общий stratified top≤4% и
C-main/C-long2x greedy с CPU overlap. C-main CE здесь отсутствует. Secrets/live
не обязательны и выключены; компактный output остаётся в `/kaggle/working`.
Watchdog130мин, backend140мин: минимум10мин до предела. Замеры и актуальный запуск — [d1-review](d1-review.md).

## Гипотеза и границы

После длинного cooldown loss truncation дал только **+0,44 п.п.** greedy4k; CI включает 0, заранее заданный eval gate не пройден ([p1-results](p1-results.md)). Это мотивирует диагностику качества разметки, но не доказывает, что плато вызвано шумом.

[clean-cooldown-analysis](clean-cooldown-analysis.md): среди повторных finite train-строк около **9,94%** находятся вне одной выбранной моды. От всего train это **1,20%**, причём число включает равные моды. Это не доля заведомо неверных программ. Singleton и согласованно неверные решения остаются невидимыми для такой проверки.

По сообщению пользователя 14.10 преподаватель разрешил **фильтрацию исходных train-пар**, включая помощь сторонних LLM. D1 использует только собственные модели. Условие в `task/` не меняется; модельные ответы служат диагностикой, **новые таргеты не создаются**, reference не исправляются. Обучение в D1 отсутствует. Heldout/test/fresh answers для фильтра не читаются.

## Полный проход CE

- Исходные **9 725 737** global compact row id, 60 шардов; manifest `46ef881b…02be`.
- Обе GPU — **C-long2x**; строки `scored_ids[gpu::2]` не пересекаются и покрывают весь train. На greedy GPU0 переключается на **C-main**, GPU1 остаётся **C-long2x**. Только финальные L3 FP16 `l2-62280`: SHA `8c11b103…b886` / `d10bf001…5057`, strict model-only load. Stable weights и P1 не используются.
- Teacher forcing, `eval`, `inference_mode`, без smoothing/градиентов. Source и labels формируются неизменённым `exp26_order.collate`.
- В каждом блоке 8192 исходных строк — bucketing по source/target длинам; начальный CE batch256, максимум32 768 target slots и65 536 общих slots. Длинные батчи уменьшаются; OOM повторяет те же строки с меньшим batch. CE считается FP32 порциями4096 токенов, без полного FP32 logits workspace.

$$
NLL_i^{(m)}=-\sum_{t=1}^{T_i}\log p_m(y_{i,t}\mid x_i,y_{i,<t}),\qquad
\ell_i^{(m)}=NLL_i^{(m)}/T_i.
$$

$i$ — исходный compact row id; $m$ — модель; $x_i$ — вопрос; $y_{i,t}$ — очередной исходный target-токен. $T_i$ включает **EOS**, исключает BOS/PAD; mode token входит в content. $NLL_i$ вычисляется FP32; в компактных блоках сохраняются row id uint32,
$T_i$ uint16 и $\ell_i$ float16. Суммарные NLL/token counts и token-weighted CE
сохраняются в JSON без повторного массива NLL. Итоговый `C-long2x/loss/ce.npy`
индексируется global row id, занимает19,5MB; для нерассчитанных строк NaN.
Shard CE/length, immutable NPZ и receipts позволяют проверить покрытие и восстановить
завершённые блоки. Timeout сохраняет также последнюю обработанную часть блока.
Частичная фаза имеет `partial.receipt.json`, а не `complete.receipt.json`.

## Подмножество и исполнение

Score — **сохранённый FP16** $\ell_i$ C-long2x. Независимо в каждом compact-content bin `<53`, `53–79`, `79–105`, `≥105` выбирается top4%; при равных score побеждает меньший row id. Это **compact BPE с mode, без EOS**, а не original60/90/120.

| Compact bin | Полный train | Верхние4%, максимум |
|---|---:|---:|
| <53 |5 307 410|212 296|
| 53–79 |2 916 253|116 650|
| 79–105 |1 113 633|44 545|
| ≥105 |388 441|15 538|
| Всего |9 725 737|**389 029**|

Первые256 ranked строк каждого bin — пилот обеих моделей. В замер входят GPU generation, restore и перекрывающееся исполнение. Для каждого bin берётся более медленная модель; оценка всего подмножества умножается на **1,6**. Если времени мало, все квоты пропорционально уменьшаются, состав остаётся префиксом ранжирования внутри bin. Подмножество фиксируется один раз, одинаково для обеих моделей; пилотные строки сохраняются и не генерируются повторно. Его точный размер до GPU-пилота неизвестен.

Frozen **v6** берётся из ранее собранного v6 notebook, включая отдельный executor-service; текущие `src/final_*` и v7 не меняются. Greedy, batch512, cap512, cross-KV sharing, async EOS32, compaction, без logprob. После каждого batch программы сразу уходят на CPU. Волна≤4096 строк; по2 nice10 execution workers на модель, **4 суммарно**. Очередь и кэш ограничены одной волной, затем executor закрывается; нет накопления сотен тысяч исходников в RAM.

Для каждой строки сохраняются row id, ответ, точный execution status, EOS/token-cap/restore status и длина генерации. C-main дополнительно сохраняет reference value/status/source. Все ответы float64, ошибочные — NaN в NPZ, в JSON — только конечные числа или null. CPU-исполнитель v6 ограничивает программу wall3с/CPU2с/768MiB и изолирует SymPy. `token_cap`, ошибки и неисполненные строки не создают сигнал для удаления.

### Где найдены train-reference

`research/outputs/data-ceiling/noise-values.npz`, `normal-duplicate-members.npz` и60 `noise-execution-*.json` содержат ответы raw-дублей. Их SHA сверены с frozen `results/clean-cooldown/train-analysis.json`. Пересечение выполнено по raw `(shard,row)` через `provenance-rows.npy`; после finite/cache exclusions пересчитаны группы, конфликты и строгие моды.

Получены **1 250 497** готовых numeric reference, включая ставшие singleton строки; duplicate rows1 177 874, conflicting251 745 — совпадают с clean-cooldown. Seed **29 642 620 bytes**, SHA `6fd712d4…52d36`, включает compact source/target lengths, sparse reference/group metadata и protected exclusion8545870. Подготовка заняла18,6с CPU. Остальные reference выбранного подмножества исполняются на GPU0 в том же CPU-service, одновременно с генерацией.

Raw Parquet и original target offsets уже отсутствуют локально. Raw row counts переиспользованы из полной переписи, сверены по pinned shard SHA и размерам сохранённых hash tables; исходные программы заново не читались. **Original BPE lengths не восстановлены**: отчёт явно помечает их недоступность, выдаёт compact bins и связь с ранее проверенными clean-cooldown группами. При наличии исходных offsets скрипт проверит SHA и добавит точные original bins.

## Бюджет и завершение

Квота друга3,00ч, refresh17.10 в00:00UTC; своего аккаунта около1,5ч.
Смоук — thisarivson, ≤720с backend/660с watchdog, максимум две итерации.

| Фаза | Deadline от первого cell |
|---|---:|
| Проверки/зависимости/SHA |5мин|
| C-long2x CE двумя GPU |78мин (по замеру≈62мин, с15% запасом≈71мин)|
| Ranking/pilot/greedy с CPU overlap |120мин (≥42мин после CE cap)|
| Хвост исполнения/запись |124мин|
| Worker stop/reap |126мин|
| Watchdog |130мин|
| Backend timeout |140мин|

Это предельные окна; точный отбор ограничивается измеренной пропускной способностью
и `choose_budget` со safety1,6 по более медленной модели в каждом bin. CE имеет
приоритет; его полный target не уменьшается. При timeout в CE ranking идёт только
по явно отмеченному scored-подмножеству. Непроверенные строки остаются в train.
Уменьшенный greedy-состав остаётся префиксом CE-ranking каждого bin и един для
обеих моделей. Завершённые пилоты и волны не генерируются повторно.

Watchdog вооружён до Torch/install/data. D1 требует две T4, но `require_secrets=False`
и `enable_live_sync=False`; owner проверяет private API metadata ревьюер, credential
файлы не инспектируются. Workers не получают auth environment; CPU executor
работает отдельными процессами. Веса и кеш читаются из input/temp, в output
полных моделей нет. Бюджет output≤2GiB, ожидаемый порядок сотенMB.

Timeout генерации сохраняет уже декодированные строки и их текущие execution
statuses; необработанные reference тоже делают волну partial. SIGTERM переводится
в graceful TimeoutError. Неожиданный SIGKILL может потерять текущий незаписанный
батч/блок, но уже опубликованные immutable данные остаются. Watchdog записывает
partial session-summary и завершает процесс кодом0. Выход partial штатно допустим;
анализ по умолчанию требует полного CE и парного покрытия, `--allow-partial`
оставляет непроверенные строки и явно сообщает coverage.

## Анализ и фильтры

Численное равенство — симметричный `math.isclose(rel_tol=abs_tol=1e-6)`; мода дублей использует прежние anchored representatives. Для A нужны оба finite numeric результата, завершённые EOS и finite numeric reference; **каждая** модель должна расходиться с reference.

| Правило удаления | Условие |
|---|---|
| A |Обе модели согласны между собой, обе против reference|
| B |A и C-long2x CE в top**2%** своего длинового bin полного scored train|
| C |B и исходный reference — меньшинство в группе≥3, единственная мода≥2/3, обе модели совпадают с этой модой|

Top2% для B заранее строже top≤4% фазы генерации: A и B не становятся одним правилом по определению. Если бюджет уменьшит генерацию ниже2%, они могут эмпирически совпасть; это будет видно по counts. C исключает ties и группы1:1.

`research/analyze-d1-labels.py` считает доли от выбранного подмножества и от совместно исполненных finite/EOS строк, распределения длин до/после фильтра, статусы и conditional rates для singleton/duplicate/conflicting/minority, в том числе внутри bins. Доля flagged от всего train — только **наблюдаемая нижняя граница**, не оценка общей доли шума: отбор сделан по высокому loss.

Для control/A/B/C создаются `order.npy` uint32 с исходными compact indices, `order.json`, `split.json`, SHA и evidence hash. Каждая оставшаяся исходная строка ровно один раз: seed2026101404 → shuffle → bucketing4096 → shuffle полных batches128. `rows` остаётся9 725 737, `slots` — размер фильтра. У всех четырёх исключён protected row8545870; membership, uniqueness и неизменённый `l2_sampler.load_order` проверяются. Непроверенные строки остаются. Model programs/answers в train не добавляются.

## Риски и последующий опыт

- **Трудные, но правильные примеры** часто имеют высокий loss. Даже стратификация длины не контролирует семантическую сложность; фильтр может упростить train и повредить длинным задачам.
- Две C коррелированы: общий tokenizer, данные и архитектура; согласованная ошибка возможна. Модель могла запомнить шум либо предпочесть частый неверный шаблон. Мода дублей тоже бывает неверной.
- CE измеряет likelihood конкретной программы, а не правильность математики; корректные альтернативные программы могут иметь разный CE.
- Программы могут совпасть по одному численному ответу случайно. `isclose` нетранзитивен, anchored clustering — соглашение. Старые reference receipts и v6 могут иметь редкие различия исполнения; statuses и source сохраняются.
- FP16 CE имеет округление и ties; precision/tie rule фиксированы. FP16 greedy чувствителен к форме batch и compaction; точный набор ответов требуется сохранить, а не обещать побитное совпадение на другом batch.
- Нет рандомизированного низкого-loss контроля; D1 не докажет причинный эффект шума и не измерит улучшение модели. До решения об обучении — ручной разбор стратифицированных flagged/retained примеров, затем отдельный matched-budget A/B исходных пар, с длиновыми метриками и заранее заданным gate. Сам D1 никакой фильтр в обучение не включает.

## Воспроизведение и проверки

Из `work/`, в CPU Python с NumPy/Torch/tokenizers/nbformat:

```bash
python research/analyze-d1-labels.py prepare
python kaggle/build_d1.py
python kaggle/build_d1.py --gpu-smoke
python kaggle/d1_tests.py
python kaggle/d1_cpu_check.py
python kaggle/d1_static_check.py
```

Friend API — только `KAGGLE_CONFIG_DIR=~/.kaggle-friend python kaggle/d1_prepare_api.py`; скрипт не открывает credential files сам, не выводит значения и не вызывает kernels API. Он создаёт private seed, повторно не перезаписывает существующий live. Файлы содержимого в ignored `kaggle/staging/d1-live/`; tracked pin — [d1-seed.json](../kaggle/d1-seed.json).

После скачивания live flattened files:

```bash
python kaggle/src/d1_live.py restore --stage PATH_TO_LIVE --destination RESTORED
python research/analyze-d1-labels.py analyze --input RESTORED --output results/d1-labels
```

Исходные8 D1 CPU tests (теперь13 после ревью): per-row/обычный CE, PAD/EOS, OOM+restart, ranked ties/budget, статусы/численное равенство, синтетический end-to-end анализ и повторяемые SHA orders, snapshot restore/tamper, worker isolation/Secrets/watchdog, frozen v6 decoder/executor. Общий CPU discover до добавления isolation test:89tests passed. На реальных compact rows и обеих L3: по16 строк CE против отдельного training-model forward и обычного flattened CE; по2 greedy+execution, cached и newly executed reference. Ошибки модели остаются execution statuses, а не ошибками инфраструктуры.

Smoke notebook:131072 равномерно выбранных train-строк для представительного
CE-mix, две непересекающиеся половины и parity обычного CE. Top4%/bin,
пилоты256/bin, обе модели, greedy cap512 и reference reuse/новое исполнение.
Входы: свой L3 live, pinned compact-cache output/2, 26b-inputs. Результаты GPU,
калибровка и ссылка боевого запуска — [d1-review](d1-review.md).

## Источники

- [p1-results](p1-results.md), [l3-results](l3-results.md), [clean-cooldown-analysis](clean-cooldown-analysis.md), [data-research-ceiling](data-research-ceiling.md) — собственные измерения и ограничения вывода о шуме.
- [Условие](../../task/hw2-code-generation.md) и разрешение фильтрации в сообщении пользователя14.10.2026; исходное условие сохранено неизменным.
- [CPU-анализ](../research/analyze-d1-labels.py), [worker](../kaggle/d1_worker.py), [сборщик](../kaggle/build_d1.py), [статика](../checks/kaggle-d1-static.json), [реальные CPU-проверки](../checks/kaggle-d1-real-cpu.json), [private live receipt](../checks/kaggle-d1-live-created.json).
- Frozen L3/P1 supervision/live/Secrets и v6 decoder/executor — исходники в embedded `SOURCES` D1 notebook, SHA в `bundles/d1/source-identity.json`. Результаты и план внесены в библиографию.
