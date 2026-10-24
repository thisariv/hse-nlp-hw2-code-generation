---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-14
updated: 2026-10-15
sources:
  - nlp-hw02-d1-independent-review
  - nlp-hw02-d1-preparation
  - kaggle-l3-final-results
aliases: []
---

# D1: независимое ревью и запуск при ограниченной квоте

**Завершение15.10:** [d1-results](d1-results.md) — production COMPLETE за79,98мин,
1271 receipts/SHA проверены; A/B не рекомендованы, строгий C121 строка — резерв.
Ниже сохранено ревью до получения production output.

Исходный коммит `b590319`, ветка `codex/nlp-hw2-code-generation`; исходное дерево
чистое. Ревьюер прочитал [d1-plan](d1-plan.md), [p1-review](p1-review.md), [l3-review](l3-review.md), исходники,
старые receipts и frozen P1/v6. В ходе работы появились параллельные R1-правки;
они не входят в D1-коммит. Основной checkout не редактировался, credentials
не инспектировались; API использует штатную авторизацию CLI.

## Находки и исправления

| Риск | Изменение и регрессия |
|---|---|
| Старые5ч/две разные CE/обязательные friend Secrets несовместимы с новой квотой. | `build_d1.py`, `d1_worker.py`, `d1_session.py`: обе GPU загружают C-long2x для непересекающихся `ids[gpu::2]`; C-main загружается только перед greedy. Watchdog130мин, backend140мин; Secrets/live optional, по умолчанию false. `test_D1_build_budget_optional_secrets_owner_and_input_contracts`, статический gate и обе реальные T4-половины. |
| Timeout CE выбрасывал текущий вычисленный блок, консолидированные массивы появлялись только после полного прохода. | `d1_loss.py`: timeout сохраняет обработанную часть блока, float16 CE/uint16 counts и global-id массив с NaN для остальных. Full/partial receipts различаются. Merge проверяет disjoint ownership, длины и counts. `test_two_long_CE_shards_cover_global_rows_once_and_partial_batches_survive`; старый OOM/restart/row-id тест сохранён. |
| Timeout/ошибка генерации до конца волны теряли уже полученные GPU-ответы; incomplete references могли выдавать complete. | `d1_generate.py`: сохраняется обработанное подмножество и точные statuses; completeness требует завершённых execution и references. Generation guard отделён от CPU-tail deadline. `test_generation_timeout_retains_completed_batch_and_reference_pending_marks_partial`. SIGTERM становится graceful TimeoutError. |
| Старый анализатор требовал CE обеих моделей и mean-CE, хотя C-main CE теперь отсутствует. | `analyze-d1-labels.py`: отдельный контракт C-long2x-only, SHA модели, полное/частичное покрытие, top2% B по long CE; старые outputs поддержаны явно. `test_single_CE_analysis_keeps_original_reference_and_rejects_incomplete`; end-to-end A/B/C и SHA исходных orders сохранены. |
| Полные FP32 NLL-массивы и дублирование двух model CE не нужны. | Production сохраняет compact CE float16, lengths uint16, row id uint32 и JSON aggregate NLL. Веса/кеш остаются в input/temp; полных моделей в working нет. Compact-output assertion в тесте partial CE и независимом GPU audit. |
| Аварийный exit нужно проверить на том же working mount. | `watchdog_probe` запускает настоящий Torch-free Watchdog в дочернем процессе, проверяет forced exit0 и ранее записанный durable batch. CPU-регрессия и GPU v2 сохраняют probe receipts в output. Watchdog вооружён до Torch/install/data. |

13 CPU-регрессий пройдены, включая обычный training CE с разными длинами/PAD/EOS,
OOM/restart, ties/bin quotas, numeric/isclose/duplicate rules, end-to-end original-pair
orders, immutable snapshot/tamper, credential isolation, настоящий watchdog и
frozen v6 decoder/service. На реальных L3 обеих моделей повторены16 строк CE
против **отдельного training-model forward** и ordinary flattened CE, а также
по2 greedy/reference executions; все проверки прошли. Frozen training/v6 modules
не изменены. Runtime исходники одинаковы у смоука v2 и production, параметры
бюджета и account/input refs отличаются.

## CE, отбор, reference и исполнение

`exp26_order.collate` формирует decoder BOS+content и labels content+EOS.
В source сохраняются исходные BOS/EOS; model padding/causal masks взяты из
training-compatible `final_core`. NLL считается FP32, PAD=0 игнорируется,
BOS не предсказывается, EOS входит в count; row CE — NLL/count. Общий CE в
receipt — **token-weighted**, а ranking использует **row-normalized** float16.
CUDA smoke сравнивает с обычным flattened CE; градиентов/optimizer/train нет.
Побитное совпадение FP16 eval с историческим FP32+AMP train не обещается;
проверены objective/labels/masks и real CPU training-model parity.
Bucketing переставляет только batches, сохранение/merge всегда по global row id.

Full train имеет9 725 737 строк; halves4 862 869 и4 862 868. Top4% отдельно в
compact target-content bins `<53`, `[53,79)`, `[79,105)`, `≥105`: квоты212296,
116650,44545,15538; максимум389029. Ties выбирают меньший global row id.
Budget одинаково масштабирует квоты и берёт ranked prefix каждого bin, без
повторной генерации пилотов. Это compact BPE с mode, без EOS; точные original
60/90/120 offsets недоступны. Высокий CE может означать трудный правильный пример.

Production seed содержит1 250 497 finite raw train-reference, SHA
`6fd712d4…52d36`. Они joined по raw `(shard,row)` через pinned compact provenance,
а не по позиции filtered arrays; исходные noise-execution receipts и SHA сверены
с clean-cooldown census. Остальные ответы исполняются из **оригинального compact
target**, после обратимого restore. Test/heldout/fresh answers не используются;
model predictions никогда не становятся targets. Cached answers не переисполняются
все заново: возможные различия старого executor и v6 остаются ограничением,
source/status сохраняются. В smoke настоящие исходные reference сначала
исполняются для8 пилотных строк, затем проверяются cached и новый пути.

Greedy cap512, batch/backoff, EOS/restore/token_cap statuses сохраняются.
Frozen v6 CPU service:2 workers/model,4 всего, nice10, wall3с/CPU2с/768MiB,
AST/import guards, свежие numeric namespaces и pristine SymPy fork. Очередь
ограничена одной волной4096; refs и генерация перекрываются. Linux parent-death
signal завершает executor group при смерти GPU worker. Capped/nonfinite/ошибочные
и неисполненные строки не дают сигнал для удаления.

## Смоук и калибровка

Private thisarivson, API без Secrets; входы L3 live, compact-cache output/2 и
26b-inputs. Manifest `46ef881b…02be`, SHA всех360 cache payloads, original tokenizer
и обеих FP16 L3 weights проверяются перед spawn. Friend fallback проверяет тот же
manifest; owner-qualified и flat mounts разрешаются прежним reviewed L3 helper.

V1 COMPLETE, оба workers complete/exits[0,0], session168,718с. CE131072 rows;
4193 парных greedy rows. CE1359,90/1307,99 rows/s/GPU; generation pipeline
86,96/79,23 rows/s, GPU-only111,46/89,02. GPU CE parity passed; все49 artifact
receipts verified по bytes/SHA, output117997901 bytes, моделей0. Tiny pilot64/bin
дал слишком консервативную экстраполяцию из-за batch/outlier overhead;
вторая и последняя разрешённая итерация проверяет batch512/pilot256/wave4096.

V2 COMPLETE, оба workers complete/exits[0,0], session197,551с; full5244 paired
rows, OOM0. Все37 artifact receipts verified, output117985147 bytes, моделей0.
CE batch512:1224,77/1244,21 rows/s/GPU — медленнее v1, поэтому production
возвращает **CE batch256**. Generation batch512/wave4096: pipeline128,04/112,15,
GPU-only160,72/126,53 rows/s. Reference reused8/executed5236; missing0, EOS/статусы
и исходные row IDs проверены. Настоящий watchdog на Kaggle working mount:
forced exit0, durable17-row JSON и partial summaries скачаны и проверены.
Обе итерации значительно ниже720с backend; третьего smoke нет.

| Боевой параметр | Значение |
|---|---:|
| Полный C-long2x CE |9 725 737; GPU halves4 862 869/4 862 868|
| CE batch/block/target slots |256/8192/32768|
| Pool top4% по длине |389029, bin quotas212296/116650/44545/15538|
| Greedy batch/pilot/wave |512/256 per-bin/4096|
| Ожидаемый common greedy subset |≈79–111тыс.; фактический размер задаёт production pilot|
| Setup / CE / greedy deadlines |5/78/120мин от первого cell|
| CPU tail / worker stop / watchdog / backend |124/126/130/140мин|

Исходные55мин CE и390k greedy за45мин **не подтверждены**. Выбрана более быстрая
CE256: full прогноз3717,8с=61,96мин, с15% reserve4275,5с=71,26мин. Оставлено
78мин с setup≤5мин. Пилот по каждому bin использует более медленную модель и
safety1,6; нормировка по коротким highest-loss batches консервативнее общей
sustained generation скорости. [Калибровка](../checks/kaggle-d1-review-calibration.json):
типичный scenario даёт110629 строк, консервативный82365, CE-cap78662.
Это **предварительные размеры**, не фактическая production selection: она ещё
не опубликована. Если pilot быстрее, общий subset может быть больше; предел
389029. Сокращение выполняется пропорционально всем bin quotas. Полный CE
остаётся приоритетом; ожидание390k не выдаётся за гарантию.

Friend API free quota10790,313с=2:59:50, refresh17.10 в00:00UTC. Боевой private
[**friend1/nlp-hw02-kaggle-d1 v1**](https://www.kaggle.com/code/friend1/nlp-hw02-kaggle-d1/versions?versionNumber=1)
отправлен14.10 в20:53:10UTC; **RUNNING** подтверждён в
20:55:06UTC. Server metadata: T4×2/private/internet on,
Secrets/live false. Все runtime module SHA и port SHA совпадают с прошедшим v2;
сменены только account/inputs/CE batch/phase deadlines. Friend weights, 26b-inputs,
fallback и seed accessible/private; original cache output/2 pinned manifest.
Окончания run ревью не ждёт. Ожидаемый финиш по measured CE и уменьшенному
subset — около01:15–01:45МСК15.10; консервативный backend предел около
02:15МСК15.10
(23:15UTC14.10). Actual first-cell clock/VM startup может
сдвинуть внутренний watchdog ETA; серверный timeout8400с ограничивает сессию.


## Завершение и пределы проверки

Deadline/частичный CE и greedy публикуют сохранённые результаты; session может
штатно завершиться partial. При зависании watchdog запишет partial summaries,
уберёт descendants и выйдет0. Полное отключение VM/SIGKILL может потерять текущий
незаписанный batch/block; опубликованные immutable данные не удаляются. Secrets,
live insurance и upload для D1 не нужны. Kaggle output сохраняется на working;
локально скачиваются JSON/NPY/NPZ/log, без моделей. Диск ограничен2GiB.

Полный train CE не обещается одной короткой выборкой; при потере скорости
незавершённый run маркируется partial. Runtime pilot может только уменьшить
общий ranked subset. Анализ по умолчанию отвергает incomplete CE/paired generation;
`--allow-partial` сообщает coverage и оставляет все непроверенные train-пары.
Flags — наблюдаемая нижняя граница в high-loss популяции, не оценка доли шума.
Следующий шаг после окончания — скачать outputs, audit/анализ, ручной разбор
flagged/retained примеров; никакого обучения/переписывания reference здесь нет.

## Источники

- [d1-plan](d1-plan.md), [p1-review](p1-review.md), [l3-review](l3-review.md), [clean-cooldown-analysis](clean-cooldown-analysis.md);
  [runs.json](../kaggle/runs.json), собственные CPU/GPU receipts и независимый аудит.
- [Kaggle CLI kernels API](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md):
  private metadata, T4×2 accelerator, timeout и version-specific status/output.
- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html):
  unreduced loss, ignored PAD и token-weighted reduction. Все выводы ревью —
  собственная интерпретация кода и измерений, без новых внешних результатов.

## Коммит

Попытка D1-only `git add` получила exit128: sandbox запрещает создание shared
worktree `index.lock` (`Operation not permitted`). Индекс не изменён, ограничения
не обходились. Все D1-правки и receipts сохранены. Параллельные R1 navigation
paragraphs не включены в [отдельный D1 patch](../checks/kaggle-d1-review-navigation.patch);
[Git receipt](../checks/kaggle-d1-review-git.json) содержит явный список D1 paths
и сообщение для повторной попытки из обычного терминала. Patch применять только
`git apply --cached`, чтобы сохранить чужие working-tree paragraphs.
