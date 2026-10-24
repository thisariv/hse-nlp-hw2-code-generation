---
type: experiment
course:
  - nlp
status: draft
understanding: 3
created: 2026-10-12
updated: 2026-10-12
sources:
  - kaggle-s3
  - kaggle-s2-v2
  - kaggle-s1-v1
  - hoffmann2022-chinchilla
  - kaggle-l1
aliases: []
---

# ДЗ2: L1 — длинное обучение C и D

> [!warning] Версия после ревью
> [Независимое ревью и CUDA smoke](l1-review.md): исходный D probe на64 получает OOM; исправленный код выбирает32, C64. Опубликованный L1 v1 остался старым. Перед UI запуском импортировать пересобранный локальный ноутбук; боевой L1 во время ревью не запускался.

Решение руководителя проекта от2026-10-12: №30c с mode token принят как стандарт по раннему сравнению1876/4096 против1888/4096, p0.638, +15.6% пар/с. Прежний frozen gate S3 на1.92M/equal-time остался незавершённым; это новое управленческое решение, не новый замер. Плейсхолдеры и B reranking отвергнуты. Основное направление — масштаб модели и больше обучения. Ограничения на GPU сняты; несколько сессий примерно по11ч T4×2 с продолжением. Оценка≈100M по Chinchilla для≈30чT4 — предоставленная руководителем эвристика, не измеренный аппаратный оптимум seq2seq.

## Данные и CPU-кеш

Ровно9 725 737 finite строк:26a0–41 (7 078 471) +26c42–59 (2 647 266). Length/hash/protected exclusions, row identities и повторяющиеся тексты неизменны. Новый таргет строится **тем же** `s3_representation.compact`/`tokenizer(...,30)`:8189 content BPE IDs,3 mode tokens, отдельные PAD/BOS/EOS → vocab8195. Source и provenance копируются byte-for-byte. Target int16, offsets int64 `.npy`, mmap. Manifest содержитSHA всех массивов, исходных manifest, tokenizer и representation code.

Новая глобальная перестановка seed**2026101201**, bucket4096, batch128: сортировка по source+target length внутри блока, shuffle128-row batches; остаток сохранён. Бие́кция проверяется на всём корпусе. Cursor продолжает циклический обход без потери последнего неполного batch и без повторов до исчерпания корпуса.

CPU round-trip: случайные834 строки каждого из60шардов =50 040 (если размеры достаточны); BPE encode/decode → restore → новое bounded execution original/restored. Нужно0 расхождений и все reference numeric. Отдельная full output проверкаSHA/offsets/order после COMPLETE обязательна до L1 push. Output CPU kernel подключается как pinned kernel source; кеш не строится на GPU.

## Модели и рецепт

| Модель | GPU | Encoder+decoder | d / h | SwiGLU | Peak LR |
|---|---:|---|---|---:|---:|
| C (53 646 403) | 0 | 6+6 | 512 /8 |1368|0.0015|
| D (103 902 979) | 1 | 8+8 | 640 /10 |1712|0.0012|

Архитектура `exp26_core.SimpleTransformer`, Pre-LN, learned positions2048, shared token embedding encoder/decoder, **untied output head**. Точное число параметров печатается каждым worker и фиксируется в summary. Обучение с нуля seed42. Token init std$d^{-1/2}$ при последующем×$\sqrt d$, dropout0, AMPFP16+GradScaler(init1024), fusedAdamW, wd0.01, clip1.

Эффективный batch128. В первые минуты полный forward/backward/Adam на длиннейших cached rows:128, при OOM/недостаточном запасе памяти96→64→32→16→8→4→2→1 с накоплением до128. T4 smoke выбрал C64/D32; поздний OOM имеет rollback/retry. Probe не меняет обучаемую траекторию/RNG/cursor. CE accumulation нормируется суммой непустых target tokens всей128, включаяEOS. Warmup1000attempted updates, затем **постоянный LR до конца L1/L2/L3…**. Финальное охлаждение основной ветки здесь отсутствует.

## Оценочные ветки и сравнение

Общие anchor marks —9000/18000/27000/36000с **чистого main training**; setup, memory probe, checkpoint I/O, evaluation и side cooldown исключены. СинхронизацияCUDA входит в измерение update. Все фактические main times/отклонения и дополнительно вся стоимость веток публикуются. Допуск общего point: max5с/0.5%target. Сравнение по этой оси фиксирует одинаковое полезное main-время; all-compute cost записывается отдельно и может различаться.

На каждой mark: сохранить full main anchor → cooldown≈10%накопленных main updates, с оценкой длины по свежему throughput и потолком1500с/25мин → evaluate → полный rollback model/AdamW/scaler/RNG/cursor. Если cooldown не завершён, point явно incomplete и не участвует в выборе. На начало ветки нужен резерв cooldown+15минeval+3минIO; поэтому поздняя точка может отсутствовать. Main продолжает учиться до штатного stop, следующий общий grid point выбирается при resume; пропущенный point не переименовывается в более поздний.

Каждая завершённая точка: model-only FP16; greedy FP16 `exp27b_fast` с4D SDPA, восстановление программ, bounded exec heldout4096 и только991finite old rows, CE компактного таргета (FP16 weights/forward, FP32 softmax/sum), original22 reference-length bins. Test1919: новый полный timer от чтения test/загрузки model/tokenizer до greedy/restore/exec/закрытия answers; gold нет, accuracy не вычисляется. Fresh не оценивается. Оценочная модель отдельная: training parameters не меняют dtype под `inference_mode`.

**Выбор L2:** последняя завершённая общая time mark; лучший exec heldout4k. Если D отстаёт отC **меньше2п.п.** и её последний наклон выше, продолжать обе. Наклон$p=(a_2-a_1)/\log_2(t_2/t_1)$ в п.п. на удвоение времени, по последним двум общим marks;$a$ — exec accuracy,$t$ — target main seconds. Без двух marks условие наклона не вычисляется, продолжается лучший last point. Exact tie → C. Old991 и длины — диагностика, не дополнительный tuning criterion.

## Resume и защита результатов

Штатный бюджет worker40680с =11ч18мин отпервойячейки, завершение train с120срезервом; watchdog TERM→20с→KILL, аварийный `os._exit`42300с =11ч45мин. Backend ceiling43200с/12ч. Watchdog вооружён до gates/install/materialization. Два независимых процесса, по одной visible GPU; credentials удалены из worker env. При отказе worker/истёкшей live-страховке supervisor останавливает второй.

Full main state каждые1800сwall и перед/послеветки: FP32 model/AdamW/scaler/RNG/cursor/scheduler/recipe/data identity. Latest atomic receipt SHA; предыдущий full file остаётся для конкурентного snapshot. В output все evaluated FP16 points, два последних full checkpoints на модель; массивы/anchors только scratch. L2 может подключить pinned output и private live dataset, выбрать максимальный verified cursor каждой модели. Strict data/recipe/code/Torch/CUDA identity; никакого неявного fresh restart.

**Первый gate:** нет Kaggle Secrets KAGGLE_USERNAME/KAGGLE_KEY или не2T4 → причина и немедленный exit доinstall/data. **Второй gate:** после CLI setup — tiny manifest upload в private `thisarivson/nlp-hw02-l1-live`, проверка server owner-private visibility и Dataset ready; failure → exit дообучения. Значения Secrets только env, не config/auth files/logs. Dataset CLI stdout/stderr отбрасывается; ошибки только type/returncode.

Live snapshot после каждой completed point и раз5400сwall (1.5ч); full latest обеихмоделей и latest evaluated weights/generations/сводки. Flat staging≤**2 500 000 000байт**, безduplicating directoryZIP. Старые points в прежних Dataset versions и output. Весьworking (включаяbinary) и staging проверяются на secret bytes поS2v2; auth username совпадающий сфиксированнымpublic owner — ресурсный идентификатор. Bounded child stage240с/privacy60с/upload600с, общийsnapshot≤900с. Failure retry300с; нет успешного snapshot≥7200с → stopworkers, сохранённое локальное состояние остаётся. Финальный bounded best-effort≤600с; watchdog≤20с внутри42300с.

API push L1 нужен дляпубликации T4×2 notebook, его ожидаемый запуск остановится наSecrets gate. Длинный запуск **только пользователь:** Edit → Add-ons/Secrets (оба прикреплены) → Accelerator GPU T4×2/InternetOn → Save Version → Save & Run All. Убедиться, что background version запущена, затем Stop Session интерактивного редактора (неCancelRun background version) и закрыть editor. Нельзя оставлять отдельную interactive GPU session.

## Проверки и состояние

Полный кеш подтверждён: **9 725 737 строк, 0/50 040 round-trip**, manifest SHA `46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be`. Pinned [CPU verifiedv2](https://www.kaggle.com/code/thisarivson/nlp-hw02-compact-cache-verified/versions?versionNumber=2):428files/360arrays SHA, offsets, исходныеsource/provenance и независимыйorder прошли. [Аудит](../checks/kaggle-l1-cache-output.json). Первая conversionv1 ERROR наexecutor gate; готовые массивы переиспользованы через private сжатыеtarget inputs. Отдельный executor без NumPy/Torch подтверждён наKaggle. Точная первичная причина не установлена; journals первого execution отсутствуют.

[L1 v1](https://www.kaggle.com/code/thisarivson/nlp-hw02-kaggle-l1/versions?versionNumber=1) опубликована private/T4×2; source exact. API gate завершился на13.35с безSecrets, logexport19.58с, training0; ERROR ожидаем послеos._exit. [Gate receipt](../checks/kaggle-l1-api-gate.json). Длинный запуск ещё не начат: нужен UI Save & Run All. [Точная инструкция](../kaggle/README.md).


ПередCPU push: tinyCPU веськеш/дваworkers/branches/eval→backward/полныйresume/гейты/mockedprivateCLI/secretrejection/staging/timeout/watchdog/decision. Реальные T4 FP16/fused/memory и eval→restore→backward/resume проверены коротким GPU-smoke в независимом ревью. Длинное качество/full1919 и runtime Secrets/live upload остаются проверками UI L1. Текущие URL/версии/outputSHA/status — `../kaggle/runs.json`.

## Источники

- [s3-representations-results](s3-representations-results.md), [32-candidate-ensemble-s3-results](32-candidate-ensemble-s3-results.md), [data-research-ceiling](data-research-ceiling.md), [26b-final-wsd-data-scaling](26b-final-wsd-data-scaling.md), [29m-28b-capacity-s2-results](29m-28b-capacity-s2-results.md) — измерения и история; решенияL1 отдельно выше.
- [Chinchilla](https://arxiv.org/abs/2203.15556), `hoffmann2022-chinchilla` — scaling laws; аппаратная оценка100M дана руководителем.
- [Kaggle kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md), [Datasets CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md) — private inputs, pinned versions, T4×2 accelerator и timeout.
- Код `../kaggle/build_l1.py`, `../kaggle/src/l1_*.py`; проверки `../checks/kaggle-l1-smoke.json`.
