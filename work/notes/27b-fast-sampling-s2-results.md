---
type: experiment
course:
  - nlp
status: learning
understanding: 3
created: 2026-10-11
updated: 2026-10-11
sources:
  - kaggle-s2-v2
  - kaggle-s1-v1
aliases: []
---

# ДЗ2, №27b: S2 — быстрый сэмплер не прошёл greedy gate

**Новый greedy отличается от сохранённого S1 на35/4096 вопросов; №27b не допущен к выбору.** [Протокол до запуска](s2-fast-sampling-and-capacity.md), [сводка S2](../results/kaggle-s2-v2-summary.json), [CPU-аудит](../checks/kaggle-s2-v2-cpu-audit.json). GPU0 остановлен с exit1 до M29. Новых training updates при разборе0.

| Проверка | Факт | Решение по протоколу |
|---|---:|---|
|Greedy token-identity|4061/4096 одинаковых;35 различий (0,854%)|Gate failed; допуск требует4096/4096|
|K16/T0,7, без greedy-голоса|GPU execution/quality gate не запускались|Gate не пройден|
|Диагностика k16 на CPU по saved samples|2702/4096 (65,967%); внутри2707±45|Post-hoc; не отменяет greedy gate|
|Новый greedy, CPU exec|2535/4096 противS1 2538;1 gain/4 losses,p0,375|Качество почти прежнее, token-identity всё равно нарушено|
|Fulltest1919 greedy / k8 / k16|Не измерены / не измерены / не измерены|K не выбран; правило≤120с неприменимо|

Batch **2048 candidate rows /128 вопросов** прошёл longest-input/full-cap probe без OOM; fallback1024 не применялся. Probe18,798с, generation18,032с, peak reserved**5.844GiB**, physical cross-KV**1.164GiB**. Это probe, а не таймер test. Heldout journal содержит4096 greedy+65536 samples; probe —128+2048. Генерации сохранены полностью, execution chunks и test отсутствуют: остановка произошла перед исполнением.

## Диагноз и границы

`gpu0-failure.json` и traceback: `AssertionError: Fast greedy token identity failed; see27b-summary`. Независимая сверка token IDs воспроизвела все35 индексов; первые различия на позициях7–429. CPU fast-greedy и vote повторно исполнены с wall3с/CPU2с и прежней numeric policy. Vote против нового greedy:241gains/74losses,+4,077п.п.,p8,632e-22. Это диагностика уже сохранённых генераций, а не новый CUDA-прогон и не успешный зарегистрированный опыт.

**Гипотеза:** FP16/SDPA или матричные ядра чувствительны к изменению question batch512→128 и padding. Совпадение большинства строк и редкие расхождения этого не доказывают; нет CUDA-сравнения logits первого отличающегося токена при одинаковых batches. CPU tiny-smoke не устанавливает причину наT4. Фактический SDPA backend не записан.

Результат №27 изS1 остаётся историческим; нового допустимого vote и нового полного greedy-таймера нет. Рабочая база остаётся A/6,4M предъявлений + прежний greedy; это не новый выбор по S2.

## Следующий шаг

Короткий CUDA-аудит35 вопросов: одинаковые batch/padding/precision, первый отличающийся logit, canonical greedy и physical4D sampling отдельно. После успешного полного4096 token gate — k16 gate и свежие test1919 greedy/k8/k16. Полный повтор M/B до этого не обоснован. [Сохранённые состояния и команды S3](../kaggle/README.md#разбор-s2-v2-и-восстановление-s3).

## Источники

- `kaggle-s2-v2`: pinned `gpu0-exp27b/27b-summary.json`,33 batch/SHA receipts, `gpu0-process.log` и `gpu0-failure.json`; hashes в CPU-аудите.
- `kaggle-s1-v1`: финальные greedy token IDs A/6,4M, reference receipts и зарегистрированное k16=2707; [S1/27](27-execution-voting-s1-results.md).
