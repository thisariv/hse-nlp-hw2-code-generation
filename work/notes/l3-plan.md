---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-13
updated: 2026-10-13
sources:
  - kaggle-l2-results
  - kaggle-l2-preparation
aliases: []
---

# L3: продолжение обеих C из L2 на втором аккаунте

По заданию пользователя обе модели продолжаются независимо: GPU0 C-main, GPU1 C-long2x. Цель — дополнительное обучение и сохранение двух восстановимых stable states для такого же L4. Это новый протокол пользователя; рекомендация одной C из [l2-results](l2-results.md) остаётся историей анализа. Гипотеза: дополнительное main-время улучшит обе C и качество будущего ансамбля; измеренный результат L3 пока отсутствует.

## Аккаунт и источники

CLI2.2.4: `KAGGLE_CONFIG_DIR=~/.kaggle-friend` выбирает аккаунт друга; без переменной работает **thisarivson**. Credential directories не открывать и не копировать в артефакты. Новый private notebook `friend1/nlp-hw02-kaggle-l3`, private live `friend1/nlp-hw02-l3-live`; live заранее создан, v1 ready/private. Internet on, T4×2, `machine_shape=NvidiaTeslaT4`.

- Private `thisarivson/nlp-hw02-l2-live`, проверенные квитанции v11: оба latest full states и исходный weighted order.
- Private `thisarivson/nlp-hw02-26b-inputs`: прежние tokenizer/heldout4k/old/test и execution references.
- **Выбран оригинальный pinned кеш** — output `thisarivson/nlp-hw02-compact-cache-verified/2`: после шаринга аккаунт друга скачал его manifest с точным SHA. При первой проверке output был запрещён; заранее создан запасной private dataset `friend1/nlp-hw02-compact-cache-verified`, v1 ready, из локального compact-cache v2. Для запасного пути явно собрать `--cache-source friend1/nlp-hw02-compact-cache-verified`; remote manifest также скачан и проверен.

Manifest кеша должен иметь SHA `46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be`, 9 725 737 строк, 60 shards, round-trip≥50 000 и 0 mismatches. До fallback upload все shard SHA проверены, payload2 603 881 817 bytes, свободно7 245 545 472 bytes; staging — hardlinks, CLI создаёт один временный ZIP. Manifest и order не меняются.

При первой проверке оба чужих datasets возвращали403 от аккаунта друга. После UI-шаринга **оба доступны через CLI**, output оригинального кеша тоже доступен. Последние проверки доступа, создание live и fallback — `../checks/kaggle-l3-*.json`. Без доступа или точного anchor notebook отказывает до обучения.

## Строгое продолжение

| GPU / модель | Full-state file в flat live | SHA256 | Updates | Cursor | main_s | l2_main_s |
|---|---|---|---:|---:|---:|---:|
|0 / C-main|`model-C-main__checkpoints__main-6275fbfc4185.pt`|`d0de7f7bd5c479159a8d6189ab5af58897edf35bdba2e9439b0b745ab72bd29c`|126452|16185856|46370.1433977189|28800.207519321037|
|1 / C-long2x|`model-C-long2x__checkpoints__main-826d479cdfe8.pt`|`a0ba33fef996f6a0c63e4059909a17796c67e1a94dadf1cc04c6afe1e94c40d2`|123842|9815936|46370.24584266532|28800.309964266355|

`l3-anchors.json` содержит полные identity/model_spec и квитанции. Startup проверяет latest↔recover-manifest↔pin, file SHA/bytes/counters/main_s, finite tensors/Adam, scheduler/scaler/CUDA RNG, semantic SHA и Torch2.11.0+cu128/CUDA12.8 **до install/cache**. Worker повторяет проверку перед CUDA update. Kaggle-expanded JSON вместо `.gz` принимается только при воспроизводимом producer gzip SHA.

Все `src/l2_*.py` переиспользуются без изменения: safe update, NaN/AMP recovery, full rollback, disposable cooldown, live-sync, watchdog, independent workers и evaluation. `l2_worker.py` получает только выбор L3 anchor/identity/phase configuration через `l3_resume.py`. AdamW/FP32 weights/scaler/RNG/history, LR0.0015/micro64/effective128 сохраняются; warmup не повторяется. C-long2x **не создаётся заново из C-main**, cursor не обнуляется и identity не пересчитывается.

C-main order SHA `09cf6db8549b7df614fa71516553d23d4cbb3ca97c7911db0c6d599a8277ab7d`. C-long2x использует **скачанный из L2 live order**, SHA `72bc3dfd3a3e738979c489c0cf8f41cde092c41d4ae90f2de8378a034603a4db`, seed2026101302, threshold79compact, multiplicity1/2, 11 227 811 slots; metadata строго равна inherited sampler. Повторного построения order нет.

Runtime сохраняет накопленное `l2_main_seconds`; L3 targets = точное anchor `l2_main_seconds` +16 740/+33 480. Новые tags остаются совместимыми с L2: `l2-45540`, `l2-62280`. Три прежние точки/history сохраняются без перезаписи; их тяжёлые файлы остаются в L2 source. Для L4 использовать новый latest и следующие накопленные targets, не повторять L3 tags.

## Бюджет от первой ячейки

| Фаза, параллельно на двух T4 | Cap, с |
|---|---:|
| Gates, preflight двух anchors, install, private live probe, cache/order |480|
| Main:16 740 +16 740 |33 480 (9ч18мин)|
| Промежуточный cooldown |900 (15мин)|
| Одна промежуточная greedy/CE/old991/test оценка |300|
| Финальный cooldown |3000 (50мин)|
| Финальные greedy heldout4k/old991/CE/test и K8 tune2048 |1800|
| Checkpoint/restore/I/O reserve |180|
| Штатный worker stop reserve |120|
| **Worker deadline** |**40 260 (11ч11мин)**|
| Финальный live-sync |до420|
| Watchdog/packaging reserve |120|
| **Hard watchdog** |**40 800 (11ч20мин)**|

Hard wall на240с ниже предела11.4ч; backend timeout12ч. Два таких hard wall=22ч40мин, около2ч20мин остаётся из25ч до расходов смоука. Смоук подготовлен отдельно: workers15мин, hard17мин, backend18мин, без Secrets/live uploads.

L2 занял34 411.55с при28 800с main и двух mid cooldown. Верхняя эмпирическая оценка:34 411.55 +4680 −900 =38 191.55с (10ч36мин32с), ещё без вычета отменённой промежуточной оценки. Это прогноз, не гарантия новой T4 throughput. L2 mid eval85.6–117.8с, final235.8–245.1с; новые caps300/1800с оставляют запас. При недостатке времени точка incomplete; stable state сохраняется, чужая модель не останавливается из-за ошибки соседа.

Финальная оценка и K8 journals — прежний L2 протокол; heldout4k и old991, greedy test1919 full pipeline без gold, K8/T0.7/seed270042 на fixed tune2048, каждый завершённый batch сохраняет code/execution/LP. Cooldown15/50мин считается по update time; после него восстанавливается полный stable anchor. Fresh и отдельный final-inference task здесь не выполняются.

## Live и проверки

Боевой gate требует прикреплённые Kaggle Secrets **KAGGLE_USERNAME/KAGGLE_KEY аккаунта друга; username дополнительно проверяется, значение key не выводится/не сериализуется. Private owner startup upload/ready probe остаётся обязательным. Workers лишены API/Secrets/IAP credentials. Live≤2.5GB, full latest обеих C + latest cooled points/journals + weighted order; heartbeat5400с, retries≥300с, insurance7200с, stable checkpoints1800с и до/после eval.

CPU: все34 L2 регрессии, оба реальных L2-state SHA и первые два batch128 шага reference↔L3 bit-identical; CPU FP32/восстановленный enabled scaler и AdamW без fused. Каждый full PT скачивается отдельно, после проверки удаляются PT и scratch copies. CPU не проверяет CUDA/FP16/fused численную идентичность. Статика: code compile, embedded source parity, metadata/private/T4/source pins/budget, empty outputs.

Ревьюер: проверить доступ/метаданные mounts от аккаунта друга, CPU receipts/source diff; выполнить один подготовленный GPU-smoke без Secrets, затем отдельно боевой private push/UI Save & Run All с прикреплёнными friend Secrets. В этой подготовке GPU/push=0; Kaggle resources не удаляются.

## Источники

- [l2-plan](l2-plan.md), [l2-review](l2-review.md), [l2-results](l2-results.md); verified L2 live v11/latest и текущие локальные источники L2.
- Задание пользователя: второй аккаунт, обе модели, ~9.3ч main, один mid cooldown и такой же L4. Бюджет и прогноз — расчёт этой подготовки.

Проверки завершены: [34 L2](../checks/kaggle-l2-regressions.json), [6 L3](../checks/kaggle-l3-regressions.json), [C-main real resume](../checks/kaggle-l3-real-resume-C-main.json), [C-long2x real resume](../checks/kaggle-l3-real-resume-C-long2x.json), [статика/nbformat](../checks/kaggle-l3-static.json). Оба скачанных PT и весь собственный temporary staging удалены; сохранены маленькие SHA/digest receipts.

Попытка коммита L3: `git add` явного L3-only списка получила `Operation not permitted` на shared worktree `index.lock`. Index не изменён; изменения сохранены, чужие final-inference файлы не staging.

Независимое ревью завершено: [l3-review](l3-review.md). L2 34/34, L3 10/10; CUDA v1/v3 COMPLETE, v3 проверил полный friend fallback по SHA. Исправлены partial-status и mixed flat/nested cache-mount; боевой API push и Secrets gate зарегистрированы в отдельном review entry runs.json.
