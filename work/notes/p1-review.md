---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-14
updated: 2026-10-14
sources:
  - kaggle-p1-independent-review
  - kaggle-l3-final-results
aliases: []
---

# ДЗ2: независимое ревью P1 на аккаунте друга

Ревью исходного коммита `17eec4f`, ветка `codex/nlp-hw2-code-generation`; исходное дерево чистое. Проверены [p1-plan](p1-plan.md), [l3-review](l3-review.md), [l4-review](l4-review.md) и новые исходники. Полный длинный P1 не запускается в рамках ревью. L4 не запускать.

## Находки и исправления

Номера строк относятся к исправленной версии.

| Риск | Исправление / проверка |
|---|---|
| Существенный для бюджета: первоначальный B делает дополнительный forward всех128, затем train forward115. План ожидает +20–35%; это не удовлетворяет требованию малой задержки loss truncation. | `src/p1_math.py:76–160`: один forward каждого примера, графы microbatches сохраняются до общей маски. Все backward используют одну маску и общий retained-token denominator. При graph OOM — удаление gradients/graphs, восстановление RNG/scaler и прежний двухпроходный fallback; Adam OOM остаётся unsafe. `p1_tests.py:90–133`: разные длины/PAD, global mask, градиенты против retained-batch oracle, OOM с загрязнёнными gradients/RNG, отказ после первого backward. Равенство градиентов проверяется с допуском float32; побитового равенства B старому двухпроходному пути не обещается. |
| Пробел доказательства restore: исходный `restore.json` проверял только state/counters; отсутствовали CUDA fingerprints стартов A/B и полного restore. | `src/p1_runtime.py:20–23,64–68,155–163`: SHA tensor bytes model/Adam/scaler, всех RNG, cursor/history перед каждой веткой и после неё. Совпадение обязательно, иначе отказ. `p1_tests.py:137–182`: четыре одинаковых restore, обнаружение tensor/RNG corruption, immutable latest. |
| Неполная reviewer CE могла быть объявлена complete: контракт validate не проверялся, исключение финальной оценки не сохраняло отдельный partial summary. | `src/p1_runtime.py:26–31,259–290`: точные split names/rows, complete=true, положительные token counts и finite CE; ошибка даёт partial, final FP16 и stable latest сохраняются. `p1_tests.py:211–224`: ложный complete, неверные rows, NaN CE, настоящий WSD→сохранение FP16→неполная оценка без complete receipt. |
| При исчерпании cap/deadline исходные disposable branches оставляли только стартовый L3 state и неполный журнал. | `src/p1_runtime.py:132–151`: безопасные partial FP16, фактические stream/time/truncation totals и memory receipt; не маркируются complete. Unsafe Adam/nonfinite model не сохраняются как веса. `p1_tests.py:171–182`: BudgetExpired обеих веток сохраняет partial weights и immutable stable. |

17/17 P1 CPU-регрессий, nbformat/compile/source parity/static/diff-check прошли. Frozen L1/L2/L3 math/runtime/sampler/eval/live не изменены. `research/score-reviewer.py` не изменён.

## Сравнимость A/B

- `p1_worker.py:41–51`, `src/p1_runtime.py:55–68`: обе ветки читают один immutable L3 latest, SHA `48f2ba3f…`, updates209144/cursor20734592. FP32 model, AdamW, GradScaler, Python/NumPy/CPU/CUDA RNG и вся history восстанавливаются. Старт B не зависит от A или её оценки.
- `src/p1_runtime.py:90–98`: один weighted order SHA `72bc3dfd…`, `cursor+(step−1)·128`, общий T31372. Поток проверяется SHA всех предъявленных uint32 IDs. Cursor stable не изменяется; `model-pilot/latest.json` остаётся L3.
- `src/p1_math.py:24–27`: $\eta_j=0{,}0015(1-\sqrt{j/31372})$, $j=1,…,31372$; первый update чуть ниже peak, последний LR0, floor0. Ось — attempted updates; AMP/nonfinite skips отдельно считаются. Порог recovery отбрасывает disposable ветку и возвращает исходный L3, без emergency peak change.
- `src/p1_math.py:30–56,76–160`: $\ell_i=\sum CE_i/n_i$ с EOS, без PAD; маска top-loss по всей effective128, **13/128=10,15625%**, не отдельно на micro64. Backward после глобального отбора, $L=\sum_{i\in K}\sum CE_i/\sum_{i\in K}n_i$. Квоты bins53/79/105 пропорциональны counts, largest remainder и минимум один retained в непустом bin. Это compact proxy, а не точные original BPE60/90/120. Переход graph→fallback сохраняет цель, возможны отличия float rounding/порядка суммирования.
- `src/l2_eval.py:50–108,110–131`: production использует неизменённую L3 функцию K8: fixed tune2048 ascending IDs, seed270042/T0,7/K8/batch32/cap512. A/B journals сохраняют represented_code, execution и untempered mean LP. Complete требует rows=target_rows=2048; OOM retry восстанавливает RNG, однако меняет sampling batch shape, поэтому его нужно учитывать при офлайн-сравнении.

## Reviewer, входы и сохранность

- `src/p1_math.py:163–178`, `p1_worker.py:57–70`: encoder `[BOS]+cached compact code с mode+[EOS]`; decoder `[BOS]+question content`, labels `question+[EOS]`. PAD/BOS не входят в CE, EOS входит. Ordinary bucketed order сохраняется, дополнительно исключается global row8545870. Новый longest-row CUDA probe для обращённого направления обязателен; original max source234/target379 → reverse encoder381/decoder233. Собственная проверка всех9725737 rows: min source7, пустых вопросов0 ([length receipt](../checks/kaggle-p1-review-lengths.json)).
- Protected exclusions закреплены SHA sealed cache; полная перепись в [clean-cooldown-analysis — Protected exclusions](clean-cooldown-analysis.md#protected-exclusions): exact/hash-heldout пересечений0, normalized heldout/old/fresh0, единственный normalized test overlap8545870. Reviewer его удаляет; GPU0 намеренно сохраняет historical L3 data policy. Новые labels correctness и heldout/reference в reviewer train не добавляются.
- `src/p1_runtime.py:197–290`: C с normalized init/seed42, warmup1000, constant LR, финальный10% **wall**, stable checkpoints1800с и перед cooldown; latest содержит uncooled full model/Adam/scaler/RNG/cursor. Final model-only FP16 `identity.model=reviewer`, vocab8195 — контракт `research/score-reviewer.py:33–49,177–199`; scorer использует именно represented_code и question EOS. Фактический final update LR сохраняется отдельно от математического endpoint0.
- `build_p1.py:44–64`: watchdog до Torch/install, Secrets gate до anchor/cache/train, production friend owner проверяется. `src/p1_session.py:16–34` очищает worker env от API/Secrets/IAP credentials и не останавливает здорового соседа при ошибке другого. `src/l3_resume.py:8–13,29–54`: owner-qualified/flat mounts и строгие latest/manifest/SHA/counters/environment pins. Никакой credential файл не инспектировался.
- Budget worker37380/hard37800/backend43200с: hard10ч30мин ограничивает backend12ч. Cutoff **уже** оставлял pilot_eval1800 и stop120; дополнительное исправление этой части не потребовалось. Setup480, отдельные A/B caps13800/17400, final sync внутри hard−120; фактический API timeout sync может быть до1200с при досрочном завершении workers, plan300с — номинальный резерв, не отдельное превышение hard wall.
- `src/l2_live.py:35–98,104–183`: private friend1/p1-live, startup upload+ready, new-point и heartbeat5400с, insurance7200с. Раздельные roots pilot-a/pilot-b сохраняют обе оценки/веса в последнем snapshot; reviewer сохраняет latest+final. Working≤10GiB, обычные states/FP16≈2,3–3GB, stage≤2,5GB; stage и CLI ZIP вне working занимают общий диск. Local output скачивается только JSON/JSONL/log, без PT/order/cache.

## GPU-смоук и публикация

[Private smokev1](https://www.kaggle.com/code/friend1/nlp-hw02-p1-gpu-smoke/versions?versionNumber=1) **COMPLETE**: session728,881с =12мин08,9с; backend log заканчивается≈754,1с, в пределах20мин. Одна итерация. [Независимый audit](../checks/kaggle-p1-review-gpu-smoke-v1.json) passed:70 малых producer outputs проверены по bytes/SHA,73 JSON/JSONL/log файла скачаны (≈2,2MiB). PT/order/cache не скачивались.

| Фаза | A | B | Reviewer |
|---|---:|---:|---:|
| CUDA updates |64|64|1750|
| Bit-exact старт/полный restore/ID stream |passed|passed|stable reload1582updates|
| Greedy heldout/old/test |64/16/256|64/16/256|reverse CE64/16|
| K8 rows/samples |36/288|36/288|—|
| K8 full pipeline, с |11,07|10,94|—|
| Allocated peak, GiB |5,28|7,78|12,10|
| Reserved peak, GiB |7,31|10,92|12,43|

Reviewer peak включает longest-row probe: micro128 OOM,96 passed12,43GiB; training принудительно micro64. Все1750 попыток finite, stable full перед cooldown1582updates, cursor202496; final FP16 сохранён и strict-loaded на CUDA по контракту scorer, reverse CE64 finite. Final cooldown168updates; после injected exit1 GPU0 reviewer завершился exit0, stop-request отсутствует. Both workers complete внутри ожидаемого partial session/exits[1,0].

Сравнение скорости не ограничено соседними короткими branches: **32 evenly spaced batches** через весь production stream31372, одинаковые четыре warmups исключены. A **2,64366 update/s**, B **2,58703 update/s**, B/A time **1,02189 (+2,19%)**; reviewer tolerance≤15% задан в audit до результатов. Все32 B batches — graph, OOM-fallback0; CPU отдельно проверил graph OOM fallback. Peak этого benchmark A/B8,79/9,76GiB. Projection T31372: A11866,9с (3ч17мин47с), B12126,6с (3ч22мин07с), ниже caps13800/17400. Это оценка sampled stream, не гарантия full run.

Finite L1↔P1 CUDA update и serialized restore→next update **bit-exact**. NaN selection обоих B paths, отказ после первого backward и GradScaler overflow проверены: model/Adam не меняются, rejected loss восстанавливает scaler и очищает gradients; found-inf уменьшает scale вдвое и не делает Adam step. A/B start/restore fingerprints5 компонентов равны; оба K8 имеют seed270042/T0,7/ascending subset fixed tune и **нет OOM batch change**. Original и friend fallback кеши прошли SHA всех360 payloads; owner/inputs/source port закреплены [server metadata](../checks/kaggle-p1-review-smoke-metadata.json).

[Боевой private P1v1](https://www.kaggle.com/code/friend1/nlp-hw02-kaggle-p1/versions?versionNumber=1) отправлен через API аккаунт друга, T4×2/timeout43200. Source port SHA47110a47… совпадает со smokev1; production T31372/q10%/peak0,0015/hard37800 восстановлены. **Status ERROR — ожидаемый Secrets gate**: `startup-gate.json` и log сообщают `Kaggle Secrets unavailable: KAGGLE_USERNAME, KAGGLE_KEY`. [Gate receipt](../checks/kaggle-p1-review-production-gate.json): remote outputs без anchors/workers/model roots, training_started=false. Пользователь: прикрепить оба friend Secrets в аккаунте друга, Edit→Save Version→Save & Run All только **P1**. Качество длинных A/B и reviewer пока не измерено. Квота API до ревью48384,644с (13ч26мин), после smoke/gate47601,456с (13ч13мин), reserved0; hard10ч30мин оставляет≈2ч43мин. [Quota receipt](../checks/kaggle-p1-review-quota-after.json). **Вердикт: готов к P1 UI Save & Run All с attached friend Secrets**; полный live/длинный run ещё не проверен.

## Некритичные риски

- Короткий smoke не доказывает безотказность10,5ч. При внезапном kill незаконченный disposable A/B не имеет full resume и теряет незаписанную работу; stable L3, завершённая A и её live snapshot сохраняются. Reviewer rollback может потерять до1800с с последнего full checkpoint.
- Graph truncation увеличивает activation memory. Fallback безопасен, но частые OOM добавляют стоимость; 32 sampled positions не гарантируют скорость всех31372 updates. Caps/deadline должны дать partial, T/q/peak молча не менять.
- Квоты q округляются; компактные bins приближённые. Они не позволяют объявлять точно10% или исходные BPE-strata. High loss может означать трудный правильный пример.
- Disk guard free512MiB меньше одного full PT≈644MB. Внезапная нехватка общего диска может сорвать checkpoint/upload; атомарный latest сохраняет предшественник. Реальный attached Secrets/live upload и сеть проверяются только UI production run.
- OOM K8 batch change нарушает побитовый поток samples относительно L3 при том же seed; journals/OOM_attempts дают диагностику. Smoke требует отсутствие такого retry. Повторное использование eval/tune и последующий selector подбор не являются новым независимым test.

## Источники

- [p1-plan](p1-plan.md), [l3-review](l3-review.md), [l4-review](l4-review.md), [l3-results](l3-results.md), [clean-cooldown-analysis](clean-cooldown-analysis.md), [runs.json](../kaggle/runs.json), собственные CPU/GPU receipts.
- [PyTorch2.11 CE](https://docs.pytorch.org/docs/2.11/generated/torch.nn.CrossEntropyLoss.html): unreduced/token-sum loss и ignored PAD. [PyTorch2.11 AMP](https://docs.pytorch.org/docs/2.11/amp.html): FP16 autocast/GradScaler и skipped optimizer steps. Публичные первичные источники; новые внешние исследования не добавлялись.

## Коммит

Попытка явного P1-review-only `git add` получила exit128: `Unable to create '<vault>/.git/worktrees/study1/index.lock': Operation not permitted`. Индекс не изменён; commit не выполнен, ограничение не обходилось. [Git receipt](../checks/kaggle-p1-review-git.json). Основной checkout не редактировался.

Повтор из обычного терминала **в этом worktree**, после проверки `git diff`:

```bash
python3 - <<'PYCOMMIT'
import json, subprocess
from pathlib import Path
p = Path('work/checks/kaggle-p1-review-git.json')
r = json.loads(p.read_text())
subprocess.run(['git', 'add', '--', *r['paths'], str(p)], check=True)
subprocess.run(['git', 'commit', '-m', r['message']], check=True)
PYCOMMIT
```
