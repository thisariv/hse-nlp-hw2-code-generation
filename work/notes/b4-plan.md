---
type: experiment
course:
  - nlp
status: learning
understanding: 4
created: 2026-10-17
updated: 2026-10-18
sources:
  - kaggle-b4-preparation
  - kaggle-b3-results
  - kaggle-b3-independent-review
  - kaggle-p1-independent-review
  - kaggle-b4-independent-review
aliases: []
---

# B4: W600 против W600 с loss truncation

18.10: [B4 полностью завершилась и разобрана](b4-results.md); plain70,90%, trunc69,95%, парный CI финальной разницы включает0. Ниже сохранён исходный план17.10.

17.10, [независимое ревью](b4-review.md): v1 выявил GPU1 graph OOM даже при micro1. Текущая сборка использует ту же CE+z цель с аналитическим backward из сохранённых logits и pinned CPU offload surplus saved activations после10GiB; model forward остаётся один. Smoke теперь200 main updates/GPU.18 B4 CPU tests passed; окончательная готовность и запуск определяются reviewer receipt/`next-b4.json`, исходные CPU-ready итоги ниже сохранены как история подготовки.

**Пользователь выбрал B4 и уже разрешил ночной запуск после ревью.** Подготовка выполняется без GPU и kernel push; оба запуска оставлены ревьюеру. Private `thisarivson/nlp-hw02-b4`, title `nlp-hw02-b4`, T4×2, internet on, default `~/.kaggle`, без Secrets/live-sync. Предыдущая рекомендация отложить B4 в [b3-results](b3-results.md) остаётся историей анализа; текущий протокол задан пользователем.

Гипотеза: исключение наиболее высоких token-normalized CE внутри каждой длиновой страты улучшает качество продолжения W600. P1 не установила надёжный выигрыш truncation; для B4 это проверяемое условие, не новый финальный default. V7 и C не меняются.

## Один общий старт

Источник: `thisarivson/nlp-hw02-b3/1`, **model-GPU1**, `latest.json` до финального cooldown; [полный pin](../kaggle/b4-anchor.json).

| Поле | Обе GPU |
| --- | --- |
| Stable | `model-GPU1/checkpoints/main-667d29a22ed6.pt` |
| Bytes / SHA256 | 1 361 617 533 / `e32b8649d41ef73133fd2dcf031a649bf8af6a07013f76bc057b897643b119c0` |
| Updates / cursor | 73 416 / 9 397 248 |
| Main_s | 59 400,86158832819 |
| LR / factor / warmup | 0,0006 / 1 / прежние 2000 updates уже пройдены |
| Архитектура | 6+6, d768, 12 heads, FFN2048, headwise QK LayerNorm, z-loss1e-4 |
| Effective / micro | 128 / исходный32; OOM может уменьшить micro |

Восстанавливаются FP32 model, AdamW со всеми моментами/step/param groups, enabled GradScaler, Python/NumPy/Torch CPU/CUDA RNG, cursor, scheduler, stability, points и sessions. GPU0 **W600** вызывает неизменённый `b2_runtime.main_step`; constant0,0006 без нового warmup. GPU1 **W600-trunc** получает тот же численный payload; отдельная identity фиксирует q/родительский SHA и реализацию B4/P1. Исходная identity GPU0 сохраняется.

Weighted order берётся из **того же output B3**, не перестраивается: SHA `8e9595f655ed2b4456bf5b94e9b5b6b3db040e3f5696a4dc0ef364588048bee8`, seed2026101501, long×2 при compact length≥79. Cache manifest `46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be`. Удалений train-строк и новых фильтров нет.

Входы: kernel_sources `thisarivson/nlp-hw02-b3` и `thisarivson/nlp-hw02-compact-cache-verified/2`; dataset `thisarivson/nlp-hw02-26b-inputs`. **b2-states не нужен**, перенос и копирование full PT отсутствуют. Preflight проверяет весь pinned B3 output manifest, exact latest, bytes/SHA stable и order; читает model-GPU1 на обеих GPU. Mount owner-qualified или flat. Runtime по-прежнему требует Torch2.11.0+cu128/CUDA12.8 и все10 frozen B2 implementation SHA; fresh fallback отсутствует.

[Read-only API/квота](../checks/b4-inputs.json): stable действительно есть в output, свободно **68 798,525с ≈19ч06мин**, reset **24.10.2026 00:00UTC**. Versioned output endpoint для `v1` вернул пустой список; использован latest endpoint с **SHA manifest, совпавшим с архивом версии1**, exact latest и SHA полного payload. Это привязка по содержимому; другой latest молча не принимается.

## Truncation и сравнимость

Путь GPU1 — [b4_math.py](../kaggle/src/b4_math.py), quota/mask из исправленной [реализации P1](p1-review.md). Для каждого примера $i$ вычисляется $\ell_i=\sum_t CE_{it}/n_i$, где $n_i$ — число target-токенов с EOS, без PAD/BOS. Выбор делается по **CE**, без z-loss.

На всей effective batch128 маска одна: округление q=0,1 даёт **13/128 =10,15625%** исключённых, если ёмкости bins допускают это. Страты compact target content length по границам53/79/105; это приближение исходных BPE60/90/120. Пропорциональные квоты, largest remainder, минимум один retained в непустом bin; ties — по позиции в batch, дубликаты являются отдельными предъявлениями. Это top-loss **внутри страт**, а не неограниченный global top13.

Для множества оставленных примеров $K$ цель:

$$L=\frac{\sum_{i\in K}\sum_t\left[CE_{it}+10^{-4}\left(\log\sum_v e^{z_{itv}}\right)^2\right]}{\sum_{i\in K}n_i}.$$

Здесь $z_{itv}$ — logit токена $v$; PAD исключён из обоих слагаемых. CE и z-loss считаются в FP32. **Два прохода (исправление после GPU-smoke v3)**: сначала no-grad scoring всех128 с FP16 autocast, затем стандартный B2 forward/backward только115 retained; граф каждого microbatch освобождается сразу. Selection сохраняет/восстанавливает Python/NumPy/Torch CPU/CUDA RNG, pinned dropout0 сохраняет общий RNG-поток. CE+z и denominator retained tokens неизменны. Custom backward и save_on_cpu удалены: offload нарушал выравнивание attention bias mem-efficient SDPA. OOM retry сохраняет B2 reset RNG/scaler/gradients и повторяет оба прохода с меньшим micro; Adam OOM остаётся unsafe. Ожидаемое замедление шага≈30% разрешено пользователем; прежний reviewer gate15% больше не применяется.

Оба курсора продвигаются на **128 предъявленных**, независимо от13 исключённых и AMP skip. Для каждого main update пишется rolling SHA всех uint32 input IDs. В начале сохраняются SHA model/Adam/scaler/enabled/RNG/state/modes. После workers [b4_session.py](../kaggle/src/b4_session.py) проверяет одинаковые старт и SHA **всего общего префикса updates**; ошибки дают partial. Truncation journals сохраняют counts/quota, IDs/позиции/длины исключённых, средние CE и presented/retained tokens; summary суммирует main statistics. Cooldown statistics находятся в полных cooldown journals.

**Равный бюджет времени не гарантирует равного числа updates.** Потоки одинаковы при одинаковом attempted update, но финальные курсоры могут различаться из-за стоимости truncation. Анализировать качество по wall и парам отдельно; по +3/+6ч оценкам нельзя утверждать строго matched-pair эффект. Independent CUDA trajectories могут численно расходиться даже при одинаковом RNG; точный инвариант smoke — непосредственный before/after полный restore, как исправлено в B3.

Серьёзная нестабильность/unsafe Adam OOM: откат к последнему stable и **partial stop**, без автоматического LR-halving и смены q/order. Соседняя GPU продолжает независимо. Обычные AMP/nonfinite skips учитываются по прежней политике B2.

## Время и результаты

Hard от первой code cell **40 800с =11ч20мин**, workers40 500, watchdog40 740; backend timeout40 800. Setup cap480с, checkpoint каждые900с. Watchdog вооружается до Torch/install. Очередь, VM setup и публикация output могут добавить платформенное wall/списание; timeout не является доказанным потолком billing Kaggle.

| Фаза на каждой GPU | Main_s от старта B4 | Cooldown | Оценка |
| --- | --- | --- | --- |
| Probe1 | +10 800с; absolute70 200,861588 | 900с linear | greedy heldout4096 +old991 +unsmoothed token CE |
| Probe2 | +21 600с; absolute81 000,861588 | 900с linear | то же |
| Final | cap+32 400с; absolute91 800,861588, фактический anchor в receipt | **3000с linear** | полная greedy/CE, K8 tune2048 journals, test1919 timer |

Все cooldown disposable и используют следующий префикс order. GPU0 q0, GPU1 **q0,1 также в probes и final**. После каждой оценки/ошибки восстанавливается полный uncooled anchor, включая RNG/Adam/cursor; охлаждённые FP16 сохраняются отдельно. Timed linear LR вычисляется до update: $\eta(t)=0{,}0006(1-\min(1,t/T))$, $t$ — сумма времени updates, $T$ —900 или3000с; фактический последний LR записан, искусственного нулевого update нет.

На probe eval450с, final eval1800с. Main всегда оставляет **5100с =3000+1800+240+60**, probe требует дополнительно1350+120с. Номинальный план32400+2×(900+450)+3000+1800+480=40 380с; I/O и запас могут сократить main. **≈9ч — cap, не обещание**. Пропущенная/неполная оценка отражается partial; worker/notebook не падают из-за финального assert на статус. Частичный успех не объявляется complete.

K8: frozen tune2048, T0,7, seed270042, исходные IDs/question SHA, represented code/logprob/execution. Test timer включает reload/tokenizer/generation/restore/execute/закрытие1919 answers; это одиночная модель, отдельный cold ensemble≤180с он не заменяет. Fresh не читается. `latest.json` и два поколения full PT на каждой GPU остаются **uncooled для B5**, summary `B5_uncooled_latest`; final FP16 — отдельные веса. Обычный output≈6,8GB плюс компактные журналы; concurrent checkpoint peak≈9,6GB, guard10GiB, ZIP нет.

Из доступных≈19ч06мин production11ч20мин и один smoke12мин оставляют≈**7ч34мин** до reset, без гарантии платформенного overhead. Live-sync отключён; атомарный stable ограничивает потерю работы при штатном завершении, но сбой VM всё ещё может потерять локальный output.

## CPU и передача на ревью

- [Реальный CPU resume](../checks/b4-real-resume.json): один скачанный B3 stable, direct full-state control против GPU0 B4 main_step, **настоящий следующий batch128** из проверенного order/cache, model/Adam/scaler/RNG/counters и CE/z-loss побитно. CPU FP32/no-autocast, enabled CPU GradScaler; fused отключён только после проверки полного restore. CUDA RNG bytes проверены, CUDA execution остаётся smoke. Временный full PT удалён после проверки.
- [Order/cache](../checks/b4-order.json): все payload SHA, multiplicities, обе next128 sequences. [18 B4 CPU tests](../checks/b4-cpu.json): strata/mask/ties, CE+z retained-token oracle и gradients с допуском float32, 128 no-grad +115 retained forward, оба курсора/потока, OOM reset, unsafe Adam/nonfinite, полный restore после двух probes/final, q0 bit-exact, partial preservation и fixed-LR stop.
- [67 регрессий B3/B2/B1/P1](../checks/b4-regressions.json), [статика](../checks/b4-static.json): frozen implementation SHA, compile/nbformat, embedded/local parity, одинаковые модули smoke/production, private/slug/mounts/budget/watchdog/Secrets-free.
- **Смоук только подготовлен**, hard≤720с, worker660/watchdog690. Обе настоящие B3 W600:9 disposable throughput/memory batches (8 по оценочному9ч потоку +самый длинный aligned batch,4 warmups исключены), затем полный reload исходника и200 main updates, mini probe/final cooldown, greedy64/old16/CE, test64, K8 на36 frozen tune IDs, прямые full restore audits и AMP-overflow. Малые сроки/cap8 не измеряют качество или full timing. Память и скорость двух проходов проверяются v4; OOM не допускается в successful smoke.

Машинная передача: [next-b4.json](../results/next-b4.json), **ready:true требует successful same-source GPU receipt**; CPU-only готовность задаётся `CPU_ready:true`. Ревьюер проверяет diff, затем один smoke с теми же module SHA и остаток квоты≥42 000с перед production. Повторного разрешения пользователя для ночного B4 не нужно.

```sh
# Из work/kaggle; сборка без запуска:
python build_b4.py
python build_b4.py --gpu-smoke
python b4_tests.py
python b4_regressions.py
python b4_static_check.py

# Только ревьюер после ревью:
KAGGLE_CONFIG_DIR=~/.kaggle kaggle kernels push -p bundles/b4_gpu_smoke_thisarivson --accelerator NvidiaTeslaT4 --timeout 720
# После successful same-source smoke и повторной проверки квоты:
KAGGLE_CONFIG_DIR=~/.kaggle kaggle kernels push -p bundles/b4_thisarivson --accelerator NvidiaTeslaT4 --timeout 40800
```

Сборки `kaggle/bundles/b4_thisarivson` и `kaggle/bundles/b4_gpu_smoke_thisarivson` локальные/ignored; notebook/builders версионируются. Повтор CPU с одним скачиванием/автоматическим удалением: `kaggle/b4_prepare.py --cpu-python <CPU-python> --cache <verified-cache>` через Kaggle SDK environment. Содержимое credentials не читается/не выводится.

## Коммит

Попытка `git add` завершилась128: `Unable to create .../.git/worktrees/study1/index.lock: Operation not permitted`. Индекс не изменён, commit не создан; ограничение песочницы не обходилось. Код, сборки и ready handoff сохранены; [квитанция](../checks/b4-git.json).

## Источники

- **kaggle-b4-preparation** — собственные код, сборки, CPU/API/static receipts17.10.2026; bibliography — kaggle-b4-preparation. Дизайн/бюджет — пользовательский протокол, оценки скорости/риска — собственные выводы.
- [b3-plan](b3-plan.md), [b3-review](b3-review.md), [b3-results](b3-results.md) — исходные states, реальные counters/order и уроки restore/partial/budget.
- [p1-review](p1-review.md) и [p1_math.py](../kaggle/src/p1_math.py) — общий mask/graph алгоритм. Формализация выше описывает **конкретный B4 вариант**: truncation в main, FP32 z-loss и два прохода с bounded microbatch memory. Новых внешних научных источников не добавлено.
