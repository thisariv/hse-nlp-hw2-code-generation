---
type: experiment
course:
  - nlp
status: complete
understanding: 4
created: 2026-10-15
updated: 2026-10-15
sources:
  - nlp-hw02-overfit-check
  - https://jmlr.org/papers/v15/srivastava14a.html
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1711.05101
  - https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html
aliases: []
---

# ДЗ2: CPU-проверка переобучения

**На проверенных точках сценарий «train CE падает, heldout стоит» не подтвердился.** Разрыв обобщения есть у всех: около0,047–0,054 CE. P1-A улучшает оба набора почти одинаково; R1 ухудшает оба. Небольшое расширение разрыва в R1 измеримо, но его нельзя приписать позднему переобучению без контроля objective и cooldown.

| Модель | CE train4096 | CE heldout4k | Разрыв heldout−train | 95% CI разрыва | Token accuracy train | Token accuracy heldout |
|---|---:|---:|---:|---|---:|---:|
| L3 C-long2x | 0,200600 | 0,248277 | +0,047677 | [0,0396; 0,0558] | 93,57% | 92,38% |
| L3 C-main | 0,203005 | 0,249934 | +0,046929 | [0,0391; 0,0553] | 93,52% | 92,32% |
| P1-A | 0,187138 | 0,234985 | +0,047848 | [0,0405; 0,0554] | 93,91% | 92,68% |
| P1-B | 0,193987 | 0,245515 | +0,051527 | [0,0432; 0,0602] | 93,81% | 92,56% |
| R1 control (6+6) | 0,218464 | 0,271842 | +0,053377 | [0,0433; 0,0640] | 93,28% | 92,04% |
| R1 deep (12+6) | 0,220268 | 0,274552 | +0,054283 | [0,0442; 0,0648] | 93,25% | 91,98% |

**L2 C-long2x l2-28800:** FP16 по указанному пути в cleanup недоступен, поэтому train CE и динамика его gap **не измерены**. Архивная GPU heldout CE L2→L3: **0,254557→0,248291**, небольшое улучшение, а не точное плато. Эти два исторических значения взяты из [l2-results](l2-results.md)/[l3-results](l3-results.md), а не из нового CPU-прогона.

### Динамика на тех же строках

Знак Δ — более поздняя/альтернативная точка минус базовая. P1 — альтернативные cooldown-ветки от uncooled L3 anchor, не последовательное продолжение оценённых FP16 weights.

| Сравнение | Δ CE train | Δ CE heldout | Δ разрыва [95% CI] |
|---|---:|---:|---|
| L3 C-long2x → P1-A | −0,01346 | −0,01329 | +0,00017 [−0,00152; +0,00182] |
| P1-A → P1-B | +0,00685 | +0,01053 | +0,00368 [+0,00186; +0,00555] |
| L3 C-long2x → R1 control (6+6) | +0,01786 | +0,02356 | +0,00570 [+0,00235; +0,00933] |
| L3 C-long2x → R1 deep (12+6) | +0,01967 | +0,02627 | +0,00661 [+0,00322; +0,00999] |
| R1 control (6+6) → R1 deep (12+6) | +0,00180 | +0,00271 | +0,00091 [−0,00163; +0,00349] |

После приведения train к token shares длиновых bins heldout разрыв остаётся **0,0468–0,0543**: различием длин он не объясняется.

**Что установлено:** у R1 рост gap к L3 составляет +0,00570/+0,00661, интервалы не включают0. При этом train CE тоже растёт на +0,01786/+0,01967. Дополнительное углубление deep против control меняет gap лишь на +0,00091, CI включает0; признака усиленного переобучения именно из-за глубины здесь нет.

**Чего сравнение не изолирует:** R1 перешла с long2x на ordinary presentations, использует q0,1 loss truncation и лишь20мин cooldown против50мин L3; deep также получает меньше updates за равное время. Обычная CE здесь включает и трудные targets, отброшенные training objective. P1-A/B получают по31372cooldown updates (около3ч37мин/3ч40мин); B дополнительно отбрасывает≈10% предъявлений. Изменение objective, достаточность cooldown и уверенность ошибок — возможные объяснения raw CE; отдельно они не проверены. По отфильтрованному training loss из журнала нельзя заключать, что падает обычная CE всего train.

**Следующий проверяемый шаг:** сначала сопоставимый cooldown у R1 и контроля с повтором этой unfiltered CE на фиксированных строках. Для чистой проверки позднего переобучения нужны две последовательные точки одной ветки с неизменными q/data policy/cooldown. Текущий CPU-анализ не устанавливает пользу дальнейшего обучения, dropout или большего decay.

**Проверка:** все6×8192 примера посчитаны; train **228870**, heldout **215035** target-токенов на модель. CPU-проход занял **19,63мин**, включая подготовку и benchmark. Saved GPU weights SHA совпали у всех шести моделей, counts совпали; максимальное отличие CPU FP32 CE от прежней GPU FP16 CE — **1.36e-05**. Это ожидаемое различие вычислительной точности, не новый GPU-прогон. Argmax PAD/BOS на оцениваемых позициях:0 у всех моделей.

## Протокол

Гипотеза: после нескольких проходов train без dropout обучение всё лучше подгоняет виденные пары, но heldout CE перестаёт улучшаться. Проверяется обычная CE на одном фиксированном наборе строк, а не loss текущего случайного minibatch.

- **Train4096:** равномерная выборка исходных row IDs без возвращения, NumPy seed `2026101503`, из всех **9 725 737** строк `nlp-hw02-compact-cache-verified/2/compact-cache`; 4095 уникальных вопросов. Без long weighting и отбора по loss. По условию диагностики C-long2x уже видела любые эти train-строки минимум дважды.
- **Heldout4k:** прежние 4096 вопросов и исходные compact30 targets, без повторного выбора split. Protected исключает все old1024, включая old991, а также heldout/test/fresh. Fresh/test используются только как строки для exclusion hashes; их качество не оценивается.
- Полный проход question SHA256 всех 60 шардов: **0 protected overlaps, 0 точных дублей heldout-вопросов**. Отдельного train4096 из таких дублей нет. Предыдущая полная [перепись](clean-cooldown-analysis.md) также нашла 0 heldout-overlaps после `lower + collapse whitespace`; нормализация здесь заново не пересчитывалась.
- Все файлы compact-кеша проверены по manifest SHA256. В sampled source заново декодированы вопросы и сверены с cached question SHA; [отдельная проверка выбранных строк](../checks/overfit-check-cpu.json) также дала 0 normalized protected overlaps. IDs, SHA и признаки выбора сохранены в [JSON](../results/overfit-check.json).
- FP16 model-only weights загружены через `model_spec` в `SimpleTransformer(..., dropout=0.)`, `strict=True`, **CPU FP32**, `eval()` и `inference_mode()`, без autocast. Нормировка embedding при загрузке повторно не применяется. Токенизация и `collate` те же, что в `l1_eval.ce`, используемой L2/R1.

Для набора $D$:

$$\operatorname{CE}(D)=-\frac{\sum_{i\in D}\sum_{t=1}^{T_i}\log p_\theta(y_{it}\mid x_i,y_{i,<t})}{\sum_{i\in D}T_i},\qquad G=\operatorname{CE}(\text{heldout})-\operatorname{CE}(\text{train}).$$

$x_i$ — вопрос, $y_i$ — compact target, $T_i$ включает EOS, $\theta$ — параметры модели. BOS подаётся декодеру, но не оценивается; PAD исключён. CE без label smoothing, в натуральных логарифмах; token losses вычислены в FP32, суммы накоплены в FP64. $G>0$ означает худшее предсказание невиденных строк.

**Token accuracy** — доля совпавших `argmax(logits)` с target при **истинном префиксе**, включая EOS и без PAD. Это teacher forcing, а не точность свободной greedy-генерации или исполнения. Оценка программ/exec, sampling и обучение не запускались.

> [!details]- Воспроизведение и проверки
> Из `work/`, Python с Torch/NumPy/tokenizers:
> ```bash
> python research/check-overfit.py --max-seconds 3000
> python research/check-overfit.py --summarize-only
> ```
> Второй вызов пересчитывает только bootstrap по [сохранённым NLL/count/correct для каждой строки](../results/overfit-check-rows.npz); forwards не повторяются. Скрипт сохраняет завершённые splits после каждого прохода. Batch16; сравнены 1/2/4 CPU threads, выбран 4. Реальная проверка трёх строк разной длины: batch/single NLL отличаются максимум на $3{,}96\cdot10^{-6}$, token counts и argmax совпали. Точные веса/модель/таймер/benchmark — в JSON.

## Как интерпретировать

Положительный train/heldout gap показывает разрыв обобщения, но сам по себе не доказывает, что дополнительное обучение уже ухудшает качество. Более сильный признак позднего переобучения — падение train CE при неизменной или растущей heldout CE **при одинаковых данных, objective и cooldown**. Здесь P1/R1 меняют эти факторы: вывод о причине нужно отделять от наблюдаемого разрыва.

Train выбран равномерно по исходным строкам, хотя C-long2x училась с дополнительными предъявлениями длинных targets. Сравнивается качество на виденных строках, а не восстановление weighted training loss. Для проверки влияния длины JSON дополнительно содержит CE по compact target bins `<53 /53–79 /79–105 /≥105` и train CE с token shares этих bins от heldout. Это описательная поправка, она не уравнивает всю сложность вопросов.

95% интервалы: 2000 bootstrap повторов по строкам вопросов; train/heldout ресэмплируются независимо, разности моделей — парно на тех же строках. Интервалы условны на фиксированных weights и многократно использованном heldout, без неопределённости seed/обучения и выбора моделей.

## Dropout и weight decay

**Собственная рекомендация: текущий рецепт автоматически не усиливать; для новой большой модели первым отдельным кандидатом проверить dropout0,05 при прежнем AdamW weight_decay0,01.** Положительного статического gap недостаточно, чтобы объявить нынешнее плато следствием недостаточной регуляризации. Dropout0,1 и повышение decay — следующие отдельные абляции при подтверждённой пользе, а не обязательные условия большей модели.

Dropout случайно выключает часть активаций во время train; при оценке выключается. В Transformer base [Vaswani2017](https://arxiv.org/html/1706.03762v7#S5.SS4) использовался0,1, но это другой режим данных/архитектуры. На нашем [опыте25](lr-dropout-after-normalized-embeddings.md) 12M/500k при одном плане обучения dropout0 превзошёл0,1 на **2,25п.п. heldout exec**, одновременно улучшив CE0,4868→0,4764 и throughput. Это один seed и меньшая модель; результат нельзя переносить на долгий54M+ train, но он не поддерживает безусловный переход на0,1.

Текущая регуляризация уже включает **AdamW weight_decay0,01**. В [PyTorch](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html) вклад decay в один update:

$$\theta\leftarrow(1-\eta_t\lambda)\theta,$$

где $\eta_t$ — LR, $\lambda$ — weight decay, $\theta$ — параметр. Это только часть update, отдельно от градиента CE. Накопленное сжатие зависит от $\sum_t\eta_t$, поэтому коэффициент0,01 сам по себе не говорит о слабом эффекте. Для иллюстрации: при постоянном LR0,0015 за100000updates множитель **только decay**, без градиентов, равен примерно $e^{-1{,}5}=0{,}223$; это не измеренное изменение норм весов. Увеличивать decay при уже ухудшившейся raw train CE без отдельного сравнения не обосновано; если проверять, сначала0,01 против0,03, сохранив dropout и schedule.

Для сравнения dropout0 против0,05 у новой модели: один split, одинаковые initial seed/порядок/число предъявлений/LR/cooldown; промежуточные **unfiltered train4096 и heldout CE в eval()**, плюс время шага. Dropout вносится параметром конструктора `SimpleTransformer`, с сохранением новой recipe identity; одного `model.train()` для изменения0→0,05 недостаточно. При фиксированном времени более медленный вариант получит меньше updates — эту цену проверять отдельно. Успех — лучшая heldout CE при приемлемой скорости, а не только меньший gap за счёт ухудшения train. Эти абляции здесь не запускались; exec-выигрыш по token CE не устанавливается.

## Git

Попытка подготовить отдельный коммит остановилась на записи shared worktree `index.lock`: `Operation not permitted` в sandbox. Индекс не изменён, коммит не создан. [Квитанция с точными путями](../checks/overfit-check-git.json) и [патч только новых навигационных строк](../checks/overfit-check-navigation.patch) сохранены отдельно от параллельных правок R1/архитектурного эксперимента.

## Источники

- **nlp-hw02-overfit-check:** собственные измерения 15.10.2026; [скрипт](../research/check-overfit.py), [агрегаты/IDs/SHA](../results/overfit-check.json), [NLL/count/correct](../results/overfit-check-rows.npz). Первичные FP16 inputs — локальные L3/P1 final и R1 `1/model-GPU{0,1}/weights/final-fp16.pt`. Файлы параллельного разбора R1 и его исходные артефакты не изменялись.
- [l2-results](l2-results.md), [l3-results — Финал L3](l3-results.md#финал-l3), [p1-results](p1-results.md), [r1-plan](r1-plan.md), [lr-dropout-after-normalized-embeddings](lr-dropout-after-normalized-embeddings.md) — локальная история рецепта, cooldown и прежняя абляция dropout; bibliography.
- Srivastava et al. [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html), JMLR 2014: случайное выключение units при обучении как регуляризация. Полезность конкретного $p$ на ДЗ2 из статьи не следует.
- Vaswani et al. [Attention Is All You Need, §5.4 и Table3](https://arxiv.org/html/1706.03762v7#S5.SS4), 2017: residual/embedding dropout 0,1 у Transformer base; абляция без dropout ухудшила dev PPL/BLEU. Это перевод и другая архитектура, не прямой подбор $p$ для текущей модели.
- Loshchilov, Hutter. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), ICLR 2019: AdamW отделяет decay от градиента loss; оптимальный decay зависит и от бюджета обучения. [PyTorch2.11 AdamW](https://docs.pytorch.org/docs/2.11/generated/torch.optim.AdamW.html) — фактическая формула decay в используемом training stack.
