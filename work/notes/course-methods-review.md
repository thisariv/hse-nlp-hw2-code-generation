---
type: reference
course:
  - nlp
status: complete
understanding: 3
created: 2026-10-12
updated: 2026-10-12
sources:
  - https://github.com/ashaba1in/hse-nlp/tree/afb4290ba2155fc826758253ec46fcb368edd49c
  - https://arxiv.org/abs/1706.03762
  - https://arxiv.org/abs/1804.00247
  - https://arxiv.org/abs/1906.02629
  - https://arxiv.org/abs/2501.00656
  - kaggle-l1
  - kaggle-s3
aliases: []
---

# ДЗ2: методы курса и выбор опыта для GPU1

**Топ-3 направлений:** 1) продолжение C с повышенным весом длинных задач; 2) независимая C для голосования; 3) продолжение C с label smoothing 0.05.

**Рекомендация:** на GPU1 коротко сравнить обычное продолжение C, повышенный вес длинных пар и smoothing из одного stable anchor. Продолжить победителя до конца сессии. Вторая C с новым seed — **альтернативное использование всей сессии**, если идею уже не успеем перенести на GPU0 и две C укладываются в180с. Усреднение checkpoints и beam2 проверить отдельно на готовых весах: им не нужно11ч обучения.

## Исходное состояние и уже проверенное

- **Из последнего сообщения пользователя:** C, 6+6/d512, около54M, 62.9% exec heldout4k после2.5ч чистого обучения; GPU0 продолжает C в L2 около11ч. D отстаёт и остановилась на NaN. Новые журналы здесь не скачивались. Записи «боевой L1 ещё не запущен» в [l1-long-training](l1-long-training.md) и [l1-review](l1-review.md) относятся к более раннему состоянию.
- **Локальный рецепт:** `l1_runtime.py`, `exp26_core.py`, `l1_cache.py`: C53 646 403; Pre-LN, learned positions2048, SwiGLU1368, общий encoder/decoder embedding, отдельная output head. Нормированная token-init, dropout0, fused AdamW, wd0.01, default betas(0.9,0.999)/eps1e-8, clip1, effective batch128, FP16/GradScaler, micro64, warmup1000→constant LR0.0015 и отдельные cooldown-ветки. CE нормирована по всем непустым токенам effective batch; smoothing отсутствует. Кеш9 725 737 строк, compact30, vocab8195, protected exclusions сохранены.
- **Не повторять:** размеры A/B/C/D, Post→Pre-LN, SwiGLU, byte/BPE4k/BPE8k с цифрами по одной, KV-кеш, AMP/bucketing/batch128, token-init scale, dropout0, LR/cosine/WSD, finite-reference фильтр/больше данных, compact target, placeholders, retrieve-and-substitute, sampling/температуры/exec-vote, эвристики выбора и B scoring.
- SDPA и быстрые attention-пути уже есть в23/27b/L1. **Нельзя считать SDPA новым опытом FlashAttention.** Конкретный backend зависит от T4, форм и масок; скорость из лекции к C не переносится автоматически.
- [data-research-ceiling](data-research-ceiling.md): 90.06% — ретроспективная мода на finite-дублях, **не доказанный test ceiling**. У A только46/1558 ошибок были runtime/syntax/nonnumeric/timeout; исправление грамматики не закроет большинство неверных вычислений. Совпадение набора чисел при неверном ответе — косвенный признак структурной ошибки. Эти доли измерены для A, не для нынешней C.
- [candidate-selection-cpu](candidate-selection-cpu.md), [32-candidate-ensemble-s3-results](32-candidate-ensemble-s3-results.md): oracle-gap большой, но эвристики и score B его почти не закрыли. **Новая модель, генерирующая собственные кандидаты, отличается от B scoring старого пула A.**

Связи: decoding-strategies, relative-position-representations, transformer-training-stability, transformer-training-efficiency, bibliography.

## Полнота обзора

Публичная ветка **2026**, snapshot `afb4290ba2155fc826758253ec46fcb368edd49c`, прочитан2026-10-12. Проверены **все15 файлов**: README, две домашки, шесть лекционных PDF, семинары1/2/3/5/6 в ipynb и семинар4 в PDF. Всего365PDF-страниц и418ячеек ноутбуков. Ячейки не исполнялись. Все45страниц S4 дополнительно просмотрены визуально: большая часть текста там встроена картинками. [Опись/SHA/ссылки](../checks/course-methods-review-sources.json).

| Материал | Что проверено |
|---|---|
| [L1][l1], [S1][s1], [HW1][hw1] | N-граммы/префиксное дерево, RNN/LSTM, units/BPE/WordPiece, teacher forcing, CE, температура и метрики. |
| [L2][l2], [S2][s2] | Attention/encoder–decoder, позиции, beam, temperature/top-k/top-p, length normalization. S2 beam: cells97–110. |
| [L3][l3], [S3][s3] | BERT/RoBERTa/ALBERT/GPT, фильтрация/дедупликация, AdamW/LAMB/Adafactor, batch/warmup/clipping/AMP. S3 masks/ELECTRA: cell22. |
| [L4][l4], [S4][s4] | Flash/KV/MQA/GQA/SWA, RPE/RoPE/ALiBi, GLU/MoE/RMSNorm; data mixing, activation checkpointing, DP/ZeRO/FSDP/TP/PP/CP/EP, async checkpoints, repetition filter. |
| [L5][l5], [S5][s5] | Full FT/LP→FT, prompts/adapters/LoRA/BitFit/QLoRA. |
| [L6][l6], [S6][s6] | Distillation, dynamic/static/QAT, calibration/granularity, ZeroQuant/LLM.int8/GPTQ/AWQ, speculative decoding, fusion. |
| [HW2][hw2] | Encoder–decoder с нуля; разрешённый датасет; без дополнительных данных/таргетов; test≤180с на одной T4. |

**Граница «все»:** ниже реестр из79семейств: полный обзор относящихся к задаче семейств из опубликованного набора и практических рецептов основных связанных работ. Подварианты одного механизма сгруппированы; произвольные значения гиперпараметров не отдельные методы. **П** — расширение из первоисточника, не отдельный пункт слайда. References внутри статей рекурсивно не обходились. Ссылки на записи перечислены в README; сами записи не прослушаны, устные дополнения не покрыты. Материалы2025/main и будущие недели не приписаны2026.

> [!note] Неточности материалов курса
> L6/41 называется ZeroQuant, но ссылка1909.05840 ведёт на **Q-BERT**; настоящий ZeroQuant —2206.01861 на соседних слайдах. Рассмотрены оба. S5/cell33 упоминает LoRA K/V, тогда как типичный опыт оригинала использует Q/V. Утверждение L2/43 «beam всегда лучше» не гарантия exec. WordPiece `uncased` из S2 разрушает регистр/пробелы Python и буквально не переносится.

## Обозначения и цена

**Д** — допустимо с исходными train-парами и Transformer encoder–decoder с нуля; **С** — серая трактовка, в план не включать до уточнения; **З** — противоречит ограничениям. PEFT/квантизация допустимы на **собственной C**, не на чужих pretrained weights и calibration data.

**Все диапазоны Δexec — наши предварительные гипотезы**, не измерения курса, не доверительные интервалы и не BLEU, переведённый в проценты. Обычно это общий exec при равном wall-бюджете и прежнем декодировании. Для decoding указан свой контроль; speed-only методы напрямую дают0. Отрицательные исходы возможны и для интервалов0…+1. Эффекты нельзя складывать.

Цена реализации дана в часах работы с L1. GPU-цена — на **одну T4 и один вариант**, с оценками; отдельный matched control оплачивается дополнительно. **F** = fresh8–10ч до полезной модели с ранним gate1–2ч, входящим в эти8–10ч. **B** = продолжение из stable C-anchor1–2ч для раннего gate. Короткий gate не доказывает финального качества. CPU-подготовка/код не расходуют11ч GPU-сессии.

> [!info]- Полный реестр:79 семейств и вариантов
> ## Обучение, данные и регуляризация
> 
> | ID / метод: что это | Курс → первоисточник | Применимость; Δexec и обоснование | Реализация; GPU-проверка |
> |---|---|---|---|
> | D01 **Oversampling длинных пар:** чаще предъявлять существующие сложные пары, target прежний. | [S4/7,37][s4] → [OLMo2,§4][olmo]; локальная идея в [data-research-ceiling](data-research-ceiling.md) | **Д**, общий−1…+3п.п., ≥90BPE+2…+6. Больше обучения слабого среза; длина — неточный proxy сложности. | 2–4ч +CPU order; B, затем5–7ч при успехе. Длинные batches могут снизить пар/с. |
> | D02 **Curriculum / late-stage mixing:** менять веса срезов по стадиям обучения. | [S4/37][s4] → [OLMo2,§2.3/4][olmo] | **Д**,−1…+2. Поздняя специализация после базовой арифметики; внешние/synthetic источники OLMo исключены. Это отдельная абляция от постоянного D01. | 2–4ч; B. |
> | D03 **Exact/near dedup:** удалить повторные пары либо ограничить вес частых вопросов; MinHash ищет близкие тексты. | [L3/34][l3] → [Lee22][dedup] | **Д**,−1…+1. Разнообразие растёт, но повторы math полезны; разные корректные программы одного вопроса не безусловный мусор. Finite filtering уже был, dedup нет. | 4–8ч CPU; B или F. |
> | D04 **Quality/repetition filter:** отбор имеющихся пар по повторяющимся n-граммам, простым признакам или консенсусу имеющихся reference. | [L3/35][l3], [S4/32][s4] → [OLMo2,§3.1][olmo] | **Д**,0…+1. Убирает часть шума/spikes, но повторы Python могут быть нужны. Консенсус обсуждался, не обучался. Новые judge/model labels — **З**. | 4–8ч CPU; B. |
> | D05 **Label smoothing (П):** CE с небольшой равномерной регуляризацией вместо чистой one-hot CE. | [L2/13][l2] → [Transformer,§5.4][transformer], [Szegedy16,§7][smooth], [Müller19][smooth-analysis] | **Д** как аналитическая регуляризация того же reference, без учителя/новой семантики.−1…+2. Может улучшить калибровку на шуме, ухудшить цифры/sampling. Начатьε0.05. | 1–2ч; B, продолжить при успехе. Инференс прежний. |
> | D06 **Inverse-square-root LR (П):** после warmup LR∝step^−1/2. | [L2/13][l2], [L3/39][l3] → [Transformer,§5.3][transformer], [Popel18,§4.6][tips] | **Д**,−1…+1. Новый schedule, но LR/cosine/WSD уже исследованы; stable+cooldown удобнее неизвестного горизонта. | 1–2ч; B с общей точкой старта. |
> | D07 **Token-budget batch / рост batch (П):** фиксировать токены вместо128пар; отдельно увеличивать effective batch вместо уменьшения LR. | [L3/40][l3], [S4/6][s4] → [Popel18,§2.2/4.5/4.8][tips], [Smith17][batch-growth] | **Д**,−1…+1. Выравнивает число токенов update, меняет шум оптимизации. Просто увеличить microbatch — не новый quality-фактор. | 3–6ч; B. Меняются cursor/LR-per-token/resume; не смешивать с D01. |
> | D08 **Adamβ2/eps (П):** быстрее обновлять второй момент, изменить стабилизатор знаменателя. | [L3/41][l3] → [Transformer,§5.3][transformer], [Popel18,Table4][tips] | **Д**,−1…+1. β2=0.98 отличается от текущего0.999; может помочь нестационарным градиентам. eps1e-8 уже совпадает с улучшенным OLMo-рецептом. | 1–2ч; B/F. Миграцию moments или reset явно зарегистрировать. |
> | D09 **Weight-decay groups (П):** исключить embeddings, отдельно norm/bias из wd. | [L3/41][l3], [S4/32][s4] → [OLMo2,§3.4.2][olmo] | **Д**,0…+1. В L1 decay всех весов; слишком малые embedding norms могут вредить стабильности. Сначала измерить norms, не объявлять причиной NaN D. | 1–3ч; B. Общий wd0.01 не повторять. |
> | D10 **Adafactor:** row/column вторые моменты, update clipping/relative scaling, опционально без momentum. | [L3/41][l3] → [Shazeer18][adafactor] | **Д**,−1…+1. Экономит optimizerRAM, но C помещается; потеря fusedAdamW может замедлить. | 3–6ч;0.5ч speed +B/F. State несовместим с Adam. |
> | D11 **LAMB:** layerwise trust ratio масштабирует адаптивный шаг для очень большого batch. | [L3/41][l3] → [You19][lamb] | **Д**,скорее0/минус при128парах. Режим десятков тысяч примеров BERT отсутствует. | 3–6ч; B +LR/batch tuning, сейчас не оправдано. |
> | D12 **Checkpoint averaging (П):** среднее весов соседних поздних состояний одной траектории. | [L2/13][l2], [L3/39][l3] → [Transformer,§6.1][transformer], [Popel18,§4.10][tips] | **Д**,0…+1. Сглаживает колебания, финальный инференс одной C. Малое позднее окно; не смешивать ранние и поздние состояния. | 1–3ч; CPU average +0.1–0.3ч eval. Сохранить snapshots до ротации. |
> | D13 **Forked checkpoint soups (П):** среднее коротких продолжений из одного anchor с разным порядком/рецептом. | [L3/39][l3], [S4/32][s4] → [Popel18,§4.10][tips], [OLMo2,§4.3][olmo], [Wortsman22][soups] | **Д**,−1…+2. Можно объединить specialist/base без двух C на инференсе; один basin надо проверить. **Fresh независимые seeds по весам не усреднять.** | 2–4ч; два продолжения по1–2ч +0.1–0.3ч eval. Готовые ветки удешевляют проверку. |
> 
> ## Архитектура и токенизация
> 
> Обычно требуется fresh модель или специальная миграция. Позиционный механизм готовой C нельзя поменять одной заменой функции без дообучения. **Длинная структура вычислений не равна выходу за context2048:** длинные задачи здесь часто90–120BPE; аргумент экстраполяции контекста сам по себе слаб.
> 
> | ID / метод: что это | Курс → первоисточник | Применимость; Δexec и обоснование | Реализация; GPU-проверка |
> |---|---|---|---|
> | A01 **Sinusoidal positions (П):** фиксированные синусы/косинусы вместо learned table. | [L2/23][l2] → [Transformer,§3.5][transformer] | **Д**,−1…+1. Поздние позиции не требуют обучения; исходная статья не показывает большого преимущества над learned. | 2–4ч, включая KV; F. |
> | A02 **Shaw relative K/V:** embeddings расстояния добавляются к ключам/значениям. | [L4/19–21][l4] → [Shaw18][shaw] | **Д**,−1…+2. Relational bias полезен, расстояние не равно вычислительной связи. L²d/RV может замедлить SDPA. | 8–16ч, training/cache; F. |
> | A03 **T5 relative bias (П):** learned scalar bias по bucket расстояния; bidirectional encoder и causal decoder. | [L4/19][l4] → [T5,§2.1][t5] | **Д**,−1…+2. Проще Shaw, учитывает относительный порядок. Cross-attention сначала без bias; T5 denoising не переносить. | 6–12ч; F, проверить маски/backend. |
> | A04 **RoPE:** поворачивать Q/K по позиции в self-attention, хранить повернутые K в кеше. | [L4/22–27][l4] → [Su21][rope] | **Д**,−1…+2. Может помочь композиции; не применять общую positional axis механически к question↔code cross-attention. | 8–16ч; F. Кандидат следующего цикла, не безопасный короткий patchC. |
> | A05 **RoPE interpolation:** сжать индексы при расширении уже обученного контекста. | [L4/28–29][l4] → [Chen23][pi] | **Д** на собственной RoPE-модели; здесь≈0: C без RoPE, overflow2048 не установлен. | После A04:3–6ч;0.5–2ч adaptation. Сейчас отдельного смысла нет. |
> | A06 **ALiBi:** фиксированный штраф расстояния для каждой головы. | [L4/30–32][l4] → [Press21][alibi] | **Д**,−1…+1. Экстраполяция полезна, но distance penalty может мешать дальним math-связям. Encoder нужен симметричный вариант. | 6–12ч; F, проверить backend с bias. |
> | A07 **RMSNorm:** нормировать по RMS без вычитания среднего. | [L4/50–52][l4] → [Zhang19][rms] | **Д**,−1…+1. Скорее скорость/стабильность.25% на операции norm не25% всего training, fusedLN уже быстр. | 4–8ч, включая cache;0.5ч speed +F. |
> | A08 **pRMSNorm:** оценивать RMS по части hidden coordinates. | [L4/52][l4] → [Zhang19][rms] | **Д**,≈0/минус: при d512 экономия мала, новая погрешность. | После A07:1–2ч;0.5ч speed +B/F. |
> | A09 **MQA:** одна KV-голова при нескольких Q-головах. | [L4/13–15][l4] → [Shazeer19][mqa] | **Д**,greedy−2…0. Качество обменивается на экономию KV; через больше samples может помочь после180с gate. | 8–16ч, custom attention/cache; F. |
> | A10 **GQA / MHA→GQA uptraining:** группы Q разделяют KV; можно усреднить KV weights собственной C и дообучить. | [L4/13–15][l4] → [Ainslie23][gqa] | **Д**,greedy−1…0. Меньше риск качества, чем MQA.5% uptraining из статьи не оценка нашего времени/T4. | 8–16ч;1–3ч conversion gate либо F, затем sampling timer. В L1 не готово. |
> | A11 **Sliding/block attention, rollingKV, chunked prefill:** окно/блоки self-attention и экономия длинного кеша. | [L4/16][l4] → [Mistral,§2][mistral] | **Д**,скорее0/минус: C имеет короткие контексты и дальние связи. Rolling/chunked напрямую дают0exec. | 8–16ч;0.5ч speed +B/F. |
> | A12 **GEGLU/ReGLU/GLU/bilinear/GELU (П):** другая activation/gate вместо SwiGLU. | [L4/35–36][l4] → [Shazeer20][glu] | **Д**,−1…+1. Нет сильного аргумента против хорошей SwiGLU; сравнить одинаковый FFN budget. | 2–4ч; F на выбранный вариант. |
> | A13 **SparseMoE top1/top2:** router выбирает FFN-экспертов для каждого токена. | [L4/37–41][l4] → [Shazeer17][moe], [Switch][switch] | **Д**,положительныйΔ при total params≤C не обоснован: меньше active capacity, dispatch overhead на одной T4. | 1–3дня; F может не хватить для вывода. Не выделять GPU1. |
> | A14 **Noisy router/jitter:** шум распределяет градиенты между экспертами. | [L4/42–43][l4] → [Shazeer17][moe] | **Д**,только с A13; лечит collapse, отдельныйΔexec неизвестен. DenseC не улучшает. | 2–4ч с готовой MoE;1–2ч router gate. |
> | A15 **MoE load balancing/capacity:** регуляризация перекоса нагрузки и предел токенов эксперта. | [L4/44][l4] → [Switch,§2][switch] | **Д** как регуляризация без новых semantic labels; только A13. Token dropping может испортить код. | 3–6ч;1–2ч router gate. |
> | A16 **Factorized embeddings:** V×d заменить V×e и e×d, e<d. | [L3/21][l3] → [ALBERT][albert] | **Д**,−1…0 напрямую. Экономит параметры, но рискует copy/digit capacity; перераспределение в новые глубины — отдельный опыт. | 4–8ч; F. |
> | A17 **Cross-layer sharing:** один Transformer block применяется несколько раз. | [L3/22][l3] → [ALBERT][albert] | **Д**,−2…+1. Экономит weights, **не FLOPs**; каждый проход требует своего KV-слота. | 6–12ч; F. |
> | A18 **Output weight tying (П):** связать decoder embedding с output head. Encoder/decoder embedding уже общая. | [L2/13][l2] → [Transformer,§3.4][transformer], [Press16][tying] | **Д**,−1…+1. Экономит8195×512=4 195 840weights, около7.8%C. Это новое ограничение, не гарантированный выигрыш от экономии. | 3–6ч; F. Простое копирование head в embedding готовой C некорректно. |
> | A19 **QK-normalization (П):** нормировать Q/K перед dot product. | [S4/32][s4] → [OLMo2,§3.3.2][olmo], [Wortsman23][instability] | **Д**,0…+1 у стабильной C; может спасти unstable run. Причина NaN D ещё не установлена. | 6–12ч, включая cache; B/F со stability gate. |
> | A20 **Output/nonparametric norm (П):** нормировать выход подслоя до residual addition; отдельно norm без обучаемых weights. | [S4/32][s4] → [OLMo2,§3.3.1–2][olmo] | **Д**,−1…+1. Это не испытанный Post-LN на residual-сумме; новая динамика, слабое seq2seq доказательство. | 6–12ч; F. |
> | A21 **Softmax z-loss (П):** малый штраф квадрату logsumexp выходных logits. | [S4/32][s4] → [OLMo2,§3.3.3][olmo], [Wortsman23][instability] | **Д** как аналитическая регуляризация;0…+1, скорее стабильность. Считать в FP32; без teacher logits. | 2–4ч; B. |
> | A22 **Full projection initialization (П):** контролируемый масштаб attention/FFN/residual weights, например truncated normal. | [S4/32][s4] → [OLMo2,§3.2][olmo] | **Д**,−1…+1. Отличается от token-init;0.02 больших LLM не универсально d512, текущая C учится. | 3–6ч +CPU gradient audit; F. |
> | A23 **Число голов при фиксированном d (П):** поменять h без изменения ширины/числа weights. | [L2/18–20][l2] → [Transformer,Table3][transformer] | **Д**,−1…+1. Размеры A/B/C/D уже проверены, чистая head ablation нет; структурный скачок не обоснован. | 3–6ч; F. |
> | T01 **Свой WordPiece:** longest-match word-internal segmentation вместо BPE. | [S1/cell19][s1], [S2/cells10–15][s2] → [официальное описание][wordpiece] | **Д** для своего train-only lossless варианта;−1…+1. Готовый BERT uncased — **С/не подходит**: внешнее обучение и потеря регистра/пробелов. | 4–8ч +CPU recache; F. Не «тот же кеш». |
> | T02 **Новые BPE настройки (П):** V16k или balanced question/code corpus с прежним digit regex. | [S1/cells13–23][s1] → [Sennrich16][bpe] | **Д**,−1…+1.4k/8k/digits уже были. Большая V сокращает длину, увеличивает head и отнимает capacity под C-cap. | 4–8ч +CPU recache; F. |
> | T03 **Разные source/target vocabularies:** encoder отдельно, head только по кодовому словарю. | [S2/cells10–11][s2] → [BPE/NMT][bpe], [GNMT,§3][gnmt] | **Д**,−1…+1. Меньше head, но исчезает shared embedding связь копирования; два train-only tokenizer. | 6–12ч +CPU recache; F. |
> 
> ## Декодирование и ансамбли
> 
> | ID / метод: что это | Курс → первоисточник | Применимость; Δexec и обоснование | Реализация; GPU-проверка |
> |---|---|---|---|
> | I01 **Beam2/4, length/coverage penalty:** несколько префиксов, ранжирование законченных кодов. | [L2/36–43][l2], [S2/cell98][s2] → [GNMT,§7][gnmt] | **Д**,−1…+2 к greedy. Вероятность не correctness; coverage всех слов/чисел вреден при лишнем контексте. A probability-sum выбирал неверные короткие коды. | 4–8ч, cached beam reorder;0.2–0.5ч eval+timer, **без обучения**. Медленный per-exampleS2 не production. |
> | I02 **Top-k filtering:** сэмплировать только из k наиболее вероятных tokens. | [L2/47–48][l2] → [Holtzman20][nucleus] | **Д**,−1…+1 кобычному vote при равном числе samples. Убирает хвост ошибок и правильные редкие operators/names. | 1–2ч;0.2–0.5ч. Температуры уже были, top-k нет. |
> | I03 **Top-p/nucleus:** минимальный token-пул с суммарной массой p. | [L2/49][l2] → [Holtzman20][nucleus] | **Д**,−1…+1 к vote при равном K. Адаптивнее top-k; разнообразие не равно верным кластерам. | 1–3ч;0.2–0.5ч. Сортировка V может стоить времени. |
> | I04 **Token-level probability ensemble:** среднее распределений двух моделей на каждом общем префиксе. | [L2/13][l2], [L3/39][l3] → [GNMT,§8][gnmt], [Popel18][tips] | **Д**,0…+2 к single-model greedy. Другой механизм, чем score B готовых candidates. Два decode шага/токен, два кеша, одинаковый tokenizer. | 6–12ч;0.2–0.5ч eval с готовыми моделями +timer; второе обучение отдельно. |
> | I05 **Независимая C +candidate vote:** два генератора дают свои пулы, затем обычное exec-vote. | [L2/13][l2] → [NMT ensembles][transformer], [Hinton15,введение][distill] | **Д**,0…+2 **сверх single-model vote при равном K**, не ещё+4. Корреляция/более молодая C2 могут дать0/минус. | 1–3ч параметризовать seed; F +0.5–1ч ансамбль/timer. Протокол ниже. |
> | I06 **Exact speculative decoding:** свой малый draft предлагает блок, C проверяет; acceptance/residual resampling сохраняют распределение. | [L6/50–56][l6] → [Leviathan23][speculative] | **Д** если draft училась только на original pairs, без distillation. Прямой exec0; больше K в180с может помочь. Speedup при большом batch неизвестен. | 1–3дня: block validation, caches/rollback;1–2ч inference gate с готовым совместимым draft. Старый A raw-target не совместим с compactC token stream. |
> | I07 **N-gram/prefix lookup draft (П, наш перенос):** предложения блока из train-only n-грамм/source-prefix lookup. | [L1/37–44][l1], [HW1][hw1], [L6/54][l6] → [Leviathan23][speculative] | **Д** как draft с C verification, не замена Transformer. Прямой exec0; шаблонный код может ускорить принятие, ошибки C сохранятся. | После I06:8–16ч;0.5–1ч speed/identity. Обучение не нужно. |
> | I08 **Train-only few-shot context:** encoder получает существующие question/code примеры, target текущей пары прежний. | [L3/26][l3], [L5/17–18][l5] → [Brown20][gpt3] | **Д**,inference-only−2…+1: C такому input не училась. Training retrieved context — неиспытанный перенос, **не повтор** retrieve-and-substitute; новые targets не нужны. | 6–12ч,index/exclusions; F с новыми inputs/recache, retrieval включить в timer. Отдельного RAG-блока в2026 нет. |
> 
> Grammar/AST-constrained decoding как метод курса в2026 не найден. Внешний опыт возможен, но низкая доля syntax/runtime ошибок A и compact target ограничивают ценность; в shortlist не входит.
> 
> ## Эффективность и сжатие
> 
> Прямой exec здесь обычно0. Польза — больше обучения/кандидатов или устойчивость. Для каждого нужен профиль **C на T4**, не speedup большой LLM из статьи.
> 
> | ID / метод: что это | Курс → первоисточник | Применимость / эффект | Реализация; GPU-проверка |
> |---|---|---|---|
> | E01 **Activation checkpointing/selective rematerialization:** пересчитать activations в backward вместо хранения. | [S4/14–15][s4] → [PyTorch][checkpointing] | **Д**,прямо0, при равном wall часто хуже; Cmicro64 помещается. Selective вариант может разрешить выгодный microbatch. | 4–8ч;0.3–0.5ч throughput/peaks/resume, не11ч. |
> | E02 **torch.compile/min-cut (П):** компиляция, fusion и выбор сохраняемых activations. | [S4/15][s4] → [PyTorch compile/AC][checkpointing] | **Д**,прямо0; только throughput. Bucketing shapes, cache mutations/fallback могут дать recompile/graph breaks. | 6–12ч;0.5–1ч bench +numerical gate. Compile/load внутри180с. |
> | E03 **Kernel/operator fusion:** слить pointwise/norm/CE; fusedAdamW/SDPA уже есть. | [L6/36–37][l6] → [ZeroQuant][zeroquant], [OLMo2,§6.4][olmo] | **Д**,прямо0; может уменьшить memory traffic/launch overhead, не гарантированно ускоряет короткие matricesC. | 8–16ч;0.5–1ч profile/bench/numerical gate. |
> | E04 **Async checkpoints/bookkeeping:** независимая CPU-копия state пишется параллельно training. | [S4/33–34][s4] → [PyTorch][async-checkpoint], [OLMo2,§6.4][olmo] | **Д**,прямо0; экономит измеренный I/O. Живые Adam/weights писать параллельно изменению нельзя; receipt после завершения. Асинхронная live-публикация L1 — не тот же механизм. | 6–12ч;0.3–0.5ч timing/failure gate. |
> | E05 **Меньше host-device sync / async metrics (П):** считать метрики пачками, реже`.item()`/sync. | [S4/31][s4] → [OLMo2,§6.4][olmo] | **Д**,прямо0. L1 sync нужны таймерам/NaN checks; speed mode отделить от accounting, проверки успеха update сохранить. | 3–6ч;0.3–0.5ч bench. |
> | E06 **Sequence packing (П):** уплотнить пары без PAD, block-diagonal masks и reset positions. | [S4/6,16][s4] → [HF packing implementation][packing] | **Д**,прямо0; seq2seq нужны отдельные source/target/cross blocks, masks могут отключить fast backend. Bucketing уже уменьшает PAD. | 1–2дня;0.5–1ч throughput/cross-pair leakage gate. L1collate/core меняются. |
> | E07 **H2D overlap/pinned buffers/prefetch (П):** готовить/передавать следующий batch параллельно текущему. | [S4/31][s4] → [PyTorch transfer guide][pinmem] | **Д**,прямо0; смысл после профиля collate/H2D при mmap-кеше. | 4–8ч;0.3–0.5ч bench/cursor identity. |
> | E08 **DP/DDP:** реплики одной модели и all-reduce gradients. | [S4/19–21][s4] → [PyTorchDDP][ddp] | **Д**,но одна свободная GPU1 не даёт применения; GPU0 занята C. Финальный инференс не на двух T4. | 1–2дня;≥2GPU, вне данного бюджета. |
> | E09 **ZeRO1/2/3,FSDP:** шардировать optimizer/gradients/weights. | [S4/22–24][s4] → [Rajbhandari20][zero], [PyTorchFSDP][fsdp] | **Д**,меж-GPU экономия на одной T4 равна0; C помещается. | 1–2дня;multi-GPU gate вне задачи. |
> | E10 **Tensor parallelism:** части matrices слоя на разных GPU. | [S4/25–26][s4] → [Megatron][megatron] | **Д**,пользы на одной T4/≤C нет; communication overhead. | 1–3дня;≥2GPU, вне задачи. |
> | E11 **Pipeline parallelism/1F1B:** разделить слои по GPU, перекрыть microbatch passes. | [S4/27,41][s4] → [GPipe][gpipe], [Playbook][playbook] | **Д**,на одной свободной GPU не применим; smallC/pipeline bubbles. | 1–3дня;≥2GPU, вне задачи. |
> | E12 **Context parallelism/RingAttention:** разделить token blocks, обмениваться KV. | [S4/28,44][s4] → [Liu23][ring] | **Д**,длинного context bottleneck у C нет; меж-GPUring не помогает одной T4. | 2–3дня;≥2GPU, вне задачи. |
> | E13 **Expert parallelism:** MoE experts на разных GPU, dispatch/combine. | [S4/42][s4] → [GShard][gshard] | **Д**,только A13+multi-GPU, здесь не применим. | 2–3дня;multi-GPU, вне задачи. |
> | E14 **CPU offload / lowbit optimizer states (П):** state в CPU либо менее точные moments. | [S4/22][s4], [S6][s6] → [ZeRO-Offload][offload], [bitsandbytes][bnb] | **Д**,прямо0, скорее wall хуже: AdamC не критичен для16GB. Quantized moments не равно испытанной AMP. | 4–8ч;0.5ч speed/memory +B. |
> | Q01 **DynamicINT8:** activation ranges вычислять на проходе, weights с фиксированными scales. | [L6/24–26][l6], [S6/cells29–52][s6] → [LLM.int8][llmint8] | **Д**,−1…0 к FP16greedy до проверки. Память меньше,decode может замедлиться; TorchCPU quantizationS6 не CUDA ускоритель. | 6–12ч custom linear/export;0.5–1ч quality/timer. |
> | Q02 **StaticINT8/calibration:** train-only minmax/moving average/histogram-KL; per-tensor/channel/group, symmetric/asymmetric. | [L6/19–29,40][l6], [S6/cells53–61][s6] → [ONNXRuntime][ort-quant], [Q-BERT][qbert] | **Д**,−1…0 к FP16. Granularity меняет ошибку/metadata; per-channel обычно точнее. Чужой calibration dataset исключён. | 8–16ч;0.5–1ч calibration/quality/timer на ограниченный config. |
> | Q03 **QAT:** fake quant/STE в training настраивает weights под lowbit inference. | [L6/25–26][l6] → [Jacob18][qat] | **Д** при original target loss;≈0/минус к FP16, может улучшить наивную quantization. C не обязательно требует. | 1–2дня;1–3ч fine-tune +deployment timer. |
> | Q04 **LLM.int8 outliers:** FP16 для outlier channels, vectorwiseINT8 для остальных. | [L6/43–45][l6] → [Dettmers22][llmint8] | **Д**,−1…0 к FP16. Outlier режим54M не обязательно похож на largeLLM;1.5×семинара не прогноз. | 6–12ч;0.5–1ч quality/timer. |
> | Q05 **ZeroQuant:** fine-grained quantization+fusion, опционально layer distillation. | [L6/37–42][l6] → [Yao22][zeroquant] | **Д** без LKD, **З** для distillation.−1…0 к FP16; speedupT4 не установлен. | 1–2дня;0.5–1ч deployment gate, сложный порт C. |
> | Q06 **GPTQ:** lowbit weights с second-order reconstruction ошибок предыдущих quantized weights. | [L6/46–47][l6] → [Frantar22][gptq] | **Д**: activation/Hessian calibration не новый семантический target.−1…0 к FP16; custom seq2seq не готовый OPT. | 1–2дня;0.5–2ч quant/calibration/quality/timer. |
> | Q07 **AWQ:** масштабировать salient channels перед weight-only quantization по train activations. | [L6/48–49][l6] → [Lin23][awq] | **Д**,−1…0 напрямую; память/больше K — косвеннаяпольза.4×speedup больших LLM не прогноз C. | 1–2дня;0.5–2ч quant/quality/timer. |
> | Q08 **Q-BERT mixed precision:** Hessian подсказывает битность групп weights. | [L6/41:назван ZeroQuant][l6] → [Shen19][qbert] | **Д** без distillation;−1…0 к FP16. Устарее GPTQ, encoderBERT не наш decoder. | 1–2дня;1–2ч quant/quality/timer, низкий приоритет. |
> 
> ## PEFT и ограничения правил
> 
> | ID / метод: что это | Курс → первоисточник | Применимость; Δexec и обоснование | Реализация; GPU-проверка |
> |---|---|---|---|
> | P01 **LP/LP→FT:** сначала head замороженной модели, затем все weights. | [L5/7–14][l5], [S5/cells26–32,53–57][s5] → [Kumar22][lpft] | **Д** на C,но новой head/задачи нет;≈0/минус к full continuation. Чужой pretrained base — **З**. | 3–6ч;0.5–1ч gate. |
> | P02 **Adapters:** малые bottleneck residual modules, base заморожена. | [L5/22–23][l5] → [Houlsby19][adapters] | **Д** на C при total params≤C budget;−1…+1 к full FT. Добавляют serial inference, мало оснований исправить структуру. | 6–12ч;1–2ч specialist gate. |
> | P03 **Soft prompt tuning:** обучать virtual input embeddings замороженной C. | [L5/19–21][l5], [S5/cells43–51][s5] → [Lester21][prompt] | **Д** на C;−2…0 к full FT. Сильный аргумент статьи — large T5, не54M; удлиняет source. | 6–12ч;1–2ч gate. |
> | P04 **Prefix tuning (П):** обучаемый KV-prefix в слоях, base заморожена. | [L5/19][l5] → [Lester21][prompt], [Li21][prefix] | **Д** на C при cap;−1…+1 к full FT. Не то же input prompt, новая cache/positions логика. | 8–16ч;1–2ч gate. |
> | P05 **LoRA:** low-rank delta linear weights, merge в base после training. | [L5/25–27][l5], [S5/cells33–40][s5] → [Hu21][lora] | **Д** на C;−1…+1 к full FT. Полезна specialist веткам, но full FT помещается. Merged модель≤C; training base+adapters превышают точный размер C: при строгом cap нужен меньший base. В shortlist не входит. | 6–12ч;1–2ч gate, измерить memory/speed. |
> | P06 **BitFit:** дообучать только bias. | [L5/28][l5] → [BenZaken21][bitfit] | **Д** на C;−2…0 к full FT. Слабая способность учить новую композицию. | 2–4ч;0.5–1ч gate. |
> | P07 **QLoRA/NF4/double quant/paged optimizer:** frozen lowbit base+LoRA, сжатие scales и offload state при пиках. | [L5/30][l5], [S6/cells62–69][s6] → [Dettmers23][qlora] | **Д** на C;−1…0 к FP16 full FT, предназначена для гораздо больших моделей. BF16 compute из S6 не переносить буквально на T4; использовать поддерживаемый backend. | 1–2дня;1–2ч fine-tune +merge/deployment. |
> | X01 **MLM/dynamic masks/NSP/SOP/ELECTRA/GPT pretraining:** вспомогательная задача/другое направление targets. | [L3/10–24][l3], [S3/cell22][s3] → [BERT][bert], [RoBERTa][roberta], [ALBERT][albert], [ELECTRA][electra] | **З** для новых masked/real-fake/order labels. Unconditional code LM по original reference или code→question — **С**: строки те же,задача/target меняются. Прирост не обоснован, в план не включать. | 1–2дня,ещё несколько часов GPU;разрешённого gate здесь нет. |
> | X02 **Distillation outputs/attention/hidden states/two stages:** приближать teacher signals. | [L6/4–14][l6] → [Hinton15][distill], [TinyBERT][tinybert] | **З**,даже teacher собственная C и original inputs: новые teacher targets. Эффект не применим к разрешённому финалу. | 1–3дня,2–10ч GPU;не проверять. |
> | X03 **Learned judge/filter/verifier, RLVR/DPO/backtranslation/synthetic rewriting (П):** дополнительные correctness/preference/rewritten targets или training на генерациях. | [S1/cells72–75][s1], [S4/7,32][s4], [L6/4][l6] → [OLMo2,§4–5][olmo], [GNMT,§4][gnmt] | **З** при прямом запрете,даже verifier на saved A. Execution vote без training допустим,уже испытывался. Возможный прирост не использовать в прогнозе. | Высокая цена, часы/сутки GPU;не проверять. |
> 
> BERTScore/MAUVE/LLM judge и прочие готовые внешние модели семинаров не дают разрешённого дополнительного тренировочного сигнала. BLEU/ROUGE/METEOR/SARI/Distinct/PPL/MAUVE не заменяют exec-match. TF–IDF/Word2Vec/RNN/LSTM/trie как основная модель не подходят по архитектуре;retrieval baseline уже исследован.
> 
> Scheduled sampling/R-Drop/Unigram/BPE-dropout/grammar verification как преподаваемые методы в snapshot не найдены; курсу не приписывать. Training насобственных sampleprefixes при scheduledsampling по нынешнему запрету как минимум**С**, не предлагается.
> 

## Протоколы трёх направлений GPU1

### 1. Продолжение C с повышенным весом длинных пар

**Гипотеза:** равномерное предъявление finite train недостаточно обучает редкие длинные вычислительные структуры. Удвоенный row weight длинных программ улучшит этот срез без потери качества на основной массе. Обсуждение long weighting в прежней заметке ещё не было опытом обучения.

**Изменение:** полный stable main C-anchor из L1, вместе с AdamW/scaler; меняется только sampler. Для train-строки $i$:

$$w_i=\begin{cases}2,&L_i\ge90,\\1,&L_i<90,\end{cases}\qquad p_i=\frac{w_i}{\sum_jw_j}.$$

$L_i$ — длина **исходного** reference в BPE до compact, без специальных токенов; $p_i$ — вероятность предъявления. Длины взять из старого train-кеша по тем же row identities. Compact/cache/tokenizer прежние. CPU заранее создаёт weighted order; effective batch128 и bucketing4096 сохранены. При дискретном повторе длинных строк weighted epoch имеет $N+N_{\ge90}$ слотов: это уже не биекция исходного корпуса. CE по токенам оставить прежней, дополнительный2×loss weight не добавлять.

**Контроль:** GPU1 короткое обычное продолжение **того же anchor**, прежние order/cursor, равные1.5ч main и одинаковый cooldown. GPU0 L2 — финальная база, но не чистый короткий контроль: накопленное время может отличаться. Weighted branch имеет свои cursor/order/recipe identity. Strict L1 resume не ослаблять. Охлаждённые evaluated weights не заменяют stable main state.

**Критерий:** на заранее выделенном tune после1.5ч main общийΔ≥+0.5п.п., срез≥90BPEΔ≥+3п.п., <60BPE не хуже−0.5п.п. При нейтральном общемΔ от−0.3до+0.5 и положительном длинном срезе оставить как specialist только если frozen mixed vote выигрывает≥+0.5п.п. при равном K. После выбора и продолжения нужен eval-прирост≥+1п.п. одиночной модели либо≥+0.7п.п. mixed vote при≤180с; показать paired gains/losses и CI. Пороги — правила решения, не гарантии статистической мощности.

**Что останется:** улучшенная одиночная C-long; либо C-base+C-long vote; либо идея для ближайшего продолжения C на GPU0. Дополнительно возможно ровно одно равное усреднение weights ordinary/long из общего anchor: один model soup без цены двух моделей на инференсе.

**Риски:** длина отражает также шум/многословие. Обычная token-CE уже даёт длинным строкам больше общей loss-массы; oversampling усиливает её ещё раз. Может упасть throughput и потребоваться меньший microbatch; короткие задачи могут ухудшиться. Не добавлять обратную длину в loss в том же опыте. Долю train≥90 посчитать на CPU, не брать15.3% из heldout.

### 2. Независимая C для ансамбля голосования

**Гипотеза:** новая инициализация даст другие структурные ошибки; при том же числе samples смесь двух генераторов будет лучше C-only. Второй greedy-oracle сам этого не доказывает.

**Изменение:** C53.646M с нуля, например seed137. Остальной L1-рецепт прежний: warmup1000→constantLR0.0015, normalized init, dropout0, compact/cache/order/batch/AMP. Первый опыт меняет **только model seed**, data order2026101201 сохранён; другой порядок — отдельная абляция. Сейчас `new_model()` вызывает `seed_all()` с default42: внешний `manual_seed137` будет перезаписан. Параметр seed должен попасть в recipe/identity/RNG. Продолжение C42 с другим RNG не независимая fresh C.

**Контроль:** saved C42 при равном накопленном main-time для greedy-кривой; для финала — лучшая доступная C42 из L2. C2 за11ч wall может быть моложе C42, которая получила2.5ч+L2; это реальный риск финальной полезности. Одинаковые cooldown, paired ошибки и их пересечение.

**Критерий:** на общем time-point C2 greedy отстаёт от C42 не больше чем на1.5п.п. К финальной C42 допускается отставание только при самостоятельном ensemble gate. Заранее фиксировать C42-onlyK8 против C42K4+C2K4, T0.7, прежнее execution voting/ties. K16 против8+8 — лишь после time gate. **Не удваивать K**, иначе выигрыш нельзя приписать seed diversity. На eval принять mixed vote приΔ≥+0.7п.п., нижней границе pairedCI>0 для основного сравнения, без сильного отрицательного сигнала old991, full pipeline≤180с.

**Риски:** обе модели разделяют шум и систематический bias; более молодая C2 может подавить верный кластер. Timer включает два encoder/load/cache пути, токенизацию, restore, execution и output. Все модели и кеши на **одной T4**. Если нужна последовательная генерация с освобождением кеша — её время тоже входит. Обученную C2 без позитивного ensemble gate в финал не добавлять.

### 3. Продолжение C с label smoothing 0.05

**Гипотеза:** слабая равномерная регуляризация снизит переуверенность на шумных/альтернативных кодах и улучшит greedy или vote. Цена инференса не меняется.

**Изменение:** тот же stable anchor, optimizer/order/cooldown; CE с ε0.05, EOS включён, PAD positions игнорируются. Записать точную конвенцию: PyTorch `label_smoothing` распределяет uniform mass по **всем8195 классам**; `ignore_index` удаляет positions, а не PAD class из uniform. Это не smoothing только по8189 content tokens. Исключение отдельных special classes — другая абляция, здесь не добавлять.

$$L_\varepsilon=(1-\varepsilon)(-\log p_\theta(y\mid x,y_{<t}))+\varepsilon\left(-\frac1V\sum_{v=1}^{V}\log p_\theta(v\mid x,y_{<t})\right).$$

$x$ — вопрос, $y_{<t}$ — **gold prefix**, $y$ — gold следующий token, $p_\theta$ — распределение C, $V=8195$, $\varepsilon=0.05$. Uniform слагаемое ослабляет вырожденность распределения. Оно не добавляет программ, учителя или новых семантических labels: обычная аналитическая регуляризация, допустимая по указанным ограничениям.

**Контроль:** обычное продолжение того же anchor на1.5ч, равные main-time/cooldown; у обеих оценивать **unsmoothed CE**. Smooth loss численно не сравнивать с ordinary CE. Greedy и K8vote, фиксированная T0.7; новую temperature grid в этот gate не добавлять.

**Критерий:** на tuneΔgreedy≥+0.5п.п. либоΔvoteK8≥+0.7п.п. при greedy не хуже−0.3п.п.; после выбора продолжить stable branch. Для финального eval≥+1п.п. одиночной C либо≥+0.7п.п. итогового пула. Низкий smooth loss/красивая энтропия сами не критерии. При отказе восстановить весь anchor, не продолжать проигравшую ветку сε0 под видом контроля.

**Риски:** uniform mass повышает вероятность неверных цифр/operators/special tokens; калибровка может улучшиться, exec ухудшиться. Более длинные/noisy samples могут замедлить vote. Метод не обучает семантическому плану, поэтому его аргумент слабее направленного long training. Положительную идею можно перенести на C без fresh обучения/изменения KV-кеша.

## Расписание: одна GPU1, не больше11ч

Три полных11-часовых опыта на одной GPU1 не помещаются. **Основное расписание — два новых рецепта и один короткий matched control. Fresh seed C занимает отдельную сессию вместо них.** Код и CPU orders готовятся заранее.

| GPU1 wall budget | Действие |
|---|---|
| 0–0.25ч | Загрузка verified C-anchor/cache, probe и baseline eval. |
| 0.25–5.75ч | Три последовательные ветки из **одного** anchor: ordinary, long, smoothing. Каждая1.5ч main +до20мин на одинаковый cooldown/eval/I/O. Cooldown≤10%branch updates. Если оценки не помещаются, сократить main budget одинаково всем до запуска, а не урезать один контроль. |
| 5.75–10.50ч | Продолжить **stable main state** tune-победителя, не cooled weights. Если новые идеи отрицательны — ordinary branch как полезное продолжение C. Финальные cooldown/eval входят в4.75ч, actual main меньше. |
| 10.50–11.00ч | Final eval/model-only/receipts, production timer и сохранение/остановка. Резерв не тратить на новый grid. |

Победитель получит примерно1.5+4ч дополнительного main сверх anchor; GPU0 обучается дольше. Назначение GPU1 — вовремя проверить переносимую идею или получить specialist. Ordinary branch отделяет рецепт от простого «ещё учили C».

Альтернативная **fresh seed C**: до0.25ч setup, около8.75ч main, до1.5ч cooldown/eval/checkpoints/ensemble timer,0.5ч резерв. Early checks после2.5/5ч не обязаны каждый иметь дорогую cooldown-ветку; число anchors зафиксировать заранее. **11ч wall≠11ч main.**

## Seed C против альтернатив: ожидаемая польза к финалу

| Выбор | Возможный эффект сверх доступной C+vote | Что остаётся | Главный риск |
|---|---|---|---|
| **Long-weighted continuation** | Общий−1…+3п.п.; направлена на слабый срез, ранний matched gate. | Одна C/перенос на GPU0, specialist vote или soup. | Шум длинных targets; same-anchor модели коррелируют. |
| **Независимая C, seed137** | Рабочее положительное ожидание0…+2п.п. mixed vote **при равном K**;0/минус возможны. | Второй генератор без архитектурного переписывания. | Почти весь budget, более молодая модель, стоимость двух C в180с. |
| **Label smoothing** | −1…+2п.п.; дешёвый перенос и прежнийинференс. | Одна C с прежней архитектурой. | Не учит новой структуре, риск цифр/голосования. |
| **RoPE/T5bias** | −1…+2п.п. relational bias; не обещание большого прироста от «длинного контекста». | Новая модель/рецепт будущегоцикла. | Fresh training, customattention/KV, не быстрый перенос на C. |
| **Checkpoint averaging / beam2** | Average0…+1п.п.; beam−1…+2 к greedy, не обязательно сверх vote. | Одна C или новый декодер. | Дешёвые проверки поверх победителя, **не замена11ч training**. |

**Вывод:** если есть бюджет для переноса идеи на C, первым делать D01 matched branch: возможен прирост одной модели без второй C на инференсе. Если перенос на GPU0 уже **не успеет**, а fullK4+K4timer проходит, independent seed становится самым предсказуемым по инженерной готовности способом получить ещё один финальный генератор. Без проверенного времени двух C его ценность условна.

Независимые Bernoulli-ошибки не модель нашего ансамбля: одинаковые систематические math-ошибки, корреляция внутри пула и разные неверные числа. $P(C_2\text{ correct}\mid C_1\text{ wrong})$ полезна для диагностики комплементарности, но использует gold. Production должна сама выбирать кластер; oracle union не доступный результат.

## Общий контроль качества опыта

1. Только original train; без обучения на генерациях, исправления reference, correctness labels и distillation. Cache/representation/tokenizer и protected exclusions прежние. Новые ветки имеют experiment IDs/recipe identity; строгий L1resume не обходить.
2. Одно изменение: long→sampler, smoothing→loss, seed→инициализация. Остальные AdamW/clip/AMP/batch/cooldown одинаковы. При обычном branch resume не перезапускать warmup/optimizer.
3. Переиспользовать fixed hash-split tune2048/eval2048 из [candidate-selection-cpu](candidate-selection-cpu.md). Рецепт/threshold/T/K задать до GPU. Ветки выбирать по tune; eval один раз для выбранной ветки, old991 диагностика. На отрицательном eval не выбирать «второе место» по eval. Heldout многократно использован, это не идеально fresh; full4k после фиксации — описательная метрика.
4. Записать paired gains/losses/CI, exec overall и originalBPEbins<60/60–90/90–120/≥120, unsmoothedCE, NaN/AMPskips, rows/targettokens, main/cooldown/eval/I/O/wall и throughput. Срез≥90 фиксирован заранее;138задач≥120 в full4k недостаточно для уверенного малогоΔ. McNemar exploratory после многих гипотез; порог+0.5 не статистическое доказательство.
5. **Fulltest1919≤180с:** только вопросы, без gold; от загрузки weights/tokenizer/test до restore/execution/закрытия answers. Все реально нужные cold-start warmup/compile/calibration включить. Одна T4, обе модели на ней; OOMfallback и budget stop сохранены.
6. Исторические A k16=309.53с и k32=493.33с превышали180с. C/compact/быстрый sampler другие: нужны новые измерения. C smoke test256≈7.51с на ранних weights не подтверждает full1919 C+vote; линейная экстраполяция не доказательство.
7. Fresh1024 оставить до фиксации одного финального рецепта/декодера. Сохранить победителя и ordinaryC как backup. **В рамках обзора GPU/Kaggle не запускались.**

## Источники

Методы/формализация в реестре — пересказ курса и первоисточников. ДиапазоныΔ, ранжирование, пороги и11-часовойплан — **наши гипотезы/выводы**. Числа ДЗ — сохранённые собственные измерения; новая C62.9% — сообщение пользователя.

- [Курс2026, pinned snapshot][course], [HW2][hw2]. Все course-ссылки зафиксированы на этот commit; номера — PDFpages или zero-based cells.
- [data-research-ceiling](data-research-ceiling.md), [candidate-selection-cpu](candidate-selection-cpu.md), [s3-representations-results](s3-representations-results.md), [32-candidate-ensemble-s3-results](32-candidate-ensemble-s3-results.md), [l1-long-training](l1-long-training.md), [l1-review](l1-review.md); [runtime](../kaggle/src/l1_runtime.py), [core](../kaggle/src/exp26_core.py), [cache](../kaggle/src/l1_cache.py).
- [Transformer][transformer]:§3.4/3.5/5.3/5.4/6.1; [TrainingTips][tips]:§2.2/4.4–4.10. Они дополняют краткий course-рецепт.
- [Label smoothing: original][smooth], [анализ][smooth-analysis]; [OLMo2][olmo]:§3/4/6; [ModelSoups][soups] — одна initialization family, не произвольные freshseeds.
- Первоисточник указан в каждой строке ивнесён в bibliography. Public материалы скачаны/прочитаны; training data семинаров и pretrained weights не скачивались.

[course]: https://github.com/ashaba1in/hse-nlp/tree/afb4290ba2155fc826758253ec46fcb368edd49c
[l1]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/week01_classification_generation/lecture1.pdf
[s1]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/week01_classification_generation/seminar1.ipynb
[hw1]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/homeworks/hw1_n_gram_generaiton.ipynb
[l2]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/week02_transformer/lecture2.pdf
[s2]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/week02_transformer/seminar2.ipynb
[l3]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/week03_bert_gpt/lecture3.pdf
[s3]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/week03_bert_gpt/seminar3.ipynb
[l4]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/week04_modern_llms/lecture4.pdf
[s4]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/week04_modern_llms/seminar4.pdf
[l5]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/week05_peft/lecture5.pdf
[s5]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/week05_peft/seminar5.ipynb
[l6]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/week06_speed_up/lecture6.pdf
[s6]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/week06_speed_up/seminar6.ipynb
[hw2]: https://github.com/ashaba1in/hse-nlp/blob/afb4290ba2155fc826758253ec46fcb368edd49c/homeworks/hw2_code_generation.md
[transformer]: https://arxiv.org/abs/1706.03762
[tips]: https://arxiv.org/abs/1804.00247
[olmo]: https://arxiv.org/abs/2501.00656
[dedup]: https://aclanthology.org/2022.acl-long.577/
[smooth]: https://arxiv.org/abs/1512.00567
[smooth-analysis]: https://arxiv.org/abs/1906.02629
[batch-growth]: https://arxiv.org/abs/1711.00489
[adafactor]: https://arxiv.org/abs/1804.04235
[lamb]: https://arxiv.org/abs/1904.00962
[soups]: https://arxiv.org/abs/2203.05482
[shaw]: https://aclanthology.org/N18-2074/
[t5]: https://arxiv.org/abs/1910.10683
[rope]: https://arxiv.org/abs/2104.09864
[pi]: https://arxiv.org/abs/2306.15595
[alibi]: https://arxiv.org/abs/2108.12409
[rms]: https://arxiv.org/abs/1910.07467
[mqa]: https://arxiv.org/abs/1911.02150
[gqa]: https://arxiv.org/abs/2305.13245
[mistral]: https://arxiv.org/abs/2310.06825
[glu]: https://arxiv.org/abs/2002.05202
[moe]: https://arxiv.org/abs/1701.06538
[switch]: https://www.jmlr.org/papers/v23/21-0998.html
[albert]: https://arxiv.org/abs/1909.11942
[tying]: https://arxiv.org/abs/1608.05859
[instability]: https://arxiv.org/abs/2309.14322
[wordpiece]: https://huggingface.co/learn/llm-course/en/chapter6/6
[bpe]: https://aclanthology.org/P16-1162/
[gnmt]: https://arxiv.org/abs/1609.08144
[nucleus]: https://arxiv.org/abs/1904.09751
[distill]: https://arxiv.org/abs/1503.02531
[speculative]: https://arxiv.org/abs/2211.17192
[gpt3]: https://arxiv.org/abs/2005.14165
[checkpointing]: https://pytorch.org/blog/activation-checkpointing-techniques/
[async-checkpoint]: https://docs.pytorch.org/tutorials/recipes/distributed_async_checkpoint_recipe.html
[playbook]: https://huggingface.co/spaces/nanotron/ultrascale-playbook
[ddp]: https://docs.pytorch.org/docs/stable/notes/ddp.html
[zero]: https://arxiv.org/abs/1910.02054
[fsdp]: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
[megatron]: https://arxiv.org/abs/1909.08053
[gpipe]: https://arxiv.org/abs/1811.06965
[ring]: https://arxiv.org/abs/2310.01889
[gshard]: https://arxiv.org/abs/2006.16668
[bnb]: https://huggingface.co/docs/bitsandbytes/en/optimizers
[llmint8]: https://arxiv.org/abs/2208.07339
[ort-quant]: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
[qbert]: https://arxiv.org/abs/1909.05840
[qat]: https://arxiv.org/abs/1712.05877
[zeroquant]: https://arxiv.org/abs/2206.01861
[gptq]: https://arxiv.org/abs/2210.17323
[awq]: https://arxiv.org/abs/2306.00978
[lpft]: https://arxiv.org/abs/2202.10054
[adapters]: https://arxiv.org/abs/1902.00751
[prompt]: https://arxiv.org/abs/2104.08691
[prefix]: https://arxiv.org/abs/2101.00190
[lora]: https://arxiv.org/abs/2106.09685
[bitfit]: https://arxiv.org/abs/2106.10199
[qlora]: https://arxiv.org/abs/2305.14314
[bert]: https://aclanthology.org/N19-1423/
[roberta]: https://arxiv.org/abs/1907.11692
[electra]: https://arxiv.org/abs/2003.10555
[tinybert]: https://arxiv.org/abs/1909.10351

[packing]: https://huggingface.co/docs/trl/sft_trainer
[pinmem]: https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
[offload]: https://www.deepspeed.ai/tutorials/zero-offload/
