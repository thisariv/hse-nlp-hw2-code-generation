---
type: experiment
course:
  - nlp
status: complete
understanding: 3
created: 2026-10-09
updated: 2026-10-09
sources:
  - work/results/22-cosine-tail-from-75000-to-100000-steps-encoder4-decoder4-qwen-bpe-8k-swiglu-500k-pairs-summary.json
  - work/results/21-deeper-encoder-4-layers-decoder-4-qwen-bpe-8k-swiglu-500k-pairs-100000-steps-summary.json
  - work/checks/22-cosine-tail-functional-quality-audit.json
aliases: []
---

# ДЗ2: cosine LR на последних 25k

[Подготовленный notebook22](../notebooks/22-cosine-tail-from-75000-to-100000-steps-encoder4-decoder4-qwen-bpe-8k-swiglu-500k-pairs.ipynb) · [полученные outputs](../results/22-cosine-tail-from-75000-to-100000-steps-encoder4-decoder4-qwen-bpe-8k-swiglu-500k-pairs-results.ipynb) · [сводка/опись/SHA](../results/22-cosine-tail-from-75000-to-100000-steps-encoder4-decoder4-qwen-bpe-8k-swiglu-500k-pairs-summary.json).

**Результат смешанный:** №22 лучше завершает программы и быстрее в сохранённом test; прироста правильных решений нет. Полный CE1024 немного выше21, reference proxy ниже, ручной74 с AugAssign одинаковый19/74. Снижение LR как причина решения плато не подтверждено. Все T4-метрики ниже — сохранённый пользовательский запуск; новые проверки CPU приведены отдельно.

## Гипотеза и одинаковый бюджет

Пользователь заметил замедление loss16 в №21 на80–100k. С50k LR был постоянным0.0001. Гипотеза22: плавное снижение шага улучшит донастройку. Источник — **тот же21/75000**, модель/AdamW/scheduler counter/Python+Torch+CUDA RNG/coverage восстановлены; повторены ровно25k updates до100k. Контроль — уже полученный21/75k→100k с constant0.0001.

Меняется только будущий LR: cosine0.0001→0.00001. Остаются encoder4/decoder4/d256/heads4, Pre-LN + final LayerNorm, SwiGLU680, learned positions2048, общие source/target embeddings и отдельная output projection: **12 086 912 параметров**. Те же500k разрешённых train, собственный Qwen-style byte BPE8k, batch8/FP32/cached IDs, AdamW/weight decay0.01/clip1/dropout0.1. Tokenizer из checkpoint без refit, SHA `261f25b5d4be006a6a00e41e0eafbe522580741da627ba670f6b9805a2659b7d`.

На старте75k CE1024 и **все1024 greedy token IDs** точно совпали со старым21. Исходные model/AdamW/scheduler/RNG/coverage проверены по состояниям. В конце Python/Torch/CUDA RNG и покрытие совпадают с21/100k: это поддерживает одинаковую последовательность случайных операций и train-batches; веса различаются из-за LR.

## Результат на100k

| Метрика |21: constant1e−4 |22: cosine1e−4→1e−5 |
| --- | --- | --- |
| CE наheldout1024 ↓ |0.775758 |0.778481 |
| NLL наисходныйUTF8-байт ↓ |0.227092 |0.227889 |
| Reference-совпадения /888 |178 |173 |
| Исходный ручной74, Assign/Return |18 |18 |
| Дополнительный ручной74, также Name AugAssign |19 |19 |
| EOS /1024 |1013 |1017 |
| Python AST /1024 |1014 |1017 |
| Undefined name /1024 |49 |45 |
| Полный test1919, с ↓ |74.70 |57.60 |
| Test EOS /1919 |1891 |1903 |
| Test AST /1919 |1895 |1903 |
| Test cap безEOS /1919 |28 |16 |
| Test undefined name /1919 |103 |97 |

CE/NLL хуже на **0.35%**. По reference888:145 сохранённых совпадений,28 новых,33 потерянных, net−5. Reference-программы содержат известные ошибки; это согласие с эталонным числом, не официальный pass@1.

По одинаковому ручному74 с поддержкой `+=/-=`:16 сохранённых верных,3 новых,3 потерянных, net0;19/74≈25.7% относится только к этому малому адаптивному набору. Исходный18/74 и дополнительный19/74 сохраняются отдельно. Unsupported не объявляется автоматически неправильным; в финальном дополнительном22 неизвестных0, неправильных55.

### Что изменилось по смыслу

[Аудит](../checks/22-cosine-tail-functional-quality-audit.json) пересчитан по прежним74 ответам без изменения train-таргетов и без исполнения сгенерированного Python.

| Индекс heldout1024 |Правильный ответ |21 |22 |Изменение |
| --- | --- | --- | --- | --- |
|12 |26 игр |38 |26 |приобретено |
|31 |240/(160/4)=6 часов |12 |6 |приобретено |
|321 |250−150=100 долларов |0 |100 |приобретено |
|42 |40+5+3=48 дюймов |48 |45 |потеряно |
|557 |60×2=120 минут=2 часа |120 |900 |потеряно |
|877 |300×4×7=8400 мл |8400 |2400 |потеряно |

Ошибки выбора чисел, направления операции и использования переменных остаются. В42 модель обновляет `this_year_height=40+3`, затем другую переменную `last_year_height=40+5` и возвращает45; условие не собрано в одну цепь40+5+3. Более аккуратный синтаксис и EOS не гарантируют правильного вычисления.

## Почему loss выглядит лучше

Частая метрика — **те же16 validation-задач**, без dropout. Финальная точка0.755881→0.730918; это выглядит как большой выигрыш, но не характеризует все1024.

| Окно updates |21: средний CE16 |22: средний CE16 |
| --- | --- | --- |
|75–80k |0.756126 |0.755734 |
|80–85k |0.749843 |0.747177 |
|85–90k |0.747577 |0.739734 |
|90–95k |0.743137 |0.733834 |
|95–100k |0.739818 |0.731321 |

В каждом окне100 общих контрольных точек с шагом50; первая дополнительная точка75001 исключена. Это коррелированные оценки одной траектории, не100 независимых опытов. Последняя точка21 заметно выше его среднего0.739818; сравнивать только endpoints недостаточно.

На validation64 CE0.751160→0.744594 тоже лучше, но extra1920.746526→0.752912 и новые768 внутри прежнегоheldout10240.784858→0.787439 хуже. Их взвешенный по target-токенам результат и есть0.775758→0.778481. Train64 также чуть хуже0.712635→0.719058. Малый train–validation gap не доказывает достаточную ёмкость или отсутствие ошибок данных.

### Динамика №22

| Шаг |CE1024 |
| --- | --- |
|75k |0.824065 |
|80k |0.811174 |
|85k |0.800201 |
|90k |0.788243 |
|95k |0.782143 |
|100k |0.778481 |

Модель продолжает обучаться:75k→100k CE−5.53%; в21 тот же участок дал−5.86%. Полное плато и польза именно такого decay не доказаны. Проверено одно расписание на одном seed; это не поиск оптимального LR. Floors/reset/новый warmup в этом опыте отдельно не сравнивались.

Чувствительность к источнику сохранена: правильные questions64 дают CE0.744594, циклически подменённые2.338761 (+1.594167). Это подтверждает использование вопроса, а не качество рассуждений.

## Скорость и покрытие

Новые25k: **970.66с/16.18мин чистого обучения,25.76 updates/s**, прежний constant-хвост1025.02с/17.08мин. Active wall с диагностиками/checkpoints1084.88с/18.08мин; setup данных/tokenizer187.60с/3.13мин отдельно, без upload/baseline. Цепочка75k-префикс21 + новый хвост22:4007.49с/66.79мин. Это не повторение всего100k и не дополнительное обучение сверх100k.

Разница training time−5.30% измерена в разных T4-сессиях; optimizer/model kernels не менялись. Нельзя приписать её снижению LR. Накопленно оба21/22:800000 предъявлений,399353 увиденных строк/79.8706%,52789363 полезных target-tokens,85115448 paddedslots/37.9791% padding.

Test1919: **57.60с/33.32 ответа/с**, запас122.40с до180 в прежней области таймера; peak allocated424.98MiB. Таймер включает encoding/transfers/encoder/KV-decode/CPU decoding/AST; loading/setup/loss/warmup/дополнительный numeric audit исключены. Официальный scope неизвестен; лимит посылки ещё не проверен официальным checker.

Средняя длина68.05→64.34 tokens (−5.46%), но batch32 ждёт самый длинный ответ. По сохранённым длинам total decoder steps16351→13333 (−18.46%),queryrows523128→426557,cap-батчей22→14. Proxy self-attention query-key работы−29.11%; полное время−22.89%. Сокращение длинных хвостов согласуется с ускорением; это расчёт работы, не GPU-профилировка/точное разложение wall time. KV-код и архитектура те же.

## Что делать дальше

Предложен следующий короткий контроль: **greedy против beam2 на готовых весах22/100k**, без обучения. Сначала старый heldout1024/те же74 ручных ответа, затем полный test1919 с одинаковыми batch/cap/precision и полным таймером. Правило ранжирования и обработки EOS фиксировать до оценки; выдавать одну программу, без выбора по reference. Проверяется влияние поиска ответа, выигрыш заранее не обещается.21/100k сохранить как контроль качества; fresh1024 — для финального сравнения выбранных моделей. Новый notebook23 не подготовлен.

Проверка мотивирована тем, что улучшение вероятностей конкретного кода не превращается в рост greedy-корректности. Beam сохраняет несколько префиксов вместо одного локального выбора; более вероятная программа всё ещё может неверно решить задачу. Скорость нужно измерить, удвоение beam не гарантирует ровно2× время или попадание в180с. Это отдельный контроль метода из лекции2 и прежнего плана, не доказательство причины ошибок.

Если этот контроль не даёт роста решений, следующий обучающий фактор — batch16 либо ширина384. Для batch фиксировать **число предъявлений/полезных tokens**, а не одинаковое число updates:25k×8=12.5k×16. Один размер batch также меняет число AdamW updates; не масштабировать LR одновременно и не называть сравнение чистой проверкой шума. Для ширины запускать с нуля при прежних data/tokenizer/schedule и измерять новую стоимость; увеличение параметров разрешено пользователем, выигрыш не гарантирован. Выбор нового обучающего опыта пока не сделан.

## Проверено сейчас, без обучения

[Статика/целостность](../checks/22-cosine-tail-results-static-checks.json) · [ручной аудит](../checks/22-cosine-tail-functional-quality-audit.json) · [кривые/работаdecode](../checks/22-cosine-tail-learning-curve-and-inference-work.json) · [CPU](../checks/22-cosine-tail-conditioning-cpu-checks.json).

25cells/13code; code и Markdown точно совпали с подготовленным22, outputs/errors0. Execution counts1,2,5…15 сохранены: пробелы не доказывают непрерывность запуска; завершение100k/добавочные25k проверены по logs/state/manifest. ZIP9members: CRC/размеры/manifest/SHA всех8 payloads.4671 сохранённый пример проверен по token bytes/UTF8/AST/numeric/repetition/aggregates/NLL; test1919 соответствует исходному порядку публичногоtest Parquet.

Три safe weights-only checkpoints75k/100k/latest: finiteFP32/AdamW,steps/LR/counters/data/tokenizer/RNG/coverage; latest state==100k.75k model/optimizer/scheduler/RNG/coverage точно равны исходному21/75k; конфигурация22 содержит будущий LR. В финале все sampling/RNG/coverage равны21/100k. Истории21/20/19/18/hash прежние; обаmanual74/парные изменения/reference888 пересчитаны. Fresh1024 совпал поhash и **не оценивался**.

Новый CPU/PyTorch2.14.1: saved22/100k,correct/rotated/reversed64 CE0.744589/2.338756/2.403257; первые два воспроизвелиT4 в1e−5.3 полныхgreedy ответа — правильные6часов,undefined name иошибочные45дюймов — совпали ordinary/KV/savedT4. Веса неизменны, optimizer updates **0**. Длительное обучение/T4/полный1919 inference/официальный checker сейчас не выполнялись.

Notebook с outputs/сводка/ограниченные аудиты сохраняются вGit. ПолныеJSON/test-generations/fresh,точный tokenizer и3weights — локальные ignored `results/generations/`/`results/checkpoints/`; опись/SHA в сводке. Для переноса копировать ignored отдельно или восстановить исходный ZIP. Исходные Downloads сохранены.

## Протокол LR и исторические проверки подготовки

Для глобального update $s\in[75001,100000]$:

$$
u(s)=\frac{s-75001}{100000-75001},\qquad
\eta(s)=\eta_{min}+\frac{\eta_0-\eta_{min}}{2}(1+\cos(\pi u(s))).
$$

$u(s)$ — доля хвоста;$\eta_0=10^{-4}$ — старт;$\eta_{min}=10^{-5}$ — floor. Первыйupdate75001 использует1e−4,последний1000001e−5; далееfloor. СчётчикLambdaLR восстановлен75000;старыйbaseLR.001 иAdamW moments сохранены,новогоwarmup/reset нет.

Источник21 `step-75000.pt`SHA `cbdc205be3842f02ea7eabc024d2417934aac2dd3de47da1058b1ccb2ff4eaa1`; проверяется до setup. На75k349535seenrows/39586597usefultokens/63842112paddedslots. Fullgreedy/manual на75k/100k,teacher-forced1024 каждые5k,частыйloss16 каждые50. Latest каждые1000,отдельные75k/100k; train rerun запрещён, partialZIP сохраняется после прерывания.

При подготовке22 проверены25/13 source/пустыеoutputs,неизменныйmodel/KV/loss/update/coverage/reference888 AST,все25000 LR/endpoints/floor,16 сохранённыхmanualscores18/19/20/21. CPU-loss1024 baseline delta−4.59e−6,точныйPT/ZIP restore. **6 optimizer updates** при подготовке:2online/2notebook/1resume/1interrupt;weights/AdamW/LR/RNG/coverage trajectory exact. Эти6updates не относятся к нынешнему анализу0updates.

Проверены abbreviated cap4/test2/fulltarget controls,RNG-preserver,resume75001,cosine послеsourceLambdaLRload,timer/rerun/interruptioncheckpoint/partial+fullZIP. Full500kcache лениво заменён32 затронутымипарами; T4/CUDA restore/25k/full1919/UI при подготовке не проверялись. Полученный T4-результат и новые проверки описаны выше. [Подготовительная статика](../checks/22-cosine-tail-static-checks.json),[CPU6updates](../checks/22-cosine-tail-cpu-checks.json).

## Источники и связи

- Пользовательские `22_cosine_tail_from_75000_to_100000_steps_encoder4_decoder4_qwen_bpe_8k_swiglu_500k_pairs.ipynb` и `hw02-encoder4-decoder4-qwen-bpe8k-swiglu-500k-cosine-tail-75000-to-100000-results.zip`;точныеSHA/membernames в [сводке22](../results/22-cosine-tail-from-75000-to-100000-steps-encoder4-decoder4-qwen-bpe-8k-swiglu-500k-pairs-summary.json).
- [21: encoder4/decoder4](deeper-encoder-4-layers-decoder-4-qwen-bpe-8k.md) и [сводка21](../results/21-deeper-encoder-4-layers-decoder-4-qwen-bpe-8k-swiglu-500k-pairs-100000-steps-summary.json): пользовательскийT4-контроль/источник75k.
- [Аудит21](../checks/21-deeper-encoder-functional-quality-audit.json): фиксированные74 ручных ответа и одинаковая поддержкаAugAssign.
- Greedy/beam search,[очередь опытов](experiment-plan.md),[10: cosine LR](cosine-learning-rate-wider-transformer.md),Pre-LN. Новое сравнение/приоритетbeam — наш вывод по сохранённым данным, не факт из источника.
