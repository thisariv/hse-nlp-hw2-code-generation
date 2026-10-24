---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-14
updated: 2026-10-14
sources:
  - nlp-hw02-final-inference-l3-v6
  - nlp-hw02-decode-profile-preparation
  - pytorch-2-11-cuda-graphs
  - pytorch-2-11-profiler
  - nvidia-t4-specifications
aliases: []
---

# Скорость финального декодирования на T4

**Обновление14.10.2026:** предоставлен T4 profilev2, полный test1919: greedy all1,91× без нагрузки/1,66× с нагрузкой, K4 all1,32×/1,64×. Greedy FP16 отличается1/3строк; пользователь разрешил это известное shape-rounding отличие при той же семантике. Исходный строгий `GPU_gate_passed:false` сохранён. Production перенос и пересчёт K — [v6](final-inference.md#v6-адаптивный-инференс-с-decoder-all), K6/оценка126,4с, CPU-ready без push. Следующие разделы описывают подготовку профиля до его запуска.

**CPU/статика прошли; профильный бандл готов, GPU/push=0.** Цель — минимум2× для генерации одной C, затем проверить, помещается ли K8+K8 в180с. Пользовательский ориентир качества75,0% против73,8% не переизмерялся. Рабочие v4/v5 и P1 не изменены. Связи: [final-inference](final-inference.md), [l3-results](l3-results.md), kv-cache, flash-attention.

## Что уже установлено

Статический разбор frozen `final_fast.py` / `final_decode.py`:

| Место | Фактическое поведение | Оценка вклада |
|---|---|---|
| Self-KV | Предвыделен на512;12 записей текущего слота на шаг, prefix — view | `cat` и копии всего prefix отсутствуют; повторно «устранить cat» нельзя |
| Cross-KV |6 проекций в `Context`, один раз на батч | В `Decoder` есть `repeat_interleave` даже приK=1; приK=4 память дублируется по lanes |
| EOS | `done.all().cpu()` раз в16 шагов | До32 блокирующих проверок на512 шагов; `.item()` и D2H **на каждом шаге отсутствуют** |
| Выходы | Два D2H в конце decode | Не являются per-step sync; входят в таймер генерации |
| Python | Цикл по шагам и6 слоям, deadline раз в16 | Цикла по строкам батча в decoder нет; `pack().tolist()/restore/submit` — после decode |
| Аллокации | Norm/linear/SDPA/residual/head, `tokens.ne`, logits; sample softmax/multinomial и log-softmax | Много маленьких вызовов/ядер и временных тензоров; число/время ядер даст profiler |
| CPU | v5 уже `set_num_threads(1)` и interop1; executor по умолчанию `os.cpu_count()` | Конкуренция4 workers с CUDA-dispatch остаётся гипотезой;22с против15с нельзя целиком приписать ей при разных batch sizes |

Числа из сохранённых timing, не новый прогон:

| v5b greedy | Последовательности | Source | Шаги | Секунды |
|---|---:|---:|---:|---:|
| test, первый батч |1024|47|112|2,097|
| test, второй батч |895|203|512|20,263|
| tune, первый батч |1024|45|176|3,853|
| tune, второй батч |1024|149|512|23,014|

Батч с512 шагами занимает90,6% суммы greedy batch wall на test,85,7% на tune. Даже одна незавершённая строка удерживает все lanes до512. Предположение «длина≤150» для этих запусков неверно. Скрипт [аудита](../kaggle/final_decode_audit.py), [компактные числа и SHA исходных timing](../checks/final-decode-audit.json).

## Почему оценки только по FLOPs недостаточно

Для C:6 decoder layers, width512, heads8, SwiGLU1368, vocab8195. На шаг1024 lanes приходится≈53,74GFLOPs проекций/FFN/head без attention и encoder. Но KV создаёт большой поток чтений:

$$
R_{KV}=2LNd b\left(\frac{T(T+1)}{2}+ST\right).
$$

$L$ — число decoder layers, $N$ — lanes, $d$ — ширина, $b$ — байт на элемент, $T$ — шаги, $S$ — длина source. Множитель2 учитывает K и V; сумма prefix длин даёт квадратичный self-read, $ST$ — повторные cross-read. Предвыделение устраняет копирование растущего кеша, но не чтение attention.

Self-cache приN1024/T512/FP16 —6GiB, cross приS203 —2,38GiB. Для реального test-батчаN895/S203/T512 модель логических чтений даёт≈2,59TB, или масштаб8,09с при320GB/s T4. Это **аналитическая оценка, не измеренный DRAM traffic и не строгий нижний предел**: caching, SDPA и реальные bandwidth меняют результат. Она показывает, почему из FP16 peak65TFLOPS не следует≤1с. Характеристики оборудования — [NVIDIA](https://www.nvidia.com/en-us/data-center/tesla-t4/); источник не измерял наш decoder.

## Флаги и ожидаемый эффект

Отдельный [final_speed.py](../kaggle/src/final_speed.py); базовые файлы сохранены. `Options` управляет только экспериментальным decoder, production v5 ещё использует frozen путь.

| Флаг | Изменение | Гипотеза |
|---|---|---|
| `preallocated_kv=True` | Существующий indexed self-cache + однократный cross projection | Уже baseline; тождественная строка без выдуманного ускорения |
| `share_cross` | ДляK=1 читает неизменяемый context cross/mask без копии | Меньше начального трафика и памяти; небольшой выигрыш, K4 без изменения |
| `async_eos`, `eos_every` | Pinned snapshot done + CUDA event; CPU только `query()`, по умолчанию32 шага | Убирает блокирующие EOS sync; задержка проверки может добавить пустые шаги |
| `compact_done` | При≥25% завершённых удаляет их из активных lanes; переносит только записанный self-prefix и cross, сохраняет исходные row/lane IDs | Главный кандидат на2×: прекращает GEMM/attention/softmax для EOS-строк, включая хвост512 |
| `race_sampler` | Тот же FP32 softmax, `argmax(p/Exp(1))` с повторным noise buffer | Убирает проверки multinomial; выигрыш может быть малым, штатный sampler уже эффективен |
| `lean_greedy` | Не обновляет длины при `compute_logprob=False` | Минус одна операция на шаг; маленький эффект |
| `workers2` |2 исполнителя во время генерации | Проверяет конкуренцию CPU; сама генерация имеет1 Torch thread как baseline |
| `nice10` | Понижает приоритет только Torch-free driver/service/workers | Дополнительная защита CUDA-dispatch; requested/effective/applied записываются |
| `all` | Все перечисленные,2 workers/nice10 | Цель≥2×; без GPU это не установленный результат |

Compaction копирует кеш только на границах проверок и освобождает старые self-слои по мере замены, уменьшая временный пик. Маски PAD/BOS, EOS, позиции, untempered mean logprob и математические операции attention сохранены. Изменение размера GEMM/SDPA способно изменить округление на GPU: **полное greedy bit-exact обязательно проверяется, автоматического допуска нет**. Async EOS может дать больше завершающих PAD; сравнение дополняет обе последовательности до512 и сравнивает каждый ID.

Распределение сэмплера: если $E_i$ независимы и имеют Exp(1), $P(\arg\max_i p_i/E_i=j)=p_j$. $p_i$ — исходный softmax сT0.7; PAD/BOS остаются нулевыми. Noise reuse/удаление lanes/лишние PAD-шаги меняют RNG-поток; это разрешённое изменение и записано в отчёте. FP32 log-softmax для mean LP не меняется.

`cuda_graphs=True` явно отклоняется. В текущем `step` Python-позиция меняет адрес записи/position view и размер prefix SDPA; bucket только по batch этого не решает. Граф на каждую позицию/форму потребовал бы сотен captures, а full-capacity masked attention меняет reduction shapes, вычисления и расходы. Согласовать это с обязательным bit-exact и одним коротким прогоном пока ненадёжно. Это отказ для данного пути, не утверждение, что CUDA graphs невозможны вообще. Ограничения форм/адресов/sync и требования side-stream warmup — [PyTorch2.11](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs).

## Один профильный прогон

[Ноутбук](../kaggle/notebooks/nlp-hw02-decode-profile.ipynb), [сборщик](../kaggle/build_final_profile.py), [runner](../kaggle/src/final_profile.py). Bundle `kaggle/bundles/final_profile_mine`; id/title slug `thisarivson/nlp-hw02-decode-profile`. Private, Internet off, T4/GPU0, входы `thisarivson/nlp-hw02-l3-live` и `thisarivson/nlp-hw02-26b-inputs`. Только C-main L3 l2-62280, SHA weights/tokenizer/test проверяются. Никакого обучения, gold, голосования, Kaggle API или Secrets.

1. Warmup greedy1024 иK4/1024 lanes; одинаковый sorted-by-source-length порядок, начальные формы, cap512 и seed270042 у парных прогонов.
2. **Полные1919**: baseline/all × greedy/K4 × без исполнения/с исполнением. Для `all` greedy все1919 ID сравниваются с full baseline в обоих режимах нагрузки.
3. **Абляции192**: фиксированные равномерные ранги source length изtest1919, baseline/каждый флаг/all × greedy/K4 × без/с нагрузкой. Собственный paired baseline: формы подвыборки отличаются от full. Workers/nice без нагрузки, KV-preallocation и Torch threads1 отмечены как тождественные без повторного замера.
4. Четыре profiler trace baseline/all × load/no-load,32 шагаK4/1024 lanes. Сохраняются сжатый Chrome trace и top25 operators/CPU, CUDA kernel count/busy, launch API, sync/memcpy API. Доли перекрываются; GPU idle — верхняя оценка host overhead, не точная доля Python. [Смысл метрик profiler](https://docs.pytorch.org/docs/2.11/profiler.html).

**Сокращение матрицы явно вынесено в scope.** Для лимита≤12мин по умолчанию отдельные абляции идут на192, full baseline/all — на1919. Это допущение при отсутствии ответа на уточнение; subset throughput не выдается за full. `build_final_profile.py --scope full` включает все1919 во всех строках, сохраняя тот же лимит, и может завершиться partial. В рамках одной сессии нельзя заранее гарантировать полную матрицу и время при неизвестном ускорении.

Нагрузка — непрерывные одинаковые численные Python-программы через настоящий `final_executor`, без Torch в worker-процессах. Это стресс CPU-конкуренции, не воспроизведение распределения реальных кандидатов. Старт/остановка нагрузки исключены из generation timer; setup/drain/logging остаются в общем бюджете. Во время generation2/4 workers, после него сервис завершается; повышение до4 в production пока не внедрено.

Работа останавливается к650с от начала первой code cell; watchdog убивает дерево к690с, предел720с. `summary.json` и `timings.csv` обновляются после каждой строки. В таблице seconds/seq/s, число questions/sequences/scope, effective nice, peak allocated/reserved. Reserved зависит от ранее прогретого CUDA allocator; для парного сравнения памяти прежде смотреть allocated. Event span включает промежутки GPU idle и не равен kernel time. При OOM/error/timeout — явный failed/partial, без скрытого изменения batch. `GPU_gate_passed`, полнота матрицы и `target_2x_all_generation` — разные поля. После2× ещё нужен полный K8+K8 pipeline timer/quality: профилирование одной модели не доказывает180с.

## CPU-результат и воспроизведение

[Проверки](../kaggle/final_profile_checks.py), [receipt](../checks/final-profile-cpu.json): Torch2.14.1 CPU; Kaggle runner строго требует2.11.0+cu128/CUDA12.8/T4.

- Реальные FP16 L3 weights загружены в **FP32**,4 вопросаtest1919 разных длин, cap512: baseline96 шагов;6 вариантов greedy bit-exact. Compaction4→3→2→1 работает на реальных вычислениях.
- Unit: стабильные адреса self/cross, запись prefix/PAD, перестановка/продолжение кеша, K4 cross, staggered EOS двенадцати lanes и сохранение LP/порядка, граница capacity без EOS. Sample без флагов сохраняет RNG bit-exact;40000 exponential-race samples согласуются сp=(0,.1,.2,.7).
- Torch-free load/service запускается и чисто закрывается. Песочница macOS запрещает даже понижение `nice`: PermissionError записан, обхода нет; реальный эффект nice на Kaggle pending.
- nbformat/compile/embedded SHA, metadata/title slug/mounts/watchdog и отсутствие изменений frozen/P1 прошли.

Из `work/`: `python kaggle/build_final_profile.py`, затем `python kaggle/final_profile_checks.py` в окружении сTorch/tokenizers/nbformat/numpy. Сборщик сначала сбрасывает ready; checker публикует `results/kaggle-runs/nlp-hw02-kaggle-l3/final/next-profile.json` с `bundle`, `ready:true` только после gates. Это готовность артефакта к ручной отправке, **не пройденный GPU gate**. Push не выполнялся.

## Источники

- **nlp-hw02-decode-profile-preparation**: пользовательский протокол14.10.2026 и собственный разбор frozen v4/v5/v5b timing/source. Факты timing/CPU отделены выше от гипотез и аналитической модели; bibliography.
- [PyTorch2.11 CUDA semantics / graphs](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs).
- [PyTorch2.11 profiler](https://docs.pytorch.org/docs/2.11/profiler.html).
- [NVIDIA T4 specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/).
