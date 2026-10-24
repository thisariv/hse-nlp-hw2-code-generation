---
type: experiment
course:
  - nlp
status: draft
understanding: 4
created: 2026-10-19
updated: 2026-10-19
sources:
  - wang2022-self-consistency
  - shi2022-mbr-exec
  - wortsman2022-model-soups
  - sennrich2016-nmt-ensemble
  - opennmt-ensemble
  - nlp-hw02-e1-ensembles
aliases: []
---

# E1: другие способы ансамблирования

Исходная точка — [v10](final-inference.md#v10-c-long2x-b6--p1-b--w-b6): C-long2x B6 / P1-B / W B6, greedy C, K1=1/1/1 → K2=1/5/6. Пользователь сообщил public78,96%, свежий tune1530 и test115,6с. Сохранённый timing подтверждает115,631356826с; E1 не генерирует test. Большее число execution-голосов уже насыщалось; проверяем изменение распределения и ранжирования, а не только числа samples. Связи: [decode-speed](decode-speed.md), [b6-results](b6-results.md), [tta-experiment](tta-experiment.md).

## Предрегистрация — до GPU и чтения E1 eval

Этот раздел и исполняемый список [e1_policies.py](../research/e1_policies.py) фиксируются SHA в launch receipt до единственного push. Eval нового E1 ещё не читался. Ранее eval использовался в TTA, поэтому это независимое подтверждение **от текущего tune-подбора**, но не совершенно нетронутый исследовательский holdout. Дизайн и проверяемые веса заданы пользователем, а не выбираются по E1 labels.

### Гипотезы и формализация

1. **Токенный ансамбль:** модели согласуют один префикс до исполнения. При одинаковом словаре геометрическое усреднение подавит токены, которым хотя бы одна модель даёт низкую вероятность. Это перенос идеи ансамблей NMT; точный log-linear вариант описан в OpenNMT. В NMT также применяется арифметическое усреднение вероятностей, здесь оно не исследуется.

$$a_t(v)=\frac13\sum_{m\in\{C,P,W\}}\log p_m(v\mid y_{<t},x),\qquad q_T(v)=\operatorname{softmax}(a_t/T)_v.$$

$x$ — вопрос, $y_{<t}$ — общий префикс, $v$ — следующий токен, $m$ — модель; $T$ — температура. Greedy выбирает argmax $a_t$. Для сохранённого LP используется нормированное $\log q_1=a_t-\log\sum_v\exp a_t(v)$; среднее log-prob само по себе не нормировано. EOS выбирается тем же распределением, отдельные encoder/cross/self-KV у каждой модели. PAD/BOS запрещены, как вv10. Ограничение — примерно тройная стоимость шага и дополнительные кэши.

2. **Перекрёстный LL:** каждая из трёх моделей оценивает каноническую токенизацию каждого уникального `represented_code` внутри строки, с одним EOS. Сохраняются $L_m=\sum_{t=1}^{n}\log p_m(y_t\mid y_{<t},x)$ и $L_m/n$, где $n$ включает EOS. Для обрезанной генерации EOS добавляется при scoring; LF/UTF8 преобразования и каноническая BPE-токенизация явно отличаются от исходного пути генерации. CPU-инвариант generation↔teacher forcing проверяется на исходных token IDs. Ограничения — предпочтение коротких программ у суммарного LL и дорогой scoring всех кандидатов. Идентичные программы исполняются один раз, но повторные samples сохраняют кратность голосов.

3. **Soup:** $\theta_\alpha=\alpha\theta_C+(1-\alpha)\theta_P$, $\alpha\in\{0.3,0.5,0.7\}$. Усреднение FP32, затем FP16 inference; строгая проверка ключей/форм. C/P происходят от общей C-long2x ветки, но разошлись по обучению; наличие хорошей общей долины не гарантировано. У W другая архитектура, в soup она не входит. Wortsman et al. мотивируют усреднение близких fine-tuned решений; здесь это собственная проверка применимости к нашим продолжениям обучения.

### Один GPU-пул

Private `friend1/nlp-hw02-e1-ensembles`; Internet off, T4×2: GPU0=tune2048, GPU1=eval2048, `fixed_split` v10. Только три JSON: tokenizer, heldout4k, reference execution; `test.json` не открывается и не извлекается. Веса/encoder/fast decoder/QK/исполнитель/базовое голосование — frozenv10; новые файлы E1 отдельно. Источники B6v1/P1v3 и полные SHA закреплены в [identity](../kaggle/e1-source-identity.json).

- На каждой строке: greedy C/P/W/E/трёх soups; K16 T0.7 и K8 T1.0 каждой модели; K8 T0.7 E. **87 кандидатов**, уникальные represented-code и cross LL всех трёх моделей.
- Первый T0.7 sample отдельно от остальных15 (E:1+7), чтобы измерить реальную форму stage1; source-length bucketing, single lanes512, ensemble lanes256, cap512. OOM → уменьшение batch с откатом RNG. Это свежий Monte Carlo pool, а не старые K8-журналы.
- До полного пула — smoke64 каждого split, равномерные ранги длины вопроса. Оба должны пройти shapes/EOS/PAD, execution completion, finite LL, gzip roundtrip; иначе остановка. Accuracy в GPU worker не считается.
- Throughput каждого типа: samples/s, ms/row, K/steps/source length, LL tokens/s; executor v10, глобальный cache без потери кратности. Атомарные `.jsonl.gz` по64строки и SHA-manifest; `/kaggle/working/e1/`.
- Ожидание≤60мин wall; backend4200с, worker deadline4050с, watchdog4080с; T4 pair максимум8400 device-с. Полнота обязательна; partial не считается подтверждением. Run All запускает тот же smoke→full протокол.
- Ровно **один** собственный push с серверной проверкой кода/private/offline/mounts/version. Если QUEUED>15мин либо RUNNING без output>15мин: сохранить наблюдение, не перезапускать, ждать до75мин. Последующие версии пользователя/оркестратора принимаются только при exact code parity. Ничего на Kaggle не удалять; содержимое `~/.kaggle*` не читать/выводить.

### Политики и выбор

Исполняемая сетка — [e1_policies.py](../research/e1_policies.py), без адаптации после eval. Все списки samples берутся в сохранённом порядке; неуспех исполнения расходует бюджет. Обычное голосование: численные connected components `isclose(1e-6)` → count → наличие greedy → mean generation LP → стабильный порядок; код greedy сохраняется, если он в победившем компоненте. Stage1 останавливается только при численном согласии **всех** его кандидатов с anchor, какv10.

| Семейство | Заранее фиксированные варианты |
|---|---|
|Baseline|C greedy, 1/1/1→1/5/6, T0.7; frozen production selector|
|Токенный|E greedy; E voteK4/K8 fixed; E vote2→4/8 adaptive; E greedy + обычные1/1/1→1/5/6 и E1→4/8|
|Cross LL|Baseline budget; weight $1+\lambda z$, λ=0.25/0.5/1; tie по сумме LL или средним LL; cluster по сумме LL; отдельно cluster по logsumexp среднего трёх LL|
|Soup|Каждый α: только greedy; soup greedy как anchor обычного baseline; замена P1-B одним soup greedy при C1/W1→C1/W6 и anchorC|
|Бюджет|Только неуверенные stage1: K2=1/8/8,1/12/12,1/16/16,4/8/8,8/8/8,16/16/16|
|Температуры|Те же baseline stage1; на неуверенных чередование T0.7/T1.0, начиная сT0.7, K2=1/5/6,1/8/8,1/16/16|

Для LL $s_c=\sum_m L_m(c)$, $z_c=(s_c-\bar s)/\max_c|s_c-\bar s|$ по уникальным **используемым** кандидатам строки, знаменатель≥1e−12. Это ограничивает z в[−1,1], веса неотрицательны; повторы голосуют с тем же z. LL tie вставляется после count, до greedy. `cluster-sum` буквально суммирует отрицательные LL с кратностью и может предпочитать маленькие кластеры; `cluster-logmass` проверяет вероятностную альтернативу $\log\sum_c\exp(\sum_mL_m(c)/3)$. Все LL-политики используют LL только после неединогласного stage1; CPU-кандидатам не передаются gold.

**Правило выбора:** на tune выбрать ровно одну primary политику среди прогнозов≤155с по maximum matches; ничья → меньше прогноз, затем имя. До открытия eval записать `results/e1-selection.json`, SHA raw manifest/предрегистрации/grid/анализатора/timing и SHA selection. Также заморозить лучшую допустимую политику каждого семейства. После этого отдельный процесс confirm читает eval и оценивает **всю** заранее объявленную сетку как вторичный описательный анализ; решение об успехе относится только к frozen primary. Нового выбора по eval нет.

Парная разница с baseline на том же пуле, percentile bootstrap по задачам20000, seed2026101904, 95% CI; gains/losses. CI tune после выбора описательный, CI по задачам не покрывает вариативность повторного sampling. Eval-интервалы остальных политик не скорректированы за множественные сравнения и не используются для объявления успеха.

**Время:** только throughput tune, без использования eval. Фиксированные фазыv10=23,082561923с, генерацияv10=92,548794903с из сохранённого test timing (SHA закреплён в [reference](../kaggle/e1-timing-reference.json)). Стоимость greedy и stage1/extra samples — по соответствующим измеренным K1/K15 (E K1/K7), T1.0 поK8. Общий множитель генерации калибруется наv10 по его реальному среднему числу samples каждой модели на1919строках. Для новой политики используются её средние counts на tune; LL — измеренные tune seconds/unique candidate × число scoring jobs ×1919/2048, дополнительно soup setup при необходимости. Фиксированные фазы сохраняются консервативно даже у single-greedy. При LL не начислять scoring неиспользуемого пула. Публиковать также uncalibrated cost и коэффициент; это прогноз, не новый cold test. Допуск≤155с, цель≤140с.

**Успех:** frozen primary на eval имеет Δ≥+0.4п.п., нижнюю95%границу>0 и frozen прогноз≤155с. Иначе сохранитьv10, честно перечислить отрицательные результаты; удачный вторичный вариант требует нового независимого подтверждения, а не переименования вprimary.

## CPU-проверки и состояние запуска

Реализация: [decoder/LL/soup](../kaggle/src/e1_decode.py), [worker](../kaggle/src/e1_worker.py), [сборщик](../kaggle/build_e1.py), [CPU checks](../kaggle/e1_checks.py). На реальных pinned C/P/W: vocab8195, PAD/BOS/EOS0/1/2, offset3, compact30, одинаковый SHA representation-source и cache manifest; token-table SHA `eb0080fcc2bda85916b20a1e76c6ef57c2c3460e1e3995fd9339644fdb49e713`. Soup C/P по212совместимых ключей.

[CPU receipt](../checks/e1-cpu.json): Torch2.11.0; slow full-prefix и cached ensemble совпали на трёх реальных tune-вопросах **до EOS, максимум123шага**, разные ширины и QK W проверены. Три копии C дают ту же greedy byte-строку и LP. Максимальная ошибка generation↔teacher-forced sum LL: C3,58e−6, P4,77e−6, W3,34e−6; каноническая токенизация совпала с генерацией во всех9проверках C/P/W×3строки. В обоих scoring-путях PAD/BOS замаскированы до log-softmax, как вv10; это условное распределение по разрешённым токенам. Отдельно прошли EOS/хвостовой PAD/compaction с общими lane IDs, разная длина teacher-forced целей, FP32 soup/отказ при несовместимых keys/shapes, реальный embedded worker/executor/gzip87candidates и дедупликация LL.

[CPU анализатора](../checks/e1-analysis-cpu.json):33политики,128синтетических строк; production baseline winner/execution/early-stop/count parity exact, в том числе нетранзитивный isclose. Gold не влияет на selector; weighted LL/tie и paired bootstrap проверены. Timing-модель восстанавливает115,631356826сv10 при исходных counts.

В15:45:20UTC19.10 выполнен **единственный push**; [private версия1](https://www.kaggle.com/code/friend1/nlp-hw02-e1-ensembles/versions?versionNumber=1). Сервер подтвердил exact code parity, private, Internet off, T4 и источники B6v1/P1v3. Notebook SHA `108ca8a3a7f1158e320ba3ccee668ae1d9ea7fcad0ba1e1c44a66cbaf37f1438`. Перед запуском свободно52295,961GPU-с, cap пары8400с. [Launch](../checks/e1-launch.json), [наблюдения](../checks/e1-observations.json), [handoff](../results/next-e1.json). Первое наблюдение—RUNNING; текущее состояние и итог ниже и в [JSON](../results/e1-experiment.json). Повторного собственного push нет. Монитор [e1_watch.py](../kaggle/e1_watch.py) ограничен17:00:20UTC; при внешнем перезапуске проверяет exact parity.

[Проверка сборки](../checks/e1-build.json): новый builder использует собственный [конфигE1](../kaggle/e1-config.json) и закоммиченные20модулейv8, побайтно равныеv10. `build_e1.py --check` воспроизводит исходный notebook SHA без зависимости от незакоммиченных файловv10.

Команда для **внешнего оркестратора**, только после пользовательской отмены/перезапуска, не для автоматического повторного push агентом:

```sh
KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle kernels push -p kaggle/bundles/e1_mine --accelerator NvidiaTeslaT4 --timeout 4200
```

После полного output: `research/e1-analysis.py tune <raw>`, затем отдельный процесс `confirm <raw>`. Eval нельзя анализировать до квитанции выбора. Артефакты v10, запрещённые progress/index/dashboard/backlog не редактировались.

## Результаты E1

Последняя [версия4](https://www.kaggle.com/code/friend1/nlp-hw02-e1-ensembles/versions?versionNumber=4): **COMPLETE**, наблюдение 2026-10-19T16:44:18.261179+00:00. Собственный push ровно1; внешние версии: 2, 3, 4. Exact code parity подтверждён; отмен/перезапусков агентом0.

Полный пул подтверждён SHA/manifest/source/config/IDs, baseline production parity2048/2048 на обоих splits. Primary **`budget-1-8-8`** выбрана на tune до открытия eval; selection SHA `ec56bc5287d1ee96377c36599eaea35b72340294e81dfbc27ff6ab6b688e29d9`. Все времена ниже — frozen прогноз1919строк, не новый cold test. CI95% парной разницы с baseline, п.п.; accuracy в%. Baseline относится к свежему E1-пулу.

|Политика (выбор на tune)|Tune, %|Eval, %|Δeval [95%CI], п.п.|Прогноз, с|
|---|---:|---:|---|---:|
|baseline|1543/2048 (75.34)|1559/2048 (76.12)|+0.000 [+0.000; +0.000]|115.8|
|token-greedy|1487/2048 (72.61)|1513/2048 (73.88)|-2.246 [-3.320; -1.172]|67.6|
|ll-weight-0.5|1547/2048 (75.54)|1562/2048 (76.27)|+0.146 [-0.244; +0.537]|138.7|
|soup-0.5-anchor|1540/2048 (75.20)|1558/2048 (76.07)|-0.049 [-0.537; +0.391]|116.1|
|**budget-1-8-8**|1550/2048 (75.68)|1574/2048 (76.86)|+0.732 [+0.098; +1.367]|137.3|
|temperature-1-8-8|1548/2048 (75.59)|1573/2048 (76.81)|+0.684 [-0.098; +1.465]|140.6|

**Критерий успеха пройден.** Набросокv11 сохранён ниже; холодное время пока не измерено.

> [!info]- Вся предобъявленная сетка (вторичный описательный eval)
> |Политика|Tune, %|Eval, %|Δeval [95%CI], п.п.|Прогноз, с|
> |---|---:|---:|---|---:|
> |baseline|1543/2048 (75.34)|1559/2048 (76.12)|+0.000 [+0.000; +0.000]|115.8|
> |token-greedy|1487/2048 (72.61)|1513/2048 (73.88)|-2.246 [-3.320; -1.172]|67.6|
> |token-vote-4-fixed|1508/2048 (73.63)|1541/2048 (75.24)|-0.879 [-1.904; +0.098]|222.4|
> |token-vote-8-fixed|1521/2048 (74.27)|1558/2048 (76.07)|-0.049 [-0.977; +0.879]|372.4|
> |token-vote-4-adaptive|1508/2048 (73.63)|1541/2048 (75.24)|-0.879 [-1.904; +0.098]|171.8|
> |token-vote-8-adaptive|1522/2048 (74.32)|1559/2048 (76.12)|+0.000 [-0.928; +0.928]|220.4|
> |token-mix-4|1543/2048 (75.34)|1561/2048 (76.22)|+0.098 [-0.537; +0.684]|240.8|
> |token-mix-8|1545/2048 (75.44)|1574/2048 (76.86)|+0.732 [+0.049; +1.416]|302.6|
> |ll-weight-0.25|1542/2048 (75.29)|1561/2048 (76.22)|+0.098 [-0.195; +0.391]|138.7|
> |ll-weight-0.5|1547/2048 (75.54)|1562/2048 (76.27)|+0.146 [-0.244; +0.537]|138.7|
> |ll-weight-1.0|1544/2048 (75.39)|1562/2048 (76.27)|+0.146 [-0.342; +0.635]|138.7|
> |ll-tie-sum|1543/2048 (75.34)|1559/2048 (76.12)|+0.000 [-0.293; +0.244]|138.7|
> |ll-tie-mean|1540/2048 (75.20)|1558/2048 (76.07)|-0.049 [-0.293; +0.146]|138.7|
> |ll-cluster-sum|1223/2048 (59.72)|1246/2048 (60.84)|-15.283 [-17.139; -13.477]|138.7|
> |ll-cluster-logmass|1516/2048 (74.02)|1552/2048 (75.78)|-0.342 [-1.123; +0.439]|138.7|
> |soup-0.3-greedy|1469/2048 (71.73)|1504/2048 (73.44)|-2.686 [-3.760; -1.611]|34.9|
> |soup-0.3-anchor|1539/2048 (75.15)|1560/2048 (76.17)|+0.049 [-0.488; +0.586]|116.6|
> |soup-0.3-replace-P|1526/2048 (74.51)|1549/2048 (75.63)|-0.488 [-1.270; +0.244]|99.7|
> |soup-0.5-greedy|1478/2048 (72.17)|1492/2048 (72.85)|-3.271 [-4.346; -2.197]|34.6|
> |soup-0.5-anchor|1540/2048 (75.20)|1558/2048 (76.07)|-0.049 [-0.537; +0.391]|116.1|
> |soup-0.5-replace-P|1524/2048 (74.41)|1550/2048 (75.68)|-0.439 [-1.221; +0.293]|99.3|
> |soup-0.7-greedy|1472/2048 (71.88)|1499/2048 (73.19)|-2.930 [-4.004; -1.855]|34.6|
> |soup-0.7-anchor|1540/2048 (75.20)|1560/2048 (76.17)|+0.049 [-0.342; +0.391]|116.2|
> |soup-0.7-replace-P|1530/2048 (74.71)|1548/2048 (75.59)|-0.537 [-1.318; +0.244]|99.1|
> |budget-1-8-8|1550/2048 (75.68)|1574/2048 (76.86)|+0.732 [+0.098; +1.367]|137.3|
> |budget-1-12-12|1552/2048 (75.78)|1576/2048 (76.95)|+0.830 [+0.146; +1.514]|173.0|
> |budget-1-16-16|1552/2048 (75.78)|1584/2048 (77.34)|+1.221 [+0.488; +1.953]|208.6|
> |budget-4-8-8|1548/2048 (75.59)|1578/2048 (77.05)|+0.928 [+0.293; +1.562]|148.8|
> |budget-8-8-8|1553/2048 (75.83)|1574/2048 (76.86)|+0.732 [+0.049; +1.416]|164.1|
> |budget-16-16-16|1560/2048 (76.17)|1583/2048 (77.29)|+1.172 [+0.488; +1.904]|266.1|
> |temperature-1-5-6|1534/2048 (74.90)|1565/2048 (76.42)|+0.293 [-0.391; +0.977]|117.9|
> |temperature-1-8-8|1548/2048 (75.59)|1573/2048 (76.81)|+0.684 [-0.098; +1.465]|140.6|
> |temperature-1-16-16|1561/2048 (76.22)|1577/2048 (77.00)|+0.879 [+0.146; +1.611]|215.2|

**Вывод по гипотезам.** На свежем пуле увеличение P1-B/W до8 только при разногласии stage1 дало29исправлений/14ухудшений. Это выбранная до eval primary; один sampling run не оценивает разброс повторного запуска.

Токенный greedy: eval1513, против одиночных C/P/W1499/1498/1499; улучшение относительно отдельных greedy есть. До baseline1559 он не дотягивает. E vote2→8 даёт те же1559 при220,4с; смесьE+обычных samples доходит до1574, но прогноз302,6с исключает её из допустимых политик.

Cross LL: выбранное λ0.5 даёт+3ответа, CI включает0. Tie по sumLL не меняет score, logmass теряет7ответов. Буквальная сумма отрицательных LL по кластеру даёт1246/2048 (−15,28п.п.). Собственная интерпретация: такой score штрафует размер кластера; это не вероятностная маргинализация.

Soupα0.5 как anchor теряет1ответ; α0.3/0.7 дают лишь+1. Замена P1-B одним soup-greedy ухудшает score. Смесь температур при1/8/8 даёт+14ответов, но CI[−0,098;+1,465] включает0; её не выбираем по вторичному eval.

**Набросокv11:** прежние C-long2x B6/P1-B/W B6, greedyC, T0.7; stage1 по1sample каждой модели. При полном численном согласии остановиться, иначе добавить7P1-B и7W. Итого максимум17samples+greedy. Голосование, порядок/приоритет, executor и guardsv10 сохраняются; дорогие token ensemble/LL/soup в этот вариант не входят.

Прогноз:23,083с фиксированных фаз +56,757с сырой генерации×2,012782 = **137,322с**. Коэффициент калибровки опубликован; это перенос throughput на формыv10, а не измерение coldv11. Cold T4, W K7 memory/OOM-retry и новые production samples ещё не проверены. До этого v10 остаётся готовым fallback.

[Tune](../results/e1-tune.json), [selection](../results/e1-selection.json), [eval](../results/e1-confirm.json), [компактный итог](../results/e1-experiment.json). Все33eval-интервала описательные, без поправки на множество сравнений; первичный критерий применён только к frozen primary.

Коммит не создан: песочница запретила Git `index.lock`; индекс остался пустым. [Квитанция](../checks/e1-git.json).

[Набросокv11](../results/e1-v11-policy.json): ровно frozen policy и веса E1; прежние guards160/166/170/172, одинT4. Прогноз 137.32с. Перед новой посылкой нужен отдельный cold gate; в E1 test не запускался.

GPU session 1140.21с; [измеренный throughput](../results/e1-throughput.json). Исходный test не читался; число кандидатов87на строку, полных строк4096.

Всего356352кандидата,222968уникальных represented-code внутри строк,668904cross-LL score. OOM0, peak allocated7,71/7,54GiB (tune/eval). Внутри каждого split2048уникальных question SHA; пересечений tune/eval0. [Финальный аудит](../checks/e1-final.json).

После HTTP502 два source-файла восстановлены из `SOURCES` **серверной версии4** с exact code parity и проверкой каждого SHA. Candidate/LL-журналы не изменялись. [Происхождение восстановленных исходников](../checks/e1-source-recovery.json).

## Источники

- **wang2022-self-consistency** — Wang X. et al. *Self-Consistency Improves Chain of Thought Reasoning in Language Models*, arXiv2022 / ICLR2023. [Оригинал](https://arxiv.org/abs/2203.11171). Выбор ответа через маргинализацию путей; наш кодовый ансамбль — собственная адаптация. bibliography — wang2022-self-consistency.
- **shi2022-mbr-exec** — Shi F., Fried D., Ghazvininejad M., Zettlemoyer L., Wang S.I. *Natural Language to Code Translation with Execution*, EMNLP2022,3533–3546. [ACL](https://aclanthology.org/2022.emnlp-main.231/). Группировка кандидатов по результату исполнения. bibliography — shi2022-mbr-exec.
- **wortsman2022-model-soups** — Wortsman M. et al. *Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time*,2022. [Оригинал](https://arxiv.org/abs/2203.05482). Усреднение близких решений; применимость к нашим веткам — гипотеза. bibliography — wortsman2022-model-soups.
- **sennrich2016-nmt-ensemble** — Sennrich R., Haddow B., Birch A. *Edinburgh Neural Machine Translation Systems for WMT16*,371–376. [ACL](https://aclanthology.org/W16-2323/). Мотивация ансамблей encoder–decoder. bibliography — sennrich2016-nmt-ensemble.
- **opennmt-ensemble** — OpenNMT-py, translate options / `avg_raw_probs`. [Документация](https://opennmt.net/OpenNMT-py/options/translate.html). Различие усреднения raw probabilities и log scores. bibliography — opennmt-ensemble.
- **nlp-hw02-e1-ensembles** — пользовательский протокол19.10.2026, собственная реализация и измерения E1; bibliography — nlp-hw02-e1-ensembles. Выводы опыта отделяются от содержания статей.
