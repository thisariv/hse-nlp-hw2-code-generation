"""Render the research note from measured aggregates, with explicit hypotheses."""
from common import *

SOURCES={
 'wang2017-dns':'https://aclanthology.org/D17-1088/',
 'xie2019-gts':'https://www.ijcai.org/Proceedings/2019/736',
 'hashimoto2018-retrieve-edit':'https://papers.nips.cc/paper_files/paper/2018/hash/cd17d3ce3b64f227987cd92cd701cc58-Abstract.html',
 'hayati2018-recode':'https://aclanthology.org/D18-1111/',
 'parvez2021-redcoder':'https://aclanthology.org/2021.findings-emnlp.232/',
 'liu2023-tinygsm':'https://arxiv.org/pdf/2312.09241',
 'eldan2023-tinystories':'https://arxiv.org/abs/2305.07759',
}

EXPERIMENTS=[
 dict(rank=1,name='Компактный обратимый таргет',hypothesis='Декодер тратит около пятой части длины на восстановимый текст; та же семантика требует меньше шагов.',
      change='Удалить комментарии, схлопнуть пустые строки с AST guard, убрать header и фиксированный tail; один control token задаёт режим восстановления. Выражение result сохранить.',
      control='GPU0: A/D0, исходный таргет. GPU1: та же A/D0, компактный таргет. Один seed, те же row identities/order, tokenizer22; обе модели с нуля. Сравнить равные предъявления и равное время.',
      t4x2_budget='20–30 минут wall на 0.96M каждой; при положительном сигнале до1.92M, 40–50 минут. Это план, не замер.',
      expected='Гипотеза: +0–3 п.п. greedy при равном времени; около20% меньше target tokens, особенно полезно для sampling. Скорость не обязана улучшиться на20%.',
      risks='Сдвиг decoder positions, ошибка mode/EOS, нестандартный tail; восстановление проверять AST и execution. Не удалять выражение результата.',
      admissibility='Допустимо: детерминированное обратимое представление существующего reference; дополнительных данных/семантических таргетов нет.'),
 dict(rank=2,name='Плейсхолдеры чисел с literal fallback',hypothesis='Копирование входных чисел проще генерации цифр; исчезнут некоторые пропуски и новые литералы, но структура остаётся задачей модели.',
      change='Заменить совпадающие числа вопроса/reference на N1…Nk; оставить константы и непокрытые литералы. Детерминированные словесные/percent mappings включать отдельной абляцией; передать исходные значения в source для сравнений величин.',
      control='GPU0: победитель №1. GPU1: тот же формат + только digit placeholders. Новые токены учатся с нуля; один seed и равные предъявления. Сначала CPU round-trip на≥50k.',
      t4x2_budget='25–40 минут wall на0.96M каждой; до1.92M только после улучшения основных и длинных бакетов.',
      expected='Гипотеза: +1–5 п.п.; оценивать реальные gains/losses, не складывать12.2% и17.7% ошибок как независимый запас.',
      risks='Повторяющиеся значения, знак, precomputed5.5, числовые имена, порядковые/даты, влияние величины на ветвление. 88.5% покрытия с эвристиками не означает88.5% accuracy.',
      admissibility='Допустимо как обратимая нормализация имеющихся пар. Переписывание программы по собственной математике либо создание уравнений-таргетов было бы серым/запрещённым; здесь не предлагается.'),
 dict(rank=3,name='B: больше обучения с сохранением длинных задач',hypothesis='Главная оставшаяся ошибка — структура; B ещё растёт, а длинные reference резко хуже коротких.',
      change='Продолжить stable main B в двух ветках; GPU0 равномерный finite train, GPU1 удвоенный вес reference≥90 BPE. Использовать только исходные пары; тот же WSD/cooldown и бюджет времени.',
      control='Одинаковый stable B main checkpoint/AdamW/RNG. Оценённый t04 cooldown не подменяет stable main. Не менять формат/архитектуру одновременно с sampling weights.',
      t4x2_budget='45–60 минут wall, две ветки B параллельно; отдельный одинаковый cooldown внутри бюджета. Это ранний gate, не окончательное обучение для первого места.',
      expected='Гипотеза: +1–4 п.п. от бюджета; дополнительный эффект length weighting −1…+2 п.п. общего качества и +2–8 п.п. на длинных.',
      risks='Ухудшить короткие задачи, переусилить шум/сложные программы, менять эффективный вопросный состав; логировать размер каждого бакета и исходные row identities.',
      admissibility='Допустимо: перевзвешивание существующих train-пар, encoder–decoder по-прежнему обучен с нуля.'),
 dict(rank=4,name='Выбор существующих reference по консенсусу',hypothesis='Конфликтующие reference дают противоречивый сигнал; их выбор может быть полезнее простого увеличения корпуса.',
      change='Только при≥3 finite reference и единственном модальном численном ответе оставить исходные reference из модального кластера; ties и singleton оставить. Сохранить суммарный sampling weight вопроса. Не удалять все трудные вопросы.',
      control='GPU0: текущий finite train. GPU1: тот же корпус с описанным отбором, A/выбранный формат/D0, один seed,0.96–1.92M предъявлений.',
      t4x2_budget='20–40 минут wall на ветку параллельно; полный consensus/cache подготовить заранее на CPU.',
      expected='Гипотеза: 0–2 п.п.; известный loss от conflicts в finite-кеше всего1.20% всех строк, поэтому большой скачок заранее не обещается.',
      risks='Мода бывает неверной, n=2 не даёт надёжного выбора, согласие коррелирует с лёгкостью; не превращать фильтр в удаление сложных задач.',
      admissibility='Допустимо: фильтрация исходных пар. Новые ответы, починка reference моделью и обучение на сэмплах запрещены.'),
 dict(rank=5,name='Retrieval как контекст encoder–decoder',hypothesis='Модель может исправлять родственника; простая подстановка и прямой override уже недостаточны.',
      change='Source: вопрос + ближайший train-вопрос/исходный код. Target: только исходный reference текущего вопроса. Training retrieval исключает все строки с тем же вопросом; без pretrained retriever и без обучения на substituted candidates.',
      control='GPU0: plain выбранная A. GPU1: та же модель + retrieved context, с нуля; один seed, одинаковые пары и target. Сначала индекс1M, затем полный только если оправдана цена.',
      t4x2_budget='30–45 минут wall,0.96M в каждой ветке; длина source и цена retrieval учитываются. Отдельно полный test1919 timer≤180с на однойT4.',
      expected='Гипотеза: 0–3 п.п.; измеренный A+aligned-retrieval20 oracle70.65% задаёт запас существующего кандидата, но editor способен создавать другую структуру и не ограничен этим oracle.',
      risks='Неподходящий сосед, копирование его ошибки, self-retrieval leakage, длинный encoder input; полный CPU поиск может съесть inference budget.',
      admissibility='Допустимо: retrieval только по разрешённому train, выходы — существующие таргеты. Чистый retrieval вместо обязательной Transformer-модели был бы серым; baseline здесь только диагностика.'),
]

def pct(x):return f'{100*x:.2f}%'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    d=read(WORK/'results/data-research-ceiling.json')
    # Routing can complete while the weighted audit is running.
    d['routing']=read(LOCAL/'routing.json');d['alignment']=read(LOCAL/'alignment.json')
    d['retrieval']['index']['tie_policy']='sp_matmul_topn arbitrary membership at equal-score shard cutoff; merge retained candidates by decreasing score then increasing shard,row'
    d['experiments']=EXPERIMENTS
    d['ceiling_assessment']=dict(working_hypothesis_percent=[85,92],central_orientation_percent=90,
       confidence='low: extrapolation, not a confidence interval or measured accuracy',
       empirical_finite_duplicate_ge2=d['noise_in_finite_training_cache']['modal_rows_ge2_ceiling'],
       all_train_empirical_optimistic=d['noise_in_finite_training_cache']['all_train_row_optimistic_ceiling'],
       note='Do not call98.8% or any oracle a realistic deployable ceiling. Private leaderboard rank is unknown.')
    d['literature_sources']=SOURCES
    write(WORK/'results/data-research-ceiling.json',d)
    t=d['targets'];w=d['population_weighted_targets'];c=d['census'];noise=d['noise'];finite=d['noise_in_finite_training_cache'];r=d['retrieval'];full=r['by_index']['9725737'];a=d['alignment'];err=r['A_errors'];routing=d['routing']
    lines=['---','type: experiment','course:','  - nlp','status: learning','understanding: 3','created: 2026-10-12','updated: 2026-10-12','sources:']
    lines+=['  - '+s for s in SOURCES]+['  - kaggle-s1-v1','  - kaggle-s2-v2','aliases: []','---','','# ДЗ2: данные, шум и потолок качества','',
      f'**Короткий вывод:** обратимый таргет сокращает длину на **{pct(w["saving_fractions"]["no_frame"])}**; с одним mode token — **{pct(w["no_frame_saving_with_one_mode_token"])}**. Все три безопасные нормализации: **0/57 702** расхождений. Главный оставшийся запас связан со структурой решения, а не только цифрами. Рабочий ориентир для борьбы за первое место — **около90%**, диапазон **85–92% как гипотеза**, не установленный потолок и не обещание результата.',
      '', '[Все числа и определения](../results/data-research-ceiling.json) · [CPU-скрипты и воспроизведение](../research/README.md) · [[26b-final-wsd-data-scaling|A/6.4M]] · [[27-execution-voting-s1-results|S1/27]] · [[29m-28b-capacity-s2-results|S2/B]].',
      '', '## Область проверки и протокол','',
      '- Только локальный CPU. Новых GPU-запусков, обучения, Kaggle push и скачанных байтов — 0. До работы свободно19GiB; все60 pinned raw-шардов уже были локально. SHA/bytes60/60 проверены.',
      '- Train: revision `b2724689ff079e9c22efd548fe134bfb29036dbb`, **11 838 003** сырых строк. Для retrieval — **9 725 737** finite строк №26a+26c: исходные length/hash/protected exclusions, включая вопросы heldout4k/old1024/test/fresh; exact overlap0.',
      '- Heldout4k:4096 numeric reference; old1024:991 eligible. Test1919: только вопросы, accuracy test не вычислялась. Fresh не оценивается.',
      '- Выборка: seed20261012, по1000 случайных eligible-before-exec строк из каждого шарда, всего60k; **57 702** finite. Основные оценки взвешены числом finite строк шарда; это поправляет малый последний шард. BPE №22:8189 content IDs, SHA `261f25b5…659b7d`, без EOS.',
      '- Original/normalization/duplicates/substituted candidates исполнены bounded executor №26: AST filter, math/sympy, wall1с/CPU2с,4 workers. Error-анализ A использует сохранённые S1 greedy и ранее проверенные execution; новой генерации модели нет.',
      '', '## 1. На что уходят target tokens','',
      'Компоненты ниже — доля BPE-токенов всех60k сырых выбранных таргетов. Токен на границе регионов делится пропорционально покрытым символам. Выражение после `result =` не считается константным.',
      '',table(['Регион','Доля токенов'],[[k,pct(t['token_component_fractions'][v])] for k,v in [('Комментарии','comments'),('Пустые физические строки','blank_lines'),('Header `def solve() -> int:`','header'),('Фиксированный tail: `result = ` и `return result`','tail_fixed')]]),
      '', 'Региональная атрибуция не равна экономии: при удалении меняются соседние BPE-разбиения и исчезают отступы/пробелы. Основная экономия ниже — оценка на всём finite-кеше, взвешенная по размерам шардов:',
      '',table(['Вариант','Средняя длина BPE','Экономия от original','Расхождения / finite'],[
          ['Original',f'{w["tokens"]["original"]["mean"]:.2f}','—','—']]+[[label,f'{w["tokens"][k]["mean"]:.2f}',pct(w['saving_fractions'][k]),'0 / 57 702'] for label,k in [('a) Без комментариев','no_comments'),('b) + без пустых строк, AST guard','no_blank'),('c) + восстанавливаемый каркас','no_frame')]]),
      '', 'Вариантc имеет режимы `header_tail`/`header` (46 014/13 986 в60k). Выражение результата остаётся в fragment; deterministic postprocessing добавляет header, префикс assignment и return по mode. Нужно передавать режим: один специальный токен уменьшает чистую экономию до19.77%. Для нестандартного модуля предусмотрен raw fallback.',
      '', '> [!warning] Не удалять пустые строки слепо\n> Наивный вариант дал1/57 702 =0.001733% расхождений: пустая строка после явного `\\` удерживала границу выражения. AST guard оставляет такие случаи без сжатия. Для0 ошибок в независимой выборке rule-of-three даёт ориентир<0.0052% на95%, но не доказывает отсутствие всех редких случаев.',
      '', '## 2. Шум reference и модальный ответ','',
      table(['Срез','Разных вопросов','Групп с≥2 raw reference','Строк в таких группах'],[
          ['Точный текст',c['exact']['unique_questions'],c['exact']['duplicate_groups'],c['exact']['duplicate_rows']],
          ['lower + collapse whitespace',c['normal']['unique_questions'],c['normal']['duplicate_groups'],c['normal']['duplicate_rows']]]),
      '', 'Нормализация регистра/пробелов объединяет лишь115 групп дополнительно. Хеши128bit использовались для поиска групп; равенство текстов внутри каждой найденной группы затем проверено. Исполнены все1 596 554 reference нормализованных дублей:1 517 271 numeric,79 283 nonnumeric/blocked/ошибки.',
      '', table(['Метрика','Полный raw train, exact','Finite training cache26a/26c'],[
          ['Вопросов с≥2 finite reference',noise['exact']['counts']['at_least_two_finite'],finite['counts']['at_least_two_finite']],
          ['Из них с конфликтом ответов',noise['exact']['counts']['conflicting_finite'],finite['counts']['conflicting_finite']],
          ['Доля конфликтующих вопросов',pct(noise['exact']['counts']['conflicting_finite']/noise['exact']['counts']['at_least_two_finite']),pct(finite['counts']['conflicting_finite']/finite['counts']['at_least_two_finite'])],
          ['Согласие случайной пары, pair-weighted',pct(noise['exact']['pair_agreement']),pct(finite['pair_agreement'])],
          ['Согласие пар, macro по вопросам',pct(noise['exact']['pair_agreement_question_macro']),pct(finite['pair_agreement_question_macro'])],
          ['Empirical mode, rows с≥2 finite в группе',pct(noise['exact']['modal_rows_ge2_ceiling']),pct(finite['modal_rows_ge2_ceiling'])],
          ['Empirical mode, macro по вопросам',pct(noise['exact']['modal_group_mean']),pct(finite['modal_group_mean'])],
          ['Все finite строки raw-дублей, включая1 finite',pct(noise['exact']['modal_duplicate_row_ceiling']),pct(finite['modal_duplicate_row_ceiling'])],
          ['Оптимистичный all-row ceiling',pct(noise['exact']['all_train_row_optimistic_ceiling']),pct(finite['all_train_row_optimistic_ceiling'])]]),
      '', r'$$', r'C_{\mathrm{mode}}=\frac{\sum_q\max_a n_{q,a}}{\sum_{q,a}n_{q,a}}.',r'$$','',r'Здесь $q$ — вопрос, $a$ — кластер численного ответа с `isclose(rel=abs=1e-6)`, $n_{q,a}$ — число его reference. Это лучший **ретроспективный** lookup для данного набора reference.',
      '', 'Потолок98.42% raw и98.80% finite считает singleton и остальные незаметные конфликты идеально предсказуемыми: известные losses187 306/117 076 делятся на весь соответствующий корпус. Это не оценка качества нового test. Частые простые вопросы усиливают pair-weighted agreement: macro существенно ниже. Finite и length-фильтр повышают согласованность, поэтому смешивать raw87.45% и short-heldout нельзя.',
      '', '> [!note] Что это говорит о test/val\n> По условию они из того же генератора; это делает шум reference релевантным. Но max по2–3 наблюдениям оптимистичен для будущего reference; singleton могут быть как легче, так и шумнее. Согласованные неверные решения не выявляются. Истинная математическая правильность и exec-match с выбранным reference — разные величины.',
      '', '## 3. Шаблонность и retrieve-and-substitute','',
      'Точный sparse cosine top20 в **фиксированном** train-only TF–IDF пространстве: слова/unigrams, lower, маскирование чисел, English stopwords, sublinear TF, L2; IDF на stratified60k,8851 признаков. Вложенные индексы — первые200k/1M/9 725 737 eligible row identities по shard,row. Повторы не удалены. На равном score библиотека может выбрать произвольную часть граничной группы внутри шарда; объединение сохранённых top20 сортируется по score, затем shard,row. Это проверка расширения корпуса в одной геометрии, а не сравнение независимо обученных IDF.',
      '', table(['Индекс','Heldout p10 / median / p90','Heldout≥0.9','Test median','Test≥0.9','KS statistic / p'],[
         [n,' / '.join(f'{x["similarity"]["heldout4k"]["quantiles"][str(p)]:.3f}' for p in [.1,.5,.9]),pct(x['similarity']['heldout4k']['ge']['0.9']),f'{x["similarity"]["test"]["quantiles"]["0.5"]:.3f}',pct(x['similarity']['test']['ge']['0.9']),f'{x["test_vs_heldout_ks"]["statistic"]:.3f} / {x["test_vs_heldout_ks"]["pvalue"]:.3f}'] for n,x in r['by_index'].items()]),
      '', 'Test имеет сопоставимых родственников: на полном индексе median0.865 против0.866;≥0.9 около43.4% в обоих. KS не обнаружил различие (p0.479), но это не доказательство равенства распределений сложности или accuracy. Cosine≈1 не означает одинаковую роль числа или правильное решение.',
      '', 'Правила подстановки: **order** сопоставляет числа по порядку, требует равного количества; **local role** использует уникальный локальный контекст, иначе order; **aligned role** монотонно выравнивает слова с числовыми масками, требует покрытия используемых source-code чисел, допускает дополнительные числа. Все отказываются при противоречивом отображении повторного значения. Заменяются AST numeric constants, совпадающие со значениями вопроса соседа; производные и не совпадающие константы остаются. Unary negative literal без положительного совпадения в вопросе может остаться неизменным — ограничение baseline.',
      '', table(['Индекс','Order top1','Oracle5','Oracle20','Aligned top1','Oracle5','Oracle20'],[
         [n]+[pct(x['heldout4k']['order'][f'top{k}']['accuracy']) for k in [1,5,20]]+[pct(a['by_index'][n]['heldout4k'][f'top{k}']['accuracy']) for k in [1,5,20]] for n,x in r['by_index'].items()]),
      '', 'Local-role правило не улучшило полный индекс:1143/1616/1965, как order. Aligned-role:1115/1615/**2015**; новый код по этой эвристике иногда хуже top1, но богаче oracle20. Oracle использует reference для выбора и недоступен при inference. Abstentions считаются ошибками; это не accuracy только удачно сопоставленных случаев.',
      '', table(['Индекс','Скелет top1','Скелет oracle5','Скелет oracle20','A∪order20 oracle','A∪aligned20 oracle'],[
         [n]+[pct(x['heldout4k']['skeleton'][f'top{k}']['accuracy']) for k in [1,5,20]]+[pct(x['heldout4k']['order']['top20']['A_union_oracle_matches']/4096),pct(a['by_index'][n]['heldout4k']['top20']['A_union_oracle_matches']/4096)] for n,x in r['by_index'].items()]),
      '', 'Скелет — AST с числами→0 и переименованием переменных по первому появлению; операции, порядок и повторные обращения сохранены. Совпадение достаточной формы не гарантирует верные numeric bindings; несовпадение не исключает эквивалентное решение.',
      '', '> [!details]- Exec-match по длине для каждого индекса\n'+ '\n'.join('> '+l for l in table(['Индекс','Ref BPE','n','Order1','Oracle5','Oracle20','Align1','Oracle5','Oracle20'],[
          [n,k,x['heldout4k']['order']['top1']['by_reference_length'][k]['n']]+[pct(x['heldout4k']['order'][f'top{z}']['by_reference_length'][k]['accuracy']) for z in [1,5,20]]+[pct(a['by_index'][n]['heldout4k'][f'top{z}']['by_reference_length'][k]['accuracy']) for z in [1,5,20]] for n,x in r['by_index'].items() for k in ['<60','60-90','90-120','>=120']]).splitlines()),
      '', 'Old991 и все quantiles/thresholds также есть в JSON. Простой gold-free override тестирован по10 правилам similarity/согласия: **все хуже control**. Best nontrivial поold991 —3/5 одинаковых ответов и cosine≥0.99: primary2515/4096 (61.40%),10 gains/33 losses; old623/991 против634. С учётом no-op выбран keep-greedy. Полный поиск на CPU и teacher-forcing rerank требуют отдельного inference timer; сейчас нет доказательства≤180с наT4.',
      '', '## 4. Числа и плейсхолдеры','',
      table(['Категория литерала','Оценка finite train'],[[k,pct(w['numeric_literal_fractions'][v])] for k,v in [('Точно есть в вопросе, учитывается знак','question'),('Словесные числа / явные percentage формы','derived_or_word'),('Эвристический список констант','constant_heuristic'),('Прочие','other')]]),
      '', f'Только N1…Nk для цифровых чисел вопроса покрывают **{pct(w["numeric_program_fractions"]["exact_only"])}** программ; с распознанными словесными/percent формами и константами — **{pct(w["numeric_program_fractions"]["placeholders_plus_constants"])}**. Остальные требуют literal fallback. Повторные значения в вопросе: **{pct(w["numeric_program_fractions"]["repeated_question_numbers"])}**; повторённое используемое значение: **{pct(w["numeric_program_fractions"]["repeated_used_value"])}**.',
      '', 'Категории — воспроизводимые эвристики, не ручная семантическая разметка: словарьzero…twenty/twice/double/triple/half/dozen/quarter/hundred/thousand; percentagev/100 и1±v/100 только при явномpercent/%; константы0,1,2,3,4,7,12,24,60,100,360,365,1000. Любой literal из whitelist не доказан как константа именно этой задачи. Precomputed5.5 из5hours+30minutes попадает в «прочие». AST хранит модуль negative literal, знак — отдельный узел.',
      '', '## 5. Ошибки A/6.4M','',
      'Greedy2538/4096;1558 ошибок. Ниже пересекающиеся признаки, denominator1558:',
      '',table(['Признак','n','Доля ошибок'],[
        ['Необъяснённое число вне question/слов/percent/констант',190,pct(190/1558)],
        ['Пропущено цифровое число question, используемое reference',275,pct(275/1558)],
        ['Тот же set литералов, неверное решение',594,pct(594/1558)],
        ['Тот же multiset литералов, неверное решение',382,pct(382/1558)],
        ['Runtime/syntax/nonnumeric/timeout',46,pct(46/1558)],
        ['Буквально есть любое число вне question',944,pct(944/1558)]]),
      '', 'Последние944 включают полезные константы/производные; называть все такие случаи галлюцинацией нельзя. Novel literal относительноreference —494/1558. Runtime46:35 execution errors,8 syntax,2 timeout,1 nonnumber. Всего17 неверных numeric решений имеют тот же reference AST-скелет.',
      '', table(['Ref BPE','n','Ошибки','Error rate','Число внеQ*','Пропуск*','Те же числа*','Runtime*','Прочее*'],[
        [k,x['n'],x['errors'],pct(x['errors']/x['n'])]+[x['exclusive'].get(z,0) for z in ['number_not_from_question_or_known_constant','used_question_number_missing','same_literal_set_wrong_program','runtime_or_non_numeric','other_numeric_or_structure']] for k,x in err['by_reference_length'].items()]),
      '', '`*` В таблице длины признаки взаимоисключающие с приоритетом runtime→необъяснённое число→пропуск→те же числа→прочее. Числа отличаются от overlapping таблицы намеренно. Лишние числа из irrelevant context не обязаны использоваться; missing считается относительно literal reference.',
      '',table(['Индекс','Скелет ближайшего соседа на ошибках','Скелет среди5','Скелет среди20'],[
        [n]+[f'{x["A_error_neighbor_skeleton"][f"top{k}"]["matches"]}/1558 ({pct(x["A_error_neighbor_skeleton"][f"top{k}"]["matches"]/1558)})' for k in [1,5,20]] for n,x in r['by_index'].items()]),
      '', 'На полном индексе это50/1558 nearest и179/1558 top20. По длинам top20:70/447,77/689,21/312,11/110. Прямой перенос структуры соседа покрывает меньшинство наших ошибок; смысловое редактирование может выходить за эти exact skeleton matches.',
      '', '## 6. Оценка реалистичного потолка','',
      '- **Данные:** empirical modal на finite группах с≥2 numeric —90.06%; macro по вопросам89.44%. Raw counterparts87.45%/86.70%. Это сигнал, что90% гораздо ближе к реальному шумовому пределу, чем98.8%; exact повторов недостаточно для универсальной оценки всех singleton.',
      '- **Текущий A:** saved S1 greedy+k16/T0.7 oracle3314/4096=80.91% противvote2707/4096=66.09%; k32+greedy3468/4096=84.67%; все65 candidates3605/4096=88.01%. Это предел конкретного candidate set, не обучения A/B. С order-retrieval20:83.13%/86.28%/89.09% для этих трёх наборов.',
      '- **Рабочая гипотеза:** ориентир≈90%, разумный исследовательский диапазон85–92% для большего обучения и улучшенного представления/отбора. Это не confidence interval и не прогноз короткого GPU gate. 90%+ остаётся stretch; обещать95% по этим измерениям нельзя. Выбранный heldout многократно использован, final fresh нужен после фиксации решения.',
      '- **Правила:** лимит курса180с/test1919/однаT4. Старые внутренние120с, greedy token-identity gate и выбранные бюджеты опытов не являются ограничениями курса. Качество и место на приватном leaderboard здесь не измерены.',
      '', '## 7. Ранжированный план экспериментов','',
      'Ожидаемые эффекты ниже — гипотезы, их нельзя складывать. Ни одного GPU-опыта в этой сессии не запущено. На обоихT4 — independent variants; one change at a time, paired gains/losses по4k, контрольold991, одинаковая политика cooldown и полный inference timer после положительного gate.', '']
    for e in EXPERIMENTS:
        lines += [f'### {e["rank"]}. {e["name"]}','',
          '- **Гипотеза:** '+e['hypothesis'], '- **Изменение:** '+e['change'], '- **Контроль:** '+e['control'],
          '- **T4×2:** '+e['t4x2_budget'], '- **Ожидание:** '+e['expected'], '- **Риск:** '+e['risks'], '- **Правила:** '+e['admissibility'],'']
    lines += ['Шестым, отдельно от data ablations: короткий inference-only опыт k8/k16 с teacher-forcing score существующей A/B для кандидатов и полного таймера≤180с. Это использует уже обученные с нуля генераторы и не создаёт обучающих таргетов. Не обучать verifier на saved samples: по текущему условию это запрещённые дополнительные данные/таргеты. AST variable-name canonicalization также стоит отдельной абляции, поскольку меняет подсказки имён, а не только длину.',
      '', '## 8. Первичные источники: что переносится','',
      '- **Wang et al.,2017, DNS:** number mapping по порядку появления чисел и генерация equation template вместо числовых значений. Переносится обратимая N-схема; значимость чисел и неоднозначность повторов остаются. Их RNN и Math23K не являются доказательством улучшения нашего Transformer/Python. [Статья](https://aclanthology.org/D17-1088.pdf).',
      '- **Xie & Sun,2019, GTS:** оператор раскладывает цель на подцели, листья — известные количества. Переносим разделение structure/number selection; отдельный tree-RNN не удовлетворяет обязательному Transformer encoder–decoder. Можно исследовать AST/token representation в той же разрешённой модели. [Статья](https://www.ijcai.org/Proceedings/2019/736).',
      '- **Hashimoto et al.,2018:** retrieve training example→edit prototype; retriever оптимизируется для лёгкости редактирования, а не общего word overlap. Для нас — сначала sparse baseline, затем encoder input с train-парой и исходным target. Не переносим чужие weights/datasets. [Статья](https://papers.neurips.cc/paper/8209-a-retrieve-and-edit-framework-for-predicting-structured-outputs.pdf).',
      '- **Hayati et al.,2018, RECODE:** retrieval AST subtrees, выравнивание описаний и перенос COPY actions. Переносим идею связывать числовые роли по lexical alignment и давать модели структуру соседа. BLEU наDjango/Hearthstone не равен exec-match наTinySMS. [Статья](https://aclanthology.org/D18-1111.pdf).',
      '- **Parvez et al.,2021, REDCODER:** retrieval code/description как дополнительный контекст генератора; dense retrieval помогает сверхBM25. Переносим схему input augmentation и контроль retrieval-only vsgeneration. Их pretrained DPR/PLBART нельзя использовать; нужен свой encoder–decoder с нуля и train-only sparse index. [Статья](https://aclanthology.org/2021.findings-emnlp.232.pdf).',
      '- **Liu et al.,2023, TinyGSM:** AppendixB: random-init125M/350M/1.3B дают53.1/55.5/57.3% наGSM8K, pretrained —63.3/65.9/68.2. Наши62% измерены на другом in-generator heldout и несопоставимы напрямую. Переносим исполняемый Python и сохранение трудных задач: авторы отмечают, что consensus filtering удаляет сложные примеры. Их81.5% получены с pretrained generator+verifier; verifier обучается на сгенерированных программах и новых binary labels, для нас это запрещено. [Статья](https://arxiv.org/pdf/2312.09241).',
      '- **Eldan & Li,2023, TinyStories:** маленькие модели с нуля выигрывают от ограниченной сложности представления. Переносим мотивацию убрать поверхностный шум; fluent stories и математическая correctness различны. Не переносим создание новых synthetic данных. [Статья](https://arxiv.org/abs/2305.07759).',
      '', '## Ограничения и следующий шаг','',
      'TF–IDF fixed unigram vocabulary не содержит всю семантику; top20 содержит дубли и не исчерпывает retrieval. Равные score на границе top20 делают oracle зависимым от выбранных библиотекой представителей: отдельная CPU-гипотеза — расширить candidates и разнообразить AST-скелеты, не используя gold. Exact AST skeleton — строгий критерий; процентные/precomputed/словесные числа обрабатываются неполно. Numeric taxonomy эвристическая. Полный census и oracle — ретроспективные; gold-free routing показывает, что высокая близость не обеспечивает улучшение. Ни GPU-качество новых представлений, ни полный test timer нового pipeline пока не измерены.',
      '', 'Следующий разумный GPU-шаг — №1: raw против compact приA/D0; затем placeholders. Сохранить нынешнюю A как контроль и B stable main для структурного/длинного продолжения. До финального решения не расходовать fresh и не создавать training targets из любых generation/retrieval diagnostics.',
      '', '## Источники','',
      '- [Pinned датасет и карточка](https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/blob/b2724689ff079e9c22efd548fe134bfb29036dbb/README.md); локальные Parquet60 и SHA-инвентарь №26.',
      '- [Неизменное условие курса](../../task/hw2-code-generation.md): разрешённые данные/архитектура/180с.',
      '- `kaggle-s1-v1`, `kaggle-s2-v2`: собственные сохранённые измерения, [[26b-final-wsd-data-scaling]], [[27-execution-voting-s1-results]], [[27b-fast-sampling-s2-results]], [[29m-28b-capacity-s2-results]].',
      '- Собственные CPU-измерения: [JSON](../results/data-research-ceiling.json), [скрипты](../research/README.md); версии packages/seed/denominators/правила классификации включены.',
      '- Теоретические источники из раздела8 внесены в [[knowledge/bibliography|общую библиографию]].', '']
    text='\n'.join(lines)
    text=re.sub(r'([А-Яа-яёЁ])(?=\d)',r'\1 ',text)
    text=re.sub(r'(?<=\d)(?=[А-Яа-яёЁ])',' ',text)
    (WORK/'notes/data-research-ceiling.md').write_text(text)
    print('note written')

if __name__=='__main__':main()
