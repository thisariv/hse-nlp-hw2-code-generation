"""Render compact Russian S3 notes from verified summary JSONs."""
from common import WORK, read


def front(title):
    return f'''---
type: experiment
course:
  - nlp
status: draft
understanding: 3
created: 2026-10-12
updated: 2026-10-12
sources:
  - kaggle-s3
  - kaggle-s1-v1
  - kaggle-s2-v2
aliases: []
---

# {title}

'''


def representation_note(d):
    text = front('ДЗ2: S3 — результаты компактного таргета и плейсхолдеров')
    text += '''**Решение по заранее записанному правилу: №30 и №31 pending.** Точек 1.92M нет; №30 также не имеет сравнения при равном времени. На 0.96M compact быстрее при близком качестве, placeholders ухудшают качество. Это ранняя диагностика, не окончательное принятие или отказ.

## Полнота, причина остановки и восстановление

Kaggle v1 COMPLETE, внутренний partial, exits `[1,1]`, wall **2211.14 с / 36.85 мин**. Скачаны все **84 output-файла** в ignored `results/kaggle-runs/nlp-hw02-kaggle-s3/1/`; с локальным receipt и API-логом — 86 файлов / 404314068 байт. Все manifest bytes совпали; служебный `__notebook__.ipynb` API не выдаёт. Полный локальный SHA-инвентарь, producer SHA весов/latest/tokenizer, original frozen26 modules и pushed port SHA проверены. Manifest сам содержит bytes, а не SHA.

Оба worker упали при первом forward после оценки и rollback к stable6250: `Inference tensors cannot be saved for backward`. `evaluate` меняла FP32→FP16→FP32 модель внутри `torch.inference_mode`; созданные при конверсии parameter storages стали inference tensors. Это ошибка режима выполнения, а не deadline, OOM или ошибка восстановления программ.

Локальный `s3_runtime.evaluate` исправлен на `torch.no_grad`. [Регрессия](../checks/kaggle-s3-dtype-regression.json) воспроизводит старую ошибку на CPU после half/float, затем проверяет настоящий patched evaluate и два AdamW updates для обоих вариантов; parameter identity сохранена. Старый tiny smoke не ловил ошибку, поскольку CPU выполнял FP32→FP32. Реальный CUDA-путь исправления ещё не проверен. Frozen v1 notebook и downloaded source сохранены; новая сборка и push не выполнялись.

Сохранены для каждого варианта: полные поколения/exec/CE обоих split на0.96M, model-only FP32 cooldown weights и latest full stable checkpoint с AdamW/scaler/scheduler/RNG. Latest: main6250, branch0, cursor800000; optimizer step6250. История включает стоимость уже оценённого cooldown — 960000 compute pairs; она не равна stable cursor. SHA и finite FP32 tensors проверены; после загрузки tensors обычные. Продолжение должно восстановить stable state и сохранить эту compute-историю; текущий worker пока всегда стартует с initial и требует отдельной resume-подготовки.

№32 **не стартовал**: ошибка GPU0 вышла до `B.run`. Generic `32-summary.json` — только partial-маркер, scores0/332800, score batches отсутствуют. Отдельное B scoring требует повтора; существующие B t04 weights SHA подтверждены.

## Равные предъявления

Δ в п.п.; McNemar — exact two-sided, gains/losses относительно того же 26b point. Heldout denominator4096, old только991 finite reference из1024.

| Точка / вариант | Exec4k | Δ / p | Exec991 | Δ / p | Чистый train, с | Пар/с |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
'''
    for tag, label in [('p006250', '0.96M'), ('p012500', '1.92M')]:
        base = d['controls'][tag]; t = base['training']
        text += f"| {label} / 26b | {base['splits']['heldout4k']['matches']} | — | {base['splits']['old1024']['matches']} | — | {t['train_seconds_total']:.2f} | {t['counts']['pairs']/t['train_seconds_total']:.1f} |\n"
        for variant in ['30', '31']:
            p = d['experiments'][variant]['points'].get(tag)
            if p is None:
                text += f'| {label} / №{variant} | missing | — | missing | — | — | — |\n'; continue
            a, b = p['splits']['heldout4k'], p['splits']['old1024']; t = p['training']
            text += f"| {label} / №{variant} | {a['matches']} | {a['McNemar_vs_26b']['delta_percentage_points']:+.2f} / {a['McNemar_vs_26b']['exact_two_sided_p']:.3g} | {b['matches']} | {b['McNemar_vs_26b']['delta_percentage_points']:+.2f} / {b['McNemar_vs_26b']['exact_two_sided_p']:.3g} | {t['train_seconds_total']:.2f} | {t['pairs_per_second_all_compute']:.1f} |\n"
    text += '\n'
    for variant in ['30', '31']:
        p = d['experiments'][variant]['points']['p006250']
        pairs = [p['splits'][s]['McNemar_vs_26b'] for s in ['heldout4k', 'old1024']]
        text += f"№{variant}: gains/losses **{pairs[0]['gains']}/{pairs[0]['losses']}** на4k, **{pairs[1]['gains']}/{pairs[1]['losses']}** на991. "
    text += '''

№30: +15.60% пар/с и −13.50% чистого времени; полезных target tokens51.245M против64.316M (−20.32%). Преобразование prefix1.92M заняло1299.38с и исключено из train timer; для №31 —829.94с. Скорость обучения не означает такой же выигрыш end-to-end при первом запуске. AMP skips0. CE измерена в собственных представлениях и численно с26b не сравнивается.

## Равное время №30

На контрольных928.12/2018.23с независимых compact-веток нет. Единственная compact-точка802.84с находится **ниже первой положительной точки control928.12с**: интерполяция по log-time не имеет нижней опоры. D0 с time0 не позволяет брать logarithm. Экстраполяция не выполнена; equal-time quality и обязательный gate остаются pending.

## Длины исходного reference

BPE22, без specials; bins и denominator совпали с control. Значения — matches/denominator на0.96M; lengths не пересчитаны в новом представлении.

| Split / вариант | <60 | 60–90 | 90–120 | ≥120 |
| --- | ---: | ---: | ---: | ---: |
'''
    for s, label in [('heldout4k', '4k'), ('old1024', '991')]:
        for variant in ['26b', '30', '31']:
            m = d['controls']['p006250']['splits'][s] if variant == '26b' else d['experiments'][variant]['points']['p006250']['splits'][s]
            vals = [f"{b['matches']}/{b['denominator']}" for b in m['length_accuracy'].values()]
            text += f"| {label} / {variant} | {' | '.join(vals)} |\n"
    text += '''
## Ошибки

Пересекающиеся метки из [[data-research-ceiling]], только среди exec ошибок; строка может иметь несколько меток. «Missing» — только цифровое число вопроса, реально использованное reference. Из outside-number исключены фиксированные, словесные и процентные эвристические константы. Same-set означает совпавший набор литералов при неверном ответе, а не доказательство конкретной структурной причины.

| Split / вариант | Все ошибки | Число вне Q/констант | Missing | Same-set | Runtime/non-number |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
    for s, label in [('heldout4k', '4k'), ('old1024', '991')]:
        for variant in ['26b', '30', '31']:
            m = d['controls']['p006250']['splits'][s] if variant == '26b' else d['experiments'][variant]['points']['p006250']['splits'][s]
            e = m['errors']; tags = e['overlapping_tags']
            vals = [e['errors']] + [tags.get(k, 0) for k in ['number_not_from_question_or_known_constant', 'used_question_number_missing', 'same_literal_set_wrong_program', 'runtime_or_non_numeric']]
            text += f"| {label} / {variant} | {' | '.join(map(str, vals))} |\n"
    text += '''
Собственный вывод: placeholders уменьшили missing на4k579→467, но увеличили same-set717→876; общий результат упал. Compact уменьшил runtime139→97, однако не дал выигрыша exec. У обоих вариантов **restore errors0/5120**. Syntax/runtime после успешного восстановления учтены отдельно. В ранних control обнаружены экстремальные int вне float range:4k ID597/866 и old ID971; taxonomy сохраняет точные int и относит их к outside-number, без падения анализа. Это узкое расширение прежней эвристики.

## CPU-проверка и решение

Заново исполнены **23113 уникальных программ**: все10240 восстановленных S3 generations,10240 control generations двух точек и5120 reference rows до дедупликации. Wall3с/CPU2с,2workers; statuses/численные ответы и полные exec-match sets совпали во всех8 evaluations, reference finite identity совпала. Повторная генерация моделью и CE forward не выполнялись.

По frozen gate требуются ≥2251/4096 и≥528/991 **на1.92M**; compact дополнительно должен пройти equal-time gate. Все необходимые финальные точки отсутствуют → **pending**, ни один вариант не принят. Early placeholders имеют отрицательный сигнал, compact — только выигрыш скорости. Рабочая модель остаётся прежней A6.4M. Fresh/test не читались;180с production timer, CUDA fix и A/B ensemble не проверены.

Следующий шаг: короткая CUDA eval→rollback→backward проверка, затем подготовка resume stable6250 для недостающих точек и независимого B scoring; новый push требует отдельной команды пользователя.

## Источники

- [[s3-data-representations-and-b-scoring]] — неизменный протокол до push; [[26b-final-wsd-data-scaling]] — контроль и frozen данные.
- `kaggle-s3` v1: downloaded source/process logs, model-only и stable latest; библиография [[knowledge/bibliography#kaggle-s3]].
- [Summary](../results/kaggle-s3-v1-summary.json), [SHA/CPU audit](../checks/kaggle-s3-v1-cpu-audit.json), [dtype regression](../checks/kaggle-s3-dtype-regression.py), [анализ](../research/s3-analysis.py).
- [[32-candidate-ensemble-s3-results]] — отдельный CPU-выбор кандидатов; [[data-research-ceiling]] — taxonomy.
'''
    if d.get('CPU_B_ensemble'):
        text = text.replace('Отдельное B scoring требует повтора; существующие B t04 weights SHA подтверждены.',
            'Существующие B t04 weights SHA подтверждены. GPU scoring не повторялось; отдельно наCPU/FP32 заполнены168960 scores всех requestedk8/k16/k32+greedy pools, и [[32-candidate-ensemble-s3-results|ансамблевый выбор завершён]]. Неиспользуемые T0.5/T1.0 остаются вне CPU-повтора.')
        text = text.replace('Fresh/test не читались;180с production timer, CUDA fix и A/B ensemble не проверены.',
            'Fresh/test не читались; CUDA fix, FP16 score и production timer≤180с не проверены; CPU A/B выбор выполнен отдельно.')
        text = text.replace('затем подготовка resume stable6250 для недостающих точек и независимого B scoring;', 'затем подготовка resume stable6250 для недостающих точек;')
    return text


def selection_note(d):
    has_b = bool(d.get('ensemble_tune_winner'))
    text = front('ДЗ2: №32 — scores B отсутствуют, CPU-контроли выбора')
    text += r'''**Ансамбль A/B pending:** S3 упала до scoring;0/332800 scores B. Ни probability-B, ни logprob-A/B, ни их dedup-варианты не измерены, доля закрытого ансамблем oracle gap неизвестна. [Причина и сохранённые checkpoints](s3-representations-results.md).

## Frozen протокол

Использованы все сохранённые S1/A6.4M candidates/execution и исходный порядок: greedy + первыеk T0.7, k8/16/32 дают9/17/33 candidates. Exact SHA, code/execution identity и historical votes подтверждены. Старый SHA256 `candidate-selection-v1:heldout4k:{id}` даёт прежние tune2048/eval2048, old991 только для проверки. ID списки и SHA входят в [JSON](../results/32-candidate-ensemble-cpu-summary.json).

Кластеры — connected components math.isclose(rel=abs1e-6), представитель с max meanA. Pure параметры выбраны **только на k16/tune**, перенесены без изменений наk8/k32. Вероятностные ties сохраняют прежний порядок кластеров; отдельный vote-dedup применяет count→greedy→meanA→первый индекс после дедупликации. Exact-code dedup сохраняет max meanA, затем первую копию; AST-нормализации нет. Scorers не получают gold.

Для кандидата c: $a_c=\ell_{A,c}L_{A,c}$, где mean logprob $\ell_A$ сохранена до температурного преобразования, $L_A$ включает EOS. Score кластера — $\log\sum_c\exp(a_c)$ либо $\log\sum_c\exp(\ell_{A,c}/\tau)$. Mean-вес является эвристикой длины, не вероятностью всей программы. A sum без параметров; A mean τ∈{0.05,0.1,0.2,0.5,1,2}, ties по сложности как в [[candidate-selection-cpu]].

B после повтора: $b_c$ — untempered teacher-forced sum по decoded code + EOS, без PAD. Предусмотрены $\log\sum_c\exp(b_c)$ и meanB с прежнейτ-сеткой; комбинация $\log\sum_c\exp(a_c+\lambda b_c)$, λ∈{0,0.25,0.5,1,2,4}, **один коэффициент**. Для mean-комбинации фиксированоτ1: meanA+λmeanB. Обе комбинации с/без exact-code dedup. Сетка расширения сохранена до outcome analysis; B результаты отсутствуют, выбор λ не выполнялся. При partial scores нельзя подменить full split удобной подвыборкой.

## Доступные A-контроли

В ограниченном семействе probability/dedup tune-победитель — **A-mean-dedup, τ0.2**, tune1349/2048. A-mean без dedup выбранаτ0.5. Прежнее более широкое исследование выбирало probability-prior τ0.1/δ0.5 с tune1351; его не заменяли на основании eval. Следующая таблица — новые requested controls, не завершённый №32.

Δ и McNemar относительно простого vote **на том же пуле**. Все p номинальные exact two-sided; это несколько сравнений и повторно использованный heldout.

| k + greedy | Правило | Eval /2048 | Δ п.п. / p | Old /991 | Δ п.п. / p |
| --- | --- | ---: | ---: | ---: | ---: |
'''
    for k, results in d['results'].items():
        for family in ['vote', 'vote-dedup', 'A-sum', 'A-sum-dedup', 'A-mean', 'A-mean-dedup'] + (['B-sum', 'B-sum-dedup', 'B-mean', 'B-mean-dedup', 'AB-sum', 'AB-sum-dedup', 'AB-mean', 'AB-mean-dedup'] if has_b else []):
            r = results[family]; a, b = r['eval'], r['old991']
            text += f"| {k} | {family} | {a['matches']} | {a['delta_vote_pp']:+.2f} / {a['mcnemar_exact_p']:.3g} | {b['matches']} | {b['delta_vote_pp']:+.2f} / {b['mcnemar_exact_p']:.3g} |\n"
    g = d['greedy']
    text += f"\nОтдельный greedy: **{g['eval']['matches']}/2048**, **{g['old991']['matches']}/991**; полный4k2538/4096.\n"
    text += '''
Полные парные gains/losses, CI и reference length bins каждого правила находятся вJSON. SumA заметно хуже vote, что согласуется с сильным штрафом длинных последовательностей; это собственная интерпретация, не отдельная абляция длины. Dedup-mean улучшает k16, но его преимущество над прежним tune-победителем не подтверждено.

## Сколько oracle gap закрыто

Идеальное исправление здесь сохраняет все верные решения baseline и исправляет ошибку, если существует правильный кластер с≥2 votes. Это gold oracle, не исполнимое правило выбора.

| k | Reserve к vote, eval / old, п.п. | Net gain A-only, eval / old | Закрытая доля, eval / old |
| --- | ---: | ---: | ---: |
'''
    for k, rr in d['oracle_gap'].items():
        a, b = rr['eval'], rr['old991']
        text += f"| {k} | {a['reserve_pp']:.2f} / {b['reserve_pp']:.2f} | {a['A_only_winner_net_gain']} / {b['A_only_winner_net_gain']} | {100*a['A_only_fraction_closed']:.1f}% / {100*b['A_only_fraction_closed']:.1f}% |\n"
    text += '''
Для ранее записанного **≈9п.п.** относительно probability-prior: reserve8.98/9.18п.п. (184/91 вопрос). Новый A-only контроль дополнительно исправляет нетто6/4 → **3.3%/4.4%** этого запаса; McNemar p0.497/0.503. Относительно простого vote приk16: +12/+9 →6.4%/9.6%. Знаменатели различаются; это не результат ансамбля. Для A/B fraction_closed остаётсяnull.

## Решение и ограничения

№32 pending; подтверждённого решения об A/B или преимуществе над старым tune-победителем нет. A-only dedup-mean — кандидат для следующей проверки, а не замена production pipeline. Полный k8/k16/k32 inference с B scoring/execution/answers write на1919 ≤180с не измерен; CPU research timer к этому ограничению не относится. Новых GPU-запусков, обучения и push нет; fresh/test не читались.

Следующий шаг: независимо получить B scores с pinned S2/2 t04, затем тем же tune-split выбрать единственный λ и проверить eval/old991 и полный T4 timer.

## Источники

- [[candidate-selection-cpu]] — frozen split, кластеры, A grids и старый oracle; [[s3-data-representations-and-b-scoring]] — B scoring convention.
- [[s3-representations-results]] — KaggleS3v1 partial; [[29m-28b-capacity-s2-results]] — B t04 origin/SHA.
- [JSON](../results/32-candidate-ensemble-cpu-summary.json), [CPU-анализ](../research/s3-analysis.py), [воспроизведение](../research/README.md).
- Собственные pinnedS1/S2/S3 outputs; записи `kaggle-s1-v1`, `kaggle-s2-v2`, `kaggle-s3` в [[knowledge/bibliography]].
'''
    if has_b:
        b = d['B_CPU_scoring']; rule = d['ensemble_tune_winner']; overall = d['tune_winner']
        text = text.replace('# ДЗ2: №32 — scores B отсутствуют, CPU-контроли выбора', '# ДЗ2: №32 — ансамблевый выбор с CPU scores B')
        text = text.replace('**Ансамбль A/B pending:** S3 упала до scoring;0/332800 scores B. Ни probability-B, ни logprob-A/B, ни их dedup-варианты не измерены, доля закрытого ансамблем oracle gap неизвестна.',
            f"**CPU-часть №32 завершена.** S3 упала до scoring; её GPU scores0/332800. Отдельно наCPU/FP32 выполнены **168960 scores** для всех9/17/33 candidates5120 вопросов (**{b['identity']['unique_programs']} уникальных внутри вопроса**). Forward+chunk I/O: {b['seconds']:.1f}с; source preparation иmodel load внеtimer. Неиспользуемые T0.5/T1.0 вне CPU-повтора. Это новый score, а не восстановленный T4/FP16 output S3.")
        text = text.replace('B после повтора:', 'Pinned S2/2 B t04, SHA`cbbe292c…f720a`, CPU FP32:')
        text = text.replace('B результаты отсутствуют, выбор λ не выполнялся. При partial scores нельзя подменить full split удобной подвыборкой.',
            'Все scores для requested pools полные. SHA chunks, codeSHA/ID/temperature и sum=mean×length проверены; при λ0 AB-sum точно совпадает сA-sum. λ по умолчанию1, ties затем abs(λ−1) и порядок сетки; scorers не получаютreference.')
        text = text.replace('Следующая таблица — новые requested controls, не завершённый №32.',
            f"Общий победитель requested rules поk16/tune — **{overall}**, параметры`{d['selected_parameters'][overall]}`; лучший среди B/AB — **{rule}**, параметры`{d['selected_parameters'][rule]}`. Все параметры зафиксированы до eval/old reporting; полный tune-grid вJSON.")
        text = text.replace('В ограниченном семействе probability/dedup tune-победитель —', 'В A-only контролях tune-победитель —')
        text = text.replace('Для A/B fraction_closed остаётсяnull.', 'Доля для B/AB рассчитана ниже относительно тех же frozen baseline.')
        start = text.index('## Решение и ограничения')
        end = text.index('## Источники', start)
        extra = '## Ансамбль: frozen tune-победитель\n\n'
        extra += f"Выбран **{rule}**, `{d['selected_parameters'][rule]}`, tune{d['results']['16'][rule]['tune_matches']}/2048. Δ и p к простомуvote на том же пуле.\n\n"
        extra += '| k + greedy | Eval /2048 | Δ / p | Old /991 | Δ / p | Доля oracle gap кvote, eval / old |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
        for k, rr in d['results'].items():
            a, c = rr[rule]['eval'], rr[rule]['old991']; gap = d['oracle_gap'][k]
            extra += f"| {k} | {a['matches']} | {a['delta_vote_pp']:+.2f} / {a['mcnemar_exact_p']:.3g} | {c['matches']} | {c['delta_vote_pp']:+.2f} / {c['mcnemar_exact_p']:.3g} | {100*gap['eval']['ensemble_fraction_closed']:.1f}% / {100*gap['old991']['ensemble_fraction_closed']:.1f}% |\n"
        gap = d['gap_against_previous_tune_winner']
        a, c = gap['eval'], gap['old991']
        extra += f"\nОтносительно ранее записанного≈9п.п. запаса (8.98/9.18 кprobability-prior): ансамбль закрывает **{100*a['ensemble_fraction_closed']:.1f}% / {100*c['ensemble_fraction_closed']:.1f}%**, нетто{a['ensemble_incremental_net_gain']}/{c['ensemble_incremental_net_gain']} ответов. McNemar против старого правила p{a['ensemble_McNemar']['mcnemar_exact_p']:.3g}/{c['ensemble_McNemar']['mcnemar_exact_p']:.3g}. Отрицательная доля означает net loss, не исчерпанный запас. Это gold oracle diagnosis, не прогноз leaderboard.\n\n"
        extra += '''## Решение и ограничения

Полный CPU-выбор A/B выполнен; таблицы показывают результат frozen tune-победителя, независимо от eval/old. **Убедительного выигрыша не показано:** наk16 Δ+0.29/+0.40п.п. кvote,p0.392/0.454; кпрежнему правилу0/−1ответ,p1/1. Лучшие отдельные eval числа других семейств не позволяют заменить победителя. B scoring вproduction по этим данным не добавлять; сохранить прежниеA/greedy/vote контроли.

McNemar exploratory после нескольких сравнений и повторно использованного heldout. CPU FP32 и будущий T4 FP16 могут дать разные вероятности: итоговый CUDA pipeline требуется проверить отдельно. Полный1919≤180с не измерен; CPU scoring time в исследовании его не устанавливает. Новых GPU, обучения и push нет; fresh/test не читались. №30/31 остаютсяpending.

Следующий шаг: CUDA eval→rollback→backward regression и подготовка resume№30 до1.92M/equal-time; fullk8/k16 timer остаётся открытым для прежнегоvote без добавленияB. №31 ранне ухудшает качество и имеет меньший приоритет, но его frozen1.92M gate формальноpending.

'''
        text = text[:start] + extra + text[end:]
    return text


if __name__ == '__main__':
    (WORK / 'notes/s3-representations-results.md').write_text(representation_note(read(WORK / 'results/kaggle-s3-v1-summary.json')))
    (WORK / 'notes/32-candidate-ensemble-s3-results.md').write_text(selection_note(read(WORK / 'results/32-candidate-ensemble-cpu-summary.json')))
