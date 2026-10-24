"""Render frozen CPU selection results; never select a rule using eval/old."""
from common import WORK, read


def pct(value):
    return f'{100*value:.2f}%'


def delta(value):
    return f'{value:+.2f}'


def params(config):
    return ', '.join(f'{k}={v:g}' for k, v in config.items()) or '—'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---']*len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def main():
    j = read(WORK / 'results/candidate-selection-cpu.json')
    rr, winners = j['results'], j['winners']
    best = winners['pure16']['family']
    primary = rr['pure16'][best]
    ev, old = primary['eval'], primary['old991']
    p = j['train_prior']
    out = ['## Результаты', '',
        f'**Победитель только по tune/k16:** `{best}`, {params(primary["parameters"])}; '
        f'tune {primary["tune_matches"]}/2048. Eval **{ev["matches"]}/2048 = {pct(ev["accuracy"])}**, '
        f'Δ {delta(ev["delta_vote_pp"])} п.п.; old991 **{old["matches"]}/991 = {pct(old["accuracy"])}**, '
        f'Δ {delta(old["delta_vote_pp"])} п.п. [Все числа, split ID, сетки и QA](../results/candidate-selection-cpu.json); '
        '[воспроизведение](../research/README.md#выбор-кандидата-на-cpu).', '',
        'Зафиксированное основное правило:', '',
        r'$$S(C)=\log\sum_{p\in C}\exp(\bar\ell_p/0.1)+0.5\,r(a_C).$$', '',
        r'Здесь $C$ — кластер численных ответов; $p$ — программа; $\bar\ell_p$ — её сохранённый mean logprob; '
        '$a_C$ — ответ представителя с максимальным mean logprob; $r$ равно0 для нецелого,1 для целого, '
        '2 для ненулевого целого кратного10. Выбирается кластер с максимальным $S(C)$; '
        'score усиливает уверенные программы и слабо предпочитает типичную форму ответа.', '',
        '> [!note] Исправление исходного числа контроля',
        '> В прежней заметке о потолке 2707/4096 ошибочно приписаны голосованию с greedy. '
        'S1/27 и независимый replay: **k16 без greedy 2707**, **k16 с greedy 2703**. '
        'Первоначальный протокол выше сохранён дословно; исправлена только проверка известного baseline до подбора новых параметров. '
        'Основные таблицы ниже сравнивают с greedy+samples на том же пуле; контроли без greedy приведены отдельно.', '',
        f'Разделение: tune2048/eval2048; old991. Протокол SHA `{j["protocol"]["protocol_sha256"]}`; '
        f'UTC фиксации `{j["protocol"]["registered_utc"]}`. '
        'Победители pure16/mixed16/all65 зафиксированы до отчёта eval/old; параметры pure16 и mixed16 перенесены на k8/k32.', '',
        f'Априорная форма train: finite {p["sample_finite"]}; целые **{pct(p["integer_fraction_weighted"])}** '
        f'(unweighted {pct(p["integer_fraction_unweighted"])}), ненулевые кратные10 '
        f'**{pct(p["nonzero_multiple10_fraction_weighted"])}** (unweighted {pct(p["nonzero_multiple10_fraction_unweighted"])}). '
        'Большая доля целых сама по себе не означает полезность бонуса: его коэффициент решает только tune.', '',
        '### Все семейства: основной k16 и all65', '',
        'Δ — в п.п.; p — exact McNemar против vote того же пула. Параметры в каждой таблице выбраны только на соответствующем tune. '
        'Полные gains/losses, CI и все длины каждого правила находятся в JSON.', '']
    for name, label in [('pure16', '16@T0.7 + greedy'), ('all64', 'Все65')]:
        out += [f'**{label}; победитель tune: `{winners[name]["family"]}`.**', '',
            table(['Правило', 'Параметры', 'Eval', 'Old991', 'Δeval', 'Δold', 'p eval / old'],
            [[f, params(x['parameters']), pct(x['eval']['accuracy']), pct(x['old991']['accuracy']),
              delta(x['eval']['delta_vote_pp']), delta(x['old991']['delta_vote_pp']),
              f'{x["eval"]["mcnemar_exact_p"]:.3g} / {x["old991"]["mcnemar_exact_p"]:.3g}'] for f, x in rr[name].items()]), '']
    out += ['### Чувствительность к числу samples', '',
        'В каждой клетке exec-match / Δ к голосованию, п.п. Параметры всех семейств взяты с pure16/tune и одинаковы для трёх k.', '',
        '> [!details]- Все семейства на k8/k16/k32',
        '\n'.join('> '+line for line in table(['Правило', 'Eval k8', 'Eval k16', 'Eval k32', 'Old k8', 'Old k16', 'Old k32'],
            [[f] + [f'{pct(rr[f"pure{k}"][f][s]["accuracy"])} / {delta(rr[f"pure{k}"][f][s]["delta_vote_pp"])}'
                    for s in ['eval', 'old991'] for k in [8, 16, 32]] for f in rr['pure16']]).splitlines()), '',
        table(['Выбранное pure16-правило', 'k samples', 'Eval', 'Old991', 'Δeval / old'],
            [[best, k, pct(rr[f'pure{k}'][best]['eval']['accuracy']), pct(rr[f'pure{k}'][best]['old991']['accuracy']),
              f'{delta(rr[f"pure{k}"][best]["eval"]["delta_vote_pp"])} / {delta(rr[f"pure{k}"][best]["old991"]["delta_vote_pp"])}']
             for k in [8, 16, 32]]), '',
        '**Контроль без greedy** (отдельный пул, Δ здесь не вычисляется):', '',
        table(['k', 'Eval', 'Old991'], [[k, pct(rr[f'nogreedy{k}']['vote']['eval']['accuracy']),
            pct(rr[f'nogreedy{k}']['vote']['old991']['accuracy'])] for k in [8, 16, 32]]), '',
        '### Смешивание температур', '',
        f'Mixed16-победитель tune: `{winners["mixed16"]["family"]}`, {params(winners["mixed16"]["parameters"])}. '
        'В mixed k/2 samples T0.7 и по k/4 T0.5/T1.0; общий budget k+1. '
        'Все семейства и разбивки mixed8/16/32 есть в JSON.', '',
        table(['Пул', 'Правило', 'Eval', 'Old991', 'Δ к mixed-vote eval/old', 'Δ к pure-vote eval/old'],
            [[f'mixed{k}', f, pct(rr[f'mixed{k}'][f]['eval']['accuracy']), pct(rr[f'mixed{k}'][f]['old991']['accuracy']),
              f'{delta(rr[f"mixed{k}"][f]["eval"]["delta_vote_pp"])} / {delta(rr[f"mixed{k}"][f]["old991"]["delta_vote_pp"])}',
              f'{delta(rr[f"mixed{k}"][f]["eval_against_pure_vote"]["delta_vote_pp"])} / {delta(rr[f"mixed{k}"][f]["old991_against_pure_vote"]["delta_vote_pp"])}']
             for k in [8, 16, 32] for f in ['vote', 'temperature-mix', winners['mixed16']['family']]]), '',
        '### Длины reference', '',
        'Бакеты заданы по reference BPE8k без EOS. Ни длина, ни reference не подаются правилу.', '',
        table(['Пул / правило', 'Срез', 'Ref BPE', 'n', 'Exec-match', 'Δvote'],
            [[f'{name} / {winners[anchor]["family"]}', s, b, x['n'], pct(x['accuracy']), delta(x['delta_vote_pp'])]
             for name, anchor in [('pure8', 'pure16'), ('pure16', 'pure16'), ('pure32', 'pure16'), ('all64', 'all64')]
             for s in ['eval', 'old991'] for b, x in rr[name][winners[anchor]['family']][s]['by_reference_length'].items()]), '',
        '### Потенциал реранкера без обучения', '',
        'Oracle использует gold только для **диагностики**. «Исправить≥2» — идеальный выбор правильного повторяющегося кластера, '
        'если он есть, с сохранением остальных предсказаний выбранного правила. Это верхняя граница, не ожидаемая обучаемая accuracy. '
        'Отдельно считать≥2 разных code: одинаковые samples не доказывают независимое согласие.', '',
        table(['Пул', 'Срез', 'Выбрано', 'Oracle любой', 'Oracle≥2', 'Oracle≥2 разных code', 'Исправить≥2', 'Запас к правилу'],
            [[name, s, f'{x["selected_matches"]}/{x["n"]}', pct(x['any_candidate_oracle']['accuracy']),
              pct(x['repeated_cluster_oracle']['accuracy']), pct(x['repeated_distinct_programs_oracle']['accuracy']),
              pct(x['repair_repeated_keep_other_predictions']['accuracy']), delta(x['repair_repeated_keep_other_predictions']['delta_best_pp'])]
             for name in ['pure8', 'pure16', 'pure32', 'all64'] for s, x in j['reranker_potential'][name].items()]), '',
        'Ранг лучшего правильного кластера по **выбранному на tune** score (число вопросов):', '',
        table(['Пул / срез', '1', '2', '3', '4–5', '6–10', '>10', 'Нет'],
            [[f'{name} / {s}'] + [x['correct_rank_histogram'][b] for b in ['1', '2', '3', '4-5', '6-10', '>10', 'absent']]
             for name in ['pure8', 'pure16', 'pure32', 'all64'] for s, x in j['reranker_potential'][name].items()]), '',
        'Ранги только на ошибках, cumulative top2/3/5/10, singleton availability и строгий oracle по разным code находятся в JSON.', '',
        '### Практический вывод и GPU-признаки', '',
        '**Надёжного большого CPU-прироста нет.** Строгий победитель pure16/tune даёт '
        f'{delta(ev["delta_vote_pp"])} п.п. eval (27 gains /21 losses, p={ev["mcnemar_exact_p"]:.3g}); '
        f'old {delta(old["delta_vote_pp"])} п.п. (12/7, p={old["mcnemar_exact_p"]:.3g}). '
        f'95% парный CI eval [{ev["delta_paired_normal_95ci_pp"][0]:+.2f}; {ev["delta_paired_normal_95ci_pp"][1]:+.2f}], '
        f'old [{old["delta_paired_normal_95ci_pp"][0]:+.2f}; {old["delta_paired_normal_95ci_pp"][1]:+.2f}] п.п. '
        'Рабочее ожидание для этого правила — небольшой эффект порядка0–0.5 п.п., включая возможность нулевого/отрицательного эффекта; '
        'ожидать закрытия oracle gap нельзя. Это интерпретация измерений, не прогноз leaderboard.', '',
        'Однопараметрическая альтернатива `probability-mean-dedup`, τ=0.2 на pure16: eval67.09% '
        '(+0.59 п.п.;21/9,p0.0428), old69.12% (+0.91;12/3,p0.0352). Её τ выбран на tune, '
        'но семейство уступило основному победителю на2 tune-примерах. Предпочесть её **после** просмотра eval было бы новой подгонкой; '
        'оставить отдельной заранее фиксированной альтернативой для будущего fresh. Nominal p без поправки на перебор семейств '
        'не является окончательным подтверждением.', '',
        'Полная вероятность последовательности ухудшает k16 на1.61 п.п.: она усиливает предпочтение коротких/уверенных программ. '
        'Умножение частоты samples на их вероятность также может повторно усилить уверенность того же генератора. '
        'Mean logprob с exact-code дедупликацией выглядит полезнее, но выигрыши малы. '
        'Pure16/tune отключил coverage/full/channel бонусы (β=0); literal penalty не переносится. '
        'Train79.16% целых подтверждает форму как слабый prior, не как средство восстановить структуру. '
        'Mixed-пулы при одинаковом k не дали устойчивого преимущества; all65 не оправдывает увеличение генерации ради такого прироста.', '',
        '**Бюджет180 с:** первым замерять k8/T0.7 на исправленном быстром sampler, затем k16 только если полный pipeline укладывается. '
        'У frozen pure16-правила k8→k16 даёт +0.68 п.п. eval/+1.41 old, k16→k32 даёт +0.44 eval/−0.30 old; '
        'all65-победитель относительно pure16-победителя +1.03 eval/+0.30 old при почти четырёхкратном числе samples. '
        'По старому S1 полный test k16=309.53 с, k32=493.33 с; они превышают180 с. '
        'k8 и новый sampler этим исследованием по T4 не проверены. Голосование сохранять практическим control: '
        'у primary-победителя на k8 old хуже на0.30 п.п. Ни k32, ни all65 сейчас не рекомендовать для сдачи.', '',
        '**Запас реранкера реален:** pure16/eval ideal repair≥2 даёт75.78% против66.80%, '
        'то есть +8.98 п.п.; old77.90% против68.72%, +9.18. Для≥2 разных программ остаётся +8.64/+8.88 п.п. '
        'На eval среди295 ошибок с доступным правильным кластером188 (63.7%) имеют его на ранге2–3. '
        'В all65 идеальный выбор повторяющегося кластера достигает83.40% eval/85.27% old, '
        'но не обещает достижимость этого качества обучаемой моделью.', '',
        '- **Score B:** следующий GPU inference-only gate — teacher forcing существующей B по уникальным A-кандидатам '
        'на k8/k16, untempered mean logprob с EOS, затем ансамбль A/B и агрегация exact-code-dedup кластеров. '
        'Один коэффициент смешивания выбирать только на tune; B уже обучена на разрешённых исходных парах, '
        'новые таргеты не создаются. Проверить full-pool score и вариант только top2–3 кластеров для времени; '
        'экономия вычислений уменьшает oracle. Выигрыш заранее неизвестен: B greedy немного хуже A, '
        'комплементарность вероятностей ещё не измерена.',
        '- **Обратная модель code→question / noisy channel:** score log p(question|code) проверяет согласие решения с вопросом, '
        'а не только уверенность прямого генератора. Для обучения нужна отдельная Transformer encoder–decoder с нуля '
        'и GPU; пары остаются теми же, вопрос становится target. **Серая зона допустимости:** новые строки данных отсутствуют, '
        'но направление и обучающий target меняются; условие запрещает дополнительные таргеты. '
        'Согласовать эту трактовку с преподавателем до обучения. Здесь модель не обучалась.',
        '- **Обучаемый verifier/reranker на correctness сгенерированных программ:** оценён только oracle. '
        'Такое обучение создаёт дополнительные binary targets и по текущему условию не предлагается. '
        'Числовые coverage/novelty, размер/доля кластера, distinct-code count, длина и форма ответа уже доступны на CPU; '
        'их вычисление само по себе GPU не требует.', '',
        '### Проверка и ограничения', '',
        f'CPU финального полного прогона: {j["timing"]["total_cpu_seconds"]:.1f} с; '
        f'цифровые признаки для обоих all65 наборов: {j["timing"]["number_feature_seconds"]:.1f} с. '
        'Это исследовательский проход по множеству правил и независимым проверкам, а не production timer. '
        f'SHA {j["sources"]["heldout4k"]["candidate_files"] + j["sources"]["old1024"]["candidate_files"]} candidate-файлов и '
        f'{j["sources"]["heldout4k"]["execution_files"] + j["sources"]["old1024"]["execution_files"]} execution-файлов, '
        f'execution identities/alignment и {j["QA"]["direct_vote_comparisons"]} независимых vote-сравнений прошли. '
        f'Нетранзитивные компоненты: {sum(v for pp in j["QA"]["nontransitive_components"].values() for v in pp.values())}. '
        'Сохранённые исполнения переиспользованы, GPU/Kaggle/обучения нет. '
        'Fresh/test не читались; ≤180 с на T4 не проверено. McNemar exploratory без поправки на несколько семейств; '
        'old991 и heldout неоднократно использовались исторически.', '',
        '## Источники', '',
        '- [[data-research-ceiling|CPU-аудит данных, раздел 6]]; существующая stratified60k train-выборка №data-research.',
        '- [[27-execution-voting-s1-results|S1/27]], сохранённые журналы candidate/execution; источник `kaggle-s1-v1` в [[knowledge/bibliography|библиографии]].',
        '- [Условие курса](../../task/hw2-code-generation.md), локальная неизменная копия.',
        '- [JSON](../results/candidate-selection-cpu.json), [CPU-скрипт](../research/candidate-selection.py), '
        '[рендер заметки](../research/write-candidate-selection-note.py): собственные измерения и выводы.', '']
    note = WORK / 'notes/candidate-selection-cpu.md'
    prefix = note.read_text().split('## Результаты')[0]
    note.write_text(prefix + '\n'.join(out))


if __name__ == '__main__':
    main()
