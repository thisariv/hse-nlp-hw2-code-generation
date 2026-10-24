"""Assemble a separate adaptive L3 bundle from frozen v4 assets, no Kaggle API."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
BUNDLE = ROOT/'bundles/final_inference_l3_v5'
READY = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final/next-inference-v5.json'


def replace(source, old, new):
    assert source.count(old) == 1, f'V4 template drift: {old}'
    return source.replace(old, new)


def build():
    BUNDLE.mkdir(parents=True, exist_ok=True)
    READY.write_text(json.dumps({'bundle': str(BUNDLE.resolve()), 'ready': False}, indent=2)+'\n')
    original = json.loads((ROOT/'bundles/final_inference_l3/nlp-hw02-final-inference-l3.ipynb').read_text())
    first = next(c['source'] for c in original['cells'] if c['cell_type'] == 'code')
    sources = ast.literal_eval(next(n.value for n in ast.parse(first).body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SOURCES'))
    base_identity = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()}
    assert base_identity == json.loads((ROOT/'final-l3-source-identity.json').read_text())['embedded_source_sha256']
    analysis = json.loads((WORK/'checks/final-inference-v5-offline.json').read_text())
    assert analysis['passed']
    k2 = analysis['selected_K2']
    selected = next(r for r in analysis['rows'] if r['K2'] == k2)
    baseline = selected['matches']
    config = {'K1': 2, 'K2': k2, 'unanimity_threshold': 1., 'sequence_budget': 1024,
              'greedy_sequence_budget': 1024,
              'scorer_enabled': False, 'scorer_weight': 0., 'expected_matches': baseline}
    sources['final_inference.py'] = (ROOT/'src/final_inference_v5.py').read_text()
    sources['final_adaptive.py'] = (ROOT/'src/final_adaptive.py').read_text()
    runner = sources['final_runner.py']
    runner = replace(runner, "budget = float(config.get('budget_seconds', 180.))",
        "budget = min(180., float(config.get('budget_seconds', 180.)))")
    sources['final_runner.py'] = replace(runner, "    timing['target_150_seconds'] = timing['full_pipeline_seconds'] <= 150.",
        "    timing['target_150_seconds'] = timing['full_pipeline_seconds'] <= 150.\n"
        "    timing['target_155_seconds'] = timing['full_pipeline_seconds'] <= 155.")
    session = sources['final_session.py']
    session = replace(session, 'from final_vote import select, validate_answers',
        'from final_vote import validate_answers\nfrom final_adaptive import replay_record, stage1_decision')
    session = replace(session, "win = select(main['greedy'], main['samples']['0.7'][:4], long['samples']['0.7'][:4])",
        f"win, _, _ = replay_record(main['greedy'], main['samples']['0.7'], long['samples']['0.7'], k1=2, k2={k2})")
    session = replace(session, 'assert matches == 1511', f'assert matches == {baseline}')
    session = replace(session, "'budget_seconds': 180., 'sample_batch': 256, 'greedy_batch': 1024}",
        "'budget_seconds': 180., 'greedy_batch': 1024, **"+repr(config)+'}')
    session = replace(session, "'test_all_under_120': all(r['target_120_seconds'] for r in report['test']),",
        "'test_with_25s_margin': all(r['target_155_seconds'] for r in report['test']),")
    session = replace(session, "report['K4_diagnostic']", "report['adaptive_diagnostic']")
    lo = session.index('def diagnose(')
    hi = session.index('\n\ndef suite(', lo)
    session = session[:lo]+'''def diagnose(records, journals, ids):
    result = {}
    for name, target in [('C-main', 'main'), ('C-long2x', 'long')]:
        code_diff = sum(c['code'] != journals[name][i]['samples']['0.7'][s]['code']
                        for r, i in zip(records, ids) for s, c in enumerate(r[target]))
        result[name] = {'sample_counts': {str(k): sum(len(r[target]) == k for r in records)
                                        for k in sorted({len(r[target]) for r in records})},
                        'journal_prefix_code_mismatches': code_diff}
    result['stage1_stopped'] = sum(stage1_decision(r, 2) is True for r in records)
    result['greedy_code_mismatches'] = sum(r['greedy']['code'] != journals['C-main'][i]['greedy']['code']
                                         for r, i in zip(records, ids))
    return result
'''+session[hi:]
    sources['final_session.py'] = session
    cells = []
    def cell(kind, source):
        row = {'cell_type': kind, 'id': f'final-v5-{len(cells):02d}', 'metadata': {}, 'source': source}
        if kind == 'code':
            compile(source, row['id'], 'exec')
            row.update(outputs=[], execution_count=None)
        cells.append(row)
    cell('markdown', f'''# ДЗ2 NLP: адаптивный инференс v5 на L3

Private, Internet off, одна T4/GPU0, аккаунт friend1. Greedy C-main → K2+K2 T0.7 → исполнение. Все пять численных ответов совпали с greedy (isclose1e-6): стоп. Остальные добираются до K{k2}+K{k2}; sequence_budget=1024, вопросов в батче =1024//K. Greedy1024; stage1 512×K2; stage2 {1024//(k2-2)}×K{k2-2}. OOM retry с восстановлением RNG и сохранением уменьшенного батча. После всех stage1 executions: сначала минимальная доля лидирующего ответа, длина только при равной доле. Каждый priority batch получает main и long подряд. Голос/код как v3/v4: count → greedy → mean LP → порядок, greedy в победившей компоненте, иначе max LP.

K1=2, K2={k2} всего на модель; unanimity_threshold=1.0. Настройки в session config, scorer выключен. Один cold test1919 + tune2048, без обучения. Таймер включает imports/load/warmup/execution/reexecution/vote/write. Цель ≤155 с (≥25 с до потолка), обязательный предел <180 с. Generation guard160 / execution166 / verification170 / supervisor172с: минимум20с после планового окончания генерации. Guard сохраняет доступные голоса; внешний supervisor сохраняет greedy при аварии.

Офлайн-ориентир {baseline}/2048, допуск новых RNG lanes ±1 п.п. Прогноз по измеренному v5 T4: test≈{selected['timing']['nominal_seconds']:.1f}с, с округлением батчей/10%/CPU barrier≤{selected['timing']['batch_bound_with_reserve_seconds']:.1f}с; оценка, не новый GPU gate. K6–K8 не дают требуемого запаса. Frozen v3/v4 vote сохранён. Веса SHA/tag L3 l2-62280. Сессия ≤720с, watchdog660/690; ровно два процесса test+tune. CPU-ready не означает пройденный GPU gate.''')
    init = first[:first.index('SOURCES=')]+'SOURCES='+repr(sources)+first[first.index('\nfor name,source in SOURCES.items()'):]
    init = init.replace('final-inference-l3', 'final-inference-l3-v5')
    cell('code', init)
    run = next(c['source'] for c in original['cells'] if c['cell_type'] == 'code' and 'def find_input' in c['source'])
    cell('code', run)
    cell('markdown', 'Проверить summary.json, test-1/timing.json/adaptive и tune/quality.json. Ответ всегда единственный test-1. При guard/partial/reexecution failure результат требует ревью. Повторяемость не измеряется одним test.')
    notebook = dict(original, cells=cells)
    name = 'nlp-hw02-final-inference-l3-v5.ipynb'
    target = ROOT/'notebooks'/name
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n')
    (BUNDLE/name).write_bytes(target.read_bytes())
    metadata = json.loads((ROOT/'bundles/final_inference_l3/kernel-metadata.json').read_text())
    metadata.update(id='friend1/nlp-hw02-adaptive-inference-l3-v5', title='NLP HW02 adaptive inference L3 v5', code_file=name)
    limits = {'kind': 'final_inference_l3_v5', 'session_timeout_seconds': 720, 'watchdog_stop_seconds': 660,
              'watchdog_exit_seconds': 690, 'cold_test_runs': 1, 'tune_runs': 1, 'adaptive': config}
    for filename, value in [('kernel-metadata.json', metadata), ('session-limits.json', limits)]:
        (BUNDLE/filename).write_text(json.dumps(value, indent=2)+'\n')
    identity = {'base_v4_sha256': base_identity, 'adaptive_config': config,
                'embedded_source_sha256': {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()}}
    (ROOT/'final-v5-source-identity.json').write_text(json.dumps(identity, indent=2)+'\n')
    print(target)
    return target, sources


if __name__ == '__main__':
    build()
