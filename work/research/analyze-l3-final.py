"""Audit final L3 journals on CPU; selection sees no gold, no new sampling."""
import importlib.util
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = Path(__file__).resolve().parent.parent
FINAL = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-l3/final'
TAG = 'l2-62280'
spec = importlib.util.spec_from_file_location('mid_analysis', WORK / 'research/analyze-l3-results.py')
mid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mid)
l2 = mid.l2
MODELS = mid.MODELS


def path(model, suffix):
    return FINAL / f'model-{model}__points__{TAG}__{suffix}.json'


def load_final(model):
    training, metrics, complete, checkpoint = [l2.read(path(model, s)) for s in
                                              ['training', 'evaluation__metrics', 'complete', 'checkpoint']]
    latest = l2.read(FINAL / f'model-{model}__latest.json')
    assert training['final'] and 3000 <= training['cooldown_seconds'] < 3001
    assert complete['metrics'] == metrics and metrics['fresh_evaluated'] == 0
    assert all(complete['training'][k] == v for k, v in training.items())
    weight = FINAL / f'model-{model}__weights__{TAG}-fp16.pt'
    assert l2.sha(weight) == checkpoint['sha256'] == complete['weights_sha256'] == metrics['weights_sha256']
    assert weight.stat().st_size == checkpoint['bytes']
    for tr, la in [('anchor_updates', 'updates'), ('anchor_cursor', 'cursor'),
                   ('main_seconds', 'main_seconds'), ('l2_main_seconds', 'l2_main_seconds')]:
        assert training[tr] == latest[la]
    records = {}
    for split, n in [('heldout4k', 4096), ('old991', 991)]:
        rows = l2.read(path(model, f'evaluation__{split}-per-example'))
        assert len(rows) == n and [r['id'] for r in rows] == list(range(n))
        assert all(r['correct'] == l2.matches(r['reference'], r['execution']) for r in rows)
        assert [r['id'] for r in rows if r['correct']] == metrics['splits'][split]['exec']['match_indices']
        assert math.isfinite(metrics['splits'][split]['CE']['CE'])
        for label, lo, hi in [('<60', 0, 60), ('60–90', 60, 90), ('90–120', 90, 120), ('≥120', 120, math.inf)]:
            chosen = [r for r in rows if lo <= r['original_bpe_length'] < hi]
            b = metrics['splits'][split]['length_accuracy'][label]
            assert len(chosen) == b['denominator'] and sum(r['correct'] for r in chosen) == b['matches']
        records[split] = rows
    for split, ids in l2.SPLIT.items():
        assert l2.score(records['heldout4k'], ids)['matches'] == metrics['paired_subsets'][split]['matches']
    point = l2.point_row(model, TAG, training, metrics, records)
    point.update(training=training, point_complete=True, evaluation_s=complete['training']['evaluation_seconds'])
    return records, point, latest


def vote_journal(model, greedy):
    files = sorted(FINAL.glob(f'model-{model}__points__{TAG}__evaluation__k8-tune2048__batch-*.json'))
    assert len(files) == 64
    rows = [r for p in files for r in l2.read(p)]
    index = {r['id']: r for r in rows}
    assert len(rows) == len(index) == 2048 and sorted(index) == l2.SPLIT['tune2048']
    for i, r in index.items():
        assert r['question_sha256'] == greedy[i]['question_sha256'] and r['reference'] == greedy[i]['reference']
        assert r['greedy']['execution'] == greedy[i]['execution']
        samples = r['samples']['0.7']
        assert len(samples) == 8 and [s['sample'] for s in samples] == list(range(8))
        assert all(s['sampling_status'] == 'complete' and math.isfinite(s['mean_logprob']) for s in samples)
        win = l2.choose({'greedy': r['greedy'], 'samples': r['samples']},
                        {'mode': 'vote', 'temperature': .7, 'k': 8, 'greedy_vote': False})
        assert win['execution'] == r['vote_execution'] and l2.matches(r['reference'], win['execution']) == r['correct']
    summary = l2.read(path(model, 'evaluation__k8-tune2048__summary'))
    assert summary['complete'] and sum(r['correct'] for r in rows) == summary['matches']
    return index


def outcomes(name, x, y):
    # Same frozen selector, original candidate order and greedy tie anchor as L2.
    mixed = name.startswith('mixed')
    long = name.startswith('C-long ')
    k = 8 if 'K8' in name else 4
    result = []
    for i in l2.SPLIT['tune2048']:
        row = y[i] if long else x[i]
        samples = x[i]['samples']['0.7'][:k] + y[i]['samples']['0.7'][:k] if mixed else row['samples']['0.7'][:k]
        win = l2.choose({'greedy': row['greedy'], 'samples': {'0.7': samples}},
                        {'mode': 'vote', 'temperature': .7, 'k': len(samples), 'greedy_vote': '+greedy' in name})
        result.append(l2.matches(row['reference'], win['execution']))
    return result


def curve(model, points):
    short = [points[0]] + [p for p in points if p['model'] == model and p['cooldown_s'] < 1000]
    finals = [p for p in points if p['model'] == model and p['cooldown_s'] > 2900]
    t = np.array([p['main_s'] for p in short]); a = np.array([p['scores']['heldout4k']['accuracy_pp'] for p in short])
    log = np.polyfit(np.log(t), a, 1)
    power = np.polyfit(np.log(t), np.log(100-a), 1)
    old, new = finals
    accuracy = lambda p: p['scores']['heldout4k']['accuracy_pp']
    slope50 = (accuracy(new)-accuracy(old))/math.log(new['main_s']/old['main_s'])
    target = new['main_s'] + 30600
    funcs = {'log_all_short': lambda z: log[0]*math.log(z),
             'log_recent_short': lambda z: (a[-1]-a[-2])/math.log(t[-1]/t[-2])*math.log(z),
             'power_short_no_floor': lambda z: 100-math.exp(power[1])*z**power[0],
             'log_two_50min_finals': lambda z: slope50*math.log(z)}
    forecasts = {k: float(accuracy(new)+f(target)-f(new['main_s'])) for k, f in funcs.items()}
    return {'target_main_s': target, 'L4_added_main_s': 30600, 'L4_accuracy_pp_scenarios': forecasts,
            'range_pp': [min(forecasts.values()), max(forecasts.values())],
            'same50_log_slope': slope50, 'short_tags': [p['tag'] for p in short],
            'policy': 'Scenarios anchored at measured L3 final, same 50min cooldown; not prediction CI. Plateau possible.'}


def submissions(model, point):
    test_path = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/test.json'
    assert l2.sha(test_path) == '20cacb334e3fc81291433d2173d9085b17a15e1834bd84aea67b350403182503'
    test = l2.read(test_path); generated = l2.read(path(model, 'evaluation__test-generations'))
    source = FINAL / f'model-{model}__points__{TAG}__evaluation__test-answers.jsonl'
    answers = [json.loads(s) for s in source.read_text().splitlines()]
    assert len(test) == len(generated) == len(answers) == 1919
    assert [r['question'] for r in generated] == [r['question'] for r in test]
    assert [r['id'] for r in generated] == [r['id'] for r in answers] == list(range(1919))
    assert [r['question'] for r in answers] == [r['question'] for r in test]
    assert all(isinstance(r['code'], str) for r in answers)
    assert [r['code'] for r in answers] == [r['generated_code'] for r in generated]
    assert l2.sha(source) == point['test']['answers_sha256']
    target = WORK / f'submissions/l3-{model.lower()}-greedy-{TAG}.jsonl'
    target.write_text(''.join(json.dumps({'code': r['code']}, ensure_ascii=False)+'\n' for r in answers))
    assert all(set(json.loads(s)) == {'code'} for s in target.read_text().splitlines())
    return {'file': str(target.relative_to(WORK)), 'rows': 1919, 'test_order_verified': True,
            'sha256': l2.sha(target), 'test_sha256': l2.sha(test_path)}


def main():
    previous = l2.read(WORK / 'results/l3-results.json')
    points = previous['points'].copy(); records = {}; gains = {}; diagnostics = {}; receipts = {}; outputs = {}
    for model in MODELS:
        rec, point, latest = load_final(model); records[model] = rec; points.append(point)
        old_rec, old = l2.load_point(model, 'l2-28800')
        for split in rec: mid.assert_aligned(rec[split], old_rec[split])
        gains[model] = mid.comparisons(old_rec, rec)
        before = l2.read(l2.RAW / f'model-{model}/summary.json')['state']
        summary = l2.read(FINAL / f'kernel-output/model-{model}/summary.json')
        state = summary['state']
        assert summary['status'] == 'complete' and len(state['l2_points']) == 5
        assert latest['identity'] == l2.read(l2.RAW / f'model-{model}/latest.json')['identity']
        assert latest['cursor']-before['cursor'] == 128*(latest['updates']-before['updates'])
        diagnostics[model] = {k: state[k] for k in ['AMP_skips', 'nonfinite_skips', 'micro_batch', 'last_lr']}
        diagnostics[model].update(stability={k: v for k, v in state['stability'].items() if k != 'window'},
            L3_AMP_skips=state['AMP_skips']-before['AMP_skips'],
            L3_updates=latest['updates']-old['updates'], L3_main_s=point['main_s']-old['main_s'],
            net_updates_per_main_s=(latest['updates']-old['updates'])/(point['main_s']-old['main_s']),
            cooldown_AMP_skips=point['training']['cooldown_AMP_skips'],
            cooldown_nonfinite_skips=point['training']['cooldown_nonfinite_skips'], restore_errors=point['restore_errors'],
            CE_change_vs_L2={s: point['CE'][s]-old['CE'][s] for s in point['CE']})
        receipts[model] = {'latest': latest, 'FP16': l2.read(path(model, 'checkpoint'))}
        outputs[model] = submissions(model, point)
    for split in records['C-main']: mid.assert_aligned(records['C-main'][split], records['C-long2x'][split])
    x, y = [vote_journal(m, records[m]['heldout4k']) for m in MODELS]
    for model, journal in zip(MODELS, [x, y]):
        diagnostics[model]['test_restore_errors'] = sum(bool(r['restore_error']) for r in
            l2.read(path(model, 'evaluation__test-generations')))
        diagnostics[model]['K8_sample_restore_errors'] = sum(bool(s['restore_error']) for r in journal.values()
            for s in r['samples']['0.7'])
    voting = l2.voting(x, y)
    singles = [r for r in voting['rows'] if not r['name'].startswith('mixed')]
    best = max(singles, key=lambda r: r['matches'])
    best_outcomes = outcomes(best['name'], x, y)
    for row in voting['rows']:
        row['against_best_single'] = l2.paired(best_outcomes, outcomes(row['name'], x, y))
    voting['best_single'] = best['name']
    curves = {m: curve(m, points) for m in MODELS}
    fig, ax = plt.subplots(figsize=(9, 5))
    for model, color in zip(MODELS, ['#2463ae', '#d26c24']):
        for short, marker, label in [(True, 'o', '15 мин'), (False, 's', '50 мин')]:
            selected = ([points[0]] if short else []) + [p for p in points if p['model'] == model and (p['cooldown_s'] < 1000) == short]
            ax.plot([p['main_s']/3600 for p in selected], [p['scores']['heldout4k']['accuracy_pp'] for p in selected],
                    marker=marker, color=color, linestyle='--' if short else '-', label=f'{model}, {label}')
        c = curves[model]; xx = c['target_main_s']/3600; lo, hi = c['range_pp']
        ax.plot([xx, xx], [lo, hi], color=color, linewidth=5, alpha=.5)
        ax.annotate(f'{model} L4: {lo:.2f}–{hi:.2f}%', (xx, hi), xytext=(-5, 10), textcoords='offset points', ha='right', color=color)
    ax.set(xlabel='Накопленное main-время, часы', ylabel='Greedy heldout4096, %',
           title='L3 финал и сценарии L4 (+8,5 ч main)', ylim=(62, 74), xlim=(1, 32))
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.text(.12, .01, 'L1 — общий предок. Прогноз — разброс fits, не CI; L4 предполагает тот же 50-минутный cooldown.', fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(WORK / 'results/l3-final-learning-curve.png', dpi=170); plt.close(fig)
    session = l2.read(FINAL / 'kernel-output/session-summary.json')
    assert session['status'] == 'complete' and session['exit_codes'] == [0, 0] and session['final_live_uploaded']
    log = l2.read(FINAL / 'kernel-output/nlp-hw02-kaggle-l3.log')
    log_text = '\n'.join(r['data'] for r in log)
    issues = {label: len(re.findall(pattern, log_text)) for label, pattern in {
        'worker_failed': 'WORKER FAILED', 'point_incomplete': 'L2 POINT INCOMPLETE',
        'tracebacks': r'Traceback \(most recent call last\)', 'CE_nan_inf': r'CE (?:nan|inf|NaN)',
        'positive_nonfinite_training_tail': r'NF [1-9][0-9]*'}.items()}
    nonfinite_paths = []
    def finite_tree(value, prefix):
        if isinstance(value, float) and not math.isfinite(value): nonfinite_paths.append(prefix)
        elif isinstance(value, dict):
            for k, v in value.items(): finite_tree(v, prefix+'/'+str(k))
        elif isinstance(value, list):
            for i, v in enumerate(value): finite_tree(v, prefix+'/'+str(i))
    for p in FINAL.rglob('*.json'):
        if p.name != 'next-inference.json': finite_tree(l2.read(p), str(p.relative_to(FINAL)))
    result = {'status': 'final_complete', 'points': points, 'paired_final_minus_L2_final': gains,
              'paired_final_long_minus_main': mid.comparisons(records['C-main'], records['C-long2x']),
              'voting_tune2048': voting, 'learning_curves': curves, 'diagnostics': diagnostics,
              'receipts': receipts, 'submissions': outputs,
              'session': {k: v for k, v in session.items() if k != 'comparison'},
              'log_issue_counts': issues, 'JSON_nonfinite_paths': nonfinite_paths,
              'audit': {'GPU': False, 'new_sampling': 0, 'model_forwards': 0, 'programs_reexecuted': 0,
                        'CI': 'paired normal 95%, unadjusted, descriptive repeated tune/heldout',
                        'local_SHA256': {str(p.relative_to(FINAL)): l2.sha(p) for p in sorted(FINAL.rglob('*'))
                                         if p.is_file() and p.name != 'next-inference.json' and '__checkpoints__' not in p.name},
                        'source_SHA256': l2.sha(Path(__file__))}}
    (WORK / 'results/l3-final-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('Final points', [(p['model'], p['scores']['heldout4k']) for p in points[-2:]])
    print('Gains', {m: g['heldout4k'] for m, g in gains.items()})
    for r in voting['rows']: print(r['name'], r['matches'], 'vs mainK8', r['against_C_main_K8'], 'vs best', r['against_best_single'])
    print('Curves', curves)


if __name__ == '__main__':
    main()
