"""CPU-only preliminary L3 analysis from saved L2 and flat L3-mid journals.

No training, model forward, program execution, network or fresh evaluation.
Reuses the L2 correctness, fixed split and paired-CI protocol without rerunning
its main(). Writes only l3-results.json and l3-learning-curve.png.
"""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = Path(__file__).resolve().parent.parent
MID = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-l3/mid'
TAG = 'l2-45540'
MODELS = ['C-main', 'C-long2x']
spec = importlib.util.spec_from_file_location('l2_analysis', WORK / 'research/analyze-l2-results.py')
l2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l2)


def assert_aligned(a, b):
    fields = ['id', 'question_sha256', 'reference', 'original_bpe_length']
    assert [[r[f] for f in fields] for r in a] == [[r[f] for f in fields] for r in b]


def load_mid(model):
    def path(suffix):
        return MID / f'model-{model}__points__{TAG}__{suffix}.json'
    training = l2.read(path('training'))
    metrics = l2.read(path('evaluation__metrics'))
    complete = l2.read(path('complete'))
    checkpoint = l2.read(path('checkpoint'))
    latest = l2.read(MID / f'model-{model}__latest.json')
    assert complete['metrics'] == metrics
    assert all(complete['training'][k] == v for k, v in training.items())
    assert checkpoint['sha256'] == complete['weights_sha256'] == metrics['weights_sha256']
    assert not training['final'] and metrics['fresh_evaluated'] == 0
    assert 900 <= training['cooldown_seconds'] < 901
    for tr, la in [('anchor_updates', 'updates'), ('anchor_cursor', 'cursor'),
                   ('main_seconds', 'main_seconds'), ('l2_main_seconds', 'l2_main_seconds')]:
        assert training[tr] == latest[la]
    records = {}
    for split, n in [('heldout4k', 4096), ('old991', 991)]:
        rows = l2.read(path(f'evaluation__{split}-per-example'))
        assert len(rows) == n and [r['id'] for r in rows] == list(range(n))
        assert all(r['correct'] == l2.matches(r['reference'], r['execution']) for r in rows)
        assert [r['id'] for r in rows if r['correct']] == metrics['splits'][split]['exec']['match_indices']
        assert sum(r['correct'] for r in rows) == metrics['splits'][split]['exec']['matches']
        assert math.isfinite(metrics['splits'][split]['CE']['CE'])
        for label, lo, hi in [('<60', 0, 60), ('60–90', 60, 90),
                              ('90–120', 90, 120), ('≥120', 120, math.inf)]:
            selected = [r for r in rows if lo <= r['original_bpe_length'] < hi]
            b = metrics['splits'][split]['length_accuracy'][label]
            assert len(selected) == b['denominator']
            assert sum(r['correct'] for r in selected) == b['matches']
        records[split] = rows
    for split, ids in l2.SPLIT.items():
        assert l2.score(records['heldout4k'], ids)['matches'] == metrics['paired_subsets'][split]['matches']
    point = l2.point_row(model, TAG, training, metrics, records)
    point['training'] = training
    point['evaluation_s'] = complete['training']['evaluation_seconds']
    point['point_complete'] = True
    return records, point, latest


def comparisons(a, b):
    h = a['heldout4k']
    subsets = {'heldout4k': list(range(4096)), **l2.SPLIT,
               'ge90': [i for i, r in enumerate(h) if r['original_bpe_length'] >= 90]}
    for label, lo, hi in [('<60', 0, 60), ('60-90', 60, 90),
                          ('90-120', 90, 120), ('>=120', 120, math.inf)]:
        subsets['bin-' + label] = [i for i, r in enumerate(h) if lo <= r['original_bpe_length'] < hi]
    return {**{name: l2.paired([a['heldout4k'][i]['correct'] for i in ids],
                              [b['heldout4k'][i]['correct'] for i in ids])
               for name, ids in subsets.items()},
            'old991': l2.paired([r['correct'] for r in a['old991']], [r['correct'] for r in b['old991']])}


def curve(model, points, previous):
    short = [points[0]] + [p for p in points if p['model'] == model and p['cooldown_s'] < 1000]
    t = np.array([p['main_s'] for p in short])
    a = np.array([p['scores']['heldout4k']['accuracy_pp'] for p in short])
    log = np.polyfit(np.log(t), a, 1)
    recent = (a[-1] - a[-2]) / math.log(t[-1] / t[-2])
    power = np.polyfit(np.log(t), np.log(100 - a), 1)
    functions = {'log_all_short': lambda z: float(log[0]) * math.log(z),
                 'log_last_short': lambda z: float(recent) * math.log(z),
                 'power_error_no_floor': lambda z: 100 - math.exp(float(power[1])) * z ** float(power[0])}
    final_l2 = next(p for p in points if p['model'] == model and p['tag'] == 'l2-28800')
    base = final_l2['scores']['heldout4k']['accuracy_pp']
    historical_residuals = previous['learning_curves'][model]['final_residual_pp_not_causal_cooldown_estimate']
    forecasts = {}
    for phase, extra in [('L3-final', 33480), ('L4-plus-9h', 33480 + 32400)]:
        target = final_l2['main_s'] + extra
        short_scores = {k: float(a[-1] + f(target) - f(t[-1])) for k, f in functions.items()}
        same50 = {k: float(base + f(target) - f(final_l2['main_s'])) for k, f in functions.items()}
        residual_transfer = {k: short_scores[k] + historical_residuals[k] for k in functions}
        forecasts[phase] = {'main_s': target, 'short_cooldown_accuracy_pp': short_scores,
                            'same_50min_L2_anchor_accuracy_pp': same50,
                            'old_L2_residual_transfer_accuracy_pp': residual_transfer,
                            'same_50min_range_pp': [min(same50.values()), max(same50.values())],
                            'sensitivity_envelope_pp': [min(*same50.values(), *residual_transfer.values()),
                                                        max(*same50.values(), *residual_transfer.values())]}
    return {'fit_tags': [p['tag'] for p in short], 'main_s': t.tolist(), 'accuracy_pp': a.tolist(),
            'log_all_slope_pp_per_ln_time': float(log[0]), 'log_recent_slope_pp_per_ln_time': float(recent),
            'power_error_exponent': float(power[0]), 'forecasts': forecasts,
            'policy': 'Fit four short-cooldown points only. Forecast increments anchored at L2 50min final; '
                      'sensitivity transfers the old L2 noncausal residual to the mid-anchored short curve. '
                      'Descriptive scenarios, no statistical prediction CI; same recipe and final cooldown assumed.'}


def draw(curves, points):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, model, color in zip(axes, MODELS, ['#2463ae', '#d26c24']):
        c = curves[model]
        ax.plot(np.array(c['main_s']) / 1000, c['accuracy_pp'], 'o-', color=color, label='Cooldown ≈15 мин')
        old = next(p for p in points if p['model'] == model and p['tag'] == 'l2-28800')
        ax.scatter(old['main_s'] / 1000, old['scores']['heldout4k']['accuracy_pp'],
                   s=70, facecolors='none', edgecolors=color, label='L2: cooldown 50 мин')
        x, low, high = [], [], []
        for phase, f in c['forecasts'].items():
            xx = f['main_s'] / 1000
            lo, hi = f['same_50min_range_pp']
            ax.plot([xx, xx], [lo, hi], color=color, linewidth=4, alpha=.65)
            ax.scatter(xx, (lo + hi) / 2, marker='D', color=color, s=35)
            ax.annotate('L3 final' if phase == 'L3-final' else 'L4 +9ч',
                        (xx, hi), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
            x.append(xx); low.append(lo); high.append(hi)
        grid = np.linspace(c['main_s'][-1], c['forecasts']['L4-plus-9h']['main_s'], 100)
        short = c['accuracy_pp'][-1] + c['log_all_slope_pp_per_ln_time'] * np.log(grid / c['main_s'][-1])
        ax.plot(grid / 1000, short, ':', color=color, label='Короткий cooldown: log fit')
        ax.set(title=model, xlabel='Накопленное main_s, тыс. секунд', xlim=(5, 120), ylim=(61, 76))
        ax.set_xticks([9, 28.37, 46.37, 63.11, 79.85, 112.25])
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Greedy heldout4096, %')
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('L3 mid: измерения и условные прогнозы финалов', fontsize=13)
    fig.text(.06, .025, 'Ромбы и полосы: разброс трёх fits при привязке к L2 final (50 мин). Это сценарии, не CI.\n'
             'L1 — общий предок; точка L2 с 50 мин cooldown исключена из fit; L4 предполагает ещё 9 ч main и 50 мин cooldown.', fontsize=8)
    fig.tight_layout(rect=(0, .1, 1, .94))
    fig.savefig(WORK / 'results/l3-learning-curve.png', dpi=180)
    plt.close(fig)


def main():
    previous = l2.read(WORK / 'results/l2-results.json')
    points = list(previous['points'])
    records, diagnostics, receipts, gains = {}, {}, {}, {}
    audit = {'GPU_used': False, 'model_forwards': 0, 'training_updates': 0,
             'programs_reexecuted': 0, 'fresh_read_or_evaluated': False,
             'CI': previous['audit']['CI'], 'local_SHA256': {},
             'weights': 'JSON SHA receipts consistent; FP16/full PT absent, not read or independently hashed',
             'source_SHA256': {'analyze-l2-results.py': l2.sha(WORK / 'research/analyze-l2-results.py'),
                              'analyze-l3-results.py': l2.sha(Path(__file__))}}
    for p in sorted(MID.iterdir()):
        assert p.is_file() and p.suffix in {'.json', '.txt'}
        audit['local_SHA256'][p.name] = l2.sha(p)
    for model in MODELS:
        rec, point, latest = load_mid(model)
        records[model] = rec; points.append(point)
        old_rec, old = l2.load_point(model, 'l2-21600')
        final_rec, final = l2.load_point(model, 'l2-28800')
        for split in rec:
            assert_aligned(rec[split], old_rec[split]); assert_aligned(rec[split], final_rec[split])
        prior_latest = l2.read(l2.RAW / f'model-{model}/latest.json')
        assert latest['identity'] == prior_latest['identity']
        assert latest['model_spec'] == prior_latest['model_spec']
        assert latest['cursor'] - prior_latest['cursor'] == 128 * (latest['updates'] - prior_latest['updates'])
        gains[model] = comparisons(old_rec, rec)
        rate_old = (final['updates'] - old['updates']) / (final['main_s'] - old['main_s'])
        rate_new = (point['updates'] - final['updates']) / (point['main_s'] - final['main_s'])
        diagnostics[model] = {'L3_net_attempted_updates': point['updates'] - final['updates'],
                              'L3_main_s': point['main_s'] - final['main_s'],
                              'L2_last_segment_updates_per_main_s': rate_old,
                              'L3_net_updates_per_main_s': rate_new,
                              'rate_change_pct': 100 * (rate_new / rate_old - 1),
                              'L3_nominal_pairs_per_main_s': 128 * rate_new,
                              'cooldown_AMP_skips': point['training']['cooldown_AMP_skips'],
                              'cooldown_nonfinite_skips': point['training']['cooldown_nonfinite_skips'],
                              'CE_change_vs_l2_21600': {s: point['CE'][s] - old['CE'][s] for s in point['CE']},
                              'CE_change_vs_l2_28800_different_cooldown': {s: point['CE'][s] - final['CE'][s] for s in point['CE']},
                              'main_skips_rollbacks_LR_microbatch': 'Not present in downloaded mid JSON',
                              'restore_errors': point['restore_errors'],
                              'test_execution_s': point['test']['execution']['seconds'],
                              'test_generation_s': point['test']['generation']['seconds']}
        receipts[model] = {'FP16': l2.read(MID / f'model-{model}__points__{TAG}__checkpoint.json'),
                           'latest': {k: latest[k] for k in ['file', 'sha256', 'bytes', 'updates', 'cursor']}}
    for split in records['C-main']:
        assert_aligned(records['C-main'][split], records['C-long2x'][split])
    pairs = comparisons(records['C-main'], records['C-long2x'])
    union = sum(a['correct'] or b['correct'] for a, b in zip(records['C-main']['heldout4k'], records['C-long2x']['heldout4k']))
    mixed = next(r for r in previous['voting_tune2048']['rows'] if r['name'] == 'mixed K4+K4 +greedy')
    main_greedy = previous['voting_tune2048']['baseline_C_main_greedy_matches']
    long_greedy = previous['voting_tune2048']['baseline_C_long_greedy_matches']
    voting = {'mid_sampling_available': False, 'L2_mixed_K4_K4_greedy_matches': mixed['matches'],
              'n': 2048, 'L2_accuracy_pp': mixed['accuracy_pp'],
              'L2_vs_main_greedy': mixed['against_C_main_greedy'],
              'L2_vs_long_greedy_delta_pp': 100 * (mixed['matches'] - long_greedy) / 2048,
              'L2_vs_mean_two_greedy_delta_pp': 100 * (mixed['matches'] - (main_greedy + long_greedy) / 2) / 2048,
              'L2_vs_main_K8': mixed['against_C_main_K8'],
              'mid_greedy_union_oracle': {'matches': union, 'accuracy_pp': 100 * union / 4096,
                                         'selectable_without_gold': False},
              'transfer_scenario': '+2 to +3 pp over a future single greedy; tune-only history, not measured L3, not a CI'}
    curves = {model: curve(model, points, previous) for model in MODELS}
    draw(curves, points)
    result = {'status': 'preliminary_mid_only', 'kernel_ref': 'friend1/nlp-hw02-kaggle-l3',
              'kernel_status_downloaded': (MID / 'kernel-status.txt').read_text().strip(),
              'audit': audit, 'points': points, 'mid_receipts': receipts,
              'paired_long_minus_main_mid': pairs, 'paired_mid_minus_l2_21600': gains,
              'learning_curves': curves, 'voting': voting, 'diagnostics': diagnostics,
              'missing_for_main_stability_audit': ['model-C-main/progress.json', 'model-C-long2x/progress.json',
                                                  'model-C-main/summary.json', 'model-C-long2x/summary.json',
                                                  'model-C-main/stability-history.json (if created)',
                                                  'model-C-long2x/stability-history.json (if created)',
                                                  'L3 worker/stdout logs with updates/main_s/CE/LR/NF']}
    (WORK / 'results/l3-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print('Verified 10174 mid rows; wrote l3-results.json and l3-learning-curve.png')
    print('Long-main:', json.dumps(pairs['heldout4k']))
    for model in MODELS:
        print(model, 'gain:', json.dumps(gains[model]['heldout4k']))
        print(model, 'forecast:', json.dumps(curves[model]['forecasts']))


if __name__ == '__main__':
    main()
