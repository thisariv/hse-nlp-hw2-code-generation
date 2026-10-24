"""CPU audit of R1 output, paired scores, frozen K8 votes and logged train CE.

No model forwards, new execution/sampling, API calls, or fresh-set access.
Run with NumPy and Matplotlib; raw Kaggle output remains ignored by Git.
"""
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = Path(__file__).resolve().parents[1]
RAW = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-r1/1'
L3 = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-l3/final'
P1 = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-p1/final'
spec = importlib.util.spec_from_file_location('l2_analysis', WORK / 'research/analyze-l2-results.py')
l2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l2)
sys.path.insert(0, str(WORK / 'kaggle/src'))
from final_ensemble import replay_record
from final_adaptive import select

BINS = [('<60', 0, 60), ('60–90', 60, 90), ('90–120', 90, 120), ('≥120', 120, math.inf)]
LOCATIONS = {
    'deep': (RAW / 'model-GPU0/points/final', None),
    'control': (RAW / 'model-GPU1/points/final', None),
    'L3-long': (L3, 'model-C-long2x__points__l2-62280'),
    'L3-main': (L3, 'model-C-main__points__l2-62280'),
    'P1-A': (P1, 'model-pilot-a__points__long'),
    'P1-B': (P1, 'model-pilot-b__points__long-truncation'),
}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def file(name, suffix):
    folder, prefix = LOCATIONS[name]
    return folder / (suffix.replace('__', '/') + '.json' if prefix is None else prefix + '__' + suffix + '.json')


def paired(a, b):
    """Variant b minus baseline a; exact paired row-bootstrap distribution."""
    result = l2.paired(a, b)
    n = result['n']; gains = result['gains']; losses = result['losses']
    draws = np.random.default_rng(2026101501).multinomial(
        n, [losses / n, (n - gains - losses) / n, gains / n], size=10000)
    deltas = 100 * (draws[:, 2] - draws[:, 0]) / n
    result['CI95_pp'] = np.quantile(deltas, [.025, .975]).tolist()
    result['method'] = 'paired row bootstrap, multinomial equivalent; 10000 draws, seed2026101501'
    return result


def subsets(rows):
    return {'heldout4k': list(range(4096)), **l2.SPLIT,
            **{'bin-' + label: [r['id'] for r in rows if lo <= r['original_bpe_length'] < hi]
               for label, lo, hi in BINS}}


def load(name):
    training, metrics, complete, checkpoint = [l2.read(file(name, s)) for s in
        ['training', 'evaluation__metrics', 'complete', 'checkpoint']]
    assert complete['metrics'] == metrics
    assert complete['weights_sha256'] == metrics['weights_sha256'] == checkpoint['sha256']
    rows = {}
    for split, n in [('heldout4k', 4096), ('old991', 991)]:
        rows[split] = l2.read(file(name, 'evaluation__' + split + '-per-example'))
        assert [r['id'] for r in rows[split]] == list(range(n))
        assert all(r['correct'] == l2.matches(r['reference'], r['execution']) for r in rows[split])
        assert sum(r['correct'] for r in rows[split]) == metrics['splits'][split]['exec']['matches']
        assert math.isfinite(metrics['splits'][split]['CE']['CE'])
    scores = {s: l2.score(rows['heldout4k'], ids) for s, ids in subsets(rows['heldout4k']).items()}
    scores['old991'] = l2.score(rows['old991'], range(991))
    for split in l2.SPLIT:
        assert scores[split]['matches'] == metrics['paired_subsets'][split]['matches']
    for label, _, _ in BINS:
        b = metrics['splits']['heldout4k']['length_accuracy'][label]
        assert (b['denominator'], b['matches']) == (scores['bin-' + label]['n'], scores['bin-' + label]['matches'])
    if name in ('deep', 'control'):
        stable = training['stable']
        model = RAW / ('model-GPU0' if name == 'deep' else 'model-GPU1')
        latest = l2.read(model / 'latest.json')
        summary = l2.read(model / 'summary.json')
        assert latest == stable and complete['training'] == training
        assert summary['status'] == 'complete' and summary['stable_preserved']
        assert summary['stable_updates'] == stable['updates']
        assert summary['stable_cursor'] == stable['cursor']
        assert stable['updates'] - 209144 == training['main_updates']
        assert stable['cursor'] - 20734592 == 128 * training['main_updates']
        assert stable['model_spec']['encoder_layers'] == (12 if name == 'deep' else 6)
        branch = stable['identity']['recipe']['r1']
        assert branch['q'] == .1 and branch['data'] == 'ordinary'
        assert stable['identity']['order_sha256'] == '72bc3dfd3a3e738979c489c0cf8f41cde092c41d4ae90f2de8378a034603a4db'
        main_s, updates = stable['main_seconds'], stable['updates']
        added_s = main_s - 79850.42887455746
        rate = training['main_updates'] / training['main_wall_seconds']
        cooldown_s = training['cooldown_wall_seconds']
        train_info = dict(main_wall_s=training['main_wall_seconds'], added_main_s=added_s,
                          added_main_updates=training['main_updates'], updates_per_wall_s=rate,
                          presented_pairs_per_wall_s=128 * rate, retained_pairs_per_wall_s=115 * rate,
                          updates_per_update_timer_s=training['main_updates'] / added_s,
                          peak_reserved_bytes=summary['peak_reserved_bytes'])
    else:
        main_s, updates = training.get('main_seconds', 79850.42887455746), training['anchor_updates']
        cooldown_s = training['cooldown_wall_seconds']
        train_info = {}
    # P1 adds only disposable cooldown, no main steps or main seconds.
    if name.startswith('P1-'):
        main_s, updates = 79850.42887455746, training['anchor_updates']
        train_info = dict(added_main_s=0, added_main_updates=0,
                          cooldown_updates_per_wall_s=training['cooldown_updates'] / cooldown_s,
                          measured_cooldown_updates_per_update_timer_s=training['throughput_updates_per_second'])
    if name == 'L3-long':
        previous = next(p for p in l2.read(WORK / 'results/l3-final-results.json')['points']
                        if p['model'] == 'C-long2x' and p['tag'] == 'l2-28800')
        added_s = main_s - previous['main_s']; added = updates - previous['updates']
        train_info = dict(added_main_s=added_s, added_main_updates=added,
                          updates_per_update_timer_s=added / added_s,
                          main_wall_s=None, reason='L3 main_s sums update timers; no comparable main-wall receipt')
    generated = l2.read(file(name, 'evaluation__test-generations'))
    assert len(generated) == 1919 and [r['id'] for r in generated] == list(range(1919))
    token_lengths = [len(r['generated_token_ids']) for r in generated]
    answer_path = file(name, 'evaluation__test-answers').with_suffix('.jsonl')
    answers = [json.loads(line) for line in answer_path.read_text().splitlines()]
    assert len(answers) == 1919 and [r['code'] for r in answers] == [r['generated_code'] for r in generated]
    assert l2.sha(answer_path) == metrics['test']['answers_sha256']
    test_audit = dict(rows=1919, generated_token_count_including_mode_EOS=sum(token_lengths),
                      mean_token_length=float(np.mean(token_lengths)), max_token_length=max(token_lengths),
                      no_EOS=sum(not r['eos_emitted'] for r in generated),
                      restore_errors=sum(bool(r['restore_error']) for r in generated), answers_SHA_verified=True)
    return rows, dict(scores=scores, main_s=main_s, updates=updates, cooldown_wall_s=cooldown_s,
                      cooldown_updates=training['cooldown_updates'], training=train_info,
                      CE={s: metrics['splits'][s]['CE']['CE'] for s in rows},
                      test=metrics['test'], test_generation_audit=test_audit, K8=metrics['K8_tune2048'], checkpoint=checkpoint,
                      restore_errors={s: metrics['splits'][s]['restore_errors'] for s in rows})


def aligned(a, b):
    for split in a:
        for field in ['id', 'question_sha256', 'reference', 'original_bpe_length']:
            assert [r[field] for r in a[split]] == [r[field] for r in b[split]]


def comparisons(a, b):
    aligned(a, b)
    result = {s: paired([a['heldout4k'][i]['correct'] for i in ids],
                        [b['heldout4k'][i]['correct'] for i in ids])
              for s, ids in subsets(a['heldout4k']).items()}
    result['old991'] = paired([r['correct'] for r in a['old991']], [r['correct'] for r in b['old991']])
    result['gain_ids'] = [i for i, (x, y) in enumerate(zip(a['heldout4k'], b['heldout4k'])) if not x['correct'] and y['correct']]
    result['loss_ids'] = [i for i, (x, y) in enumerate(zip(a['heldout4k'], b['heldout4k'])) if x['correct'] and not y['correct']]
    return result


def journal(name, greedy):
    folder, prefix = LOCATIONS[name]
    paths = sorted((folder / 'evaluation/k8-tune2048').glob('batch-*.json')) if prefix is None else sorted(
        folder.glob(prefix + '__evaluation__k8-tune2048__batch-*.json'))
    assert len(paths) == 64
    rows = [r for p in paths for r in l2.read(p)]
    assert [r['id'] for r in rows] == l2.SPLIT['tune2048']
    for r in rows:
        g = greedy[r['id']]
        assert r['question_sha256'] == g['question_sha256'] and r['reference'] == g['reference']
        assert r['greedy']['execution'] == g['execution']
        samples = r['samples']['0.7']
        assert len(samples) == 8 and [s['sample'] for s in samples] == list(range(8))
        assert all(s['sampling_status'] == 'complete' and math.isfinite(s['mean_logprob']) for s in samples)
        v = l2.choose(r, dict(mode='vote', temperature=.7, k=8, greedy_vote=False))
        assert v['execution'] == r['vote_execution'] and l2.matches(r['reference'], v['execution']) == r['correct']
    summary = l2.read(file(name, 'evaluation__k8-tune2048__summary'))
    assert summary['complete'] and summary['rows'] == summary['target_rows'] == 2048 and not summary['OOM_attempts']
    assert sum(r['correct'] for r in rows) == summary['matches']
    return rows, {str(p.relative_to(WORK)): l2.sha(p) for p in paths}


def voting(journals):
    results, outcomes = {}, {}
    def run(label, names, ks, greedy_vote=True, adaptive=False):
        correct, counts, stops, oracle = [], [], [], []
        for j in range(2048):
            base = journals['L3-main'][j] if len(names) > 1 else journals[names[0]][j]
            g = base['greedy']
            pool = [c for name, k in zip(names, ks) for c in journals[name][j]['samples']['0.7'][:k]]
            if adaptive:
                models = [dict(name=name, K1=1, K2=k) for name, k in zip(names, ks)]
                win, stop, count = replay_record(g, {name: journals[name][j]['samples']['0.7'] for name in names}, models)
                used = [c for name, k in zip(names, [1] * len(names) if stop else ks)
                        for c in journals[name][j]['samples']['0.7'][:k]]
            else:
                win = l2.choose({'greedy': g, 'samples': {'0.7': pool}},
                                dict(mode='vote', temperature=.7, k=len(pool), greedy_vote=greedy_vote))
                if greedy_vote:
                    production = select(g, pool, [], k=len(pool))
                    assert l2.matches(base['reference'], win['execution']) == l2.matches(base['reference'], production['execution'])
                stop, count, used = False, len(pool), pool
            correct.append(l2.matches(base['reference'], win['execution']))
            oracle.append(any(l2.matches(base['reference'], c['execution']) for c in ([g] if greedy_vote else []) + used))
            stops.append(stop); counts.append(count)
        outcomes[label] = correct
        results[label] = dict(models=names, K2=ks, adaptive_K1=[1] * len(names) if adaptive else None,
                              greedy_vote=greedy_vote, matches=sum(correct), n=2048, accuracy_pp=100 * np.mean(correct),
                              max_samples=sum(ks), mean_samples=float(np.mean(counts)), stopped=sum(stops),
                              oracle_matches=sum(oracle), missed_oracle=sum(o and not c for o, c in zip(oracle, correct)))

    run('v7 frozen', ['L3-main', 'L3-long', 'P1-B'], [3, 5, 4], adaptive=True)
    assert results['v7 frozen']['matches'] == 1535
    assert results['v7 frozen']['stopped'] == 1232 and results['v7 frozen']['mean_samples'] == 6.5859375
    for name in ['deep', 'control', 'L3-long', 'P1-A', 'P1-B']:
        for g in [False, True]:
            run(name + ' single8' + ('+g' if g else ''), [name], [8], g)
    for name in ['deep', 'control', 'L3-long']:
        for k in [4, 8]:
            for g in [False, True]:
                run('main+' + name + f' {k}+{k}' + ('+g' if g else ''), ['L3-main', name], [k, k], g)
    for label, names in [
        ('main+long+deep', ['L3-main', 'L3-long', 'deep']),
        ('main+control+deep', ['L3-main', 'control', 'deep']),
        ('main+long+control', ['L3-main', 'L3-long', 'control']),
        ('main+B+deep', ['L3-main', 'P1-B', 'deep']),
        ('main+B+control', ['L3-main', 'P1-B', 'control'])]:
        for ks in [[4, 4, 4], [8, 8, 8], [3, 5, 4]]:
            run(label + ' ' + '+'.join(map(str, ks)) + '+g', names, ks)
        run(label + ' adaptive1→3/5/4', names, [3, 5, 4], adaptive=True)
    for label, r in results.items():
        r['paired_vs_v7'] = paired(outcomes['v7 frozen'], outcomes[label])
        r['correctness_bits_tune_order'] = ''.join('1' if b else '0' for b in outcomes[label])
    results['deep single8']['paired_vs_control_single8'] = paired(outcomes['control single8'], outcomes['deep single8'])
    return results


def curves():
    kernel = l2.read(RAW / 'nlp-hw02-kaggle-r1.log')
    pattern = re.compile(r'R1 model-GPU([01]) (main|cooldown) (\d+) CE ([0-9.eE+-]+)')
    observed = {}
    for entry in kernel:
        for gpu, phase, update, ce in pattern.findall(entry['data']):
            observed.setdefault((gpu, phase, int(update)), (float(entry['time']) / 3600, float(ce)))
    result = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    common = {}
    for gpu, name, color in [('0', 'deep', '#d36d27'), ('1', 'control', '#2463ae')]:
        process = (RAW / f'gpu{gpu}-process.log').read_text()
        rows = [(phase, int(update), float(ce)) for g, phase, update, ce in pattern.findall(process) if g == gpu]
        main = [(update, *observed[(gpu, phase, update)]) for phase, update, ce in rows if phase == 'main']
        # Supervisor timestamps bound the first observation; polling is 30 seconds.
        assert len(main) == len({r[0] for r in main})
        xx = np.array([r[1] for r in main]); yy = np.array([r[2] for r in main])
        assert np.all(np.diff(xx) > 0)
        bins = []
        for lo in np.arange(0, 10, .5):
            mask = (xx >= lo) & (xx < lo + .5)
            # Exclude the tiny final partial bin (3/4 batches), which otherwise
            # creates a spurious late slope from a few differently sized rows.
            if mask.sum() >= 20:
                bins.append(dict(hour=float(xx[mask].mean()), n=int(mask.sum()), median=float(np.median(yy[mask])),
                                 q25=float(np.quantile(yy[mask], .25)), q75=float(np.quantile(yy[mask], .75))))
        slopes = {}
        for hours in [2, 4]:
            tail = [b for b in bins if b['hour'] >= xx.max() - hours]
            slopes[f'last{hours}h_median_CE_per_hour'] = float(np.polyfit([b['hour'] for b in tail], [b['median'] for b in tail], 1)[0])
        axes[0].scatter(xx, yy, s=4, alpha=.11, color=color)
        axes[0].plot([b['hour'] for b in bins], [b['median'] for b in bins], 'o-', ms=3, color=color, label=name)
        axes[0].fill_between([b['hour'] for b in bins], [b['q25'] for b in bins], [b['q75'] for b in bins], color=color, alpha=.1)
        common[name] = {r[0]: r[2] for r in main}
        update_bins = []
        for lo in range(0, 90000, 5000):
            chosen = [r[2] for r in main if lo < r[0] <= lo + 5000]
            if chosen:
                update_bins.append((lo + 2500, float(np.median(chosen))))
        axes[1].plot([a / 1000 for a, b in update_bins], [b for a, b in update_bins], 'o-', ms=3, color=color, label=name)
        result[name] = dict(periodic_heldout_CE_points=0, train_main_log_points=len(main), half_hour_bins=bins,
                            median_bin_minimum_logged_batches=20,
                            **slopes, timing='first kernel-supervisor observation from session start, polling about30s',
                            metric='retained-token training CE of one batch every100 updates, not validation CE')
    common_ids = sorted(common['deep'].keys() & common['control'].keys())
    diffs = [common['deep'][i] - common['control'][i] for i in common_ids]
    result['same_update_stream'] = dict(n=len(common_ids), max_update=max(common_ids),
                                       mean_deep_minus_control_CE=float(np.mean(diffs)),
                                       median_deep_minus_control_CE=float(np.median(diffs)))
    axes[0].set(xlabel='Часы с начала сессии, первое наблюдение', ylabel='Train CE по оставленным токенам', title='R1: batch CE, медианы по 30 минут')
    axes[1].set(xlabel='Новые attempted main updates, тысячи', ylabel='Train CE по оставленным токенам', title='Общий префикс order, медианы по 5000 updates')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(); ax.set_ylim(.13, .28)
    fig.savefig(WORK / 'results/r1-train-ce.png', dpi=170)
    plt.close(fig)
    return result


def truncation():
    result = {}; prefixes = []
    for gpu, name in [('0', 'deep'), ('1', 'control')]:
        phases = {}; prefix = []
        with (RAW / f'model-GPU{gpu}/truncation.jsonl').open() as handle:
            for line in handle:
                row = json.loads(line)
                r = phases.setdefault(row['phase'], dict(updates=0, presented=0, excluded=0,
                    presented_tokens=0, retained_tokens=0, graph_updates=0, OOM_fallbacks=0))
                assert row['presented'] == 128 and row['retained'] == 115 and row['requested_q'] == .1
                r['updates'] += 1; r['presented'] += 128; r['excluded'] += 13
                for k in ['presented_tokens', 'retained_tokens']: r[k] += row[k]
                r['graph_updates'] += row['forward_mode'] == 'graph'
                r['OOM_fallbacks'] += row['graph_OOM_fallback']
                if row['phase'] == 'main':
                    prefix.append((row['update'], row['counts'], row['presented_tokens']))
        tr = l2.read(file(name, 'training'))
        assert phases['main']['updates'] == tr['main_updates']
        assert phases['cooldown']['updates'] == tr['cooldown_updates']
        result[name] = phases; prefixes.append(prefix)
    assert prefixes[0] == prefixes[1][:len(prefixes[0])]
    result['shared_main_prefix_presented_length_token_checks'] = len(prefixes[0])
    return result


def main():
    session = l2.read(RAW / 'session-summary.json')
    assert session['status'] == 'complete' and session['exit_codes'] == [0, 0]
    assert 'COMPLETE' in (RAW / 'kernel-status.txt').read_text()
    manifest = l2.read(RAW / 'r1-output-manifest.json')
    missing, bad, hashes = [], [], {}
    for r in manifest['files']:
        p = RAW / r['file']
        if not p.is_file(): missing.append(r['file']); continue
        digest = l2.sha(p); hashes[r['file']] = digest
        if p.stat().st_size != r['bytes'] or digest != r['sha256']: bad.append(r['file'])
    # Kaggle's executed notebook was listed by the producer but not downloaded.
    # It is not an input to metric/vote/curve calculations.
    assert set(missing) <= {'__notebook__.ipynb'} and not bad, (missing, bad)
    source_parity = {}
    for name in ['r1_runtime.py', 'r1_growth.py', 'p1_math.py', 'exp27_decode.py', 'l2_sampler.py']:
        a, b = RAW / 'src' / name, WORK / 'kaggle/src' / name
        assert a.read_bytes() == b.read_bytes()
        source_parity[name] = l2.sha(a)
    records, models, journals, journal_hashes = {}, {}, {}, {}
    for name in LOCATIONS:
        records[name], models[name] = load(name)
        if name != 'deep': aligned(records['deep'], records[name])
        journals[name], journal_hashes[name] = journal(name, records[name]['heldout4k'])
    pairs = {label: comparisons(records[a], records[b]) for label, a, b in [
        ('deep-control', 'control', 'deep'), ('control-L3', 'L3-long', 'control'), ('deep-L3', 'L3-long', 'deep')]}
    report = dict(status='complete', session=session, manifest=dict(entries=len(hashes), missing=missing, bad=bad, sha256=hashes),
                  models=models, paired=pairs, voting_tune2048=voting(journals), curves=curves(), truncation=truncation(),
                  audit=dict(GPU=False, model_forwards=0, new_sampling=0, program_reexecution=0, fresh_access=0,
                             producer_source_parity=source_parity,
                             K8_batches=384, K8_samples=98304, aligned_greedy_rows=30522,
                             journal_sha256=journal_hashes, source_sha256=l2.sha(Path(__file__)),
                             CI='paired bootstrap95%, unadjusted; repeated heldout and tune; no training-seed variability'))
    write(WORK / 'results/r1-results.json', report)
    print('manifest', report['manifest']['entries'], 'missing', missing, 'bad', bad)
    print('models', {n: (m['scores']['heldout4k']['matches'], m['scores']['tune2048']['matches'], m['scores']['eval2048']['matches']) for n, m in models.items()})
    print('paired', {n: p['heldout4k'] for n, p in pairs.items()})
    for label, r in report['voting_tune2048'].items():
        print(label, r['matches'], 'samples', r['mean_samples'], 'vs v7', r['paired_vs_v7']['delta_pp'], r['paired_vs_v7']['CI95_pp'])
    print('slopes', {n: {k: v for k, v in r.items() if 'slope' in k or 'per_hour' in k or k == 'train_main_log_points'} for n, r in report['curves'].items()})


if __name__ == '__main__':
    main()
