"""CPU replay and conservative v6 timing from measured profile/v5 phases."""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FINAL = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final'
PROFILE = WORK/'results/kaggle-runs/nlp-hw02-decode-profile/2/decode-profile/summary.json'
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from final_v5_analysis import load_journals
from final_adaptive import replay_record, stage1_decision, stage1_leader_fraction, select_available, numeric, same
from final_vote import select as v3_select
from final_split import fixed_split


def read(path):
    return json.loads(path.read_text())


def timing_estimate(k):
    profile = read(PROFILE)
    def measurement(variant, mode):
        return next(r['seconds'] for r in profile['rows'] if r['scope'] == 'test1919' and
                    r['variant'] == variant and r['mode'] == mode and r['load_workers'])
    greedy_speedup = measurement('baseline', 'greedy')/measurement('all', 'greedy')
    measured_sample_speedup = measurement('baseline', 'K4')/measurement('all', 'K4')
    sample_speedup = min(1.6, measured_sample_speedup)
    measured = {}
    worst = 0.
    for label in ['test-1', 'tune']:
        path = FINAL/f'inference-v5/final-inference-l3-v5/{label}/timing.json'
        t = read(path); p = t['phases_seconds']
        s1 = sum(p[f'stage1_{name}_sampling_restore_submit'] for name in ['main', 'long'])
        n1 = sum(b['stage'] == 1 for b in t['generation_batches'])
        n2 = sum(b['stage'] == 2 for b in t['generation_batches'])
        proxy = p['stage2_sampling_restore_submit']/n2
        worst = max(worst, proxy, s1/n1)
        greedy = p['greedy_generation_restore_submit']
        fixed = t['full_pipeline_seconds']-s1-greedy-p['stage2_sampling_restore_submit']-p['stage1_decision_wait']
        second_batches = 2*math.ceil(len(t['adaptive']['escalated_rows'])/(1024//(k-2)))
        measured[label] = {'fixed_cold_load_restore_tail_seconds': fixed,
            'greedy_seconds': greedy/greedy_speedup, 'stage1_seconds': s1/sample_speedup,
            'stage2_batches_both_models': second_batches,
            'nominal_seconds': fixed+greedy/greedy_speedup+(s1+second_batches*s1/n1)/sample_speedup+2.,
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for r in measured.values():
        # Retain all unscaled fixed overhead (incl. real cold imports/load, execution,
        # vote/verification/write), round batch count, reserve 10% on worst stage2
        # proxy plus 2 seconds for the new global CPU barrier / worker expansion.
        r['conservative_seconds'] = (r['fixed_cold_load_restore_tail_seconds']+r['greedy_seconds']+
            r['stage1_seconds']+r['stage2_batches_both_models']*worst*1.10/sample_speedup+2.)
    bound = max(r['conservative_seconds'] for r in measured.values())
    return {'measured_v5_runs': measured, 'profile_sha256': hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
        'sample_speedup_assumed': sample_speedup, 'sample_speedup_measured': measured_sample_speedup,
        'greedy_speedup_loaded': greedy_speedup, 'stage2_worst_old_batch_proxy_seconds': worst,
        'conservative_seconds': bound, 'fits_150_with_reserve': bound <= 150.,
        'reserve': 'rounded batches; stage2 worst proxy +10%; fixed overhead unchanged; +2s CPU barrier/expansion',
        'margin_to_160_seconds': 160-bound, 'margin_to_180_seconds': 180-bound,
        'uncertainty': 'C-main K4 synthetic load profile extrapolated to K2/K6 and C-long; not a full v6 GPU timer',
        'GPU_measured_v6': False}


def analyze():
    models, receipts = load_journals()
    previous = read(WORK/'checks/final-inference-v5-offline.json')
    assert previous['journal_sha256'] == receipts
    result = []
    for k in [4, 6, 8]:
        winners = {}; records = {}; stopped = 0; matches = 0
        for i in fixed_split()['tune2048']:
            a, b = models['C-main'][i], models['C-long2x'][i]
            g, main, long = a['greedy'], a['samples']['0.7'], b['samples']['0.7']
            winner, stop, count = replay_record(g, main, long, k1=2, k2=k)
            stopped += stop
            winners[i] = winner
            matches += numeric(winner) and same(winner['execution']['value'], a['reference']['value'])
            records[i] = {'greedy': g, 'main': main[:2], 'long': long[:2]}
            if k == 4:
                old = v3_select(g, main, long)
                assert (winner['code'], winner['execution']) == (old['code'], old['execution'])
        priority = sorted((i for i, r in records.items() if not stage1_decision(r)),
                          key=lambda i: (stage1_leader_fraction(records[i]), i))
        for i in priority:
            records[i].update(main=models['C-main'][i]['samples']['0.7'][:k],
                              long=models['C-long2x'][i]['samples']['0.7'][:k])
        for i, r in records.items():
            actual, _ = select_available(r, k)
            assert (actual['code'], actual['execution']) == (winners[i]['code'], winners[i]['execution'])
        ordered = [{'id': i, 'code': w['code'], 'execution': w['execution']} for i, w in winners.items()]
        digest = hashlib.sha256(json.dumps(ordered, ensure_ascii=False).encode()).hexdigest()
        old = next(r for r in previous['rows'] if r['K2'] == k)
        assert digest == old['winner_code_execution_sha256']
        assert matches == {4: 1511, 6: 1533, 8: 1531}[k] and stopped == 1133
        result.append({'K2': k, 'matches': matches, 'stopped': stopped, 'n': 2048,
                       'v5_winner_code_execution_sha256': digest, 'priority_mismatches': 0,
                       'timing': timing_estimate(k)})
    # User's policy prioritizes K8 if its conservative estimate clears 150 seconds.
    selected = 8 if result[-1]['timing']['fits_150_with_reserve'] else 6
    assert next(r for r in result if r['K2'] == selected)['timing']['fits_150_with_reserve']
    checks = [r for r in read(PROFILE)['greedy_checks'] if r['variant'] == 'all' and r['rows'] == 1919]
    return {'passed': True, 'selected_K2': selected, 'rows': result, 'journal_sha256': receipts,
        'v3_v4_K4_and_v5_K6_K8_exact': True, 'stage1_stop_and_priority_equivalent': True,
        'known_FP16_shape_difference': checks, 'GPU': False, 'push': False}


if __name__ == '__main__':
    report = analyze()
    (WORK/'checks/final-inference-v6-offline.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'selected_K2': report['selected_K2'], 'rows': [
        {'K2': r['K2'], 'matches': r['matches'], 'seconds': r['timing']['conservative_seconds']} for r in report['rows']]}, indent=2))
