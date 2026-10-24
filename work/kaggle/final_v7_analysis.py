"""K8 tune replay, independent frozen vote, paired CI and batch-based T4 forecast.

No new generation, GPU, or API calls. K selection on tune is exploratory; the
pointwise paired CI is not adjusted for selection among configurations.
"""
import gzip
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from final_v5_analysis import load_journals, simulation_functions
from final_v6_analysis import FINAL, PROFILE, read
from final_adaptive import replay_record as v6_replay, numeric, same
from final_ensemble import DEFAULT_MODELS, parameters, replay_record, stage1_decision, stage1_leader_fraction, select_available
from final_split import fixed_split

P1 = WORK/'results/kaggle-runs/nlp-hw02-kaggle-p1/final'


def load_all():
    models, receipts = load_journals()
    manifest = {r['original_path']: r for r in read(P1/'recover-manifest.json')['files']}
    rows, hashes = [], {}
    paths = sorted(P1.glob('model-pilot-b__points__long-truncation__evaluation__k8-tune2048__batch-*.json'))
    assert len(paths) == 64
    for path in paths:
        raw = path.read_bytes()
        entry = manifest[path.name.replace('__', '/')]
        packed_path = P1/entry['file']
        if packed_path.exists():
            packed = packed_path.read_bytes()
            assert len(packed) == entry['bytes'] and hashlib.sha256(packed).hexdigest() == entry['sha256']
            assert raw == (gzip.decompress(packed) if entry['gzip'] else packed)
        else:
            packed = bytearray(gzip.compress(raw, compresslevel=1, mtime=0))
            assert entry['gzip'] and len(packed) == entry['bytes']
            digests = []
            for os_byte in (3, 255):
                packed[9] = os_byte
                digests.append(hashlib.sha256(packed).hexdigest())
            assert entry['sha256'] in digests
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        rows.extend(json.loads(raw))
    models['pilot-b'] = {r['id']: r for r in rows}
    receipts['pilot-b'] = hashes
    assert len(rows) == len(models['pilot-b']) == 2048
    assert sorted(models['pilot-b']) == fixed_split()['tune2048']
    for i in fixed_split()['tune2048']:
        for name in models:
            assert len(models[name][i]['samples']['0.7']) == 8
            assert models[name][i]['reference'] == models['C-main'][i]['reference']
            assert models[name][i]['question_sha256'] == models['C-main'][i]['question_sha256']
    return models, receipts


def specs(k1, k2):
    return parameters({'models': [dict(m, K1=a, K2=b) for m, a, b in zip(DEFAULT_MODELS, k1, k2)]})[0]


@lru_cache(maxsize=1)
def question_lengths():
    from final_representation import tokenizer
    inputs = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    heldout = read(inputs/'heldout4k.json')
    tok, _ = tokenizer(inputs/'tokenizer.json', 30)
    return {i: len(tok.encode(heldout[i]['question'], add_special_tokens=False).ids)+2 for i in fixed_split()['tune2048']}


def independent_select(greedy, samples):
    """Separate union-find oracle: count/greedy/meanLP/order, then code policy."""
    samples = [c for c in samples if c.get('mean_logprob') is not None and math.isfinite(c['mean_logprob'])]
    good = [c for c in [greedy]+samples if numeric(c)]
    if not good:
        return greedy
    parent = list(range(len(good)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, c in enumerate(good):
        for j in range(i):
            if same(c['execution']['value'], good[j]['execution']['value']):
                parent[root(i)] = root(j)
    components = {}
    for i, c in enumerate(good):
        components.setdefault(root(i), []).append(c)
    component = max(components.values(), key=lambda group: (
        len(group), any(c is greedy for c in group), sum(c['mean_logprob'] for c in group)/len(group)))
    return greedy if any(c is greedy for c in component) else max(component, key=lambda c: c['mean_logprob'])


def paired(correct, baseline):
    differences = [int(a)-int(b) for a, b in zip(correct, baseline)]
    n = len(differences); mean = sum(differences)/n
    se = math.sqrt(sum((d-mean)**2 for d in differences)/(n-1)/n)
    return {'delta_pp': 100*mean, 'CI95_pp': [100*(mean-1.96*se), 100*(mean+1.96*se)],
            'gained': differences.count(1), 'lost': differences.count(-1), 'n': n,
            'method': 'paired normal CI of per-question correctness differences; pointwise, tune-selected'}


def timing_estimate(model_specs, stopped):
    """Same conservative proxy as v6, with actual N/K-dependent waves and cold load.

    Stage two uses a shared priority wave (lanes / largest added K). All model
    calls are priced as full batches even when a smaller extra underfills lanes.
    Test disagreement uses the measured test:v6-tune ratio, not its gold labels.
    """
    profile = read(PROFILE)
    def measurement(variant, mode):
        return next(r['seconds'] for r in profile['rows'] if r['scope'] == 'test1919' and
                    r['variant'] == variant and r['mode'] == mode and r['load_workers'])
    gs = measurement('baseline', 'greedy')/measurement('all', 'greedy')
    ss = min(1.6, measurement('baseline', 'K4')/measurement('all', 'K4'))
    extra = [m['K2']-m['K1'] for m in model_specs]
    wave = 1024//max(extra) if max(extra) else 0
    old_runs = {label: read(FINAL/f'inference-v5/final-inference-l3-v5/{label}/timing.json')
                for label in ['test-1', 'tune']}
    worst = max(max(t['phases_seconds']['stage2_sampling_restore_submit']/sum(b['stage'] == 2 for b in t['generation_batches']),
                    sum(t['phases_seconds'][f'stage1_{n}_sampling_restore_submit'] for n in ['main', 'long'])/
                    sum(b['stage'] == 1 for b in t['generation_batches'])) for t in old_runs.values())
    runs = {}
    for label, t in old_runs.items():
        p = t['phases_seconds']; n = t['rows']
        first = sum(p[f'stage1_{name}_sampling_restore_submit'] for name in ['main', 'long'])
        mean_batch = first/sum(b['stage'] == 1 for b in t['generation_batches'])
        unresolved = (2048-stopped) if label == 'tune' else math.ceil(
            len(t['adaptive']['escalated_rows'])*(2048-stopped)/915)
        unresolved = min(n, unresolved)
        first_batches = sum(math.ceil(n/(1024//m['K1'])) for m in model_specs)
        second_batches = math.ceil(unresolved/wave)*sum(k > 0 for k in extra) if wave else 0
        fixed = t['full_pipeline_seconds']-first-p['greedy_generation_restore_submit']-p['stage2_sampling_restore_submit']-p['stage1_decision_wait']
        # Deliberately also retain tokenizer/encode in this per-extra-model proxy.
        third_load = max(0, len(model_specs)-2)*(p['load_models_tokenizer_encode']+p['warmup'])/2
        greedy = p['greedy_generation_restore_submit']/gs
        nominal = fixed+third_load+greedy+(first_batches+second_batches)*mean_batch/ss+2.
        conservative = fixed+third_load+greedy+first_batches*mean_batch/ss+second_batches*worst*1.10/ss+2.
        runs[label] = {'rows': n, 'escalated_rows_proxy': unresolved, 'stage1_model_batches': first_batches,
                       'stage2_model_batches': second_batches, 'stage2_wave_questions': wave,
                       'fixed_seconds': fixed, 'extra_model_load_warmup_reserve_seconds': third_load,
                       'nominal_seconds': nominal, 'conservative_seconds': conservative,
                       'source_sha256': hashlib.sha256((FINAL/f'inference-v5/final-inference-l3-v5/{label}/timing.json').read_bytes()).hexdigest()}
    bound = max(r['conservative_seconds'] for r in runs.values())
    return {'runs': runs, 'conservative_seconds': bound, 'fits_140': bound <= 140.,
            'nominal_test_seconds': runs['test-1']['nominal_seconds'], 'sample_speedup': ss,
            'worst_old_batch_seconds': worst, 'profile_sha256': hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
            'GPU_measured_v7': False, 'margin_to_guard_seconds': 160-bound,
            'limitations': 'C-main K4 profile extrapolated to K1/extras/pilot-b; test disagreements projected from tune; not a GPU measurement'}


def winners_for(journals, model_specs):
    ordered, correct, stopped, counts = [], [], [], []
    records = {}
    for i in fixed_split()['tune2048']:
        g = journals['C-main'][i]['greedy']
        samples = {m['name']: journals[m['name']][i]['samples']['0.7'] for m in model_specs}
        winner, stop, count = replay_record(g, samples, model_specs)
        first = [g]+[c for m in model_specs for c in samples[m['name']][:m['K1']]]
        independent_stop = all(numeric(c) for c in first) and all(same(c['execution']['value'], g['execution']['value']) for c in first)
        assert stop == independent_stop
        flattened = [c for m in model_specs for c in samples[m['name']][:m['K1' if stop else 'K2']]]
        oracle = independent_select(g, flattened)
        assert (winner['code'], winner['execution']) == (oracle['code'], oracle['execution'])
        records[i] = {'greedy': g, 'samples': {m['name']: samples[m['name']][:m['K1']] for m in model_specs}}
        ordered.append({'id': i, 'code': winner['code'], 'execution': winner['execution']})
        ref = journals['C-main'][i]['reference']
        correct.append(numeric(winner) and numeric({'execution': ref}) and same(winner['execution']['value'], ref['value']))
        stopped.append(stop); counts.append(count)
    # Replay the complete live priority wave schedule and compare exact code/execution.
    lengths = question_lengths()
    priority = sorted((i for i, r in records.items() if not stage1_decision(r, model_specs)),
                      key=lambda i: (stage1_leader_fraction(records[i], model_specs), lengths[i], i))
    max_extra = max(m['K2']-m['K1'] for m in model_specs)
    wave = 1024//max_extra if max_extra else 0
    for offset in range(0, len(priority), wave) if wave else []:
        for m in model_specs:
            for i in priority[offset:offset+wave]:
                records[i]['samples'][m['name']] = journals[m['name']][i]['samples']['0.7'][:m['K2']]
    for expected in ordered:
        actual, _ = select_available(records[expected['id']], model_specs)
        assert (actual['code'], actual['execution']) == (expected['code'], expected['execution'])
    return ordered, correct, stopped, counts


def analyze(simulation_script):
    journals, receipts = load_all()
    ids = fixed_split()['tune2048']
    base_winners = [v6_replay(journals['C-main'][i]['greedy'], journals['C-main'][i]['samples']['0.7'],
                            journals['C-long2x'][i]['samples']['0.7'], k1=2, k2=6) for i in ids]
    baseline = [numeric(w) and same(w['execution']['value'], journals['C-main'][i]['reference']['value'])
                for i, (w, _, _) in zip(ids, base_winners)]
    assert sum(baseline) == 1533
    base_mean = sum(c for _, _, c in base_winners)/2048
    # Small declared candidate grid; max12 samples, with different stage-one evidence.
    variants = [((1, 1, 1), k) for k in [(3, 3, 3), (4, 4, 3), (4, 3, 4), (3, 4, 4),
        (4, 4, 4), (5, 4, 3), (5, 3, 4), (4, 5, 3), (3, 5, 4), (4, 3, 5), (3, 4, 5), (6, 3, 3)]]
    variants += [(k, (4, 4, 4)) for k in [(2, 1, 1), (1, 2, 1), (1, 1, 2), (2, 2, 1), (2, 1, 2), (1, 2, 2), (2, 2, 2)]]
    rows = []
    for k1, k2 in variants:
        model_specs = specs(k1, k2)
        winners, correct, stopped, counts = winners_for(journals, model_specs)
        rows.append({'K1': list(k1), 'K2': list(k2), 'matches': sum(correct), 'accuracy_pp': 100*sum(correct)/2048,
                     'n': 2048, 'stopped': sum(stopped), 'mean_samples': sum(counts)/2048,
                     'max_samples': sum(k2), 'paired_vs_v6': paired(correct, baseline),
                     'winner_code_execution_sha256': hashlib.sha256(json.dumps(winners, ensure_ascii=False).encode()).hexdigest(),
                     'independent_vote_and_priority_mismatches': 0,
                     'timing': timing_estimate(model_specs, sum(stopped))})
    eligible = [r for r in rows if r['max_samples'] <= 12 and r['mean_samples'] <= base_mean and r['timing']['fits_140']]
    selected = max(eligible, key=lambda r: (r['matches'], -r['timing']['conservative_seconds'], -r['mean_samples']))
    # Reproduce user's full655 result separately; it exceeds the requested v7 budget.
    sim, sim_sha = simulation_functions(simulation_script)
    full_counts = {}
    for ks in [(6, 5, 5), (8, 8, 0)]:
        frozen_matches = sim_matches = 0
        for i in ids:
            g = journals['C-main'][i]['greedy']
            samples = [c for m, k in zip(DEFAULT_MODELS, ks) for c in journals[m['name']][i]['samples']['0.7'][:k]]
            win = independent_select(g, samples)
            ref = journals['C-main'][i]['reference']['value']
            frozen_matches += numeric(win) and same(win['execution']['value'], ref)
            value = sim['select']([{'src': 'g' if j == 0 else 'a', 'ex': c['execution'],
                'lp': c['mean_logprob'], 'code': c['code'], 'cov': 1.} for j, c in enumerate([g]+samples)], {})
            sim_matches += value is not None and same(value, ref)
        full_counts[str(list(ks))] = {'frozen_matches': frozen_matches, 'user_simulation_matches': sim_matches}
    assert full_counts['[6, 5, 5]']['user_simulation_matches'] == 1538
    assert full_counts['[8, 8, 0]']['user_simulation_matches'] == 1530
    return {'passed': True, 'selected': selected, 'rows': rows, 'journal_sha256': receipts,
            'baseline_v6': {'K1': [2, 2], 'K2': [6, 6], 'matches': 1533, 'mean_samples': base_mean},
            'full_pool_user_comparison': full_counts, 'simulation_script_sha256': sim_sha,
            'selection': 'best tune matches among declared variants with max12, mean<=v6, conservative<=140; ties faster',
            'CI_limit': 'pointwise paired95% CI on the same tune set used for K selection; no heldout confirmation',
            'GPU': False, 'push': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('simulation_script', type=Path, help='User selection_analysis.py (SHA recorded in receipt)')
    result = analyze(parser.parse_args().simulation_script)
    (WORK/'checks/final-inference-v7-offline.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'selected': result['selected'], 'rows': [{k: r[k] for k in
          ['K1', 'K2', 'matches', 'stopped', 'mean_samples']} | {'seconds': r['timing']['conservative_seconds']} for r in result['rows']],
          'full_pool': result['full_pool_user_comparison']}, indent=2))
