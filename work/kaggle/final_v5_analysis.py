"""L3 journal replay through production v5 decisions; CPU only, no resampling."""
import argparse
import ast
import hashlib
import gzip
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FINAL = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final'
sys.path.insert(0, str(ROOT/'src'))
from final_adaptive import replay_record, select, select_available, stage1_decision, stage1_leader_fraction, numeric, same
from final_vote import select as v3_select
from final_split import fixed_split


def read(path):
    return json.loads(path.read_text())


def load_journals():
    models, receipts = {}, {}
    manifest = {r['original_path']: r for r in read(FINAL/'recover-manifest.json')['files']}
    for name in ['C-main', 'C-long2x']:
        files = sorted(FINAL.glob(f'model-{name}__points__l2-62280__evaluation__k8-tune2048__batch-*.json'))
        assert len(files) == 64
        rows = []
        hashes = {}
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entry = manifest[path.name.replace('__', '/')]
            payload = FINAL/entry['file']
            if payload.exists():
                assert hashlib.sha256(payload.read_bytes()).hexdigest() == entry['sha256']
                assert payload.stat().st_size == entry['bytes']
                assert path.read_bytes() == (gzip.decompress(payload.read_bytes()) if entry['gzip'] else payload.read_bytes())
            else:
                assert entry['gzip'], 'Missing uncompressed manifest payload'
                # l2_live producer: level1/mtime0; Linux/macOS differ in OS byte.
                packed = bytearray(gzip.compress(path.read_bytes(), compresslevel=1, mtime=0))
                assert len(packed) == entry['bytes']
                verified = False
                for os_byte in (3, 255):
                    packed[9] = os_byte
                    verified |= hashlib.sha256(packed).hexdigest() == entry['sha256']
                assert verified, 'Recovered journal does not reproduce live manifest SHA'
            hashes[path.name] = digest
            rows.extend(read(path))
        models[name] = {r['id']: r for r in rows}
        assert len(rows) == len(models[name]) == 2048
        assert sorted(models[name]) == fixed_split()['tune2048']
        assert all(len(r['samples']['0.7']) == 8 for r in rows)
        receipts[name] = hashes
    return models, receipts


def simulation_functions(path):
    source = Path(path).read_text()
    tree = ast.parse(source)
    names = {'isnum', 'same', 'clusters', 'select'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = {'math': math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'user simulation functions', 'exec'), namespace)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def timing_estimate(escalation_fraction, k2):
    """Batch estimate calibrated on actual v5 T4 test+tune, not sample-volume scaling.

    Stage two was interrupted: phase time / completed batches is an upper proxy,
    not a measured individual batch latency. Add 10% plus 2s for the global CPU
    barrier. Priority sorting changes padding/RNG, so this remains a hypothesis.
    """
    runs = {}
    upper_batch_seconds = 0.
    for label in ['test-1', 'tune']:
        path = FINAL/f'inference-v5/final-inference-l3-v5/{label}/timing.json'
        timing = read(path)
        phases = timing['phases_seconds']
        batches = timing['generation_batches']
        first = [r for r in batches if r['stage'] == 1]
        second = [r for r in batches if r['stage'] == 2]
        assert first and second and not timing['OOM_attempts']
        assert all(r['questions'] <= 1024//2 for r in first)
        assert all(r['sequences'] <= 1024 for r in batches)
        first_seconds = sum(phases[f'stage1_{name}_sampling_restore_submit'] for name in ['main', 'long'])
        second_upper = phases['stage2_sampling_restore_submit']/len(second)
        upper_batch_seconds = max(upper_batch_seconds, second_upper, first_seconds/len(first))
        # Keep all measured cold overhead/greedy/stage1/finalization. CPU service
        # wall overlaps generation and must never be added to these phases.
        fixed = timing['full_pipeline_seconds']-phases['stage2_sampling_restore_submit']-phases['stage1_decision_wait']
        tail = sum(phases[key] for key in ['execution_tail_wait', 'executor_shutdown', 'vote',
                                           'winner_reexecution', 'answer_write'])
        unresolved = len(timing['adaptive']['escalated_rows'])
        question_batch = 1024//(k2-2)
        second_batches = math.ceil(unresolved/question_batch)
        nominal = fixed+2*second_batches*first_seconds/len(first)
        runs[label] = {'rows': timing['rows'], 'escalated_rows': unresolved,
                      'stage1_batches_per_model': math.ceil(timing['rows']/512),
                      'stage2_question_batch': question_batch, 'stage2_batches_per_model': second_batches,
                      'stage1_mean_batch_seconds': first_seconds/len(first),
                      'stage2_phase_per_completed_batch_upper_seconds': second_upper,
                      'fixed_seconds_including_stage1_and_tail': fixed, 'observed_tail_seconds': tail,
                      'nominal_seconds': nominal, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for run in runs.values():
        run['batch_bound_with_reserve_seconds'] = (run['fixed_seconds_including_stage1_and_tail']+
            2*run['stage2_batches_per_model']*upper_batch_seconds*1.10+2.)
        run['generation_bound_seconds'] = run['batch_bound_with_reserve_seconds']-run['observed_tail_seconds']
    conservative = max(r['batch_bound_with_reserve_seconds'] for r in runs.values())
    generation = max(r['generation_bound_seconds'] for r in runs.values())
    return {'measured_v5_runs': runs, 'sequence_budget': 1024,
            'escalation_fraction_proxy_from_journals': escalation_fraction,
            'stage2_batch_seconds_upper_proxy': upper_batch_seconds,
            'nominal_seconds': runs['test-1']['nominal_seconds'],
            'batch_bound_with_reserve_seconds': conservative,
            'generation_bound_seconds': generation, 'generation_deadline_seconds': 160.,
            'reserve_model': 'measured fixed + rounded stage2 batches * worst batch proxy * 1.10 + 2s CPU barrier',
            'stage1_batches_per_model': runs['test-1']['stage1_batches_per_model'],
            'stage2_batches_per_model': runs['test-1']['stage2_batches_per_model'],
            'margin_to_180_seconds': 180-conservative,
            'fits_15_second_margin': conservative <= 165. and generation <= 160.,
            'GPU_measured': False}


def analyze(simulation_script):
    models, receipts = load_journals()
    sim, sim_sha = simulation_functions(simulation_script)
    rows = []
    old_mismatches = []
    for k2 in [4, 5, 6, 7, 8]:
        matches = sim_matches = full_matches = stops = sample_count = 0
        differences, winners, full_diff = [], [], []
        scheduling_records = {}
        for i in fixed_split()['tune2048']:
            a, b = models['C-main'][i], models['C-long2x'][i]
            assert a['reference'] == b['reference'] and a['question_sha256'] == b['question_sha256']
            g, main, long = a['greedy'], a['samples']['0.7'], b['samples']['0.7']
            win, stop, count = replay_record(g, main, long, k1=2, k2=k2)
            scheduling_records[i] = {'greedy': g, 'main': main[:2], 'long': long[:2]}
            stops += stop
            sample_count += count
            used = 2 if stop else k2
            candidates = [g]+main[:used]+long[:used]
            simulated = sim['select']([{'src': 'g' if j == 0 else 'a', 'ex': c['execution'],
                                      'lp': c['mean_logprob'], 'code': c['code'], 'cov': 1.}
                                     for j, c in enumerate(candidates)], {})
            first = [g]+main[:2]+long[:2]
            sim_stop = all(sim['isnum'](c['execution']) for c in first) and all(
                sim['same'](c['execution']['value'], g['execution']['value']) for c in first)
            assert stop == sim_stop
            ref = a['reference']
            matches += numeric(win) and same(win['execution']['value'], ref['value'])
            sim_matches += simulated is not None and sim['same'](simulated, ref['value'])
            if simulated is not None and (not numeric(win) or not same(simulated, win['execution']['value'])):
                differences.append(i)
            full = select(g, main, long, k2)
            full_matches += numeric(full) and same(full['execution']['value'], ref['value'])
            if win['execution'] != full['execution']:
                full_diff.append(i)
            winners.append({'id': i, 'code': win['code'], 'execution': win['execution']})
            if k2 == 4:
                old = v3_select(g, main, long)
                if old['execution'] != win['execution'] or old['code'] != win['code']:
                    old_mismatches.append(i)
        # Replay the live global priority schedule while preserving each row's
        # main-before-long vote order; all answers/codes must remain bit-exact.
        escalated = [i for i, r in scheduling_records.items() if not stage1_decision(r)]
        priority = sorted(escalated, key=lambda i: (stage1_leader_fraction(scheduling_records[i]), i))
        for i in priority:
            scheduling_records[i].update(main=models['C-main'][i]['samples']['0.7'][:k2],
                                         long=models['C-long2x'][i]['samples']['0.7'][:k2])
        schedule_mismatches = []
        for expected in winners:
            actual, _ = select_available(scheduling_records[expected['id']], k2)
            if actual['code'] != expected['code'] or actual['execution'] != expected['execution']:
                schedule_mismatches.append(expected['id'])
        assert not schedule_mismatches
        rows.append({'K1': 2, 'K2': k2, 'matches': matches, 'accuracy_pp': 100*matches/2048,
                     'simulation_matches': sim_matches, 'simulation_accuracy_pp': 100*sim_matches/2048,
                     'simulation_value_difference_ids': differences,
                     'full_vote_matches': full_matches, 'adaptive_vs_full_execution_difference_ids': full_diff,
                     'priority_schedule_code_execution_mismatch_ids': schedule_mismatches,
                     'stopped': stops, 'stop_fraction': stops/2048, 'mean_samples': sample_count/2048,
                     'winner_code_execution_sha256': hashlib.sha256(json.dumps(winners, ensure_ascii=False).encode()).hexdigest(),
                     'timing': timing_estimate(1-stops/2048, k2)})
    assert not old_mismatches and rows[0]['matches'] == 1511
    k8 = rows[-1]
    assert k8['simulation_matches'] == 1530 and k8['matches'] == 1531 and k8['stopped'] == 1133
    assert k8['mean_samples'] == 9.361328125
    selected = max((r for r in rows if r['timing']['fits_15_second_margin']),
                   key=lambda r: (r['matches'], -r['K2']))
    return {'passed': True, 'rows': rows, 'selected_K2': selected['K2'], 'journal_sha256': receipts,
            'simulation_script_sha256': sim_sha, 'stage1_stop_equivalence': True,
            'v3_v4_K4_code_execution_mismatches': old_mismatches,
            'exact_7471_with_frozen_vote': False,
            'reason': 'User simulation ties: count/greedy/order. Frozen v3/v4: count/greedy/mean LP/order.',
            'GPU': False, 'push': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation-script', required=True)
    args = parser.parse_args()
    result = analyze(args.simulation_script)
    (WORK/'checks/final-inference-v5-offline.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': result['passed'], 'selected_K2': result['selected_K2'],
                      'rows': [{k: r[k] for k in ['K2', 'matches', 'simulation_matches', 'stopped', 'mean_samples']}
                               for r in result['rows']]}, indent=2))
