"""CPU-only B7 audit and six fixed v11-policy replays on saved tune K8.

Run with NumPy/Matplotlib (inherited B6 helpers). No tensors, forwards,
sampling, program execution, network or changes to frozen B6/E1/E2 outputs.
All six policies use K1=1/1/1 -> K2=1/8/8, greedy first model.
Grouped splits and policy order are fixed here, before running the analysis.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('b6_analysis', HERE/'analyze-b6-results.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
B, M, np = S.B, S.M, S.np
WORK, RUNS, IDS = S.WORK, S.RUNS, S.IDS
RAW = RUNS/'nlp-hw02-b7/1'
NAMES = ['C-long2x-B6', 'pilot-b', 'W-B6', 'C-long2x-B7', 'W-B7']
PAIRS = [('W-B7', 'W-B6'), ('C-long2x-B7', 'C-long2x-B6'),
         ('W-B7', 'pilot-b'), ('C-long2x-B7', 'pilot-b'), ('W-B7', 'C-long2x-B7')]
POLICIES = [
    ('v11', ['C-long2x-B6', 'pilot-b', 'W-B6']),
    ('replace-C', ['C-long2x-B7', 'pilot-b', 'W-B6']),
    ('replace-W', ['C-long2x-B6', 'pilot-b', 'W-B7']),
    ('replace-both', ['C-long2x-B7', 'pilot-b', 'W-B7']),
    ('replace-P-with-C6', ['C-long2x-B7', 'C-long2x-B6', 'W-B7']),
    ('replace-P-with-W6', ['C-long2x-B7', 'W-B6', 'W-B7']),
]
INPUTS, RECORDS = {}, {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bitstring(bits):
    """Position j corresponds to IDS[j]; keep row outcomes compact in Git."""
    return ''.join('1' if b else '0' for b in bits)


def read(path):
    INPUTS[str(path.relative_to(WORK))] = digest(path)
    return json.loads(path.read_text())


def paired(candidate, baseline):
    """Always candidate minus baseline; CI resamples paired task outcomes."""
    out = M.paired(candidate, baseline, boot=False)
    n, g, l = out['n'], out['gained'], out['lost']
    draws = np.random.default_rng(2026102007).multinomial(
        n, [g/n, l/n, 1-(g+l)/n], size=20000)
    out['bootstrap_CI95_pp'] = np.quantile(100*(draws[:, 0]-draws[:, 1])/n, [.025, .975]).tolist()
    out.update(matches=int(np.sum(candidate)), baseline_matches=int(np.sum(baseline)))
    return out


def getters():
    old = S.getters()
    result = {n: old[n] for n in NAMES[:3]}
    for gpu, name in [(0, 'W-B7'), (1, 'C-long2x-B7')]:
        root = RAW/f'model-GPU{gpu}/points/b6-final'
        result[name] = lambda suffix, root=root: root/suffix
    return result


def audit():
    session = read(RAW/'session-summary.json')
    assert 'COMPLETE' in (RAW/'kernel-status.txt').read_text()
    assert session['status'] == 'complete' and session['exit_codes'] == [0, 0]
    assert not session['unreaped_worker_pids'] and session['wall_seconds'] < 40800
    checked, missing = [], []
    manifest = read(RAW/'b1-output-manifest.json')
    for entry in manifest['files']:
        path = RAW/entry['file']
        assert path.resolve().is_relative_to(RAW.resolve())
        if not path.is_file():
            missing.append(entry)
            continue
        assert path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'], path
        INPUTS[str(path.relative_to(WORK))] = entry['sha256']
        checked.append(entry['file'])
    config = read(RAW/'session-config.json')
    assert config['gpu_smoke'] is False and config['owner'] == 'friend1'
    assert config['N_by_gpu'] == {'0': 33000, '1': 65800}
    old_config = read(RUNS/'nlp-hw02-b6/1/session-config.json')
    for key in ['b6_pins', 'port_sha256', 'anchors', 'dropout', 'final_q']:
        assert config[key] == old_config[key], key
    identity = read(WORK/'checks/b7-review-gpu-smoke.json')
    sources = {n: (WORK/'kaggle/src'/n if (WORK/'kaggle/src'/n).exists() else WORK/'kaggle'/n).read_text()
               for n in identity['modules']}
    overrides = {n: (WORK/'kaggle/legacy-c'/n).read_text() for n in identity['C_overrides']}
    assert config['port_sha256'] == hashlib.sha256(json.dumps(
        dict(SOURCES=sources, C_OVERRIDES=overrides), sort_keys=True).encode()).hexdigest()
    stability = {}
    for gpu, name in [(0, 'W-B7'), (1, 'C-long2x-B7')]:
        root = RAW/f'model-GPU{gpu}'
        s = read(root/'summary.json'); cd = s['cooldown']
        anchor = config['b6_pins']['anchors'][str(gpu)]
        old = read(RUNS/f'nlp-hw02-b6/1/model-GPU{gpu}/summary.json')
        assert s['status'] == 'complete' and s['final_complete'] and s['main_added_updates'] == 0
        assert cd['N'] == cd['updates'] == config['N_by_gpu'][str(gpu)] == 2*old['cooldown']['N']
        assert cd['source_sha256'] == old['cooldown']['source_sha256'] == anchor['sha256']
        assert cd['peak_lr'] == old['cooldown']['peak_lr'] and cd['schedule'] == old['cooldown']['schedule']
        assert cd['q'] == cd['last_lr'] == 0 and s['main_state'] == old['main_state']
        assert cd['origin_updates'] == anchor['updates'] and cd['origin_cursor'] == anchor['cursor']
        assert cd['cursor'] == anchor['cursor']+128*cd['N']
        assert read(root/'start-fingerprints.json')['source_sha256'] == anchor['sha256']
        assert read(root/'latest.json') == s['latest']
        assert cd['nonfinite_skips'] == cd['stability']['total'] == cd['stability']['escalations'] == 0
        log = (RAW/f'gpu{gpu}-process.log').read_text()
        assert not re.search(r'Traceback|WORKER FAILED|POINT INCOMPLETE|\b(?:NaN|nan|inf)\b', log)
        training = read(root/'points/b6-final/training.json')
        assert training['cooldown'] == cd and training['completed_updates'] == cd['N']
        assert training['endpoint_lr'] == training['main_added_updates'] == training['q'] == 0
        # Compact download omits the two step journals. Do not claim step-level audit.
        step_path = root/'cooldown-updates.jsonl'
        step_checked = 0
        if step_path.is_file():
            rows = [json.loads(line) for line in step_path.read_bytes().splitlines()]
            assert len(rows) == cd['N']
            elapsed = 0.
            for j, x in enumerate(rows, 1):
                assert x['branch_update'] == j and x['cursor'] == anchor['cursor']+128*j
                assert x['lr'] == cd['peak_lr']*(1-math.sqrt(j/cd['N'])) and x['pairs'] == 128
                assert 'truncation' not in x and not x['OOM_retries'] and x['nonfinite_skip'] == 0
                assert math.isfinite(x['CE']) and (x['AMP_skip'] or math.isfinite(x['gradient_norm']))
                elapsed += x['seconds']; assert elapsed == x['cooldown_seconds']
            assert elapsed == cd['seconds'] and sum(x['AMP_skip'] for x in rows) == cd['AMP_skips']
            assert rows[-1]['stream_sha256'] == cd['stream_sha256']
            step_checked = len(rows)
        stability[name] = dict(N=cd['N'], seconds=cd['seconds'], pairs=128*cd['N'], AMP=cd['AMP_skips'],
                               nonfinite=0, endpoint_lr=0, source_sha256=cd['source_sha256'],
                               midpoint=s['midpoint'], step_rows_checked=step_checked)
    models, refs = {}, None
    for name, getter in getters().items():
        metrics = read(getter('evaluation/metrics.json')); ck = read(getter('checkpoint.json'))
        complete = read(getter('complete.json'))
        assert metrics == complete['metrics'] and metrics['weights_sha256'] == ck['sha256'] == complete['weights_sha256']
        rr = {}
        for split, n in [('heldout4k', 4096), ('old991', 991)]:
            rows = read(getter(f'evaluation/{split}-per-example.json'))
            assert [x['id'] for x in rows] == list(range(n))
            for i, x in enumerate(rows):
                assert x['correct'] == B.correct(x['reference'], x['execution'])
                if refs:
                    assert all(x[k] == refs[split][i][k] for k in ['id', 'question_sha256', 'reference', 'original_bpe_length'])
            assert sum(x['correct'] for x in rows) == metrics['splits'][split]['exec']['matches']
            rr[split] = rows
        if refs is None: refs = rr
        RECORDS[name] = rr
        groups = dict(heldout4k=list(range(4096)), tune2048=IDS,
                      eval2048=[i for i in range(4096) if i not in set(IDS)])
        groups.update({f'bin{lo}-{hi}': [x['id'] for x in rr['heldout4k'] if lo <= x['original_bpe_length'] < hi]
                       for lo, hi in S.BINS})
        scores = {key: B.scores([rr['heldout4k'][i] for i in ids]) for key, ids in groups.items()}
        scores['old991'] = B.scores(rr['old991'])
        for key in ['tune2048', 'eval2048']:
            assert scores[key]['matches'] == metrics['paired_subsets'][key]['matches']
        k = metrics['K8_tune2048']
        assert k['complete'] and k['rows'] == k['target_rows'] == 2048 and k['K'] == 8
        assert k['T'] == .7 and k['seed'] == 270042 and not k['OOM_attempts']
        assert k['weights_sha256'] == ck['sha256'] and metrics['test']['rows'] == 1919
        models[name] = dict(scores=scores, CE={s: metrics['splits'][s]['CE']['CE'] for s in rr},
                            checkpoint=ck, K8=k, greedy_test_seconds=metrics['test']['full_pipeline_seconds'])
    comparisons = {}
    for a, b in PAIRS:
        out = {}
        for split in ['heldout4k', 'old991']:
            groups = {'all': list(range(len(RECORDS[a][split])))}
            if split == 'heldout4k':
                groups.update(groups_for_heldout(RECORDS[a][split]))
            for key, ids in groups.items():
                out[f'{split}/{key}'] = paired(*[[RECORDS[n][split][i]['correct'] for i in ids] for n in [a, b]])
        out['CE_delta'] = {s: models[a]['CE'][s]-models[b]['CE'][s] for s in ['heldout4k', 'old991']}
        comparisons[f'{a} minus {b}'] = out
    return dict(session=session, producer_checked=checked, producer_missing=missing,
                stability=stability, models=models, comparisons=comparisons)


def groups_for_heldout(rows):
    groups = dict(tune2048=IDS, eval2048=[i for i in range(4096) if i not in set(IDS)])
    groups.update({f'bin{lo}-{hi}': [x['id'] for x in rows if lo <= x['original_bpe_length'] < hi]
                   for lo, hi in S.BINS})
    return groups


def ensembles(result):
    journals, receipts, singles = {}, {}, []
    for name, getter in getters().items():
        pattern = getter('evaluation/k8-tune2048/batch-*.json')
        files = sorted(pattern.parent.glob(pattern.name))
        assert len(files) == 64
        rows = [r for f in files for r in read(f)]
        receipts.update({str(f.relative_to(WORK)): digest(f) for f in files})
        assert [r['id'] for r in rows] == IDS
        journals[name] = {r['id']: r for r in rows}
        for row in rows:
            ref = RECORDS[name]['heldout4k'][row['id']]
            assert row['question_sha256'] == ref['question_sha256'] and row['reference'] == ref['reference']
            assert row['greedy']['execution'] == ref['execution'] and len(row['samples']['0.7']) == 8
            for sample in row['samples']['0.7']:
                assert sample['sampling_status'] == 'complete' and math.isfinite(sample['mean_logprob'])
            win = B.choose(row, dict(mode='vote', temperature=.7, k=8, greedy_vote=False))
            assert B.correct(row['reference'], win['execution']) == row['correct']
        sim = B.Replay(journals, name, [name])
        for greedy in [False, True]:
            bits, _, winners = sim.evaluate([8], greedy=greedy)
            for row, bit, winner in zip(rows, bits, winners):
                actual = B.choose(row, dict(mode='vote', temperature=.7, k=8, greedy_vote=greedy))
                assert (winner['code'], winner['execution']) == (actual['code'], actual['execution'])
                assert bit == B.correct(row['reference'], actual['execution'])
                if not greedy: assert bit == row['correct']
            if not greedy: assert sum(bits) == result['models'][name]['K8']['matches']
            singles.append(dict(model=name, greedy_vote=greedy, matches=sum(bits),
                                correct_bits=bitstring(bits)))
    single_comparisons = {}
    for a, b in PAIRS:
        for greedy in [False, True]:
            vectors = [[c == '1' for c in next(x['correct_bits'] for x in singles if x['model'] == n and x['greedy_vote'] == greedy)]
                       for n in [a, b]]
            single_comparisons[f'{a} minus {b}/greedy={greedy}'] = paired(*vectors)
    matrix, policies, parity = [], [], 0
    rates = read(WORK/'results/e1-selection.json')['time_model']['rates']
    timer = read(RUNS/'nlp-hw02-ensemble-inference-v11/interactive/test-1-timing.json')
    # Field names are the actual frozen v11 phase names, not accumulated sampling counts.
    stage2 = timer['phases_seconds']['stage2_sampling_restore_submit']
    cold = timer['full_pipeline_seconds']
    baseline_extra_cost = None
    for label, names in POLICIES:
        sim = M.FastReplay(journals, names[0], names)
        bits = sim.bits([1, 8, 8]); matrix.append(bits)
        parity += sim.validate([1, 8, 8], [1, 1, 1])
        p = (2048-sim.stopped)/2048
        kinds = ['W' if n.startswith('W-') else 'P' if n == 'pilot-b' else 'C' for n in names]
        extra_cost = p*7*sum(rates[k+'/0.7:K15'] for k in kinds[1:])
        if baseline_extra_cost is None: baseline_extra_cost = extra_cost
        stage1_delta = timer['phases_seconds']['stage1_pilot-b_sampling_restore_submit']*(
            rates[kinds[1]+'/0.7:K1']/rates['P/0.7:K1']-1)
        forecast = cold-stage2+stage2*extra_cost/baseline_extra_cost+stage1_delta
        policies.append(dict(name=label, models=names, anchor=names[0], K1=[1]*3, K2=[1, 8, 8],
                             matches=int(bits.sum()), early_stops=sim.stopped, mean_samples=3+14*p,
                             forecast_seconds=forecast, correct_bits=bitstring(bits)))
    matrix = np.asarray(matrix, dtype=bool); baseline = matrix[0]
    assert int(baseline.sum()) == 1551 and policies[0]['early_stops'] == 1266
    for row, bits in zip(policies, matrix): row['paired_vs_v11'] = paired(bits, baseline)
    questions = read(RUNS/'nlp-hw02-kaggle-s1/1/inputs/heldout4k.json')
    for i in IDS:
        assert hashlib.sha256(questions[i]['question'].encode()).hexdigest() == journals[NAMES[0]][i]['question_sha256']
    # Prefer the frozen control on a score tie; then shorter forecast and fixed order.
    def rank(j, counts):
        return (int(counts[j]), j == 0, -policies[j]['forecast_seconds'], -j)
    cv = {}
    for family, choices in [('including-control', list(range(6))), ('B7-only', list(range(1, 6)))]:
        repeats = []
        for repeat in range(10):
            folds = S.halves(questions, repeat)
            oof = np.zeros(2048, dtype=bool); fits = []
            for held in [0, 1]:
                train, test = folds[1-held], folds[held]
                counts = matrix[:, train].sum(axis=1)
                selected = max(choices, key=lambda j: rank(j, counts))
                oof[test] = matrix[selected, test]
                fits.append(dict(policy=policies[selected]['name'], train_n=len(train), held_n=len(test),
                                 train_matches=int(counts[selected]), held_matches=int(matrix[selected, test].sum()),
                                 baseline_held_matches=int(baseline[test].sum()),
                                 train_ids_sha256=hashlib.sha256(json.dumps([IDS[i] for i in train]).encode()).hexdigest(),
                                 held_ids_sha256=hashlib.sha256(json.dumps([IDS[i] for i in test]).encode()).hexdigest()))
            repeats.append(dict(repeat=repeat, fits=fits, OOF_matches=int(oof.sum()),
                                paired_vs_v11=paired(oof, baseline), correct_bits=bitstring(oof)))
        cv[family] = dict(primary=repeats[0], fixed_repeats=repeats,
                          mean_OOF_matches=float(np.mean([r['OOF_matches'] for r in repeats])),
                          range_OOF_matches=[min(r['OOF_matches'] for r in repeats), max(r['OOF_matches'] for r in repeats)])
    full_counts = matrix.sum(axis=1)
    return dict(policies=policies, best=policies[max(range(6), key=lambda j: rank(j, full_counts))]['name'],
                best_B7=policies[max(range(1, 6), key=lambda j: rank(j, full_counts))]['name'],
                single_K8=singles, single_comparisons=single_comparisons, split_half=cv,
                bit_positions='correct_bits[j] corresponds to fixed_split tune2048 IDS[j]; ASCII 0/1',
                primary_halves=[[IDS[i] for i in f] for f in S.halves(questions, 0)],
                production_independent_parity_rows=parity, graph_fixtures=B.graph_fixtures(),
                journals=320, samples=81920, journal_sha256=receipts,
                search='Six fixed compositions, K1=1/1/1 -> K2=1/8/8, T0.7; no K/anchor/temperature search',
                CV='Inherited B6 grouped question splits, repeats0..9, repeat0 primary; train-only composition choice; ties prefer control, then time, then fixed order',
                timing='v11 cold fixed part plus measured stage2 scaled by pending fraction and E1 K15 cost; replacing P additionally rescales its stage1 by E1 K1 rates. Other phases/lengths/batch/OOM behavior assumed unchanged; forecast only',
                baseline=dict(old_K8=1551, fresh_GPU_tune=1547, public_from_user=79.37, cold_test_seconds=cold))


def main():
    result = audit()
    print('audited', len(result['producer_checked']), 'producer files', flush=True)
    result['ensemble'] = ensembles(result)
    e = result['ensemble']
    # Bind every reused B6/P1 file (including all K8 journals) to audited B6 inputs.
    prior = read(WORK/'results/b6-results.json')
    inherited = 0
    for path, sha in list(INPUTS.items()):
        if path in prior['inputs_sha256']:
            assert sha == prior['inputs_sha256'][path], path
            inherited += 1
    # P1's recovered flat journals are bound through the inherited manifest
    # receipts, not B6's inputs_sha256 (which did not contain these 64 paths).
    p1_receipts = prior['ensemble']['journal_sha256']['pilot-b']
    p1_files = [path for path in e['journal_sha256'] if 'model-pilot-b__' in path]
    assert len(p1_files) == 64
    for path in p1_files:
        assert INPUTS[path] == p1_receipts[Path(path).name], path
    inherited += len(p1_files)
    result['inherited_inputs_verified'] = inherited
    gates = {name: result['comparisons'][f'{name} minus {old}']['heldout4k/all']
             for name, old in PAIRS[:2]}
    passed = [n for n, r in gates.items() if r['delta_pp'] >= .3 and r['bootstrap_CI95_pp'][0] > 0]
    qualifying_policies = [r['name'] for r in e['policies'][1:4]
                          if r['matches'] >= e['policies'][0]['matches'] and
                          any(n in r['models'] for n in passed)]
    result['recommendation'] = dict(single_gate_models=passed, qualifying_v11_replacements=qualifying_policies,
                                    build_v12=bool(passed and qualifying_policies),
                                    criterion='At least one B7 greedy heldout4k delta>=0.3pp and paired bootstrap lower>0; v11 fixed-policy replacement using that model has >= baseline old-K8 matches. No claim of statistical noninferiority.')
    result.update(status='complete', GPU=False, new_forward=False, new_execution=False, new_sampling=False, API=False,
                  provenance=dict(kernel='friend1/nlp-hw02-b7', version_from_user=1),
                  CI='20000 paired task multinomial bootstrap draws; seed2026102007; percentile95, pointwise descriptive; CE lacks per-example NLL',
                  limits=['B7 compact output omits step journals and model tensors; summaries/source/config/endpoints checked, no full numerical training/restore audit',
                          'Same uncooled B5 sources as B6; B5 pre-final tensors had been rotated away',
                          'Heldout, tune and old991 have historical reuse; split-half only addresses current composition selection',
                          'No independent B7 K8 eval2048 or sampling replicates; paired CIs condition on saved generation',
                          'OOF CI conditions on fitted policies, without refitting inside bootstrap; overlapping repeats are not independent tests',
                          'Forecast is not a cold mixed T4 timing/memory gate; B6-for-P variants change resident architecture/cost'])
    for path in [Path(__file__), HERE/'analyze-b6-results.py', HERE/'analyze-b5-results.py',
                 HERE/'analyze-b4-results.py', HERE/'analyze-b3-results.py', WORK/'kaggle/src/final_ensemble.py',
                 WORK/'kaggle/src/final_choose.py', WORK/'kaggle/final_v7_analysis.py']:
        INPUTS[str(path.relative_to(WORK))] = digest(path)
    result['inputs_sha256'] = INPUTS
    B.write(WORK/'results/b7-results.json', result)
    B.write(WORK/'checks/b7-results-cpu.json', dict(passed=True, producer_files=len(result['producer_checked']),
            missing_producer_files=len(result['producer_missing']), inherited_inputs_verified=inherited,
            cooldown_updates_from_summaries=98800,
            cooldown_step_rows_checked=sum(r['step_rows_checked'] for r in result['stability'].values()),
            K8_journals=e['journals'], K8_samples=e['samples'], fixed_policies=6, grouped_split_half_repeats=10,
            production_independent_parity_rows=e['production_independent_parity_rows'], graph_fixtures=e['graph_fixtures'],
            no_GPU=True, no_forward=True, no_execution=True, no_sampling=True, no_API=True))
    print(json.dumps(dict(recommendation=result['recommendation'],
                          policies=[{k: r[k] for k in ['name', 'matches', 'paired_vs_v11', 'forecast_seconds']} for r in e['policies']],
                          split_half={k: dict(primary=v['primary']['OOF_matches'], mean=v['mean_OOF_matches']) for k, v in e['split_half'].items()}), indent=2))


if __name__ == '__main__':
    main()
