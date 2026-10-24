"""CPU audit of B6 and saved K8 replay with grouped split-half selection.

Run with NumPy/Matplotlib and a C++17 compiler. No model forwards, program
execution, sampling, GPU, or network. Raw journals and temporary binaries stay
outside Git. Split definitions and search space are fixed in this source.
"""
import csv
import ctypes
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('b5_analysis', HERE/'analyze-b5-results.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
B, np = M.B, M.np
WORK, RUNS, IDS = M.WORK, M.RUNS, M.IDS
RAW = RUNS/'nlp-hw02-b6/1'
NAMES = ['C-main', 'C-long2x', 'C-long2x-B5', 'C-long2x-B6',
         'pilot-b', 'W-B4', 'W-B5', 'W-B6']
INPUTS, RECORDS = {}, {}
BINS = [(0, 60), (60, 90), (90, 120), (120, math.inf)]


def read(path):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(WORK))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def paired(a, b):
    out = M.paired(a, b, boot=False)
    n, g, l = out['n'], out['gained'], out['lost']
    draws = np.random.default_rng(2026101906).multinomial(
        n, [g/n, l/n, 1-(g+l)/n], size=20000)
    out['bootstrap_CI95_pp'] = np.quantile(100*(draws[:, 0]-draws[:, 1])/n, [.025, .975]).tolist()
    return out


def getters():
    def flat(run, prefix):
        return lambda suffix: RUNS/run/(prefix+'__'+suffix.replace('/', '__'))
    def tree(run, gpu, tag):
        return lambda suffix: RUNS/run/f'model-GPU{gpu}/points'/tag/suffix
    return {
        'C-main': flat('nlp-hw02-kaggle-l3/final', 'model-C-main__points__l2-62280'),
        'C-long2x': flat('nlp-hw02-kaggle-l3/final', 'model-C-long2x__points__l2-62280'),
        'pilot-b': flat('nlp-hw02-kaggle-p1/final', 'model-pilot-b__points__long-truncation'),
        'W-B4': tree('nlp-hw02-b4/1', 0, 'b4-final-091801'),
        'W-B5': tree('nlp-hw02-b5/1', 0, 'b5-final-125641'),
        'C-long2x-B5': tree('nlp-hw02-b5/1', 1, 'b5-final-113690'),
        'W-B6': tree('nlp-hw02-b6/1', 0, 'b6-final'),
        'C-long2x-B6': tree('nlp-hw02-b6/1', 1, 'b6-final'),
    }


def audit():
    session = read(RAW/'session-summary.json')
    assert 'COMPLETE' in (RAW/'kernel-status.txt').read_text()
    assert session['status'] == 'complete' and session['exit_codes'] == [0, 0]
    assert not session['unreaped_worker_pids'] and session['wall_seconds'] < 19200
    checked, missing = [], []
    for entry in read(RAW/'b1-output-manifest.json')['files']:
        path = RAW/entry['file']
        if not path.is_file():
            missing.append(entry)
            continue
        raw = path.read_bytes()
        assert len(raw) == entry['bytes'] and hashlib.sha256(raw).hexdigest() == entry['sha256'], path
        INPUTS[str(path.relative_to(WORK))] = entry['sha256']
        checked.append(entry['file'])
    config = read(RAW/'session-config.json')
    identity = read(WORK/'checks/b6-review-gpu-smoke.json')
    assert config['gpu_smoke'] is False and config['owner'] == 'friend1'
    pins = read(WORK/'kaggle/b6-anchors.json')
    assert config['b6_pins'] == pins
    # Verify port hash against the same module namespace used by production.
    sources = {n: (WORK/'kaggle/src'/n if (WORK/'kaggle/src'/n).exists() else WORK/'kaggle'/n).read_text()
               for n in identity['modules']}
    overrides = {n: (WORK/'kaggle/legacy-c'/n).read_text() for n in identity['C_overrides']}
    assert config['port_sha256'] == hashlib.sha256(json.dumps(
        dict(SOURCES=sources, C_OVERRIDES=overrides), sort_keys=True).encode()).hexdigest()
    stability = {}
    for gpu, name in [(0, 'W-B6'), (1, 'C-long2x-B6')]:
        root = RAW/f'model-GPU{gpu}'
        s = read(root/'summary.json'); cd = s['cooldown']; anchor = pins['anchors'][str(gpu)]
        assert s['status'] == 'complete' and s['final_complete'] and s['main_added_updates'] == 0
        assert cd['N'] == cd['updates'] == pins['budget'][str(gpu)]['N']
        assert cd['source_sha256'] == anchor['sha256'] and cd['q'] == cd['last_lr'] == 0
        assert cd['origin_updates'] == s['main_state']['updates'] == anchor['updates']
        assert cd['origin_cursor'] == s['main_state']['cursor'] == anchor['cursor']
        assert cd['cursor'] == anchor['cursor']+128*cd['N']
        old = read(RUNS/f'nlp-hw02-b5/1/model-GPU{gpu}/summary.json')
        assert s['main_state'] == old['state']
        assert read(root/'start-fingerprints.json')['source_sha256'] == anchor['sha256']
        assert read(root/'latest.json') == s['latest']
        path = root/'cooldown-updates.jsonl'; raw = path.read_bytes()
        INPUTS[str(path.relative_to(WORK))] = hashlib.sha256(raw).hexdigest()
        logs = [json.loads(line) for line in raw.splitlines()]
        assert len(logs) == cd['N']
        elapsed = 0.
        for j, x in enumerate(logs, 1):
            assert x['branch_update'] == j and x['cursor'] == anchor['cursor']+128*j
            assert x['lr'] == cd['peak_lr']*(1-math.sqrt(j/cd['N'])) and x['pairs'] == 128
            assert 'truncation' not in x and not x['OOM_retries'] and x['nonfinite_skip'] == 0
            assert math.isfinite(x['CE']) and (x['AMP_skip'] or math.isfinite(x['gradient_norm']))
            if gpu == 0: assert math.isfinite(x['z_loss'])
            elapsed += x['seconds']
            assert elapsed == x['cooldown_seconds']
        assert elapsed == cd['seconds'] and logs[-1]['stream_sha256'] == cd['stream_sha256']
        assert sum(x['AMP_skip'] for x in logs) == cd['AMP_skips']
        assert cd['nonfinite_skips'] == cd['stability']['total'] == cd['stability']['escalations'] == 0
        text = (RAW/f'gpu{gpu}-process.log').read_text()
        assert not re.search(r'Traceback|WORKER FAILED|POINT INCOMPLETE|\b(?:NaN|nan|inf)\b', text)
        training = read(root/'points/b6-final/training.json')
        assert training['cooldown'] == cd and training['completed_updates'] == cd['N']
        assert training['endpoint_lr'] == training['main_added_updates'] == training['q'] == 0
        stability[name] = dict(N=cd['N'], seconds=cd['seconds'], pairs=128*cd['N'],
            AMP=cd['AMP_skips'], nonfinite=0, OOM=0, endpoint_lr=0,
            source_sha256=cd['source_sha256'], full_latest=s['latest'], midpoint=s['midpoint'])

    models = {}
    refs = None
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
        groups = {'heldout4k': list(range(4096)), 'tune2048': IDS,
                  'eval2048': [i for i in range(4096) if i not in set(IDS)]}
        groups.update({f'bin{lo}-{hi}': [x['id'] for x in rr['heldout4k'] if lo <= x['original_bpe_length'] < hi] for lo, hi in BINS})
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
    prior = read(WORK/'results/b5-results.json')
    # Every inherited local input read above must still match the B5 audit.
    inherited = 0
    for path, digest in list(INPUTS.items()):
        if path in prior['inputs_sha256']:
            assert digest == prior['inputs_sha256'][path], path
            inherited += 1
    comparisons = {}
    for a, b in [('W-B6', 'W-B5'), ('C-long2x-B6', 'C-long2x-B5'),
                 ('W-B6', 'pilot-b'), ('C-long2x-B6', 'pilot-b'), ('W-B6', 'C-long2x-B6')]:
        out = {}
        for split in ['heldout4k', 'old991']:
            aa, bb = RECORDS[a][split], RECORDS[b][split]
            groups = {'all': list(range(len(aa)))}
            if split == 'heldout4k':
                groups.update(tune2048=IDS, eval2048=[i for i in range(4096) if i not in set(IDS)])
                groups.update({f'bin{lo}-{hi}': [x['id'] for x in aa if lo <= x['original_bpe_length'] < hi] for lo, hi in BINS})
            for key, ids in groups.items():
                out[f'{split}/{key}'] = paired([aa[i]['correct'] for i in ids], [bb[i]['correct'] for i in ids])
        out['CE_delta'] = {s: models[a]['CE'][s]-models[b]['CE'][s] for s in ['heldout4k', 'old991']}
        comparisons[f'{a} minus {b}'] = out
    return dict(session=session, producer_checked=checked, producer_missing=missing, inherited_inputs_verified=inherited,
                stability=stability, models=models, comparisons=comparisons)


def load_journals():
    journals, receipts = B.load_all()
    for name, getter in getters().items():
        if name in journals: continue
        files = sorted(getter('evaluation/k8-tune2048').glob('batch-*.json'))
        assert len(files) == 64
        rows = [r for f in files for r in read(f)]
        assert [r['id'] for r in rows] == IDS
        journals[name] = {r['id']: r for r in rows}
    for name in NAMES:
        for i in IDS:
            row, ref = journals[name][i], RECORDS[name]['heldout4k'][i]
            assert row['question_sha256'] == ref['question_sha256'] and row['reference'] == ref['reference']
            assert row['greedy']['execution'] == ref['execution']
            assert len(row['samples']['0.7']) == 8
            for sample in row['samples']['0.7']:
                assert sample['sampling_status'] == 'complete' and math.isfinite(sample['mean_logprob'])
            win = B.choose(row, dict(mode='vote', temperature=.7, k=8, greedy_vote=False))
            assert B.correct(row['reference'], win['execution']) == row['correct']
    return journals, receipts


def halves(questions, repeat):
    groups = {}
    for pos, i in enumerate(IDS):
        q = re.sub(r'\s+', ' ', questions[i]['question'].strip()).casefold()
        groups.setdefault(q, []).append(pos)
    folds = [[], []]
    for q, positions in sorted(groups.items(), key=lambda kv: hashlib.sha256(f'b6-cv-v1:{repeat}:{kv[0]}'.encode()).digest()):
        folds[0 if len(folds[0]) <= len(folds[1]) else 1].extend(positions)
    return [np.asarray(sorted(f), dtype=int) for f in folds]


def ensembles(result, library):
    journals, receipts = load_journals()
    ratios = {n: dict(sample=1., greedy=1.) for n in NAMES}
    for n in NAMES:
        if n.startswith('W-'):
            ratios[n] = read(WORK/'results/b3-results.json')['ensemble']['ratios']['W-lr600']
    sim = M.FastReplay(journals, 'C-long2x', ['C-long2x', 'pilot-b', 'W-B4'])
    baseline = sim.bits([1, 8, 1]); assert baseline.sum() == 1541
    baseline_cost = sum((1+(2048-sim.stopped)/2048*(k-1))*ratios[n]['sample']
                        for n, k in zip(sim.names, [1, 8, 1]))
    parity = sim.validate([1, 8, 1], [1, 1, 1])
    singles = []
    for name in NAMES:
        one = B.Replay(journals, name, [name])
        for greedy in [False, True]:
            bits, _, _ = one.evaluate([8], greedy=greedy)
            if not greedy: assert sum(bits) == result['models'][name]['K8']['matches']
            singles.append(dict(model=name, greedy_vote=greedy, matches=sum(bits), correct_ids=[i for i, b in zip(IDS, bits) if b]))
    single_comparisons = {}
    for a, b in [('W-B6', 'W-B5'), ('C-long2x-B6', 'C-long2x-B5'), ('W-B6', 'pilot-b'), ('C-long2x-B6', 'pilot-b')]:
        for greedy in [False, True]:
            aa, bb = [next(x for x in singles if x['model'] == n and x['greedy_vote'] == greedy) for n in [a, b]]
            sets = [set(x['correct_ids']) for x in [aa, bb]]
            single_comparisons[f'{a} minus {b}/greedy={greedy}'] = paired(*[[i in s for i in IDS] for s in sets])

    grid, eligible, bitlist = [], [], []
    fields = ['models', 'anchor', 'K2', 'matches', 'stopped', 'max_samples', 'mean_samples', 'mean_C_units', 'conservative_s', 'stress15_s']
    bitchecks, families = 0, 0
    path = WORK/'results/b6-ensemble-grid.csv'
    with path.open('w') as handle:
        writer = csv.writer(handle); writer.writerow(fields)
        for count in [2, 3, 4]:
            configs = [ks for ks in itertools.product(range(1, 9), repeat=count) if sum(ks) <= 14]
            for family in itertools.combinations(NAMES, count):
                for anchor in (family if count == 2 else family[:1]):
                    names = (anchor,)+tuple(n for n in family if n != anchor)
                    replay = M.FastReplay(journals, anchor, list(names))
                    bits = M.native_replay(replay, configs, library)
                    families += 1; bitchecks += 3*2048
                    for ks, rowbits in zip(configs, bits):
                        p = (2048-replay.stopped)/2048
                        means = [1+p*(k-1) for k in ks]
                        t = M.timing(names, anchor, ks, replay.stopped, ratios)
                        row = dict(models=list(names), anchor=anchor, K1=[1]*count, K2=list(ks), matches=int(rowbits.sum()),
                                   stopped=replay.stopped, max_samples=sum(ks), mean_samples=sum(means),
                                   mean_C_units=sum(v*ratios[n]['sample'] for n, v in zip(names, means)),
                                   conservative_s=t['conservative_s'], stress15_s=t['stress15_s'])
                        writer.writerow(['+'.join(names), anchor, '/'.join(map(str, ks))]+[row[k] for k in fields[3:]])
                        grid.append(row)
                        if row['conservative_s'] <= 140:
                            eligible.append(row); bitlist.append(rowbits.copy())
                print('replayed', family, 'total', len(grid), flush=True)
    # Keep the complete derived table compact; raw generations remain ignored.
    packed_path = path.with_suffix('.csv.gz')
    packed_path.write_bytes(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0))
    path.unlink()
    path = packed_path
    matrix = np.asarray(bitlist, dtype=np.uint8)
    def order(j, counts):
        r = eligible[j]
        return (int(counts[j]), -r['conservative_s'], -r['mean_C_units'], -j)
    totals = matrix.sum(axis=1)
    def best(predicate):
        choices = [j for j, row in enumerate(eligible) if predicate(row)]
        return max(choices, key=lambda j: order(j, totals))
    chosen = dict(all_140=best(lambda r: True), stress140=best(lambda r: r['stress15_s'] <= 140),
                  both_B6=best(lambda r: all(n in r['models'] for n in ['C-long2x-B6', 'W-B6'])),
                  contains_B6=best(lambda r: any(n.endswith('-B6') for n in r['models'])),
                  at_v8_cost=best(lambda r: r['mean_C_units'] <= baseline_cost and r['max_samples'] <= 12))
    def report(j):
        r = dict(eligible[j]); r['paired_vs_v8'] = paired(matrix[j], baseline)
        return r
    winners = {label: report(j) for label, j in chosen.items()}
    questions = read(RUNS/'nlp-hw02-kaggle-s1/1/inputs/heldout4k.json')
    for i in IDS:
        assert hashlib.sha256(questions[i]['question'].encode()).hexdigest() == journals['C-main'][i]['question_sha256']
    cv = []
    # Fixed repeats, never choose a favorable split. Repeat 0 is primary.
    for repeat in range(10):
        folds = halves(questions, repeat)
        oof = np.zeros(2048, dtype=bool); fits = []
        for held in [0, 1]:
            train, test = folds[1-held], folds[held]
            counts = matrix[:, train].sum(axis=1)
            selected = max(range(len(eligible)), key=lambda j: order(j, counts))
            oof[test] = matrix[selected, test]
            fits.append(dict(train_n=len(train), held_n=len(test), train_matches=int(counts[selected]),
                             held_matches=int(matrix[selected, test].sum()), baseline_held_matches=int(baseline[test].sum()),
                             policy=dict(eligible[selected]), train_ids=[IDS[i] for i in train], held_ids=[IDS[i] for i in test],
                             held_paired_vs_v8=paired(matrix[selected, test], baseline[test])))
        cv.append(dict(repeat=repeat, fits=fits, OOF_matches=int(oof.sum()), paired_vs_v8=paired(oof, baseline),
                       correct_ids=[i for i, b in zip(IDS, oof) if b]))
    replacements = []
    for c, w in itertools.product(['C-long2x', 'C-long2x-B5', 'C-long2x-B6'], ['W-B4', 'W-B5', 'W-B6']):
        replay = M.FastReplay(journals, c, [c, 'pilot-b', w]); bits = replay.bits([1, 8, 1])
        replacements.append(dict(models=replay.names, anchor=c, K1=[1]*3, K2=[1, 8, 1], matches=int(bits.sum()),
                                 paired_vs_v8=paired(bits, baseline)))
    # Production and independent oracle parity for every reported full-grid or CV fit.
    policies = list(winners.values())+[fit['policy'] for row in cv for fit in row['fits']]
    validated = set()
    for r in policies:
        key = (tuple(r['models']), r['anchor'], tuple(r['K2']))
        if key in validated: continue
        replay = M.FastReplay(journals, r['anchor'], r['models'])
        parity += replay.validate(r['K2'], r['K1']); validated.add(key)
        assert replay.bits(r['K2']).sum() == r['matches']
        assert M.timing(tuple(r['models']), r['anchor'], tuple(r['K2']), replay.stopped, ratios)['conservative_s'] == r['conservative_s']
    return dict(best=winners, replacements=replacements, single_K8=singles, single_comparisons=single_comparisons,
                split_half=dict(primary=cv[0], fixed_repeats=cv, mean_OOF_matches=float(np.mean([r['OOF_matches'] for r in cv])),
                                range_OOF_matches=[min(r['OOF_matches'] for r in cv), max(r['OOF_matches'] for r in cv)],
                                method='10 fixed grouped two-fold splits; train-only policy choice over entire eligible grid; repeat 0 primary; ties time/cost/grid order',
                                limits='Repeated tune, historical v8 selection and frozen model-pool decisions are not undone. OOF CIs condition on fitted policies; no resampling selection or Monte Carlo run variability. Repeats overlap and are not 10 independent tests.'),
                grid_configurations=len(grid), eligible_140=len(eligible), families=families, ratios=ratios,
                grid_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), journal_sha256=receipts,
                production_independent_parity_rows=parity, native_python_bit_comparisons=bitchecks,
                graph_fixtures=B.graph_fixtures(), native_fixtures=M.native_fixtures(library),
                search='All subsets 2-4 of eight models, both pair anchors and canonical first-C for triples/quads; K1=1 each, K2=1..8, sum<=14; max(test,tune) cautious proxy<=140s',
                baseline=dict(v8_offline_matches=1541, v8_mean_C_units=baseline_cost,
                              v8_v9_GPU_tune_matches=1532, public_from_user=dict(v7=77.92, v8=78.54, v9=78.44)))


def main():
    result = audit()
    B.write(WORK/'results/b6-results.json', result)
    print('audited', len(result['producer_checked']), 'producer files', flush=True)
    with tempfile.TemporaryDirectory(prefix='nlp-b6-replay-') as tmp:
        path = Path(tmp)/'replay.dylib'
        subprocess.run(['c++', '-O3', '-shared', '-fPIC', '-std=c++17', str(HERE/'b5-replay.cpp'), '-o', str(path)], check=True)
        library = ctypes.CDLL(str(path)); library.replay_many.restype = None
        result['ensemble'] = ensembles(result, library)
    result.update(status='complete', GPU=False, new_forward=False, new_execution=False, new_sampling=False, API=False,
                  provenance=dict(kernel='friend1/nlp-hw02-b6', version_from_user=1, start_UTC_from_user='2026-10-19 10:05'),
                  CI='20000 paired multinomial row-bootstrap draws; seed2026101906, pointwise. CE has no CI without per-example NLL.',
                  limits=['Full/FP16 tensors not downloaded or loaded; no new CUDA restore/finiteness audit',
                          'B5 pre-final tensors were rotated away; B6 pins match audited post-restore uncooled latest, see b6-plan',
                          'Cooldown duration and shape change together; against P1-B main history and truncation also differ',
                          'No eval2048 K8 for B6 or B5; TTA eval covers old v8 weights only and cannot evaluate B6',
                          'Proxy is not measured mixed timing; multiple resident W models require memory/timing verification'])
    result['inputs_sha256'] = {**INPUTS, **B.INPUTS}
    for path in [Path(__file__), HERE/'analyze-b5-results.py', HERE/'analyze-b4-results.py', HERE/'analyze-b3-results.py', HERE/'b5-replay.cpp']:
        result['inputs_sha256'][str(path.relative_to(WORK))] = hashlib.sha256(path.read_bytes()).hexdigest()
    B.write(WORK/'results/b6-results.json', result)
    e = result['ensemble']
    B.write(WORK/'checks/b6-results-cpu.json', dict(passed=True, producer_files=len(result['producer_checked']),
            missing_producer_files=len(result['producer_missing']), cooldown_updates=sum(r['N'] for r in result['stability'].values()),
            K8_journals=512, K8_samples=131072, grid_configurations=e['grid_configurations'], eligible_140=e['eligible_140'],
            native_python_bit_comparisons=e['native_python_bit_comparisons'], production_independent_parity_rows=e['production_independent_parity_rows'],
            graph_fixtures=e['graph_fixtures'], native_fixtures=e['native_fixtures'], grouped_split_half_repeats=10,
            no_GPU=True, no_forward=True, no_execution=True, no_sampling=True, no_API=True))
    print(json.dumps(dict(best=e['best'], split_half={k:v for k,v in e['split_half'].items() if k not in ['primary','fixed_repeats']}), indent=2))


if __name__ == '__main__':
    main()
