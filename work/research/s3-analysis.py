"""Offline pinned S3 audit and frozen tune-only candidate selection; no training/push."""
import argparse
import ast
import hashlib
import importlib.util
import importlib.metadata
import math
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from common import WORK, INPUT, S1, bucket, literals, number_info, question_numbers, read, write
from exp26_common import sha_file, json_hash
import exp26_exec as N
import s3_representation as R
from exp26_fast_exec import run_chunks
import s2_exec

ROOT = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-s3/1'
CONTROL = WORK / 'kaggle/staging/drive-merged/26-wsd-data-scaling'
OUT = WORK / 'research/outputs/s3-analysis'
MIX = [0., .25, .5, 1., 2., 4.]


def taxonomy(rows, gens, execution, score, lengths):
    """Exact data-research taxonomy, including its unsigned-literal convention."""
    eligible, matched = set(score['eligible_indices']), set(score['match_indices'])
    tags, exclusive, statuses = [], [], Counter()
    oversized_literal_rows = []
    bins = {b: Counter() for b in ['<60', '60-90', '90-120', '>=120']}
    for i in sorted(eligible - matched):
        r, g, x = rows[i], gens[i], execution['rows'][i]
        refvals = literals(r['code'])
        try:
            predvals = literals(g['generated_code'])
            info = number_info(r['question'], g['generated_code'])
        except OverflowError:
            # Earlier A6.4M diagnostics had no literal outside float range. Retain
            # exact oversized ints for set comparisons; count them as novel numbers.
            tree = ast.parse(g['generated_code'])
            predvals = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and type(n.value) in (int, float)]
            oversized = 0
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and type(n.value) is int and abs(n.value) > 10**308:
                    n.value = 0; oversized += 1
            info = number_info(r['question'], ast.unparse(tree))
            info.setdefault('counts', {})['other'] = info.get('counts', {}).get('other', 0) + oversized
            oversized_literal_rows.append(i)
        qvals = {a[0] for a in question_numbers(r['question'])}
        tt = []
        if x['status'] != 'numeric': tt.append('runtime_or_non_numeric')
        if info.get('counts', {}).get('other', 0): tt.append('number_not_from_question_or_known_constant')
        if predvals is not None:
            required = {v for v in refvals if v in qvals}
            if required - set(predvals): tt.append('used_question_number_missing')
            if set(predvals) == set(refvals): tt.append('same_literal_set_wrong_program')
        if not tt: tt.append('other_numeric_or_structure')
        tags.extend(tt); exclusive.append(tt[0]); statuses[x['status']] += 1
        bins[lengths[i]][tt[0]] += 1
    errors = len(eligible - matched)
    return dict(errors=errors, statuses=dict(statuses), overlapping_tags=dict(Counter(tags)),
                exclusive_priority_counts=dict(Counter(exclusive)), by_reference_length={b: dict(v) for b, v in bins.items()},
                oversized_literal_rows=oversized_literal_rows,
                caveat='Heuristic overlapping tags, exactly data-research: AST unsigned literal sets; missing only question digits used by reference; known/word/percent constants discounted.')


def same(a, b):
    return a['status'] == b['status'] and (a['status'] != 'numeric' or
        math.isclose(a['value'], b['value'], rel_tol=1e-6, abs_tol=1e-6))


def audit():
    import torch
    import exp26_eval as E
    import exp26_core as C
    s2_exec.install()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = read(ROOT / 'download-version.json')
    assert receipt['ref'] == 'thisarivson/nlp-hw02-kaggle-s3/1' and receipt['version_label'] == 'v1'
    manifest = read(ROOT / 's3-output-manifest.json')
    missing, byte_differences = [], []
    inventory = [dict(file=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size, sha256=sha_file(p))
                 for p in sorted(ROOT.rglob('*')) if p.is_file()]
    assert not list(ROOT.rglob('*.download-tmp'))
    for row in manifest['files']:
        p = ROOT / row['file']
        if not p.exists(): missing.append(row['file'])
        elif p.stat().st_size != row['bytes']: byte_differences.append(row['file'])
    assert missing == ['__notebook__.ipynb'] and not byte_differences
    assert len(inventory) == receipt['files'] + 2  # local version receipt + API kernel log
    original_source = read(WORK / 'kaggle/source-identity.json')
    for name, digest in original_source['modules'].items(): assert sha_file(ROOT / 'src' / name) == digest
    cfg = read(ROOT / 'session-config.json')
    sources = {p.name: p.read_text() for p in sorted((ROOT / 'src').glob('*.py'))}
    import json
    assert hashlib.sha256(json.dumps(sources, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == cfg['port_sha']
    report = dict(date='2026-10-12', pinned_version=1, API_status='COMPLETE',
        python=platform.python_version(), packages={name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'tokenizers', 'sympy']},
        internal_session=read(ROOT / 'session-summary.json'), download=receipt,
        SHA_inventory=inventory, bytes=sum(x['bytes'] for x in inventory),
        manifest=dict(missing=missing, byte_differences=byte_differences,
            note='Kaggle internal notebook omitted by SDK; manifest has bytes only, SHA is local plus producer receipts.'),
        source_identity='original frozen26 modules and full pushed port SHA passed', experiments={})
    report['source_receipts'] = {p.relative_to(WORK).as_posix(): sha_file(p)
        for p in [INPUT / 'tokenizer.json'] + [INPUT / (s + suffix) for s in ['heldout4k', 'old1024']
            for suffix in ['.json', '-reference-execution.json']] +
        [CONTROL / 'points' / tag / relative for tag in ['p006250', 'p012500']
            for relative in ['training.json', 'evaluation/metrics.json', 'evaluation/heldout4k-generations.json',
                'evaluation/heldout4k-execution.json', 'evaluation/old1024-generations.json', 'evaluation/old1024-execution.json']]}
    sets = {s: read(INPUT / (s + '.json')) for s in ['heldout4k', 'old1024']}
    refs = {s: read(INPUT / (s + '-reference-execution.json')) for s in sets}
    tok = Tokenizer.from_file(str(INPUT / 'tokenizer.json'))
    lengths = {s: [bucket(len(t.ids)) for t in tok.encode_batch([r['code'] for r in rows])] for s, rows in sets.items()}
    baselines, evaluations, codes = {}, [], []
    for tag in ['p006250', 'p012500']:
        m = read(CONTROL / 'points' / tag / 'evaluation/metrics.json')
        baselines[tag] = dict(training={k: v for k, v in read(CONTROL / 'points' / tag / 'training.json').items() if k != 'metrics'}, splits={})
        for s, rows in sets.items():
            directory = CONTROL / 'points' / tag / 'evaluation'
            g, x = read(directory / (s + '-generations.json')), read(directory / (s + '-execution.json'))
            assert len(g) == len(rows) and all(v['id'] == i and v['question'] == rows[i]['question'] for i, v in enumerate(g))
            score = N.compare_exec(refs[s], x)
            assert score == m['splits'][s]['exec']
            assert E.length_accuracy(rows, tok, score) == m['splits'][s]['length_accuracy']
            baselines[tag]['splits'][s] = dict(matches=score['matches'], denominator=score['denominator'],
                length_accuracy=m['splits'][s]['length_accuracy'], errors=taxonomy(rows, g, x, score, lengths[s]))
            evaluations.append(('26b', tag, s, g, x, score))
            codes.extend(v['generated_code'] for v in g)
    for variant in [30, 31]:
        root = ROOT / f'gpu{variant-30}-exp{variant}'
        summary = read(root / f'{variant}-summary.json')
        assert summary['identity']['code_sha256'] == cfg['port_sha']
        assert summary['identity']['original_tokenizer_sha256'] == sha_file(INPUT / 'tokenizer.json')
        assert summary['identity']['tokenizer_sha256'] == sha_file(root / 'tokenizer.json')
        latest = read(root / 'latest.json'); p = torch.load(root / latest['file'], map_location='cpu', weights_only=True)
        assert sha_file(root / latest['file']) == latest['sha256'] and (root / latest['file']).stat().st_size == latest['bytes']
        assert E.state_digest(p['model']) == latest['model_tensor_sha256']
        assert p['training']['main_updates'] == 6250 and p['training']['branch_updates'] == 0 and p['training']['data_cursor'] == 800000
        assert summary['identity']['order_sha256'] == '737dac5a146ad15bb0ec70cb06fea59493277e4332dea440b2bdf2a6316c1637'
        assert p['scheduler'] == __import__('exp26_train').schedule_state(p['training'], p['config'])
        assert all(int(v['step']) == 6250 for v in p['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for v in p['model'].values())
        assert not any(torch.is_inference(v) for v in p['model'].values())
        checkpoint = dict(latest, optimizer_steps=6250, scaler=p['scaler'],
            history_train_seconds=p['experiment_history']['train_seconds'],
            history_compute_pairs=p['experiment_history']['all_counts']['pairs'],
            phase_note='stable anchor, not evaluated cooldown; history includes completed cooldown cost; load yields normal tensors')
        del p
        rep_tok, rep_bytes = R.tokenizer(INPUT / 'tokenizer.json', variant)
        result = dict(status=summary['status'], missing_points=summary['missing_points'],
            preparation=read(root / 'representation-preparation.json'), checkpoint=checkpoint, points={},
            failure=read(ROOT / f'gpu{variant-30}-failure.json'), decision='pending: missing registered1.92M point')
        for tag, point in summary['points'].items():
            directory = root / 'points' / tag / 'evaluation'
            assert read(directory / 'metrics.json') == point['metrics']
            assert read(root / 'points' / tag / 'training.json') == {k: v for k, v in point.items() if k != 'metrics'}
            assert sha_file(root / 'weights' / (tag + '.pt')) == point['weights_sha256']
            weights = torch.load(root / 'weights' / (tag + '.pt'), map_location='cpu', weights_only=True)
            assert weights['format'] == 'nlp-hw02-model-only-v1' and weights['model_spec'] == summary['model_spec']
            assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for v in weights['model'].values())
            del weights
            if variant == 30:
                assert point['counts']['source_tokens'] == baselines[tag]['training']['counts']['source_tokens']
            pp = dict(training={k: v for k, v in point.items() if k != 'metrics'}, splits={})
            for s, rows in sets.items():
                g, x = read(directory / (s + '-generations.json')), read(directory / (s + '-execution.json'))
                assert len(g) == len(x['rows']) == len(rows)
                for i, v in enumerate(g):
                    assert v['id'] == i and v['question'] == rows[i]['question']
                    text = C.decode_bytes(rep_bytes, v['generated_token_ids']).decode('utf-8', errors='replace')
                    assert text == v['represented_code'] and v['eos_emitted'] == (C.EOS in v['generated_token_ids'])
                    try: restored, error = R.restore(text, rows[i]['question'], variant), None
                    except (ValueError, IndexError): restored, error = '', 'invalid representation'
                    assert restored == v['generated_code'] and error == v['restore_error']
                score = N.compare_exec(refs[s], x)
                m = point['metrics']['splits'][s]
                assert score == m['exec'] and E.length_accuracy(rows, tok, score) == m['length_accuracy']
                control_score = read(CONTROL / 'points' / tag / 'evaluation/metrics.json')['splits'][s]['exec']
                pair = E.paired_mcnemar(score, control_score)
                assert pair == m['McNemar_against_26b']
                pp['splits'][s] = dict(matches=score['matches'], denominator=score['denominator'],
                    accuracy=score['accuracy'], McNemar_vs_26b=pair, length_accuracy=m['length_accuracy'],
                    restore_errors=sum(v['restore_error'] is not None for v in g),
                    restore_error_ids=[i for i, v in enumerate(g) if v['restore_error'] is not None],
                    errors=taxonomy(rows, g, x, score, lengths[s]), generation_seconds=m['generation_seconds'])
                evaluations.append((str(variant), tag, s, g, x, score)); codes.extend(v['generated_code'] for v in g)
            pp['training']['speed_ratio_vs_26b'] = point['pairs_per_second_all_compute'] / (baselines[tag]['training']['counts']['pairs'] / baselines[tag]['training']['train_seconds_total'])
            result['points'][tag] = pp
        result['equal_time'] = dict(status='pending', interpolation_performed=False,
            reason='No saved equal-time branch; compact802.84s is below earliest positive control928.12s, log-time interpolation cannot bracket it; extrapolation not used.') if variant == 30 else {'status': 'not required'}
        report['experiments'][str(variant)] = result
    unique = list(dict.fromkeys(codes + [r['code'] for rows in sets.values() for r in rows]))
    identity = json_hash(unique); cache = OUT / 'reexecution.json'
    if cache.exists() and read(cache)['codes_sha256'] == identity:
        replay = read(cache)['replay']
    else:
        print('CPU replay', len(unique), 'unique programs', flush=True)
        replay = run_chunks(unique, workers=2, wall_limit=3., cpu_limit=2.,
            progress=lambda done, total, sec: print('REPLAY', done, '/', total, round(sec, 1), flush=True))
        write(cache, dict(codes_sha256=identity, replay=replay))
    lookup = dict(zip(unique, replay['rows']))
    report['CPU_reexecution'] = dict(replay['summary'], saved_rows=len(codes), unique_programs=len(unique), evaluations={})
    for s, rows in sets.items():
        differences = [i for i, r in enumerate(rows) if not same(lookup[r['code']], refs[s]['rows'][i])]
        assert not differences, differences
    for variant, tag, s, g, x, score in evaluations:
        fresh = [lookup[v['generated_code']] for v in g]
        fresh_score = N.compare_exec(refs[s], fresh)
        differences = [i for i, (a, b) in enumerate(zip(fresh, x['rows'])) if not same(a, b)]
        assert fresh_score == score and not differences, (variant, s, differences)
        report['CPU_reexecution']['evaluations'][f'{variant}/{tag}/{s}'] = dict(matches=score['matches'], denominator=score['denominator'],
            different_execution_indices=differences, exact_match_set=True)
    report['controls'] = baselines
    report['experiment32'] = dict(status='pending: B scoring never started', scored_candidates=0, expected=332800,
        cause='GPU0 RuntimeError escaped representation worker before B.run; generic partial summary is not a score receipt')
    assert not list((ROOT / 'gpu0-exp32').rglob('batch-*.json'))
    b = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-s2/2/gpu1-exp28'
    b_receipt = read(b / 'points/t04/checkpoint.json')
    assert sha_file(b / b_receipt['file']) == b_receipt['sha256'] == 'cbbe292cf65e37ec4876b8206b19b06b6fbe7541fd7be1115799c3ce085f720a'
    report['experiment32']['B_weights_SHA_verified'] = b_receipt['sha256']
    report['acceptance_rule'] = dict(primary_1920000_min_matches=2251, secondary_min_matches=528,
        compact_equal_time_required=True, missing_or_misaligned='pending, not accepted or rejected based on early point')
    report['local_fix'] = dict(file='../kaggle/src/s3_runtime.py', change='evaluate: inference_mode -> no_grad',
        regression='../checks/kaggle-s3-dtype-regression.json', GPU_verified=False,
        frozen_v1_source_unchanged=True, rebuilt_notebook=False)
    report['unverified'] = ['30/31 at1.92M', '30 equal-time quality', 'B scores and A/B selection',
        'real CUDA fix and speed/memory', 'production full1919 T4 pipeline<=180s', 'fresh/official score']
    report['new_GPU_runs'] = report['pushes'] = 0
    report['seconds'] = time.perf_counter() - started
    report['status'] = 'passed saved-artifact audit; experiment decisions pending'
    write(WORK / 'checks/kaggle-s3-v1-cpu-audit.json', report)
    summary = {k: v for k, v in report.items() if k not in ['SHA_inventory', 'CPU_reexecution']}
    summary['audit'] = '../checks/kaggle-s3-v1-cpu-audit.json'
    summary['CPU_reexecution'] = {k: v for k, v in report['CPU_reexecution'].items() if k != 'evaluations'}
    ensemble_path = WORK / 'results/32-candidate-ensemble-cpu-summary.json'
    if ensemble_path.exists():
        ensemble = read(ensemble_path)
        if ensemble.get('ensemble_tune_winner'):
            b = ensemble['B_CPU_scoring']; rule = ensemble['tune_winner']
            summary['CPU_B_ensemble'] = dict(summary=ensemble_path.name, precision='CPU FP32',
                requested_candidates=b['scored_candidates'], unique_programs=b['identity']['unique_programs'],
                scoring_seconds=b['seconds'], winner=rule, parameters=ensemble['selected_parameters'][rule],
                decision=ensemble['decision']['status'], GPU_S3_scores_remain_missing=True)
            summary['unverified'] = [v for v in summary['unverified'] if v != 'B scores and A/B selection']
            summary['unverified'].append('CPU FP32 scores versus T4 FP16 equivalence; unused T0.5/T1.0 scores absent')
    write(WORK / 'results/kaggle-s3-v1-summary.json', summary)
    print('AUDIT PASSED', round(report['seconds'], 1), flush=True)


def selection():
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location('candidate_selection', Path(__file__).with_name('candidate-selection.py'))
    D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
    OUT.mkdir(parents=True, exist_ok=True)
    # Persist the extension before loading/tuning any outcomes; existing A grids unchanged.
    protocol = dict(split='candidate-selection-v1 hash; exact previous tune2048/eval2048 and old991',
        anchor='k16/tune, frozen for k8/k32', probability='sum exp(meanA*lengthA), sum exp(meanB*lengthB), sum exp(sumA+lambda*sumB)',
        mean_extension='existing tau grid for A and B; AB uses meanA+lambda*meanB, tau=1, one free coefficient',
        mix_grid=MIX, dedup='exact code; retain maximum meanA then first; original tie priority preserved',
        families=['vote', 'vote-dedup', 'A-sum', 'A-mean', 'B-sum', 'B-mean', 'AB-sum', 'AB-mean'],
        B_missing='do not fabricate score, subset evaluation or winner; retain pending',
        gold='only k16/tune labels choose parameters; scorers accept no labels/reference')
    frozen = OUT / 'ensemble-protocol.json'
    if frozen.exists(): assert read(frozen) == protocol
    else: write(frozen, protocol)
    rows, originals, refs, sources = {}, {}, {}, {}
    for s in ['heldout4k', 'old1024']:
        rows[s], originals[s], sources[s] = D.load_split(s)
        refs[s] = read(INPUT / (s + '-reference-execution.json'))['rows']
        for r in rows[s]:
            for c in r['flat']: c['number_features'] = (0., 0., 0.)
    b_root = WORK / 'research/outputs/s3-cpu-b-scores'
    b_summary = read(b_root / 'summary.json') if (b_root / 'summary.json').exists() else None
    has_b = bool(b_summary and b_summary['status'] == 'complete')
    b_receipts = []
    if has_b:
        seen = set()
        assert b_summary['scored_candidates'] == 168960 and b_summary['identity']['precision'] == 'CPU FP32'
        for path in sorted(b_root.glob('chunk-*.json')):
            if '.sha.' in path.name: continue
            assert sha_file(path) == read(path.with_suffix('.sha.json'))['sha256']
            b_receipts.append([path.name, sha_file(path)])
            for score in read(path):
                s, i, j = score['split'], score['id'], score['candidate_index']
                assert (s, i, j) not in seen and 0 <= j < 33; seen.add((s, i, j))
                c = rows[s][i]['flat'][j]
                assert score['code_sha256'] == hashlib.sha256(c['code'].encode()).hexdigest()
                assert all(math.isfinite(score[k]) for k in ['sum_logprob', 'mean_logprob'])
                assert math.isclose(score['sum_logprob'], score['mean_logprob'] * score['length_with_eos'], rel_tol=1e-7, abs_tol=1e-4)
                assert score['temperature'] == c['temperature']
                c['B_score'] = score
        assert len(seen) == 168960
    previous = read(WORK / 'results/candidate-selection-cpu.json')
    ids = sorted(range(4096), key=lambda i: hashlib.sha256(f'candidate-selection-v1:heldout4k:{i}'.encode()).digest())
    tune_ids, eval_ids = sorted(ids[:2048]), sorted(ids[2048:])
    assert tune_ids == previous['split']['tune_ids'] and eval_ids == previous['split']['eval_ids']
    tm, em = np.zeros(4096, bool), np.zeros(4096, bool); tm[tune_ids] = True; em[eval_ids] = True
    om = np.array([r['status'] == 'numeric' for r in refs['old1024']]); assert om.sum() == 991
    tok = Tokenizer.from_file(str(INPUT / 'tokenizer.json'))
    lengths = {s: np.array([bucket(len(t.ids)) for t in tok.encode_batch([r['code'] for r in rr])]) for s, rr in originals.items()}
    families = ['vote', 'vote-dedup', 'A-sum', 'A-sum-dedup', 'A-mean', 'A-mean-dedup']
    a_families = list(families)
    b_families = ['B-sum', 'B-sum-dedup', 'B-mean', 'B-mean-dedup', 'AB-sum', 'AB-sum-dedup', 'AB-mean', 'AB-mean-dedup']
    if has_b: families += b_families
    features, labels = {}, {}
    for k in [8, 16, 32]:
        features[k] = {s: D.pack(rows[s], 'pure', k) for s in rows}
        for s, a in features[k].items():
            # Count dedup votes and apply the same vote tie hierarchy AFTER dedup.
            # Probability-dedup ties keep the historical pack ordering.
            picks = []
            for row in rows[s]:
                groups = D.clusters(row, D.indices('pure', k))
                order = {id(c): i for i, c in enumerate(row['flat'])}
                def key(cc):
                    unique = {}
                    for c in cc:
                        if c['code'] not in unique or c['mean_logprob'] > unique[c['code']]['mean_logprob']:
                            unique[c['code']] = c
                    kept = list(unique.values())
                    greedy = row['flat'][0]['execution']
                    return (len(kept), greedy['status'] == 'numeric' and any(D.near(c['execution']['value'], greedy['value']) for c in kept),
                        sum(c['mean_logprob'] for c in kept)/len(kept), -min(order[id(c)] for c in kept))
                picks.append(max(range(len(groups)), key=lambda i: key(groups[i])) if groups else 0)
            a['dedup_vote_picks'] = np.array(picks)
            if has_b:
                a['B_weights'] = [[None] * a['n'].shape[1] for _ in rows[s]]
                for i, row in enumerate(rows[s]):
                    for j, cc in enumerate(D.clusters(row, D.indices('pure', k))):
                        kept = {}
                        for c in cc:
                            if c['code'] not in kept or c['mean_logprob'] > kept[c['code']]['mean_logprob']:
                                kept[c['code']] = c
                        def weights(candidates):
                            return {name: np.array([c['mean_logprob'] * c['length'] if name == 'a_sum' else
                                c['mean_logprob'] if name == 'a_mean' else c['B_score']['sum_logprob'] if name == 'b_sum' else
                                c['B_score']['mean_logprob'] for c in candidates]) for name in ['a_sum', 'a_mean', 'b_sum', 'b_mean']}
                        a['B_weights'][i][j] = (weights(cc), weights(list(kept.values())))
        labels[k] = {s: D.correct_matrix(a, refs[s]) for s, a in features[k].items()}
        print('SELECTION packed', k, flush=True)
    def scores(a, family, cfg):
        if family.startswith(('B-', 'AB-')):
            out = np.full(a['n'].shape, -np.inf)
            dedup = int(family.endswith('-dedup'))
            for i, groups in enumerate(a['B_weights']):
                for j, pair in enumerate(groups):
                    if pair is None: continue
                    w = pair[dedup]
                    if family.startswith('B-'):
                        values = w['b_mean'] / cfg['tau'] if 'mean' in family else w['b_sum']
                    else:
                        suffix = 'mean' if 'mean' in family else 'sum'
                        values = w['a_' + suffix] + cfg['lam'] * w['b_' + suffix]
                    out[i, j] = D.logsumexp(values)
            return out
        field = {'vote': 'n', 'vote-dedup': 'unique', 'A-sum': 'sum', 'A-sum-dedup': 'sum-dedup'}.get(family)
        if field is None: field = f"mean-{'dedup-' if family.endswith('dedup') else ''}{cfg['tau']}"
        return np.where(a['valid'], a[field], -np.inf)
    def picks(a, family, cfg):
        return a['dedup_vote_picks'] if family == 'vote-dedup' else scores(a, family, cfg).argmax(1)
    def complexity(cfg):
        if 'lam' in cfg: return (int(cfg['lam'] != 1.), abs(cfg['lam'] - 1.))
        return D.complexity(cfg)
    if has_b:
        for pp in features.values():
            for a in pp.values():
                for suffix in ['', '-dedup']:
                    assert np.allclose(scores(a, 'AB-sum'+suffix, {'lam': 0.}), scores(a, 'A-sum'+suffix, {}))
    chosen, searches = {}, {}
    for family in families:
        grid = [dict(lam=v) for v in MIX] if family.startswith('AB-') else [dict(tau=t) for t in D.TAUS] if 'mean' in family else [{}]
        search = [dict(parameters=c, matches=int(D.flags(labels[16]['heldout4k'], picks(features[16]['heldout4k'], family, c))[tm].sum()),
                       complexity=list(complexity(c)), order=j) for j, c in enumerate(grid)]
        best = min(search, key=lambda x: (-x['matches'], *x['complexity'], x['order']))
        chosen[family], searches[family] = best['parameters'], search
    def winner_from(ff):
        return min(ff, key=lambda f: (-max(x['matches'] for x in searches[f]), *complexity(chosen[f]), families.index(f)))
    a_winner = winner_from(a_families)
    winner = winner_from(families)
    b_winner = winner_from(b_families) if has_b else None
    write(OUT / 'ensemble-frozen-choices.json', dict(winner=winner, parameters=chosen, search=searches))
    result = dict(protocol=protocol, split=previous['split'], sources=sources, search=searches, selected_parameters=chosen,
        python=platform.python_version(), packages={name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'tokenizers']},
        tune_winner=winner, A_only_tune_winner=a_winner, ensemble_tune_winner=b_winner,
        B_status='complete CPU FP32 on every requested pool; S3 GPU scores remain missing' if has_b else 'missing; B/AB families pending, no ensemble conclusion',
        B_CPU_scoring=b_summary if has_b else None, B_score_receipts_sha256=json_hash(b_receipts) if has_b else None,
        lambda_tie_rule='default1, then abs(lambda-1), then grid order; zero included', results={}, oracle_gap={}, reranker_potential={})
    result['extension_protocol_sha256'] = sha_file(frozen)
    greedy = {s: np.array([r['flat'][0]['execution']['status'] == ref['status'] == 'numeric' and
        D.near(r['flat'][0]['execution']['value'], ref['value']) for r, ref in zip(rows[s], refs[s])]) for s in rows}
    result['greedy'] = dict(tune_matches=int(greedy['heldout4k'][tm].sum()),
        eval=D.metrics(greedy['heldout4k'], greedy['heldout4k'], em, lengths['heldout4k']),
        old991=D.metrics(greedy['old1024'], greedy['old1024'], om, lengths['old1024']))
    assert int(greedy['heldout4k'].sum()) == 2538 and int(greedy['old1024'][om].sum()) == 634
    for k in [8, 16, 32]:
        pp, ll = features[k], labels[k]
        baseline = {s: D.flags(ll[s], scores(a, 'vote', {}).argmax(1)) for s, a in pp.items()}
        assert int(baseline['heldout4k'].sum()) == previous['results'][f'pure{k}']['vote']['tune_matches'] + previous['results'][f'pure{k}']['vote']['eval']['matches']
        result['results'][str(k)] = {}
        for family in families:
            cfg = chosen[family]
            flags = {s: D.flags(ll[s], picks(a, family, cfg)) for s, a in pp.items()}
            result['results'][str(k)][family] = dict(parameters=cfg, tune_matches=int(flags['heldout4k'][tm].sum()),
                parameter_anchor='k16/tune', free_parameters=len(cfg),
                eval=D.metrics(flags['heldout4k'], baseline['heldout4k'], em, lengths['heldout4k']),
                old991=D.metrics(flags['old1024'], baseline['old1024'], om, lengths['old1024']))
        if not has_b:
            for family in b_families: result['results'][str(k)][family] = dict(status='pending: no saved B scores')
        result['oracle_gap'][str(k)] = {}
        if has_b: result['reranker_potential'][str(k)] = {}
        for name, s, mask in [('eval', 'heldout4k', em), ('old991', 'old1024', om)]:
            potential = baseline[s] | (ll[s] & (pp[s]['n'] >= 2)).any(1)
            reserve = int((potential[mask] & ~baseline[s][mask]).sum())
            gain = result['results'][str(k)][a_winner][name]['matches'] - int(baseline[s][mask].sum())
            ensemble_gain = result['results'][str(k)][b_winner][name]['matches'] - int(baseline[s][mask].sum()) if has_b else None
            result['oracle_gap'][str(k)][name] = dict(vote_matches=int(baseline[s][mask].sum()), ideal_repair_matches=int(potential[mask].sum()),
                reserve_matches=reserve, reserve_pp=100*reserve/int(mask.sum()), A_only_winner_net_gain=gain,
                A_only_fraction_closed=gain/reserve if reserve else None, ensemble_net_gain=ensemble_gain,
                ensemble_fraction_closed=ensemble_gain/reserve if has_b and reserve else None)
            if has_b:
                result['reranker_potential'][str(k)][name] = D.potential(pp[s], ll[s], picks(pp[s], b_winner, chosen[b_winner]),
                    scores(pp[s], b_winner, chosen[b_winner]), rows[s], refs[s], mask)
    old_rule = previous['winners']['pure16']
    result['gap_against_previous_tune_winner'] = {}
    for name, s, mask in [('eval', 'heldout4k', em), ('old991', 'old1024', om)]:
        a = features[16][s]; ll = labels[16][s]
        old_flags = D.flags(ll, D.choose(a, old_rule['family'], old_rule['parameters']))
        assert int(old_flags[mask].sum()) == previous['results']['pure16'][old_rule['family']][name]['matches']
        new_flags = D.flags(ll, picks(a, a_winner, chosen[a_winner]))
        reserve = int(((ll & (a['n'] >= 2)).any(1) & ~old_flags)[mask].sum())
        gain = int(new_flags[mask].sum() - old_flags[mask].sum())
        ensemble_flags = D.flags(ll, picks(a, b_winner, chosen[b_winner])) if has_b else None
        ensemble_gain = int(ensemble_flags[mask].sum() - old_flags[mask].sum()) if has_b else None
        repairable = (ll & (a['n'] >= 2)).any(1) & ~old_flags
        result['gap_against_previous_tune_winner'][name] = dict(previous_rule=old_rule,
            reserve_matches=reserve, reserve_pp=100*reserve/int(mask.sum()), incremental_net_gain=gain,
            A_only_fraction_closed=gain/reserve, ensemble_incremental_net_gain=ensemble_gain,
            ensemble_fraction_closed=ensemble_gain/reserve if has_b else None,
            ensemble_recovered_repairable_cases=int((ensemble_flags & repairable)[mask].sum()) if has_b else None,
            ensemble_gains_outside_repeated_reserve=int((ensemble_flags & ~old_flags & ~repairable)[mask].sum()) if has_b else None,
            ensemble_lost_previously_correct=int((old_flags & ~ensemble_flags)[mask].sum()) if has_b else None,
            ensemble_McNemar=D.metrics(ensemble_flags, old_flags, mask, lengths[s]) if has_b else None,
            McNemar=D.metrics(new_flags, old_flags, mask, lengths[s]))
    result['decision'] = dict(status='no_convincing_gain', selected_rule=winner, parameters=chosen[winner],
        interpretation='B-mean improves vote slightly but shows no net gain over previous tune winner; no eval-based replacement',
        production='keep previous A/greedy and voting controls; do not add B scoring on this evidence',
        limitations='exploratory McNemar, reused heldout, CPU FP32; full T4 timer and fresh not measured') if has_b else 'A-only tune winner reported without substituting for unmeasured A/B ensemble; B gate pending'
    result['QA'] = dict(previous_hash_split_exact=True, historical_votes_exact=True, no_eval_or_old_parameter_selection=True,
        source_SHA_checked=True, nontransitive_components={str(k): {s: a['nontransitive_components'] for s, a in pp.items()} for k, pp in features.items()})
    result['timing'] = dict(CPU_selection_seconds=time.perf_counter()-started, execution_reused=True,
        B_scoring_seconds=b_summary['seconds'] if has_b else None,
        B_timer='forward and chunk writes; source parse/tokenization and model load excluded', production_T4_timer_verified=False)
    if has_b:
        result['B_CPU_scoring_script_sha256'] = sha_file(WORK / 'research/s3-score-b-cpu.py')
        result['QA']['B_all_requested_candidates_codeSHA_ID_sum_mean_complete'] = True
        result['QA']['AB_lambda0_equal_A_sum'] = True
    assert int(labels[16]['heldout4k'].any(1).sum()) == 3314
    assert int(labels[32]['heldout4k'].any(1).sum()) == 3468
    write(WORK / 'results/32-candidate-ensemble-cpu-summary.json', result)
    print('SELECTION WINNERS', dict(overall=winner, A_only=a_winner, ensemble=b_winner), chosen[winner], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['audit', 'selection'])
    args = parser.parse_args()
    audit() if args.phase == 'audit' else selection()
