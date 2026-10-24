#!/usr/bin/env python3
"""Audit pinned S1 artifacts and replay every saved program; no training/GPU."""
import argparse
import collections
import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'kaggle/src'))
from exp26_common import json_hash, read_json, sha_file, write_json
from exp26_fast_exec import run_chunks
import exp26_exec as numeric


def paired(score, control):
    assert score['eligible_indices'] == control['eligible_indices']
    a, b = set(score['match_indices']), set(control['match_indices'])
    gains, losses = len(a - b), len(b - a)
    n = gains + losses
    p = min(1., 2 * sum(math.comb(n, i) for i in range(min(gains, losses) + 1)) / (1 << n)) if n else 1.
    return dict(gains=gains, losses=losses, exact_two_sided_p=p,
                delta_percentage_points=100 * (gains - losses) / score['denominator'])


def same(a, b):
    return a['status'] == b['status'] and (a['status'] != 'numeric' or
        math.isclose(a['value'], b['value'], rel_tol=1e-6, abs_tol=1e-6))


def audit(root, output):
    from tokenizers import Tokenizer
    started = time.perf_counter()
    report = dict(date='2026-10-11', new_training_updates=0, new_GPU_runs=0,
                  root=str(root.relative_to(WORK)), checks={}, execution_differences={})
    receipt = read_json(root / 's1-archive-receipt.json')
    archive = root / receipt['file']
    assert archive.stat().st_size == receipt['bytes'] and sha_file(archive) == receipt['sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read('zip-manifest.json'))
        for record in manifest['files']:
            p = root / record['file']
            assert p.stat().st_size == record['bytes'] and sha_file(p) == record['sha256'], record['file']
            with z.open(record['file']) as f:
                h = hashlib.sha256()
                for chunk in iter(lambda: f.read(4 * 1024 ** 2), b''):
                    h.update(chunk)
            assert h.hexdigest() == record['sha256']
    report['archive'] = dict(receipt, manifest_files=len(manifest['files']), CRC='passed', all_SHA_bytes='passed')
    report['session'] = read_json(root / 's1-summary.json')
    report['checks']['archive_and_unpacked_members'] = 'passed'
    starter = WORK / 'results/checkpoints/26b-training-inputs-d0-initial-and-evaluation.zip'
    with zipfile.ZipFile(starter) as z:
        sets = {s: json.loads(z.read(s + '.json')) for s in ('heldout4k', 'old1024')}
        test = json.loads(z.read('test.json'))
        tok = Tokenizer.from_str(z.read('tokenizer.json').decode())
        input_identity = read_json(root / 'gpu0-a26/input-identity.json')
        for name, digest in input_identity['files'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    refs = {s: read_json(root / ('gpu0-a26/references/' + s + '-reference-execution.json')) for s in sets}
    lengths = {s: [len(e.ids) for e in tok.encode_batch([r['code'] for r in rows])] for s, rows in sets.items()}
    bounds = {'<60': lambda n: n < 60, '60–90': lambda n: 60 <= n < 90,
              '90–120': lambda n: 90 <= n < 120, '≥120': lambda n: n >= 120}
    codes = [r['code'] for rows in sets.values() for r in rows]
    saved = []
    metrics = {}
    for mp in sorted((root / 'gpu0-a26').rglob('metrics.json')):
        m = read_json(mp)
        metrics[mp.parent.relative_to(root).as_posix()] = m
        for split, rows in sets.items():
            d = m['splits'][split]
            gens = read_json(mp.parent / (split + '-generations.json'))
            execution = read_json(mp.parent / (split + '-execution.json'))
            answers = [json.loads(l) for l in (mp.parent / (split + '-answers.jsonl')).read_text().splitlines()]
            assert len(gens) == len(rows) == len(answers) == len(execution['rows'])
            assert sha_file(mp.parent / (split + '-answers.jsonl')) == d['generation_time']['answers_sha256']
            for i, (g, r, a) in enumerate(zip(gens, rows, answers)):
                assert g['id'] == a['id'] == i and g['question'] == a['question'] == r['question']
                assert a['code'] == g['generated_code'] == tok.decode([t-3 for t in g['generated_token_ids'] if t >= 3], skip_special_tokens=False)
                assert len(g['generated_token_ids']) <= 512 and g['eos_emitted'] == (2 in g['generated_token_ids'])
            score = numeric.compare_exec(refs[split], execution)
            assert score == d['exec']
            eligible, matched = set(score['eligible_indices']), set(score['match_indices'])
            for label, predicate in bounds.items():
                indices = {i for i in eligible if predicate(lengths[split][i])}
                b = d['length_accuracy'][label]
                assert b['denominator'] == len(indices) and b['matches'] == len(indices & matched)
            assert d['CE']['useful_target_tokens'] == sum(lengths[split]) + len(rows)
            tag = mp.parent.relative_to(root).as_posix() + '/' + split
            saved.append((tag, split, gens, execution, d))
            codes.extend(g['generated_code'] for g in gens)
    report['checks']['input_SHA_order_tokens_scores_CE_denominators_length_bins'] = 'passed'
    # Saved test programs have no gold; replay their numeric/status results only.
    plain_tests = []
    for p in sorted(root.rglob('test-generations.json')):
        gens = read_json(p)
        assert len(gens) == 1919
        for i, (g, t) in enumerate(zip(gens, test)):
            assert g['id'] == i and g['question'] == t['question']
        plain_tests.append((p.relative_to(root).as_posix(), gens))
        codes.extend(g['generated_code'] for g in gens)
    grids = []
    for directory in sorted((root / 'gpu0-exp27').rglob('candidates')):
        paths = sorted(p for p in directory.glob('batch-*.json') if not p.name.endswith('.sha.json'))
        if not paths:
            continue
        rows = []
        for p in paths:
            assert sha_file(p) == read_json(p.with_suffix('.sha.json'))['sha256']
            rows.extend(read_json(p))
        rows.sort(key=lambda r: r['id'])
        assert [r['id'] for r in rows] == list(range(len(rows)))
        flat = [c for r in rows for c in [r['greedy']] + [c for group in r['samples'].values() for c in group]]
        execution_dir = directory.parent / 'execution'
        assert json_hash([c['code'] for c in flat]) == read_json(execution_dir / 'identity.json')['codes_sha256']
        execution = []
        for p in sorted(execution_dir.glob('chunk-*.json')):
            assert sha_file(p) == read_json(p.with_suffix('.sha'))['sha256']
            execution.extend(read_json(p))
        assert len(flat) == len(execution)
        for c, e in zip(flat, execution):
            c['execution'] = e
        grids.append((directory.parent.relative_to(root).as_posix(), rows, flat, execution))
        codes.extend(c['code'] for c in flat)
    report['checks']['candidate_batches_chunks_SHA_flat_order'] = 'passed'
    unique = list(dict.fromkeys(codes))
    print('CPU replay:', len(codes), 'rows /', len(unique), 'unique programs', flush=True)
    def progress(done, total, seconds):
        if done % 20000 == 0 or done == total:
            print('CPU', done, '/', total, 'seconds', round(seconds, 1), flush=True)
    replay = run_chunks(unique, workers=2, chunk_rows=2000, progress=progress)
    lookup = dict(zip(unique, replay['rows']))
    report['CPU_execution'] = dict(replay['summary'], total_saved_program_rows=len(codes), unique_programs=len(unique))
    newrefs = {s: [lookup[r['code']] for r in rows] for s, rows in sets.items()}
    report['reference_replay'] = {}
    for s in sets:
        diffs = [i for i, (a, b) in enumerate(zip(newrefs[s], refs[s]['rows'])) if not same(a, b)]
        report['reference_replay'][s] = dict(numeric=sum(r['status']=='numeric' for r in newrefs[s]), different_indices=diffs)
    report['greedy_replay'] = {}
    for tag, split, gens, execution, d in saved:
        fresh = [lookup[g['generated_code']] for g in gens]
        score = numeric.compare_exec(newrefs[split], fresh)
        diff = sorted(set(score['match_indices']) ^ set(d['exec']['match_indices']))
        statuses = [i for i, (a, b) in enumerate(zip(fresh, execution['rows'])) if not same(a, b)]
        report['greedy_replay'][tag] = dict(matches=score['matches'], denominator=score['denominator'],
                                         different_match_indices=diff, different_execution_indices=statuses)
    # Use the same registered voting rules, now with independently replayed execution.
    import exp27_decode as decoding
    summary27 = read_json(root / 'gpu0-exp27/27-summary.json')
    final = read_json(root / 'gpu0-a26/points/p050000/evaluation/metrics.json')
    report['voting_replay'] = {}
    report['candidate_execution_replay'] = {}
    for tag, rows, flat, execution in grids:
        differences = [i for i, (c, e) in enumerate(zip(flat, execution)) if not same(lookup[c['code']], e)]
        report['candidate_execution_replay'][tag] = dict(programs=len(flat), different_execution_count=len(differences),
                                                        different_indices= differences[:100])
        for c in flat:
            c['execution'] = lookup[c['code']]
        split = tag.split('/')[-1]
        if split in sets:
            calculated = decoding.score_configs(rows, newrefs[split], sets[split], tok, final['splits'][split]['exec'])
            for calc, original in zip(calculated, summary27['splits'][split]['configs']):
                diff = sorted(set(calc['exec']['match_indices']) ^ set(original['exec']['match_indices']))
                assert calc['config'] == original['config']
                report['voting_replay'][split+'/'+calc['config']['name']] = dict(matches=calc['exec']['matches'],
                    denominator=calc['exec']['denominator'], different_match_indices=diff)
                p = paired(original['exec'], final['splits'][split]['exec'])
                old = original['McNemar_against_greedy']
                assert p['gains']==old['gains'] and p['losses']==old['losses'] and math.isclose(p['exact_two_sided_p'], old['exact_two_sided_p'], rel_tol=1e-10, abs_tol=1e-300)
        else:
            cfg = read_json(root / tag / 'timing.json')['config']
            answers = [json.loads(l) for l in (root / tag / 'test-answers.jsonl').read_text().splitlines()]
            changed = [i for i, (r, a) in enumerate(zip(rows, answers)) if decoding.choose(r, cfg)['code'] != a['code']]
            report['voting_replay'][tag] = dict(rows=len(rows), different_selected_program_indices=changed)
    report['test_numeric_replay'] = {tag:dict(collections.Counter(lookup[g['generated_code']]['status'] for g in gens)) for tag, gens in plain_tests}
    checks = list(report['greedy_replay'].values()) + list(report['voting_replay'].values())
    report['all_match_sets_and_test_winners_exact'] = all(not r.get('different_match_indices') and not r.get('different_selected_program_indices') for r in checks)
    report['all_reference_values_and_finite_indices_exact'] = all(not r['different_indices'] for r in report['reference_replay'].values())
    report['wall_seconds'] = time.perf_counter()-started
    report['status'] = 'passed' if report['all_match_sets_and_test_winners_exact'] and report['all_reference_values_and_finite_indices_exact'] else 'differences; see replay sections'
    write_json(output, report)
    print('AUDIT', report['status'], 'seconds', round(report['wall_seconds'], 1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=WORK / 'results/kaggle-runs/nlp-hw02-kaggle-s1/1')
    p.add_argument('--output', type=Path, default=WORK / 'checks/kaggle-s1-v1-cpu-audit.json')
    args = p.parse_args()
    audit(args.root, args.output)
