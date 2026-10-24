"""CPU-only D1 evidence preparation, diagnostics and deterministic original-pair orders.

prepare: join existing duplicate executions by raw (shard,row), never by position
in a filtered array. analyze: untested/non-numeric/capped rows are NEVER removed.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK/'kaggle/src'))
from kaggle_io import sha, write
from d1_data import (CACHE_SHA, ROWS, MODELS, WEIGHTS, EDGES, record, publish, save_npz, verify_receipt,
                     cluster, close, ranked_candidates, filtered_order)

DEFAULT_CACHE = WORK/'results/kaggle-runs/nlp-hw02-compact-cache-verified/2/compact-cache'


def checked(path, expected):
    assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], str(path)


def prepare(cache_root, destination):
    """Use the complete audited train census. No model forward or new execution."""
    cache_root, destination = Path(cache_root), Path(destination); destination.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    clean_path = WORK/'results/clean-cooldown/train-analysis.json'
    clean = json.loads(clean_path.read_text()); manifest = json.loads((cache_root/'compact-manifest.json').read_text())
    assert sha(cache_root/'compact-manifest.json') == clean['cache_manifest_sha256'] == CACHE_SHA
    assert manifest['rows'] == ROWS and [r['shard'] for r in manifest['shards']] == list(range(60))
    inputs = clean['execution_reuse']['inputs']
    for rec in inputs: checked(WORK/rec['file'], rec)
    local = WORK/'research/outputs/data-ceiling'
    with np.load(local/'normal-duplicate-members.npz', allow_pickle=False) as z:
        raw_ids, groups = z['row'], z['group']
    with np.load(local/'noise-values.npz', allow_pickle=False) as z:
        assert np.array_equal(raw_ids, z['rows']); values, statuses = z['values'], z['statuses']
    inventory = json.loads((WORK/'checks/26-pinned-train-shard-inventory.json').read_text())
    census_path = local/'census.json'; census = json.loads(census_path.read_text())
    assert census['revision'] == inventory['revision']
    receipt_records = clean['execution_reuse']['source_execution_receipts']
    indices, positions, original, sources, targets = [], [], [], [], []
    raw_offset = cache_offset = 0
    original_available = True
    for rec, spec, exrec, raw_spec in zip(manifest['shards'], inventory['files'], receipt_records, census['shards']):
        folder = cache_root/rec['folder']
        for name in ('provenance-rows.npy', 'source-offsets.npy', 'target-offsets.npy', 'question-sha256.npy'):
            checked(folder/name, next(f for f in rec['files'] if f['file'] == name))
        ids = np.load(folder/'provenance-rows.npy', allow_pickle=False)
        # Raw Parquet was purged after the census. Reuse its audited row counts,
        # cross-check pinned raw SHA and the complete retained per-row hash table.
        assert raw_spec['shard'] == spec['shard'] == rec['shard'] and raw_spec['sha256'] == spec['sha256']
        raw_rows = raw_spec['rows']
        assert len(np.load(local/f'hashes-{rec["shard"]:02d}.npy', mmap_mode='r', allow_pickle=False)) == raw_rows
        checked(WORK/exrec['file'], exrec)
        executions = json.loads((WORK/exrec['file']).read_text())['rows']
        lo, hi = np.searchsorted(raw_ids, [raw_offset, raw_offset+raw_rows])
        assert len(executions) == hi-lo
        for j, execution in enumerate(executions, lo):
            assert (execution['status'] == 'numeric') == (statuses[j] == 0)
            if statuses[j] == 0: assert execution['value'] == values[j] and np.isfinite(values[j])
        pos = np.searchsorted(raw_ids, ids.astype(np.int64)+raw_offset)
        found = pos < len(raw_ids)
        jj = np.flatnonzero(found)
        found[jj] = raw_ids[pos[jj]] == ids[jj].astype(np.int64)+raw_offset
        assert np.all(statuses[pos[found]] == 0)
        indices.append(np.flatnonzero(found).astype(np.uint32)+cache_offset); positions.append(pos[found])
        sources.append(np.diff(np.load(folder/'source-offsets.npy', allow_pickle=False)).astype(np.uint16))
        targets.append(np.diff(np.load(folder/'target-offsets.npy', allow_pickle=False)).astype(np.uint16))
        raw_root = (WORK/'kaggle/staging/drive-merged/26-token-cache' if rec['shard'] < 42 else
                    WORK/'results/kaggle-runs/nlp-hw02-26c-cpu-cache/2/26c-token-cache')
        expected = next(f for f in rec['original_files'] if f['file'] == 'target-offsets.npy')
        raw_lengths = raw_root/rec['folder']/'target-offsets.npy'
        if raw_lengths.exists():
            checked(raw_lengths, expected)
            original.append(np.diff(np.load(raw_lengths, allow_pickle=False)).astype(np.uint16))
        else: original_available = False
        cache_offset += len(ids); raw_offset += raw_rows
    row = np.concatenate(indices); pos = np.concatenate(positions)
    assert cache_offset == ROWS and len(row) == clean['execution_reuse']['numeric_duplicate_rows']
    group = groups[pos]; reference = values[pos]
    size = np.ones(len(row), np.uint16); conflict = np.zeros(len(row), bool)
    minority = np.zeros(len(row), bool); majority = np.full(len(row), np.nan)
    order = np.argsort(group, kind='stable'); gg = group[order]
    bounds = np.r_[0, np.flatnonzero(gg[1:] != gg[:-1])+1, len(gg)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        ii = order[a:b]; labels, counts, reps = cluster(reference[ii])
        size[ii] = len(ii); conflict[ii] = len(counts) > 1
        winners = np.flatnonzero(counts == counts.max())
        if len(ii) >= 3 and len(winners) == 1 and 3*counts[winners[0]] >= 2*len(ii):
            majority[ii] = reps[winners[0]]; minority[ii] = labels != winners[0]
    assert np.all(row[1:] > row[:-1])
    assert int((size >= 2).sum()) == clean['census']['duplicate_rows']
    assert int(conflict.sum()) == clean['census']['conflicting_rows']
    target = destination/'seed-evidence.npz'
    optional = dict(original_length=np.concatenate(original)) if original_available else {}
    save_npz(target, row_id=row, reference=reference, group=group.astype(np.uint32), group_size=size,
        conflicting=conflict, minority=minority, majority=majority,
        source_length=np.concatenate(sources), target_length=np.concatenate(targets),
        protected=np.asarray(clean['protected']['removed_cache_indices'], dtype=np.uint32), **optional)
    publish(destination/'seed.receipt.json', [target], phase='seed', cache_manifest_sha256=CACHE_SHA,
        rows=ROWS, duplicate_reference_rows=len(row), duplicate_rows=int((size >= 2).sum()),
        conflicting_rows=int(conflict.sum()), train_analysis_sha256=sha(clean_path), inputs=inputs,
        execution_receipt_sha256=[r['sha256'] for r in receipt_records],
        original_lengths_available=original_available, census_sha256=sha(census_path),
        raw_row_count_source='previous full census + pinned raw SHA + retained hash-table shapes; no raw reread',
        seconds=time.monotonic()-start)
    write(WORK/'kaggle/d1-seed.json', dict(**record(target), cache_manifest_sha256=CACHE_SHA,
          rows=ROWS, duplicate_reference_rows=len(row), receipt_sha256=sha(destination/'seed.receipt.json')))
    print('D1 seed prepared:', len(row), 'cached references;', target.stat().st_size, 'bytes')


def filter_masks(main, long, reference, main_status, long_status, ref_status, main_gen, long_gen,
                 high_loss, group_size, minority, majority):
    eligible = ((main_status == 'numeric') & (long_status == 'numeric') & (ref_status == 'numeric') &
                (main_gen == 'eos') & (long_gen == 'eos') &
                np.isfinite(main) & np.isfinite(long) & np.isfinite(reference))
    a = eligible & close(main, long) & ~close(main, reference) & ~close(long, reference)
    b = a & high_loss
    c = b & (group_size >= 3) & minority & np.isfinite(majority) & close(main, majority) & close(long, majority)
    return eligible, {'A': a, 'B': b, 'C': c}


def stats(mask, eligible, filters):
    n = int(mask.sum()); good = int((mask & eligible).sum())
    return dict(rows=n, numeric_complete=good, filters={k: dict(rows=int((mask & v).sum()),
        fraction_of_rows=float((mask & v).sum()/n) if n else None,
        fraction_of_numeric_complete=float((mask & v).sum()/good) if good else None) for k, v in filters.items()})


def read_generation(root):
    records = {}; receipts = []
    for receipt in sorted(root.glob('*.receipt.json')):
        meta = verify_receipt(receipt)
        if meta.get('phase') != 'generation': continue
        receipts.append(dict(file=receipt.name, sha256=sha(receipt)))
        with np.load(root/meta['files'][0]['file'], allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        for j, row in enumerate(arrays['row_id']):
            row = int(row); assert row not in records, 'Duplicate D1 generation row'
            records[row] = {k: a[j] for k, a in arrays.items() if k != 'row_id'}
    return records, receipts


def analyze(source, output, allow_partial=False):
    source, output = Path(source), Path(output); output.mkdir(parents=True, exist_ok=True)
    root = source/'d1' if (source/'d1').is_dir() else source
    seed_meta = verify_receipt(root/'seed.receipt.json')
    assert seed_meta['cache_manifest_sha256'] == CACHE_SHA
    with np.load(root/'seed-evidence.npz', allow_pickle=False) as z: seed = {k: z[k] for k in z.files}
    n = len(seed['target_length']); assert n == ROWS or allow_partial
    scope = verify_receipt(root/'scored-row-ids.receipt.json')
    scored = np.load(root/'scored-row-ids.npy', allow_pickle=False)
    assert scored.dtype == np.uint32 and scored.ndim == 1
    assert len(np.unique(scored)) == len(scored) and int(scored.max()) < n
    if not allow_partial: assert len(scored) == n and scope['all_train']
    selection_meta = verify_receipt(root/'selection.receipt.json')
    assert selection_meta['cache_manifest_sha256'] == CACHE_SHA
    with np.load(root/'selection.npz', allow_pickle=False) as z: selected = z['row_id']
    assert selected.ndim == 1 and len(selected) and len(np.unique(selected)) == len(selected)
    assert int(selected.max()) < n
    assert np.isin(selected, scored).all()
    ce = {}; loss_metadata = {}; generations = {}; gen_receipts = {}
    single_ce = not (root/'C-main/loss/complete.receipt.json').exists()
    if single_ce:
        loss_root = root/'C-long2x/loss'
        loss_path = loss_root/('complete.receipt.json' if (loss_root/'complete.receipt.json').exists() else 'partial.receipt.json')
        meta = verify_receipt(loss_path)
        assert meta['model'] == 'C-long2x' and meta['weight_sha256'] == WEIGHTS['C-long2x']
        if not allow_partial: assert meta['complete']
        scores = np.load(loss_root/'ce.npy', mmap_mode='r', allow_pickle=False)
        assert scores.shape == (n,) and scores.dtype == np.float16
        scored = scored[np.isfinite(scores[scored])]
        assert len(scored) == meta['rows'] and np.all(scores[scored] >= 0)
        assert int((seed['target_length'][scored].astype(np.int64)+1).sum()) == meta['tokens']
        assert np.isin(selected, scored).all()
        loss_metadata['C-long2x'] = meta
        ce = {m:scores for m in MODELS}  # compatibility of filter implementation; no C-main CE was run
    for m in MODELS:
        identity = json.loads((root/m/'identity.json').read_text())
        assert identity['model'] == m and identity['weight_sha256'] == WEIGHTS[m]
        assert identity['cache_manifest_sha256'] == CACHE_SHA and identity['training'] is False
        if not single_ce:
            loss_metadata[m] = verify_receipt(root/m/'loss/complete.receipt.json')
            assert loss_metadata[m]['complete']
            ce[m] = np.load(root/m/'loss/ce.npy', mmap_mode='r', allow_pickle=False)
            lengths = np.load(root/m/'loss/length.npy', mmap_mode='r', allow_pickle=False)
            nll = np.load(root/m/'loss/nll.npy', mmap_mode='r', allow_pickle=False)
            assert ce[m].dtype == np.float16 and lengths.dtype == np.uint16 and nll.dtype == np.float32
            assert ce[m].shape == (n,) and np.array_equal(lengths[scored], seed['target_length'][scored]+1)
            assert nll.shape == (n,) and np.isfinite(nll[scored]).all() and np.all(nll[scored] >= 0)
            assert np.array_equal(ce[m][scored], (nll[scored]/lengths[scored]).astype(np.float16))
            assert abs(float(nll[scored].sum(dtype=np.float64))-loss_metadata[m]['sum_NLL']) < 1e-3
            assert int(lengths[scored].sum()) == loss_metadata[m]['tokens']
            assert loss_metadata[m]['rows'] == len(scored)
        generations[m], gen_receipts[m] = read_generation(root/m)
        assert set(generations[m]) <= set(map(int, selected))
    available = set(generations[MODELS[0]]) & set(generations[MODELS[1]])
    if not allow_partial: assert available == set(map(int, selected)), 'Incomplete paired generation; use --allow-partial explicitly'
    ids = np.asarray(sorted(available), dtype=np.uint32)
    assert len(ids), 'No paired results'
    data = {m: {k: np.asarray([generations[m][int(i)][k] for i in ids])
                for k in next(iter(generations[m].values()))} for m in MODELS}
    # B is stricter than the initial top8%: top2% in each full train length bin.
    ranked, score = ranked_candidates(ce[MODELS[0]][scored], ce[MODELS[1]][scored], seed['target_length'][scored], .02)
    high_loss = np.isin(ids, scored[np.concatenate(ranked)])
    mean_scores = (ce[MODELS[0]][ids].astype(np.float32)+ce[MODELS[1]][ids].astype(np.float32))*.5
    pos = np.searchsorted(seed['row_id'], ids); found = pos < len(seed['row_id'])
    jj = np.flatnonzero(found); found[jj] = seed['row_id'][pos[jj]] == ids[jj]
    group_size = np.ones(len(ids), np.uint16); conflict = np.zeros(len(ids), bool)
    minority = np.zeros(len(ids), bool); majority = np.full(len(ids), np.nan)
    for out, key in [(group_size, 'group_size'), (conflict, 'conflicting'), (minority, 'minority'), (majority, 'majority')]:
        out[found] = seed[key][pos[found]]
    a, b = data[MODELS[0]], data[MODELS[1]]
    eligible, masks = filter_masks(a['answer'], b['answer'], a['reference'], a['status'], b['status'],
        a['reference_status'], a['generation_status'], b['generation_status'], high_loss,
        group_size, minority, majority)
    report = dict(format='nlp-hw02-d1-analysis-v1', cache_manifest_sha256=CACHE_SHA, cache_rows=n,
        selected_rows=len(selected), paired_rows=len(ids), scored_rows=len(scored),
        partial=len(ids) != len(selected) or len(scored) != n or (single_ce and not meta['complete']),
        coverage_fraction=len(ids)/n, all_train_prevalence_identified=False,
        metrics=stats(np.ones(len(ids), bool), eligible, masks),
        rules={'A': 'both numeric EOS predictions agree and each differs from finite reference',
               'B': 'A + top2% C-long2x persisted CE within each compact length bin' if single_ce else
                    'A + top2% mean persisted CE within each compact length bin',
               'C': 'B + minority reference in duplicate n>=3, unique >=2/3 mode, both models match mode'},
        by_compact_length={}, by_original_length={}, by_duplicates={},
        loss=loss_metadata, score_policy='C-long2x only' if single_ce else 'mean of two CE', generation_receipts=gen_receipts,
        seed_sha256=sha(root/'seed-evidence.npz'), selection_sha256=sha(root/'selection.npz'),
        numerical_equivalence='symmetric math.isclose rel=abs=1e-6; anchored duplicate clusters',
        caveat='High-loss selected population only; agreement can be a shared model error. No reference rewriting.')
    report['fraction_of_selected'] = {k:float(v.sum()/len(selected)) for k,v in masks.items()}
    report['full_train_flagged_lower_bound'] = {k:float(v.sum()/n) for k,v in masks.items()}
    report['status_counts'] = {m:dict(Counter(map(str, data[m]['status']))) for m in MODELS}
    report['reference_status_counts'] = dict(Counter(map(str, a['reference_status'])))
    length_views = [('by_compact_length', EDGES, seed['target_length'])]
    if 'original_length' in seed: length_views.append(('by_original_length', (60, 90, 120), seed['original_length']))
    else: report['by_original_length'] = dict(available=False, reason='original offsets unavailable; compact bins reported explicitly')
    for key, edges, lengths in length_views:
        bins = np.searchsorted(edges, lengths, side='right')
        for bin_id in range(4):
            s = stats(bins[ids] == bin_id, eligible, masks)
            s['full_train_rows'] = int((bins == bin_id).sum())
            s['selected_rows'] = int((bins[selected] == bin_id).sum())
            report[key][str(bin_id)] = s
    for key, mask in [('singleton', group_size < 2), ('duplicate', group_size >= 2),
                      ('conflicting', conflict), ('minority', minority)]:
        report['by_duplicates'][key] = stats(mask, eligible, masks)
        report['by_duplicates'][key]['mean_CE'] = float(mean_scores[mask].mean()) if mask.any() else None
        bins = np.searchsorted(EDGES, seed['target_length'][ids], side='right')
        report['by_duplicates'][key]['by_compact_length'] = {str(b):stats(mask & (bins == b), eligible, masks) for b in range(4)}
    evidence_sha = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    orders = {}
    for rule, excluded in [('control', np.zeros(len(ids), bool)), *masks.items()]:
        keep = np.ones(n, bool); keep[seed['protected']] = False; keep[ids[excluded]] = False
        orders[rule] = filtered_order(keep, seed['source_length'], seed['target_length'], output/rule,
                                      CACHE_SHA, rule, evidence_sha)
    save_npz(output/'flags.npz', row_id=ids, eligible=eligible, mean_CE=mean_scores, **masks)
    report['orders'] = orders; write(output/'analysis.json', report)
    print(json.dumps(dict(paired_rows=len(ids), metrics=report['metrics'], orders={k:v['slots'] for k,v in orders.items()}), indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    prep.add_argument('--output', type=Path, default=WORK/'kaggle/staging/d1-live')
    analysis = sub.add_parser('analyze'); analysis.add_argument('--input', type=Path, required=True)
    analysis.add_argument('--output', type=Path, default=WORK/'results/d1-labels')
    analysis.add_argument('--allow-partial', action='store_true')
    args = p.parse_args()
    if args.command == 'prepare': prepare(args.cache, args.output)
    else: analyze(args.input, args.output, args.allow_partial)


if __name__ == '__main__': main()
