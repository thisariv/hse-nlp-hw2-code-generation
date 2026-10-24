"""Full CPU census and original-pair-only cooldown orders. No network or Torch.

Reuse the complete raw duplicate execution census, intersect by (shard, row)
with the pinned compact cache, and recluster AFTER filtering. Never use heldout
answers to select train rows. Temporary arrays live in RAM; only orders persist.
"""
import argparse
import ast
import hashlib
import json
import math
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

WORK = Path(__file__).resolve().parents[1]
LOCAL = WORK / 'research/outputs/data-ceiling'
CACHE = WORK / 'results/kaggle-runs/nlp-hw02-compact-cache-verified/2/compact-cache'
RAW = WORK / 'kaggle/staging/drive-merged/26-token-cache/raw'
INPUT = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
OUTPUT = WORK / 'results/clean-cooldown'
RESULT = WORK / 'results/clean-cooldown-analysis.json'
CACHE_SHA = '46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be'
SEED = 2026101304
BINS = ['<60', '60-90', '90-120', '>=120']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    return {'file': str(path.relative_to(WORK)), 'sha256': sha(path), 'bytes': path.stat().st_size}


def verify(path, expected):
    assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], path


def normal(q):
    return ' '.join(q.lower().split())


def clusters(values):
    """Stable raw-row first representative, same anchored isclose as noise.py.

    isclose is not transitive. Preserve the existing census convention rather
    than silently switching to rounded keys or transitive connected components.
    """
    reps, counts, labels = [], [], []
    for x in values:
        for i, representative in enumerate(reps):
            if math.isclose(x, representative, rel_tol=1e-6, abs_tol=1e-6):
                counts[i] += 1
                labels.append(i)
                break
        else:
            labels.append(len(reps))
            reps.append(float(x))
            counts.append(1)
    return np.asarray(labels), np.asarray(counts)


def select(values):
    labels, counts = clusters(values)
    top = counts.max()
    winners = np.flatnonzero(counts == top)
    strict = (len(winners) == 1 and 3 * top >= 2 * len(values))
    # R2: keep ties; they supply no evidence that any one mode is preferable.
    conservative = np.isin(labels, winners)
    return labels, counts, (labels == winners[0]) & strict, conservative


def histogram(lengths, mask=None):
    values = lengths if mask is None else lengths[mask]
    ids = np.searchsorted([60, 90, 120], values, side='right')
    counts = np.bincount(ids, minlength=4)
    return {'rows': len(values), 'mean_BPE': float(values.mean()),
            'bins': {name: {'rows': int(n), 'fraction': float(n / len(values))}
                     for name, n in zip(BINS, counts)}}


def load_function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def build_order(keep, source_lengths, target_lengths, rule, evidence):
    root = OUTPUT / rule
    root.mkdir(parents=True, exist_ok=True)
    order = np.flatnonzero(keep).astype(np.uint32)
    rng = np.random.default_rng(SEED)
    rng.shuffle(order)
    for lo in range(0, len(order), 4096):
        part = order[lo:lo + 4096]
        ns = source_lengths[part].astype(np.int32)
        nt = target_lengths[part].astype(np.int32)
        part = part[np.lexsort((nt, ns, ns + nt + 1))]
        full = len(part) // 128
        if full:
            part[:full * 128] = part[:full * 128].reshape(full, 128)[rng.permutation(full)].reshape(-1)
        order[lo:lo + len(part)] = part
    # Membership and uniqueness, not just min/max: each selected original row once.
    assert np.array_equal(np.sort(order), np.flatnonzero(keep).astype(np.uint32))
    np.save(root / 'order.npy', order, allow_pickle=False)
    f = record(root / 'order.npy')
    f['file'] = 'order.npy'
    meta = {'seed': SEED, 'rows': len(keep), 'slots': len(order), 'batch': 128,
            'chunk': 4096, 'bijection': False, 'selected_bijection': True,
            'cache_manifest_sha256': CACHE_SHA, 'file': f,
            'rule': rule, 'selection': evidence['rules'].get(rule, evidence['control']),
            'multiplicity': 'one slot per retained original (shard,row); no question reweighting',
            'answer_policy': 'existing train reference executions only; no new targets',
            'normalization': 'lower + collapse whitespace; no number masking',
            'answer_equivalence': 'stable first representative; math.isclose rel=abs=1e-6',
            'protected_exclusions': evidence['protected'],
            'analysis_sha256': hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
            'selected_cache_indices_sha256': hashlib.sha256(
                np.flatnonzero(keep).astype('<u4').tobytes()).hexdigest()}
    write(root / 'order.json', meta)
    # Invoke the existing L2 loader itself, without importing Torch or a worker.
    loader = load_function(WORK / 'kaggle/src/l2_sampler.py', 'load_order',
                           {'Path': Path, 'json': json, 'np': np, 'sha': sha})
    class CacheIdentity:
        def __len__(self):
            return len(keep)
    loaded, receipt = loader(root, CacheIdentity(), f['sha256'])
    assert np.array_equal(loaded, order) and receipt == meta
    split = load_function(WORK / 'kaggle/src/l2_sampler.py', 'fixed_split', {'hashlib': hashlib})()
    write(root / 'split.json', {k: {'ids': v, 'sha256': hashlib.sha256(
        json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()}
                              for k, v in split.items()})
    return {'folder': str(root.relative_to(WORK)), 'metadata': meta,
            'loader_verified': 'unmodified l2_sampler.load_order',
            'manifest_sha256': sha(root / 'order.json')}


def error_rate(rows, ids):
    n = len(ids)
    errors = sum(not rows[i]['correct'] for i in ids)
    if not n:
        return {'n': 0, 'errors': 0, 'error_rate': None}
    p = errors / n
    z = 1.96
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {'n': n, 'errors': errors, 'error_rate': p,
            'Wilson95': [center-half, center+half]}


def heldout_diagnostic(heldout, peers, train_peers):
    """A clearly separated raw-source proxy; those references are excluded train.

    Remove one occurrence of the actual heldout pair, if found, BEFORE labeling
    conflicts. A heldout reference cannot create its own diagnostic label.
    """
    labels, peer_counts = [], []
    own_removed = 0
    for row, siblings in zip(heldout, peers):
        own = next((j for j, r in enumerate(siblings) if r['own_pair']), None)
        if own is not None:
            siblings = siblings[:own] + siblings[own + 1:]
            own_removed += 1
        finite = [r['value'] for r in siblings if r['status'] == 0]
        peer_counts.append(len(finite))
        if len(finite) < 2:
            labels.append('insufficient')
        else:
            _, counts = clusters(finite)
            labels.append('conflicting' if len(counts) > 1 else 'unanimous')
    fixed_split = load_function(WORK / 'kaggle/src/l2_sampler.py', 'fixed_split', {'hashlib': hashlib})()
    train_labels = ['absent' if not v else 'singleton' if len(v) == 1 else
                    'conflicting' if len(clusters(v)[1]) > 1 else 'unanimous' for v in train_peers]
    report = {'exact_train_question_overlap': 0,
              'normalized_input_cache_overlap_questions': sum(bool(v) for v in train_peers),
              'normalized_safe_order_overlap_questions': 0,
              'actual_cache_join_labels': dict(Counter(train_labels)),
              'direct_train_conflict_test': 'not identifiable: no overlapping conflicting train groups' if
                  'conflicting' not in train_labels else 'tiny normalized overlap only; see actual_cache_join',
              'proxy': 'raw-source duplicate peers; all excluded from train; own heldout pair removed once',
              'own_heldout_pairs_removed': own_removed, 'label_counts': dict(Counter(labels)),
              'peer_numeric_count_histogram': dict(Counter(peer_counts)), 'models': {}}
    # Resample normalized questions as clusters, retaining both model outcomes.
    keys = [normal(r['question']) for r in heldout]
    for model in ['C-main', 'C-long2x']:
        path = WORK / f'results/kaggle-runs/nlp-hw02-kaggle-l2/1/model-{model}/points/l2-28800/evaluation/heldout4k-per-example.json'
        rows = read(path)
        assert len(rows) == 4096
        for i, (row, q) in enumerate(zip(rows, heldout)):
            assert row['id'] == i
            assert row['question_sha256'] == hashlib.sha256(q['question'].encode()).hexdigest()
            good = row['execution']['status'] == row['reference']['status'] == 'numeric' and math.isclose(
                row['execution']['value'], row['reference']['value'], rel_tol=1e-6, abs_tol=1e-6)
            assert row['correct'] == good
        subsets = {'heldout4k': list(range(4096)), **fixed_split}
        model_report = {'input': record(path), 'subsets': {},
                        'actual_cache_join': {label: error_rate(rows, [i for i, value in enumerate(train_labels) if value == label])
                                              for label in ['conflicting', 'unanimous', 'singleton', 'absent']}}
        for name, ids in subsets.items():
            by_label = {label: [i for i in ids if labels[i] == label]
                        for label in ['conflicting', 'unanimous', 'insufficient']}
            scores = {label: error_rate(rows, ii) for label, ii in by_label.items()}
            by_bin = {}
            for b, bin_name in enumerate(BINS):
                by_bin[bin_name] = {label: error_rate(rows, [i for i in ii if
                    np.searchsorted([60, 90, 120], rows[i]['original_bpe_length'], side='right') == b])
                                   for label, ii in by_label.items()}
            scores['by_original_length'] = by_bin
            comparable = by_label['conflicting'] + by_label['unanimous']
            if by_label['conflicting'] and by_label['unanimous']:
                scores['conflict_minus_unanimous_error_pp'] = 100 * (
                    scores['conflicting']['error_rate'] - scores['unanimous']['error_rate'])
                # Question-cluster bootstrap for the unpaired observational contrast.
                grouped = defaultdict(list)
                for i in comparable:
                    grouped[keys[i]].append(i)
                vectors = []
                for ii in grouped.values():
                    vectors.append([sum(labels[i] == 'conflicting' and not rows[i]['correct'] for i in ii),
                                    sum(labels[i] == 'conflicting' for i in ii),
                                    sum(labels[i] == 'unanimous' and not rows[i]['correct'] for i in ii),
                                    sum(labels[i] == 'unanimous' for i in ii)])
                vectors = np.asarray(vectors)
                rng = np.random.default_rng(SEED)
                deltas = []
                for _ in range(4000):
                    e1, n1, e0, n0 = vectors[rng.integers(0, len(vectors), len(vectors))].sum(axis=0)
                    if n1 and n0:
                        deltas.append(100 * (e1/n1 - e0/n0))
                scores['question_cluster_bootstrap95_pp'] = np.quantile(deltas, [.025, .975]).tolist()
                scores['bootstrap_question_clusters'] = len(vectors)
                eligible_bins = [v for v in by_bin.values() if v['conflicting']['n'] and v['unanimous']['n']]
                total = sum(v['conflicting']['n'] + v['unanimous']['n'] for v in eligible_bins)
                scores['length_standardized_delta_pp'] = sum(
                    (v['conflicting']['n'] + v['unanimous']['n']) / total * 100 *
                    (v['conflicting']['error_rate'] - v['unanimous']['error_rate']) for v in eligible_bins) if total else None
                scores['length_standardization_rows'] = total
            model_report['subsets'][name] = scores
        report['models'][model] = model_report
    write(OUTPUT / 'heldout-source-proxy.json', {'labels': labels, 'numeric_peer_counts': peer_counts,
                                                'policy': report['proxy']})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-orders', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = read(CACHE / 'compact-manifest.json')
    assert sha(CACHE / 'compact-manifest.json') == CACHE_SHA
    assert manifest['sealed'] and manifest['rows'] == 9725737
    assert [r['shard'] for r in manifest['shards']] == list(range(60))
    inventory = read(WORK / 'checks/26-pinned-train-shard-inventory.json')
    membership = np.load(LOCAL / 'normal-duplicate-members.npz', allow_pickle=False)
    noise = np.load(LOCAL / 'noise-values.npz', allow_pickle=False)
    raw_ids, groups = membership['row'], membership['group']
    values, statuses = noise['values'], noise['statuses']
    assert np.array_equal(raw_ids, noise['rows'])
    assert read(LOCAL / 'noise.json')['hash_text_equality_verified']
    protected_path = INPUT / 'protected-questions.json'
    protected_groups = read(protected_path)['groups']
    protected = {q for qq in protected_groups.values() for q in qq}
    normalized_protected = {normal(q) for q in protected}
    protected_normal_sets = {k: {normal(q) for q in qq} for k, qq in protected_groups.items()}
    protected_overlap_counts = Counter()
    safe_keep = np.ones(manifest['rows'], dtype=bool)
    heldout = read(INPUT / 'heldout4k.json')
    heldout_index = defaultdict(list)
    for i, q in enumerate(heldout):
        heldout_index[normal(q['question'])].append(i)
    peers = [[] for _ in heldout]
    train_peers = [[] for _ in heldout]
    duplicate_indices, duplicate_positions = [], []
    original_lengths, source_lengths, compact_lengths = [], [], []
    canonical_text = {}
    inputs = [record(CACHE / 'compact-manifest.json'), record(protected_path),
              record(LOCAL / 'normal-duplicate-members.npz'), record(LOCAL / 'noise-values.npz'),
              record(LOCAL / 'noise.json'), record(WORK / 'checks/26-pinned-train-shard-inventory.json')]
    evidence = []
    cache_offset = raw_offset = normalized_protected_overlaps = 0
    for rec, spec in zip(manifest['shards'], inventory['files']):
        shard = rec['shard']
        assert spec['shard'] == shard
        folder = CACHE / rec['folder']
        for f in rec['files']:
            verify(folder / f['file'], f)
        ids = np.load(folder / 'provenance-rows.npy', mmap_mode='r', allow_pickle=False)
        assert len(ids) == rec['rows'] and np.all(np.diff(ids.astype(np.int64)) > 0)
        hashes = np.load(folder / 'question-sha256.npy', mmap_mode='r', allow_pickle=False)
        offsets = np.load(folder / 'target-offsets.npy', mmap_mode='r', allow_pickle=False)
        compact_lengths.append(np.diff(offsets).astype(np.uint16))
        source_lengths.append(np.diff(np.load(folder / 'source-offsets.npy', mmap_mode='r')).astype(np.uint16))
        original_root = (WORK / 'kaggle/staging/drive-merged/26-token-cache' if shard < 42 else
                         WORK / 'results/kaggle-runs/nlp-hw02-26c-cpu-cache/2/26c-token-cache')
        original_folder = original_root / rec['folder']
        expected = {f['file']: f for f in rec['original_files']}
        for name in ['provenance-rows.npy', 'question-sha256.npy', 'source-offsets.npy', 'target-offsets.npy']:
            verify(original_folder / name, expected[name])
            if name != 'target-offsets.npy':
                assert sha(original_folder / name) == sha(folder / name)
        raw_lengths = np.diff(np.load(original_folder / 'target-offsets.npy', mmap_mode='r'))
        assert len(raw_lengths) == len(ids)
        original_lengths.append(raw_lengths.astype(np.uint16))
        raw_path = RAW / f'part-{shard:05d}.parquet'
        verify(raw_path, spec)
        questions = pq.read_table(raw_path, columns=['question']).column('question').to_pylist()
        old_hashes = np.load(LOCAL / f'hashes-{shard:02d}.npy', mmap_mode='r', allow_pickle=False)
        assert len(questions) == len(old_hashes)
        lo, hi = np.searchsorted(raw_ids, [raw_offset, raw_offset + len(questions)])
        # Verify receipt -> union row ordering and numeric values before reuse.
        execution_path = LOCAL / f'noise-execution-{shard:02d}.json'
        execution = read(execution_path)['rows']
        assert len(execution) == hi - lo
        for j, ex in enumerate(execution, lo):
            assert (ex['status'] == 'numeric') == (statuses[j] == 0)
            if statuses[j] == 0:
                assert ex['value'] == values[j] and math.isfinite(values[j])
        evidence.append(record(execution_path))
        positions = np.searchsorted(raw_ids, ids.astype(np.int64) + raw_offset)
        valid = positions < len(raw_ids)
        found = np.flatnonzero(valid)
        valid[found] = raw_ids[positions[found]] == ids[found].astype(np.int64) + raw_offset
        selected_pos = positions[valid]
        assert np.all(statuses[selected_pos] == 0), 'Missing numeric duplicate execution'
        duplicate_indices.append(np.flatnonzero(valid).astype(np.uint32) + cache_offset)
        duplicate_positions.append(selected_pos)
        # Fresh references are never read; fresh questions only enforce exclusions.
        for j, raw_row in enumerate(ids):
            q = questions[int(raw_row)]
            nq = normal(q)
            assert q not in protected
            if nq in normalized_protected:
                normalized_protected_overlaps += 1
                safe_keep[cache_offset + j] = False
                for name, qq in protected_normal_sets.items():
                    protected_overlap_counts[name] += nq in qq
                if nq in heldout_index:
                    assert valid[j] and statuses[positions[j]] == 0
                    for i in heldout_index[nq]:
                        train_peers[i].append(float(values[positions[j]]))
            assert hashlib.sha256(q.encode()).digest() == hashes[j].tobytes()
            assert hashlib.sha256(nq.encode()).digest()[:16] == old_hashes[int(raw_row)]['normal'].tobytes()
            assert int.from_bytes(hashlib.sha256(f'42:{q}'.encode()).digest()[:8], 'big') % 10 != 0
        # Check equality of normalized strings within every selected raw group.
        for pos in selected_pos:
            nq = normal(questions[int(raw_ids[pos] - raw_offset)])
            g = int(groups[pos])
            if g in canonical_text:
                assert canonical_text[g] == nq
            else:
                canonical_text[g] = nq
        # Heldout-only source proxy. It has no path into duplicate_indices/order.
        relevant = [(pos, int(raw_ids[pos] - raw_offset), normal(questions[int(raw_ids[pos] - raw_offset)]))
                    for pos in range(lo, hi) if normal(questions[int(raw_ids[pos] - raw_offset)]) in heldout_index]
        if relevant:
            codes = pq.read_table(raw_path, columns=['code']).column('code').to_pylist()
            for pos, raw_row, nq in relevant:
                for i in heldout_index[nq]:
                    peers[i].append({'status': int(statuses[pos]), 'value': float(values[pos]),
                                     'own_pair': questions[raw_row] == heldout[i]['question'] and
                                                 codes[raw_row] == heldout[i]['code']})
            del codes
        cache_offset += len(ids)
        raw_offset += len(questions)
        print(f'VERIFIED shard {shard:02d}: cache={len(ids)} duplicates={len(selected_pos)}', flush=True)
    assert cache_offset == manifest['rows']
    assert int((~safe_keep).sum()) == normalized_protected_overlaps
    del canonical_text, questions, old_hashes, membership, noise
    indices = np.concatenate(duplicate_indices)
    positions = np.concatenate(duplicate_positions)
    gg, vv = groups[positions], values[positions]
    sorting = np.argsort(gg, kind='stable')
    gg, vv, indices = gg[sorting], vv[sorting], indices[sorting]
    bounds = np.r_[0, np.flatnonzero(gg[1:] != gg[:-1]) + 1, len(gg)]
    sizes = np.diff(bounds)
    train_duplicate_groups = int((sizes >= 2).sum())
    train_duplicate_rows = int(sizes[sizes >= 2].sum())
    singleton_rows = cache_offset - train_duplicate_rows
    total_groups = singleton_rows + train_duplicate_groups
    strict_keep = np.ones(cache_offset, dtype=bool)
    conservative_keep = np.ones(cache_offset, dtype=bool)
    counters = Counter()
    theoretical_loss = 0
    tie_rows = conflict_rows = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b-a == 1:
            continue
        labels, counts, strict, conservative = select(vv[a:b])
        strict_keep[indices[a:b]] = strict
        conservative_keep[indices[a:b]] = conservative
        top = int(counts.max())
        theoretical_loss += b-a-top
        if len(counts) > 1:
            counters['conflicting_groups'] += 1
            conflict_rows += b-a
        else:
            counters['unanimous_groups'] += 1
        if (counts == top).sum() > 1:
            counters['tied_mode_groups'] += 1
            tie_rows += b-a
        if 3*top >= 2*(b-a):
            counters['strict_two_thirds_groups'] += 1
        else:
            counters['below_two_thirds_groups'] += 1
    # Strengthen the original exact exclusions for every order, including control.
    strict_keep &= safe_keep
    conservative_keep &= safe_keep
    sl = np.concatenate(source_lengths)
    cl = np.concatenate(compact_lengths)
    ol = np.concatenate(original_lengths)
    protected_report = {'file': record(protected_path),
                        'group_question_counts': {k: len(v) for k, v in protected_groups.items()},
                        'exact_overlaps': 0, 'normalized_overlaps': 0,
                        'input_cache_normalized_overlaps_removed': normalized_protected_overlaps,
                        'input_overlap_rows_by_split': dict(protected_overlap_counts),
                        'removed_cache_indices': np.flatnonzero(~safe_keep).tolist(),
                        'hash_heldout_42_overlaps': 0,
                        'fresh_reference_reads': 0,
                        'checked': 'every compact row against original question/hash/provenance'}
    rules = {}
    base = histogram(ol)
    for name, mask, description in [
        ('strict-two-thirds', strict_keep, 'singletons + original rows in a unique modal answer cluster with count >= 2/3 group; other duplicate groups entirely dropped'),
        ('all-modal-ties-kept', conservative_keep, 'singletons + all maximum-frequency answer clusters; ties retained; only strictly lower-frequency clusters dropped')]:
        h = histogram(ol, mask)
        shift = {k: 100*(h['bins'][k]['fraction']-base['bins'][k]['fraction']) for k in BINS}
        rules[name] = {'description': description, 'rows': int(mask.sum()),
                       'dropped_rows': int((~mask).sum()), 'dropped_fraction': float((~mask).mean()),
                       'consensus_removed_rows': int((safe_keep & ~mask).sum()),
                       'original_BPE': h, 'original_bin_shift_pp': shift,
                       'compact_BPE': histogram(cl, mask),
                       'removed_original_BPE': histogram(ol, ~mask)}
    report = {'hypothesis': 'late linear cooldown on reliable original reference pairs reduces conflicting CE gradients',
              'scope': 'full verified compact train, CPU only; no sampling estimates',
              'cache_rows': cache_offset, 'raw_rows': raw_offset, 'cache_manifest_sha256': CACHE_SHA,
              'raw_revision': inventory['revision'], 'normalization': 'lower + collapse whitespace',
              'answer_equivalence': 'anchored first representative, isclose(rel_tol=1e-6, abs_tol=1e-6)',
              'census': {'groups': total_groups, 'singleton_rows': singleton_rows,
                         'duplicate_groups': train_duplicate_groups, 'duplicate_rows': train_duplicate_rows,
                         'duplicate_row_fraction': train_duplicate_rows/cache_offset,
                         **dict(counters), 'conflicting_group_fraction_among_duplicates': counters['conflicting_groups']/train_duplicate_groups,
                         'conflicting_group_fraction_all_groups': counters['conflicting_groups']/total_groups,
                         'conflicting_rows': int(conflict_rows), 'tied_mode_rows': int(tie_rows),
                         'nonmodal_rows_one_arbitrary_mode': int(theoretical_loss),
                         'nonmodal_row_fraction_all_train': float(theoretical_loss/cache_offset),
                         'nonmodal_row_fraction_duplicate_rows': float(theoretical_loss/train_duplicate_rows),
                         'rows_strictly_below_any_modal_cluster': int((safe_keep & ~conservative_keep).sum()),
                         'train_group_size_histogram': dict(Counter(map(int, sizes[sizes >= 2])) | Counter({1: singleton_rows})),
                         'note': 'arbitrary-mode loss includes ties; conservative removal does not'},
              'lengths': {'original_BPE': base, 'compact_BPE': histogram(cl),
                          'content_only': True, 'EOS_included': False,
                          'alignment': 'full original-offset SHA plus compact source/provenance/question SHA exact; no compact proxy'},
              'rules': rules, 'protected': protected_report,
              'control': {'description': 'uniform full compact train with the same strengthened protected exclusions',
                          'rows': int(safe_keep.sum()), 'dropped_rows': int((~safe_keep).sum()),
                          'original_BPE': histogram(ol, safe_keep)},
              'execution_reuse': {'numeric_duplicate_rows': len(indices), 'missing_numeric_receipts': 0,
                                 'new_program_executions': 0, 'all_singletons': 'finite by sealed cache receipts; no consensus information',
                                 'source_execution_receipts': evidence, 'inputs': inputs},
              'selected_rule': 'strict-two-thirds',
              'provenance': {'python': platform.python_version(), 'numpy': np.__version__,
                             'date': '2026-10-13', 'analysis_script': record(Path(__file__)),
                             'GPU_runs': 0, 'Kaggle_runs': 0, 'Kaggle_datasets_created': 0}}
    # Write frozen train-only evidence before touching model outcomes.
    write(OUTPUT / 'train-analysis.json', report)
    if not args.no_orders:
        report['orders'] = {name: build_order(mask, sl, cl, name, report) for name, mask in [
            ('strict-two-thirds', strict_keep), ('all-modal-ties-kept', conservative_keep),
            ('ordinary-protected-safe', safe_keep)]}
    report['heldout'] = heldout_diagnostic(heldout, peers, train_peers)
    report['seconds'] = time.monotonic() - started
    write(RESULT, report)
    write(OUTPUT / 'manifest.json', {'format': 'nlp-hw02-clean-cooldown-v1',
                                   'cache_manifest_sha256': CACHE_SHA, 'analysis': record(RESULT),
                                   'orders': {k: {'folder': v['folder'], 'order': v['metadata']['file'],
                                                  'manifest_sha256': v['manifest_sha256']}
                                              for k, v in report.get('orders', {}).items()},
                                   'protected': protected_report})
    print(json.dumps({'census': report['census'], 'rules': {k: {n: v[n] for n in
          ['rows', 'dropped_rows', 'dropped_fraction', 'original_bin_shift_pp']} for k, v in rules.items()},
          'heldout_labels': report['heldout']['label_counts'], 'seconds': report['seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
