"""Deterministic 1/2 row multiplicity; reference lengths never enter training loss."""
import hashlib, json
from pathlib import Path
import numpy as np
from kaggle_io import sha, write
from exp26_common import json_hash

SEED = 2026101302

def fixed_split(n=4096):
    ordered = sorted(range(n), key=lambda i: hashlib.sha256(
        f'candidate-selection-v1:heldout4k:{i}'.encode()).digest())
    return {'tune2048': sorted(ordered[:n//2]), 'eval2048': sorted(ordered[n//2:])}

def threshold_for(lengths, fraction):
    """Nearest attainable >=threshold fraction; ties choose the higher threshold."""
    counts = np.bincount(np.asarray(lengths, dtype=np.int64))
    tails = np.cumsum(counts[::-1])[::-1]
    options = [(abs(int(tails[t])/len(lengths)-fraction), -t, t)
               for t in np.flatnonzero(counts)]
    options.append((fraction, -len(counts), len(counts)))
    return min(options)[2]

def original_lengths(cache, root):
    """Accept raw offsets only with exact source/provenance/row SHA alignment."""
    root = Path(root) if root else None
    if root is None: return None
    values = []
    for rec in cache.manifest['shards']:
        folder = root/rec['folder']; compact = cache.root/rec['folder']
        expected = {r['file']: r for r in rec['original_files']}
        for name in ('source-offsets.npy', 'provenance-rows.npy', 'question-sha256.npy', 'target-offsets.npy'):
            p = folder/name; r = expected[name]
            assert p.stat().st_size == r['bytes'] and sha(p) == r['sha256'], 'Unverified original lengths'
            if name != 'target-offsets.npy': assert sha(p) == sha(compact/name), 'Row identity mismatch'
        offsets = np.load(folder/'target-offsets.npy', mmap_mode='r', allow_pickle=False)
        assert len(offsets)-1 == rec['rows']
        values.append(np.diff(offsets))
    return np.concatenate(values)

def build_order(cache, heldout_lengths, destination, original_root=None, seed=SEED, batch=128, chunk=4096):
    assert 0 < len(cache) <= np.iinfo(np.uint32).max and batch > 0 and chunk > 0
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    raw = original_lengths(cache, original_root)
    lengths = raw if raw is not None else cache.target_lengths
    fraction = float(np.mean(np.asarray(heldout_lengths) >= 90))
    threshold = threshold_for(lengths, fraction); long = np.asarray(lengths) >= threshold
    slots = np.concatenate((np.arange(len(cache), dtype=np.uint32), np.flatnonzero(long).astype(np.uint32)))
    rng = np.random.default_rng(seed); rng.shuffle(slots)
    # Identical L1 bucketing: sort inside 4096, then permute complete batches.
    for lo in range(0, len(slots), chunk):
        part = slots[lo:lo+chunk]; ns = cache.source_lengths[part].astype(np.int32)
        nt = cache.target_lengths[part].astype(np.int32)
        part = part[np.lexsort((nt, ns, ns+nt+1))]; full = len(part)//batch
        if full: part[:full*batch] = part[:full*batch].reshape(full, batch)[rng.permutation(full)].reshape(-1)
        slots[lo:lo+len(part)] = part
    assert np.array_equal(np.bincount(slots, minlength=len(cache)), 1+long.astype(np.int64))
    np.save(destination/'order.npy', slots, allow_pickle=False)
    meta = {'seed': seed, 'rows': len(cache), 'slots': len(slots), 'batch': batch, 'chunk': chunk,
            'length_source': 'original BPE offsets; exact row identity' if raw is not None else 'verified compact target-offsets',
            'original_lengths_available': raw is not None, 'threshold': int(threshold),
            'heldout_original_ge90': int(sum(np.asarray(heldout_lengths) >= 90)),
            'heldout_rows': len(heldout_lengths), 'target_long_fraction': fraction,
            'train_long_rows': int(long.sum()), 'train_long_fraction': float(long.mean()),
            'fraction_error': float(long.mean())-fraction, 'weighted_long_fraction': float(2*long.sum()/len(slots)),
            'multiplicity': {'short': 1, 'long': 2}, 'cache_manifest_sha256': cache.manifest_sha,
            'file': {'file': 'order.npy', 'sha256': sha(destination/'order.npy'), 'bytes': (destination/'order.npy').stat().st_size},
            'selection': 'nearest attainable threshold; higher threshold wins ties; no loss reweighting',
            'lengths_sha256': hashlib.sha256(np.asarray(lengths, dtype='<i8').tobytes()).hexdigest()}
    write(destination/'order.json', meta)
    write(destination/'split.json', {k: {'ids': v, 'sha256': json_hash(v)} for k, v in fixed_split().items()})
    return slots, meta

def load_order(root, cache, expected_sha):
    """Check metadata and every index before the first CUDA update, including weighted tails."""
    root = Path(root); meta = json.loads((root/'order.json').read_text())
    path = root/'order.npy'
    assert sha(path) == meta['file']['sha256'] == expected_sha, 'Order SHA mismatch'
    assert path.stat().st_size == meta['file']['bytes'] and meta['rows'] == len(cache)
    order = np.load(path, mmap_mode='r', allow_pickle=False)
    assert order.ndim == 1 and order.dtype == np.uint32 and len(order) > 0
    assert int(order.min()) >= 0 and int(order.max()) < len(cache), 'Order index out of bounds'
    if 'slots' in meta: assert len(order) == meta['slots']
    return order, meta
