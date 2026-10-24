"""Pinned inherited orders or a verified filtered membership with a new seed."""
import hashlib
import json
from pathlib import Path
import numpy as np
from kaggle_io import sha, write
from l2_sampler import load_order


def filtered_order(path, cache, expected_sha, seed, destination):
    path = Path(path)
    root = path.parent if path.is_file() else path
    assert path.name == 'order.npy' or path.is_dir()
    order, meta = load_order(root, cache, expected_sha)
    assert meta['cache_manifest_sha256'] == cache.manifest_sha
    assert meta.get('batch', 128) == 128
    # Original compact indices. Repetition is allowed ONLY for a declared,
    # fully checked long2x order; deleting a row must delete both occurrences.
    counts = np.bincount(order, minlength=len(cache))
    retained = np.flatnonzero(counts).astype('<u4')
    multiplicity = meta.get('filtered_multiplicity')
    if multiplicity is None:
        assert np.all(counts <= 1), 'Filtered membership must be unique'
    else:
        assert multiplicity == dict(format='nlp-hw02-filtered-long2x-v1', threshold=79, short=1, long=2)
        expected = 1+(cache.target_lengths[retained] >= multiplicity['threshold'])
        assert np.array_equal(counts[retained], expected), 'Filtered long2x multiplicity mismatch'
        assert 'selected_cache_indices_sha256' in meta
    if 'selected_cache_indices_sha256' in meta:
        assert hashlib.sha256(retained.tobytes()).hexdigest() == meta['selected_cache_indices_sha256']
    assert not np.isin(order, [8545870]).any(), 'Protected test-overlap row'
    slots = np.asarray(order).copy()
    rng = np.random.default_rng(seed); rng.shuffle(slots)
    for lo in range(0, len(slots), 4096):
        part = slots[lo:lo+4096]
        ns = cache.source_lengths[part].astype(np.int32); nt = cache.target_lengths[part].astype(np.int32)
        part = part[np.lexsort((nt, ns, ns+nt+1))]; full = len(part)//128
        if full:
            part[:full*128] = part[:full*128].reshape(full,128)[rng.permutation(full)].reshape(-1)
        slots[lo:lo+len(part)] = part
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    np.save(destination/'order.npy', slots, allow_pickle=False)
    result = dict(meta, seed=seed, parent_order_sha256=expected_sha, slots=len(slots),
                  file=dict(file='order.npy', sha256=sha(destination/'order.npy'),
                            bytes=(destination/'order.npy').stat().st_size))
    write(destination/'order.json', result)
    # Preserve the frozen evaluation split and its evidence alongside membership.
    split = root/'split.json'
    if split.exists():
        (destination/'split.json').write_bytes(split.read_bytes())
    return load_order(destination, cache, result['file']['sha256'])
