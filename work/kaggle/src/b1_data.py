"""C-long2x multiplicity (compact length >=79), new fixed bucketed permutation."""
from pathlib import Path
import numpy as np
from kaggle_io import sha, write
from l2_sampler import load_order

SEED = 2026101501


def build_order(cache, destination, seed=SEED):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    long = cache.target_lengths >= 79
    slots = np.concatenate((np.arange(len(cache), dtype=np.uint32), np.flatnonzero(long).astype(np.uint32)))
    rng = np.random.default_rng(seed); rng.shuffle(slots)
    for lo in range(0, len(slots), 4096):
        part = slots[lo:lo+4096]
        ns = cache.source_lengths[part].astype(np.int32); nt = cache.target_lengths[part].astype(np.int32)
        part = part[np.lexsort((nt, ns, ns+nt+1))]; full = len(part)//128
        if full: part[:full*128] = part[:full*128].reshape(full, 128)[rng.permutation(full)].reshape(-1)
        slots[lo:lo+len(part)] = part
    np.save(destination/'order.npy', slots, allow_pickle=False)
    meta = dict(format='nlp-hw02-b1-order-v1', seed=seed, rows=len(cache), slots=len(slots),
                batch=128, chunk=4096, threshold=79, multiplicity=dict(short=1, long=2),
                cache_manifest_sha256=cache.manifest_sha,
                file=dict(file='order.npy', sha256=sha(destination/'order.npy'), bytes=(destination/'order.npy').stat().st_size))
    write(destination/'order.json', meta)
    return verify_order(destination, cache)


def verify_order(root, cache):
    import json
    meta = json.loads((Path(root)/'order.json').read_text())
    order, _ = load_order(root, cache, meta['file']['sha256'])
    assert meta['format'] == 'nlp-hw02-b1-order-v1' and meta['cache_manifest_sha256'] == cache.manifest_sha
    assert meta['threshold'] == 79 and meta['batch'] == 128 and meta['chunk'] == 4096
    assert meta['multiplicity'] == dict(short=1, long=2)
    assert np.array_equal(np.bincount(order, minlength=len(cache)), 1+(cache.target_lengths >= 79))
    return order, meta
