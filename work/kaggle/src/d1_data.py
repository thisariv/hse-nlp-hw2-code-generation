"""D1 CPU contracts: immutable artifacts, train identities, ranks and L2 orders."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from kaggle_io import sha, write

CACHE_SHA = '46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be'
ROWS = 9725737
MODELS = ('C-main', 'C-long2x')
EDGES = (53, 79, 105)  # Compact CONTENT, mode included, EOS excluded.
SEED = 2026101404
WEIGHTS = {
    'C-main': '8c11b1038855422b6912e243049a554317978f848edb412ed48aebbe4cb6b886',
    'C-long2x': 'd10bf0018d0b14fd6cc3c8b9c7121f127b5cec97057544fd63e5824253a55057'}


def record(path):
    p = Path(path)
    return dict(file=p.name, bytes=p.stat().st_size, sha256=sha(p))


def save_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def publish(path, files, **metadata):
    """Receipt is the commit point; its blobs must never be modified afterwards."""
    path = Path(path)
    write(path, dict(format='nlp-hw02-d1-artifact-v1', files=[record(p) for p in files], **metadata))


def verify_receipt(path):
    path = Path(path); meta = json.loads(path.read_text())
    assert meta['format'] == 'nlp-hw02-d1-artifact-v1'
    for f in meta['files']:
        assert Path(f['file']).name == f['file']
        p = path.parent/f['file']
        assert p.stat().st_size == f['bytes'] and sha(p) == f['sha256'], str(p)
    return meta


def length_bins(lengths):
    return np.searchsorted(EDGES, lengths, side='right')


def ranked_candidates(main, long, lengths, fraction=.08):
    """Top q separately per length bin. Float32 mean of persisted FP16 scores.

    Deterministic row-id ties. Rounded bin quotas may differ from q*N by <=2.
    """
    a, b = np.asarray(main), np.asarray(long)
    assert a.shape == b.shape == np.asarray(lengths).shape
    assert np.isfinite(a).all() and np.isfinite(b).all() and 0 < fraction <= .08
    score = (a.astype(np.float32)+b.astype(np.float32))*.5
    bins = length_bins(lengths); ranked = []
    for bin_id in range(4):
        ids = np.flatnonzero(bins == bin_id)
        k = min(len(ids), int(math.floor(len(ids)*fraction+.5)))
        # Partition narrows the expensive sort; explicitly resolve the boundary tie.
        if k:
            threshold = np.partition(score[ids], len(ids)-k)[len(ids)-k]
            greater = ids[score[ids] > threshold]
            tied = ids[score[ids] == threshold][:k-len(greater)]
            ids = np.r_[greater, tied]
            ids = ids[np.lexsort((ids, -score[ids]))]
        else:
            ids = ids[:0]
        ranked.append(ids.astype(np.uint32))
    return ranked, score


def choose_budget(ranked, pilots, remaining, safety=1.6, tail=240.):
    """One proportional reduction across bins; both GPUs get identical row IDs.

    Pilot wall includes generation AND overlapped execution. Use the slower model
    in EACH bin, so a fast short bin cannot hide the long tail. No upward resizing.
    """
    rates = []
    for b in range(4):
        if not len(ranked[b]):
            rates.append(0.); continue
        rates.append(max(p['bins'][b]['seconds']/max(1, p['bins'][b]['rows']) for p in pilots))
    cost = safety*sum(len(ids)*rate for ids, rate in zip(ranked, rates))
    scale = min(1., max(0., remaining-tail)/max(cost, 1e-9))
    quotas = [int(len(ids)*scale) for ids in ranked]
    pilot_counts = [max(p['bins'][b]['rows'] for p in pilots) for b in range(4)]
    if any(q < count for q, count in zip(quotas, pilot_counts)):
        raise TimeoutError('Insufficient budget even for pilot-sized common subset')
    selected = np.concatenate([ids[:q] for ids, q in zip(ranked, quotas)])
    return selected, dict(maximum_rows=sum(map(len, ranked)), selected_rows=len(selected),
                         quotas=quotas, seconds_per_row_by_bin=rates, safety=safety,
                         estimated_seconds=safety*sum(q*r for q, r in zip(quotas, rates)),
                         available_seconds=remaining, execution_tail_reserve=tail)


def close(a, b):
    # Symmetric math.isclose, unlike asymmetric np.isclose.
    return np.abs(a-b) <= np.maximum(1e-6, 1e-6*np.maximum(np.abs(a), np.abs(b)))


def cluster(values):
    reps, labels, counts = [], [], []
    for x in values:
        for j, rep in enumerate(reps):
            if math.isclose(float(x), rep, rel_tol=1e-6, abs_tol=1e-6):
                labels.append(j); counts[j] += 1; break
        else:
            labels.append(len(reps)); reps.append(float(x)); counts.append(1)
    return np.asarray(labels), np.asarray(counts), np.asarray(reps)


def filtered_order(keep, sl, tl, destination, cache_sha, rule, evidence_sha):
    """Same shuffle/bucketing as clean-cooldown; unmodified L2 loader accepts it."""
    from l2_sampler import load_order, fixed_split
    from exp26_common import json_hash
    root = Path(destination); root.mkdir(parents=True, exist_ok=True)
    order = np.flatnonzero(keep).astype(np.uint32)
    assert len(order), 'Refuse an empty train order'
    rng = np.random.default_rng(SEED); rng.shuffle(order)
    for lo in range(0, len(order), 4096):
        part = order[lo:lo+4096]
        ns, nt = np.asarray(sl[part], dtype=np.int32), np.asarray(tl[part], dtype=np.int32)
        part = part[np.lexsort((nt, ns, ns+nt+1))]; full = len(part)//128
        if full:
            part[:full*128] = part[:full*128].reshape(full, 128)[rng.permutation(full)].reshape(-1)
        order[lo:lo+len(part)] = part
    assert np.array_equal(np.sort(order), np.flatnonzero(keep))
    np.save(root/'order.npy', order, allow_pickle=False)
    meta = dict(seed=SEED, rows=len(keep), slots=len(order), batch=128, chunk=4096,
                selected_bijection=True, cache_manifest_sha256=cache_sha, file=record(root/'order.npy'),
                rule=rule, evidence_sha256=evidence_sha,
                selected_cache_indices_sha256=hashlib.sha256(np.flatnonzero(keep).astype('<u4').tobytes()).hexdigest(),
                answer_policy='retain original pairs only; model programs never become targets')
    write(root/'order.json', meta)
    write(root/'split.json', {k: dict(ids=v, sha256=json_hash(v)) for k, v in fixed_split().items()})
    class Identity:
        def __len__(self): return len(keep)
    loaded, receipt = load_order(root, Identity(), meta['file']['sha256'])
    assert np.array_equal(loaded, order) and receipt == meta
    return meta
