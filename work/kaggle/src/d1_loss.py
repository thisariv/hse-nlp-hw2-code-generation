"""Teacher-forced row NLL, exact training labels, FP32 CE on FP16 final weights."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from exp26_order import collate
from kaggle_io import write
from d1_data import publish, save_npz, verify_receipt


@torch.inference_mode()
def row_nll(model, cache, ids, device, ce_chunk=4096):
    src, tgt, labels, _ = collate(cache, ids, device)
    logits = model(src, tgt)
    # A full float() of B*T*8195 is unnecessary; limit the FP32 CE workspace.
    flat = logits.reshape(-1, model.vocab); y = labels.reshape(-1)
    losses = torch.empty(len(y), dtype=torch.float32, device=device)
    for lo in range(0, len(y), ce_chunk):
        losses[lo:lo+ce_chunk] = F.cross_entropy(flat[lo:lo+ce_chunk].float(), y[lo:lo+ce_chunk],
                                               ignore_index=0, reduction='none')
    nll = losses.reshape_as(labels).sum(1)
    lengths = labels.ne(0).sum(1)
    if not bool(torch.isfinite(nll).all()): raise FloatingPointError('Nonfinite D1 NLL')
    return nll.cpu().numpy(), lengths.cpu().numpy().astype(np.uint16)


def bucket_order(cache, ids):
    ids = np.asarray(ids, dtype=np.uint32)
    sl = cache.source_lengths[ids].astype(np.int32)
    tl = cache.target_lengths[ids].astype(np.int32)+1
    return ids[np.lexsort((ids, sl, tl, sl+tl))]


@torch.inference_mode()
def parity_check(model, cache, ids, device):
    """Reviewer GPU smoke also compares against the ordinary flattened CE path."""
    nn, ll = row_nll(model, cache, ids, device)
    src, tgt, labels, counts = collate(cache, ids, device)
    logits = model(src, tgt)
    ordinary = F.cross_entropy(logits.float().reshape(-1, model.vocab), labels.reshape(-1), ignore_index=0)
    delta = abs(float(nn.sum(dtype=np.float64)/ll.sum())-float(ordinary))
    assert delta < 1e-5 and int(ll.sum()) == counts['target_tokens']
    assert all(p.grad is None for p in model.parameters())
    return dict(passed=True, rows=len(ids), tokens=int(ll.sum()), token_CE=float(ordinary), absolute_delta=delta)


def score_all(model, cache, ids, root, device, check, batch=256, token_slots=32768, block=8192,
              compact=False):
    """Immutable row-id blocks permit live recovery; consolidated arrays index full cache.

    Memory stays bounded on long examples; OOM retries the same rows with half batch.
    A receipt is published only after the corresponding complete block is durable.
    """
    root = Path(root); root.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    ids = np.asarray(ids, dtype=np.uint32); model.eval()
    n = len(cache); total = 0.; tokens = 0; done = 0; oom = 0; interrupted = False
    for lo in range(0, len(ids), block):
        try: check()
        except TimeoutError: interrupted = True; break
        output = root/f'block-{lo:08d}.npz'; receipt = output.with_suffix('.receipt.json')
        current = bucket_order(cache, ids[lo:lo+block])
        if receipt.exists():
            verify_receipt(receipt)
            with np.load(output, allow_pickle=False) as saved:
                assert np.array_equal(saved['row_id'], current)
                meta = verify_receipt(receipt)
                total += meta['sum_NLL']; tokens += meta['tokens']
        else:
            nlls = np.empty(len(current), np.float32); lens = np.empty(len(current), np.uint16)
            offset = 0
            while offset < len(current):
                try: check()
                except TimeoutError: interrupted = True; break
                count = min(batch, len(current)-offset)
                while count > 1:
                    take = current[offset:offset+count]
                    max_t = int(cache.target_lengths[take].max())+1
                    max_s = int(cache.source_lengths[take].max())
                    if count*max_t <= token_slots and count*(max_t+max_s) <= 2*token_slots: break
                    count //= 2
                try:
                    a, b = row_nll(model, cache, current[offset:offset+count], device)
                    nlls[offset:offset+count], lens[offset:offset+count] = a, b
                    offset += count
                except torch.cuda.OutOfMemoryError:
                    oom += 1; batch = count//2
                    if not batch: raise
                    import gc
                    gc.collect()
                    if device.type == 'cuda': torch.cuda.empty_cache()
            current, nlls, lens = current[:offset], nlls[:offset], lens[:offset]
            if not offset: break
            block_total = float(nlls.sum(dtype=np.float64)); block_tokens = int(lens.sum())
            total += block_total; tokens += block_tokens
            arrays = dict(row_id=current, length=lens, ce=(nlls/lens).astype(np.float16))
            if not compact: arrays['nll'] = nlls
            save_npz(output, **arrays)
            publish(receipt, [output], phase='loss-block', rows=len(current),
                    sum_NLL=block_total, tokens=block_tokens, complete=not interrupted)
        done += len(current)
        write(root/'progress.json', dict(rows=done, target_rows=len(ids), seconds=time.monotonic()-started,
                                       sum_NLL=total, tokens=tokens, batch=batch, OOM_retries=oom))
        print('D1 CE', done, '/', len(ids), 'seconds', round(time.monotonic()-started, 1), flush=True)
        if interrupted: break
    paths = [root/'ce.npy', root/'length.npy'] if compact else [root/'ce.npy', root/'nll.npy', root/'length.npy']
    dtypes = [np.float16, np.uint16] if compact else [np.float16, np.float32, np.uint16]
    arrays = [np.lib.format.open_memmap(str(p)+'.tmp', mode='w+', dtype=dt, shape=(n,))
              for p, dt in zip(paths, dtypes)]
    arrays[0][:] = np.nan; arrays[-1][:] = 0
    if not compact: arrays[1][:] = np.nan
    for receipt in sorted(root.glob('block-*.receipt.json')):
        verify_receipt(receipt)
        with np.load(receipt.with_name(receipt.name.replace('.receipt.json', '.npz')), allow_pickle=False) as saved:
            ii, ll = saved['row_id'], saved['length']
            arrays[0][ii] = saved['ce']; arrays[-1][ii] = ll
            if not compact: arrays[1][ii] = saved['nll']
    for a in arrays: a.flush()
    del arrays
    for path in paths: Path(str(path)+'.tmp').replace(path)
    summary = dict(phase='loss', complete=done == len(ids), rows=done, cache_rows=n,
                   sum_NLL=total, tokens=tokens, token_CE=total/tokens if tokens else None, OOM_retries=oom,
                   seconds=time.monotonic()-started, normalized_dtype='float16', nll_dtype='float32',
                   scoring='target content + EOS; PAD/BOS excluded; FP32 CE; no smoothing',
                   compact=compact, target_rows=len(ids), rows_per_second=done/max(time.monotonic()-started, 1e-9))
    publish(root/('complete.receipt.json' if summary['complete'] else 'partial.receipt.json'), paths, **summary)
    return summary


def merge_shards(common, ids, cache):
    """Publish one C-long2x CE array, retaining NaN for unscored global row IDs."""
    common = Path(common); ids = np.asarray(ids, dtype=np.uint32)
    root = common/'C-long2x/loss'; root.mkdir(parents=True, exist_ok=True)
    output = root/'ce.npy'
    ce = np.lib.format.open_memmap(str(output)+'.tmp', mode='w+', dtype=np.float16, shape=(len(cache),))
    ce[:] = np.nan; seen = np.zeros(len(cache), bool); total = 0.; tokens = 0; reports = []
    for gpu in (0, 1):
        shard = common/f'loss-shards/gpu{gpu}'
        meta = verify_receipt(shard/('complete.receipt.json' if (shard/'complete.receipt.json').exists()
                                     else 'partial.receipt.json'))
        a = np.load(shard/'ce.npy', mmap_mode='r', allow_pickle=False)
        length = np.load(shard/'length.npy', mmap_mode='r', allow_pickle=False)
        current = np.flatnonzero(np.isfinite(a))
        assert not seen[current].any() and np.isin(current, ids[gpu::2]).all()
        assert np.array_equal(length[current], cache.target_lengths[current]+1)
        assert len(current) == meta['rows'] and int(length[current].sum()) == meta['tokens']
        ce[current] = a[current]; seen[current] = True
        total += meta['sum_NLL']; tokens += meta['tokens']; reports.append(meta)
    ce.flush(); del ce; Path(str(output)+'.tmp').replace(output)
    summary = dict(phase='loss', model='C-long2x', weight_sha256=__import__('d1_data').WEIGHTS['C-long2x'],
                   complete=bool(seen[ids].all()), rows=int(seen.sum()), target_rows=len(ids),
                   cache_rows=len(cache), sum_NLL=total, tokens=tokens, token_CE=total/tokens if tokens else None,
                   normalized_dtype='float16', shards=reports, score_policy='C-long2x only')
    publish(root/('complete.receipt.json' if summary['complete'] else 'partial.receipt.json'), [output], **summary)
    return summary
