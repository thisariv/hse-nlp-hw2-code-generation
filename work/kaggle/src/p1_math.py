"""P1-only objectives. Frozen L1/L2 update, retry and finite-state checks are reused."""
import math
import copy
import gc
import time
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import exp26_core as C
import l1_runtime as L
import l2_runtime as R
import exp26_train as T
from exp26_order import collate

# Compact content lengths, including the mode token. Approximate 60/90/120 raw BPE;
# 79 is the inherited L2 compact analogue of 90. Never call these raw BPE bins.
COMPACT_EDGES = (53, 79, 105)
PROTECTED_CACHE_INDICES = (8545870,)


def sqrt_lr(step, total, peak=.0015):
    if total <= 0 or not 0 <= step <= total:
        raise ValueError('Invalid cooldown position')
    return peak * (1. - math.sqrt(step / total))


def truncation_mask(scores, lengths, q=.1, edges=COMPACT_EDGES):
    """Nearest integer q*N; proportional quotas, largest remainder, >=1 kept/bin.

    Ties in scores use position in the effective batch, including duplicate rows.
    Quotas use counts only: high-loss long rows cannot take another bin's quota.
    """
    scores = np.asarray(scores, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)
    if scores.ndim != 1 or scores.shape != lengths.shape or not np.isfinite(scores).all() or not 0 <= q < 1:
        raise FloatingPointError('Invalid per-example loss')
    bins = np.searchsorted(edges, lengths, side='right')
    counts = np.bincount(bins, minlength=len(edges)+1)
    capacities = np.maximum(0, counts-1)
    desired = min(int(math.floor(q*len(scores)+.5)), int(capacities.sum()))
    quotas = np.minimum(np.floor(q*counts).astype(int), capacities)
    while int(quotas.sum()) < desired:
        available = [b for b in range(len(counts)) if quotas[b] < capacities[b]]
        b = min(available, key=lambda b: (-(q*counts[b]-quotas[b]), b))
        quotas[b] += 1
    keep = np.ones(len(scores), dtype=bool)
    for b, quota in enumerate(quotas):
        positions = np.flatnonzero(bins == b)
        ranked = sorted(positions, key=lambda j: (-scores[j], int(j)))
        keep[ranked[:quota]] = False
    return keep, dict(counts=counts.tolist(), excluded=quotas.tolist(),
                      actual_fraction=float((~keep).mean()) if len(keep) else 0.,
                      compact_edges=list(edges), requested_q=q)


@torch.no_grad()
def per_example_ce(model, cache, indices, device, micro, amp=False):
    scores = []
    for lo in range(0, len(indices), micro):
        src, tgt, labels, _ = collate(cache, indices[lo:lo+micro], device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(src, tgt)
        losses = F.cross_entropy(logits.float().transpose(1, 2), labels,
                                 ignore_index=C.PAD, reduction='none')
        values = losses.sum(1)/labels.ne(C.PAD).sum(1)
        if not bool(torch.isfinite(values).all()):
            raise FloatingPointError('Nonfinite selection loss')
        scores.extend(values.cpu().tolist())
        del src, tgt, labels, logits, losses, values
    return np.asarray(scores)


def graph_truncated_update(model, opt, scaler, cache, indices, device, micro, lr):
    """One forward per example; hold microbatch graphs until the global mask exists."""
    C.sync(device); begin = time.perf_counter(); opt.zero_grad(set_to_none=True)
    graphs = []; scores = []; lengths = np.asarray([len(cache[int(i)][1]) for i in indices])
    for lo in range(0, len(indices), micro):
        src, tgt, labels, _ = collate(cache, indices[lo:lo+micro], device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler.is_enabled()):
            logits = model(src, tgt)
            losses = F.cross_entropy(logits.reshape(-1, model.vocab), labels.reshape(-1),
                                     ignore_index=C.PAD, reduction='none').reshape_as(labels).sum(1)
        values = losses.detach()/labels.ne(C.PAD).sum(1)
        if not bool(torch.isfinite(values).all()): raise FloatingPointError('Nonfinite selection loss')
        scores.extend(values.cpu().tolist()); graphs.append(losses)
        del src, tgt, labels, logits, losses, values
    scores = np.asarray(scores); keep, stats = truncation_mask(scores, lengths)
    tokens = int(sum(lengths[keep]+1)); total = 0.
    # Each microbatch uses the SAME effective-batch mask and retained-token denominator.
    # Pop graphs before backward, releasing each activation set as soon as possible.
    lo = 0
    while graphs:
        losses = graphs.pop(0)
        mask = torch.as_tensor(keep[lo:lo+len(losses)], device=device)
        loss = losses[mask].sum()/tokens
        if not bool(torch.isfinite(loss).detach()): raise FloatingPointError('Nonfinite retained loss')
        scaler.scale(loss).backward(); total += float(loss.detach()); lo += len(losses)
        del losses, loss, mask
    for group in opt.param_groups: group['lr'] = lr
    scaler.unscale_(opt); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    skipped = bool(any(float(t)>0 for t in scaler._found_inf_per_device(opt).values())) if scaler.is_enabled() else False
    if not skipped and not bool(torch.isfinite(norm)): raise FloatingPointError('Nonfinite successful gradient')
    try: scaler.step(opt)
    except torch.cuda.OutOfMemoryError as error:
        error.optimizer_may_be_modified = True
        raise
    scaler.update(); C.sync(device)
    return dict(seconds=time.perf_counter()-begin, CE=total, AMP_skip=int(skipped), lr=lr,
                pairs=len(indices), tokens=tokens, gradient_norm=float(norm) if torch.isfinite(norm) else None), scores, keep, stats


def safe_truncated_update(model, opt, scaler, cache, indices, device, micro, lr, strategy='graph'):
    """Selection over all 128 BEFORE backward; retry repeats selection with smaller micro.

    Fast path retains graphs, eliminating the selection-only forward. If those graphs
    exceed memory, discard ALL gradients/graphs and restore RNG/scaler before the
    bounded two-pass fallback. Adam OOM always propagates as unsafe, never retries.
    """
    base_update = L.update

    def truncated(m, o, s, data, ids, d, mb, rate):
        begin = time.perf_counter()
        lengths = np.asarray([len(data[int(i)][1]) for i in ids])
        rng = T.cpu(T.rng_state()); scale = copy.deepcopy(s.state_dict()); fallback = False
        if strategy == 'graph':
            try:
                log, scores, keep, stats = graph_truncated_update(m, o, s, data, ids, d, mb, rate)
            except torch.cuda.OutOfMemoryError as error:
                if getattr(error, 'optimizer_may_be_modified', False): raise
                fallback = True
            if fallback:
                o.zero_grad(set_to_none=True); gc.collect()
                if d.type == 'cuda': torch.cuda.empty_cache()
                s.load_state_dict(scale)
                if s.is_enabled(): s._per_optimizer_states.clear()
                T.restore_rng(rng, d)
        if strategy == 'two-pass' or fallback:
            scores = per_example_ce(m, data, ids, d, mb, s.is_enabled())
            keep, stats = truncation_mask(scores, lengths)
            log = base_update(m, o, s, data, np.asarray(ids)[keep], d, mb, rate)
        selected = np.asarray(ids)[keep]
        stats.update(presented=len(ids), retained=len(selected),
                     presented_tokens=int(sum(lengths+1)), retained_tokens=log['tokens'],
                     excluded_positions=np.flatnonzero(~keep).tolist(),
                     excluded_indices=np.asarray(ids)[~keep].astype(int).tolist(),
                     excluded_target_lengths=lengths[~keep].tolist(),
                     excluded_source_lengths=[len(data[int(i)][0]) for i in np.asarray(ids)[~keep]],
                     excluded_mean_CE=float(scores[~keep].mean()) if (~keep).any() else None,
                     retained_mean_CE=float(scores[keep].mean()),
                     forward_mode='two-pass' if strategy == 'two-pass' or fallback else 'graph',
                     graph_OOM_fallback=fallback)
        log.update(truncation=stats, pairs=len(ids), seconds=time.perf_counter()-begin)
        return log

    # Process-local, synchronous scope; safe_update and resilient_update stay frozen.
    with patch.object(L, 'update', truncated):
        return R.safe_update(model, opt, scaler, cache, indices, device, micro, lr)


class ReverseCache:
    """Code + mode -> question content. Original question source has BOS/EOS."""
    def __init__(self, base):
        self.base = base
        self.manifest_sha = getattr(base, 'manifest_sha', 'heldout')
        self.source_lengths = np.asarray(base.target_lengths)+2
        self.target_lengths = np.asarray(base.source_lengths)-2

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        question, code = self.base[int(index)]
        assert int(question[0]) == C.BOS and int(question[-1]) == C.EOS
        assert len(question) >= 3 and all(int(t) >= 3 for t in code)
        return np.asarray([C.BOS, *code, C.EOS]), np.asarray(question[1:-1])
