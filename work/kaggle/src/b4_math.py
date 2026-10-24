"""B4 bounded two-pass truncation; ordinary B2 CE/z-loss autograd and safety."""
import time
from unittest.mock import patch
import numpy as np
import b2_optim as O
from b2_optim import update as retained_update
import exp26_core as C
import exp26_train as T
from p1_math import truncation_mask, per_example_ce, COMPACT_EDGES


def two_pass_update(model, opt, scaler, cache, indices, device, micro, lr):
    """Score all presentations without graphs, then train only retained examples.

    Selection uses the same training mode and FP16 autocast as B2. Restore its
    RNG even on failure, so the extra forward does not consume training RNG.
    B4's pinned dropout=0 model consumes no RNG in either forward: both workers
    keep identical RNG and advance the same input cursor by all 128 examples.
    Only one microbatch graph is live; no activation offload/custom backward.
    """
    C.sync(device); begin=time.perf_counter(); opt.zero_grad(set_to_none=True)
    indices=np.asarray(indices)
    lengths=np.asarray([len(cache[int(i)][1]) for i in indices])
    rng=T.cpu(T.rng_state())
    try:
        scores=per_example_ce(model,cache,indices,device,micro,scaler.is_enabled())
    finally:
        T.restore_rng(rng,device)
    keep,stats=truncation_mask(scores,lengths,q=.1)
    # This unpatched alias is exactly GPU0's standard autograd path: global
    # retained-token CE+1e-4*z, micro backward, one unscale/clip/step/update.
    log=retained_update(model,opt,scaler,cache,indices[keep],device,micro,lr)
    stats.update(presented=len(indices),retained=int(keep.sum()),presented_tokens=int(sum(lengths+1)),
                 retained_tokens=log['tokens'],excluded_positions=np.flatnonzero(~keep).tolist(),
                 excluded_indices=indices[~keep].astype(int).tolist(),
                 excluded_target_lengths=lengths[~keep].tolist(),
                 excluded_source_lengths=[len(cache[int(i)][0]) for i in indices[~keep]],
                 excluded_mean_CE=float(scores[~keep].mean()) if (~keep).any() else None,
                 retained_mean_CE=float(scores[keep].mean()),forward_mode='two-pass',extra_selection_forward=True,
                 selection_rng_restored=True,CE_z_backward='standard B2 autograd')
    log.update(pairs=len(indices),seconds=time.perf_counter()-begin,truncation=stats)
    return log


def safe_update(*args,**kwargs):
    # B2 retries restore RNG/scaler/gradients and repeat BOTH passes on OOM;
    # an OOM inside Adam is unsafe and never retried with modified weights.
    with patch.object(O,'update',two_pass_update):return O.safe_update(*args,**kwargs)
