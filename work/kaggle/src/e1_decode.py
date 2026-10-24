"""E1: log-linear token ensemble, teacher-forced likelihood and FP32 soups.

PAD/BOS are masked before normalization, as in v10. Ensemble probabilities
are the normalized geometric mean; EOS is an ordinary ensemble token.
"""
import math
import torch
from torch.nn import functional as F
import final_core as C
from final_speed import Decoder, race_pick


def log_probs(logits):
    logits = logits.float()
    logits[..., :2] = -torch.inf
    return F.log_softmax(logits, -1)


@torch.inference_mode()
def decode(contexts, k=1, temperature=None, deadline=None, check_every=16,
           compact=True, race=True, stats=None):
    """Independent encoder/cross/self KV per member; one shared prefix.

    Returns CPU token IDs, normalized untempered sum LP, EOS-inclusive length.
    All members compact with exactly the same active-lane index.
    """
    assert contexts and k >= 1
    first = contexts[0]
    assert all(c.capacity == first.capacity and torch.equal(c.src, first.src) for c in contexts)
    decoders = [Decoder(c, k, share_cross=True) for c in contexts]
    n, cap, device = len(first.src)*k, first.capacity, first.src.device
    token = torch.full((n,), C.BOS, dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    original = torch.arange(n, device=device)
    output = torch.full((n, cap), C.PAD, dtype=torch.long, device=device)
    sums = torch.zeros(n, device=device)
    lengths = torch.zeros(n, dtype=torch.long, device=device)
    final_sums, final_lengths = sums.clone(), lengths.clone()
    noise = None
    history = []
    for t in range(cap):
        if t % check_every == 0:
            C.check_deadline(deadline)
        # Avoid stacking M x N x V. Member logits/caches are never shared.
        pooled = None
        for d in decoders:
            lp = log_probs(d.step(token))
            if pooled is None:
                pooled = lp
            else:
                pooled.add_(lp)
        pooled.div_(len(decoders))
        # Geometric mean is unnormalized; subtract log Z for actual LP.
        lp = F.log_softmax(pooled, -1)
        if temperature is None:
            pick = pooled.argmax(-1)
        else:
            probabilities = F.softmax(pooled/temperature, -1)
            if race:
                if noise is None:
                    noise = torch.empty_like(probabilities)
                pick = race_pick(probabilities, noise)
            else:
                pick = torch.multinomial(probabilities, 1).squeeze(-1)
        sums.add_(torch.where(done, 0., lp.gather(-1, pick[:, None]).squeeze(-1)))
        lengths.add_(~done)
        token = pick.masked_fill(done, C.PAD)
        output[original, t] = token
        done |= token.eq(C.EOS)
        if (t+1) % check_every == 0 or t+1 == cap:
            snapshot = done.cpu()
            final_sums.index_copy_(0, original, sums)
            final_lengths.index_copy_(0, original, lengths)
            if bool(snapshot.all()):
                break
            keep = (~snapshot).nonzero().flatten()
            if compact and len(keep) <= len(done)*.75:
                index = keep.to(device)
                for d in decoders:
                    d.select(index)
                history.append(dict(step=t+1, before=len(done), after=len(keep)))
                token, done, sums, lengths, original = [x[index] for x in (token, done, sums, lengths, original)]
                noise = None
    final_sums.index_copy_(0, original, sums)
    final_lengths.index_copy_(0, original, lengths)
    if stats is not None:
        stats.update(steps=t+1, compactions=history, members=len(contexts), check_every=check_every)
    return (output[:, :t+1].reshape(len(first.src), k, t+1).cpu(),
            final_sums.reshape(-1, k).cpu(), final_lengths.reshape(-1, k).cpu())


@torch.inference_mode()
def teacher_score(model, src, targets):
    """targets: canonical represented-code IDs + EOS; no BOS/PAD inside.

    Scores with the same PAD/BOS mask as generation. Padding never contributes;
    EOS contributes exactly once, including for originally truncated candidates.
    """
    assert len(src) == len(targets) and targets
    assert all(t and t[-1] == C.EOS and C.EOS not in t[:-1]
               and all(x >= C.TOKEN_OFFSET for x in t[:-1]) for t in targets)
    device = src.device
    target = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(t, dtype=torch.long) for t in targets], batch_first=True, padding_value=C.PAD).to(device)
    prefix = torch.full_like(target, C.PAD)
    prefix[:, 0] = C.BOS
    prefix[:, 1:] = target[:, :-1]
    lp = log_probs(model(src, prefix))
    # Gather at PAD would yield -inf; replace BEFORE multiplying by the mask.
    selected = lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = target.ne(C.PAD)
    sums = selected.masked_fill(~mask, 0.).sum(-1)
    lengths = mask.sum(-1)
    assert bool(torch.isfinite(sums).all())
    return sums.cpu(), (sums/lengths).cpu(), lengths.cpu()


def soup_state(left, right, alpha):
    assert math.isfinite(alpha) and 0 <= alpha <= 1
    assert left.keys() == right.keys(), 'Soup keys differ'
    out = {}
    for key, a in left.items():
        b = right[key]
        assert a.shape == b.shape, 'Soup shape differs: '+key
        assert a.is_floating_point() and b.is_floating_point(), 'Unexpected nonfloating soup tensor'
        out[key] = a.float()*alpha + b.float()*(1-alpha)
        assert bool(torch.isfinite(out[key]).all()), 'Nonfinite soup: '+key
    return out
