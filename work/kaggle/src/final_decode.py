"""V3 decode loop; frozen Context/Decoder mathematics live in final_fast."""
import torch
from torch.nn import functional as F
import final_core as C


@torch.inference_mode()
def decode(context, k=1, temperature=None, deadline=None, check_every=16, compute_logprob=True):
    from final_fast import Decoder
    d = Decoder(context, k)
    device = context.src.device
    token = torch.full((d.n,), C.BOS, device=device, dtype=torch.long)
    done = torch.zeros(d.n, dtype=torch.bool, device=device)
    # KV, output, running LP/length buffers are allocated once, never concatenated.
    output = torch.full((d.n, context.capacity), C.PAD, device=device, dtype=torch.long)
    sums = torch.zeros(d.n, device=device)
    lengths = torch.zeros(d.n, dtype=torch.long, device=device)
    for t in range(context.capacity):
        if t % check_every == 0:
            C.check_deadline(deadline)
        logits = d.step(token)
        if temperature is not None or compute_logprob:
            logits = logits.float()
        logits[:, :2] = -torch.inf  # PAD/BOS: contiguous slice avoids a per-step index tensor.
        pick = logits.argmax(-1) if temperature is None else torch.multinomial(
            F.softmax(logits / temperature, -1), 1).squeeze(-1)
        if compute_logprob:
            logprob = F.log_softmax(logits, -1).gather(-1, pick[:, None]).squeeze(-1)
            sums.add_(torch.where(done, 0., logprob))
        lengths.add_(~done)
        token = pick.masked_fill(done, C.PAD)
        output[:, t] = token
        done |= token.eq(C.EOS)
        if (t + 1) % check_every == 0 and bool(done.all().cpu()):
            break
    return (output[:, :d.t].reshape(len(context.src), k, d.t).cpu(),
            (sums / lengths.clamp_min(1)).reshape(-1, k).cpu(), d.cross_bytes)
