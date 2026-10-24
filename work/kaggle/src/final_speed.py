"""Opt-in decoder experiments. Frozen final_fast/final_decode remain the control."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
import final_core as C
import final_fast as Fast


GRAPH_REFUSAL = (
    'CUDA graphs disabled: prefix length, KV slice addresses and active batch change every step. '
    'One graph per position/batch would require hundreds of captures; full-capacity masked SDPA '
    'changes reduction shapes and cannot promise bit-exact greedy. No untested graph fallback.')


@dataclass(frozen=True)
class Options:
    preallocated_kv: bool = True  # Already present in the frozen baseline; no new speedup.
    share_cross: bool = False
    async_eos: bool = False
    eos_every: int = 16
    compact_done: bool = False
    compact_fraction: float = .25
    race_sampler: bool = False
    lean_greedy: bool = False
    cuda_graphs: bool = False

    def __post_init__(self):
        if not self.preallocated_kv:
            raise ValueError('Baseline already preallocates KV; a cat regression is not an optimization')
        if self.cuda_graphs:
            raise NotImplementedError(GRAPH_REFUSAL)
        if self.eos_every < 1 or not 0 < self.compact_fraction <= 1:
            raise ValueError('Invalid EOS/compaction interval')


class Decoder(Fast.Decoder):
    @torch.inference_mode()
    def __init__(self, context, k, share_cross=False):
        if not share_cross or k != 1:
            super().__init__(context, k)
            return
        # Immutable cross KV and source mask can be shared for greedy, no replication.
        self.c, self.k, self.t, self.n = context, k, 0, len(context.src)
        self.allowed, self.cross = context.allowed, context.cross
        self.cross_bytes = sum(t.numel()*t.element_size() for pair in self.cross for t in pair)
        shape = (self.n, context.heads, context.capacity, context.dim)
        self.self_kv = [(context.positions.new_empty(shape), context.positions.new_empty(shape))
                        for _ in context.cross]
        self.valid = torch.zeros(self.n, context.capacity, dtype=torch.bool, device=context.src.device)

    @torch.inference_mode()
    def select(self, index):
        """Copy only initialized self-KV prefix, then continue at the SAME position."""
        if index.ndim != 1 or not len(index):
            raise ValueError('Nonempty 1-D active indices required')
        n, cap = len(index), self.c.capacity
        self.allowed = self.allowed.index_select(0, index)
        self.cross = [tuple(t.index_select(0, index) for t in pair) for pair in self.cross]
        # Release each old layer as soon as its replacement is ready. Holding both
        # complete caches would make compaction itself an avoidable T4 OOM peak.
        for number in range(len(self.self_kv)):
            pair = self.self_kv[number]
            new_pair = []
            for old in pair:
                new = old.new_empty((n, self.c.heads, cap, self.c.dim))
                new[:, :, :self.t].copy_(old[:, :, :self.t].index_select(0, index))
                new_pair.append(new)
            self.self_kv[number] = tuple(new_pair)
        self.n = n
        self.valid = self.valid.index_select(0, index)


def race_pick(probabilities, noise):
    """Independent Exp(1) race: argmax(p/E) has categorical probabilities p.

    Uses exactly the same FP32 softmax as baseline; RNG consumption can change.
    Avoids multinomial's probability validation/reductions. Noise storage is reused.
    """
    noise.exponential_()
    return torch.div(probabilities, noise, out=noise).argmax(-1)


@torch.inference_mode()
def decode(context, k=1, temperature=None, deadline=None, compute_logprob=True,
           options=None, stats=None):
    options = options or Options()
    d = Decoder(context, k, options.share_cross)
    device, n = context.src.device, d.n
    token = torch.full((n,), C.BOS, device=device, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    output = torch.full((n, context.capacity), C.PAD, device=device, dtype=torch.long)
    sums, lengths = torch.zeros(n, device=device), torch.zeros(n, dtype=torch.long, device=device)
    final_sums, final_lengths = torch.zeros_like(sums), torch.zeros_like(lengths)
    original = torch.arange(n, device=device)
    compacted = False
    noise = None
    host = event = None
    pending = False
    async_eos = options.async_eos and device.type == 'cuda'
    if async_eos:
        host = torch.empty(n, dtype=torch.bool, pin_memory=True)
        event = torch.cuda.Event()
    history = []
    checks = 0

    def flush():
        final_sums.index_copy_(0, original, sums)
        final_lengths.index_copy_(0, original, lengths)

    def inspect(snapshot):
        nonlocal token, done, sums, lengths, original, compacted, noise, host
        if bool(snapshot.all()):
            return True
        if not options.compact_done:
            return False
        keep = (~snapshot).nonzero().flatten()  # CPU only: never CUDA nonzero/item.
        if options.compact_done and len(keep) <= d.n*(1-options.compact_fraction):
            flush()
            index = keep.to(device)
            before = d.n
            d.select(index)
            token, done = token[index], done[index]
            sums, lengths, original = sums[index], lengths[index], original[index]
            compacted, noise = True, None
            history.append({'step': d.t, 'before': before, 'after': d.n})
            if async_eos:
                host = torch.empty(d.n, dtype=torch.bool, pin_memory=True)
        return False

    try:
        for t in range(context.capacity):
            if t % 16 == 0:
                C.check_deadline(deadline)
            if pending and event.query():
                pending = False
                if inspect(host):
                    break
            logits = d.step(token)
            if temperature is not None or compute_logprob:
                logits = logits.float()
            logits[:, :2] = -torch.inf
            if temperature is None:
                pick = logits.argmax(-1)
            else:
                probabilities = F.softmax(logits / temperature, -1)
                if options.race_sampler:
                    if noise is None:
                        noise = torch.empty_like(probabilities)
                    pick = race_pick(probabilities, noise)
                else:
                    pick = torch.multinomial(probabilities, 1).squeeze(-1)
            if compute_logprob:
                lp = F.log_softmax(logits, -1).gather(-1, pick[:, None]).squeeze(-1)
                sums.add_(torch.where(done, 0., lp))
            if compute_logprob or not options.lean_greedy:
                lengths.add_(~done)
            token = pick.masked_fill(done, C.PAD)
            if compacted:
                output[:, t].index_copy_(0, original, token)
            else:
                output[:, t] = token
            done |= token.eq(C.EOS)
            if (t+1) % options.eos_every == 0:
                if async_eos:
                    if not pending:
                        host.copy_(done, non_blocking=True)
                        event.record()
                        pending = True
                        checks += 1
                else:
                    checks += 1
                    if inspect(done.cpu()):
                        break
    finally:
        # Also retain pinned storage on deadline/OOM exceptions until copy finishes.
        if pending:
            event.synchronize()
    flush()
    if stats is not None:
        stats.update(steps=d.t, eos_checks=checks, compactions=history,
                     async_eos=async_eos, rng_stream_may_change=options.race_sampler or options.compact_done)
    return (output[:, :d.t].reshape(len(context.src), k, d.t).cpu(),
            (final_sums / final_lengths.clamp_min(1)).reshape(-1, k).cpu(), d.cross_bytes)
