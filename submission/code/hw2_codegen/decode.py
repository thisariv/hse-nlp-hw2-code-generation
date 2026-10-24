"""Быстрый v11 decoder: KV-cache, compaction, asynchronous EOS и Exp(1) sampling.

Compaction и sampling намеренно сохраняют порядок операций final_speed.py.
Одинаковые seed, batch, precision и runtime нужны для воспроизводимых samples.
"""

import math
import time
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from . import model as C


class BudgetExpired(Exception):
    """Новые batches больше не начинаются; готовые кандидаты сохраняются."""


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetExpired("Inference deadline reached")


class Context:
    """Encoder и неизменные cross-attention K/V вычисляются один раз на batch."""

    @torch.inference_mode()
    def __init__(self, model, src, capacity):
        self.model, self.src, self.capacity = (model, src, capacity)
        self.width = model.width
        self.heads = model.transformer.decoder.layers[0].self_attn.num_heads
        self.dim = self.width // self.heads
        memory = model.transformer.encoder(
            model.embed(src), src_key_padding_mask=src.eq(C.PAD)
        )
        self.allowed = src.ne(C.PAD)[:, None, None, :]
        self.positions = model.positions(torch.arange(capacity, device=src.device))
        self.cross = []
        for layer in model.transformer.decoder.layers:
            a = layer.multihead_attn
            k, v = F.linear(
                memory, a.in_proj_weight[self.width :], a.in_proj_bias[self.width :]
            ).chunk(2, -1)
            kh = (
                k.reshape(len(src), -1, self.heads, self.dim)
                .transpose(1, 2)
                .contiguous()
            )
            if hasattr(a, "k_norm"):
                kh = a.k_norm(kh)
            self.cross.append(
                (
                    kh,
                    v.reshape(len(src), -1, self.heads, self.dim)
                    .transpose(1, 2)
                    .contiguous(),
                )
            )


class BaseDecoder:
    """Один autoregressive шаг; self-attention K/V предвыделены до cap."""

    @torch.inference_mode()
    def __init__(self, context, k):
        self.c, self.k, self.t = (context, k, 0)
        self.n = len(context.src) * k
        self.allowed = context.allowed.repeat_interleave(k, dim=0)
        self.cross = [
            tuple(t.repeat_interleave(k, dim=0).contiguous() for t in pair)
            for pair in context.cross
        ]
        self.cross_bytes = sum(
            t.numel() * t.element_size() for pair in self.cross for t in pair
        )
        shape = (self.n, context.heads, context.capacity, context.dim)
        self.self_kv = [
            (context.positions.new_empty(shape), context.positions.new_empty(shape))
            for _ in context.cross
        ]
        self.valid = torch.zeros(
            self.n, context.capacity, dtype=torch.bool, device=context.src.device
        )

    def heads(self, x):
        return x.reshape(self.n, 1, self.c.heads, self.c.dim).transpose(1, 2)

    def attend(self, a, q, k, v, mask):
        assert q.ndim == k.ndim == v.ndim == 4
        if hasattr(a, "q_norm"):
            q = a.q_norm(q)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        return a.out_proj(
            y.transpose(1, 2).contiguous().reshape(self.n, 1, self.c.width)
        )

    @torch.inference_mode()
    def step(self, tokens):
        c = self.c
        t = self.t
        x = (
            c.model.tokens(tokens[:, None]) * math.sqrt(c.width)
            + c.positions[t][None, None, :]
        )
        self.valid[:, t] = tokens.ne(C.PAD)
        for layer, (sk, sv), (ck, cv) in zip(
            c.model.transformer.decoder.layers, self.self_kv, self.cross
        ):
            a = layer.self_attn
            q, k, v = F.linear(layer.norm1(x), a.in_proj_weight, a.in_proj_bias).chunk(
                3, -1
            )
            sk[:, :, t : t + 1] = (
                a.k_norm(self.heads(k)) if hasattr(a, "k_norm") else self.heads(k)
            )
            sv[:, :, t : t + 1] = self.heads(v)
            x = x + layer.dropout1(
                self.attend(
                    a,
                    self.heads(q),
                    sk[:, :, : t + 1],
                    sv[:, :, : t + 1],
                    self.valid[:, None, None, : t + 1],
                )
            )
            a = layer.multihead_attn
            q = F.linear(
                layer.norm2(x), a.in_proj_weight[: c.width], a.in_proj_bias[: c.width]
            )
            x = x + layer.dropout2(self.attend(a, self.heads(q), ck, cv, self.allowed))
            x = x + layer.dropout3(
                layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(layer.norm3(x))))
                )
            )
        if c.model.transformer.decoder.norm is not None:
            x = c.model.transformer.decoder.norm(x)
        self.t += 1
        return c.model.head(x[:, 0])


@dataclass(frozen=True)
class Options:
    """Настройки decoder all, использованные в v11."""

    share_cross: bool = True
    async_eos: bool = True
    eos_every: int = 32
    compact_done: bool = True
    compact_fraction: float = 0.25
    race_sampler: bool = True
    lean_greedy: bool = True


class Decoder(BaseDecoder):
    @torch.inference_mode()
    def __init__(self, context, k, share_cross=False):
        if not share_cross or k != 1:
            super().__init__(context, k)
            return
        self.c, self.k, self.t, self.n = (context, k, 0, len(context.src))
        self.allowed, self.cross = (context.allowed, context.cross)
        self.cross_bytes = sum(
            t.numel() * t.element_size() for pair in self.cross for t in pair
        )
        shape = (self.n, context.heads, context.capacity, context.dim)
        self.self_kv = [
            (context.positions.new_empty(shape), context.positions.new_empty(shape))
            for _ in context.cross
        ]
        self.valid = torch.zeros(
            self.n, context.capacity, dtype=torch.bool, device=context.src.device
        )

    @torch.inference_mode()
    def select(self, index):
        """Copy only initialized self-KV prefix, then continue at the SAME position."""
        if index.ndim != 1 or not len(index):
            raise ValueError("Nonempty 1-D active indices required")
        n, cap = (len(index), self.c.capacity)
        self.allowed = self.allowed.index_select(0, index)
        self.cross = [
            tuple(t.index_select(0, index) for t in pair) for pair in self.cross
        ]
        for number in range(len(self.self_kv)):
            pair = self.self_kv[number]
            new_pair = []
            for old in pair:
                new = old.new_empty((n, self.c.heads, cap, self.c.dim))
                new[:, :, : self.t].copy_(old[:, :, : self.t].index_select(0, index))
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
def decode(
    context,
    k=1,
    temperature=None,
    deadline=None,
    compute_logprob=True,
    options=None,
    stats=None,
):
    options = options or Options()
    d = Decoder(context, k, options.share_cross)
    device, n = (context.src.device, d.n)
    token = torch.full((n,), C.BOS, device=device, dtype=torch.long)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    output = torch.full((n, context.capacity), C.PAD, device=device, dtype=torch.long)
    sums, lengths = (
        torch.zeros(n, device=device),
        torch.zeros(n, dtype=torch.long, device=device),
    )
    final_sums, final_lengths = (torch.zeros_like(sums), torch.zeros_like(lengths))
    original = torch.arange(n, device=device)
    compacted = False
    noise = None
    host = event = None
    pending = False
    async_eos = options.async_eos and device.type == "cuda"
    if async_eos:
        host = torch.empty(n, dtype=torch.bool, pin_memory=True)
        event = torch.cuda.Event()
    # One temperature per stochastic slot, shared across questions. Compaction
    # must retain each sequence's original slot temperature.
    temperatures = None
    if isinstance(temperature, (list, tuple)):
        if len(temperature) != k or any(not math.isfinite(t) or t <= 0 for t in temperature):
            raise ValueError("Require K positive finite slot temperatures")
        temperatures = torch.tensor(temperature, device=device, dtype=torch.float32).repeat(len(context.src))[:, None]
    history = []
    checks = 0

    def flush():
        final_sums.index_copy_(0, original, sums)
        final_lengths.index_copy_(0, original, lengths)

    def inspect(snapshot):
        nonlocal token, done, sums, lengths, original, compacted, noise, host, temperatures
        if bool(snapshot.all()):
            return True
        if not options.compact_done:
            return False
        keep = (~snapshot).nonzero().flatten()
        if options.compact_done and len(keep) <= d.n * (1 - options.compact_fraction):
            flush()
            index = keep.to(device)
            before = d.n
            d.select(index)
            token, done = (token[index], done[index])
            sums, lengths, original = (sums[index], lengths[index], original[index])
            if temperatures is not None:
                temperatures = temperatures[index]
            compacted, noise = (True, None)
            history.append({"step": d.t, "before": before, "after": d.n})
            if async_eos:
                host = torch.empty(d.n, dtype=torch.bool, pin_memory=True)
        return False

    try:
        for t in range(context.capacity):
            if t % 16 == 0:
                check_deadline(deadline)
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
                probabilities = F.softmax(logits / (temperatures if temperatures is not None else temperature), -1)
                if options.race_sampler:
                    if noise is None:
                        noise = torch.empty_like(probabilities)
                    pick = race_pick(probabilities, noise)
                else:
                    pick = torch.multinomial(probabilities, 1).squeeze(-1)
            if compute_logprob:
                lp = F.log_softmax(logits, -1).gather(-1, pick[:, None]).squeeze(-1)
                sums.add_(torch.where(done, 0.0, lp))
            if compute_logprob or not options.lean_greedy:
                lengths.add_(~done)
            token = pick.masked_fill(done, C.PAD)
            if compacted:
                output[:, t].index_copy_(0, original, token)
            else:
                output[:, t] = token
            done |= token.eq(C.EOS)
            if (t + 1) % options.eos_every == 0:
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
        if pending:
            event.synchronize()
    flush()
    if stats is not None:
        stats.update(
            steps=d.t,
            eos_checks=checks,
            compactions=history,
            async_eos=async_eos,
            rng_stream_may_change=options.race_sampler or options.compact_done,
        )
    return (
        output[:, : d.t].reshape(len(context.src), k, d.t).cpu(),
        (final_sums / final_lengths.clamp_min(1)).reshape(-1, k).cpu(),
        d.cross_bytes,
    )


def question_batch(sequence_budget, k):
    """Лимит задан в последовательностях: batch вопросов = budget // K."""
    if not isinstance(k, int) or k < 1 or sequence_budget < k:
        raise ValueError("Require sequence_budget >= K >= 1")
    return sequence_budget // k
