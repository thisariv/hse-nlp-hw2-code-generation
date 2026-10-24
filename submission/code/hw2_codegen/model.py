"""C (6+6/d512) и W (6+6/d768, QK LayerNorm), совместимые с FP16 checkpoints."""

import math

import torch
from torch import nn
from torch.nn import functional as F

PAD, BOS, EOS, TOKEN_OFFSET = 0, 1, 2, 3
VOCAB, MAX_POSITIONS, SEED = 8195, 2048, 42


def norm_q(a, x):
    return a.q_norm(x) if hasattr(a, "q_norm") else x


def norm_k(a, x):
    return a.k_norm(x) if hasattr(a, "k_norm") else x


class HeadLayerNorm(nn.LayerNorm):
    """Статистики и affine в FP32, выход в dtype attention."""

    def forward(self, x):
        # FP32 statistics/affines also when inference casts the entire model to FP16.
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float(),
            self.bias.float(),
            self.eps,
        ).to(x.dtype)


class QKAttention(nn.MultiheadAttention):
    """Headwise LayerNorm Q/K с прежними именами MHA-параметров."""

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        assert self.batch_first and query.ndim == 3 and self._qkv_same_embed_dim
        assert not self.add_zero_attn and self.bias_k is None and self.bias_v is None
        width = self.embed_dim
        n, t, _ = query.shape
        length = key.shape[1]
        dim = width // self.num_heads
        wq, wk, wv = self.in_proj_weight.chunk(3)
        bq, bk, bv = (
            self.in_proj_bias.chunk(3) if self.in_proj_bias is not None else (None,) * 3
        )

        def heads(x):
            return x.reshape(n, -1, self.num_heads, dim).transpose(1, 2)

        q = norm_q(self, heads(F.linear(query, wq, bq)))
        k = norm_k(self, heads(F.linear(key, wk, bk)))
        v = heads(F.linear(value, wv, bv))
        mask = None
        if attn_mask is not None:
            mask = attn_mask
            if mask.dtype == torch.bool:
                mask = torch.zeros_like(mask, dtype=q.dtype).masked_fill(
                    mask, float("-inf")
                )
            if mask.ndim == 2:
                mask = mask[None, None]
            elif mask.ndim == 3:
                mask = mask.reshape(n, self.num_heads, t, length)
        elif is_causal:
            mask = torch.full(
                (t, length), float("-inf"), dtype=q.dtype, device=q.device
            ).triu(1)[None, None]
        if key_padding_mask is not None:
            padding = key_padding_mask
            if padding.dtype == torch.bool:
                padding = torch.zeros_like(padding, dtype=q.dtype).masked_fill(
                    padding, float("-inf")
                )
            padding = padding[:, None, None, :]
            mask = padding if mask is None else mask + padding
        if getattr(self, "record_logits", False):
            # Cheap sampled maximum: first batch, all heads, first eight queries, all keys.
            with torch.no_grad(), torch.autocast(
                device_type=q.device.type, enabled=False
            ):
                sample = (
                    q[:1, :, :8].float() @ k[:1].float().transpose(-1, -2)
                ) / math.sqrt(dim)
                self.sampled_logit_max = sample.abs().amax().detach()
        p = self.dropout if self.training else 0.0
        if need_weights:
            weights = (q @ k.transpose(-1, -2)) / math.sqrt(dim)
            if mask is not None:
                weights = weights + mask
            weights = F.dropout(weights.softmax(-1), p, training=self.training)
            y = weights @ v
            returned = weights.mean(1) if average_attn_weights else weights
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
            returned = None
        return (
            self.out_proj(y.transpose(1, 2).contiguous().reshape(n, t, width)),
            returned,
        )


def install_qk_norm(transformer, flag):
    """Установить QK-нормировку во всех self/cross attention без смены весов MHA."""
    if flag in (False, None, "none"):
        return
    if flag not in (True, "layernorm"):
        raise ValueError("qk_norm must be false or layernorm")
    for layer in [*transformer.encoder.layers, *transformer.decoder.layers]:
        # Disable PyTorch's fused encoder path, which would bypass the QKAttention forward.
        if isinstance(layer, nn.TransformerEncoderLayer):
            layer.activation_relu_or_gelu = 0
        for a in (
            layer.self_attn,
            *([layer.multihead_attn] if hasattr(layer, "multihead_attn") else []),
        ):
            a.__class__ = QKAttention
            a.q_norm = HeadLayerNorm(a.head_dim, eps=1e-5)
            a.k_norm = HeadLayerNorm(a.head_dim, eps=1e-5)


class SwiGLUInput(nn.Module):
    """Две проекции: SiLU(gate(x)) * value(x)."""

    def __init__(self, width, hidden):
        super().__init__()
        self.gate = nn.Linear(width, hidden)
        self.value = nn.Linear(width, hidden)

    def forward(self, x):
        return F.silu(self.gate(x)) * self.value(x)


def replace_ffn_with_swiglu(transformer, width, hidden):
    """SwiGLU вместо FFN; фиксированный локальный RNG исходного рецепта."""
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(SEED).get_state())
        for layer in list(transformer.encoder.layers) + list(
            transformer.decoder.layers
        ):
            layer.linear1 = SwiGLUInput(width, hidden)
            layer.linear2 = nn.Linear(hidden, width)
            layer.activation = nn.Identity()
            if isinstance(layer, nn.TransformerEncoderLayer):
                layer.activation_relu_or_gelu = 0
            for module in [layer.linear1.gate, layer.linear1.value, layer.linear2]:
                nn.init.xavier_uniform_(module.weight)


class SimpleTransformer(nn.Module):
    """Pre-LN encoder–decoder с learned positions и отдельной выходной головой."""

    def __init__(
        self,
        d_model=256,
        heads=4,
        encoder_layers=4,
        decoder_layers=4,
        swiglu_dim=680,
        vocab=VOCAB,
        max_positions=MAX_POSITIONS,
        dropout=0.1,
        normalized_init=True,
        qk_norm=False,
    ):
        super().__init__()
        self.width = d_model
        self.vocab = vocab
        self.tokens = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.positions = nn.Embedding(max_positions, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer.encoder.enable_nested_tensor = False
        self.transformer.encoder.use_nested_tensor = False
        self.head = nn.Linear(d_model, vocab)
        replace_ffn_with_swiglu(self.transformer, d_model, swiglu_dim)
        if qk_norm:
            install_qk_norm(self.transformer, qk_norm)
        if normalized_init:
            with torch.no_grad():
                self.tokens.weight.div_(math.sqrt(self.width))

    def embed(self, ids):
        if ids.shape[1] > self.positions.num_embeddings:
            raise ValueError("Position overflow")
        return self.tokens(ids) * math.sqrt(self.width) + self.positions(
            torch.arange(ids.shape[1], device=ids.device)
        )

    def decode_prefix(self, tgt, memory, source_padding):
        mask = torch.ones(
            tgt.shape[1], tgt.shape[1], device=tgt.device, dtype=torch.bool
        ).triu(1)
        return self.transformer.decoder(
            self.embed(tgt),
            memory,
            tgt_mask=mask,
            tgt_key_padding_mask=tgt.eq(PAD),
            memory_key_padding_mask=source_padding,
        )

    def forward(self, src, tgt):
        memory = self.transformer.encoder(
            self.embed(src), src_key_padding_mask=src.eq(PAD)
        )
        return self.head(self.decode_prefix(tgt, memory, src.eq(PAD)))

    @torch.inference_mode()
    def generate(self, src, max_new_tokens=512):
        self.eval()
        memory = self.transformer.encoder(
            self.embed(src), src_key_padding_mask=src.eq(PAD)
        )
        tgt = torch.full((len(src), 1), BOS, device=src.device, dtype=torch.long)
        done = torch.zeros(len(src), device=src.device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            logits = self.head(self.decode_prefix(tgt, memory, src.eq(PAD))[:, -1])
            logits[:, [PAD, BOS]] = -torch.inf
            token = logits.argmax(-1).masked_fill(done, PAD)
            done |= token.eq(EOS)
            tgt = torch.cat([tgt, token[:, None]], dim=1)
            if bool(done.all()):
                break
        return tgt[:, 1:]


MODEL_C = dict(
    d_model=512,
    heads=8,
    encoder_layers=6,
    decoder_layers=6,
    swiglu_dim=1368,
    vocab=VOCAB,
)
MODEL_W = dict(
    d_model=768,
    heads=12,
    encoder_layers=6,
    decoder_layers=6,
    swiglu_dim=2048,
    vocab=VOCAB,
    qk_norm="layernorm",
)


def from_spec(model_spec, dropout=0.0):
    """Создать модель по словарю из checkpoint; веса загружаются отдельно."""
    return SimpleTransformer(**model_spec, dropout=dropout)
