"""Headwise Q/K LayerNorm before scaled SDPA; legacy MHA parameter names retained."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def norm_q(a, x):
    return a.q_norm(x) if hasattr(a, 'q_norm') else x


def norm_k(a, x):
    return a.k_norm(x) if hasattr(a, 'k_norm') else x


class HeadLayerNorm(nn.LayerNorm):
    def forward(self, x):
        # FP32 statistics/affines also when inference casts the entire model to FP16.
        return F.layer_norm(x.float(),self.normalized_shape,self.weight.float(),self.bias.float(),self.eps).to(x.dtype)


class QKAttention(nn.MultiheadAttention):
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        assert self.batch_first and query.ndim == 3 and self._qkv_same_embed_dim
        assert not self.add_zero_attn and self.bias_k is None and self.bias_v is None
        width=self.embed_dim; n,t,_=query.shape; length=key.shape[1]; dim=width//self.num_heads
        wq,wk,wv=self.in_proj_weight.chunk(3)
        bq,bk,bv=self.in_proj_bias.chunk(3) if self.in_proj_bias is not None else (None,)*3
        def heads(x):return x.reshape(n,-1,self.num_heads,dim).transpose(1,2)
        q=norm_q(self,heads(F.linear(query,wq,bq)));k=norm_k(self,heads(F.linear(key,wk,bk)))
        v=heads(F.linear(value,wv,bv));mask=None
        if attn_mask is not None:
            mask=attn_mask
            if mask.dtype==torch.bool:mask=torch.zeros_like(mask,dtype=q.dtype).masked_fill(mask,float('-inf'))
            if mask.ndim==2:mask=mask[None,None]
            elif mask.ndim==3:mask=mask.reshape(n,self.num_heads,t,length)
        elif is_causal:
            mask=torch.full((t,length),float('-inf'),dtype=q.dtype,device=q.device).triu(1)[None,None]
        if key_padding_mask is not None:
            padding=key_padding_mask
            if padding.dtype==torch.bool:padding=torch.zeros_like(padding,dtype=q.dtype).masked_fill(padding,float('-inf'))
            padding=padding[:,None,None,:];mask=padding if mask is None else mask+padding
        if getattr(self,'record_logits',False):
            # Cheap sampled maximum: first batch, all heads, first eight queries, all keys.
            with torch.no_grad(),torch.autocast(device_type=q.device.type,enabled=False):
                sample=(q[:1,:,:8].float()@k[:1].float().transpose(-1,-2))/math.sqrt(dim)
                self.sampled_logit_max=sample.abs().amax().detach()
        p=self.dropout if self.training else 0.
        if need_weights:
            weights=(q@k.transpose(-1,-2))/math.sqrt(dim)
            if mask is not None:weights=weights+mask
            weights=F.dropout(weights.softmax(-1),p,training=self.training)
            y=weights@v;returned=weights.mean(1) if average_attn_weights else weights
        else:
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=p);returned=None
        return self.out_proj(y.transpose(1,2).contiguous().reshape(n,t,width)),returned


def install_qk_norm(transformer, flag):
    if flag in (False,None,'none'):return
    if flag not in (True,'layernorm'):raise ValueError('qk_norm must be false or layernorm')
    for layer in [*transformer.encoder.layers,*transformer.decoder.layers]:
        # Disable PyTorch's fused encoder path, which would bypass the QKAttention forward.
        if isinstance(layer,nn.TransformerEncoderLayer):layer.activation_relu_or_gelu=0
        for a in (layer.self_attn, *([layer.multihead_attn] if hasattr(layer,'multihead_attn') else [])):
            a.__class__=QKAttention
            a.q_norm=HeadLayerNorm(a.head_dim,eps=1e-5)
            a.k_norm=HeadLayerNorm(a.head_dim,eps=1e-5)
