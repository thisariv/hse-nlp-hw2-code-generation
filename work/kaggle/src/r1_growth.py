"""Identity Pre-LN encoder insertion; retain parameter objects and Adam state."""
import copy
import torch
from torch import nn
import exp26_core as C
import exp26_train as T


def deepen(model, opt, spec, seed=2026101701, encoder_layers=12):
    assert spec['encoder_layers'] == 6 and encoder_layers in (9, 12)
    assert len(model.transformer.encoder.layers) == 6
    device = next(model.parameters()).device
    rng = T.cpu(T.rng_state())
    # Use the model constructor's MHA/linear/SwiGLU recipe, without cloning trained
    # weights. Only new encoder blocks are kept. Preserve ALL training RNGs.
    try:
        torch.manual_seed(seed)
        positions = set(range(6) if encoder_layers == 12 else (1, 3, 5))
        template = C.SimpleTransformer(**dict(spec, encoder_layers=encoder_layers), dropout=0.)
        additions = [template.transformer.encoder.layers[i] for i in sorted(positions)]
        del template
        layers = []
        pending = iter(additions)
        for i, old in enumerate(model.transformer.encoder.layers):
            layers.append(old)
            if i not in positions: continue
            new = next(pending)
            assert old.norm_first and new.norm_first
            with torch.no_grad():
                for layer in (new.self_attn.out_proj, new.linear2):
                    nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
                for norm in (new.norm1, new.norm2):
                    nn.init.ones_(norm.weight); nn.init.zeros_(norm.bias)
            layers.append(new.to(device=device, dtype=next(old.parameters()).dtype))
        model.transformer.encoder.layers = nn.ModuleList(layers)
        model.transformer.encoder.num_layers = encoder_layers
        params = [p for layer in additions for p in layer.parameters()]
        opt.add_param_group(dict(params=params, lr=0., r1_new=True))
        assert all(p not in opt.state for p in params), 'New Adam moments must be lazy zero'
        return dict(spec, encoder_layers=encoder_layers)
    finally:
        T.restore_rng(rng, device)


def groups_by_name(model, opt):
    names = {id(p): name for name, p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in opt.param_groups]


def set_rates(opt, base_lr, step, warmup):
    for group in opt.param_groups:
        group['lr'] = base_lr * min(1., step / max(1, warmup)) if group.get('r1_new') else base_lr


def update(model, opt, scaler, cache, ids, device, micro, lr, step, warmup, q):
    """Set per-group rates at Adam.step, after frozen L1/P1 sets scalar LR.

    Pre-hook also runs for OOM retries; skipped AMP steps do not call Adam.
    No hook or wrapper is installed for the unchanged none/q=0 route.
    """
    import l2_runtime as R
    import p1_math as P
    from unittest.mock import patch
    handle = None
    if len(opt.param_groups) > 1:
        handle = opt.register_step_pre_hook(lambda o, args, kwargs: set_rates(o, lr, step, warmup))
    try:
        if not q:
            return R.safe_update(model, opt, scaler, cache, ids, device, micro, lr)
        mask = P.truncation_mask
        # P1 stays byte-identical; R1 exposes q through a process-local scope.
        with patch.object(P, 'truncation_mask', lambda scores, lengths: mask(scores, lengths, q=q)):
            return P.safe_truncated_update(model, opt, scaler, cache, ids, device, micro, lr)
    finally:
        if handle is not None:
            handle.remove()
            set_rates(opt, lr, step, warmup)
