"""B2 token-weighted CE + FP32 z-loss, same Adam/AMP/OOM/nonfinite policy as B1."""
import copy,gc,time
import torch
from torch.nn import functional as F
import exp26_core as C
import exp26_train as T
import l1_runtime as L
from exp26_order import collate
from l1_runtime import MICRO_BATCHES, UnsafeAdamOOM
from l2_runtime import UnsafeState


def loss_terms(logits, labels, tokens, coefficient=1e-4):
    flat=logits.float().reshape(-1,logits.shape[-1]); labels=labels.reshape(-1)
    ce=F.cross_entropy(flat,labels,ignore_index=C.PAD,reduction='sum')/tokens
    z=coefficient*torch.logsumexp(flat,dim=-1).square().masked_fill(labels.eq(C.PAD),0.).sum()/tokens
    return ce,z

def update(model,opt,scaler,cache,indices,device,micro,lr):
    """An exact effective batch; 96+32 and 64+64 have token-weighted CE."""
    C.sync(device); started=time.perf_counter(); opt.zero_grad(set_to_none=True)
    tokens=sum(len(cache[int(i)][1])+1 for i in indices); total=0.; z_total=0.
    for lo in range(0,len(indices),micro):
        src,tgt,labels,counts=collate(cache,indices[lo:lo+micro],device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=scaler.is_enabled()):
            logits=model(src,tgt)
            ce,z=loss_terms(logits,labels,tokens);loss=ce+z
        if not bool(torch.isfinite(loss).detach()):raise FloatingPointError('Nonfinite loss')
        scaler.scale(loss).backward(); total+=float(ce.detach()); z_total+=float(z.detach()); del src,tgt,labels,logits,loss,ce,z
    for group in opt.param_groups:group['lr']=lr
    scaler.unscale_(opt); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    skipped=bool(any(float(t)>0 for t in scaler._found_inf_per_device(opt).values())) if scaler.is_enabled() else False
    if not skipped and not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite successful gradient')
    try:
        scaler.step(opt)
    except torch.cuda.OutOfMemoryError as error:
        # Adam may have modified only some parameters. Never retry this state.
        error.optimizer_may_be_modified=True
        raise
    scaler.update(); C.sync(device)
    return {'seconds':time.perf_counter()-started,'CE':total,'z_loss':z_total,'AMP_skip':int(skipped),'lr':lr,
            'pairs':len(indices),'tokens':tokens,'gradient_norm':float(norm) if torch.isfinite(norm) else None}


def resilient_update(model,opt,scaler,cache,indices,device,micro,lr):
    rng=T.cpu(T.rng_state());scale=copy.deepcopy(scaler.state_dict());attempts=[]
    while True:
        try:
            result=update(model,opt,scaler,cache,indices,device,micro,lr)
            result['OOM_retries']=attempts
            return result,micro
        except torch.cuda.OutOfMemoryError as error:
            unsafe=getattr(error,'optimizer_may_be_modified',False)
        # The exception frame (and its CUDA tensors) must be gone before retry.
        model.zero_grad(set_to_none=True);gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        if unsafe:raise UnsafeAdamOOM('OOM inside Adam: restore last committed full checkpoint')
        smaller=[m for m in MICRO_BATCHES if m<micro]
        if not smaller:raise RuntimeError('OOM even at micro-batch1; last checkpoint preserved')
        attempts.append({'micro_batch':micro,'status':'late OOM before optimizer step'})
        micro=smaller[0];scaler.load_state_dict(scale)
        if scaler.is_enabled():scaler._per_optimizer_states.clear()
        T.restore_rng(rng,device)
        print('OOM RETRY micro',micro,flush=True)


def safe_update(model, opt, scaler, cache, indices, device, micro, lr):
    """Never call Adam after rejected loss/gradient. Normal AMP skips stay normal."""
    started = time.perf_counter(); scale = copy.deepcopy(scaler.state_dict())
    try:
        log, micro = resilient_update(model, opt, scaler, cache, indices, device, micro, lr)
        tensors = list(model.parameters()) + [v for s in opt.state.values() for v in s.values()
                                              if torch.is_tensor(v) and v.is_floating_point()]
        norms = torch.stack(torch._foreach_norm(tensors))
        if not bool(torch.isfinite(norms).all()): raise UnsafeState('Nonfinite model/Adam after update')
        log['nonfinite_skip'] = 0
        log['seconds'] = time.perf_counter()-started
        return log, micro
    except FloatingPointError:
        opt.zero_grad(set_to_none=True)
        scaler.load_state_dict(scale)
        if scaler.is_enabled(): scaler._per_optimizer_states.clear()
        for group in opt.param_groups: group['lr'] = lr
        C.sync(device)
        return {'seconds': time.perf_counter()-started, 'CE': None, 'AMP_skip': 0, 'nonfinite_skip': 1,
                'lr': lr, 'pairs': len(indices), 'tokens': 0, 'gradient_norm': None, 'OOM_retries': []}, micro

