"""Bounded real-anchor CUDA growth, AMP, representative timing and v7 probes."""
import copy
import time
from pathlib import Path
import numpy as np
import torch
import exp26_core as C
import exp26_train as T
import l1_runtime as L
import l2_runtime as R
import r1_runtime as Q
import r1_growth as G
from exp26_order import collate
from l2_gpu_smoke import digest
from kaggle_io import write


def probe(anchor, branch, cache, order, root, device):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    report=dict(passed=False,checks={},device=torch.cuda.get_device_name(0))
    def check(name,**fields):
        report['checks'][name]=dict(passed=True,**fields)
        write(root/'probe.json',report);print('R1 CUDA',name,flush=True)
    # Identity in the training AMP path and pure FP16 inference, including padding.
    for mode in ('AMP','FP16') if branch['growth']=='encoder-deepen' else ():
        m,o,s,state,spec=Q.restore_anchor(anchor,dict(branch,growth='none'),device)
        src,tgt,_,_=collate(cache,L.take(order,state['cursor'],4),device)
        if mode=='FP16':m.half().eval()
        else:m.train()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=mode=='AMP'):
            before=m(src,tgt).float()
        rng=digest(T.rng_state());old_params=set(m.parameters())
        moments=digest(o.state_dict())
        spec=G.deepen(m,o,spec,encoder_layers=branch.get('encoder_layers',12))
        assert rng==digest(T.rng_state()) and old_params.issubset(set(m.parameters()))
        # Old group/state survive byte-for-byte (new group was appended).
        saved=o.state_dict();saved['param_groups']=saved['param_groups'][:1]
        assert moments==digest(saved)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=mode=='AMP'):
            after=m(src,tgt).float()
        assert torch.isfinite(after).all()
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        check('growth_identity_'+mode,max_abs=float((before-after).abs().max()),encoder_layers=spec['encoder_layers'])
        del m,o,s,src,tgt,before,after,old_params,saved;R.release(device)
    m,o,s,state,spec=Q.restore_anchor(anchor,branch,device)
    micro=min(64,state['micro_batch']);lr=state['last_lr'];ids=L.take(order,state['cursor'],128)
    # Overflow must skip old and newly inserted parameters, and back off the scaler.
    existing=list(o.state);before=digest((m.state_dict(),[o.state[p] for p in existing]));scale=s.get_scale()
    hook=next(m.parameters()).register_hook(lambda grad:torch.full_like(grad,float('inf')))
    try:log,_=G.update(m,o,s,cache,ids,device,micro,lr,1,branch['new_warmup'],branch['q'])
    finally:hook.remove()
    assert log['AMP_skip']==1 and s.get_scale()==scale/2
    assert before==digest((m.state_dict(),[o.state[p] for p in existing]))
    for p in set(o.state)-set(existing):
        assert all(torch.count_nonzero(v)==0 for v in o.state[p].values() if torch.is_tensor(v))
    del existing
    check('GradScaler_overflow',scale_before=scale,scale_after=s.get_scale())
    del m,o,s;R.release(device)
    m,o,s,state,spec=Q.restore_anchor(anchor,branch,device)
    # Same evenly-spaced weighted-order positions on both GPUs; exclude warmups.
    seconds=[];fallbacks=0;micros=[];amp_skips=0
    torch.cuda.reset_peak_memory_stats(device)
    for j in range(36):
        offset=state['cursor']+(0 if j<4 else ((j-4)*len(order)//32//128)*128)
        C.sync(device);start=time.perf_counter()
        log,micro=G.update(m,o,s,cache,L.take(order,offset,128),device,micro,lr,j+1,branch['new_warmup'],branch['q'])
        C.sync(device)
        assert not log['nonfinite_skip']
        amp_skips+=log['AMP_skip']
        if j>=4:seconds.append(time.perf_counter()-start);micros.append(micro)
        fallbacks+=int(log.get('truncation',{}).get('graph_OOM_fallback',False))
    check('representative_timing',seconds_per_step=float(np.mean(seconds)),samples=seconds,
          warmups=4,micro_batches=micros,graph_OOM_fallbacks=fallbacks,AMP_skips=amp_skips,q=branch['q'])
    report.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    # Full deep optimizer groups/scaler/RNG -> exactly the same next CUDA update.
    Q.save_full(root,m,o,s,state,spec,anchor['identity'])
    G.update(m,o,s,cache,ids,device,micro,lr,37,branch['new_warmup'],branch['q'])
    expected=digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state()))
    del m,o,s;R.release(device)
    m,o,s,state,_,_=Q.reload_latest(root,device)
    G.update(m,o,s,cache,ids,device,micro,lr,37,branch['new_warmup'],branch['q'])
    assert expected==digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state()))
    check('serialized_next_update_bit_exact')
    del m,o,s;R.release(device)
    # Probe weights are disposable; production branches start again from L3.
    for p in (root/'checkpoints').glob('*.pt'):p.unlink()
    (root/'latest.json').unlink()
    report['passed']=True;write(root/'probe.json',report)
    return report


def v7(point,cache,order,root,device):
    import final_inference_v7 as V
    m=V.load(point,device)
    spec=torch.load(point,map_location='cpu',weights_only=True)['model_spec']
    assert len(m.transformer.encoder.layers)==spec['encoder_layers']
    src,tgt,_,_=collate(cache,L.take(order,0,4),device)
    with torch.inference_mode():
        logits=m(src,tgt)
        assert torch.isfinite(logits).all()
        # Exercise v7's actual cached decoder and optimized greedy path.
        ctx=V.Fast.Context(m,src,8)
        tokens,lps,_=V.Speed.decode(ctx,k=1,deadline=time.monotonic()+30,options=V.DECODE_OPTIONS)
        assert tokens.shape[0]==4
    write(Path(root)/'v7-load.json',dict(passed=True,encoder_layers=spec['encoder_layers'],rows=4,
                                      logits_shape=list(logits.shape),tokens_shape=list(tokens.shape)))
    del m,ctx,logits,src,tgt,tokens,lps;R.release(device)
