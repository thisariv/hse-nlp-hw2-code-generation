#!/usr/bin/env python3
"""One downloaded L3 full state: exact restore and next two batch128 P1 updates."""
import argparse
import contextlib
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import torch
import exp26_core as C
import exp26_train as T
import l1_runtime as L
import l2_runtime as R
import p1_runtime as P
from p1_math import sqrt_lr
from l1_cache import CompactCache
from l2_resume import cpu_restore
from l2_sampler import load_order
from l3_resume import verify_anchor
from l2_gpu_smoke import digest
from kaggle_io import write


def main():
    p=argparse.ArgumentParser()
    for name in ('anchor-root','cache','weighted-order','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);device=torch.device('cpu')
    pin=json.loads((ROOT/'l4-anchors.json').read_text())['anchors']['C-long2x']
    file,anchor=verify_anchor(a.anchor_root,pin,False,5);start=time.monotonic()
    initial={k:digest(anchor[k]) for k in ('model','optimizer','scaler','rng','state')}
    cache=CompactCache(a.cache,verify=False)
    order,meta=load_order(a.weighted_order,cache,pin['identity']['order_sha256'])
    assert meta==anchor['identity']['recipe']['sampler'] and cache.manifest_sha==pin['identity']['cache_manifest_sha256']
    reference=[];actual=[];length=round(.15*pin['updates'])
    def record(m,o,s,ids):
        return dict(model=digest(m.state_dict()),optimizer=digest(o.state_dict()),scaler=digest(s.state_dict()),
                    rng=digest(T.rng_state()),indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
    m=C.SimpleTransformer(**anchor['model_spec'],dropout=0.)
    o=L.optimizer(m,.0015,True);state=cpu_restore(anchor,m,o)
    s=torch.amp.GradScaler('cpu',enabled=True);s.load_state_dict(anchor['scaler'])
    with patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
        for step in (1,2):
            ids=L.take(order,state['cursor']+(step-1)*128,128)
            log,_=L.resilient_update(m,o,s,cache,ids,device,state['micro_batch'],sqrt_lr(step,length))
            assert not log['AMP_skip'];reference.append(record(m,o,s,ids))
            print('P1 CPU reference',step,round(log['seconds'],2),flush=True)
    del m,o,s,anchor;gc.collect()
    _,anchor=verify_anchor(a.anchor_root,pin,False,5)
    original=L.restore_full;old_scaler=torch.amp.GradScaler;restores=[]
    def restore(payload,m,o,s,d):
        state=original(payload,m,o,s,d)
        restored=dict(model=digest(m.state_dict()),optimizer=digest(o.state_dict()),scaler=digest(s.state_dict()),state=digest(state))
        assert all(restored[k]==initial[k] for k in restored)
        restores.append(restored)
        for group in o.param_groups:group['fused']=False
        return state
    with patch.object(L,'restore_full',restore),patch.object(torch.amp,'GradScaler',
         lambda *args,**kw:old_scaler('cpu',enabled=True,init_scale=1024)),\
         patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
        m,o,s,state=P.restore_anchor(anchor,device,cpu=True)
        for step in (1,2):
            ids=L.take(order,state['cursor']+(step-1)*128,128)
            log,_=R.safe_update(m,o,s,cache,ids,device,state['micro_batch'],sqrt_lr(step,length))
            assert not log['AMP_skip'] and not log['nonfinite_skip'];actual.append(record(m,o,s,ids))
            print('P1 CPU route',step,round(log['seconds'],2),flush=True)
        # A and B each start at the exact same original full state, including RNG.
        del m,o,s,anchor;gc.collect()
        _,anchor=verify_anchor(a.anchor_root,pin,False,5)
        m,o,s,state=P.restore_anchor(anchor,device,cpu=True)
    assert actual==reference and len(restores)==2
    assert state==anchor['state'] and state['cursor']==20734592
    del m,o,s,anchor;gc.collect();file.unlink()
    write(a.out,dict(passed=True,source='friend1/nlp-hw02-l3-live/12',anchor_sha256=pin['sha256'],
          anchor_bytes=pin['bytes'],initial_full_state_digests=initial,restores=restores,steps=actual,
          steps_bit_exact=2,effective_batch=128,pilot_updates=length,full_restore_before_both_branches=True,
          downloaded_anchor_removed=not file.exists(),seconds=time.monotonic()-start,GPU=False,
          limitation='CPU FP32/no autocast, enabled CPU GradScaler from L3, fused disabled; CUDA smoke still required'))
    print('P1 REAL RESUME PASSED; downloaded PT removed',flush=True)


if __name__=='__main__':main()
