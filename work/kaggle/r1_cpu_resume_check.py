#!/usr/bin/env python3
"""Real L3 -> R1 none: FP32 CPU, inherited enabled scaler, exact two updates."""
import argparse
import contextlib
import gc
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
import r1_growth as G
import r1_runtime as Q
from l1_cache import CompactCache
from l2_resume import cpu_restore
from l2_sampler import load_order
from l3_resume import verify_anchor
from l2_gpu_smoke import digest
from kaggle_io import write
from build_r1 import default_branches


def main():
    p=argparse.ArgumentParser()
    for name in ('anchor-root','cache','weighted-order'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model',choices=['C-main','C-long2x'],required=True)
    p.add_argument('--keep-anchor',action='store_true',help='Only for streaming transfer; uploader deletes immediately')
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);d=torch.device('cpu');start=time.monotonic()
    pin=json.loads((ROOT/'r1-anchors.json').read_text())['anchors'][a.model]
    file,anchor=verify_anchor(a.anchor_root,pin,False,5)
    cache=CompactCache(a.cache,verify=False)
    order,_=load_order(a.weighted_order if a.model=='C-long2x' else a.cache,cache,pin['identity']['order_sha256'])
    initial={k:digest(anchor[k]) for k in ('model','optimizer','scaler','state','rng')}
    m=C.SimpleTransformer(**anchor['model_spec'],dropout=0.)
    o=L.optimizer(m,.0015,True);state=cpu_restore(anchor,m,o)
    s=torch.amp.GradScaler('cpu',enabled=True);s.load_state_dict(anchor['scaler'])
    def record(m,o,s,state):return dict(model=digest(m.state_dict()),optimizer=digest(o.state_dict()),
                                      scaler=digest(s.state_dict()),rng=digest(T.rng_state()),state=digest(state))
    reference=[];actual=[]
    def advance(state,log,micro):
        state['micro_batch']=micro;state['updates']+=1;state['cursor']+=128
        state['AMP_skips']+=log['AMP_skip'];state['nonfinite_skips']+=log.get('nonfinite_skip',0)
    with patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
        for j in (1,2):
            log,micro=R.safe_update(m,o,s,cache,L.take(order,state['cursor'],128),d,state['micro_batch'],.0015)
            assert not log['AMP_skip'] and not log['nonfinite_skip'];advance(state,log,micro)
            reference.append(record(m,o,s,state));print('R1 reference',a.model,j,round(log['seconds'],2),flush=True)
        # Adam.load_state_dict may share CPU moment tensors with its payload.
        # Re-read immutable source bytes; never reuse a reference-updated payload.
        del m,o,s,anchor;gc.collect()
        _,anchor=verify_anchor(a.anchor_root,pin,False,5)
        m,o,s,state,_=Q.restore_anchor(anchor,default_branches()[1],d,True)
        restored=record(m,o,s,state)
        for k in ('model','scaler','state'):assert restored[k]==initial[k]
        normalized=__import__('copy').deepcopy(anchor['optimizer'])
        for group in normalized['param_groups']:group['fused']=False
        assert restored['optimizer']==digest(normalized)
        for j in (1,2):
            log,micro=G.update(m,o,s,cache,L.take(order,state['cursor'],128),d,state['micro_batch'],.0015,j,1000,0)
            assert not log['AMP_skip'] and not log['nonfinite_skip'];advance(state,log,micro)
            actual.append(record(m,o,s,state));print('R1 route',a.model,j,round(log['seconds'],2),flush=True)
    assert actual==reference
    del m,o,s,anchor;gc.collect()
    if not a.keep_anchor:file.unlink()
    write(ROOT.parent/f'checks/kaggle-r1-real-resume-{a.model}.json',dict(passed=True,model=a.model,
          source='friend1/nlp-hw02-l3-live/12',anchor_sha256=pin['sha256'],anchor_bytes=pin['bytes'],
          initial_digests=initial,steps=actual,steps_bit_exact=2,effective_batch=128,
          downloaded_anchor_removed=not file.exists(),kept_only_for_immediate_streaming_upload=a.keep_anchor,
          seconds=time.monotonic()-start,GPU=False,
          limitation='CPU FP32 / autocast disabled / enabled inherited scaler / fused disabled; CUDA pending'))
    print('R1 REAL NONE RESUME PASSED',a.model,flush=True)


if __name__=='__main__':
    import fcntl,tempfile
    with (Path(tempfile.gettempdir())/'nlp-r1-cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);main()
