#!/usr/bin/env python3
"""CPU relative full training update cost, real C6/6 vs grown C12/6."""
import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import l1_runtime as L
import r1_growth as G
from l1_cache import CompactCache
from l2_sampler import load_order
from kaggle_io import write
from exp26_order import collate


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--order',type=Path,required=True);p.add_argument('--repeats',type=int,default=5)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);d=torch.device('cpu')
    cache=CompactCache(a.cache,verify=False)
    pins=json.loads((ROOT/'r1-anchors.json').read_text());pin=pins['anchors']['C-long2x']
    order,_=load_order(a.order,cache,pin['identity']['order_sha256'])
    # Same next real order positions; no synthetic targets and no heldout/test.
    ids=L.take(order,pin['cursor'],16); results={}
    for growth in ('none','encoder-deepen'):
        m=L.new_model(L.SPECS['C'],d);o=L.optimizer(m,.0015,True);o.defaults['fused']=False
        for g in o.param_groups:g['fused']=False
        s=torch.amp.GradScaler('cpu',enabled=False)
        identity=None
        if growth!='none':
            src,tgt,_,_=collate(cache,ids,d)
            with torch.no_grad():before=m(src,tgt)
            G.deepen(m,o,L.SPECS['C'])
            with torch.no_grad():after=m(src,tgt)
            difference=float((before-after).abs().max())
            assert difference<=1e-6
            identity=dict(bit_exact=torch.equal(before,after),max_abs=difference,model='real C512 6/6 -> 12/6')
            del before,after,src,tgt
        # Both use q=.1 and the same global graph objective. Warm Adam allocation.
        timings=[]
        for j in range(a.repeats+3):
            start=time.perf_counter();log,_=G.update(m,o,s,cache,ids,d,16,.0015,1001+j,1000,.1)
            elapsed=time.perf_counter()-start
            if j>=3:timings.append(elapsed)
            print('CPU R1 benchmark',growth,j,round(elapsed,3),flush=True)
        results[growth]=dict(seconds=timings,median_seconds=statistics.median(timings),
                             parameters=sum(p.numel() for p in m.parameters()),function_identity=identity)
        del m,o,s;gc.collect()
    ratio=results['encoder-deepen']['median_seconds']/results['none']['median_seconds']
    t4_base=1/2.573 # observed net L3 C-long2x updates / main seconds
    report=dict(passed=True,device='CPU',torch=str(torch.__version__),threads=1,effective_batch=16,
                source_lengths=cache.source_lengths[ids].astype(int).tolist(),
                target_lengths=cache.target_lengths[ids].astype(int).tolist(),q=.1,repeats=a.repeats,warmup_updates=3,
                results=results,relative_step_cost=ratio,T4_baseline_step_seconds=t4_base,
                T4_estimated_deepen_seconds=t4_base*ratio,
                limitation='CPU FP32/batch16 -> T4 FP16/batch128 ratio is an estimate; memory/fused/graph fallback unmeasured',GPU=False)
    write(ROOT.parent/'checks/kaggle-r1-cpu-benchmark.json',report)
    print('Relative deepen cost',ratio,flush=True)


if __name__=='__main__':
    import fcntl,tempfile
    with (Path(tempfile.gettempdir())/'nlp-r1-cpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);main()
