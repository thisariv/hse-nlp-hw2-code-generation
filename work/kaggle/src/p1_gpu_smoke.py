"""CUDA probes for the new pilot route; run only by the independent reviewer."""
from pathlib import Path
from unittest.mock import patch
import time
import numpy as np
import torch
import exp26_train as T
import l1_runtime as L
import l2_runtime as R
from l3_resume import verify_anchor
from l2_gpu_smoke import digest,numerical
from p1_runtime import restore_anchor
from p1_math import safe_truncated_update
from kaggle_io import write


def check_anchor_update(path,pin,cache,order,device,root,cfg):
    _,anchor=verify_anchor(Path(path).parent,pin,expected_points=5)
    m,o,s,state=restore_anchor(anchor,device)
    ids=L.take(order,state['cursor'],128);micro=state['micro_batch']
    L.update(m,o,s,cache,ids,device,micro,.0015);reference=numerical(m,o,s)
    L.restore_full(anchor,m,o,s,device)
    R.safe_update(m,o,s,cache,ids,device,micro,.0015)
    assert reference==numerical(m,o,s)
    L.restore_full(anchor,m,o,s,device)
    before=numerical(m,o,s);ce=L.F.cross_entropy
    with patch.object(L.F,'cross_entropy',lambda *a,**kw:ce(*a,**kw)*float('nan')):
        log,_=R.safe_update(m,o,s,cache,ids,device,micro,.0015)
    assert log['nonfinite_skip']==1 and before==numerical(m,o,s)
    checks={}
    # Exercise BOTH selection implementations, not only the inherited A path.
    for strategy in ('graph','two-pass'):
        L.restore_full(anchor,m,o,s,device); before=numerical(m,o,s)
        with patch.object(L.F,'cross_entropy',lambda *a,**kw:ce(*a,**kw)*float('nan')):
            log,_=safe_truncated_update(m,o,s,cache,ids,device,micro,.0015,strategy=strategy)
        assert log['nonfinite_skip']==1 and before==numerical(m,o,s)
        assert all(p.grad is None for p in m.parameters())
        L.restore_full(anchor,m,o,s,device)
        before=digest((m.state_dict(),o.state_dict()));old_scale=s.get_scale()
        hook=next(m.parameters()).register_hook(lambda grad:torch.full_like(grad,float('inf')))
        try:log,_=safe_truncated_update(m,o,s,cache,ids,device,micro,.0015,strategy=strategy)
        finally:hook.remove()
        assert log['AMP_skip']==1 and s.get_scale()==old_scale/2
        assert before==digest((m.state_dict(),o.state_dict()))
        checks[strategy]=dict(nonfinite_selection_rejected=True,GradScaler_overflow_preserves_Adam=True)
    L.restore_full(anchor,m,o,s,device);before=numerical(m,o,s)
    scale=s.scale;calls=[0]
    def fail_after_backward(loss):
        calls[0]+=1
        if calls[0]==2:raise FloatingPointError('Injected after first backward')
        return scale(loss)
    with patch.object(s,'scale',fail_after_backward):
        log,_=safe_truncated_update(m,o,s,cache,ids,device,micro,.0015)
    assert log['nonfinite_skip']==1 and before==numerical(m,o,s)
    assert all(p.grad is None for p in m.parameters())
    checks['partial_backward']={'nonfinite_rejected_and_gradients_cleared':True}
    # Fixed, evenly spaced batch positions over the entire planned production stream.
    # Timings include collate, selection, finite checks and CUDA synchronization.
    offsets=np.linspace(0,31371,32,dtype=int);bench={}
    for label,update in [('a',R.safe_update),('b',safe_truncated_update)]:
        L.restore_full(anchor,m,o,s,device);times=[];modes=[];torch.cuda.reset_peak_memory_stats(device)
        for step in [-1]*4+offsets.tolist():
            batch=L.take(order,state['cursor']+max(0,step)*128,128)
            log,_=update(m,o,s,cache,batch,device,micro,.0015)
            assert not log['AMP_skip'] and not log['nonfinite_skip']
            if step>=0:
                times.append(log['seconds']);modes.append(log.get('truncation',{}).get('forward_mode','ordinary'))
        bench[label]=dict(seconds=times,updates_per_second=len(times)/sum(times),forward_modes=modes,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    bench['B_over_A_time']=sum(bench['b']['seconds'])/sum(bench['a']['seconds'])
    bench['projected_seconds']={k:31372/v['updates_per_second'] for k,v in bench.items() if k in ('a','b')}
    bench['sampling']='32 evenly spaced positions across 31372 planned updates; 4 warmups excluded'
    write(Path(root)/'pilot-throughput.json',bench)
    L.restore_full(anchor,m,o,s,device)
    # Serialize a real stable state and compare next update with restored execution.
    probe=Path(root)/'model-pilot'
    R.save_full(probe,m,o,s,state,anchor['model_spec'],anchor['identity'])
    R.safe_update(m,o,s,cache,ids,device,micro,.0015);expected=numerical(m,o,s)
    del m,o,s;R.release(device)
    m,o,s,state=R.reload_latest(probe,anchor['identity'],device)
    R.safe_update(m,o,s,cache,ids,device,micro,.0015)
    assert expected==numerical(m,o,s)
    write(Path(root)/'pilot-smoke.json',dict(passed=True,finite_update_bit_exact=True,
              serialized_resume_bit_exact=True,nonfinite_rejected=True,truncation_faults=checks,
              throughput=bench,GPU=True))
    del m,o,s,anchor;R.release(device)
