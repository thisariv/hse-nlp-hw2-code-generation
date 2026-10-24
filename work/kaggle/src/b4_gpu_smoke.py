"""Reviewer-only real B3 resume/truncation/evaluation/restore smoke <=720s."""
import copy,time
import numpy as np
import l1_runtime as L
from pathlib import Path
import torch
import b2_runtime as B
import b4_runtime as G
import b1_eval as E
import l2_runtime as R
from b3_gpu_smoke import BranchAudit,fingerprints
from kaggle_io import write

SMOKE_UPDATES = 200


def exercise(name,cache,order,meta,root,device,cfg,inputs,sets,refs,resume,pin):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);begin=time.monotonic();q=cfg['truncation_q']
    steps=cfg.get('smoke_updates',SMOKE_UPDATES)
    assert steps>=SMOKE_UPDATES, 'B4 smoke requires hundreds of main updates on each GPU'
    p=G.read_anchor(resume,pin,cache,meta);m,o,s,state=G.restore(p,device);del p
    initial=copy.deepcopy(state);start=fingerprints(m,o,s,state)
    # Disposable memory/throughput samples throughout an estimated +9h stream,
    # plus the largest inherited aligned batch. Reload source before real smoke.
    n=(len(order)//128)*128
    lengths=cache.source_lengths.astype(np.int32)+cache.target_lengths.astype(np.int32)+1
    worst=int(np.argmax(lengths[np.asarray(order[:n])].reshape(-1,128).sum(1)))*128
    positions=[initial['cursor']+int(v)*128 for v in np.linspace(0,40000,8,dtype=int)]+[worst]
    benchmark=[];micro=state['micro_batch'];torch.cuda.reset_peak_memory_stats(device)
    for cursor in [initial['cursor']]*4+positions:
        log,micro=G.update(m,o,s,cache,L.take(order,cursor,128),device,micro,.0006,q)
        assert not log['nonfinite_skip']
        benchmark.append(dict(cursor=cursor,seconds=log['seconds'],micro_batch=micro,
                              OOM_retries=log['OOM_retries'],AMP_skip=log['AMP_skip'],
                              forward_mode=log.get('truncation',{}).get('forward_mode','ordinary')))
    benchmark=dict(samples=benchmark[4:],warmups_excluded=4,main_updates_estimate=40000,
                   largest_aligned_cursor=worst,seconds_per_update=float(np.mean([r['seconds'] for r in benchmark[4:]])),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device),two_pass_selection=bool(q))
    write(root/'sampled-throughput.json',benchmark)
    del lengths,m,o,s;R.release(device)
    mini={k:v[:64 if k=='heldout4k' else 16] for k,v in sets.items()}
    mini_refs={k:dict(rows=v['rows'][:len(mini[k])]) for k,v in refs.items()}
    def evaluate(point,out,deadline,final):return E.evaluate(point,out,inputs,mini,mini_refs,device,deadline,final,True)
    short=dict(cfg,main_seconds=9999.,max_updates=steps,
               probe_main_marks=[initial['main_seconds']+.001],probe_cooldown_seconds=2.,probe_eval_seconds=30.,
               final_cooldown_seconds=3.,final_eval_seconds=60.,io_reserve_seconds=20.)
    audit=BranchAudit(root);torch.cuda.reset_peak_memory_stats(device)
    result=G.run(name,cache,order,meta,root,device,short,evaluate,resume,pin,branch_audit=audit)
    assert result['status']=='complete' and result['final_complete'] and len(result['B4_points'])==2
    assert len(audit.rows)==2 and all(r['passed'] for r in audit.rows)
    identity=G.branch_identity(pin['identity'],q,pin['sha256'])
    m,o,s,state=G.restore(B.read_full(root,identity),device)
    assert state['updates']==initial['updates']+steps and state['cursor']==initial['cursor']+steps*128
    assert state['nonfinite_skips']==initial['nonfinite_skips'], 'Main smoke rejected a nonfinite update'
    assert state['stability']['escalations']==initial['stability']['escalations']
    assert state['last_lr']==.0006 and state['lr_factor']==1. and state['micro_batch']<=initial['micro_batch']
    assert state['points'][:len(initial['points'])]==initial['points']
    assert state['sessions'][:-1]==initial['sessions']
    if q:assert result['truncation_totals']['excluded']>0
    # AMP overflow must never modify Adam/model; the retained stable stays immutable.
    from l2_gpu_smoke import digest
    before=digest((m.state_dict(),o.state_dict()['state']));scale=s.get_scale()
    hook=next(m.parameters()).register_hook(lambda grad:torch.full_like(grad,float('inf')))
    try:log,_=G.main_step(m,o,s,state,cache,order,device,q)
    finally:hook.remove()
    assert log['AMP_skip'] and s.get_scale()==scale/2
    assert before==digest((m.state_dict(),o.state_dict()['state']))
    peak=torch.cuda.max_memory_reserved(device);del m,o,s;R.release(device)
    write(root/'gpu-smoke.json',dict(passed=True,model=name,q=q,source_checkpoint_sha256=pin['sha256'],
          start_fingerprints=start,main_updates=steps,restore_audits=audit.rows,
          nonfinite_skips=state['nonfinite_skips']-initial['nonfinite_skips'],
          AMP_skips=state['AMP_skips']-initial['AMP_skips']-log['AMP_skip'],
          constant_lr=True,AMP_overflow_skip=True,historical_state_preserved=True,mini_full_evaluation=True,
          two_pass_selection=bool(q),truncation_totals=result['truncation_totals'],
          peak_reserved_bytes=peak,seconds=time.monotonic()-begin,full_test_timer=False,
          sampled_throughput=benchmark))
