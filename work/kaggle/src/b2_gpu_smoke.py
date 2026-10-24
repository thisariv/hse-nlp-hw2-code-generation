"""Reviewer-only real T4 gate: allocation, throughput, peak LR, AMP, exact resume, eval/restore."""
import time
from pathlib import Path
import numpy as np
import torch
import b2_runtime as B
import b1_eval as E
import exp26_train as T
import l1_runtime as L
import l2_runtime as R
from l2_gpu_smoke import digest
from kaggle_io import write


def exercise(name,cache,order,meta,root,device,cfg,inputs,sets,refs):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);spec=B.SPECS[name];peak=B.PEAKS[name]
    report=dict(passed=False,model=name,spec=spec)
    m=B.new_model(spec,device);micro,attempts=B.probe(m,cache,device,peak,name)
    report['memory_probe']=dict(micro_batch=micro,attempts=attempts)
    o=L.optimizer(m,peak,False);s=torch.amp.GradScaler('cuda',init_scale=1024)
    state=B.new_state(micro,peak);ident=B.identity_for(name,spec,cache,meta)
    times=[];maxima=[];z_losses=[]
    torch.cuda.reset_peak_memory_stats(device)
    for j in range(256):
        if j<36:state['cursor']=((j*len(order)//36)//128)*128
        elif j==36:state['cursor']=state['updates']*128
        for a in m.modules():
            if hasattr(a,'q_norm'):a.record_logits=(j%100==0)
        log,escalate=B.main_step(m,o,s,state,cache,order,device)
        assert not escalate and not log['nonfinite_skip']
        if j%100==0:
            maxima.append(float(torch.stack([a.sampled_logit_max for a in m.modules()
                                            if getattr(a,'record_logits',False)]).max()))
        z_losses.append(log['z_loss'])
        if 4<=j<36:times.append(log['seconds'])
    for a in m.modules():
        if hasattr(a,'q_norm'):a.record_logits=False
    state['cursor']=state['updates']*128
    report.update(pairs_per_second=128/float(np.mean(times)),AMP_skips=state['AMP_skips'],
                  nonfinite_skips=state['nonfinite_skips'],parameter_count=sum(p.numel() for p in m.parameters()),
                  main_attempts=256,step_seconds=times,NaN_escalations=state['stability']['escalations'],
                  sampled_attention_abs_logit_max=max(maxima),attention_logit_samples=maxima,
                  z_loss_range=[min(z_losses),max(z_losses)],
                  training_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  training_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    B.save_full(root,m,o,s,state,spec,ident)
    B.main_step(m,o,s,state,cache,order,device)
    expected=digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state()))
    del m,o,s;R.release(device)
    m,o,s,state=B.restore(B.read_full(root,ident),device)
    B.main_step(m,o,s,state,cache,order,device)
    assert expected==digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state()))
    report['CUDA_next_update_bit_exact']=True
    state.update(updates=2000,cursor=256000)
    peak_skips=0
    for _ in range(16):
        log,escalate=B.main_step(m,o,s,state,cache,order,device)
        assert log['lr']==peak and not escalate and not log['nonfinite_skip']
        peak_skips+=log['AMP_skip']
    report.update(peak_lr_attempts=16,peak_lr_AMP_skips=peak_skips)
    before=digest((m.state_dict(),o.state_dict()['state']));scale=s.get_scale()
    hook=next(m.parameters()).register_hook(lambda g:torch.full_like(g,float('inf')))
    try:log,_=B.main_step(m,o,s,state,cache,order,device)
    finally:hook.remove()
    assert log['AMP_skip'] and s.get_scale()==scale/2 and before==digest((m.state_dict(),o.state_dict()['state']))
    report['AMP_overflow_skip']=True
    del m,o,s;R.release(device)
    payload=B.read_full(root,ident);m,o,s,state=B.restore(payload,device)
    for _ in range(12):B.main_step(m,o,s,state,cache,order,device)
    control=digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state()))
    del m,o,s;R.release(device)
    m,o,s,state=B.restore(payload,device);del payload
    # Real short cooldown using production run, then strict rollback before next updates.
    short=dict(cfg,main_seconds=0.,probe_cooldown_seconds=3.,probe_eval_seconds=120.,
               io_reserve_seconds=15.,final_session=False)
    del m,o,s;R.release(device)
    def evaluate(p,out,deadline,final):return E.evaluate(p,out,inputs,sets,refs,device,deadline,False,True)
    # B2 run's production completeness check uses this explicit mini-eval mode.
    result=B.run(name,cache,order,meta,root,device,short,evaluate,resume=root)
    assert result['status']=='complete'
    m,o,s,state=B.restore(B.read_full(root,ident),device)
    for _ in range(12):B.main_step(m,o,s,state,cache,order,device)
    assert control==digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state()))
    del m,o,s;R.release(device)
    import final_inference_v7 as V
    import s3_representation as S
    import exp26_core as C
    point=root/'weights'/(result['state']['points'][-1]['tag']+'-fp16.pt')
    m=V.load(point,device);tok,_=S.tokenizer(Path(inputs)/'tokenizer.json',30)
    rows=sets['heldout4k'][:32];src=C.source_batch(tok,[r['question'] for r in rows],device)
    begin=time.perf_counter();ctx=V.Fast.Context(m,src,128)
    tokens,_,_=V.Speed.decode(ctx,k=1,deadline=min(cfg['deadline_monotonic']-30,time.monotonic()+120),options=V.DECODE_OPTIONS)
    C.sync(device)
    report['v7_QK_generation']=dict(seconds=time.perf_counter()-begin,rows=len(rows),cap=128,
                                    model_spec_qk_norm=spec['qk_norm'],full_180s_gate=False)
    del m,src,ctx,tokens;R.release(device)
    report.update(passed=True,evaluation_restore_12_updates_bit_exact=True,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    # Keep root latest valid, compact receipt for supervisor.
    write(root/'gpu-smoke.json',report);write(root/'summary.json',dict(status='complete',smoke=report))
