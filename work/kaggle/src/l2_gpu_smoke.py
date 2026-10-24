"""Reviewer-only CUDA faults around the exact production update/run/evaluation functions."""
import copy, gc, hashlib, json, time
from pathlib import Path
from unittest.mock import patch
import torch
import l1_runtime as L, l2_runtime as R, l2_eval as E, exp26_core as C, exp26_train as T
from l2_resume import initial_state
from kaggle_io import write

def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()

def numerical(model,opt,scaler):
    return digest({'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':T.rng_state()})

def exercise(gpu,name,anchor,identity,cache,order,inputs,sets,refs,device,cfg):
    assert device.type=='cuda' and not torch.is_inference_mode_enabled() and torch.is_grad_enabled()
    root=Path(cfg['working_root'])/('model-'+name);root.mkdir(parents=True,exist_ok=True)
    report={'passed':False,'GPU':True,'device':torch.cuda.get_device_name(0),'checks':{},'stage':'start'}
    def stage(label,**fields):
        report['stage']=label;report.update(fields);write(root/'gpu-smoke.json',report)
        print('CUDA SMOKE',name,label,flush=True)
    def check(label,**fields):
        report['checks'][label]=fields or {'passed':True};stage(label)
    spec=anchor['model_spec'];lr=anchor['state']['last_lr'];micro=anchor['state']['micro_batch']
    stage('longest-row memory probe')
    model=C.SimpleTransformer(**spec,dropout=0.).to(device);model.load_state_dict(anchor['model'])
    choice,attempts=L.probe(model,cache,device,lr)
    assert choice>=micro, 'Inherited micro-batch failed longest-row CUDA probe'
    report['probe']={'choice':choice,'inherited_micro_batch':micro,'attempts':attempts}
    opt=L.optimizer(model,lr,False);scaler=torch.amp.GradScaler('cuda')
    state=L.restore_full(anchor,model,opt,scaler,device)
    cursor=0 if gpu==1 else state['cursor'];ids=L.take(order,cursor,128)
    stage('L1 versus L2 finite CUDA update')
    reference=L.update(model,opt,scaler,cache,ids,device,micro,lr);expected=numerical(model,opt,scaler)
    L.restore_full(anchor,model,opt,scaler,device)
    actual,_=R.safe_update(model,opt,scaler,cache,ids,device,micro,lr)
    assert expected==numerical(model,opt,scaler) and reference['AMP_skip']==actual['AMP_skip']
    check('bit_exact_L1_CUDA_update',AMP_skip=actual['AMP_skip'],micro_batch=micro)
    L.restore_full(anchor,model,opt,scaler,device)
    before=digest((model.state_dict(),opt.state_dict()));old_scale=scaler.get_scale()
    hook=next(model.parameters()).register_hook(lambda grad:torch.full_like(grad,float('inf')))
    try:overflow,_=R.safe_update(model,opt,scaler,cache,ids,device,micro,lr)
    finally:hook.remove()
    assert overflow['AMP_skip']==1 and overflow['nonfinite_skip']==0
    assert scaler.get_scale()==old_scale/2 and before==digest((model.state_dict(),opt.state_dict()))
    check('GradScaler_overflow_skip')
    L.restore_full(anchor,model,opt,scaler,device)
    before=numerical(model,opt,scaler);ce=L.F.cross_entropy;calls=[0]
    def bad_loss(*args,**kwargs):
        calls[0]+=1;return ce(*args,**kwargs)*(float('nan') if calls[0]==2 else 1.)
    with patch.object(L.F,'cross_entropy',bad_loss):rejected,_=R.safe_update(model,opt,scaler,cache,ids,device,micro,lr)
    assert rejected['nonfinite_skip']==1 and before==numerical(model,opt,scaler)
    assert all(p.grad is None for p in model.parameters())
    check('partial_backward_nonfinite_loss_skip')
    # Serialize/restore and compare the next real CUDA update with uninterrupted execution.
    state=initial_state(anchor,name);R.save_full(root,model,opt,scaler,state,spec,identity)
    R.safe_update(model,opt,scaler,cache,ids,device,micro,lr);expected=numerical(model,opt,scaler)
    del model,opt,scaler;gc.collect();torch.cuda.empty_cache()
    model,opt,scaler,state=R.reload_latest(root,identity,device)
    R.safe_update(model,opt,scaler,cache,ids,device,micro,lr)
    assert expected==numerical(model,opt,scaler)
    check('serialized_resume_bit_exact_CUDA')
    del model,opt,scaler;gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    stage('production train cooldown eval restore train')
    calls=[0];original=R.safe_update
    def inject(*args,**kwargs):
        calls[0]+=1
        if gpu==0 and calls[0]<=3:
            with patch.object(L.F,'cross_entropy',lambda *a,**k:ce(*a,**k)*float('nan')):
                return original(*args,**kwargs)
        return original(*args,**kwargs)
    evaluate=lambda point,out,deadline,final:E.evaluate(point,out,inputs,sets,refs,device,deadline,final,
                                                         vote_cap_seconds=cfg['vote_cap_seconds'],test_rows=256)
    with patch.object(R,'safe_update',inject):
        result=R.run(name,anchor,identity,cache,order,root,device,cfg['deadline_monotonic'],evaluate,**cfg['phase_recipe'])
    assert result['status']=='complete', result['status']
    state=result['state'];assert len(state['l2_points'])==2 and all(p['complete'] for p in state['l2_points'])
    assert state['updates']-anchor['state']['updates']>=300
    assert state['stability']['escalations']==(1 if gpu==0 else 0)
    assert state['last_lr']==lr*(.5 if gpu==0 else 1.)
    check('nonfinite_escalation_full_rollback_LR' if gpu==0 else 'constant_LR_weighted_fork',
          updates=state['updates'],cursor=state['cursor'],LR=state['last_lr'],stability=state['stability'])
    for item in state['l2_points']:
        out=root/'points'/item['tag']/'evaluation';rows=json.loads((out/'heldout4k-per-example.json').read_text())
        assert len(rows)==64 and all(isinstance(r['correct'],bool) for r in rows)
        assert item['metrics']['test']['rows']==256
    final=root/'points'/state['l2_points'][-1]['tag']/'evaluation/k8-tune2048'
    vote=json.loads((final/'summary.json').read_text());assert vote['complete'] and vote['rows']>0
    journals=[r for p in sorted(final.glob('batch-*.json')) for r in json.loads(p.read_text())]
    assert len(journals)==vote['rows'] and all(len(r['samples']['0.7'])==8 for r in journals)
    assert all('execution' in c and 'mean_logprob' in c and c['sampling_status']=='complete'
               for r in journals for c in r['samples']['0.7'])
    check('K8_executed_samples_and_greedy256',rows=vote['rows'],samples=8*vote['rows'],K8_seconds=vote['seconds'],
          test_seconds=[p['metrics']['test']['full_pipeline_seconds'] for p in state['l2_points']])
    report['peak_reserved_bytes']=torch.cuda.max_memory_reserved(device)
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
    report['micro_batch']=state['micro_batch'];report['main_added_updates']=state['updates']-anchor['state']['updates']
    report['main_seconds']=state['l2_main_seconds']
    receipt=json.loads((root/'latest.json').read_text());saved=R.read_full(root/receipt['file'],identity)
    report['passed']=True;stage('all primary CUDA phases passed')
    if gpu==0:
        # Intentional nonzero exit only after all CUDA diagnostics/stable checkpoints exist.
        write(root.parent/'smoke-worker-crash.json',{'expected_exit':1,'GPU':0,'primary_checks_passed':True})
        raise RuntimeError('Reviewer injected worker failure')
    stage('wait for sibling crash, then resume healthy worker')
    marker=root.parent/'smoke-worker-crash.json'
    while not marker.exists():
        C.check_deadline(cfg['deadline_monotonic']-150);time.sleep(.1)
    old=copy.deepcopy(saved['state'])
    tail=R.run(name,saved,identity,cache,order,root,device,cfg['deadline_monotonic'],evaluate,
               main_targets=(1e6,),cooldown_seconds=(1.,),eval_seconds=(1.,),max_updates=64)
    state=tail['state'];assert state['updates']==old['updates']+64 and state['cursor']==old['cursor']+64*128
    assert state['last_lr']==old['last_lr'] and state['l2_points']==old['l2_points']
    check('healthy_worker_resumed_after_sibling_crash',continued_updates=64,cursor=state['cursor'])
    report['passed']=True;stage('complete')
    tail['status']='complete';tail['reviewer_smoke_tail_updates']=64;write(root/'summary.json',tail)
