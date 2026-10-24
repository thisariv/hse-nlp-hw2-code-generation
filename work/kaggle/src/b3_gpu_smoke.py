"""Reviewer-only <=12min: real B2 CUDA resume, both branches, cooldown/eval/restore."""
import copy, time
from pathlib import Path
import torch
import b2_runtime as B
import b3_runtime as G
import b1_eval as E
import exp26_train as T
import l2_runtime as R
from l2_gpu_smoke import digest
from kaggle_io import write

SMOKE_UPDATES = 200
REPLAY_UPDATES = 8


def fingerprints(m,o,s,state=None):
    """Only CPU hashes survive; do not retain copies of the full model/Adam on GPU."""
    fields=dict(model=m.state_dict(),optimizer=o.state_dict(),scaler=s.state_dict(),
                scaler_enabled=s.is_enabled(),training={n:v.training for n,v in m.named_modules()},
                grad_enabled=torch.is_grad_enabled(),inference_mode=torch.is_inference_mode_enabled())
    fields.update({'rng_'+k:v for k,v in T.rng_state().items()})
    if state is not None: fields['state']=state
    return {k:digest(v) for k,v in fields.items()}


class BranchAudit:
    def __init__(self,root):
        self.path=Path(root)/'restore-audits.json'; self.anchors={}; self.rows=[]

    def __call__(self,phase,tag,m,o,s,state):
        actual=fingerprints(m,o,s,state)
        if phase=='before': self.anchors[tag]=actual; return
        assert phase=='restored' and tag in self.anchors
        before=self.anchors.pop(tag)
        different=[k for k in before if before[k]!=actual[k]]
        self.rows.append(dict(tag=tag,passed=not different,before=before,restored=actual,different=different))
        write(self.path,dict(passed=all(r['passed'] for r in self.rows),branches=self.rows))
        assert not different, 'B3 restore changed '+', '.join(different)


def control_step(m,o,s,state,cache,order,device):
    # Match G.run's diagnostic forward path as well as its training update.
    for a in m.modules():
        if hasattr(a,'q_norm'): a.record_logits=(state['updates']%100==0)
    return B.main_step(m,o,s,state,cache,order,device)


def exercise(name,cache,order,meta,root,device,cfg,inputs,sets,refs,resume,pin):
    root=Path(root); root.mkdir(parents=True,exist_ok=True); start=time.monotonic()
    def stage(label):
        write(root/'gpu-smoke.json',dict(passed=False,model=name,stage=label,seconds=time.monotonic()-start))
        print('B3 SMOKE',name,label,flush=True)
    stage('strict real B2 resume')
    p=G.read_anchor(resume,pin,cache,meta); initial=copy.deepcopy(p['state'])
    m,o,s,state=G.restore(p,device); del p
    torch.cuda.reset_peak_memory_stats(device)
    # No-eval control, from the actual B2 full state at constant LR.
    steps=cfg.get('smoke_updates',SMOKE_UPDATES); assert steps>=SMOKE_UPDATES
    stage('no-evaluation control '+str(steps)+' updates')
    control_AMP=0
    for i in range(steps):
        log,escalate=control_step(m,o,s,state,cache,order,device)
        assert log['lr']==B.PEAKS[name]*initial['lr_factor'] and not escalate and not log['nonfinite_skip']
        control_AMP+=log['AMP_skip']
        if i+1==REPLAY_UPDATES: prefix=fingerprints(m,o,s)
    control_state=copy.deepcopy(state)
    control=fingerprints(m,o,s)
    peak=torch.cuda.max_memory_reserved(device)
    del m,o,s; R.release(device)
    # An independent no-eval replay distinguishes CUDA variation from branch restore.
    stage('independent no-evaluation replay '+str(REPLAY_UPDATES)+' updates')
    p=G.read_anchor(resume,pin,cache,meta); m,o,s,state=G.restore(p,device); del p
    for _ in range(REPLAY_UPDATES): control_step(m,o,s,state,cache,order,device)
    replay=fingerprints(m,o,s)
    replay_different=[k for k in prefix if prefix[k]!=replay[k]]
    noeval=dict(updates=REPLAY_UPDATES,bit_exact=not replay_different,different=replay_different,
                control=prefix,replay=replay)
    write(root/'noeval-replay.json',noeval)
    del m,o,s; R.release(device)
    mini_sets={k:v[:64 if k=='heldout4k' else 16] for k,v in sets.items()}
    mini_refs={k:dict(rows=v['rows'][:len(mini_sets[k])]) for k,v in refs.items()}
    def evaluate(point,out,deadline,final): return E.evaluate(point,out,inputs,mini_sets,mini_refs,device,deadline,final,True)
    short=dict(cfg,main_seconds=9999.,max_updates=steps,probe_main_marks=[initial['main_seconds']+.001],
               probe_cooldown_seconds=2.,probe_eval_seconds=30.,final_cooldown_seconds=3.,final_eval_seconds=60.,
               io_reserve_seconds=20.)
    stage('continuation with probe and final K8')
    audit=BranchAudit(root)
    result=G.run(name,cache,order,meta,root,device,short,evaluate,resume,pin,branch_audit=audit)
    assert result['status']=='complete' and result['final_complete'] and len(result['B3_points'])==2
    m,o,s,state=G.restore(B.read_full(root,pin['identity']),device)
    assert len(audit.rows)==2 and not audit.anchors and all(r['passed'] for r in audit.rows)
    continued=fingerprints(m,o,s)
    trajectory_different=[k for k in control if control[k]!=continued[k]]
    # Independent CUDA trajectories are diagnostic, not a restore invariant.
    write(root/'trajectory-comparison.json',dict(bit_exact=not trajectory_different,different=trajectory_different,
                                                control=control,continuation=continued))
    assert state['updates']==initial['updates']+steps and state['cursor']==initial['cursor']+steps*128
    for key in ('last_lr','lr_factor'):
        assert state[key]==control_state[key],key
    assert state['nonfinite_skips']==initial['nonfinite_skips'] and state['stability']['escalations']==initial['stability']['escalations']
    assert state['points'][:len(initial['points'])]==initial['points'] and state['sessions'][:-1]==initial['sessions']
    stage('bit-exact restore passed; injected overflow and recovery')
    # Real AMP overflow on a disposable loaded state, never on the retained stable.
    before=digest((m.state_dict(),o.state_dict()['state'])); scale=s.get_scale()
    hook=next(m.parameters()).register_hook(lambda grad:torch.full_like(grad,float('inf')))
    try: log,_=B.main_step(m,o,s,state,cache,order,device)
    finally: hook.remove()
    assert log['AMP_skip'] and s.get_scale()==scale/2
    assert before==digest((m.state_dict(),o.state_dict()['state']))
    del m,o,s; R.release(device)
    # Actual rollback API, preserving time/history and emergency factor.
    h=copy.deepcopy(initial['stability'])
    m,o,s,recovered,stop=B.recover(root,pin['identity'],device,h,state['main_seconds'],initial['lr_factor'],initial['micro_batch'],'smoke injected nonfinite threshold')
    assert not stop and recovered['lr_factor']==initial['lr_factor']/2
    assert recovered['cursor']==initial['cursor']+steps*128
    del m,o,s; R.release(device)
    report=dict(passed=True,model=name,source_checkpoint_sha256=pin['sha256'],
                main_updates=steps,control_updates=steps,nonfinite_skips=0,AMP_skips=control_AMP,
                probe_final_restore_bit_exact=True,AMP_overflow_skip=True,NaN_recovery=True,constant_lr=True,
                restore_audits=audit.rows,noeval_replay=noeval,trajectory_bit_exact=not trajectory_different,
                trajectory_different=trajectory_different,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                historical_points_sessions_preserved=True,mini_full_evaluation=True,peak_reserved_bytes=peak,
                seconds=time.monotonic()-start,full_test_timer=False)
    write(root/'gpu-smoke.json',report)
