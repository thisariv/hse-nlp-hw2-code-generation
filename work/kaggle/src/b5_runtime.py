"""Independent W/B4 and C/L3 continuations; one disposable probe and linear final."""
import copy, json, math, time
from pathlib import Path
import torch
import b2_runtime as B
import b5_state as S
import hashlib
import numpy as np
import exp26_core as C
import l1_runtime as L
import l2_runtime as R
from exp26_common import json_hash
from kaggle_io import write, sha

restore = S.restore
read_anchor = S.read_anchor
main_step = S.main_step


def final_reserve(cfg):
    return cfg['final_cooldown_seconds']+cfg['final_eval_seconds']+cfg['io_reserve_seconds']+60.


def metric_complete(metrics, final=False, smoke=False):
    if not (final and smoke): return B.metric_complete(metrics,final,smoke)
    from l2_sampler import fixed_split
    # The first 64 heldout IDs contain 36 frozen tune IDs, not 32.
    expected=sum(i<64 for i in fixed_split()['tune2048'])
    try:
        vote=metrics['K8_tune2048']
        return (B.metric_complete(metrics,False,True) and vote['complete'] is True
                and vote['rows']==vote['target_rows']==expected and metrics['test']['rows']==64)
    except (KeyError,TypeError,ValueError): return False


def run(name, cache, order, meta, root, device, cfg, evaluate, resume, pin, cpu=False, branch_audit=None):
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    if cpu:
        # Tiny regression fixtures use their own strict B2 CPU identity.
        p = S.read_full(resume,pin['identity'],environment=False)
    else: p = read_anchor(resume,pin,cache,meta)
    identity = p['identity']; spec = p['model_spec']; m,o,s,state = restore(p,device); del p
    origin = state['main_seconds']; origin_updates = state['updates']; historical_points = len(state.get('b5_points',[]))
    from b3_gpu_smoke import fingerprints
    write(root/'start-fingerprints.json',dict(source_sha256=pin['sha256'],fingerprints=fingerprints(m,o,s,state)))
    state.setdefault('b5_points',[]);state.setdefault('b5_sessions',[])
    stream=hashlib.sha256()
    target = origin+cfg['main_seconds']; deadline = cfg['deadline_monotonic']; reserve = cfg['io_reserve_seconds']
    marks = list(cfg.get('probe_main_marks',[origin+16200.]))
    marks = [v for v in marks if origin<v<target]; next_mark = 0
    # The original B2 state/point/session history is retained, never reset.
    write(root/'identity.json',identity)
    write(root/'b5-origin.json',dict(source=cfg.get('resume_source'),anchor=pin,target_main_seconds=target,
                                   probe_main_marks=marks,main_pairs=state['cursor']))
    S.save_full(root,m,o,s,state,spec,identity); last_save = time.monotonic(); final_done = False; status = 'complete'
    missed = []; budget_stop = False
    def stopped(): return (root.parent/'stop-request.json').exists() or time.monotonic()>=deadline-reserve
    def branch(final, mark):
        nonlocal m,o,s,state,last_save
        tag = ('b5-final-' if final else 'b5-probe-')+f'{int(state["main_seconds"]):06d}'
        out = root/'points'/tag; out.mkdir(parents=True,exist_ok=True)
        anchor = S.save_full(root,m,o,s,state,spec,identity)
        if branch_audit is not None: branch_audit('before',tag,m,o,s,state)
        cap = cfg['final_cooldown_seconds'] if final else cfg['probe_cooldown_seconds']
        eval_cap = cfg['final_eval_seconds'] if final else cfg['probe_eval_seconds']
        peak = state['last_lr']; branch_s = 0.; done = 0; metrics = None; begin = time.monotonic()
        train = dict(session='B5',anchor=anchor,main_seconds=state['main_seconds'],main_pairs=state['cursor'],
                     target_main_seconds=mark,final=final,cooldown='C timed linear',cooldown_target_seconds=cap,
                     q=0.,start_lr=peak,main_target_seconds=target)
        try:
            assert deadline-time.monotonic()>=cap+eval_cap+reserve, 'Insufficient branch reserve'
            stability = copy.deepcopy(state['stability'])
            while branch_s<cap:
                if stopped() or time.monotonic()-begin>cap+max(30.,cap*.05): raise C.BudgetExpired('Cooldown incomplete')
                lr = peak*(1-min(1.,branch_s/cap))
                log,micro = S.update_for(identity,m,o,s,cache,L.take(order,state['cursor']+done*128,128),
                                          device,state['micro_batch'],lr)
                done+=1; branch_s+=log['seconds']; state['micro_batch']=micro
                with (out/'cooldown-updates.jsonl').open('a') as f:
                    f.write(json.dumps(dict(log,branch_update=done,cooldown_seconds=branch_s),allow_nan=False)+'\n')
                if R.observe(stability,log['nonfinite_skip'],amp_skip=log['AMP_skip']): raise R.UnsafeState('Cooldown unstable')
            train.update(completed_updates=done,cooldown_seconds=branch_s,endpoint_lr=lr)
            point = L.save_point(root,tag,m,spec,identity,train)
            del m,o,s; m=o=s=None; R.release(device)
            metrics = evaluate(point,out/'evaluation',min(deadline-reserve,time.monotonic()+eval_cap),final)
            assert metric_complete(metrics,final,cfg.get('gpu_smoke',False)), 'Incomplete B5 evaluation'
            write(out/'complete.json',dict(training=train,metrics=metrics,weights_sha256=sha(point)))
        except Exception as error:
            metrics=None; write(out/'incomplete.json',dict(reason=type(error).__name__,training=train,completed_updates=done))
        finally:
            if m is not None: del m,o,s
            R.release(device); m,o,s,state = restore(S.read_full(root,identity,environment=not cpu),device)
            # Smoke-only callback: compare the anchor itself before adding branch history.
            if branch_audit is not None: branch_audit('restored',tag,m,o,s,state)
            state['b5_branch_seconds']=state.get('b5_branch_seconds',0.)+branch_s
            state['b5_branch_attempted_updates']=state.get('b5_branch_attempted_updates',0)+done
            state['b5_points'].append(dict(tag=tag,training=train,complete=metrics is not None,metrics=metrics))
            S.save_full(root,m,o,s,state,spec,identity); last_save=time.monotonic()
        return metrics is not None
    try:
        while state['main_seconds']<target and not stopped():
            if cfg.get('max_updates') and state['updates']-origin_updates>=cfg['max_updates']: break
            if deadline-time.monotonic()<final_reserve(cfg): budget_stop=True; break
            if next_mark<len(marks) and state['main_seconds']>=marks[next_mark]:
                mark=marks[next_mark]; next_mark+=1
                needed = cfg['probe_cooldown_seconds']+cfg['probe_eval_seconds']+final_reserve(cfg)+120.
                if deadline-time.monotonic()>=needed: branch(False,mark)
                else: missed.append(mark)
                continue
            unsafe=None; escalate=False; begin=time.perf_counter()
            try:
                for a in m.modules():
                    if hasattr(a,'q_norm'): a.record_logits=(state['updates']%100==0)
                ids=L.take(order,state['cursor'],128)
                log,escalate=main_step(m,o,s,state,cache,order,device,identity)
                maxima=[a.sampled_logit_max for a in m.modules() if getattr(a,'record_logits',False) and hasattr(a,'sampled_logit_max')]
                if maxima:
                    v=float(torch.stack(maxima).max()); log['sampled_attention_abs_logit_max']=v if math.isfinite(v) else None
                stream.update(np.asarray(ids,dtype='<u4').tobytes())
                log.update(stream_sha256=stream.hexdigest(),updates=state['updates'],main_pairs=state['cursor'],main_seconds=state['main_seconds'])
                with (root/'updates.jsonl').open('a') as f: f.write(json.dumps(log,allow_nan=False)+'\n')
            except L.UnsafeAdamOOM: unsafe='Adam OOM'
            except R.UnsafeState: unsafe='nonfinite post-Adam'
            if unsafe:
                C.sync(device); state['main_seconds']+=time.perf_counter()-begin
            if unsafe or escalate:
                # Keep the requested LR and historical policy; stop on last stable.
                write(root/'recovery-stop.json',dict(reason=unsafe or 'nonfinite threshold',constant_lr_preserved=True))
                del m,o,s;R.release(device)
                m,o,s,state=restore(S.read_full(root,identity,environment=not cpu),device)
                status='partial; stability_stop';break
            if state['updates']%100==0:
                write(root/'progress.json',dict(state=state,last_update=log))
                print('B5',name,state['updates'],round(state['main_seconds'],1),'pairs',state['cursor'],'LR',state['last_lr'],flush=True)
            if time.monotonic()-last_save>=cfg.get('checkpoint_seconds',900.):
                S.save_full(root,m,o,s,state,spec,identity); last_save=time.monotonic()
        S.save_full(root,m,o,s,state,spec,identity)
        missed.extend(marks[next_mark:])
        if status=='complete' and not stopped(): final_done=branch(True,target)
        new_points=state['b5_points'][historical_points:]
        if status=='complete' and (not final_done or missed or any(not v['complete'] for v in new_points)
                or (state['main_seconds']<target and not cfg.get('max_updates'))):
            status='partial; B5 evaluation incomplete'
        state['b5_sessions'].append(dict(session='B5',owner=cfg['owner'],origin_main_seconds=origin,
                                      end_main_seconds=state['main_seconds'],status=status,missed_points=missed,
                                      budget_stopped_main=budget_stop,final=True))
        stable=S.save_full(root,m,o,s,state,spec,identity)
        result=dict(status=status,model=name,state=state,identity=identity,final_complete=final_done,
                    origin_main_seconds=origin,main_target_seconds=target,main_target_reached=state['main_seconds']>=target,
                    budget_stopped_main=budget_stop,missed_points=missed,B6_uncooled_latest=stable,
                    stream=dict(origin_cursor=pin['cursor'],attempted_batches=state['updates']-origin_updates,sha256=stream.hexdigest()),
                    B5_points=[v['tag'] for v in new_points])
        write(root/'summary.json',result); return result
    finally:
        del m,o,s; R.release(device)
