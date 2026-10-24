"""B3 orchestration only. B2 training implementation and full-state identity stay frozen."""
import copy, json, math, time
from pathlib import Path
import torch
import b2_runtime as B
import exp26_core as C
import l1_runtime as L
import l2_runtime as R
from exp26_common import json_hash
from kaggle_io import write, sha

restore = B.restore


def validate_anchor(p, pin):
    assert p['format']=='nlp-hw02-b1-full-v1' and p['identity']==pin['identity']
    assert p['model_spec']==pin['model_spec']==p['identity']['recipe']['architecture']
    recipe = p['identity']['recipe']; state = p['state']
    assert json_hash(recipe)==p['identity']['recipe_sha256']
    assert p['stability_policy']==R.POLICY and R.finite_payload(p)
    assert all(state[k]==pin[k] for k in ('updates','cursor','main_seconds'))
    assert state['cursor']==state['updates']*128 and state['updates']>state['warmup_updates']==2000
    assert state['peak_lr']==recipe['peak_lr']==B.PEAKS[p['identity']['model']]
    assert B.learning_rate(state)==state['last_lr']==recipe['peak_lr']*state['lr_factor']
    assert p['scheduler']['last_lr']==state['last_lr'] and p['scheduler']['lr_factor']==state['lr_factor']
    assert p['scaler_enabled'] and p['scaler'] and p['rng']['torch_cuda']
    assert all(g['lr']==state['last_lr'] for g in p['optimizer']['param_groups'])
    steps = state['updates']-state['AMP_skips']-state['nonfinite_skips']
    assert p['optimizer']['state'] and all(int(v['step'])==steps for v in p['optimizer']['state'].values())


def read_anchor(root, pin, cache, meta):
    rec = json.loads((Path(root)/'latest.json').read_text())
    assert rec==pin, 'B3 requires the pinned uncooled B2 latest, no fallback'
    identity = pin['identity']
    assert identity['cache_manifest_sha256']==cache.manifest_sha
    assert identity['order_sha256']==meta['file']['sha256'] and identity['order_seed']==meta['seed']
    p = B.read_full(root,identity)  # Includes strict Torch/CUDA and every frozen implementation SHA.
    validate_anchor(p,pin)
    return p


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
        p = B.read_full(resume,pin['identity'])
    else: p = read_anchor(resume,pin,cache,meta)
    identity = p['identity']; spec = p['model_spec']; m,o,s,state = restore(p,device); del p
    origin = state['main_seconds']; origin_updates = state['updates']; historical_points = len(state['points'])
    target = origin+cfg['main_seconds']; deadline = cfg['deadline_monotonic']; reserve = cfg['io_reserve_seconds']
    marks = list(cfg.get('probe_main_marks',(37800.,48600.)))
    marks = [v for v in marks if origin<v<target]; next_mark = 0
    # The original B2 state/point/session history is retained, never reset.
    write(root/'identity.json',identity)
    write(root/'b3-origin.json',dict(source=cfg.get('resume_source'),anchor=pin,target_main_seconds=target,
                                   probe_main_marks=marks,main_pairs=state['cursor']))
    B.save_full(root,m,o,s,state,spec,identity); last_save = time.monotonic(); final_done = False; status = 'complete'
    missed = []; budget_stop = False
    def stopped(): return (root.parent/'stop-request.json').exists() or time.monotonic()>=deadline-reserve
    def branch(final, mark):
        nonlocal m,o,s,state,last_save
        tag = ('b3-final-' if final else 'b3-probe-')+f'{int(state["main_seconds"]):06d}'
        out = root/'points'/tag; out.mkdir(parents=True,exist_ok=True)
        anchor = B.save_full(root,m,o,s,state,spec,identity)
        if branch_audit is not None: branch_audit('before',tag,m,o,s,state)
        cap = cfg['final_cooldown_seconds'] if final else cfg['probe_cooldown_seconds']
        eval_cap = cfg['final_eval_seconds'] if final else cfg['probe_eval_seconds']
        peak = state['last_lr']; branch_s = 0.; done = 0; metrics = None; begin = time.monotonic()
        train = dict(session='B3',anchor=anchor,main_seconds=state['main_seconds'],main_pairs=state['cursor'],
                     target_main_seconds=mark,final=final,cooldown='C timed linear',cooldown_target_seconds=cap,
                     q=0.,start_lr=peak,main_target_seconds=target)
        try:
            assert deadline-time.monotonic()>=cap+eval_cap+reserve, 'Insufficient branch reserve'
            stability = copy.deepcopy(state['stability'])
            while branch_s<cap:
                if stopped() or time.monotonic()-begin>cap+max(30.,cap*.05): raise C.BudgetExpired('Cooldown incomplete')
                lr = peak*(1-min(1.,branch_s/cap))
                log,micro = B.O.safe_update(m,o,s,cache,L.take(order,state['cursor']+done*128,128),
                                          device,state['micro_batch'],lr)
                done+=1; branch_s+=log['seconds']; state['micro_batch']=micro
                with (out/'cooldown-updates.jsonl').open('a') as f:
                    f.write(json.dumps(dict(log,branch_update=done,cooldown_seconds=branch_s),allow_nan=False)+'\n')
                if R.observe(stability,log['nonfinite_skip'],amp_skip=log['AMP_skip']): raise R.UnsafeState('Cooldown unstable')
            train.update(completed_updates=done,cooldown_seconds=branch_s,endpoint_lr=lr)
            point = L.save_point(root,tag,m,spec,identity,train)
            del m,o,s; m=o=s=None; R.release(device)
            metrics = evaluate(point,out/'evaluation',min(deadline-reserve,time.monotonic()+eval_cap),final)
            assert metric_complete(metrics,final,cfg.get('gpu_smoke',False)), 'Incomplete B3 evaluation'
            write(out/'complete.json',dict(training=train,metrics=metrics,weights_sha256=sha(point)))
        except Exception as error:
            metrics=None; write(out/'incomplete.json',dict(reason=type(error).__name__,training=train,completed_updates=done))
        finally:
            if m is not None: del m,o,s
            R.release(device); m,o,s,state = restore(B.read_full(root,identity),device)
            # Smoke-only callback: compare the anchor itself before adding branch history.
            if branch_audit is not None: branch_audit('restored',tag,m,o,s,state)
            state['branch_seconds']+=branch_s; state['branch_attempted_updates']+=done
            state['points'].append(dict(tag=tag,training=train,complete=metrics is not None,metrics=metrics))
            B.save_full(root,m,o,s,state,spec,identity); last_save=time.monotonic()
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
                log,escalate=B.main_step(m,o,s,state,cache,order,device)
                maxima=[a.sampled_logit_max for a in m.modules() if getattr(a,'record_logits',False) and hasattr(a,'sampled_logit_max')]
                if maxima:
                    v=float(torch.stack(maxima).max()); log['sampled_attention_abs_logit_max']=v if math.isfinite(v) else None
                log.update(updates=state['updates'],main_pairs=state['cursor'],main_seconds=state['main_seconds'])
                with (root/'updates.jsonl').open('a') as f: f.write(json.dumps(log,allow_nan=False)+'\n')
            except L.UnsafeAdamOOM: unsafe='Adam OOM'
            except R.UnsafeState: unsafe='nonfinite post-Adam'
            if unsafe:
                C.sync(device); state['main_seconds']+=time.perf_counter()-begin
            if unsafe or escalate:
                h=copy.deepcopy(state['stability']); elapsed=state['main_seconds']; factor=state['lr_factor']; micro=state['micro_batch']
                with (root/'recovery-events.jsonl').open('a') as f:
                    f.write(json.dumps(dict(reason=unsafe or 'nonfinite threshold',attempted_updates=state['updates'],
                                            main_seconds=elapsed,main_pairs=state['cursor'],lr_factor=factor))+'\n')
                del m,o,s; R.release(device)
                m,o,s,state,stop=B.recover(root,identity,device,h,elapsed,factor,micro,unsafe or 'nonfinite threshold')
                B.save_full(root,m,o,s,state,spec,identity); last_save=time.monotonic()
                if stop: status='stability_stop'; break
                continue
            if state['updates']%100==0:
                write(root/'progress.json',dict(state=state,last_update=log))
                print('B3',name,state['updates'],round(state['main_seconds'],1),'pairs',state['cursor'],'LR',state['last_lr'],flush=True)
            if time.monotonic()-last_save>=cfg.get('checkpoint_seconds',900.):
                B.save_full(root,m,o,s,state,spec,identity); last_save=time.monotonic()
        B.save_full(root,m,o,s,state,spec,identity)
        missed.extend(marks[next_mark:])
        if status=='complete' and not stopped(): final_done=branch(True,target)
        new_points=state['points'][historical_points:]
        if status=='complete' and (not final_done or missed or any(not v['complete'] for v in new_points)):
            status='partial; B3 evaluation incomplete'
        state['sessions'].append(dict(session='B3',owner=cfg['owner'],origin_main_seconds=origin,
                                      end_main_seconds=state['main_seconds'],status=status,missed_points=missed,
                                      budget_stopped_main=budget_stop,final=True))
        stable=B.save_full(root,m,o,s,state,spec,identity)
        result=dict(status=status,model=name,state=state,identity=identity,final_complete=final_done,
                    origin_main_seconds=origin,main_target_seconds=target,main_target_reached=state['main_seconds']>=target,
                    budget_stopped_main=budget_stop,missed_points=missed,B4_uncooled_latest=stable,
                    B3_points=[v['tag'] for v in new_points])
        write(root/'summary.json',result); return result
    finally:
        del m,o,s; R.release(device)
