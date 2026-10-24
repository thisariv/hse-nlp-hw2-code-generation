"""Only fixed-N q0 cooldown; full periodic recovery, FP16 midpoint and bounded final evaluation."""
import copy, hashlib, json, time
from pathlib import Path
import numpy as np
import b5_state as S
import b5_runtime as B5
import b6_state as D
import exp26_core as C
import l1_runtime as L
import l2_runtime as R
from kaggle_io import sha, write


def run(name,cache,order,meta,root,device,cfg,evaluate,resume,pin,cpu=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    p=(S.read_full(resume,pin['identity'],environment=False) if cpu else S.read_anchor(resume,pin,cache,meta))
    identity=p['identity'];spec=p['model_spec'];m,o,s,state=S.restore(p,device);del p
    from b3_gpu_smoke import fingerprints
    write(root/'start-fingerprints.json',dict(source_sha256=pin['sha256'],fingerprints=fingerprints(m,o,s,state)))
    cd=D.new_cooldown(state,cfg['N_by_gpu'][cfg['gpu']],pin['sha256'])
    write(root/'identity.json',identity)
    write(root/'b6-origin.json',dict(anchor=pin,N=cd['N'],q=0.,main_added_updates=0,schedule=cd['schedule']))
    deadline=cfg['deadline_monotonic'];reserve=cfg['io_reserve_seconds'];eval_cap=cfg['final_eval_seconds']
    train_until=deadline-eval_cap-reserve
    stream=hashlib.sha256();status='partial';reason=None;metrics=None;point=None;final_complete=False;midpoint=None
    last_save=time.monotonic();checkpoint=None;start=time.monotonic()
    def save():return D.save_full(root,m,o,s,state,spec,identity,cd)
    def training():
        return dict(session='B6',source_sha256=pin['sha256'],anchor_updates=pin['updates'],anchor_cursor=pin['cursor'],
            main_seconds=pin['main_seconds'],main_added_updates=0,q=0.,cooldown=copy.deepcopy(cd),
            planned_cooldown_updates=cd['N'],completed_updates=cd['updates'],endpoint_lr=cd['last_lr'])
    try:
        t=time.monotonic();checkpoint=save();initial_save_seconds=time.monotonic()-t;last_save=time.monotonic()
        while cd['updates']<cd['N']:
            if (root.parent/'stop-request.json').exists() or time.monotonic()>=train_until:
                reason='deadline/stop; fixed N unchanged';break
            ids=L.take(order,cd['cursor'],128)
            try:
                log,unsafe=D.step(m,o,s,state,cd,cache,order,device,identity)
                if unsafe:raise R.UnsafeState('Cooldown stability threshold')
            except (L.UnsafeAdamOOM,R.UnsafeState) as error:
                # Never serialize suspect Adam/model. Fall back to last finite full state.
                attempted=cd['updates'];del m,o,s;m=o=s=None;R.release(device)
                p=D.read_full(root,identity,environment=not cpu);m,o,s,state,cd=D.restore(p,device);del p
                reason=type(error).__name__
                write(root/'recovery-stop.json',dict(reason=reason,attempted_updates=attempted,
                    restored_updates=cd['updates'],fixed_N=cd['N'],LR_halving=False));break
            stream.update(np.asarray(ids,dtype='<u4').tobytes());cd['stream_sha256']=stream.hexdigest()
            with (root/'cooldown-updates.jsonl').open('a') as f:
                f.write(json.dumps(dict(log,branch_update=cd['updates'],cursor=cd['cursor'],
                    cooldown_seconds=cd['seconds'],stream_sha256=cd['stream_sha256']),allow_nan=False)+'\n')
            if cd['updates']%100==0 or cd['updates']==cd['N']:
                write(root/'progress.json',dict(cooldown=cd,last_update=log))
                print('B6',name,cd['updates'],'/',cd['N'],'LR',cd['last_lr'],flush=True)
            if cfg.get('save_midpoint',True) and cd['updates']==cd['N']//2:
                # Admission uses measured initial full-checkpoint I/O; FP16 is ~1/6 of full.
                if initial_save_seconds<=30 and train_until-time.monotonic()>120:
                    t=time.monotonic();path=L.save_point(root,'b6-mid',m,spec,identity,training())
                    midpoint=dict(file=path.relative_to(root).as_posix(),seconds=time.monotonic()-t,evaluated=False)
                    write(root/'midpoint.json',midpoint)
                else:write(root/'midpoint.json',dict(skipped=True,reason='I/O estimate or reserve >60s'))
            if time.monotonic()-last_save>=cfg.get('checkpoint_seconds',900.):
                checkpoint=save();last_save=time.monotonic()
        if reason is None and cd['updates']==cd['N'] and cd['last_lr']==0.:
            tag='b6-final';point=L.save_point(root,tag,m,spec,identity,training())
        elif reason is None:reason='fixed-N cooldown incomplete'
        checkpoint=save()
        if point is not None:
            del m,o,s;m=o=s=None;R.release(device)
            try:
                metrics=evaluate(point,root/'points/b6-final/evaluation',min(deadline-reserve,time.monotonic()+eval_cap),True)
                final_complete=B5.metric_complete(metrics,True,cfg.get('gpu_smoke',False))
                if not final_complete:reason='incomplete final evaluation'
            except Exception as error:
                reason='evaluation: '+type(error).__name__
            receipt=dict(training=training(),weights_sha256=sha(point),metrics=metrics)
            write(root/'points/b6-final'/('complete.json' if final_complete else 'incomplete.json'),receipt)
        status='complete' if final_complete else 'partial'
    except Exception as error:
        reason=type(error).__name__
        # Existing atomic full checkpoint and any completed FP16/evaluation files survive.
        write(root/'runtime-failure.json',dict(reason=reason,last_finite_checkpoint=checkpoint))
    finally:
        if m is not None:del m,o,s
        R.release(device)
    result=dict(status=status,model=name,reason=reason,final_complete=final_complete,cooldown=cd,
        main_added_updates=0,main_state=state,latest=checkpoint,midpoint=midpoint,
        FP16_final=str(point.relative_to(root)) if point else None,wall_seconds=time.monotonic()-start)
    write(root/'summary.json',result)
    return result
