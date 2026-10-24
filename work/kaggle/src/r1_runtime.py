"""R1 stable continuation and a disposable terminal cooldown, bounded by wall time."""
import copy
import json
import math
import time
import uuid
from pathlib import Path
import torch
import exp26_core as C
import exp26_train as T
import l1_runtime as L
import l2_runtime as R
from kaggle_io import sha, write
from exp26_common import json_hash
from r1_growth import deepen, groups_by_name, update


def restore_anchor(anchor, branch, device, cpu=False):
    model = C.SimpleTransformer(**anchor['model_spec'], dropout=0.).to(device)
    opt = L.optimizer(model, anchor['state']['last_lr'], cpu)
    scaler = torch.amp.GradScaler(device.type, enabled=anchor['scaler_enabled'], init_scale=1024)
    state = L.restore_full(anchor, model, opt, scaler, device)
    if cpu:
        opt.defaults['fused'] = False
        for group in opt.param_groups: group['fused'] = False
    spec = copy.deepcopy(anchor['model_spec'])
    if branch['growth'] == 'encoder-deepen':
        spec = deepen(model, opt, spec, branch.get('growth_seed', 2026101701), branch.get('encoder_layers',12))
    else:
        assert branch['growth'] == 'none'
    return model, opt, scaler, state, spec


def payload(model, opt, scaler, state, spec, identity):
    p = L.state_payload(model, opt, scaler, state, spec, identity)
    p.update(format='nlp-hw02-r1-full-v1', optimizer_names=groups_by_name(model,opt),
             runtime_sha256={n:sha(Path(__file__).with_name(n))
                            for n in ('r1_growth.py','r1_runtime.py','r1_data.py')})
    p['scheduler'].update(kind='R1 constant old / warmup new; stable before terminal cooldown',
                          group_lrs=[g['lr'] for g in opt.param_groups])
    return p


def save_full(root, model, opt, scaler, state, spec, identity):
    root = Path(root); folder = root/'checkpoints'; folder.mkdir(parents=True,exist_ok=True)
    C.sync(next(model.parameters()).device)
    p = payload(model,opt,scaler,state,spec,identity)
    assert R.finite_payload(p), 'Refuse unsafe R1 checkpoint'
    path = folder/('main-'+uuid.uuid4().hex[:12]+'.pt'); tmp = path.with_suffix('.pt.tmp')
    torch.save(p,tmp); tmp.replace(path)
    receipt = dict(file=path.relative_to(root).as_posix(),sha256=sha(path),bytes=path.stat().st_size,
                   updates=state['updates'],cursor=state['cursor'],main_seconds=state['main_seconds'],
                   model_spec=spec,identity=identity,purpose='R1 full stable; before terminal cooldown')
    write(root/'latest.json',receipt)
    for old in sorted(folder.glob('main-*.pt'),key=lambda p:p.stat().st_mtime_ns,reverse=True)[2:]:
        old.unlink(missing_ok=True)
    return receipt


def reload_latest(root, device, cpu=False):
    root=Path(root); rec=json.loads((root/'latest.json').read_text()); path=root/rec['file']
    assert path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256']
    p=torch.load(path,map_location='cpu',weights_only=True)
    assert p['format']=='nlp-hw02-r1-full-v1' and p['identity']==rec['identity'] and R.finite_payload(p)
    for name,h in p['runtime_sha256'].items(): assert sha(Path(__file__).with_name(name))==h
    model=C.SimpleTransformer(**p['model_spec'],dropout=0.).to(device)
    names=dict(model.named_parameters()); groups=[]
    for keys,saved in zip(p['optimizer_names'],p['optimizer']['param_groups']):
        group=dict(params=[names[k] for k in keys])
        if 'r1_new' in saved:group['r1_new']=saved['r1_new']
        groups.append(group)
    opt=torch.optim.AdamW(groups,lr=p['state']['last_lr'],weight_decay=.01,fused=not cpu)
    scaler=torch.amp.GradScaler(device.type,enabled=p['scaler_enabled'],init_scale=1024)
    state=L.restore_full(p,model,opt,scaler,device)
    if cpu:
        for g in opt.param_groups:g['fused']=False
    return model,opt,scaler,state,p['model_spec'],p['identity']


def schedule(branch, main_updates):
    if branch['cooldown']=='linear':return None
    assert branch['cooldown']=='1-sqrt'
    f=branch['cooldown_fraction']
    assert 0<f<1
    return max(1,round(main_updates*f/(1-f)))


def metrics_complete(metrics, expected):
    if not isinstance(metrics,dict):return False
    vote=metrics.get('K8_tune2048',{})
    return (vote.get('complete') is True and vote.get('rows')==vote.get('target_rows')==expected['k8']
            and all(metrics.get('splits',{}).get(s,{}).get('exec',{}).get('denominator')==n
                    and math.isfinite(metrics.get('splits',{}).get(s,{}).get('CE',{}).get('CE',float('nan')))
                    for s,n in expected['splits'].items())
            and metrics.get('test',{}).get('rows')==expected['test'])


def run(anchor, branch, cache, order, order_meta, root, device, cfg, evaluate, cpu=False):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    model,opt,scaler,state,spec=restore_anchor(anchor,branch,device,cpu)
    identity=copy.deepcopy(anchor['identity'])
    identity.update(model=root.name.removeprefix('model-'),parent_recipe_sha256=identity['recipe_sha256'],
                    order_sha256=order_meta['file']['sha256'])
    identity['recipe'].update(architecture=spec,r1=branch)
    identity['recipe_sha256']=json_hash(identity['recipe'])
    origin=state['updates']; peak=state['last_lr']
    state['r1']=dict(branch=branch,anchor_updates=origin,anchor_cursor=anchor['state']['cursor'],
                     order_start_cursor=state['cursor'],order_start_slot=state['cursor']%len(order),
                     added_updates=0,main_wall_seconds=0.)
    del anchor
    write(root/'identity.json',identity)
    save_full(root,model,opt,scaler,state,spec,identity)
    begin=time.monotonic(); last_save=begin; deadline=cfg['deadline_monotonic']
    # Reserve cooldown, eval and I/O before allowing main updates. Main counts WALL.
    cool_cap=branch['cooldown_seconds']; eval_cap=branch['eval_seconds']; io=cfg.get('io_reserve_seconds',180)
    checkpoint_reserve=cfg.get('stable_save_reserve_seconds',60 if not cpu else 0)
    main_until=min(begin+branch['main_seconds'],deadline-cool_cap-eval_cap-io-checkpoint_reserve)
    report=dict(status='partial',growth=branch['growth']); phase='main'; cooled=0; metrics=None
    def stop(until):
        return time.monotonic()>=until or (root.parent/'stop-request.json').exists()
    def step(lr):
        nonlocal model,opt,scaler,state
        log,micro=update(model,opt,scaler,cache,L.take(order,state['cursor'],128),device,
                         state['micro_batch'],lr,state['r1']['added_updates']+1,branch['new_warmup'],branch['q'])
        state.update(micro_batch=micro,last_lr=lr,updates=state['updates']+1,cursor=state['cursor']+128)
        state['AMP_skips']+=log['AMP_skip'];state['nonfinite_skips']+=log['nonfinite_skip']
        state['r1']['added_updates']+=1
        if R.observe(state['stability'],log['nonfinite_skip'],amp_skip=log['AMP_skip']):
            raise R.UnsafeState('R1 stability threshold; retain last stable, stop worker')
        if phase=='main':
            state['main_seconds']+=log['seconds'];state['l2_main_seconds']+=log['seconds']
        if state['r1']['added_updates']%100==0 or cfg.get('gpu_smoke'):
            write(root/'progress.json',dict(state=state,phase=phase,last_update=log))
            print('R1',root.name,phase,state['r1']['added_updates'],'CE',log['CE'],flush=True)
        if 'truncation' in log:
            with (root/'truncation.jsonl').open('a') as f:
                f.write(json.dumps(dict(update=state['updates'],phase=phase,**log['truncation']))+'\n')
        return log
    try:
        while not stop(main_until):
            if branch.get('max_main_updates') and state['r1']['added_updates']>=branch['max_main_updates']:break
            step(peak)
            state['r1']['main_wall_seconds']=time.monotonic()-begin
            if time.monotonic()-last_save>=cfg.get('checkpoint_seconds',1800):
                save_full(root,model,opt,scaler,state,spec,identity);last_save=time.monotonic()
        if (root.parent/'stop-request.json').exists():raise C.BudgetExpired('Stop requested')
        if state['r1']['added_updates']==0:raise C.BudgetExpired('No main updates')
        stable=save_full(root,model,opt,scaler,state,spec,identity)
        training=dict(stable=stable,main_wall_seconds=time.monotonic()-begin,
                      main_updates=state['r1']['added_updates'],schedule=branch['cooldown'],endpoint_lr=0.)
        count=schedule(branch,training['main_updates']);phase='cooldown';start=time.monotonic()
        if deadline-start<cool_cap+eval_cap+io:raise C.BudgetExpired('Cooldown reserve lost')
        until=min(start+cool_cap,deadline-eval_cap-io)
        while not stop(until):
            if count is not None and cooled>=count:break
            progress=(cooled+1)/count if count is not None else min(1.,(time.monotonic()-start)/cool_cap)
            lr=peak*(1-math.sqrt(progress) if count is not None else 1-progress)
            step(lr);cooled+=1
        if (root.parent/'stop-request.json').exists() or (count is not None and cooled<count) or cooled==0:
            raise C.BudgetExpired('Incomplete terminal cooldown')
        training.update(cooldown_updates=cooled,planned_cooldown_updates=count,
                        cooldown_wall_seconds=time.monotonic()-start,last_update_lr=state['last_lr'])
        point=L.save_point(root,'final',model,spec,identity,training)
        del model,opt,scaler;model=opt=scaler=None;R.release(device)
        metrics=evaluate(point,root/'points/final/evaluation',min(deadline-60*(not cpu),time.monotonic()+eval_cap))
        if not metrics_complete(metrics,cfg['expected_rows']):raise RuntimeError('Incomplete final evaluation')
        write(root/'points/final/complete.json',dict(training=training,metrics=metrics,weights_sha256=sha(point)))
        report.update(status='complete',training=training,metrics=metrics)
    except Exception as error:
        report.update(reason=type(error).__name__,phase=phase,cooldown_updates=cooled,metrics=metrics)
        write(root/'incomplete.json',report)
    finally:
        if model is not None:del model,opt,scaler
        R.release(device)
        # Never overwrite stable with cooldown weights/cursor/RNG or unsafe Adam.
        m,o,s,restored,_,_=reload_latest(root,device,cpu)
        report.update(stable_updates=restored['updates'],stable_cursor=restored['cursor'],
                      stable_preserved=True,peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
        del m,o,s;R.release(device);write(root/'summary.json',report)
    return report
