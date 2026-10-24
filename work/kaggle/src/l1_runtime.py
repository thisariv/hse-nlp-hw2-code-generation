"""Long constant-LR main trajectory, disposable cooldowns, exact complete resume."""
import copy, gc, json, math, random, time, uuid
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import exp26_core as C
import exp26_train as T
from exp26_order import collate
from kaggle_io import sha, write

SPECS = {
    'C':dict(d_model=512,heads=8,encoder_layers=6,decoder_layers=6,swiglu_dim=1368,vocab=8195),
    'D':dict(d_model=640,heads=10,encoder_layers=8,decoder_layers=8,swiglu_dim=1712,vocab=8195)}
PEAK_LR={'C':0.0015,'D':0.0012}
MICRO_BATCHES=(128,96,64,32,16,8,4,2,1)
class UnsafeAdamOOM(RuntimeError):pass

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def new_model(spec, device):
    seed_all(); model=C.SimpleTransformer(**spec,dropout=0.).to(device)
    with torch.no_grad(): model.tokens.weight.div_(math.sqrt(model.width))
    return model

def optimizer(model, lr, smoke):
    return torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01,**({} if smoke else {'fused':True}))

def take(order,cursor,batch):
    return np.asarray(order[(np.arange(batch,dtype=np.int64)+cursor)%len(order)])

def lr_at(update, peak, warmup):
    return peak*min(1.,update/max(1,warmup))

def update(model,opt,scaler,cache,indices,device,micro,lr):
    """An exact effective batch; 96+32 and 64+64 have token-weighted CE."""
    C.sync(device); started=time.perf_counter(); opt.zero_grad(set_to_none=True)
    tokens=sum(len(cache[int(i)][1])+1 for i in indices); total=0.
    for lo in range(0,len(indices),micro):
        src,tgt,labels,counts=collate(cache,indices[lo:lo+micro],device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=scaler.is_enabled()):
            logits=model(src,tgt)
            loss=F.cross_entropy(logits.reshape(-1,model.vocab),labels.reshape(-1),ignore_index=C.PAD,reduction='sum')/tokens
        if not bool(torch.isfinite(loss).detach()):raise FloatingPointError('Nonfinite loss')
        scaler.scale(loss).backward(); total+=float(loss.detach()); del src,tgt,labels,logits,loss
    for group in opt.param_groups:group['lr']=lr
    scaler.unscale_(opt); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    skipped=bool(any(float(t)>0 for t in scaler._found_inf_per_device(opt).values())) if scaler.is_enabled() else False
    if not skipped and not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite successful gradient')
    try:
        scaler.step(opt)
    except torch.cuda.OutOfMemoryError as error:
        # Adam may have modified only some parameters. Never retry this state.
        error.optimizer_may_be_modified=True
        raise
    scaler.update(); C.sync(device)
    return {'seconds':time.perf_counter()-started,'CE':total,'AMP_skip':int(skipped),'lr':lr,
            'pairs':len(indices),'tokens':tokens,'gradient_norm':float(norm) if torch.isfinite(norm) else None}

def resilient_update(model,opt,scaler,cache,indices,device,micro,lr):
    rng=T.cpu(T.rng_state());scale=copy.deepcopy(scaler.state_dict());attempts=[]
    while True:
        try:
            result=update(model,opt,scaler,cache,indices,device,micro,lr)
            result['OOM_retries']=attempts
            return result,micro
        except torch.cuda.OutOfMemoryError as error:
            unsafe=getattr(error,'optimizer_may_be_modified',False)
        # The exception frame (and its CUDA tensors) must be gone before retry.
        model.zero_grad(set_to_none=True);gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        if unsafe:raise UnsafeAdamOOM('OOM inside Adam: restore last committed full checkpoint')
        smaller=[m for m in MICRO_BATCHES if m<micro]
        if not smaller:raise RuntimeError('OOM even at micro-batch1; last checkpoint preserved')
        attempts.append({'micro_batch':micro,'status':'late OOM before optimizer step'})
        micro=smaller[0];scaler.load_state_dict(scale)
        if scaler.is_enabled():scaler._per_optimizer_states.clear()
        T.restore_rng(rng,device)
        print('OOM RETRY micro',micro,flush=True)

def probe(model, cache, device, peak, smoke=False, batch=128):
    """Full forward/backward/Adam allocation on longest rows, then exact reset."""
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; rng=T.cpu(T.rng_state()); attempts=[]; choice=None
    lengths=cache.source_lengths.astype(np.int32)+cache.target_lengths.astype(np.int32)
    count=min(batch,len(cache)); selected=np.argpartition(lengths,-count)[-count:]
    # Also exercise the individually longest source and target.
    selected[0]=int(cache.source_lengths.argmax())
    selected[min(1,len(selected)-1)]=int(cache.target_lengths.argmax())
    indices=np.resize(selected,batch)
    for micro in ([batch] if smoke else [m for m in MICRO_BATCHES if m<=batch]):
        opt=None; scaler=None
        try:
            model.load_state_dict(initial); T.restore_rng(rng,device)
            if device.type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
            opt=optimizer(model,peak,smoke); scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda',init_scale=1024)
            result=update(model,opt,scaler,cache,indices,device,micro,peak/1000)
            memory=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None
            enough=smoke or memory<.93*torch.cuda.get_device_properties(device).total_memory
            attempts.append({'micro_batch':micro,'status':'passed' if enough else 'insufficient_headroom',
                             'peak_reserved_bytes':memory,'probe_seconds':result['seconds']})
            if enough:choice=micro;break
        except torch.cuda.OutOfMemoryError:attempts.append({'micro_batch':micro,'status':'OOM'})
        finally:
            model.zero_grad(set_to_none=True); del opt,scaler;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
    model.load_state_dict(initial); T.restore_rng(rng,device);del initial
    if choice is None:raise RuntimeError('No safe micro-batch down to1')
    return choice,attempts

def state_payload(model,opt,scaler,state,spec,identity):
    return {'format':'nlp-hw02-l1-full-v1','model':T.cpu(model.state_dict()),'optimizer':T.cpu(opt.state_dict()),
            'scaler':scaler.state_dict(),'scaler_enabled':scaler.is_enabled(),'rng':T.cpu(T.rng_state()),
            'state':copy.deepcopy(state),'model_spec':spec,'identity':identity,
            'scheduler':{'kind':'warmup-constant; final cooldown only at end of all sessions',
                         'warmup_updates':identity['recipe']['warmup_updates'],'last_lr':state['last_lr']}}

def save_full(root,model,opt,scaler,state,spec,identity):
    root=Path(root); folder=root/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
    path=folder/('main-'+uuid.uuid4().hex[:12]+'.pt'); temporary=path.with_suffix('.pt.tmp')
    C.sync(next(model.parameters()).device)
    torch.save(state_payload(model,opt,scaler,state,spec,identity),temporary); temporary.replace(path)
    receipt={'file':path.relative_to(root).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size,
             'updates':state['updates'],'cursor':state['cursor'],'main_seconds':state['main_seconds'],
             'model_spec':spec,'identity':identity,'purpose':'constant-LR main resume; never evaluated cooldown weights'}
    write(root/'latest.json',receipt)
    # Keep a predecessor during concurrent live snapshots; never accumulate all periodic Adam states.
    older=sorted(folder.glob('main-*.pt'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
    for p in older[2:]:p.unlink(missing_ok=True)
    return receipt

def read_full(path,identity):
    p=torch.load(path,map_location='cpu',weights_only=True)
    assert p['format']=='nlp-hw02-l1-full-v1' and p['identity']==identity,'Resume identity mismatch'
    assert p['scheduler']['last_lr']==p['state']['last_lr']
    assert p['scheduler']['warmup_updates']==identity['recipe']['warmup_updates']
    assert p['state']['last_lr']==lr_at(p['state']['updates'],identity['recipe']['peak_lr'],identity['recipe']['warmup_updates'])
    if p['state']['updates']:
        assert all(g['lr']==p['state']['last_lr'] for g in p['optimizer']['param_groups'])
    assert all(int(s['step'])==p['state']['updates']-p['state']['AMP_skips'] for s in p['optimizer']['state'].values())
    assert p['state']['cursor']==p['state']['updates']*identity['recipe']['effective_batch']
    return p

def restore_full(payload,model,opt,scaler,device):
    assert payload['scaler_enabled']==scaler.is_enabled(),'Resume AMP/scaler mode mismatch'
    model.load_state_dict(payload['model']);opt.load_state_dict(payload['optimizer']);scaler.load_state_dict(payload['scaler'])
    T.restore_rng(payload['rng'],device);model.train(); return copy.deepcopy(payload['state'])

def save_point(root,tag,model,spec,identity,training):
    path=Path(root)/'weights'/f'{tag}-fp16.pt';path.parent.mkdir(parents=True,exist_ok=True)
    tensors={k:(v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()) for k,v in model.state_dict().items()}
    torch.save({'format':'nlp-hw02-model-only-v1','model':tensors,'model_spec':spec,'identity':identity,
                'dtype':'FP16','optimizer_included':False,'point':tag},path)
    receipt={'file':path.relative_to(root).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size,
             'dtype':'FP16','optimizer_included':False}
    write(Path(root)/'points'/tag/'checkpoint.json',receipt);write(Path(root)/'points'/tag/'training.json',training)
    return path

def identity_for(name,spec,cache,order_sha,smoke=False,batch=128,warmup=1000):
    # Semantic recipe hashes permit L2/L3 packaging changes without weakening data/state checks.
    from exp26_common import json_hash
    recipe={'architecture':spec,'representation':30,'normalized_token_init':True,'seed':42,'dropout':0.,
            'AMP':'FP16+GradScaler','fused_AdamW':not smoke,'weight_decay':.01,'clip':1.,
            'effective_batch':batch,'peak_lr':PEAK_LR[name],'warmup_updates':warmup,'main_schedule':'constant',
            'data_policy':'fixed bucketed permutation, circular continuation without dropping tail',
            'implementation_sha256':{n:sha(Path(__file__).with_name(n)) for n in
                ('l1_runtime.py','exp26_core.py','exp26_train.py','exp26_order.py','s3_representation.py')},
            'torch':str(torch.__version__),'CUDA':torch.version.cuda}
    return {'model':name,'recipe':recipe,'recipe_sha256':json_hash(recipe),
            'cache_manifest_sha256':cache.manifest_sha,'order_sha256':order_sha,'smoke':smoke}

def run(name,cache,order,order_sha,root,scratch,device,deadline,evaluate,smoke=False,spec=None,
        checkpoint_seconds=1800.,point_seconds=9000.,cooldown_cap=1500.,max_updates=None,resume=None,
        batch=128,warmup=1000,point_updates=None,evaluation_seconds=900.,max_points=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);scratch=Path(scratch);scratch.mkdir(parents=True,exist_ok=True)
    spec=spec or SPECS[name]; identity=identity_for(name,spec,cache,order_sha,smoke,batch,warmup)
    model=new_model(spec,device); params=sum(p.numel() for p in model.parameters())
    print('MODEL',name,'parameters',params,'spec',spec,flush=True)
    opt=optimizer(model,PEAK_LR[name],smoke);scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda',init_scale=1024)
    state={'updates':0,'cursor':0,'main_seconds':0.,'branch_seconds':0.,'branch_attempted_updates':0,
           'AMP_skips':0,'last_lr':0.,'next_point_seconds':point_seconds,'points':[], 'micro_batch':batch}
    if resume:
        receipt=json.loads((Path(resume)/'latest.json').read_text());file=Path(resume)/receipt['file']
        assert sha(file)==receipt['sha256'] and file.stat().st_size==receipt['bytes']
        p=read_full(file,identity);state=restore_full(p,model,opt,scaler,device);del p
        if 'deferred_point_seconds' in state:
            state['missed_point_seconds']=state.pop('deferred_point_seconds')
            state['next_point_seconds']=(math.floor(state['main_seconds']/point_seconds)+1)*point_seconds
        print('RESUME',name,state['updates'],'cursor',state['cursor'],'main seconds',state['main_seconds'],flush=True)
    else:
        del opt,scaler
        micro,attempts=probe(model,cache,device,PEAK_LR[name],smoke,batch)
        state['micro_batch']=micro
        write(root/'memory-probe.json',{'effective_batch':batch,'micro_batch':micro,'attempts':attempts,'data_cursor_consumed':0})
        opt=optimizer(model,PEAK_LR[name],smoke);scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda',init_scale=1024)
    write(root/'identity.json',identity)
    write(root/'summary.json',{'model':name,'status':'training','parameter_count':params,'state':state,'identity':identity})
    save_full(root,model,opt,scaler,state,spec,identity);last_save=time.monotonic();start_updates=state['updates']
    logs=[];unsafe_ooms=0
    try:
        while time.monotonic()<deadline-(.2 if smoke else 120.):
            if (root.parent/'stop-request.json').exists():break
            if max_updates is not None and state['updates']-start_updates>=max_updates:break
            due=(state['updates']>=point_updates and not state.get('point_attempts',0)) if point_updates else state['main_seconds']>=state['next_point_seconds']
            if max_points is not None and state.get('point_attempts',0)>=max_points:due=False
            if due:
                remaining=deadline-time.monotonic()
                if smoke or remaining>=cooldown_cap+evaluation_seconds+180:
                    tag=f't{int(round(state["next_point_seconds"]*(1000 if smoke else 1))):06d}'
                    anchor=state_payload(model,opt,scaler,state,spec,identity)
                    anchor_path=scratch/'anchor.pt';torch.save(anchor,anchor_path);del anchor
                    save_full(root,model,opt,scaler,state,spec,identity);last_save=time.monotonic()
                    mean_seconds=(sum(r['seconds'] for r in logs[-100:])/len(logs[-100:]) if logs else
                                  state['main_seconds']/max(1,state['updates']))
                    length=max(1,min(int(.1*state['updates']),int(.90*cooldown_cap/max(.001,mean_seconds))))
                    branch_start=time.monotonic(); branch_seconds=0.; done=0; branch_skips=0
                    for j in range(1,length+1):
                        # Emergency reserve also covers an unexpectedly slow update.
                        if (root.parent/'stop-request.json').exists() or time.monotonic()>=min(branch_start+cooldown_cap-mean_seconds*2,deadline-(.1 if smoke else evaluation_seconds+150)):break
                        lr=state['last_lr']*(1-j/length)
                        try:
                            log,micro=resilient_update(model,opt,scaler,cache,take(order,state['cursor']+done*batch,batch),device,state['micro_batch'],lr)
                        except UnsafeAdamOOM:
                            print('BRANCH Adam OOM: discard branch and restore anchor',flush=True)
                            break
                        state['micro_batch']=micro
                        branch_seconds+=log['seconds'];done+=1;branch_skips+=log['AMP_skip']
                    training={'main_seconds':state['main_seconds'],'target_main_seconds':state['next_point_seconds'],
                              'anchor_updates':state['updates'],'anchor_cursor':state['cursor'],
                              'cooldown_updates':done,'planned_cooldown_updates':length,'cooldown_seconds':branch_seconds,
                              'branch_AMP_skips':branch_skips,'all_training_seconds':state['main_seconds']+state['branch_seconds']+branch_seconds,
                              'main_AMP_skips':state['AMP_skips'],'parameter_count':params,
                              'common_time_basis':'pure main update time; excludes setup, probes, I/O, evaluation and side cooldowns'}
                    metrics=None
                    eval_started=time.monotonic();branch_micro=state['micro_batch']
                    if done==length:
                        point=save_point(root,tag,model,spec,identity,training)
                        # Evaluation uses an independent model: no half/float conversion of training parameters.
                        del model,opt,scaler;model=opt=scaler=None;gc.collect()
                        if device.type=='cuda':torch.cuda.empty_cache()
                        try:
                            metrics=evaluate(point,root/'points'/tag/'evaluation',min(time.monotonic()+evaluation_seconds,deadline-(.1 if smoke else 150)))
                        except Exception as error:
                            write(root/'points'/tag/'incomplete.json',{'reason':'evaluation failed; main anchor restored',
                                  'error_type':type(error).__name__,**training})
                            print('EVAL INCOMPLETE',name,type(error).__name__,flush=True)
                        if metrics is not None:
                            training['evaluation_seconds']=time.monotonic()-eval_started
                            write(root/'points'/tag/'complete.json',{'training':training,'metrics':metrics,'weights_sha256':sha(point)})
                        model=C.SimpleTransformer(**spec,dropout=0.).to(device)
                        opt=optimizer(model,PEAK_LR[name],smoke);scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda',init_scale=1024)
                    else:write(root/'points'/tag/'incomplete.json',{'reason':'cooldown interrupted; no evaluated point',**training})
                    anchor=read_full(anchor_path,identity);state=restore_full(anchor,model,opt,scaler,device);del anchor
                    state['micro_batch']=min(state['micro_batch'],branch_micro)
                    anchor_path.unlink(missing_ok=True)
                    state['branch_seconds']+=branch_seconds;state['branch_attempted_updates']+=done
                    state['point_attempts']=state.get('point_attempts',0)+1
                    if metrics is not None:state['points'].append({'tag':tag,**training,'metrics':metrics})
                    state['next_point_seconds']+=point_seconds
                    save_full(root,model,opt,scaler,state,spec,identity);last_save=time.monotonic()
                    write(root/'summary.json',{'model':name,'status':'training','parameter_count':params,'state':state,'identity':identity})
                    continue
                # Preserve the pending time point for L2; spend remaining session on stable main.
                state['deferred_point_seconds']=state['next_point_seconds'];state['next_point_seconds']=float('1e100')
            try:
                log,micro=resilient_update(model,opt,scaler,cache,take(order,state['cursor'],batch),device,state['micro_batch'],
                                          lr_at(state['updates']+1,PEAK_LR[name],warmup))
            except UnsafeAdamOOM:
                unsafe_ooms+=1;smaller=[m for m in MICRO_BATCHES if m<state['micro_batch']]
                if unsafe_ooms>2 or not smaller:raise
                old_updates=state['updates'];del model,opt,scaler;gc.collect()
                if device.type=='cuda':torch.cuda.empty_cache()
                rec=json.loads((root/'latest.json').read_text());assert sha(root/rec['file'])==rec['sha256']
                payload=read_full(root/rec['file'],identity)
                model=C.SimpleTransformer(**spec,dropout=0.).to(device);opt=optimizer(model,PEAK_LR[name],smoke)
                scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda',init_scale=1024)
                state=restore_full(payload,model,opt,scaler,device);del payload
                state['micro_batch']=smaller[0];logs=[]
                write(root/'adam-oom-recovery.json',{'rolled_back_from_updates':old_updates,
                      'restored_updates':state['updates'],'micro_batch':state['micro_batch'],'attempt':unsafe_ooms})
                continue
            state['micro_batch']=micro
            state['updates']+=1;state['cursor']+=batch;state['main_seconds']+=log['seconds']
            state['AMP_skips']+=log['AMP_skip'];state['last_lr']=log['lr'];logs.append(log)
            if len(logs)>100:logs=logs[-100:]
            if state['updates']%(1 if smoke else 100)==0:
                print('TRAIN',name,'updates',state['updates'],'main_s',round(state['main_seconds'],1),
                      'CE',round(log['CE'],4),'micro',state['micro_batch'],'LR',state['last_lr'],flush=True)
                write(root/'progress.json',dict(state,last_chunk=log,epoch=state['cursor']//len(order)))
            if time.monotonic()-last_save>=checkpoint_seconds:
                save_full(root,model,opt,scaler,state,spec,identity);last_save=time.monotonic()
        if 'deferred_point_seconds' in state:
            # Resume evaluates the first future common grid point, never labels a later anchor as the missed time.
            deferred=state.pop('deferred_point_seconds');state['missed_point_seconds']=deferred
            state['next_point_seconds']=(math.floor(state['main_seconds']/point_seconds)+1)*point_seconds
        save_full(root,model,opt,scaler,state,spec,identity)
        summary={'model':name,'status':'session_complete; main LR remains constant','parameter_count':params,'state':state,'identity':identity}
        write(root/'summary.json',summary);return summary
    finally:
        del model,opt,scaler;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
