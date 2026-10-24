"""L1 finite-update math, independent nonfinite recovery and disposable timed cooldowns."""
import copy, gc, json, time, uuid
from pathlib import Path
import torch
import l1_runtime as L
import exp26_core as C
import exp26_train as T
from kaggle_io import sha, write
from l2_resume import initial_state

POLICY = {'consecutive': 3, 'window_size': 100, 'window_failures': 10, 'max_escalations': 3,
          'amp_consecutive': 32}
class UnsafeState(RuntimeError): pass

def safe_update(model, opt, scaler, cache, indices, device, micro, lr):
    """Never call Adam after rejected loss/gradient. Normal AMP skips stay normal."""
    started = time.perf_counter(); scale = copy.deepcopy(scaler.state_dict())
    try:
        log, micro = L.resilient_update(model, opt, scaler, cache, indices, device, micro, lr)
        tensors = list(model.parameters()) + [v for s in opt.state.values() for v in s.values()
                                              if torch.is_tensor(v) and v.is_floating_point()]
        norms = torch.stack(torch._foreach_norm(tensors))
        if not bool(torch.isfinite(norms).all()): raise UnsafeState('Nonfinite model/Adam after update')
        log['nonfinite_skip'] = 0
        log['seconds'] = time.perf_counter()-started
        return log, micro
    except FloatingPointError:
        opt.zero_grad(set_to_none=True)
        scaler.load_state_dict(scale)
        if scaler.is_enabled(): scaler._per_optimizer_states.clear()
        for group in opt.param_groups: group['lr'] = lr
        C.sync(device)
        return {'seconds': time.perf_counter()-started, 'CE': None, 'AMP_skip': 0, 'nonfinite_skip': 1,
                'lr': lr, 'pairs': len(indices), 'tokens': 0, 'gradient_norm': None, 'OOM_retries': []}, micro

def observe(stability, rejected, policy=POLICY, amp_skip=0):
    stability['total'] += int(rejected)
    stability['consecutive'] = stability['consecutive']+1 if rejected else 0
    stability['window'] = (stability['window']+[int(rejected)])[-policy['window_size']:]
    stability['amp_total'] = stability.get('amp_total', 0)+int(amp_skip)
    stability['amp_consecutive'] = stability.get('amp_consecutive', 0)+1 if amp_skip else 0
    return (stability['consecutive'] >= policy['consecutive'] or sum(stability['window']) >= policy['window_failures']
            or stability['amp_consecutive'] >= policy['amp_consecutive'])

def finite_payload(payload):
    for value in list(payload['model'].values()) + [v for s in payload['optimizer']['state'].values() for v in s.values()]:
        if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all()): return False
    return True

def payload(model, opt, scaler, state, spec, identity):
    p = L.state_payload(model, opt, scaler, state, spec, identity)
    p['format'] = 'nlp-hw02-l2-full-v1'; p['stability_policy'] = POLICY
    p['runtime_sha256'] = {n: sha(Path(__file__).with_name(n)) for n in ('l2_runtime.py', 'l2_resume.py', 'l2_sampler.py')}
    p['scheduler']['kind'] = 'constant inherited LR; recorded emergency halvings; no repeated warmup'
    return p

def read_full(path, identity, check_environment=True):
    p = torch.load(path, map_location='cpu', weights_only=True)
    assert p['format'] == 'nlp-hw02-l2-full-v1' and p['identity'] == identity
    assert p['stability_policy'] == POLICY
    for n, digest in p['runtime_sha256'].items(): assert sha(Path(__file__).with_name(n)) == digest
    for n, digest in identity['recipe']['implementation_sha256'].items(): assert sha(Path(__file__).with_name(n)) == digest
    if check_environment:
        assert identity['recipe']['torch'] == str(torch.__version__) and identity['recipe']['CUDA'] == torch.version.cuda
    s = p['state']; assert s['cursor'] == s['cursor_origin']+s['updates']*128
    assert p['scheduler']['last_lr'] == s['last_lr']
    assert all(g['lr'] == s['last_lr'] for g in p['optimizer']['param_groups'])
    steps = s['updates']-s['AMP_skips']-s['nonfinite_skips']
    assert all(int(v['step']) == steps for v in p['optimizer']['state'].values())
    assert finite_payload(p), 'Nonfinite stable payload'
    return p

def save_full(root, model, opt, scaler, state, spec, identity):
    root = Path(root); folder = root/'checkpoints'; folder.mkdir(parents=True, exist_ok=True)
    C.sync(next(model.parameters()).device); p = payload(model, opt, scaler, state, spec, identity)
    if not finite_payload(p): raise FloatingPointError('Refuse nonfinite stable state')
    path = folder/('main-'+uuid.uuid4().hex[:12]+'.pt'); tmp = path.with_suffix('.pt.tmp')
    torch.save(p, tmp); tmp.replace(path); del p
    rec = {'file': path.relative_to(root).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size,
           'updates': state['updates'], 'cursor': state['cursor'], 'main_seconds': state['main_seconds'],
           'l2_main_seconds': state['l2_main_seconds'], 'identity': identity, 'model_spec': spec,
           'purpose': 'stable full L2 resume; uncooled; model/AdamW/scaler/all RNG/history'}
    write(root/'latest.json', rec)
    for old in sorted(folder.glob('main-*.pt'), key=lambda p:p.stat().st_mtime_ns, reverse=True)[2:]: old.unlink(missing_ok=True)
    return rec

def reload_latest(root, identity, device, smoke=False):
    rec = json.loads((Path(root)/'latest.json').read_text()); path = Path(root)/rec['file']
    assert sha(path) == rec['sha256'] and path.stat().st_size == rec['bytes']
    p = read_full(path, identity)
    m = C.SimpleTransformer(**p['model_spec'], dropout=0.).to(device)
    o = L.optimizer(m, p['state']['last_lr'], smoke)
    s = torch.amp.GradScaler(device.type, enabled=device.type=='cuda', init_scale=1024)
    state = L.restore_full(p, m, o, s, device); del p
    return m, o, s, state

def release(device, *objects):
    # Callers must drop their references before empty_cache.
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()

def run(name, anchor, identity, cache, order, root, device, deadline, evaluate,
        main_targets=(10800.,21600.,28800.), cooldown_seconds=(900.,900.,3000.),
        eval_seconds=(900.,900.,1800.), checkpoint_seconds=1800., smoke=False, max_updates=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True); spec = anchor['model_spec']
    model = C.SimpleTransformer(**spec, dropout=0.).to(device)
    opt = L.optimizer(model, anchor['state']['last_lr'], smoke)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type=='cuda', init_scale=1024)
    state = L.restore_full(anchor, model, opt, scaler, device)
    if anchor['format'] == 'nlp-hw02-l1-full-v1': state = initial_state(anchor, name)
    # L3 can pass read_full() directly: order/cursor/Adam/scaler/RNG/history survive.
    del anchor
    write(root/'identity.json', identity); write(root/'stability-policy.json', POLICY)
    save_full(root, model, opt, scaler, state, spec, identity)
    last_save = time.monotonic(); started_updates = state['updates']; status = 'complete'; logs = []; pending = None
    def stopped(): return (root.parent/'stop-request.json').exists() or time.monotonic() >= deadline-120*(not smoke)
    def summary(status):
        result = {'model': name, 'status': status, 'state': state, 'identity': identity}
        write(root/'summary.json', result); return result
    summary('training')
    try:
        for mark, cool_s, eval_s in zip(main_targets, cooldown_seconds, eval_seconds):
            if f'l2-{int(mark):05d}' in {p['tag'] for p in state['l2_points']}: continue
            while state['l2_main_seconds'] < mark and not stopped():
                if max_updates is not None and state['updates']-started_updates >= max_updates: break
                unsafe = pending; pending = None; log = None
                if unsafe is None:
                    attempt_started = time.perf_counter()
                    try:
                        log, micro = safe_update(model,opt,scaler,cache,L.take(order,state['cursor'],128),device,state['micro_batch'],state['last_lr'])
                    except L.UnsafeAdamOOM: unsafe = 'Adam OOM'
                    except UnsafeState: unsafe = 'nonfinite post-Adam model'
                    if unsafe is not None:
                        C.sync(device)
                        spent = time.perf_counter()-attempt_started
                        state['main_seconds'] += spent; state['l2_main_seconds'] += spent
                if log is not None:
                    state['micro_batch'] = micro; state['updates'] += 1; state['cursor'] += 128
                    state['main_seconds'] += log['seconds']; state['l2_main_seconds'] += log['seconds']
                    state['AMP_skips'] += log['AMP_skip']; state['nonfinite_skips'] += log['nonfinite_skip']
                    logs = (logs+[log])[-100:]
                    escalate = observe(state['stability'], log['nonfinite_skip'], amp_skip=log['AMP_skip'])
                else: escalate = True
                if escalate:
                    history = copy.deepcopy(state['stability']); elapsed = state['l2_main_seconds']
                    history['escalations'] += 1
                    event = {'reason': unsafe or 'nonfinite threshold', 'from_updates': state['updates'],
                             'l2_main_seconds': elapsed, 'escalation': history['escalations'], 'old_lr': state['last_lr']}
                    old_micro = state['micro_batch']; old_lr = state['last_lr']
                    del model,opt,scaler; release(device)
                    model,opt,scaler,state = reload_latest(root,identity,device,smoke)
                    # Work already spent remains in the common main-time axis after rollback.
                    state['l2_main_seconds'] = elapsed; state['main_seconds'] = state['l2_origin']['main_seconds']+elapsed
                    history['consecutive'] = 0; history['window'] = []; history['amp_consecutive'] = 0
                    event.update(restored_updates=state['updates'], restored_cursor=state['cursor'])
                    if history['escalations'] >= POLICY['max_escalations']:
                        event['action'] = 'graceful worker stop; keep last good LR/state'; status = 'stability_stop'
                    else:
                        state['last_lr'] = old_lr*.5
                        for group in opt.param_groups: group['lr'] = state['last_lr']
                        event.update(action='rollback and halve LR', new_lr=state['last_lr'])
                        if unsafe == 'Adam OOM':
                            choices = [m for m in L.MICRO_BATCHES if m < old_micro]
                            if not choices: status = 'stability_stop'
                            else: state['micro_batch'] = choices[0]
                    history['history'].append(event); state['stability'] = history
                    write(root/'stability-history.json', history)
                    save_full(root,model,opt,scaler,state,spec,identity); last_save = time.monotonic()
                    if status == 'stability_stop': break
                    continue
                if state['updates'] % (1 if smoke else 100) == 0:
                    write(root/'progress.json', state)
                    print('L2 TRAIN',name,'updates',state['updates'],'main_s',round(state['l2_main_seconds'],1),
                          'CE',log['CE'],'LR',state['last_lr'],'NF',state['stability']['total'],flush=True)
                if time.monotonic()-last_save >= checkpoint_seconds:
                    try: save_full(root,model,opt,scaler,state,spec,identity)
                    except FloatingPointError:
                        # A corrupted Adam/model must never replace the last known-good receipt.
                        pending = 'nonfinite stable candidate'; continue
                    last_save = time.monotonic()
            if status == 'stability_stop' or stopped() or state['l2_main_seconds'] < mark:
                status = status if status == 'stability_stop' else 'partial; target main time not reached'; break
            # Both models evaluate the same L2 main-time target and the same timed cooldown.
            rec = save_full(root,model,opt,scaler,state,spec,identity); last_save = time.monotonic()
            tag = f'l2-{int(mark):05d}'; out = root/'points'/tag; out.mkdir(parents=True,exist_ok=True)
            train = {'target_l2_main_seconds': mark, 'l2_main_seconds': state['l2_main_seconds'],
                     'main_seconds': state['main_seconds'], 'anchor_updates': state['updates'],
                     'anchor_cursor': state['cursor'], 'anchor_sha256': rec['sha256'],
                     'cooldown_target_seconds': cool_s, 'cooldown_updates': 0, 'cooldown_AMP_skips': 0,
                     'cooldown_nonfinite_skips': 0, 'final': mark == main_targets[-1]}
            write(out/'training.json',train)
            branch_start = time.monotonic(); branch_s = 0.; branch_failed = False; metrics = None
            branch_stability = copy.deepcopy(state['stability']); branch_failure_reason = None
            try:
                if deadline-time.monotonic() < cool_s+eval_s+120*(not smoke):
                    raise C.BudgetExpired('Insufficient reserved cooldown/evaluation time')
                while branch_s < cool_s and not stopped():
                    lr = state['last_lr']*max(0.,1.-branch_s/cool_s)
                    log,micro = safe_update(model,opt,scaler,cache,L.take(order,state['cursor']+train['cooldown_updates']*128,128),
                                           device,state['micro_batch'],lr)
                    state['micro_batch'] = micro; branch_s += log['seconds']; train['cooldown_updates'] += 1
                    train['cooldown_AMP_skips'] += log['AMP_skip']; train['cooldown_nonfinite_skips'] += log['nonfinite_skip']
                    if observe(branch_stability,log['nonfinite_skip'],amp_skip=log['AMP_skip']):
                        branch_failed = True; branch_failure_reason = 'nonfinite cooldown threshold'; break
                train['cooldown_seconds'] = branch_s; train['cooldown_wall_seconds'] = time.monotonic()-branch_start
                if branch_failed or branch_s < cool_s: raise RuntimeError('Incomplete cooldown')
                if not all(bool(torch.isfinite(v).all()) for v in model.parameters()): raise FloatingPointError('Nonfinite cooled model')
                point = L.save_point(root,tag,model,spec,identity,train)
                del model,opt,scaler; model=opt=scaler=None; release(device)
                begin = time.monotonic()
                metrics = evaluate(point,out/'evaluation',min(deadline-60*(not smoke),time.monotonic()+eval_s),train['final'])
                train['evaluation_seconds'] = time.monotonic()-begin
                write(out/'complete.json',{'training':train,'metrics':metrics,'weights_sha256':sha(point)})
            except Exception as error:
                if isinstance(error,(UnsafeState,L.UnsafeAdamOOM)):
                    branch_failed = True; branch_failure_reason = type(error).__name__+' in cooldown'
                write(out/'incomplete.json',{'reason':type(error).__name__,'training':train})
                print('L2 POINT INCOMPLETE',name,tag,type(error).__name__,flush=True)
            finally:
                if model is not None: del model,opt,scaler
                release(device); model,opt,scaler,state = reload_latest(root,identity,device,smoke)
                state['branch_seconds'] += branch_s; state['branch_attempted_updates'] += train['cooldown_updates']
                state['stability'] = branch_stability
                if branch_failed:
                    h = state['stability']; h['escalations'] += 1
                    event = {'reason':branch_failure_reason,'point':tag,'restored_updates':state['updates'],
                             'restored_cursor':state['cursor'],'old_lr':state['last_lr'],'escalation':h['escalations']}
                    h['consecutive'] = 0; h['window'] = []; h['amp_consecutive'] = 0
                    if h['escalations'] >= POLICY['max_escalations']:
                        event['action'] = 'graceful worker stop; cooldown discarded'; status = 'stability_stop'
                    else:
                        state['last_lr'] *= .5
                        for group in opt.param_groups: group['lr'] = state['last_lr']
                        event.update(action='cooldown rollback and halve worker LR',new_lr=state['last_lr'])
                    h['history'].append(event); write(root/'stability-history.json',h)
                state['l2_points'].append({'tag':tag,'training':train,'complete':metrics is not None,'metrics':metrics})
                save_full(root,model,opt,scaler,state,spec,identity); last_save = time.monotonic(); summary('training')
            if status == 'stability_stop': break
        # Even a stability stop publishes the restored last-good state, with cumulative history.
        save_full(root,model,opt,scaler,state,spec,identity)
        return summary(status)
    finally:
        del model,opt,scaler; release(device)
