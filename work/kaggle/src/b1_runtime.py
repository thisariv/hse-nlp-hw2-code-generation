"""Fresh E/W, one warmup across accounts, verified full states and disposable probes."""
import copy
import gc
import numpy as np
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
from exp26_common import json_hash
from kaggle_io import sha, write, safe_name

SPECS = {
    'E': dict(d_model=768, heads=12, encoder_layers=8, decoder_layers=4, swiglu_dim=2048, vocab=8195),
    'W': dict(d_model=768, heads=12, encoder_layers=6, decoder_layers=6, swiglu_dim=2048, vocab=8195)}
MICRO_CAPS = {'E': 64, 'W': 32}
PROBE_MARKS = (9000., 28370.)

IMPLEMENTATION = ('b1_runtime.py', 'b1_data.py', 'exp26_core.py', 'exp26_train.py', 'exp26_order.py',
                  'l1_runtime.py', 'l2_runtime.py', 'p1_math.py', 'r1_growth.py', 's3_representation.py')


def parameter_count(spec):
    d, f, v = spec['d_model'], spec['swiglu_dim'], spec['vocab']
    # token/position embeddings, untied head; two final norms; biased MHA/SwiGLU/LayerNorm.
    return (2*v*d+2048*d+v+4*d + spec['encoder_layers']*(4*d*d+3*d*f+9*d+2*f)
            + spec['decoder_layers']*(8*d*d+3*d*f+15*d+2*f))


def new_model(spec, device, dropout=0., seed=42):
    assert 0 <= dropout < 1
    L.seed_all(seed)
    model = C.SimpleTransformer(**spec, dropout=dropout).to(device)
    with torch.no_grad(): model.tokens.weight.div_(math.sqrt(model.width))
    return model


def probe(model, cache, device, peak, name, cpu=False):
    """Worst cached rows, bounded by measured T4 shape; require a real Adam step.

    Scale=1 avoids falsely passing an overflow-skipped lazy Adam allocation.
    Restore initialization and every RNG after all attempts.
    """
    initial = T.cpu({k:v.detach().clone() for k,v in model.state_dict().items()})
    rng = T.cpu(T.rng_state()); attempts = []; choice = None
    lengths = cache.source_lengths.astype(np.int32)+cache.target_lengths.astype(np.int32)
    count = min(128, len(cache)); selected = np.argpartition(lengths, -count)[-count:]
    selected[0] = int(cache.source_lengths.argmax())
    selected[min(1, len(selected)-1)] = int(cache.target_lengths.argmax())
    indices = np.resize(selected, 128)
    try:
        for micro in (v for v in L.MICRO_BATCHES if v <= MICRO_CAPS[name]):
            opt = scaler = None
            try:
                model.load_state_dict(initial); T.restore_rng(rng, device)
                if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
                opt = L.optimizer(model, peak, cpu)
                scaler = torch.amp.GradScaler(device.type, enabled=device.type=='cuda', init_scale=1.)
                log = L.update(model, opt, scaler, cache, indices, device, micro, peak/1000)
                assert not log['AMP_skip'], 'Memory probe skipped Adam: allocation unverified'
                assert opt.state and all(int(v['step']) == 1 for v in opt.state.values())
                reserved = torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0
                enough = cpu or reserved < .93*torch.cuda.get_device_properties(device).total_memory
                attempts.append(dict(micro_batch=micro, status='passed' if enough else 'insufficient_headroom',
                    peak_reserved_bytes=reserved, peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
                    if device.type=='cuda' else 0, Adam_step_verified=True, probe_seconds=log['seconds']))
                if enough: choice = micro; break
            except torch.cuda.OutOfMemoryError:
                attempts.append(dict(micro_batch=micro, status='OOM'))
            finally:
                model.zero_grad(set_to_none=True); opt = scaler = None; gc.collect()
                if device.type=='cuda': torch.cuda.empty_cache()
    finally:
        model.load_state_dict(initial); T.restore_rng(rng, device)
    assert choice is not None, 'No safe B1 micro-batch'
    return choice, attempts


def identity_for(name, spec, cache, meta, dropout=0., cpu=False, seed=42):
    recipe = dict(architecture=spec, normalized_token_init=True, seed=seed, dropout=dropout,
                  peak_lr=.001, warmup_updates=1000, effective_batch=128, weight_decay=.01, clip=1.,
                  AMP='FP16+GradScaler' if not cpu else 'CPU test', fused_AdamW=not cpu,
                  main_truncation_q=0., main_schedule='warmup then constant; emergency factor persists',
                  torch=str(torch.__version__), CUDA=torch.version.cuda,
                  implementation_sha256={n:sha(Path(__file__).with_name(n)) for n in IMPLEMENTATION})
    return dict(model=name, recipe=recipe, recipe_sha256=json_hash(recipe),
                cache_manifest_sha256=cache.manifest_sha, order_sha256=meta['file']['sha256'],
                order_seed=meta['seed'], cpu_test=cpu)


def new_state(micro):
    return dict(updates=0, cursor=0, main_seconds=0., branch_seconds=0., branch_attempted_updates=0,
                AMP_skips=0, nonfinite_skips=0, last_lr=0., lr_factor=1., micro_batch=micro,
                next_point_main_seconds=9000., points=[], sessions=[],
                stability=dict(total=0, consecutive=0, window=[], escalations=0, history=[]))


def learning_rate(state):
    return L.lr_at(state['updates']+1, .001*state['lr_factor'], 1000)


def payload(model, opt, scaler, state, spec, identity):
    p = L.state_payload(model, opt, scaler, state, spec, identity)
    p.update(format='nlp-hw02-b1-full-v1', stability_policy=R.POLICY)
    p['scheduler'].update(kind='B1 warmup-constant with recorded emergency factor', lr_factor=state['lr_factor'])
    return p


def save_full(root, model, opt, scaler, state, spec, identity):
    root = Path(root); folder = root/'checkpoints'; folder.mkdir(parents=True, exist_ok=True)
    C.sync(next(model.parameters()).device)
    p = payload(model, opt, scaler, state, spec, identity)
    assert R.finite_payload(p), 'Refuse nonfinite stable state'
    path = folder/('main-'+uuid.uuid4().hex[:12]+'.pt'); tmp = path.with_suffix('.pt.tmp')
    torch.save(p, tmp); tmp.replace(path); del p
    receipt = dict(file=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha(path),
                   updates=state['updates'], cursor=state['cursor'], main_seconds=state['main_seconds'],
                   model_spec=spec, identity=identity, purpose='uncooled full B1 stable')
    write(root/'latest.json', receipt)
    for old in sorted(folder.glob('main-*.pt'), key=lambda p:p.stat().st_mtime_ns, reverse=True)[2:]: old.unlink()
    return receipt


def read_full(root, identity):
    root = Path(root); rec = json.loads((root/'latest.json').read_text()); path = root/safe_name(rec['file'])
    assert path.stat().st_size == rec['bytes'] and sha(path) == rec['sha256'], 'Stable SHA/bytes mismatch'
    p = torch.load(path, map_location='cpu', weights_only=True)
    assert p['format'] == 'nlp-hw02-b1-full-v1' and p['identity'] == identity == rec['identity']
    assert p['model_spec'] == identity['recipe']['architecture'] == rec['model_spec']
    assert p['stability_policy'] == R.POLICY and R.finite_payload(p)
    recipe = identity['recipe']
    assert identity['recipe_sha256'] == json_hash(recipe)
    for n, h in recipe['implementation_sha256'].items(): assert sha(Path(__file__).with_name(n)) == h, n
    assert recipe['torch'] == str(torch.__version__) and recipe['CUDA'] == torch.version.cuda, 'Environment changed'
    s = p['state']
    assert s['cursor'] == s['updates']*128
    assert all(rec[k] == s[k] for k in ('updates', 'cursor', 'main_seconds'))
    assert p['scheduler']['last_lr'] == s['last_lr'] and p['scheduler']['lr_factor'] == s['lr_factor']
    if s['updates']: assert all(g['lr'] == s['last_lr'] for g in p['optimizer']['param_groups'])
    steps = s['updates']-s['AMP_skips']-s['nonfinite_skips']
    assert all(int(v['step']) == steps for v in p['optimizer']['state'].values())
    return p


def restore(p, device):
    recipe = p['identity']['recipe']; cpu = p['identity']['cpu_test']
    m = C.SimpleTransformer(**p['model_spec'], dropout=recipe['dropout']).to(device)
    o = L.optimizer(m, p['state']['last_lr'], cpu)
    s = torch.amp.GradScaler(device.type, enabled=p['scaler_enabled'], init_scale=1024)
    state = L.restore_full(p, m, o, s, device)
    return m, o, s, state


def main_step(model, opt, scaler, state, cache, order, device):
    log, micro = R.safe_update(model, opt, scaler, cache, L.take(order, state['cursor'], 128),
                               device, state['micro_batch'], learning_rate(state))
    state.update(updates=state['updates']+1, cursor=state['cursor']+128, micro_batch=micro, last_lr=log['lr'])
    state['main_seconds'] += log['seconds']
    state['AMP_skips'] += log['AMP_skip']; state['nonfinite_skips'] += log['nonfinite_skip']
    return log, R.observe(state['stability'], log['nonfinite_skip'], amp_skip=log['AMP_skip'])


def recover(root, identity, device, history, elapsed, old_lr_factor, old_micro, reason):
    """L2 policy: rollback model/Adam/scaler/RNG/cursor, preserve spent time/history, halve LR."""
    m, o, s, state = restore(read_full(root, identity), device)
    history = copy.deepcopy(history); history['escalations'] += 1
    event = dict(reason=reason, restored_updates=state['updates'], restored_cursor=state['cursor'],
                 old_lr_factor=old_lr_factor, escalation=history['escalations'])
    history.update(consecutive=0, window=[], amp_consecutive=0)
    stopped = history['escalations'] >= R.POLICY['max_escalations']
    if not stopped:
        state['lr_factor'] = old_lr_factor*.5
        state['last_lr'] = L.lr_at(state['updates'], .001*state['lr_factor'], 1000)
        for g in o.param_groups: g['lr'] = state['last_lr']
        if reason == 'Adam OOM':
            choices = [v for v in L.MICRO_BATCHES if v < old_micro]
            if not choices: stopped = True
            else: state['micro_batch'] = choices[0]
    event.update(action='stop with last good stable' if stopped else 'rollback and halve LR',
                 new_lr_factor=state['lr_factor'])
    history['history'].append(event); state.update(stability=history, main_seconds=elapsed)
    write(Path(root)/'stability-history.json', history)
    return m, o, s, state, stopped


def metric_complete(metrics, final=False, smoke=False):
    try:
        for split, n in (('heldout4k', 64 if smoke else 4096), ('old991', 16 if smoke else 991)):
            row = metrics['splits'][split]
            if row['exec']['denominator'] != n or not math.isfinite(row['CE']['CE']): return False
        if final:
            vote = metrics['K8_tune2048']
            if not (vote['complete'] is True and vote['rows'] == vote['target_rows'] == (32 if smoke else 2048)
                    and metrics['test']['rows'] == (64 if smoke else 1919)): return False
        return True
    except (KeyError, TypeError, ValueError): return False


def next_mark(mark):
    return next((v for v in PROBE_MARKS if v > mark), 1.e12)


def run(name, cache, order, meta, root, device, cfg, evaluate, resume=None, spec=None, cpu=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True); spec = spec or SPECS[name]
    identity = identity_for(name, spec, cache, meta, cfg['dropout'], cpu)
    if resume:
        p = read_full(resume, identity); m, o, s, state = restore(p, device); del p
    else:
        m = new_model(spec, device, cfg['dropout'])
        micro, attempts = probe(m, cache, device, .001, name, cpu=cpu)
        write(root/'memory-probe.json', dict(micro_batch=micro, attempts=attempts, data_cursor_consumed=0))
        o = L.optimizer(m, .001, cpu); s = torch.amp.GradScaler(device.type, enabled=device.type=='cuda', init_scale=1024)
        state = new_state(micro)
    write(root/'identity.json', identity)
    save_full(root, m, o, s, state, spec, identity)
    origin = state['main_seconds']; start_updates = state['updates']; last_save = time.monotonic()
    deadline = cfg['deadline_monotonic']; reserve = cfg.get('io_reserve_seconds', 240.)
    final = cfg.get('final_session', False); final_done = False; status = 'complete'
    target = origin+cfg['main_seconds']
    def stopped(): return (root.parent/'stop-request.json').exists() or time.monotonic() >= deadline-reserve
    def branch(is_final, at_end=False):
        nonlocal m, o, s, state, last_save
        tag = ('final-' if is_final else 'end-main-' if at_end else 'probe-')+f'{int(state["main_seconds"]):06d}'
        out = root/'points'/tag; out.mkdir(parents=True, exist_ok=True)
        anchor = save_full(root, m, o, s, state, spec, identity)
        cap = cfg['final_cooldown_seconds'] if is_final else cfg['probe_cooldown_seconds']
        eval_cap = cfg['final_eval_seconds'] if is_final else cfg['probe_eval_seconds']
        # C uses timed linear cooldown over accumulated update seconds.
        estimate = state['main_seconds']/max(1, state['updates'])
        count = max(1, int(.85*cap/max(.001, estimate)))
        begin = time.monotonic(); done = 0; branch_s = 0.; metrics = None
        peak = state['last_lr']; q = cfg.get('final_q', .1) if is_final else 0.
        training = dict(anchor=anchor, main_seconds=state['main_seconds'], final=is_final, at_end=at_end,
                        target_main_seconds=target if at_end or is_final else state['next_point_main_seconds'],
                        cooldown_target_seconds=cap, planned_updates=count if is_final else None,
                        cooldown='1-sqrt' if is_final else 'C timed linear', q=q)
        try:
            assert deadline-time.monotonic() >= cap+eval_cap+reserve, 'Insufficient branch reserve'
            from r1_growth import update
            stability = copy.deepcopy(state['stability'])
            while done < count if is_final else branch_s < cap:
                if stopped() or time.monotonic()-begin >= cap+max(5., estimate*3):
                    raise C.BudgetExpired('Cooldown incomplete')
                j = done+1
                progress = j/count if is_final else min(1., branch_s/cap)
                lr = peak*(1-math.sqrt(progress) if is_final else 1-progress)
                log, micro = update(m, o, s, cache, L.take(order, state['cursor']+done*128, 128),
                                    device, state['micro_batch'], lr, j, 1, q)
                done += 1; branch_s += log['seconds']; state['micro_batch'] = micro
                if R.observe(stability, log['nonfinite_skip'], amp_skip=log['AMP_skip']): raise R.UnsafeState('Cooldown unstable')
                if 'truncation' in log:
                    with (out/'truncation.jsonl').open('a') as f: f.write(json.dumps(log['truncation'])+'\n')
            training.update(completed_updates=done, cooldown_seconds=branch_s, endpoint_lr=lr)
            assert (done == count and lr == 0.) if is_final else branch_s >= cap
            point = L.save_point(root, tag, m, spec, identity, training)
            del m, o, s; m = o = s = None; R.release(device)
            metrics = evaluate(point, out/'evaluation', min(deadline-reserve, time.monotonic()+eval_cap), is_final)
            assert metric_complete(metrics, is_final, cfg.get('gpu_smoke', False)), 'Incomplete evaluation'
            write(out/'complete.json', dict(training=training, metrics=metrics, weights_sha256=sha(point)))
        except Exception as error:
            metrics = None
            write(out/'incomplete.json', dict(reason=type(error).__name__, training=training, completed_updates=done))
        finally:
            if m is not None: del m, o, s
            R.release(device); m, o, s, state = restore(read_full(root, identity), device)
            state['branch_seconds'] += branch_s; state['branch_attempted_updates'] += done
            state['points'].append(dict(tag=tag, training=training, complete=metrics is not None, metrics=metrics))
            if not is_final and not at_end: state['next_point_main_seconds'] = next_mark(state['next_point_main_seconds'])
            save_full(root, m, o, s, state, spec, identity); last_save = time.monotonic()
        return metrics is not None
    try:
        while state['main_seconds'] < target and not stopped():
            if cfg.get('max_updates') and state['updates']-start_updates >= cfg['max_updates']: break
            if final and deadline-time.monotonic() < cfg['final_cooldown_seconds']+cfg['final_eval_seconds']+reserve+60: break
            if state['main_seconds'] >= state['next_point_main_seconds']:
                needed = cfg['probe_cooldown_seconds']+cfg['probe_eval_seconds']+reserve
                if final: needed += cfg['final_cooldown_seconds']+cfg['final_eval_seconds']+60
                if deadline-time.monotonic() >= needed:
                    branch(False); continue
                # Persist missed mark; never relabel an evaluation at a later time.
                state.setdefault('missed_points', []).append(state['next_point_main_seconds'])
                state['next_point_main_seconds'] = next_mark(state['next_point_main_seconds'])
            unsafe = None; begin = time.perf_counter()
            try: log, escalate = main_step(m, o, s, state, cache, order, device)
            except L.UnsafeAdamOOM: unsafe = 'Adam OOM'
            except R.UnsafeState: unsafe = 'nonfinite post-Adam'
            if unsafe:
                C.sync(device); state['main_seconds'] += time.perf_counter()-begin
            if unsafe or escalate:
                h, elapsed, factor, micro = copy.deepcopy(state['stability']), state['main_seconds'], state['lr_factor'], state['micro_batch']
                del m, o, s; R.release(device)
                m, o, s, state, stop = recover(root, identity, device, h, elapsed, factor, micro, unsafe or 'nonfinite threshold')
                save_full(root, m, o, s, state, spec, identity); last_save = time.monotonic()
                if stop: status = 'stability_stop'; break
                continue
            if state['updates'] % 100 == 0:
                write(root/'progress.json', dict(state=state, last_update=log))
                print('B1', name, state['updates'], round(state['main_seconds'], 1), 'CE', log['CE'], 'LR', state['last_lr'], flush=True)
            if time.monotonic()-last_save >= cfg.get('checkpoint_seconds', 900.):
                save_full(root, m, o, s, state, spec, identity); last_save = time.monotonic()
        save_full(root, m, o, s, state, spec, identity)
        reached = state['main_seconds'] >= target
        if status == 'complete' and reached and not stopped():
            final_done = branch(final, at_end=True)
        if status == 'complete' and (not reached or not final_done or state.get('missed_points')
                or any(not p['complete'] for p in state['points'])):
            status = 'partial; main target or evaluation incomplete'
        state['sessions'].append(dict(owner=cfg.get('owner'), origin_main_seconds=origin, final=final,
                                      end_main_seconds=state['main_seconds'], status=status))
        save_full(root, m, o, s, state, spec, identity)
        result = dict(status=status, model=name, parameter_count=sum(p.numel() for p in m.parameters()),
                      state=state, identity=identity, final_complete=final_done if final else False, end_main_complete=final_done,
                      main_target_seconds=target, main_target_reached=state['main_seconds'] >= target)
        write(root/'summary.json', result)
        return result
    finally:
        del m, o, s; R.release(device)
