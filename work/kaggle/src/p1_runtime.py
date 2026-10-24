"""Two disposable paired cooldowns and an independent reverse D0/WSD trajectory."""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import exp26_core as C
import exp26_train as T
import l1_runtime as L
import l2_runtime as R
from exp26_common import json_hash
from kaggle_io import sha, write
from p1_math import sqrt_lr, safe_truncated_update
from l2_gpu_smoke import digest


def state_fingerprints(model, opt, scaler, state):
    """Compare tensor bytes, including all RNGs, rather than only counters."""
    return {k:digest(v) for k,v in dict(model=model.state_dict(), optimizer=opt.state_dict(),
                                      scaler=scaler.state_dict(), rng=T.rng_state(), cursor_history=state).items()}


def reviewer_metrics_complete(metrics, expected_rows):
    return (isinstance(metrics, dict) and set(metrics) == set(expected_rows)
            and all(isinstance(metrics[k], dict) and metrics[k].get('complete') is True
                    and metrics[k].get('rows') == n and metrics[k].get('tokens', 0) > 0
                    and np.isfinite(metrics[k].get('CE', float('nan')))
                    for k,n in expected_rows.items()))


def restore_anchor(anchor, device, cpu=False):
    model = C.SimpleTransformer(**anchor['model_spec'], dropout=0.).to(device)
    opt = L.optimizer(model, anchor['state']['last_lr'], cpu)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda', init_scale=1024)
    state = L.restore_full(anchor, model, opt, scaler, device)
    return model, opt, scaler, state


def stopped(root, deadline):
    return (Path(root).parent/'stop-request.json').exists() or time.monotonic() >= deadline


def metrics_complete(metrics, rows):
    vote = (metrics or {}).get('K8_tune2048', {})
    return vote.get('complete') is True and vote.get('rows') == vote.get('target_rows') == rows


def run_pilot(anchor_path, pin, cache, order, root, device, cfg, evaluate, cpu=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    from l3_resume import verify_anchor

    def fresh():
        _, p = verify_anchor(Path(anchor_path).parent, pin, check_environment=not cpu, expected_points=5)
        return p

    results = []; initial_fingerprints = None
    length = cfg['pilot_updates']
    for branch, truncate, cap in [('a', False, cfg['pilot_a_cap_seconds']),
                                   ('b', True, cfg['pilot_b_cap_seconds'])]:
        # Reload from immutable L3 input for EACH branch. No A state enters B.
        anchor = fresh()
        model, opt, scaler, state = restore_anchor(anchor, device, cpu)
        fingerprints = state_fingerprints(model, opt, scaler, state)
        if initial_fingerprints is None: initial_fingerprints = fingerprints
        assert fingerprints == initial_fingerprints, 'A/B full starting state mismatch'
        identity = anchor['identity']; spec = anchor['model_spec']; del anchor
        if branch == 'a':
            R.save_full(root, model, opt, scaler, state, spec, identity)
        branch_root = root.parent/f'model-pilot-{branch}'
        out = branch_root/'points'/('long' if branch == 'a' else 'long-truncation')
        out.mkdir(parents=True, exist_ok=True)
        begin = time.monotonic()
        until = min(begin+cap, cfg['deadline_monotonic']-cfg['pilot_eval_seconds']-120*(not cpu))
        training = dict(branch=branch, truncation=truncate, planned_cooldown_updates=length,
                        cooldown_updates=0, cooldown_AMP_skips=0, cooldown_nonfinite_skips=0,
                        anchor_updates=state['updates'], anchor_cursor=state['cursor'],
                        anchor_sha256=pin['sha256'], peak_lr=state['last_lr'], floor_lr=0.,
                        schedule='1-sqrt; update axis; step 1..T', main_added_updates=0,
                        start_fingerprints=fingerprints)
        stream = hashlib.sha256(); metrics = None; failure = None
        stability = copy.deepcopy(state['stability'])
        aggregate = dict(presented=0, excluded=0, presented_tokens=0, retained_tokens=0,
                         counts=[0]*4, excluded_by_bin=[0]*4, graph_updates=0, graph_OOM_fallbacks=0)
        update_seconds = 0.; measured_seconds = 0.; measured_updates = 0
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        try:
            for step in range(1, length+1):
                if stopped(root, until):
                    raise C.BudgetExpired('Full paired cooldown cannot fit')
                ids = L.take(order, state['cursor']+(step-1)*128, 128)
                stream.update(np.asarray(ids, dtype='<u4').tobytes())
                lr = sqrt_lr(step, length, state['last_lr'])
                update = safe_truncated_update if truncate else R.safe_update
                log, micro = update(model, opt, scaler, cache, ids, device, state['micro_batch'], lr)
                state['micro_batch'] = micro
                update_seconds += log['seconds']
                if step > 8:
                    measured_seconds += log['seconds']; measured_updates += 1
                training['cooldown_updates'] += 1
                training['cooldown_AMP_skips'] += log['AMP_skip']
                training['cooldown_nonfinite_skips'] += log['nonfinite_skip']
                if truncate and 'truncation' in log:
                    stats = log['truncation']
                    with (out/'truncation.jsonl').open('a') as f:
                        f.write(json.dumps(dict(step=step, **stats), allow_nan=False)+'\n')
                    aggregate['presented'] += stats['presented']
                    aggregate['excluded'] += stats['presented']-stats['retained']
                    for k in ('presented_tokens', 'retained_tokens'):
                        aggregate[k] += stats[k]
                    aggregate['graph_updates'] += int(stats['forward_mode'] == 'graph')
                    aggregate['graph_OOM_fallbacks'] += int(stats['graph_OOM_fallback'])
                    aggregate['counts'] = [a+b for a,b in zip(aggregate['counts'], stats['counts'])]
                    aggregate['excluded_by_bin'] = [a+b for a,b in zip(aggregate['excluded_by_bin'], stats['excluded'])]
                if R.observe(stability, log['nonfinite_skip'], amp_skip=log['AMP_skip']):
                    raise FloatingPointError('Cooldown recovery threshold; discard branch')
                if step % 100 == 0 or step == length:
                    write(out/'progress.json', dict(step=step, planned=length, lr=lr, last_update=log))
                    print('P1 PILOT', branch, step, length, 'LR', lr, 'CE', log['CE'], flush=True)
            training.update(cooldown_wall_seconds=time.monotonic()-begin, stream_sha256=stream.hexdigest(),
                            final_lr=lr, truncation_summary=aggregate)
            write(out/'training.json', training)
            point = L.save_point(branch_root, out.name, model, spec, identity, training)
            del model, opt, scaler; model = opt = scaler = None; R.release(device)
            metrics = evaluate(point, out/'evaluation',
                               min(cfg['deadline_monotonic']-60*(not cpu), time.monotonic()+cfg['pilot_eval_seconds']))
            if not metrics_complete(metrics, cfg['pilot_k8_rows']):
                raise RuntimeError('Missing/incomplete K8')
            write(out/'complete.json', dict(training=training, metrics=metrics, weights_sha256=sha(point)))
        except Exception as error:
            failure = type(error).__name__
            # Retain finite partial weights and actual stream/time; never label them final.
            if model is not None and failure not in ('UnsafeState', 'UnsafeAdamOOM', 'FloatingPointError'):
                L.save_point(branch_root, out.name, model, spec, identity, training)
            write(out/'incomplete.json', dict(reason=failure, training=training))
        finally:
            training.setdefault('cooldown_wall_seconds', time.monotonic()-begin)
            training.update(branch_wall_seconds=time.monotonic()-begin, update_seconds=update_seconds,
                            throughput_updates_per_second=measured_updates/measured_seconds if measured_seconds else None,
                            timed_updates_after_warmup=measured_updates, stream_sha256=stream.hexdigest(),
                            truncation_summary=aggregate)
            if device.type == 'cuda':
                training.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            write(out/'training.json', training)
            # Rewrite receipts with timings finalized even when evaluation failed.
            if failure is None:
                write(out/'complete.json', dict(training=training, metrics=metrics, weights_sha256=sha(point)))
            else: write(out/'incomplete.json', dict(reason=failure, training=training))
            if model is not None:
                del model, opt, scaler
            R.release(device)
            # Restore EVERYTHING even after a corrupted optimizer or evaluation crash.
            restored = fresh(); model, opt, scaler, state = restore_anchor(restored, device, cpu)
            assert state == restored['state']
            after = state_fingerprints(model, opt, scaler, state)
            assert after == initial_fingerprints, 'Full model/optimizer/scaler/RNG restore mismatch'
            write(out/'restore.json', dict(passed=True, full_source_sha256=pin['sha256'],
                  fingerprints=after,
                  updates=state['updates'], cursor=state['cursor'], last_lr=state['last_lr'],
                  main_continued=False, inherited_history_unchanged=True))
            del model, opt, scaler, restored; R.release(device)
        results.append(dict(branch=branch, complete=failure is None, failure=failure,
                            training=training, metrics=metrics))
        write(root/'summary.json', dict(model='pilot', status='running', branches=results))
    paired = all(r['complete'] for r in results)
    if paired:
        assert results[0]['training']['stream_sha256'] == results[1]['training']['stream_sha256']
    report = dict(model='pilot', status='complete' if paired else 'partial', branches=results,
                  same_stream=paired, main_added_updates=0)
    write(root/'summary.json', report)
    return report


def reviewer_identity(spec, cache, order_sha, cpu=False, warmup=1000):
    identity = L.identity_for('C', spec, cache, order_sha, cpu, warmup=warmup)
    identity['model'] = 'reviewer'
    identity['recipe'].update(direction='compact-code-with-mode -> original question',
                               schedule='warmup -> constant -> disposable final 1-sqrt10% wall',
                               protected_cache_indices=[8545870])
    identity['recipe']['implementation_sha256'].update(
        {n:sha(Path(__file__).with_name(n)) for n in ('p1_math.py', 'p1_runtime.py')})
    identity['recipe_sha256'] = json_hash(identity['recipe'])
    return identity


def new_reviewer_state():
    return dict(updates=0, cursor=0, cursor_origin=0, main_seconds=0., l2_main_seconds=0.,
                branch_seconds=0., branch_attempted_updates=0, AMP_skips=0, nonfinite_skips=0,
                last_lr=.0015, micro_batch=64, l2_points=[], emergency_factor=1.,
                l2_origin=dict(main_seconds=0., updates=0, cursor=0),
                stability=dict(total=0, consecutive=0, window=[], escalations=0, history=[]))


def run_reviewer(cache, order, order_sha, root, device, cfg, validate, cpu=False, spec=None):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    warmup = cfg.get('reviewer_warmup_updates',1000)
    spec = spec or L.SPECS['C']; identity = reviewer_identity(spec, cache, order_sha, cpu, warmup)
    model = L.new_model(spec, device); opt = L.optimizer(model, .0015, cpu)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda', init_scale=1024)
    state = new_reviewer_state()
    # Reverse direction has a different memory shape: probe its longest rows anew.
    micro, attempts = L.probe(model, cache, device, .0015, smoke=cpu, batch=128)
    state['micro_batch'] = min(micro, 64)
    write(root/'memory-probe.json', dict(choice=micro, attempts=attempts, reverse=True))
    write(root/'identity.json', identity)
    R.save_full(root, model, opt, scaler, state, spec, identity)
    start = time.monotonic()
    train_until = cfg['deadline_monotonic']-cfg['reviewer_eval_seconds']-cfg['reviewer_io_reserve_seconds']
    span = max(0., train_until-start)
    cool_start = start + .9*span
    last_save = time.monotonic(); last_eval = start; cooldown = False; cooled = 0; status = 'complete'
    metrics = []
    expected_rows = cfg.get('reviewer_expected_rows', {'heldout4k':4096, 'old991':991})
    try:
        while not stopped(root, train_until):
            now = time.monotonic()
            if not cooldown and now >= cool_start:
                R.save_full(root, model, opt, scaler, state, spec, identity)
                write(root/'stable-before-cooldown.json', json.loads((root/'latest.json').read_text()))
                cooldown = True
            lr = (sqrt_lr(min(now-cool_start, train_until-cool_start), train_until-cool_start)
                  if cooldown else L.lr_at(state['updates']+1, .0015, warmup)) * state['emergency_factor']
            try:
                log, micro = R.safe_update(model, opt, scaler, cache, L.take(order, state['cursor'],128),
                                          device, state['micro_batch'], lr)
                unsafe = R.observe(state['stability'], log['nonfinite_skip'], amp_skip=log['AMP_skip'])
            except (R.UnsafeState, L.UnsafeAdamOOM):
                unsafe = True; log = None
            if unsafe:
                history = copy.deepcopy(state['stability']); history['escalations'] += 1
                from_updates = state['updates']; old_micro = state['micro_batch']
                del model, opt, scaler; R.release(device)
                model, opt, scaler, state = R.reload_latest(root, identity, device, cpu)
                history.update(consecutive=0, window=[], amp_consecutive=0)
                history['history'].append(dict(from_updates=from_updates, restored_updates=state['updates'],
                                                phase='cooldown' if cooldown else 'stable'))
                state['stability'] = history
                write(root/'stability-history.json', history)
                if cooldown or history['escalations'] >= R.POLICY['max_escalations']:
                    status = 'partial; stability_stop'; break
                state['emergency_factor'] *= .5; state['last_lr'] *= .5
                for group in opt.param_groups: group['lr'] = state['last_lr']
                state['micro_batch'] = max(1, min(state['micro_batch'], old_micro//2))
                R.save_full(root, model, opt, scaler, state, spec, identity)
                continue
            state.update(micro_batch=micro, last_lr=lr)
            state['updates'] += 1; state['cursor'] += 128
            state['AMP_skips'] += log['AMP_skip']; state['nonfinite_skips'] += log['nonfinite_skip']
            state['main_seconds'] += log['seconds']; state['l2_main_seconds'] += log['seconds']
            cooled += int(cooldown)
            if state['updates'] % 100 == 0:
                write(root/'progress.json', dict(state, cooldown=cooldown, last_chunk=log))
                print('P1 REVIEWER', state['updates'], 'CE', log['CE'], 'LR', lr, flush=True)
            if not cooldown and time.monotonic()-last_save >= cfg.get('checkpoint_seconds',1800):
                R.save_full(root, model, opt, scaler, state, spec, identity); last_save = time.monotonic()
            if not cooldown and time.monotonic()-last_eval >= cfg.get('reviewer_validation_interval',10800):
                periodic = validate(model, min(cool_start, time.monotonic()+180), 'periodic')
                if not reviewer_metrics_complete(periodic, expected_rows): raise RuntimeError('Incomplete reviewer periodic CE')
                metrics.append(periodic)
                write(root/'validation.json', metrics); last_eval = time.monotonic(); model.train()
        if not cooldown or cooled == 0 or (root.parent/'stop-request.json').exists():
            status = 'partial; cooldown not completed'
        if status == 'complete':
            # Last discrete update approaches zero; record exact endpoint, no extra data.
            training = dict(state=state, cooldown_updates=cooled, cooldown_fraction=.1,
                            train_wall_seconds=time.monotonic()-start, schedule_endpoint_lr=0.,
                            final_update_lr=state['last_lr'],
                            stable_resume='latest.json; before final cooldown')
            point = L.save_point(root, 'final', model, spec, identity, training)
            final = validate(model, min(cfg['deadline_monotonic']-60*(not cpu),
                                       time.monotonic()+cfg['reviewer_eval_seconds']), 'final')
            metrics.append(final)
            if not reviewer_metrics_complete(final, expected_rows): raise RuntimeError('Incomplete reviewer final CE')
            write(root/'points/final/complete.json', dict(training=training, metrics=final, weights_sha256=sha(point)))
        elif not cooldown:
            R.save_full(root, model, opt, scaler, state, spec, identity)
        report = dict(model='reviewer', status=status, state=state, validation=metrics,
                      stable_before_cooldown=(root/'latest.json').exists())
        write(root/'summary.json', report)
        return report
    except Exception as error:
        # FP16 final weights (if made) and uncooled latest survive failed evaluation.
        report = dict(model='reviewer', status='partial', failure=type(error).__name__,
                      state=state, validation=metrics, stable_before_cooldown=(root/'latest.json').exists())
        write(root/'incomplete.json', report); write(root/'summary.json', report)
        return report
    finally:
        del model, opt, scaler; R.release(device)
