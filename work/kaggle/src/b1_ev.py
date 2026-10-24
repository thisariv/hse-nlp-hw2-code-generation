"""EV only: frozen B1 full-state restore, timed C cooldown, greedy/CE; no main steps."""
import copy
import json
import time
from pathlib import Path
import torch
import b1_runtime as B
import b1_eval as E
import l1_runtime as L
import l2_runtime as R
from kaggle_io import sha, write


def compact_metrics(value):
    if isinstance(value,dict):return {k:compact_metrics(v) for k,v in value.items() if not k.endswith('_indices')}
    if isinstance(value,list):return [compact_metrics(v) for v in value]
    return value


def cooldown(anchor, cache, order, device, root, deadline, seconds=900.):
    identity = json.loads((Path(anchor)/'latest.json').read_text())['identity']
    payload = B.read_full(anchor, identity)
    m, o, s, state = B.restore(payload, device); spec = payload['model_spec']; del payload
    estimate = state['main_seconds']/max(1, state['updates'])
    begin = time.monotonic(); spent = 0.; done = 0; peak = state['last_lr']
    stability = copy.deepcopy(state['stability'])
    training = dict(main_seconds=state['main_seconds'], cursor=state['cursor'], cooldown='C timed linear',
                    cooldown_target_seconds=seconds, q=0., peak_lr=peak, new_main_updates=0)
    try:
        while spent < seconds:
            assert time.monotonic() < min(deadline, begin+seconds+max(5.,3*estimate)), 'Cooldown deadline'
            lr = peak*(1-min(1.,spent/seconds))
            log, micro = R.safe_update(m,o,s,cache,L.take(order,state['cursor']+done*128,128),
                                       device,state['micro_batch'],lr)
            state['micro_batch'] = micro; done += 1; spent += log['seconds']
            assert not R.observe(stability,log['nonfinite_skip'],amp_skip=log['AMP_skip']), 'Cooldown unstable'
        training.update(completed_updates=done,cooldown_seconds=spent,endpoint_lr=lr)
        return L.save_point(root,'ev-034200',m,spec,identity,training), training
    finally:
        del m,o,s; R.release(device)


def run(gpu, cfg, cache, order, root, inputs, sets, refs, device):
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    source = Path(cfg['resume_root'])/f'model-GPU{gpu}'
    results = {}; end = cfg['deadline_monotonic']-30
    # Mandatory final E and control W first; prepared 28370 follows independently.
    try:
        point, training = cooldown(source,cache,order,device,root,end, cfg['cooldown_seconds'])
        results['34200'] = dict(training=training,metrics=E.evaluate(point,root/'eval-34200',inputs,sets,refs,
                                   device,min(end,time.monotonic()+cfg['eval_seconds'])))
        assert B.metric_complete(results['34200']['metrics'])
        results['34200']['metrics']=compact_metrics(results['34200']['metrics'])
        if gpu==1:
            results['34200']['control_reference_heldout4k']=2488/4096
            results['34200']['control_delta_pp']=100*(results['34200']['metrics']['splits']['heldout4k']['exec']['accuracy']-2488/4096)
        write(root/'ev-results.json',results)
    except Exception as error:
        import traceback
        traceback.print_exc(); results['34200'] = dict(error=type(error).__name__)
        write(root/'ev-results.json',results)
    point = source/'weights/probe-028370-fp16.pt'
    metrics = E.evaluate(point,root/'eval-28370',inputs,sets,refs,device,min(end,time.monotonic()+cfg['eval_seconds']))
    assert B.metric_complete(metrics)
    results['28370'] = dict(weights_sha256=sha(point),prepared_cooldown=True,metrics=compact_metrics(metrics))
    write(root/'ev-results.json',results)
    passed = 'metrics' in results['34200']
    write(root/'summary.json',dict(status='complete' if passed else 'partial',model=('E','W')[gpu],points=results))
    assert passed, 'Final cooldown/evaluation incomplete; inspect ev-results'
