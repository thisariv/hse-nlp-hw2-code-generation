"""Pinned L2 continuation: keep both identities, weighted order, cursor and full history."""
import gzip, json, shutil
from pathlib import Path
import torch
import l1_runtime as L, l2_runtime as R
from kaggle_io import sha, safe_name, write, resolve_dataset

def resolve_cache_mount(input_root, source, kind):
    """Foreign sources use owner-qualified mounts; own datasets may still be flat."""
    root=Path(input_root);owner,slug=source.split('/')[:2]
    for candidate in (root/kind/owner/slug,root/slug):
        if candidate.is_dir():return candidate
    return resolve_dataset(root,owner+'/'+slug)

def live_bytes(source, entry):
    """Accept original gzip or Kaggle's expanded mount, verifying the producer's bytes."""
    path=Path(source)/safe_name(entry['file'])
    if path.exists():
        raw=path.read_bytes()
        assert len(raw)==entry['bytes'] and sha(path)==entry['sha256'], 'Live entry mismatch'
        return gzip.decompress(raw) if entry['gzip'] else raw
    assert entry['gzip'] and path.suffix=='.gz', 'Missing live entry'
    raw=path.with_suffix('').read_bytes()
    packed=gzip.compress(raw,compresslevel=1,mtime=0)
    import hashlib
    assert len(packed)==entry['bytes'] and hashlib.sha256(packed).hexdigest()==entry['sha256']
    return raw

def mounted_expected(source, name, pin):
    source=Path(source);manifest=json.loads((source/'recover-manifest.json').read_text())
    assert manifest['format']=='nlp-hw02-l2-live-v1'
    latest=next(r for r in manifest['files'] if r['original_path']==f'model-{name}/latest.json')
    receipt=json.loads(live_bytes(source,latest));original=receipt['file']
    blob=next(r for r in manifest['files'] if r['original_path']==f'model-{name}/{original}')
    assert blob['file']==pin['file'] and blob['sha256']==receipt['sha256']==pin['sha256']
    assert blob['bytes']==receipt['bytes']==pin['bytes']
    assert dict(receipt,file=blob['file'],original_file=original)==pin, 'Pinned L2 latest changed'
    return pin

def verify_anchor(source, expected, check_environment=True, expected_points=3):
    path=Path(source)/safe_name(expected['file'])
    assert path.stat().st_size==expected['bytes'] and sha(path)==expected['sha256'], 'L2 anchor SHA/bytes mismatch'
    p=R.read_full(path,expected['identity'],check_environment=check_environment)
    for key in ('updates','cursor','main_seconds','l2_main_seconds'):
        assert p['state'][key]==expected[key], 'L2 anchor '+key+' mismatch'
    assert p['model_spec']==expected['model_spec']==L.SPECS['C']
    assert p['state']['last_lr']==.0015 and p['scaler_enabled'] and p['rng']['torch_cuda']
    assert p['state']['stability']['escalations']==0, 'Unexpected recovered L2 recipe'
    history=p['state']['l2_points']
    assert len(history)==expected_points and all(r['complete'] for r in history)
    if expected_points==5:
        assert [r['tag'] for r in history]==['l2-10800','l2-21600','l2-28800','l2-45540','l2-62280']
        assert history[-1].get('metrics',{}).get('K8_tune2048',{}).get('complete') is True, 'Incomplete L3 final K8'
    return path,p

def restore_order(source, destination, expected):
    source,destination=Path(source),Path(destination);destination.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'recover-manifest.json').read_text())
    for filename in ('order.npy','order.json','split.json'):
        entry=next(r for r in manifest['files'] if r['original_path']=='weighted-order/'+filename)
        target=destination/filename
        if filename=='order.npy':
            path=source/safe_name(entry['file'])
            assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']==expected['file']['sha256']
            shutil.copyfile(path,target)
        else:target.write_bytes(live_bytes(source,entry))
    assert json.loads((destination/'order.json').read_text())==expected, 'Weighted sampler metadata changed'
    return destination

def phase_recipe(config, anchor):
    # R.run uses cumulative L2 main time. Offset each model from its exact saved value.
    origin=anchor['state']['l2_main_seconds'];recipe=config['phase_recipe']
    return dict(recipe,main_targets=[origin+seconds for seconds in recipe['main_targets']])

def completion_status(result, targets, final_k8_rows=None):
    """The inherited runtime preserves state on eval failure, but can still say complete."""
    if result['status'] != 'complete':
        return result
    points={p['tag']:p for p in result['state']['l2_points']}
    for target in targets:
        point=points.get(f'l2-{int(target):05d}')
        metrics=point.get('metrics') if point else None
        if not point or not point['complete'] or metrics is None or not metrics.get('K8_tune2048',{'complete':True}).get('complete',False):
            result['status']='partial; L3 cooldown/evaluation incomplete'
            break
    if result['status']=='complete' and final_k8_rows is not None:
        final=points[f'l2-{int(targets[-1]):05d}']['metrics']
        vote=final.get('K8_tune2048',{})
        if (vote.get('complete') is not True or vote.get('rows')!=final_k8_rows
                or vote.get('target_rows')!=final_k8_rows):
            result['status']='partial; L4 final K8 missing or incomplete'
    return result

def run_continuation(name, anchor, identity, cache, order, root, device, deadline, evaluate, config):
    recipe=phase_recipe(config,anchor)
    result=R.run(name,anchor,identity,cache,order,root,device,deadline,evaluate,**recipe)
    completion_status(result,recipe['main_targets'],
                      final_k8_rows=2048 if config.get('session_kind')=='l4' and not config.get('gpu_smoke') else None)
    write(Path(root)/'summary.json',result)
    return result

def exercise_smoke(gpu,name,anchor,identity,cache,order,inputs,sets,refs,device,config):
    """L2's existing CUDA checks, rebased without modifying its frozen runtime modules."""
    import copy
    from unittest.mock import patch
    import l2_gpu_smoke as S
    original_run=R.run;origin=anchor['state']['l2_main_seconds'];history=copy.deepcopy(anchor['state']['l2_points'])
    def continued_state(payload, unused_name):
        # The L2 smoke's serialization probe must save the actual inherited full state.
        return copy.deepcopy(payload['state'])
    def run(*args,**kwargs):
        primary=len(kwargs['main_targets'])==2
        kwargs['main_targets']=[origin+t for t in kwargs['main_targets']]
        result=original_run(*args,**kwargs)
        assert result['state']['l2_points'][:len(history)]==history
        # Its legacy assertions count only the two newly evaluated smoke marks.
        if primary:
            result=copy.deepcopy(result)
            result['state']['l2_points']=result['state']['l2_points'][len(history):]
        return result
    # Legacy smoke uses GPU1 cursor0. Feed an equivalent rotated view for its finite-math
    # comparison only; production run/reload always receives the original order/cursor.
    old_take=L.take
    def take(candidate,cursor,count):
        return old_take(order,anchor['state']['cursor'],count) if gpu==1 and cursor==0 else old_take(candidate,cursor,count)
    # R.run uses actual nonzero cursors; the patched take only changes the legacy probe.
    with patch.object(S,'initial_state',continued_state),patch.object(R,'run',run),patch.object(L,'take',take):
        S.exercise(gpu,name,anchor,identity,cache,order,inputs,sets,refs,device,config)
