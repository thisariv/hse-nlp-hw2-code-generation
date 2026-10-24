"""Strict adapters for the two historical formats. Frozen training math is untouched."""
import json
from pathlib import Path
import torch
import b2_runtime as B
import b4_runtime as W
import exp26_core as C
import l1_runtime as L
import l2_runtime as R
from exp26_common import json_hash
from kaggle_io import sha, safe_name


def is_w(identity):
    return identity['model']=='W-lr600'


def validate(p, pin, environment=True):
    identity=pin['identity']; recipe=identity['recipe']; state=p['state']
    assert p['identity']==identity and p['model_spec']==pin['model_spec']==recipe['architecture']
    assert identity['recipe_sha256']==json_hash(recipe) and R.finite_payload(p)
    assert p['stability_policy']==R.POLICY
    for key in ('updates','cursor','main_seconds'):
        assert state[key]==pin[key], key
    for name,h in recipe['implementation_sha256'].items():
        assert sha(Path(__file__).with_name(name))==h, name
    if environment:
        assert str(torch.__version__)==recipe['torch'] and torch.version.cuda==recipe['CUDA'], 'Environment changed'
    if is_w(identity):
        W.validate_anchor(p,pin)
        assert recipe['z_loss']==1e-4 and recipe['main_truncation_q']==0.
    else:
        assert identity['model']=='C-long2x' and p['format']=='nlp-hw02-l2-full-v1'
        for name,h in p['runtime_sha256'].items(): assert sha(Path(__file__).with_name(name))==h
        assert state['l2_main_seconds']==pin['l2_main_seconds']
        assert state['cursor']==state['cursor_origin']+state['updates']*128
        assert state['updates']>recipe['warmup_updates'] and state['last_lr']==.0015
        assert p['scheduler']['last_lr']==state['last_lr']
        assert state['stability']['escalations']==0
        assert all(g['lr']==state['last_lr'] for g in p['optimizer']['param_groups'])
        steps=state['updates']-state['AMP_skips']-state['nonfinite_skips']
        assert p['optimizer']['state'] and all(int(v['step'])==steps for v in p['optimizer']['state'].values())
    assert p['scaler_enabled'] and p['scaler'] and p['rng']['torch_cuda']


def read_full(root, identity, environment=True):
    root=Path(root); rec=json.loads((root/'latest.json').read_text()); path=root/safe_name(rec['file'])
    assert rec['identity']==identity and path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256']
    if environment:
        p=B.read_full(root,identity) if is_w(identity) else R.read_full(path,identity)
    else:
        p=torch.load(path,map_location='cpu',weights_only=True)
    # Every saved B5 latest must obey the same historical format and constant LR.
    validate(p,rec,environment)
    return p


def read_anchor(root, pin, cache, meta, environment=True):
    assert json.loads((Path(root)/'latest.json').read_text())==pin, 'Wrong stable anchor; no fallback'
    identity=pin['identity']
    assert cache.manifest_sha==identity['cache_manifest_sha256']==meta['cache_manifest_sha256']
    assert identity['order_sha256']==meta['file']['sha256']
    assert meta['seed']==(identity.get('order_seed') if is_w(identity) else identity['recipe']['sampler']['seed'])
    return read_full(root,identity,environment)


def restore(p,device):
    identity=p['identity']
    if is_w(identity): return B.restore(p,device)
    m=C.SimpleTransformer(**p['model_spec'],dropout=identity['recipe']['dropout']).to(device)
    o=L.optimizer(m,p['state']['last_lr'],identity.get('cpu_test',False))
    s=torch.amp.GradScaler(device.type,enabled=p['scaler_enabled'],init_scale=1024)
    state=L.restore_full(p,m,o,s,device)
    return m,o,s,state


def save_full(root,m,o,s,state,spec,identity):
    return (B.save_full if is_w(identity) else R.save_full)(root,m,o,s,state,spec,identity)


def update_for(identity,m,o,s,cache,ids,device,micro,lr):
    return (B.O.safe_update if is_w(identity) else R.safe_update)(m,o,s,cache,ids,device,micro,lr)


def main_step(m,o,s,state,cache,order,device,identity):
    if is_w(identity): return B.main_step(m,o,s,state,cache,order,device)
    lr=state['last_lr']; assert lr==.0015
    log,micro=R.safe_update(m,o,s,cache,L.take(order,state['cursor'],128),device,state['micro_batch'],lr)
    state.update(updates=state['updates']+1,cursor=state['cursor']+128,micro_batch=micro)
    state['main_seconds']+=log['seconds'];state['l2_main_seconds']+=log['seconds']
    state['AMP_skips']+=log['AMP_skip'];state['nonfinite_skips']+=log['nonfinite_skip']
    return log,R.observe(state['stability'],log['nonfinite_skip'],amp_skip=log['AMP_skip'])
