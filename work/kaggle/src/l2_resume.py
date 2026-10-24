"""Fail-fast pinned L1 C preflight. No D blobs, no newest-state guessing."""
import copy, json
from pathlib import Path
import torch
import l1_runtime as L
from kaggle_io import sha, safe_name
from exp26_common import json_hash

def verify_anchor(source, expected, check_environment=True):
    source = Path(source); path = source/safe_name(expected['file'])
    assert path.stat().st_size == expected['bytes'], 'L1 C bytes mismatch'
    assert sha(path) == expected['sha256'], 'L1 C SHA mismatch'
    p = L.read_full(path, expected['identity'])
    s = p['state']
    for key in ('updates', 'cursor', 'main_seconds'):
        assert s[key] == expected[key], 'L1 C '+key+' mismatch'
    assert p['model_spec'] == L.SPECS['C'] and p['identity']['model'] == 'C'
    assert s['last_lr'] == L.PEAK_LR['C'] and s['updates'] > 1000
    assert p['scaler_enabled'] and p['rng']['torch_cuda'], 'Missing real AMP/CUDA RNG state'
    recipe = p['identity']['recipe']
    for name, digest in recipe['implementation_sha256'].items():
        assert sha(Path(L.__file__).with_name(name)) == digest, 'L1 implementation changed: '+name
    if check_environment:
        assert recipe['torch'] == str(torch.__version__) and recipe['CUDA'] == torch.version.cuda, 'Torch/CUDA resume mismatch'
    assert all(bool(torch.isfinite(v).all()) for v in p['model'].values() if v.is_floating_point())
    assert all(bool(torch.isfinite(v).all()) for a in p['optimizer']['state'].values()
               for v in a.values() if torch.is_tensor(v) and v.is_floating_point())
    return path, p

def branch_identity(parent, order_meta):
    identity = copy.deepcopy(parent); identity['model'] = 'C-long2x'
    identity['parent_recipe_sha256'] = parent['recipe_sha256']
    identity['order_sha256'] = order_meta['file']['sha256']
    identity['recipe']['data_policy'] = 'fixed bucketed weighted order; long rows twice, others once; circular'
    identity['recipe']['sampler'] = copy.deepcopy(order_meta)
    identity['recipe_sha256'] = json_hash(identity['recipe'])
    return identity

def initial_state(payload, name):
    state = copy.deepcopy(payload['state'])
    state['l2_origin'] = {'updates': state['updates'], 'cursor': state['cursor'], 'main_seconds': state['main_seconds']}
    if name == 'C-long2x': state['cursor'] = 0
    state['cursor_origin'] = state['cursor']-state['updates']*128
    state['l2_main_seconds'] = 0.; state['nonfinite_skips'] = 0
    state['l2_points'] = []; state['stability'] = {'total': 0, 'consecutive': 0, 'window': [], 'escalations': 0, 'history': []}
    return state

def mounted_expected(source, expected_user):
    """Read the mounted live receipt before any cache scan; validate the requested C, never D."""
    import gzip
    source = Path(source); manifest = json.loads((source/'recover-manifest.json').read_text())
    assert manifest['format'] == 'nlp-hw02-l1-live-v1'
    blob = next(r for r in manifest['files'] if r['file'] == expected_user['file'])
    latest = next(r for r in manifest['files'] if r['original_path'] == 'model-C/latest.json')
    raw = (source/latest['file']).read_bytes()
    assert sha(source/latest['file']) == latest['sha256'] and len(raw) == latest['bytes']
    receipt = json.loads(gzip.decompress(raw) if latest['gzip'] else raw)
    assert receipt['file'].replace('/','__') == expected_user['file'].removeprefix('model-C__')
    assert receipt['sha256'] == blob['sha256'] and receipt['bytes'] == blob['bytes']
    assert receipt['updates'] == expected_user['updates'] and receipt['cursor'] == expected_user['cursor']
    assert abs(receipt['main_seconds']-expected_user['main_seconds']) <= 5., 'Rounded user main_s mismatch'
    return dict(receipt, file=blob['file'])

def cpu_restore(payload, model, opt):
    """CPU audit only: preserve CUDA RNG/scaler in payload; never claim CUDA numerical identity."""
    import exp26_train as T
    model.load_state_dict(payload['model']); opt.load_state_dict(payload['optimizer'])
    # CUDA fused flag cannot run on CPU, but moments/steps/LR remain unchanged.
    for g in opt.param_groups: g['fused'] = False
    T.restore_rng(payload['rng'], torch.device('cpu')); model.train()
    return copy.deepcopy(payload['state'])
