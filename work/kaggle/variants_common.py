"""Frozen five variants and file identities; no network or credentials."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
SUBMISSION = WORK.parent / 'submission'
BUNDLE = ROOT / 'bundles/variants-2026-10-20'
STAGE = ROOT / 'staging/variants-2026-10-20'
KERNEL = 'friend1/nlp-hw02-variants'
SOURCES = ['friend1/nlp-hw02-b6/1', 'friend1/nlp-hw02-b7/1', 'friend1/nlp-hw02-kaggle-p1/3']
LEDGER = WORK / 'checks/final-variants-launch.json'


def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''): h.update(b)
    return h.hexdigest()

def models():
    c, p, w = read(ROOT/'frozen-v11-budget640.json')['models']
    cb7 = dict(c, name='C-long2x-B7', filename='c-long2x-b7-fp16.pt',
        sha256='bcb41c54d031b70523d357dd7d14dc24ec7d1393a6cb8714d8eacfe761cca32b',
        dataset='friend1/nlp-hw02-b7', source_ref=SOURCES[1])
    wb7 = dict(w, name='W-B7', filename='w-b7-fp16.pt',
        sha256='3378c368f601c7cd046d40de22cf5cc6f5081deff190819bb4f616f0eb2761ac',
        dataset='friend1/nlp-hw02-b7', source_ref=SOURCES[1])
    result = {m['name']: copy.deepcopy(m) for m in [c,p,w,cb7,wb7]}
    for m in result.values():
        if m['name'] == 'pilot-b':
            m['source_path'] = 'model-pilot-b/weights/long-truncation-fp16.pt'
            m['local_path'] = 'results/kaggle-runs/nlp-hw02-kaggle-p1/v9-pinned-3/' + m['source_path']
        else:
            gpu = '0' if m['name'].startswith('W') else '1'
            m['source_path'] = f'model-GPU{gpu}/weights/b6-final-fp16.pt'
            version = 'b7/1' if m['name'].endswith('B7') else 'b6/v10-pinned-1'
            m['local_path'] = 'results/kaggle-runs/nlp-hw02-' + version + '/' + m['source_path']
    return result

def variants():
    base = read(ROOT/'frozen-v11-budget640.json'); pool = models()
    orders = [('W-B7','C-long2x-B7','W-B6'), ('W-B7','C-long2x-B7','pilot-b'),
        ('C-long2x-B6','pilot-b','W-B7'), ('C-long2x-B6','pilot-b','W-B6'),
        ('C-long2x-B6','W-B6','C-long2x-B7')]
    result = []
    for n, names in enumerate(orders, 1):
        cfg = copy.deepcopy(base)
        cfg['models'] = [dict(pool[name], K1=1, K2=1 if i == 0 else 8) for i,name in enumerate(names)]
        for m in cfg['models']:
            m.pop('local_path')
        cfg['variant'] = f'v{n}'
        if n == 4:
            cfg['e2_primary'] = True
            for m,k2,k3,mixed in zip(cfg['models'], [2,8,8], [2,16,8], [False,True,True]):
                m.update(K2=k2, K3=k3, mixed_temperature=mixed)
        result.append(cfg)
    return result
