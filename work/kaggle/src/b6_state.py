"""Full cooldown checkpoints. Original main state/identity stay immutable, LR lives in cooldown."""
import copy, json, uuid
from pathlib import Path
import torch
import exp26_core as C
import l1_runtime as L
import l2_runtime as R
import b5_state as S
from p1_math import sqrt_lr
from kaggle_io import sha, write, safe_name

MODULES = ('b6_state.py', 'b6_runtime.py', 'p1_math.py')


def new_cooldown(state, n, source_sha256):
    assert isinstance(n, int) and n > 0
    return dict(N=n, updates=0, cursor=state['cursor'], origin_cursor=state['cursor'],
        origin_updates=state['updates'], peak_lr=state['last_lr'], last_lr=state['last_lr'],
        micro_batch=state['micro_batch'], seconds=0., AMP_skips=0, nonfinite_skips=0,
        stability=copy.deepcopy(state['stability']), source_sha256=source_sha256,
        stream_sha256=__import__('hashlib').sha256().hexdigest(), schedule='1-sqrt; attempted updates j=1..N; endpoint LR0', q=0.)


def step(m, o, scaler, state, cd, cache, order, device, identity):
    """Exactly one attempted batch128. Skips consume cursor/schedule as in P1."""
    j = cd['updates']+1
    assert j <= cd['N'] and cd['cursor'] == cd['origin_cursor']+128*(j-1)
    lr = sqrt_lr(j, cd['N'], cd['peak_lr'])
    log, micro = S.update_for(identity, m, o, scaler, cache, L.take(order, cd['cursor'], 128),
                              device, cd['micro_batch'], lr)
    cd.update(updates=j, cursor=cd['cursor']+128, last_lr=lr, micro_batch=micro)
    cd['seconds'] += log['seconds']; cd['AMP_skips'] += log['AMP_skip']; cd['nonfinite_skips'] += log['nonfinite_skip']
    unsafe = R.observe(cd['stability'], log['nonfinite_skip'], amp_skip=log['AMP_skip'])
    return log, unsafe


def save_full(root, m, o, scaler, state, spec, identity, cd):
    root=Path(root); folder=root/'checkpoints'; folder.mkdir(parents=True,exist_ok=True)
    C.sync(next(m.parameters()).device)
    p=L.state_payload(m,o,scaler,state,spec,identity)
    p.update(format='nlp-hw02-b6-full-v1', cooldown=copy.deepcopy(cd), stability_policy=R.POLICY,
             runtime_sha256={n:sha(Path(__file__).with_name(n)) for n in MODULES})
    p['scheduler']=dict(kind=cd['schedule'],last_lr=cd['last_lr'],N=cd['N'],updates=cd['updates'])
    assert R.finite_payload(p), 'Refuse nonfinite cooldown checkpoint'
    path=folder/('cooldown-'+uuid.uuid4().hex[:12]+'.pt'); tmp=path.with_suffix('.pt.tmp')
    torch.save(p,tmp);tmp.replace(path);del p
    rec=dict(file=path.relative_to(root).as_posix(),sha256=sha(path),bytes=path.stat().st_size,
             cooldown_updates=cd['updates'],N=cd['N'],cursor=cd['cursor'],identity=identity,
             source_sha256=cd['source_sha256'],purpose='full B6 cooldown resume; not an uncooled main state')
    write(root/'latest.json',rec)
    for old in sorted(folder.glob('cooldown-*.pt'),key=lambda p:p.stat().st_mtime_ns,reverse=True)[2:]:old.unlink()
    return rec


def read_full(root, identity, environment=True):
    root=Path(root); rec=json.loads((root/'latest.json').read_text()); path=root/safe_name(rec['file'])
    assert path.stat().st_size==rec['bytes'] and sha(path)==rec['sha256']
    p=torch.load(path,map_location='cpu',weights_only=True)
    assert p['format']=='nlp-hw02-b6-full-v1' and p['identity']==identity==rec['identity']
    assert p['stability_policy']==R.POLICY and R.finite_payload(p)
    for n,h in dict(identity['recipe']['implementation_sha256'],**p['runtime_sha256']).items():
        assert sha(Path(__file__).with_name(n))==h,n
    if environment:
        assert str(torch.__version__)==identity['recipe']['torch'] and torch.version.cuda==identity['recipe']['CUDA']
    cd=p['cooldown']; state=p['state']
    assert 0<=cd['updates']<=cd['N'] and rec['N']==cd['N'] and rec['cooldown_updates']==cd['updates']
    assert cd['cursor']==state['cursor']+128*cd['updates']==rec['cursor']
    assert cd['origin_cursor']==state['cursor'] and cd['origin_updates']==state['updates']
    assert cd['last_lr']==sqrt_lr(cd['updates'],cd['N'],cd['peak_lr'])==p['scheduler']['last_lr']
    assert p['scheduler']['N']==cd['N'] and p['scheduler']['updates']==cd['updates']
    assert all(g['lr']==cd['last_lr'] for g in p['optimizer']['param_groups'])
    count=state['updates']-state['AMP_skips']-state['nonfinite_skips']+cd['updates']-cd['AMP_skips']-cd['nonfinite_skips']
    assert p['optimizer']['state'] and all(int(v['step'])==count for v in p['optimizer']['state'].values())
    assert cd['source_sha256']==rec['source_sha256']
    return p


def restore(p, device):
    m,o,scaler,state=S.restore(p,device)
    return m,o,scaler,state,copy.deepcopy(p['cooldown'])
