"""B7 reuses every embedded B6 module and executable cell; only config/metadata change."""
import argparse, ast, copy, hashlib, json, shutil
from build_l3 import ROOT, literals, write
from b7_budget import calculate, HARD, WORKER, WATCHDOG


def build(smoke=False):
    suffix = '-gpu-smoke' if smoke else ''
    source = ROOT/'notebooks'/f'nlp-hw02-b6{suffix}.ipynb'
    nb = json.loads(source.read_text())
    code = next(c for c in nb['cells'] if c['cell_type'] == 'code')
    values = literals(code['source']); cfg = copy.deepcopy(values['CONFIG'])
    for name, body in values['SOURCES'].items():
        path = ROOT/'src'/name if (ROOT/'src'/name).exists() else ROOT/name
        assert body == path.read_text(), ('B6 embedded source changed', name)
    cfg.update(session_kind='b7', comparison_protocol=dict(status='B7 vs B6: same B5 uncooled sources, q0, eta0, order and optimizer; only cooldown N doubled'))
    if not smoke:
        cfg.update(N_by_gpu=calculate()['N_by_gpu'], budget_seconds=WORKER,
            worker_deadline_seconds=WORKER, hard_shutdown_seconds=WATCHDOG, session_ceiling_seconds=HARD)
    # Keep B6 module names, checkpoint format, output filenames and all executable code.
    text = code['source']; a = text.index('CONFIG='); b = text.index('WORKING=Path', a)
    code['source'] = text[:a]+'CONFIG='+repr(cfg)+'\n'+text[b:]
    nb['cells'][0]['source'] = '# B7: q0 cooldown N x2 from the same B5 uncooled states\n\n'+(
        'Reviewer-only smoke: N200 each, hard720s.' if smoke else 'W N33000 / C-long2x N65800; full evaluation, hard40800s (11h20min).')+\
        ' Private friend1, T4x2. Frozen B6 runtime and B6 output filenames retained.'
    for i, c in enumerate(nb['cells']):
        c['id'] = f'b7-{i:02d}'
        if c['cell_type'] == 'code':
            compile(c['source'], 'B7 notebook', 'exec'); c.update(outputs=[], execution_count=None)
    slug = 'nlp-hw02-b7'+suffix
    target = ROOT/'notebooks'/f'{slug}.ipynb'; write(target, nb)
    folder = ROOT/'bundles'/('b7_gpu_smoke_friend1' if smoke else 'b7_friend1')
    folder.mkdir(parents=True, exist_ok=True); shutil.copyfile(target, folder/target.name)
    pins = cfg['b6_pins']
    meta = dict(id='friend1/'+slug, title=slug, code_file=target.name, language='python',
        kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False, enable_internet=True,
        machine_shape='NvidiaTeslaT4', dataset_sources=[pins['starter_source']],
        kernel_sources=[pins['cache_source'], pins['source']], competition_sources=[])
    write(folder/'kernel-metadata.json', meta)
    write(folder/'session-limits.json', dict(session_timeout_seconds=cfg['session_ceiling_seconds'],
        budget_seconds=cfg['budget_seconds'], watchdog_exit_seconds=cfg['hard_shutdown_seconds']))
    overrides = next(ast.literal_eval(n.value) for n in ast.parse(text).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'C_OVERRIDES' for t in n.targets))
    identity = dict(modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in values['SOURCES'].items()},
        C_overrides={n:hashlib.sha256(s.encode()).hexdigest() for n,s in overrides.items()})
    identity['config_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    write(folder/'source-identity.json', identity)
    return target, cfg


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--gpu-smoke', action='store_true'); a = p.parse_args()
    print(build(a.gpu_smoke)[0])
