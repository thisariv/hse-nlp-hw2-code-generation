"""Assemble D1 production/smoke, borrowing FROZEN P1/v6 sources. No GPU or push."""
import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path
from build_l3 import ROOT, literals, write


def embedded(filename):
    nb = json.loads((ROOT/'notebooks'/filename).read_text())
    code = [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']
    return literals(code[0])['SOURCES'], code


def build(smoke=False):
    base, p1code = embedded('nlp-hw02-kaggle-p1.ipynb')
    final, _ = embedded('nlp-hw02-adaptive-inference-l3-v6.ipynb')
    available = {**base, **final}
    # V7's live final_* files are never read into D1; use immutable v6 notebook.
    for path in sorted((ROOT/'src').glob('d1_*.py')): available[path.name] = path.read_text()
    available['d1_worker.py'] = (ROOT/'d1_worker.py').read_text()
    names = {'d1_worker.py', 'd1_session.py', 'd1_live.py', 'l2_live.py',
             'final_execute_v6.py', 'final_execute.py', 'final_numeric.py'}
    while True:
        old = names.copy()
        for name in old:
            tree = ast.parse(available[name])
            for node in ast.walk(tree):
                modules = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                           [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                names.update(m.split('.')[0]+'.py' for m in modules if m.split('.')[0]+'.py' in available)
        if names == old: break
    sources = {n: available[n] for n in sorted(names)}
    seed = json.loads((ROOT/'d1-seed.json').read_text())
    cfg = dict(owner='thisarivson' if smoke else 'friend1', session_kind='d1', gpu_smoke=smoke,
        working_root='/kaggle/working', input_root='/kaggle/input', live_dataset='friend1/nlp-hw02-d1-live',
        weights_source=('thisarivson' if smoke else 'friend1')+'/nlp-hw02-l3-live', require_secrets=False,
        enable_live_sync=False,
        cache_source='thisarivson/nlp-hw02-compact-cache-verified/2',
        cache_fallback_source='friend1/nlp-hw02-compact-cache-verified', seed_sha256=seed['sha256'],
        setup_cap_seconds=300, loss_stop_seconds=4680, generation_stop_seconds=7200,
        execution_stop_seconds=7440, worker_deadline_seconds=7560, hard_shutdown_seconds=7800,
        budget_seconds=7560, session_ceiling_seconds=8400, terminate_grace_seconds=20,
        loss_batch=256, loss_token_slots=32768, loss_block=8192, fraction=.04,
        pilot_per_bin=256, timing_safety=1.6, pilot_tail_reserve=240., wave_rows=4096,
        generation_batch=512, cap=512, wave_timeout_seconds=600, smoke_rows=131072,
        port_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest())
    if smoke:
        cfg.update(setup_cap_seconds=240, loss_stop_seconds=420, generation_stop_seconds=580,
            execution_stop_seconds=610, worker_deadline_seconds=630, hard_shutdown_seconds=660,
            budget_seconds=630, session_ceiling_seconds=720, pilot_per_bin=256, wave_rows=4096,
            loss_block=8192, pilot_tail_reserve=20., generation_batch=512, wave_timeout_seconds=120, loss_batch=512)
    startup = '''import time,json,os,sys,tempfile
from pathlib import Path
SESSION_START=time.monotonic()
'''+ 'CONFIG='+repr(cfg)+'''
WORKING=Path(CONFIG['working_root']);WORKING.mkdir(parents=True,exist_ok=True)
CONFIG['session_start_monotonic']=SESSION_START
CONFIG['deadline_monotonic']=SESSION_START+CONFIG['worker_deadline_seconds']
CONFIG['hard_deadline_monotonic']=SESSION_START+CONFIG['hard_shutdown_seconds']
CONFIG['temp_root']=tempfile.mkdtemp(prefix='nlp-d1-',dir='/kaggle/temp' if Path('/kaggle/temp').is_dir() else None)
MODULE_DIR=WORKING/'src';MODULE_DIR.mkdir(exist_ok=True)
'''+ 'SOURCES='+repr(sources)+'''
for name,source in SOURCES.items():(MODULE_DIR/name).write_text(source)
sys.path.insert(0,str(MODULE_DIR));os.environ['TOKENIZERS_PARALLELISM']='false'
from d1_session import Watchdog
from kaggle_io import write
WATCHDOG=Watchdog(CONFIG).start()
import hashlib
write(WORKING/'d1-protocol.json',{'config':{k:v for k,v in CONFIG.items() if not k.endswith('_monotonic')},
      'modules':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in SOURCES.items()},'training':False})
import torch
from l2_live import gate
GATE=gate(torch,require_secrets=CONFIG['require_secrets']);write(WORKING/'startup-gate.json',GATE)
if not GATE['passed']:WATCHDOG.finish();raise RuntimeError(GATE['reason'])
if CONFIG['require_secrets'] and os.environ.get('KAGGLE_USERNAME')!=CONFIG['owner']:
    WATCHDOG.finish();raise RuntimeError('D1 requires friend1 Secrets')
'''
    setup = '''from d1_session import setup,launch
from d1_live import D1LiveSync
setup(CONFIG)
LIVE=None
if CONFIG['enable_live_sync']:
    LIVE=D1LiveSync(WORKING,CONFIG['temp_root'],CONFIG['live_dataset'],interval=1800.)
    assert LIVE.sync('startup-probe',timeout=300.,wait_ready=True),'Private D1 live probe failed'
    WATCHDOG.live_sync=LIVE;LIVE.start()
assert time.monotonic()-SESSION_START<CONFIG['setup_cap_seconds'],'D1 setup including live probe exceeded budget'
report=launch(CONFIG,MODULE_DIR/'d1_worker.py',WATCHDOG,LIVE)
print(json.dumps(report,indent=2))
# Partial is a valid saved result. The review gate inspects completeness explicitly.
assert not report['unreaped_worker_pids']
'''
    cells = []
    def cell(kind, text):
        c = dict(id=f'd1-{len(cells):02d}', cell_type=kind, metadata={}, source=text)
        if kind == 'code': compile(text, c['id'], 'exec'); c.update(outputs=[], execution_count=None)
        cells.append(c)
    cell('markdown', '# D1: диагностика train-reference\n\nCE: обе GPU C-long2x, непересекающиеся строки. '
         'Greedy: GPU0 C-main / GPU1 C-long2x L3 l2-62280. '
         'Только inference; полный CE long → stratified top≤4% → v6 greedy + CPU execution. '
         'Smoke:131072 CE-строк, все фазы,≤12мин. Production: hard130мин/backend140мин; Secrets/live optional. '
         'Согласие моделей — подозрение, не новая разметка. GPU-smoke/push выполняет ревьюер.')
    cell('code', startup); cell('code', p1code[1]); cell('code', setup)
    slug = 'nlp-hw02-d1-gpu-smoke' if smoke else 'nlp-hw02-kaggle-d1'
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')), cells=cells)
    target = ROOT/'notebooks'/f'{slug}.ipynb'; write(target, notebook)
    bundle = ROOT/'bundles'/('d1_gpu_smoke' if smoke else 'd1'); bundle.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(target, bundle/target.name)
    datasets = [cfg['weights_source'], 'thisarivson/nlp-hw02-26b-inputs']
    if not smoke: datasets += [cfg['cache_fallback_source'], cfg['live_dataset']]
    write(bundle/'kernel-metadata.json', dict(id=cfg['owner']+'/'+slug, title=slug.replace('-', ' '), code_file=target.name,
        language='python', kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False,
        enable_internet=True, machine_shape='NvidiaTeslaT4', dataset_sources=datasets,
        kernel_sources=[cfg['cache_source']], competition_sources=[]))
    write(bundle/'session-limits.json', dict(session_timeout_seconds=cfg['session_ceiling_seconds'],
        watchdog_exit_seconds=cfg['hard_shutdown_seconds'], worker_seconds=cfg['worker_deadline_seconds'], GPU_smoke=smoke))
    write(bundle/'source-identity.json', dict(config=cfg, notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        modules={n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()},
        frozen_v6_notebook_sha256=hashlib.sha256((ROOT/'notebooks/nlp-hw02-adaptive-inference-l3-v6.ipynb').read_bytes()).hexdigest()))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--gpu-smoke', action='store_true')
    print(build(parser.parse_args().gpu_smoke))
