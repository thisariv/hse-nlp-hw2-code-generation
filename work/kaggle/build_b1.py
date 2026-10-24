#!/usr/bin/env python3
"""Assemble private B1 or reviewer smoke. No GPU execution or push."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from build_l3 import ROOT, literals, write
from build import INSTALL

CACHE_SHA = '46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be'
OWNERS = ('friend2', 'thisarivson', 'friend1')


def build(owner='friend2', dropout=0., smoke=False, resume_source=None, transfer_sha=None,
          final=False, final_q=.1, main_seconds=None,
          cache_source='thisarivson/nlp-hw02-compact-cache-verified/2'):
    assert owner in OWNERS and 0 <= dropout < 1 and 0 <= final_q < 1
    assert bool(resume_source) == bool(transfer_sha)
    if transfer_sha: assert len(transfer_sha) == 64 and all(c in '0123456789abcdef' for c in transfer_sha)
    main_seconds = main_seconds if main_seconds is not None else (28800. if final else 34200.)
    assert 0 < main_seconds <= 34200.
    old = json.loads((ROOT/'notebooks/nlp-hw02-kaggle-r1.ipynb').read_text())
    source_map = literals(next(c['source'] for c in old['cells'] if c['cell_type'] == 'code'))['SOURCES']
    # Refresh all inherited modules from disk, preserving reviewed code byte-for-byte.
    sources = {n:(ROOT/'src'/n if (ROOT/'src'/n).exists() else ROOT/n).read_text() for n in source_map}
    for p in (ROOT/'src').glob('b1_*.py'): sources[p.name] = p.read_text()
    for n in ('shape_bench.py', 'shape_specs.py'): sources[n] = (ROOT/'src'/n).read_text()
    sources['b1_worker.py'] = (ROOT/'b1_worker.py').read_text()
    cfg = dict(owner=owner, input_root='/kaggle/input', working_root='/kaggle/working', session_kind='b1',
               gpu_smoke=smoke, dropout=dropout, main_seconds=main_seconds,
               shape_bench_lengths=json.loads((ROOT/'shape-lengths.json').read_text()),
               final_session=final, final_q=final_q, probe_cooldown_seconds=900., probe_eval_seconds=600.,
               final_cooldown_seconds=3600., final_eval_seconds=2400., checkpoint_seconds=900.,
               io_reserve_seconds=240., setup_cap_seconds=480.,
               cache_source=cache_source, cache_manifest_sha256=CACHE_SHA,
               resume_source=resume_source, transfer_sha256=transfer_sha,
               budget_seconds=40260, worker_deadline_seconds=40260, hard_shutdown_seconds=40800,
               session_ceiling_seconds=40800, terminate_grace_seconds=20., live_sync_optional=True,
               live_sync_enabled=False, port_sha256=hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest())
    if smoke:
        assert not resume_source, 'Smoke is fresh and uses internal serialized continuation'
        cfg.update(budget_seconds=780, worker_deadline_seconds=780, hard_shutdown_seconds=840,
                   session_ceiling_seconds=900, io_reserve_seconds=30.)
    cells = []
    def cell(kind, src):
        c = dict(id=f'b1-{len(cells):02d}', cell_type=kind, metadata={}, source=src)
        if kind == 'code': compile(src, 'B1 notebook cell', 'exec'); c.update(outputs=[], execution_count=None)
        cells.append(c)
    cell('markdown', '# B1: E8-D4 / E6-D6, независимые T4 workers\n\n'
         + ('Короткий GPU smoke для независимого ревьюера. Полный 180с gate выполняется отдельно.\n' if smoke else
            'Fresh либо строгий full-state resume. Secrets не нужны: всё сохраняется в output.\n')
         + 'Параметры и команды: `notes/b1-plan.md`. Hard wall ≤11ч20мин от первой code cell.\n')
    cell('code', '''import time,json,os,sys,tempfile
from pathlib import Path
SESSION_START=time.monotonic()
'''+'CONFIG='+repr(cfg)+'''
WORKING=Path(CONFIG['working_root']);WORKING.mkdir(parents=True,exist_ok=True)
CONFIG['session_start_monotonic']=SESSION_START
CONFIG['deadline_monotonic']=SESSION_START+CONFIG['worker_deadline_seconds']
CONFIG['hard_deadline_monotonic']=SESSION_START+CONFIG['hard_shutdown_seconds']
CONFIG['temp_root']=tempfile.mkdtemp(prefix='nlp-b1-')
MODULE_DIR=WORKING/'src';MODULE_DIR.mkdir(exist_ok=True)
'''+'SOURCES='+repr(sources)+'''
for name,source in SOURCES.items():(MODULE_DIR/name).write_text(source)
sys.path.insert(0,str(MODULE_DIR))
os.environ['TOKENIZERS_PARALLELISM']='false'
from b1_session import Watchdog
WATCHDOG=Watchdog(CONFIG).start()
import torch
from r1_live import gate
from kaggle_io import write
GATE=gate(torch,require_secrets=False);write(WORKING/'startup-gate.json',GATE)
assert GATE['passed'],GATE['reason']
print('B1 T4x2 gate passed; local stable/output; live disabled',flush=True)
''')
    cell('code', INSTALL)
    cell('code', '''from kaggle_io import sha,materialize_dataset
from kaggle_runtime import prepare_starter
from l3_resume import resolve_cache_mount
from l1_cache import CompactCache,ROWS
from b1_data import build_order,verify_order
import shutil
inp=Path(CONFIG['input_root']);scratch=Path(CONFIG['temp_root'])
starter=resolve_cache_mount(inp,'thisarivson/nlp-hw02-26b-inputs','datasets')
CONFIG['starter_dataset']=str(prepare_starter(starter,scratch/'starter','training-inputs',read_only=True))
kind='notebooks' if CONFIG['cache_source'].endswith('/2') else 'datasets'
source=resolve_cache_mount(inp,CONFIG['cache_source'],kind)
cache=materialize_dataset(source,scratch/'compact','compact-manifest.json')
assert sha(cache/'compact-manifest.json')==CONFIG['cache_manifest_sha256']
summary=json.loads((cache/'cache-summary.json').read_text())
assert summary['rows']==ROWS and summary['roundtrip_rows']>=50000 and summary['answer_mismatches']==0
validated=CompactCache(cache,verify=True);assert len(validated)==ROWS
CONFIG['compact_cache']=str(cache)
if CONFIG['resume_source']:
    from b1_resume import materialize
    source=resolve_cache_mount(inp,CONFIG['resume_source'],'datasets')
    assert sha(source/'b1-transfer.json')==CONFIG['transfer_sha256'],'Transfer manifest SHA mismatch'
    recovered=materialize(source,scratch/'resume');CONFIG['resume_root']=str(recovered)
    (WORKING/'weighted-order').mkdir(exist_ok=True)
    for name in ('order.npy','order.json'):shutil.copyfile(recovered/'weighted-order'/name,WORKING/'weighted-order'/name)
    verify_order(WORKING/'weighted-order',validated)
else:
    build_order(validated,WORKING/'weighted-order')
write(WORKING/'input-preflight.json',dict(passed=True,starter_mount=str(starter),cache_mount=str(source),
      cache_manifest_sha256=CONFIG['cache_manifest_sha256'],cache_rows=ROWS,all_cache_SHA_verified=True))
del validated
assert time.monotonic()-SESSION_START<CONFIG['setup_cap_seconds'],'B1 setup exceeded reserve'
from b1_session import launch
report=launch(CONFIG,MODULE_DIR/'b1_worker.py',WATCHDOG)
print(json.dumps(report,indent=2))
assert report['status']=='complete' and report['exit_codes']==[0,0],'Inspect per-worker stable/output'
''')
    slug = 'nlp-hw02-b1-gpu-smoke' if smoke else 'nlp-hw02-kaggle-b1'
    nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=old['metadata'])
    target = ROOT/'notebooks'/f'{slug}.ipynb'; write(target, nb)
    folder = ROOT/'bundles'/('b1_gpu_smoke' if smoke else 'b1'); folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(target, folder/target.name)
    kernel_cache = cache_source.endswith('/2')
    datasets = ['thisarivson/nlp-hw02-26b-inputs']
    if not kernel_cache: datasets.append(cache_source)
    if resume_source: datasets.append(resume_source)
    write(folder/'kernel-metadata.json', dict(id=owner+'/'+slug, title=slug.replace('-', ' '), code_file=target.name,
          language='python', kernel_type='notebook', is_private=True, enable_gpu=True, enable_tpu=False,
          enable_internet=True, machine_shape='NvidiaTeslaT4', dataset_sources=datasets,
          kernel_sources=[cache_source] if kernel_cache else [], competition_sources=[]))
    write(folder/'session-limits.json', dict(session_timeout_seconds=cfg['session_ceiling_seconds'],
          budget_seconds=cfg['budget_seconds'], watchdog_exit_seconds=cfg['hard_shutdown_seconds']))
    write(folder/'source-identity.json', dict(modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}))
    return target, cfg


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--owner', choices=OWNERS, default=OWNERS[0])
    p.add_argument('--dropout', type=float, default=0.); p.add_argument('--gpu-smoke', action='store_true')
    p.add_argument('--resume-source'); p.add_argument('--transfer-sha'); p.add_argument('--final-session', action='store_true')
    p.add_argument('--final-q', type=float, default=.1); p.add_argument('--main-hours', type=float)
    p.add_argument('--cache-source', default='thisarivson/nlp-hw02-compact-cache-verified/2'); a=p.parse_args()
    print(build(a.owner, a.dropout, a.gpu_smoke, a.resume_source, a.transfer_sha, a.final_session, a.final_q,
                a.main_hours*3600 if a.main_hours is not None else None, a.cache_source)[0])
