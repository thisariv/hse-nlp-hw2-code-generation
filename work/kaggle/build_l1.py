#!/usr/bin/env python3
"""Build CPU compact cache and UI-started L1. Build never launches a kernel."""
import argparse, hashlib, json, re, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import write
from build import INSTALL

def build(kind,cache_source='thisarivson/nlp-hw02-compact-cache-verified/2',cache_sha=None,resume_source=None,resume_live=False,slug_override=None):
    sources={p.name:p.read_text() for p in sorted((ROOT/'src').glob('*.py'))}
    sources['l1_worker.py']=(ROOT/'l1_worker.py').read_text()
    frozen=json.loads((ROOT/'source-identity.json').read_text())
    assert all(hashlib.sha256(sources[n].encode()).hexdigest()==h for n,h in frozen['modules'].items())
    code_sha=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    cpu=kind.startswith('compact');repair=kind=='compact_repair';gpu_smoke=kind=='l1_gpu_smoke'
    slug='nlp-hw02-compact-cache-verified' if repair else 'nlp-hw02-compact-cpu-cache' if cpu else 'nlp-hw02-kaggle-l1'
    if slug_override:
        assert not cpu and re.fullmatch(r'[a-z0-9-]+',slug_override)
        slug=slug_override
    cfg={'input_root':'/kaggle/input','working_root':'/kaggle/working','session_kind':kind,
         'budget_seconds':9000 if cpu else 40680,'worker_deadline_seconds':9000 if cpu else 40680,
         'hard_shutdown_seconds':9300 if cpu else 42300,'session_ceiling_seconds':9600 if cpu else 43200,
         'terminate_grace_seconds':20,'live_dataset':'thisarivson/nlp-hw02-l1-live',
         'cache_source':cache_source,'cache_manifest_sha256':cache_sha,'port_sha256':code_sha,
         'resume_source':resume_source,'resume_live':resume_live}
    if repair:cfg.update(budget_seconds=1800,worker_deadline_seconds=1800,hard_shutdown_seconds=2100,session_ceiling_seconds=2400,
                         recovery_dataset='nlp-hw02-compact-targets-input')
    if gpu_smoke:
        slug=slug_override or 'nlp-hw02-l1-gpu-smoke'
        cfg.update(gpu_smoke=True,budget_seconds=900,worker_deadline_seconds=900,hard_shutdown_seconds=1020,
                   session_ceiling_seconds=1080,smoke_main_updates=256,smoke_resume_updates=32,
                   gpu_smoke_recipe={'max_updates':256,'point_updates':192,'max_points':1,
                                     'cooldown_cap':180.,'evaluation_seconds':240.})
    cells=[]
    def cell(kind,source):
        c={'id':f'l1-{len(cells):02d}','cell_type':kind,'metadata':{},'source':source}
        if kind=='code':c.update(execution_count=None,outputs=[])
        cells.append(c)
    cell('markdown',('# CPU compact cache — все finite26a+26c\n\nGPU отключён; полный SHA и≥50k CPU answer round-trip.' if cpu else
        '# L1 — C54M / D104M, T4×2, продолжение WSD\n\n**Запуск только Edit → Save Version → Save & Run All.** Attach Secrets KAGGLE_USERNAME/KAGGLE_KEY; Accelerator GPU T4×2; Internet on. После отправки background version остановить интерактивную сессию редактора. Гейт и пробный private live upload обязательны.\n\n'+(ROOT.parent/'notes/l1-long-training.md').read_text().split('---',2)[-1].lstrip()))
    startup='''import time,json,os,sys,tempfile
from pathlib import Path
SESSION_START=time.monotonic()
'''+ 'CONFIG='+repr(cfg)+'''
WORKING=Path(CONFIG['working_root']);WORKING.mkdir(parents=True,exist_ok=True)
CONFIG['session_start_monotonic']=SESSION_START
CONFIG['deadline_monotonic']=SESSION_START+CONFIG['worker_deadline_seconds']
CONFIG['hard_deadline_monotonic']=SESSION_START+CONFIG['hard_shutdown_seconds']
scratch_base=Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path(tempfile.gettempdir())
CONFIG['temp_root']=tempfile.mkdtemp(prefix='nlp-l1-',dir=scratch_base)
MODULE_DIR=WORKING/'src';MODULE_DIR.mkdir(exist_ok=True)
'''
    startup+='SOURCES='+repr(sources)+"\nfor name,source in SOURCES.items():(MODULE_DIR/name).write_text(source)\nsys.path.insert(0,str(MODULE_DIR))\nos.environ['TOKENIZERS_PARALLELISM']='false'\n"
    startup+='''from l1_session import Watchdog
WATCHDOG=Watchdog(CONFIG).start()
print('Watchdog armed before gates/setup; hard exit',CONFIG['hard_shutdown_seconds'],'s',flush=True)
'''
    if not cpu:
        startup+='''import torch
from l1_live import gate
from kaggle_io import write
GATE=gate(torch,require_secrets=not CONFIG.get('gpu_smoke',False));write(WORKING/'startup-gate.json',GATE)
print('L1 STARTUP GATE:',GATE['reason'],flush=True)
if not GATE['passed']:
    WATCHDOG.finish()
    print('L1 EXIT: no training, no installs, no data preparation',flush=True)
    os._exit(0)
'''
    cell('code',startup)
    # GPU gate runs before dependency installs. CLI then precedes the tiny live probe.
    cell('code',INSTALL.replace("'sympy':'1.14.0'","'sympy':'1.14.0','kaggle':'2.2.4'") if not cpu else INSTALL)
    if not cpu and not gpu_smoke:
        cell('code','''from l1_live import L1LiveSync
LIVE=L1LiveSync(WORKING,CONFIG['temp_root'],CONFIG['live_dataset'])
write(WORKING/'boot-manifest.json',{'session':'L1','purpose':'startup private live insurance probe','training_started':False})
if not LIVE.sync('startup-probe',timeout=300.,wait_ready=True):
    write(WORKING/'startup-live-gate.json',{'passed':False,'reason':'Private live probe failed; training refused'})
    print('L1 EXIT: private live probe failed; no training',flush=True)
    WATCHDOG.finish();os._exit(0)
write(WORKING/'startup-live-gate.json',{'passed':True,'private_live_dataset':CONFIG['live_dataset']})
print('L1 LIVE GATE PASSED: private startup probe uploaded',flush=True)
WATCHDOG.live_sync=LIVE
LIVE.start()
''')
    if gpu_smoke:cell('code',"LIVE=None\nprint('GPU smoke: Secrets/live upload not required; full C/D/AMP/fused recipe',flush=True)\n")
    if not cpu:
        cell('code','''from kaggle_io import resolve_dataset,materialize_dataset,sha
from kaggle_runtime import prepare_starter
from l1_cache import CompactCache,ROWS
from l1_live import restore_snapshot
inp=Path(CONFIG['input_root']);scratch=Path(CONFIG['temp_root'])
CONFIG['starter_dataset']=str(prepare_starter(resolve_dataset(inp,'nlp-hw02-26b-inputs'),scratch/'starter','training-inputs',read_only=True))
cache=materialize_dataset(resolve_dataset(inp,CONFIG['cache_source'].split('/')[1]),scratch/'compact','compact-manifest.json')
assert sha(cache/'compact-manifest.json')==CONFIG['cache_manifest_sha256']
summary=json.loads((cache/'cache-summary.json').read_text())
assert summary['rows']==ROWS and summary['roundtrip_rows']>=50000 and summary['answer_mismatches']==0
validated=CompactCache(cache,verify=True);assert len(validated)==ROWS;del validated
CONFIG['compact_cache']=str(cache)
if CONFIG['resume_source'] or CONFIG['resume_live']:
    candidates=[]
    if CONFIG['resume_source']:candidates.append(resolve_dataset(inp,CONFIG['resume_source'].split('/')[1]))
    if CONFIG['resume_live']:
        live_source=resolve_dataset(inp,'nlp-hw02-l1-live')
        candidates.append(restore_snapshot(live_source,scratch/'live-resume'))
    from l1_resume import select_resume
    recovered=select_resume(candidates,scratch/'selected-resume')
    CONFIG['resume_root']=str(recovered)
from l1_session import launch
report=launch(CONFIG,MODULE_DIR/'l1_worker.py',WATCHDOG,LIVE)
print(json.dumps(report,indent=2))
''')
    else:
        cpu_cell='''from kaggle_io import write
import subprocess,signal
write(WORKING/'cache-config.json',CONFIG)
child=subprocess.Popen([sys.executable,str(MODULE_DIR/'l1_cache.py'),'--config',str(WORKING/'cache-config.json')],start_new_session=True)
WATCHDOG.register(child)
while child.poll() is None and time.monotonic()<CONFIG['deadline_monotonic']:
    time.sleep(10)
if child.poll() is None:
    WATCHDOG.terminate();time.sleep(20);WATCHDOG.kill()
    raise RuntimeError('CPU cache budget exhausted')
assert child.returncode==0,'Compact CPU cache failed'
print((WORKING/'compact-cache/cache-summary.json').read_text(),flush=True)
WATCHDOG.finish()
'''
        if repair:cpu_cell=cpu_cell.replace("MODULE_DIR/'l1_cache.py'","MODULE_DIR/'l1_cache_repair.py'")
        cell('code',cpu_cell)
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},
         'language_info':{'name':'python'}},'cells':cells}
    target=ROOT/'notebooks'/f'{slug}.ipynb';write(target,nb)
    folder=ROOT/'bundles'/kind;folder.mkdir(parents=True,exist_ok=True);shutil.copyfile(target,folder/target.name)
    datasets=['thisarivson/nlp-hw02-26b-inputs']
    kernels=[cache_source] if not cpu else ['thisarivson/nlp-hw02-26c-cpu-cache/2']
    if cpu:datasets.append('thisarivson/nlp-hw02-26-token-cache')
    if repair:datasets=['thisarivson/nlp-hw02-compact-targets-input','thisarivson/nlp-hw02-26-token-cache'];kernels=['thisarivson/nlp-hw02-26c-cpu-cache/2']
    if resume_source:kernels.append(resume_source)
    if resume_live:datasets.append('thisarivson/nlp-hw02-l1-live')
    metadata={'id':'thisarivson/'+slug,'title':slug.replace('-',' '),'code_file':target.name,'language':'python',
              'kernel_type':'notebook','is_private':True,'enable_gpu':not cpu,'enable_tpu':False,'enable_internet':True,
              'dataset_sources':datasets,'kernel_sources':kernels,'competition_sources':[]}
    if not cpu:metadata['machine_shape']='NvidiaTeslaT4'
    write(folder/'kernel-metadata.json',metadata)
    write(folder/'session-limits.json',{'session_timeout_seconds':cfg['session_ceiling_seconds'],
          'kind':kind,'budget_seconds':cfg['budget_seconds'],'watchdog_exit_seconds':cfg['hard_shutdown_seconds']})
    print(folder);return target,cfg

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['compact','compact_repair','l1','l1_gpu_smoke'],required=True)
    p.add_argument('--cache-source',default='thisarivson/nlp-hw02-compact-cache-verified/2');p.add_argument('--cache-sha')
    p.add_argument('--resume-source');p.add_argument('--resume-live',action='store_true');p.add_argument('--slug');a=p.parse_args()
    if a.kind in ('l1','l1_gpu_smoke') and not a.cache_sha:p.error('L1 requires the verified CPU output manifest SHA')
    build(a.kind,a.cache_source,a.cache_sha,a.resume_source,a.resume_live,a.slug)
