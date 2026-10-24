#!/usr/bin/env python3
"""Build self-contained Kaggle commit notebooks; building never pushes/runs anything."""
import argparse,hashlib,json,pathlib,re,shutil,sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import write,sha,place

INSTALL="""import importlib.metadata,subprocess,sys
required={'tokenizers':'0.23.2','pyarrow':'23.0.1','sympy':'1.14.0'}
missing=[]
for package,version in required.items():
    try:installed=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:installed=None
    if installed!=version:missing.append(package+'=='+version)
if missing:
    installer=subprocess.Popen([sys.executable,'-m','pip','install','--quiet',*missing],start_new_session=True)
    if 'WATCHDOG' in globals():WATCHDOG.register(installer)
    try:
        if installer.wait(timeout=180)!=0:raise RuntimeError('Dependency install failed')
    except subprocess.TimeoutExpired:
        import os,signal
        os.killpg(installer.pid,signal.SIGKILL)
        raise
print({p:importlib.metadata.version(p) for p in required})
"""

def notebook(kind,config):
    sources={p.name:p.read_text() for p in sorted((ROOT/'src').glob('*.py'))}
    sources['s1_worker.py']=(ROOT/'s1_worker.py').read_text()
    sources['s2_worker.py']=(ROOT/'s2_worker.py').read_text()
    original=json.loads((ROOT/'source-identity.json').read_text())
    for n,h in original['modules'].items():assert hashlib.sha256(sources[n].encode()).hexdigest()==h
    port_sha=hashlib.sha256(json.dumps(sources,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    config.update(port_sha=port_sha,source_sha=original['code_sha256'])
    cells=[]
    def cell(t,s):
        obj={'id':f'{kind}-{len(cells):02d}','cell_type':t,'metadata':{},'source':s}
        if t=='code':obj.update(outputs=[],execution_count=None)
        cells.append(obj)
    cell('markdown', '# '+('S1 — №26b final / №27 / №28, T4×2' if kind=='s1' else '№26c — CPU extension42–59')+'\n\nPrivate; commit / Save & Run All. Без интерактивных ячеек. Fresh только исключается.\n')
    if kind=='s2':cell('markdown',(ROOT.parent/'notes/s2-fast-sampling-and-capacity.md').read_text().split('---',2)[-1].lstrip())
    startup="import time,json,os,sys,tempfile\nfrom pathlib import Path\nSESSION_START=time.monotonic()\n"+'CONFIG='+repr(config)+'\nWORKING=Path(CONFIG["working_root"]);WORKING.mkdir(parents=True,exist_ok=True)\n'
    if kind=='s2':
        startup+="""CONFIG['session_start_monotonic']=SESSION_START
CONFIG['deadline_monotonic']=SESSION_START+CONFIG['budget_seconds']-300
CONFIG['hard_deadline_monotonic']=min(CONFIG['deadline_monotonic']+600,SESSION_START+3*3600+40*60)
scratch_base=Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path(tempfile.gettempdir())
CONFIG['temp_root']=tempfile.mkdtemp(prefix='nlp-s2-',dir=scratch_base)
MODULE_DIR=WORKING/'src';MODULE_DIR.mkdir(exist_ok=True)
"""
        startup+="(MODULE_DIR/'session_watchdog.py').write_text("+repr(sources['session_watchdog.py'])+")\nsys.path.insert(0,str(MODULE_DIR))\nfrom session_watchdog import Watchdog\nWATCHDOG=Watchdog(CONFIG).start()\nprint('Watchdog armed: workers stop at3:10; emergency notebook exit at3:20',flush=True)\n"
    cell('code',startup)
    cell('code',INSTALL.replace("'sympy':'1.14.0'","'sympy':'1.14.0','kaggle':'2.2.4'") if kind=='s2' else INSTALL)
    cell('code','SOURCES='+repr(sources)+"\nMODULE_DIR=WORKING/'src';MODULE_DIR.mkdir(exist_ok=True)\nfor name,source in SOURCES.items():(MODULE_DIR/name).write_text(source)\nsys.path.insert(0,str(MODULE_DIR))\nos.environ['TOKENIZERS_PARALLELISM']='false'\n")
    if kind=='s2':
        cell('code',"""from s2_live_sync import LiveSync,load_secrets
LIVE=None
if load_secrets():
    LIVE=LiveSync(WORKING,CONFIG['temp_root'],CONFIG['live_dataset']).start()
    WATCHDOG.live_sync=LIVE
    print('LIVE: private snapshots armed; credentials stay in environment',flush=True)
""")
        cell('code',"""import torch
assert torch.cuda.device_count()==2, 'S2 requires exactly two GPUs'
assert all('T4' in torch.cuda.get_device_name(i) for i in (0,1)), 'Wrong accelerator'
print('GPUs:',[torch.cuda.get_device_name(i) for i in (0,1)])
assert CONFIG['budget_seconds']==11700
from kaggle_io import materialize_dataset,resolve_dataset
from kaggle_runtime import prepare_starter
inp=Path(CONFIG['input_root']);scratch=Path(CONFIG['temp_root'])
# Expanded mounts stay read-only; archive fallback and anchors live outside working.
CONFIG['starter_dataset']=str(prepare_starter(resolve_dataset(inp,CONFIG['starter_dataset']),scratch/'inputs','training-inputs',read_only=True))
CONFIG['cache_dataset']=str(materialize_dataset(resolve_dataset(inp,CONFIG['cache_dataset']),scratch/'unpacked-cache','cache-manifest.json'))
CONFIG['resume_dataset']=str(materialize_dataset(resolve_dataset(inp,CONFIG['resume_dataset']),scratch/'unpacked-resume','latest.json'))
from session_runner import launch
report=launch(CONFIG,MODULE_DIR/'s2_worker.py',watchdog=WATCHDOG,live_sync=LIVE)
print(json.dumps(report,indent=2))
print('OUTPUT: individual files, no ZIP',WORKING)
""")
    elif kind=='s1':
        cell('code',"""import torch
assert torch.cuda.device_count()==2, 'S1 requires exactly two GPUs'
assert all('T4' in torch.cuda.get_device_name(i) for i in (0,1)), 'Wrong accelerator'
print('GPUs:',[torch.cuda.get_device_name(i) for i in (0,1)])
assert 0<CONFIG['budget_seconds']<=12*3600-600
CONFIG['session_start_monotonic']=SESSION_START
CONFIG['deadline_monotonic']=min(SESSION_START+CONFIG['budget_seconds']-300,SESSION_START+12*3600-600)
# Materialize archive wrappers once, before processes race; token files then read through memmap.
from kaggle_io import materialize_dataset,resolve_dataset
from kaggle_runtime import prepare_starter
inp=Path(CONFIG['input_root'])
CONFIG['starter_dataset']=str(prepare_starter(resolve_dataset(inp,CONFIG['starter_dataset']),WORKING/'inputs','training-inputs'))
CONFIG['cache_dataset']=str(materialize_dataset(resolve_dataset(inp,CONFIG['cache_dataset']),WORKING/'unpacked-cache','cache-manifest.json'))
CONFIG['resume_dataset']=str(materialize_dataset(resolve_dataset(inp,CONFIG['resume_dataset']),WORKING/'unpacked-resume','latest.json'))
from session_runner import launch
report=launch(CONFIG,MODULE_DIR/('s2_worker.py' if CONFIG.get('session_kind')=='s2' else 's1_worker.py'))
print(json.dumps(report,indent=2))
print('OUTPUT:',WORKING/(CONFIG.get('session_kind','s1')+'-results.zip'))
""")
    else:
        cell('code',"""from kaggle_io import materialize_dataset,resolve_dataset,find_one,disk_guard
from exp26_input import import_starter
from cache_extension import run
from exp26_common import write_json,sha_file
import zipfile
# CPU only: this cell and its spawned workers never import Torch.
inputs_root=resolve_dataset(CONFIG['input_root'],CONFIG['cpu_dataset'])
print('Resolved CPU input',inputs_root,flush=True)
from kaggle_runtime import prepare_starter
inputs=prepare_starter(inputs_root,WORKING/'inputs','cache-inputs')
manifest=run(inputs,WORKING/'26c-token-cache',CONFIG['source_sha'],workers=2)
disk_guard(WORKING)
with zipfile.ZipFile(WORKING/'26c-metadata.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted((WORKING/'26c-token-cache').rglob('*.json')):z.write(p,p.relative_to(WORKING))
write_json(WORKING/'26c-output.json',{'experiment':'26c','GPU':False,'status':'complete','rows':manifest['rows'],
    'manifest_sha256':sha_file(WORKING/'26c-token-cache/cache-manifest.json'),'session_seconds':time.monotonic()-SESSION_START,
    'cache_output':'26c-token-cache; independent42–59 extension, base0–41 untouched'})
print(json.dumps(manifest['status']));print('CPU output',WORKING/'26c-output.json')
""")
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},
                'language_info':{'name':'python'}},'cells':cells}

def build(kind,owner,budget_hours=3,previous_source=None,slug=None,batch_rows=512,resume_source=None):
    if not 0<budget_hours*3600<=12*3600-600:raise ValueError('Budget must leave at least10min before12h')
    title='nlp hw02 kaggle '+kind if kind in ('s1','s2') else 'nlp hw02 26c cpu cache'
    slug=slug or title.replace(' ','-')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*',slug):raise ValueError('Invalid kernel slug')
    config={'input_root':'/kaggle/input','working_root':'/kaggle/working','sample_batch_rows':batch_rows,
            'budget_seconds':int(budget_hours*3600),'cache_dataset':'nlp-hw02-26-token-cache','resume_dataset':'nlp-hw02-26-resume',
            'starter_dataset':'nlp-hw02-26b-inputs','cpu_dataset':'nlp-hw02-26c-inputs'}
    if previous_source:
        if kind not in ('s1','s2'):raise ValueError('Previous output is an S1 source')
        if not re.fullmatch(r'[\w-]+/[\w-]+/[1-9][0-9]*',previous_source):raise ValueError('Pin previous owner/slug/N')
        # CLI validates the attachment; read-only verification uses the vN compatibility shim.
        config['previous_output']=previous_source.split('/')[1]
    if kind=='s2':
        if not previous_source:raise ValueError('S2 requires pinned S1 output')
        if budget_hours!=3.25:raise ValueError('Registered S2 quota budget is3h15m')
        # Times come from the verified S1 table, not rounded literals.
        table=json.loads((ROOT.parent/'results/26b-final-summary.json').read_text())['table']
        config.update(session_kind='s2',terminate_grace_seconds=20,output_policy='individual files; no ZIP',
            hard_shutdown_seconds=12000,session_ceiling_seconds=13200,
            live_dataset=owner+'/nlp-hw02-s2-live',live_interval_seconds=900,live_staging_limit_bytes=450000000,
            targets=[{'A_point':p['point'],'seconds':p['train_seconds_total'],'actual_pairs':round(p['actual_M_pairs']*1e6)} for p in table])
        if resume_source:
            if not re.fullmatch(r'[\w-]+/[\w-]+/[1-9][0-9]*',resume_source):raise ValueError('Pin S2 resume owner/slug/N')
            if resume_source.split('/')[1]==slug:raise ValueError('Resume must use a new kernel slug')
            config['s2_resume_output']=resume_source.split('/')[1]
    # Preserve the registered S1 source when preparing a separately pinned continuation.
    filename=slug+'.ipynb' if previous_source else ('s1-t4x2-26b-27-28.ipynb' if kind=='s1' else '26c-cpu-cache-extension.ipynb')
    nb=notebook(kind,config)
    if previous_source:
        nb['cells'][0]['source']='# '+slug+(' — №27b → №29 M / №28 B, T4×2' if kind=='s2' else ' — продолжение №26b / №27 / №28, T4×2')+'\n\nPrivate; источник '+previous_source+'. Fresh только исключается.\n'
    target=ROOT/'notebooks'/filename
    write(target,nb)
    bundle=ROOT/'bundles'/kind;bundle.mkdir(parents=True,exist_ok=True);place_path=bundle/target.name
    shutil.copyfile(target,place_path)
    meta={'id':owner+'/'+slug,'title':slug.replace('-',' '),'code_file':target.name,'language':'python','kernel_type':'notebook',
          'is_private':True,'enable_gpu':kind in ('s1','s2'),'enable_tpu':False,'enable_internet':True,
          'machine_shape':'NvidiaTeslaT4' if kind in ('s1','s2') else '',
          'dataset_sources':[owner+'/nlp-hw02-26c-inputs'] if kind=='26c' else [owner+'/'+config[k] for k in ('cache_dataset','resume_dataset','starter_dataset')],
          'kernel_sources':([previous_source] if previous_source else [])+([resume_source] if resume_source else []),'competition_sources':[]}
    write(bundle/'kernel-metadata.json',meta)
    if kind=='s2':write(bundle/'session-limits.json',{'session_timeout_seconds':13200,'budget_seconds':11700,'watchdog_exit_seconds':12000})
    print(bundle)
    return bundle

def static_inputs():
    for kind,name in [('26c','26a-cache-inputs-tokenizer-and-exclusions.zip'),('26b','26b-training-inputs-d0-initial-and-evaluation.zip')]:
        folder=ROOT/'staging'/f'{kind}-inputs';folder.mkdir(parents=True,exist_ok=True)
        from exp26_protocol import STARTER_A_SHA,STARTER_B_SHA
        import zipfile
        source=ROOT.parent/'results/checkpoints'/name
        assert sha(source)==(STARTER_A_SHA if kind=='26c' else STARTER_B_SHA)
        with zipfile.ZipFile(source) as z:assert z.testzip() is None
        place(source,folder/name)
        # Archive verifies itself; no keys or local environment copied.
        print(kind,folder)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--kind',choices=['s1','s2','26c','static-inputs'],required=True)
    p.add_argument('--owner',default='thisarivson');p.add_argument('--budget-hours',type=float,default=3)
    p.add_argument('--previous-source');p.add_argument('--resume-source');p.add_argument('--slug');p.add_argument('--sample-batch-rows',type=int,choices=[512,1024,2048],default=512)
    a=p.parse_args()
    if a.kind=='static-inputs':static_inputs()
    else:build(a.kind,a.owner,a.budget_hours,a.previous_source,a.slug,a.sample_batch_rows,a.resume_source)
