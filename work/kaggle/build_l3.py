#!/usr/bin/env python3
"""Configure the L2 notebook/worker for two strict L2 continuations; no push/run."""
import argparse, ast, copy, hashlib, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def literals(source):
    return {s.targets[0].id:ast.literal_eval(s.value) for s in ast.parse(source).body
            if isinstance(s,ast.Assign) and isinstance(s.targets[0],ast.Name)
            and s.targets[0].id in ('CONFIG','SOURCES')}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def build(smoke=False,cache_source='thisarivson/nlp-hw02-compact-cache-verified/2',continuation='l3'):
    assert continuation in ('l3','l4')
    phase=continuation.upper()
    # Reuse L2's install/live/supervisor/evaluation cells; do not collect concurrent final_* files.
    nb=json.loads((ROOT/'notebooks/nlp-hw02-kaggle-l2.ipynb').read_text())
    codes=[c['source'] for c in nb['cells'] if c['cell_type']=='code'];old=literals(codes[0])
    sources={name:((ROOT/name) if name in ('l2_worker.py','l2_compare.py') else ROOT/'src'/name).read_text()
             for name in old['SOURCES'] if not name.startswith('final_')}
    sources['l3_resume.py']=(ROOT/'src/l3_resume.py').read_text()
    frozen=json.loads((ROOT/'source-identity.json').read_text())
    assert all(hashlib.sha256(sources[n].encode()).hexdigest()==h for n,h in frozen['modules'].items())
    pins=json.loads((ROOT/f'{continuation}-anchors.json').read_text())
    cfg=copy.deepcopy(old['CONFIG'])
    for key in ('anchor_user','anchor_pin','preparation_blocker'):cfg.pop(key,None)
    cfg.update(owner='friend1',session_kind='l3',gpu_smoke=smoke,live_dataset='friend1/nlp-hw02-l3-live',
               anchor_source=pins['source'],resume_anchors=pins['anchors'],weighted_order_pin=pins['weighted_order'],
               cache_source=cache_source,main_added_seconds=33480.,
               phase_recipe={'main_targets':[16740.,33480.],'cooldown_seconds':[900.,3000.],'eval_seconds':[300.,1800.]},
               vote_cap_seconds=900.,setup_cap_seconds=480.,budget_seconds=40260,worker_deadline_seconds=40260,
               hard_shutdown_seconds=40800,session_ceiling_seconds=43200,
               port_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest())
    if continuation=='l4':
        cfg.update(session_kind='l4',live_dataset='friend1/nlp-hw02-l4-live',inherited_points=5,
                   main_added_seconds=30600.,
                   phase_recipe={'main_targets':[15300.,30600.],'cooldown_seconds':[900.,3000.],'eval_seconds':[300.,1800.]},
                   budget_seconds=37380,worker_deadline_seconds=37380,hard_shutdown_seconds=37800)
    if smoke:
        cfg.update(budget_seconds=900,worker_deadline_seconds=900,hard_shutdown_seconds=1020,session_ceiling_seconds=1080,
                   phase_recipe={'main_targets':[150.,300.],'cooldown_seconds':[20.,20.],'eval_seconds':[120.,150.]},
                   vote_cap_seconds=60.)
        if cache_source.startswith('thisarivson/'):
            cfg['review_fallback_source']='friend1/nlp-hw02-compact-cache-verified'
    startup='''import time,json,os,sys,tempfile
from pathlib import Path
SESSION_START=time.monotonic()
'''+ 'CONFIG='+repr(cfg)+'''
WORKING=Path(CONFIG['working_root']);WORKING.mkdir(parents=True,exist_ok=True)
CONFIG['session_start_monotonic']=SESSION_START
CONFIG['deadline_monotonic']=SESSION_START+CONFIG['worker_deadline_seconds']
CONFIG['hard_deadline_monotonic']=SESSION_START+CONFIG['hard_shutdown_seconds']
CONFIG['temp_root']=tempfile.mkdtemp(prefix='nlp-l3-',dir='/kaggle/temp' if Path('/kaggle/temp').is_dir() else None)
MODULE_DIR=WORKING/'src';MODULE_DIR.mkdir(exist_ok=True)
'''+ 'SOURCES='+repr(sources)+'''
for name,source in SOURCES.items():(MODULE_DIR/name).write_text(source)
sys.path.insert(0,str(MODULE_DIR));os.environ['TOKENIZERS_PARALLELISM']='false'
from l2_session import Watchdog
from kaggle_io import write,resolve_dataset,sha
WATCHDOG=Watchdog(CONFIG).start()
import torch
from l2_live import gate
GATE=gate(torch,require_secrets=not CONFIG['gpu_smoke']);write(WORKING/'startup-gate.json',GATE)
if not GATE['passed']:WATCHDOG.finish();raise RuntimeError(GATE['reason'])
if not CONFIG['gpu_smoke'] and os.environ.get('KAGGLE_USERNAME')!=CONFIG['owner']:
    WATCHDOG.finish();raise RuntimeError('L3 requires friend1 Secrets')
# Both full SHA/counters/cursors/main_s, finite Adam/model, code/Torch/CUDA before install/cache.
from l3_resume import mounted_expected,verify_anchor
try:
    source=resolve_dataset(CONFIG['input_root'],CONFIG['anchor_source'])
    checks={}
    for name,pin in CONFIG['resume_anchors'].items():
        expected=mounted_expected(source,name,pin)
        _,checked=verify_anchor(source,expected,expected_points=CONFIG.get('inherited_points',3))
        checks[name]={k:checked['state'][k] for k in ('updates','cursor','main_seconds','l2_main_seconds')}
        checks[name]['sha256']=expected['sha256'];del checked
    write(WORKING/'anchor-preflight.json',{'passed':True,'anchors':checks,
          'preflight_wall_seconds':time.monotonic()-SESSION_START,'training_started':False})
    CONFIG['anchor_root']=str(source)
except Exception as error:
    write(WORKING/'anchor-preflight.json',{'passed':False,'error_type':type(error).__name__,'training_started':False})
    WATCHDOG.finish();raise
'''
    startup=startup.replace('nlp-l3-',f'nlp-{continuation}-').replace('L3 requires',f'{phase} requires')
    live=codes[2].replace("'session':'L2'",f"'session':'{phase}'").replace('Private L2 live probe',f'Private {phase} live probe')
    setup='''from kaggle_io import materialize_dataset
from kaggle_runtime import prepare_starter
from l1_cache import CompactCache,ROWS
from l3_resume import restore_order,resolve_cache_mount
scratch=Path(CONFIG['temp_root']);inp=Path(CONFIG['input_root'])
CONFIG['starter_dataset']=str(prepare_starter(resolve_dataset(inp,'nlp-hw02-26b-inputs'),scratch/'starter','training-inputs',read_only=True))
cache_kind='notebooks' if CONFIG['cache_source'].endswith('/2') else 'datasets'
cache_root=materialize_dataset(resolve_cache_mount(inp,CONFIG['cache_source'],cache_kind),scratch/'compact','compact-manifest.json')
assert sha(cache_root/'compact-manifest.json')==CONFIG['cache_manifest_sha256']
summary=json.loads((cache_root/'cache-summary.json').read_text())
assert summary['rows']==ROWS and summary['roundtrip_rows']>=50000 and summary['answer_mismatches']==0
cache=CompactCache(cache_root,verify=True);assert len(cache)==ROWS
assert sha(cache_root/'order.npy')==CONFIG['resume_anchors']['C-main']['identity']['order_sha256']
order_root=restore_order(CONFIG['anchor_root'],WORKING/'weighted-order',CONFIG['weighted_order_pin'])
CONFIG['weighted_order_root']=str(order_root);CONFIG['compact_cache']=str(cache_root)
write(WORKING/'input-preflight.json',{'passed':True,'account':CONFIG['owner'],
      'anchor_source':CONFIG['anchor_source'],'anchor_mount':str(source),
      'starter_mount':str(CONFIG['starter_dataset']),'cache_source':CONFIG['cache_source'],
      'cache_mount':str(cache_root),'cache_manifest_sha256':cache.manifest_sha,
      'cache_rows':len(cache),'cache_shards':len(cache.manifest['shards']),
      'all_shard_SHA_verified':True,'main_order_sha256':sha(cache_root/'order.npy'),
      'weighted_order_sha256':sha(order_root/'order.npy'),
      'setup_wall_seconds':time.monotonic()-SESSION_START,'training_started':False})
del cache
if CONFIG.get('review_fallback_source'):
    fallback_root=materialize_dataset(resolve_cache_mount(inp,CONFIG['review_fallback_source'],'datasets'),scratch/'fallback','compact-manifest.json')
    assert sha(fallback_root/'compact-manifest.json')==CONFIG['cache_manifest_sha256']
    fallback_summary=json.loads((fallback_root/'cache-summary.json').read_text())
    assert fallback_summary['rows']==ROWS and fallback_summary['roundtrip_rows']>=50000 and fallback_summary['answer_mismatches']==0
    fallback=CompactCache(fallback_root,verify=True);assert len(fallback)==ROWS
    assert sha(fallback_root/'order.npy')==CONFIG['resume_anchors']['C-main']['identity']['order_sha256']
    assert sha(fallback_root/'tokenizer.json')==fallback.manifest['tokenizer']['sha256']
    write(WORKING/'fallback-preflight.json',{'passed':True,'source':CONFIG['review_fallback_source'],
          'mount':str(fallback_root),'manifest_sha256':fallback.manifest_sha,'rows':len(fallback),
          'shards':len(fallback.manifest['shards']),'all_shard_SHA_verified':True,
          'order_sha256':sha(fallback_root/'order.npy'),'tokenizer_SHA_verified':True,
          'roundtrip_rows':fallback_summary['roundtrip_rows'],'answer_mismatches':0})
    del fallback
assert time.monotonic()-SESSION_START<CONFIG['setup_cap_seconds'],'Setup exceeded fixed budget; no unequal truncated experiment'
from l2_session import launch
report=launch(CONFIG,MODULE_DIR/'l2_worker.py',WATCHDOG,LIVE)
print(json.dumps(report,indent=2))
'''+"if CONFIG['gpu_smoke']:"+codes[3].split("if CONFIG['gpu_smoke']:",1)[1]
    cells=[]
    contents=[('markdown',f'# {phase} — продолжение обеих C, friend1, private T4×2\n\n'+
               (ROOT.parent/f'notes/{continuation}-plan.md').read_text().split('---',2)[-1].lstrip()),
              ('code',startup),('code',codes[1]),('code',live),('code',setup)]
    for kind,source in contents:
        c={'id':f'{continuation}-{len(cells):02d}','cell_type':kind,'metadata':{},'source':source}
        if kind=='code':c.update(outputs=[],execution_count=None);compile(source,'L3 cell','exec')
        cells.append(c)
    slug=f'nlp-hw02-{continuation}-gpu-smoke' if smoke else f'nlp-hw02-kaggle-{continuation}'
    nb['cells']=cells;target=ROOT/'notebooks'/f'{slug}.ipynb';write(target,nb)
    bundle=ROOT/'bundles'/(continuation+'_gpu_smoke' if smoke else continuation);bundle.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(target,bundle/target.name)
    kernel_cache=cache_source.startswith('thisarivson/') and cache_source.endswith('/2')
    datasets=['thisarivson/nlp-hw02-26b-inputs',pins['source']]
    if not kernel_cache:datasets.append(cache_source)
    if cfg.get('review_fallback_source'):datasets.append(cfg['review_fallback_source'])
    write(bundle/'kernel-metadata.json',{'id':'friend1/'+slug,'title':slug.replace('-',' '),'code_file':target.name,
          'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,
          'dataset_sources':datasets,'kernel_sources':[cache_source] if kernel_cache else [],
          'competition_sources':[],'machine_shape':'NvidiaTeslaT4'})
    write(bundle/'session-limits.json',{'session_timeout_seconds':cfg['session_ceiling_seconds'],'kind':continuation+'_gpu_smoke' if smoke else continuation,
          'budget_seconds':cfg['budget_seconds'],'watchdog_exit_seconds':cfg['hard_shutdown_seconds']})
    return target,cfg

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-smoke',action='store_true')
    p.add_argument('--cache-source',default='thisarivson/nlp-hw02-compact-cache-verified/2');a=p.parse_args()
    target,_=build(a.gpu_smoke,a.cache_source);print(target)
