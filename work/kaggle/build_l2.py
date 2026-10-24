#!/usr/bin/env python3
"""Build private L2 and reviewer smoke artifacts. Never push or start Kaggle."""
import argparse, hashlib, json, shutil, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import write, sha
from build import INSTALL
USER_ANCHOR={'file':'model-C__checkpoints__main-5746a4a2897e.pt','updates':47155,'cursor':6035840,'main_seconds':17570.}
CACHE_SHA='46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be'

def build(smoke=False):
    sources={p.name:p.read_text() for p in sorted((ROOT/'src').glob('*.py'))}
    for n in ('l2_worker.py','l2_compare.py'):sources[n]=(ROOT/n).read_text()
    frozen=json.loads((ROOT/'source-identity.json').read_text())
    assert all(hashlib.sha256(sources[n].encode()).hexdigest()==h for n,h in frozen['modules'].items())
    limits={'budget_seconds':39600,'worker_deadline_seconds':39600,'hard_shutdown_seconds':41040,'session_ceiling_seconds':43200}
    if smoke:limits={'budget_seconds':900,'worker_deadline_seconds':900,'hard_shutdown_seconds':1020,'session_ceiling_seconds':1080}
    cfg=dict(limits,input_root='/kaggle/input',working_root='/kaggle/working',session_kind='l2',
             terminate_grace_seconds=20,live_dataset='thisarivson/nlp-hw02-l2-live',gpu_smoke=smoke,
             cache_source='thisarivson/nlp-hw02-compact-cache-verified/2',cache_manifest_sha256=CACHE_SHA,
             anchor_user=USER_ANCHOR,port_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest())
    pin=ROOT/'l2-anchor.json'
    if pin.exists():cfg['anchor_pin']=json.loads(pin.read_text())
    else:cfg['preparation_blocker']='Real L1 C CPU audit / live private precreation not verified; production gate refuses run'
    order_pin=ROOT.parent/'checks/kaggle-l2-weighted-order.json'
    if order_pin.exists():cfg['weighted_order_sha256']=json.loads(order_pin.read_text())['file']['sha256']
    if smoke:
        cfg.update(phase_recipe={'main_targets':[150.,300.],'cooldown_seconds':[20.,20.],
                                 'eval_seconds':[120.,150.]},vote_cap_seconds=60.)
    cells=[]
    def cell(kind,s):
        obj={'id':f'l2-{len(cells):02d}','cell_type':kind,'metadata':{},'source':s}
        if kind=='code':obj.update(outputs=[],execution_count=None)
        cells.append(obj)
    cell('markdown','# L2 — C-main / C-long2x, private T4×2\n\nНе запускать до независимого ревью. Боевой режим: Kaggle Secrets KAGGLE_USERNAME/KAGGLE_KEY, Internet on. Смоук не требует Secrets и не публикует live. Полный anchor C проверяется до install/cache; D не загружается.\n\n'+(ROOT.parent/'notes/l2-plan.md').read_text().split('---',2)[-1].lstrip())
    startup='''import time,json,os,sys,tempfile
from pathlib import Path
SESSION_START=time.monotonic()
'''+ 'CONFIG='+repr(cfg)+'''
WORKING=Path(CONFIG['working_root']);WORKING.mkdir(parents=True,exist_ok=True)
CONFIG['session_start_monotonic']=SESSION_START
CONFIG['deadline_monotonic']=SESSION_START+CONFIG['worker_deadline_seconds']
CONFIG['hard_deadline_monotonic']=SESSION_START+CONFIG['hard_shutdown_seconds']
CONFIG['temp_root']=tempfile.mkdtemp(prefix='nlp-l2-',dir='/kaggle/temp' if Path('/kaggle/temp').is_dir() else None)
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
if CONFIG.get('preparation_blocker'):WATCHDOG.finish();raise RuntimeError(CONFIG['preparation_blocker'])
# SHA, counters, main_s, semantic code/Torch/CUDA, finite Adam/model: all before large cache reads.
from l2_resume import mounted_expected,verify_anchor
try:
    source=resolve_dataset(CONFIG['input_root'],'nlp-hw02-l1-live')
    expected=mounted_expected(source,CONFIG['anchor_user'])
    pin=CONFIG['anchor_pin']
    assert expected['sha256']==pin['sha256'] and expected['bytes']==pin['bytes']
    assert expected['identity']==pin['identity'] and expected['main_seconds']==pin['main_seconds']
    _,checked=verify_anchor(source,expected)
    write(WORKING/'anchor-preflight.json',{'passed':True,'sha256':expected['sha256'],'updates':checked['state']['updates'],
          'cursor':checked['state']['cursor'],'main_seconds':checked['state']['main_seconds'],
          'preflight_wall_seconds':time.monotonic()-SESSION_START,'D_loaded':False})
    del checked
    CONFIG['anchor_root']=str(source);CONFIG['anchor_expected']=expected
except Exception as error:
    write(WORKING/'anchor-preflight.json',{'passed':False,'error_type':type(error).__name__,'training_started':False})
    WATCHDOG.finish();raise
'''
    cell('code',startup)
    cell('code',INSTALL.replace("'sympy':'1.14.0'","'sympy':'1.14.0','kaggle':'2.2.4'"))
    cell('code','''from l2_live import L2LiveSync
LIVE=None
if not CONFIG['gpu_smoke']:
    LIVE=L2LiveSync(WORKING,CONFIG['temp_root'],CONFIG['live_dataset'])
    write(WORKING/'boot-manifest.json',{'session':'L2','training_started':False,'private_live_probe':True})
    assert LIVE.sync('startup-probe',timeout=300.,wait_ready=True),'Private L2 live probe refused training'
    WATCHDOG.live_sync=LIVE;LIVE.start()
''')
    cell('code','''from kaggle_io import materialize_dataset
from kaggle_runtime import prepare_starter
from l1_cache import CompactCache,ROWS
from l2_sampler import build_order
from tokenizers import Tokenizer
scratch=Path(CONFIG['temp_root']);inp=Path(CONFIG['input_root'])
CONFIG['starter_dataset']=str(prepare_starter(resolve_dataset(inp,'nlp-hw02-26b-inputs'),scratch/'starter','training-inputs',read_only=True))
cache_root=materialize_dataset(resolve_dataset(inp,CONFIG['cache_source'].split('/')[1]),scratch/'compact','compact-manifest.json')
assert sha(cache_root/'compact-manifest.json')==CONFIG['cache_manifest_sha256']
summary=json.loads((cache_root/'cache-summary.json').read_text())
assert summary['rows']==ROWS and summary['roundtrip_rows']>=50000 and summary['answer_mismatches']==0
cache=CompactCache(cache_root,verify=True);assert len(cache)==ROWS
original=Tokenizer.from_file(str(Path(CONFIG['starter_dataset'])/'tokenizer.json'))
heldout=json.loads((Path(CONFIG['starter_dataset'])/'heldout4k.json').read_text())
lengths=[len(original.encode(r['code'],add_special_tokens=False).ids) for r in heldout]
assert len(lengths)==4096
# Required inputs contain compact offsets. Accept an explicitly mounted original cache only with exact identities.
raw_root=None
raw_candidates=[p.parent for p in inp.rglob('cache-manifest.json') if (p.parent/'shards').is_dir()]
if len(raw_candidates)==1:
    raw_meta=json.loads((raw_candidates[0]/'cache-manifest.json').read_text())
    if raw_meta.get('rows')==ROWS:raw_root=raw_candidates[0]
order_root=WORKING/'weighted-order'
_,order_stats=build_order(cache,lengths,order_root,original_root=raw_root)
assert order_stats['file']['sha256']==CONFIG['weighted_order_sha256'],'Weighted order differs from CPU-prepared SHA'
CONFIG['weighted_order_root']=str(order_root);CONFIG['compact_cache']=str(cache_root)
del cache,heldout,lengths,original
assert time.monotonic()-SESSION_START<900,'Setup exceeded fixed budget; no unequal truncated experiment'
from l2_session import launch
report=launch(CONFIG,MODULE_DIR/'l2_worker.py',WATCHDOG,LIVE)
print(json.dumps(report,indent=2))
if CONFIG['gpu_smoke']:
    # An injected worker failure is successful only if both primary CUDA suites passed
    # and the healthy sibling resumed AFTER that failure; packaging must still exist.
    reports={name:json.loads((WORKING/('model-'+name)/'gpu-smoke.json').read_text()) for name in ('C-main','C-long2x')}
    assert report['exit_codes']==[1,0] and report['failed_workers']==[0]
    assert all(r['passed'] for r in reports.values())
    assert 'healthy_worker_resumed_after_sibling_crash' in reports['C-long2x']['checks']
    assert (WORKING/'l2-output-manifest.json').exists() and not (WORKING/'stop-request.json').exists()
    write(WORKING/'cuda-smoke-summary.json',{'passed':True,'expected_fault_exit_codes':[1,0],
          'packaging_passed':True,'reports':reports,'wall_seconds':time.monotonic()-SESSION_START})
''')
    slug='nlp-hw02-l2-gpu-smoke' if smoke else 'nlp-hw02-kaggle-l2'
    nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python'}},'cells':cells}
    target=ROOT/'notebooks'/f'{slug}.ipynb';write(target,nb)
    folder=ROOT/'bundles'/('l2_gpu_smoke' if smoke else 'l2');folder.mkdir(parents=True,exist_ok=True);shutil.copyfile(target,folder/target.name)
    write(folder/'kernel-metadata.json',{'id':'thisarivson/'+slug,'title':slug.replace('-',' '),'code_file':target.name,
          'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,
          'dataset_sources':['thisarivson/nlp-hw02-26b-inputs','thisarivson/nlp-hw02-l1-live'],
          'kernel_sources':['thisarivson/nlp-hw02-compact-cache-verified/2'],'competition_sources':[],'machine_shape':'NvidiaTeslaT4'})
    write(folder/'session-limits.json',{'session_timeout_seconds':cfg['session_ceiling_seconds'],'kind':'l2_gpu_smoke' if smoke else 'l2',
          'budget_seconds':cfg['budget_seconds'],'watchdog_exit_seconds':cfg['hard_shutdown_seconds']})
    return target,cfg

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-smoke',action='store_true');a=p.parse_args();target,_=build(a.gpu_smoke);print(target)
