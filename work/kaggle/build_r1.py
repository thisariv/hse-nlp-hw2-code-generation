#!/usr/bin/env python3
"""Parameterised private R1 assembly. Does not execute or push a notebook."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from build_l3 import ROOT, literals, write


def default_branches():
    return [dict(anchor='C-long2x',growth=g,data='ordinary',order_path=None,order_sha256=None,
                 order_source=None,order_seed=2026101701,q=.1,main_seconds=34200.,cooldown='linear',
                 cooldown_seconds=1200.,cooldown_fraction=.1,eval_seconds=3600.,new_warmup=1000,encoder_layers=12)
            for g in ('encoder-deepen','none')]


def validate(branches):
    assert len(branches)==2
    for b in branches:
        assert b['anchor'] in ('C-main','C-long2x') and b['growth'] in ('none','encoder-deepen')
        assert b.get('encoder_layers',12) in (9,12)
        assert b['data'] in ('ordinary','filtered') and 0<=b['q']<1 and b['new_warmup']>0
        assert b['main_seconds']>0 and b['eval_seconds']>0 and b['cooldown_seconds']>0
        assert b['cooldown'] in ('linear','1-sqrt') and 0<b['cooldown_fraction']<1
        assert 480+b['main_seconds']+b['cooldown_seconds']+b['eval_seconds']+180<=40260
        if b['data']=='filtered':
            assert b['order_path'] and b['order_sha256'] and len(b['order_sha256'])==64
            assert isinstance(b['order_seed'],int)
    return branches


def build(owner='friend1',branches=None,smoke=False,cache_source='thisarivson/nlp-hw02-compact-cache-verified/2'):
    assert owner in ('friend1','thisarivson','friend2')
    branches=validate(copy.deepcopy(branches or default_branches()))
    nb=json.loads((ROOT/'notebooks/nlp-hw02-kaggle-p1.ipynb').read_text())
    codes=[c['source'] for c in nb['cells'] if c['cell_type']=='code'];old=literals(codes[0])
    sources={n:(ROOT/n if n in ('l2_worker.py','l2_compare.py','p1_worker.py') else ROOT/'src'/n).read_text()
             for n in old['SOURCES']}
    for n in ('r1_growth.py','r1_runtime.py','r1_data.py','r1_session.py','r1_live.py','r1_gpu_smoke.py'):
        sources[n]=(ROOT/'src'/n).read_text()
    final=json.loads((ROOT/'notebooks/nlp-hw02-ensemble-inference-v7.ipynb').read_text())
    sources.update(literals(next(c['source'] for c in final['cells'] if c['cell_type']=='code'))['SOURCES'])
    sources['final_inference_v7.py']=(ROOT/'src/final_inference_v7.py').read_text()
    sources['r1_worker.py']=(ROOT/'r1_worker.py').read_text()
    pins=json.loads((ROOT/'r1-anchors.json').read_text())
    cfg=copy.deepcopy(old['CONFIG'])
    for key in list(cfg):
        if key.startswith(('pilot_','reviewer_')) or key in ('phase_recipe','main_added_seconds'):
            cfg.pop(key)
    cfg.update(owner=owner,session_kind='r1',gpu_smoke=smoke,branches=branches,
               resume_anchors=pins['anchors'],weighted_order_pin=pins['weighted_order'],
               anchor_source=pins['source'] if owner=='friend1' else 'thisarivson/nlp-hw02-l3-states',
               require_secrets=False,live_sync_optional=True,checkpoint_seconds=900,
               live_dataset=owner+'/nlp-hw02-r1-live',cache_source=cache_source,
               setup_cap_seconds=480,io_reserve_seconds=180,budget_seconds=40260,worker_deadline_seconds=40260,
               hard_shutdown_seconds=40800,session_ceiling_seconds=43200,vote_cap_seconds=1200,
               expected_rows=dict(splits=dict(heldout4k=4096,old991=991),test=1919,k8=2048),
               port_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest())
    cfg.pop('review_fallback_source',None)
    if smoke:
        for b in branches:b.update(main_seconds=300.,max_main_updates=256,cooldown_seconds=20.,eval_seconds=120.,new_warmup=16)
        cfg.update(budget_seconds=780,worker_deadline_seconds=780,hard_shutdown_seconds=840,
                   session_ceiling_seconds=900,io_reserve_seconds=30,vote_cap_seconds=60,
                   expected_rows=dict(splits=dict(heldout4k=64,old991=16),test=64,k8=36))
    # Inherited startup arms the watchdog before Torch and checks owner/Secrets.
    startup='\n'.join('CONFIG='+repr(cfg) if line.startswith('CONFIG=') else
                      'SOURCES='+repr(sources) if line.startswith('SOURCES=') else line for line in codes[0].splitlines())+'\n'
    startup=startup.replace('nlp-p1-','nlp-r1-').replace('P1 requires friend1 Secrets','R1 requires configured owner Secrets')
    startup=startup.replace('from p1_session import Watchdog','from r1_session import Watchdog')
    startup=startup.replace('from l2_live import gate','from r1_live import optional_gate')
    startup=startup.replace("GATE=gate(torch,require_secrets=not CONFIG['gpu_smoke'])", "GATE=optional_gate(torch,CONFIG['owner'],CONFIG['gpu_smoke'])")
    startup=startup.replace("if not CONFIG['gpu_smoke'] and os.environ.get('KAGGLE_USERNAME')!=CONFIG['owner']:", "if GATE.get('live_enabled') and os.environ.get('KAGGLE_USERNAME')!=CONFIG['owner']:")
    startup=startup.replace("CONFIG['anchor_root']=str(source)", "CONFIG['anchor_root']=str(source)\n    print('R1 anchor-preflight PASSED',json.dumps(checks),flush=True)")
    startup=startup.replace("source=resolve_dataset(CONFIG['input_root'],CONFIG['anchor_source'])",
        "from l3_resume import resolve_cache_mount\n    source=resolve_cache_mount(CONFIG['input_root'],CONFIG['anchor_source'],'datasets')")
    # Copied thisarivson states retain the original producer manifest and receipts.
    install=codes[1]
    live=codes[2].replace("'session':'P1'","'session':'R1'").replace('Private P1 live probe','Private R1 live probe')
    live=live.replace('from l2_live import L2LiveSync','from r1_live import R1LiveSync').replace('LIVE=L2LiveSync','LIVE=R1LiveSync')
    live=live.replace("if not CONFIG['gpu_smoke']:", "if GATE.get('live_enabled'):")
    live+="\nprint('R1 live-sync', 'enabled' if LIVE else 'disabled; local stable checkpoints',flush=True)\n"
    setup=codes[3].split("if CONFIG.get('review_fallback_source'):",1)[0]
    setup+='''
from r1_session import launch
assert time.monotonic()-SESSION_START<CONFIG['setup_cap_seconds'],'R1 setup exceeded budget'
report=launch(CONFIG,MODULE_DIR/'r1_worker.py',WATCHDOG,LIVE)
print(json.dumps(report,indent=2))
if CONFIG['gpu_smoke']:
    assert report['status']=='complete' and report['exit_codes']==[0,0]
    for gpu in (0,1):assert json.loads((WORKING/f'r1-smoke-gpu{gpu}.json').read_text())['passed']
'''
    # main_order_sha256 remains required by the inherited input preflight.
    cfg['main_order_sha256']=pins['anchors']['C-main']['identity']['order_sha256']
    startup='\n'.join('CONFIG='+repr(cfg) if line.startswith('CONFIG=') else line for line in startup.splitlines())+'\n'
    nb['cells']=[]
    note=ROOT.parent/'notes/r1-plan.md'
    for kind,source in [('markdown',note.read_text().split('---',2)[-1].lstrip() if note.exists() else '# R1'),
                        ('code',startup),('code',install),('code',live),('code',setup)]:
        cell=dict(id=f'r1-{len(nb["cells"]):02d}',cell_type=kind,metadata={},source=source)
        if kind=='code':cell.update(outputs=[],execution_count=None);compile(source,'R1 cell','exec')
        nb['cells'].append(cell)
    slug='nlp-hw02-r1-gpu-smoke' if smoke else 'nlp-hw02-kaggle-r1'
    kind='r1_gpu_smoke' if smoke else 'r1';target=ROOT/'notebooks'/f'{slug}.ipynb';write(target,nb)
    bundle=ROOT/'bundles'/kind;bundle.mkdir(parents=True,exist_ok=True);shutil.copyfile(target,bundle/target.name)
    kernel_cache=cache_source.endswith('/2') and cache_source.startswith('thisarivson/')
    datasets=['thisarivson/nlp-hw02-26b-inputs',cfg['anchor_source']]
    datasets.extend(b['order_source'] for b in branches if b.get('order_source'))
    if not kernel_cache:datasets.append(cache_source)
    write(bundle/'kernel-metadata.json',dict(id=owner+'/'+slug,title=slug.replace('-',' '),code_file=target.name,
          language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,
          enable_internet=True,dataset_sources=sorted(set(datasets)),kernel_sources=[cache_source] if kernel_cache else [],
          competition_sources=[],machine_shape='NvidiaTeslaT4'))
    write(bundle/'session-limits.json',dict(session_timeout_seconds=cfg['session_ceiling_seconds'],kind=kind,
          budget_seconds=cfg['budget_seconds'],watchdog_exit_seconds=cfg['hard_shutdown_seconds']))
    write(bundle/'source-identity.json',dict(modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}))
    return target,cfg


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--owner',choices=['thisarivson','friend1','friend2'],default='friend2')
    p.add_argument('--config',type=Path,help='JSON list of two per-GPU configs; partial configs merge defaults')
    p.add_argument('--gpu-smoke',action='store_true');p.add_argument('--cache-source',default='thisarivson/nlp-hw02-compact-cache-verified/2')
    a=p.parse_args();branches=default_branches()
    if a.config:
        overrides=json.loads(a.config.read_text());assert len(overrides)==2
        for b,over in zip(branches,overrides):b.update(over)
    print(build(a.owner,branches,a.gpu_smoke,a.cache_source)[0])
