#!/usr/bin/env python3
"""Local P1 production/smoke assembly using reviewed L4 notebook cells. No push."""
import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from build_l3 import ROOT, literals, write


def build(smoke=False, cache_source='thisarivson/nlp-hw02-compact-cache-verified/2'):
    nb=json.loads((ROOT/'notebooks/nlp-hw02-kaggle-l4.ipynb').read_text())
    codes=[c['source'] for c in nb['cells'] if c['cell_type']=='code']
    original=literals(codes[0]);cfg=copy.deepcopy(original['CONFIG'])
    pins=json.loads((ROOT/'l4-anchors.json').read_text())
    sources={n:(ROOT/n if n in ('l2_worker.py','l2_compare.py') else ROOT/'src'/n).read_text()
             for n in original['SOURCES']}
    for n in ('p1_math.py','p1_runtime.py','p1_eval.py','p1_session.py','p1_gpu_smoke.py'):
        sources[n]=(ROOT/'src'/n).read_text()
    sources['p1_worker.py']=(ROOT/'p1_worker.py').read_text()
    cfg.update(owner='friend1',session_kind='p1',gpu_smoke=smoke,
        live_dataset='friend1/nlp-hw02-p1-live',
        resume_anchors={'C-long2x':pins['anchors']['C-long2x']},
        main_order_sha256=pins['anchors']['C-main']['identity']['order_sha256'],
        cache_source=cache_source,pilot_updates=round(.15*pins['anchors']['C-long2x']['updates']),
        pilot_a_cap_seconds=13800,pilot_b_cap_seconds=17400,pilot_eval_seconds=1800,
        pilot_k8_rows=2048,reviewer_eval_seconds=900,reviewer_io_reserve_seconds=180,
        reviewer_expected_rows={'heldout4k':4096,'old991':991},
        checkpoint_seconds=1800,reviewer_validation_interval=10800,reviewer_warmup_updates=1000,
        budget_seconds=37380,worker_deadline_seconds=37380,hard_shutdown_seconds=37800,
        session_ceiling_seconds=43200,setup_cap_seconds=480,vote_cap_seconds=900,
        port_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest())
    for key in ('phase_recipe','main_added_seconds'):
        cfg.pop(key,None)
    if smoke:
        cfg.update(pilot_updates=64,pilot_a_cap_seconds=120,pilot_b_cap_seconds=180,
                   pilot_eval_seconds=150,pilot_k8_rows=36,reviewer_eval_seconds=120,
                   reviewer_io_reserve_seconds=60,reviewer_validation_interval=180,
                   checkpoint_seconds=120,reviewer_warmup_updates=20,budget_seconds=900,worker_deadline_seconds=900,
                   hard_shutdown_seconds=1020,session_ceiling_seconds=1080,vote_cap_seconds=60)
        cfg['reviewer_expected_rows']={'heldout4k':64,'old991':16}
        if cache_source.startswith('thisarivson/'):
            cfg['review_fallback_source']='friend1/nlp-hw02-compact-cache-verified'
    lines=codes[0].splitlines()
    startup='\n'.join('CONFIG='+repr(cfg) if line.startswith('CONFIG=') else
                      'SOURCES='+repr(sources) if line.startswith('SOURCES=') else line for line in lines)+'\n'
    startup=startup.replace('nlp-l4-','nlp-p1-').replace('L4 requires','P1 requires')
    # Watchdog is armed before imports/install, just as in the reviewed L4 notebook.
    startup=startup.replace('from l2_session import Watchdog','from p1_session import Watchdog')
    live=codes[2].replace("'session':'L4'","'session':'P1'").replace('Private L4 live probe','Private P1 live probe')
    setup=codes[3].split("if CONFIG['gpu_smoke']:",1)[0]
    setup=setup.replace("resolve_dataset(inp,'nlp-hw02-26b-inputs')",
                        "resolve_cache_mount(inp,'thisarivson/nlp-hw02-26b-inputs','datasets')")
    setup=setup.replace("CONFIG['resume_anchors']['C-main']['identity']['order_sha256']","CONFIG['main_order_sha256']")
    setup=setup.replace('from l2_session import launch','from p1_session import launch')
    setup=setup.replace("MODULE_DIR/'l2_worker.py'","MODULE_DIR/'p1_worker.py'")
    setup+='''
if CONFIG['gpu_smoke']:
    assert json.loads((WORKING/'pilot-smoke.json').read_text())['passed']
    assert json.loads((WORKING/'reviewer-smoke.json').read_text())['passed']
    assert report['exit_codes']==[1,0] and not report['unreaped_worker_pids']
    assert not (WORKING/'stop-request.json').exists()
    write(WORKING/'p1-smoke-summary.json',{'passed':True,'expected_pilot_crash':True,
          'reviewer_completed':True,'no_cross_worker_stop':True,'GPU':True})
'''
    contents=[('markdown',(ROOT.parent/'notes/p1-plan.md').read_text().split('---',2)[-1].lstrip()),
              ('code',startup),('code',codes[1]),('code',live),('code',setup)]
    nb['cells']=[]
    for kind,source in contents:
        c=dict(id=f'p1-{len(nb["cells"]):02d}',cell_type=kind,metadata={},source=source)
        if kind=='code':c.update(outputs=[],execution_count=None);compile(source,'P1 notebook','exec')
        nb['cells'].append(c)
    slug='nlp-hw02-p1-gpu-smoke' if smoke else 'nlp-hw02-kaggle-p1'
    target=ROOT/'notebooks'/f'{slug}.ipynb';write(target,nb)
    bundle=ROOT/'bundles'/('p1_gpu_smoke' if smoke else 'p1');bundle.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(target,bundle/target.name)
    kernel_cache=cache_source.startswith('thisarivson/') and cache_source.endswith('/2')
    datasets=['thisarivson/nlp-hw02-26b-inputs',pins['source']]
    if not kernel_cache:datasets.append(cache_source)
    if cfg.get('review_fallback_source'):datasets.append(cfg['review_fallback_source'])
    write(bundle/'kernel-metadata.json',dict(id='friend1/'+slug,title=slug.replace('-',' '),
        code_file=target.name,language='python',kernel_type='notebook',is_private=True,
        enable_gpu=True,enable_tpu=False,enable_internet=True,dataset_sources=datasets,
        kernel_sources=[cache_source] if kernel_cache else [],competition_sources=[],machine_shape='NvidiaTeslaT4'))
    write(bundle/'session-limits.json',dict(session_timeout_seconds=cfg['session_ceiling_seconds'],
          kind='p1_gpu_smoke' if smoke else 'p1',budget_seconds=cfg['budget_seconds'],
          watchdog_exit_seconds=cfg['hard_shutdown_seconds']))
    write(ROOT/'p1-anchors.json',dict(source=pins['source'],source_version=pins['source_version'],
          anchors=cfg['resume_anchors'],weighted_order=pins['weighted_order']))
    return target,cfg


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu-smoke',action='store_true')
    p.add_argument('--cache-source',default='thisarivson/nlp-hw02-compact-cache-verified/2');a=p.parse_args()
    print(build(a.gpu_smoke,a.cache_source)[0])
