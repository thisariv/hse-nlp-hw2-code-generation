#!/usr/bin/env python3
"""L3 assembly/source/budget/privacy checks. Does not execute notebooks or touch Kaggle."""
import ast,hashlib,json,sys
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import sha,write
from build_l3 import literals

def main():
    original=literals(next(c['source'] for c in json.loads((ROOT/'notebooks/nlp-hw02-kaggle-l2.ipynb').read_text())['cells'] if c['cell_type']=='code'))
    # All frozen L2 runtime/recovery/live/evaluation/sampler modules stay byte-identical.
    for name,text in original['SOURCES'].items():
        if name.startswith('l2_') and name not in ('l2_worker.py',):
            path=ROOT/name if name=='l2_compare.py' else ROOT/'src'/name
            assert text==path.read_text(), 'L2 module changed: '+name
    pins=json.loads((ROOT/'l3-anchors.json').read_text())
    results={}
    for kind,slug in [('l3','nlp-hw02-kaggle-l3'),('l3_gpu_smoke','nlp-hw02-l3-gpu-smoke')]:
        path=ROOT/'notebooks'/f'{slug}.ipynb';nb=json.loads(path.read_text());codes=[c for c in nb['cells'] if c['cell_type']=='code']
        nbformat.validate(nbformat.from_dict(nb))
        for c in codes:
            assert c['execution_count'] is None and not c['outputs'];compile(c['source'],slug,'exec')
        values=literals(codes[0]['source']);cfg=values['CONFIG'];sources=values['SOURCES']
        for name,content in sources.items():
            assert not name.startswith('final_');compile(content,name,'exec')
            source=ROOT/name if name in ('l2_worker.py','l2_compare.py') else ROOT/'src'/name
            assert source.read_text()==content
        assert cfg['resume_anchors']==pins['anchors'] and cfg['weighted_order_pin']==pins['weighted_order']
        assert cfg['weighted_order_sha256']==pins['weighted_order']['file']['sha256']
        assert cfg['session_kind']=='l3' and cfg['owner']=='friend1' and cfg['live_dataset']=='friend1/nlp-hw02-l3-live'
        assert cfg['cache_manifest_sha256']=='46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be'
        tree=ast.parse(codes[0]['source']);calls={n.func.id:n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        assert calls['Watchdog']<calls['gate']<calls['verify_anchor']
        assert 'build_order' not in codes[-1]['source'] and 'restore_order' in codes[-1]['source']
        assert "WORKING/'input-preflight.json'" in codes[-1]['source']
        assert codes[-1]['source'].index('verify=True') < codes[-1]['source'].index("WORKING/'input-preflight.json'") < codes[-1]['source'].index('report=launch')
        meta=json.loads((ROOT/'bundles'/kind/'kernel-metadata.json').read_text())
        limits=json.loads((ROOT/'bundles'/kind/'session-limits.json').read_text())
        assert meta['id']=='friend1/'+slug and meta['is_private'] and meta['enable_internet']
        assert meta['enable_gpu'] and not meta['enable_tpu'] and meta['machine_shape']=='NvidiaTeslaT4'
        expected=['thisarivson/nlp-hw02-26b-inputs','thisarivson/nlp-hw02-l2-live']
        if cfg['cache_source'].startswith('friend1/'):
            expected.append(cfg['cache_source']);assert meta['kernel_sources']==[]
        else:assert meta['kernel_sources']==['thisarivson/nlp-hw02-compact-cache-verified/2']
        if cfg.get('review_fallback_source'):
            expected.append(cfg['review_fallback_source'])
            assert "WORKING/'fallback-preflight.json'" in codes[-1]['source']
        assert meta['dataset_sources']==expected
        assert sha(ROOT/'bundles'/kind/path.name)==sha(path)
        if kind=='l3':
            recipe=cfg['phase_recipe'];assert recipe=={'main_targets':[16740.,33480.],'cooldown_seconds':[900.,3000.],'eval_seconds':[300.,1800.]}
            assert 480+33480+900+3000+300+1800+180+120==cfg['worker_deadline_seconds']==40260
            assert cfg['worker_deadline_seconds']+420+120==cfg['hard_shutdown_seconds']==40800<41040
            assert 2*cfg['hard_shutdown_seconds']+1080<25*3600
        else:assert cfg['gpu_smoke'] and cfg['hard_shutdown_seconds']==1020 and limits['session_timeout_seconds']==1080
        results[kind]={'notebook_sha256':sha(path),'metadata':meta,'limits':limits,'embedded_modules':len(sources)}
    cpu={}
    for name in pins['anchors']:
        r=json.loads((ROOT.parent/f'checks/kaggle-l3-real-resume-{name}.json').read_text())
        assert r['passed'] and r['steps_bit_exact']==2 and r['downloaded_anchor_removed'];cpu[name]=r['anchor_sha256']
    created=json.loads((ROOT.parent/'checks/kaggle-l3-live-created.json').read_text())
    fallback=json.loads((ROOT.parent/'checks/kaggle-l3-cache-fallback.json').read_text())
    l2=json.loads((ROOT.parent/'checks/kaggle-l2-regressions.json').read_text())
    l3=json.loads((ROOT.parent/'checks/kaggle-l3-regressions.json').read_text())
    assert l2['passed'] and l3['passed']
    assert created['private'] and created['status']=='ready' and fallback['private'] and fallback['status']=='ready'
    assert fallback['downloaded_remote_manifest_sha256']==pins['anchors']['C-main']['identity']['cache_manifest_sha256']
    write(ROOT.parent/'checks/kaggle-l3-static.json',{'passed':True,'artifacts':results,'L2_modules_unchanged':True,
          'real_CPU_anchors_verified':cpu,'L2_regressions':l2,'L3_regressions':l3,'nbformat_valid':True,
          'GPU':False,'smoke_push':False,'production_push':False})
    print('L3 STATIC PASSED')

if __name__=='__main__':main()
