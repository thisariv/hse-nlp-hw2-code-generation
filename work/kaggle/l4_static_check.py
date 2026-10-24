"""Local L4 assembly, quota, pinned live identity and real CPU resume receipts."""
import ast
import hashlib
import json
import sys
from pathlib import Path
import nbformat
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from build_l3 import literals
from l3_resume import mounted_expected
from kaggle_io import sha,write


def main():
    pins=json.loads((ROOT/'l4-anchors.json').read_text())
    source=ROOT.parent/'results/kaggle-runs/nlp-hw02-kaggle-l3/final'
    for name,pin in pins['anchors'].items(): mounted_expected(source,name,pin)
    old=json.loads((ROOT/'notebooks/nlp-hw02-kaggle-l2.ipynb').read_text())
    original=literals(next(c['source'] for c in old['cells'] if c['cell_type']=='code'))
    artifacts={}
    for kind,slug in [('l4','nlp-hw02-kaggle-l4'),('l4_gpu_smoke','nlp-hw02-l4-gpu-smoke')]:
        nb_path=ROOT/'notebooks'/f'{slug}.ipynb';nb=json.loads(nb_path.read_text())
        nbformat.validate(nbformat.from_dict(nb))
        codes=[c for c in nb['cells'] if c['cell_type']=='code']
        for c in codes:
            assert c['execution_count'] is None and not c['outputs'];compile(c['source'],slug,'exec')
        values=literals(codes[0]['source']);cfg,sources=values['CONFIG'],values['SOURCES']
        assert cfg['port_sha256']==hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
        assert cfg['resume_anchors']==pins['anchors'] and cfg['anchor_source']=='friend1/nlp-hw02-l3-live'
        assert cfg['inherited_points']==5 and cfg['session_kind']=='l4'
        assert cfg['weighted_order_pin']==pins['weighted_order'] and cfg['weighted_order_sha256']==pins['weighted_order']['file']['sha256']
        assert cfg['owner']=='friend1' and cfg['live_dataset']=='friend1/nlp-hw02-l4-live'
        for name,content in sources.items():
            compile(content,name,'exec');path=ROOT/name if name in ['l2_worker.py','l2_compare.py'] else ROOT/'src'/name
            assert path.read_text()==content
            if name.startswith('l2_') and name!='l2_worker.py':assert content==original['SOURCES'][name],name
        assert "in ('l3','l4')" in sources['l2_worker.py'] and "expected_points=cfg.get('inherited_points',3)" in sources['l2_worker.py']
        assert "expected_points=CONFIG.get('inherited_points',3)" in codes[0]['source']
        tree=ast.parse(codes[0]['source']);calls={n.func.id:n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        assert calls['Watchdog']<calls['gate']<calls['verify_anchor']
        assert 'build_order' not in codes[-1]['source'] and 'restore_order' in codes[-1]['source']
        assert codes[-1]['source'].index('verify=True')<codes[-1]['source'].index("WORKING/'input-preflight.json'")<codes[-1]['source'].index('report=launch')
        bundle=ROOT/'bundles'/kind
        metadata=json.loads((bundle/'kernel-metadata.json').read_text());limits=json.loads((bundle/'session-limits.json').read_text())
        assert sha(bundle/nb_path.name)==sha(nb_path)
        assert metadata['id']=='friend1/'+slug and metadata['is_private'] and metadata['enable_internet']
        assert metadata['enable_gpu'] and not metadata['enable_tpu'] and metadata['machine_shape']=='NvidiaTeslaT4'
        expected=['thisarivson/nlp-hw02-26b-inputs','friend1/nlp-hw02-l3-live']
        if cfg['cache_source'].startswith('friend1/'):
            expected.append(cfg['cache_source']);assert not metadata['kernel_sources']
        else:assert metadata['kernel_sources']==['thisarivson/nlp-hw02-compact-cache-verified/2']
        if cfg.get('review_fallback_source'):expected.append(cfg['review_fallback_source'])
        assert metadata['dataset_sources']==expected
        if kind=='l4':
            recipe=cfg['phase_recipe'];assert recipe=={'main_targets':[15300.,30600.],'cooldown_seconds':[900.,3000.],'eval_seconds':[300.,1800.]}
            assert 480+30600+900+3000+300+1800+180+120==cfg['worker_deadline_seconds']==37380
            assert cfg['worker_deadline_seconds']+300+120==cfg['hard_shutdown_seconds']==37800<=10.5*3600
            for pin in pins['anchors'].values():assert [int(pin['l2_main_seconds']+t) for t in recipe['main_targets']]==[77580,92880]
        else:assert cfg['gpu_smoke'] and cfg['hard_shutdown_seconds']==1020 and limits['session_timeout_seconds']==1080
        artifacts[kind]={'notebook_sha256':sha(nb_path),'metadata':metadata,'limits':limits,'embedded_modules':len(sources)}
    regression={}
    for session in ['l2','l3','l4']:
        receipt=json.loads((ROOT.parent/f'checks/kaggle-{session}-regressions.json').read_text())
        assert receipt['passed'];regression[session]=receipt
    resumes={}
    for name,pin in pins['anchors'].items():
        r=json.loads((ROOT.parent/f'checks/kaggle-l4-real-resume-{name}.json').read_text())
        assert r['passed'] and r['steps_bit_exact']==2 and r['downloaded_anchor_removed']
        assert r['anchor_sha256']==pin['sha256'] and r['anchor_bytes']==pin['bytes']
        assert r['identity_history_cursor_preserved'] and r['temporary_runtime_checkpoints_removed']
        assert not (source/pin['file']).exists();resumes[name]=r['anchor_sha256']
    live=json.loads((ROOT.parent/'checks/kaggle-l4-live-created.json').read_text())
    assert live['private'] and live['account']=='friend1' and live['status'].startswith('ready')
    write(ROOT.parent/'checks/kaggle-l4-static.json',{'passed':True,'artifacts':artifacts,'regressions':regression,
          'L2_frozen_modules_unchanged':True,'pinned_latest_manifest_verified':True,'CPU_real_resumes':resumes,
          'private_live':live,'hard_wall_seconds':37800,'nbformat_valid':True,'GPU':False,'push':False})
    print('L4 STATIC PASSED; hard wall 37800s; private live ready; GPU/push=0')


if __name__=='__main__':main()
