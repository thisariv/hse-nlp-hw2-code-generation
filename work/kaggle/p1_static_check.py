"""P1 notebook, source parity, inherited pins, budgets, CPU receipts and privacy."""
import ast
import hashlib
import json
import sys
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from build_l3 import literals
from l3_resume import mounted_expected
from p1_session import Watchdog
from kaggle_io import write,sha


def main():
    pins=json.loads((ROOT/'p1-anchors.json').read_text())
    final=ROOT.parent/'results/kaggle-runs/nlp-hw02-kaggle-l3/final'
    pin=pins['anchors']['C-long2x'];mounted_expected(final,'C-long2x',pin)
    old=json.loads((ROOT/'notebooks/nlp-hw02-kaggle-l4.ipynb').read_text())
    old_sources=literals(next(c['source'] for c in old['cells'] if c['cell_type']=='code'))['SOURCES']
    artifacts={}
    for kind,slug in [('p1','nlp-hw02-kaggle-p1'),('p1_gpu_smoke','nlp-hw02-p1-gpu-smoke')]:
        path=ROOT/'notebooks'/f'{slug}.ipynb';nb=json.loads(path.read_text());nbformat.validate(nbformat.from_dict(nb))
        codes=[c for c in nb['cells'] if c['cell_type']=='code']
        for c in codes:
            assert c['execution_count'] is None and not c['outputs'];compile(c['source'],slug,'exec')
        values=literals(codes[0]['source']);cfg,sources=values['CONFIG'],values['SOURCES']
        assert cfg['port_sha256']==hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
        assert cfg['session_kind']=='p1' and cfg['resume_anchors']==pins['anchors'] and cfg['inherited_points']==5
        assert cfg['owner']=='friend1' and cfg['live_dataset']=='friend1/nlp-hw02-p1-live'
        assert cfg['anchor_source']=='friend1/nlp-hw02-l3-live' and cfg['weighted_order_pin']==pins['weighted_order']
        for n,content in sources.items():
            compile(content,n,'exec')
            src=ROOT/n if n in ('p1_worker.py','l2_worker.py','l2_compare.py') else ROOT/'src'/n
            assert src.read_text()==content, 'stale embedded '+n
            if n.startswith('l2_') or n in ('l3_resume.py','l1_runtime.py','exp26_core.py','l1_cache.py','exp26_order.py'):
                assert old_sources[n]==content, 'inherited source changed '+n
        source=codes[0]['source']
        tree=ast.parse(source)
        calls={n.func.id:n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        torch_line=next(n.lineno for n in tree.body if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
        assert calls['Watchdog']<torch_line<calls['gate']<calls['verify_anchor']
        setup=codes[-1]['source']
        assert setup.index('verify=True')<setup.index("WORKING/'input-preflight.json'")<setup.index('report=launch')
        assert "MODULE_DIR/'p1_worker.py'" in setup and 'build_order' not in setup
        assert 'from p1_session import launch' in setup and 'resolve_cache_mount' in setup
        check_cfg=dict(cfg,session_start_monotonic=0,deadline_monotonic=cfg['worker_deadline_seconds'],
                       hard_deadline_monotonic=cfg['hard_shutdown_seconds'])
        assert Watchdog(check_cfg).hard_deadline==cfg['hard_shutdown_seconds']
        bundle=ROOT/'bundles'/kind
        meta=json.loads((bundle/'kernel-metadata.json').read_text());limits=json.loads((bundle/'session-limits.json').read_text())
        assert meta['is_private'] and meta['enable_gpu'] and meta['enable_internet'] and not meta['enable_tpu']
        assert meta['id']=='friend1/'+slug and meta['machine_shape']=='NvidiaTeslaT4'
        assert sha(bundle/path.name)==sha(path)
        assert limits['watchdog_exit_seconds']==cfg['hard_shutdown_seconds']
        if cfg['cache_source'].startswith('thisarivson/'):
            assert meta['kernel_sources']==['thisarivson/nlp-hw02-compact-cache-verified/2']
        else:assert not meta['kernel_sources'] and cfg['cache_source'] in meta['dataset_sources']
        assert 'thisarivson/nlp-hw02-26b-inputs' in meta['dataset_sources'] and pins['source'] in meta['dataset_sources']
        if kind=='p1':
            assert cfg['pilot_updates']==31372 and cfg['reviewer_warmup_updates']==1000
            assert 480+13800+1800+17400+1800+180+120+1800==cfg['worker_deadline_seconds']==37380
            assert cfg['hard_shutdown_seconds']==37380+300+120==37800<=10.5*3600
            assert cfg['pilot_k8_rows']==2048 and cfg['reviewer_eval_seconds']==900
        else:
            assert cfg['pilot_updates']==64 and cfg['reviewer_warmup_updates']==20
            assert cfg['pilot_k8_rows']==36 and cfg['hard_shutdown_seconds']==1020
        artifacts[kind]=dict(notebook_sha256=sha(path),modules=len(sources),metadata=meta,limits=limits)
    regressions={}
    for session in ('l2','l3','l4','p1'):
        r=json.loads((ROOT.parent/f'checks/kaggle-{session}-regressions.json').read_text());assert r['passed'];regressions[session]=r['tests']
    resume=json.loads((ROOT.parent/'checks/kaggle-p1-real-resume-C-long2x.json').read_text())
    assert resume['passed'] and resume['steps_bit_exact']==2 and resume['downloaded_anchor_removed']
    assert resume['anchor_sha256']==pin['sha256'] and resume['full_restore_before_both_branches']
    assert not (Path('/tmp/nlp-p1-anchor')/pin['file']).exists()
    live=json.loads((ROOT.parent/'checks/kaggle-p1-live-created.json').read_text())
    assert live['private'] and live['account']=='friend1' and live['status'].startswith('ready')
    scoring=json.loads((ROOT.parent/'checks/reviewer-score-cpu-smoke.json').read_text())
    assert scoring['passed'] and scoring['scored_candidates']==128
    assert scoring['full_vote_replay']=={'C-main':{'rows':2048,'matches':1502},'C-long2x':{'rows':2048,'matches':1507}}
    write(ROOT.parent/'checks/kaggle-p1-static.json',dict(passed=True,artifacts=artifacts,regressions=regressions,
          inherited_modules_unchanged=True,pinned_latest_manifest_verified=True,private_live=live,
          CPU_resume=resume['anchor_sha256'],scorer_smoke_candidates=128,nbformat_valid=True,hard_wall_seconds=37800,
          GPU=False,push=False))
    print('P1 STATIC PASSED; private live created; GPU/push=0')


if __name__=='__main__':main()
