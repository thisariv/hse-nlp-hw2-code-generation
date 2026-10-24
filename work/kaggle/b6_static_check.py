"""Compile/source/SHA/mount/budget gates and a CPU-only handoff; never pushes kernels."""
import ast,hashlib,json,math
from pathlib import Path
import nbformat
from build_l3 import ROOT,literals,write
from build_b6 import build
from b5_sources import c_overrides


def merge_review(base,previous,receipt=None):
    same=all(previous.get(k)==base[k] for k in ('modules','C_overrides','production_config_sha256','smoke_config_sha256'))
    if same:
        for k in ('production_url','production_version','production_status','status','started_at_utc','ETA_utc',
                  'smoke_url','smoke_version','smoke_status'):
            if k in previous:base[k]=previous[k]
        if receipt and receipt.get('passed') is True and receipt.get('smoke') is True and all(receipt.get(k)==base[k] for k in ('modules','C_overrides')) and receipt.get('config_sha256')==base['smoke_config_sha256']:
            base.update(smoke_passed=True,GPU_smoke_pending=False,GPU_smoke_receipt=previous['GPU_smoke_receipt'],
                        readiness='CPU/static and reviewed same-source/config GPU smoke passed')
    return base


def inspect(smoke):
    _,cfg=build(smoke);folder=ROOT/'bundles'/('b6_gpu_smoke_friend1' if smoke else 'b6_friend1')
    meta=json.loads((folder/'kernel-metadata.json').read_text());nb=nbformat.read(folder/meta['code_file'],as_version=4);nbformat.validate(nb)
    codes=[c.source for c in nb.cells if c.cell_type=='code'];v=literals(codes[0]);assert v['CONFIG']==cfg
    overrides=next(ast.literal_eval(n.value) for n in ast.parse(codes[0]).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='C_OVERRIDES' for t in n.targets))
    assert overrides==c_overrides()
    for c in codes:compile(c,'B6 notebook','exec')
    for n,s in v['SOURCES'].items():
        compile(s,n,'exec');p=ROOT/'src'/n if (ROOT/'src'/n).exists() else ROOT/n;assert s==p.read_text(),n
    modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in v['SOURCES'].items()}
    hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in overrides.items()}
    identity=json.loads((folder/'source-identity.json').read_text())
    assert identity['modules']==modules and identity['C_overrides']==hashes
    assert identity['config_sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    assert cfg['port_sha256']==hashlib.sha256(json.dumps(dict(SOURCES=v['SOURCES'],C_OVERRIDES=overrides),sort_keys=True).encode()).hexdigest()
    pins=json.loads((ROOT/'b6-anchors.json').read_text());assert cfg['b6_pins']==pins and cfg['anchors']==pins['anchors']
    assert pins['source']=='friend1/nlp-hw02-b5/6' and cfg['resume_source']==pins['source']
    assert pins['anchors']['0']['sha256']=='3b11360d5242e0e8b3fc1ac38254f0d4d544fba8bb62ee4c7bb034943e6cdc32'
    assert pins['anchors']['1']['sha256']=='69dff4405d2236097f48733adbfc9e3bfd5d20b501a3fcfd1dae62ec2ce63074'
    for g,pin in pins['anchors'].items():
        for n,h in pin['identity']['recipe']['implementation_sha256'].items():assert (hashes.get(n,modules[n]) if g=='1' else modules[n])==h,(g,n)
    slug='nlp-hw02-b6-gpu-smoke' if smoke else 'nlp-hw02-b6'
    assert meta['id']=='friend1/'+slug and meta['title']==slug
    assert meta['is_private'] is True and meta['enable_gpu'] is True and meta['enable_internet'] is True and not meta['enable_tpu']
    assert meta['machine_shape']=='NvidiaTeslaT4'
    assert meta['dataset_sources']==[pins['starter_source']] and meta['kernel_sources']==[pins['cache_source'],pins['source']]
    assert cfg['owner']=='friend1' and cfg['main_seconds']==0. and cfg['final_q']==0. and not cfg['live_sync_enabled']
    assert cfg['N_by_gpu']==({'0':200,'1':200} if smoke else {g:b['N'] for g,b in pins['budget'].items()})
    assert cfg['session_ceiling_seconds']==(720 if smoke else 19200)
    assert cfg['worker_deadline_seconds']<cfg['hard_shutdown_seconds']<cfg['session_ceiling_seconds']
    assert cfg['checkpoint_seconds']==(90 if smoke else 900)
    assert 'require_secrets=False' in codes[0] and 'from b1_session import launch' in codes[-1]
    assert "MODULE_DIR/'b6_worker.py'" in codes[-1] and 'b6_resume import mount' in codes[-1]
    assert 'build_order(' not in codes[-1] and 'assert report' not in codes[-1]
    assert "src-C' if a.gpu==1" in v['SOURCES']['b6_worker.py']
    assert 'assert result' not in v['SOURCES']['b6_worker.py'] and 'main_step(' not in v['SOURCES']['b6_runtime.py']
    tree=ast.parse(codes[0]).body
    armed=next(n.lineno for n in tree if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WATCHDOG' for t in n.targets))
    imported=next(n.lineno for n in tree if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names));assert armed<imported
    assert all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=='code')
    assert json.loads((folder/'session-limits.json').read_text())['session_timeout_seconds']==cfg['session_ceiling_seconds']
    return dict(bundle_path=str(folder.relative_to(ROOT.parent)),kernel=meta['id'],modules=modules,C_overrides=hashes,config_sha256=identity['config_sha256'])


def main():
    handoff=ROOT.parent/'results/next-b6.json'
    previous=json.loads(handoff.read_text()) if handoff.exists() else {}
    # Invalidate readiness before any check which could fail.
    write(handoff,dict(previous,ready=False,CPU_ready=False,readiness='CPU/static verification in progress'))
    prod=inspect(False);smoke=inspect(True)
    assert prod['modules']==smoke['modules'] and prod['C_overrides']==smoke['C_overrides']
    names=['b6-inputs','b6-order-gpu0','b6-order-gpu1','b6-final-anchor-gpu0','b6-final-anchor-gpu1',
           'b6-real-resume-gpu0','b6-real-resume-gpu1','b6-cpu','b6-regressions']
    checks={n:json.loads((ROOT.parent/'checks'/f'{n}.json').read_text()) for n in names};assert all(r['passed'] for r in checks.values())
    pins=json.loads((ROOT/'b6-anchors.json').read_text())
    for name in ('b6-cpu','b6-regressions','b6-real-resume-gpu0','b6-real-resume-gpu1'):
        for n,h in checks[name]['modules'].items():
            if n in prod['modules']:assert prod['modules'][n]==h,('Stale CPU source',name,n)
    for name,file in (('b6-cpu','b6_tests.py'),('b6-regressions','b6_regressions.py')):
        assert checks[name]['script_sha256']==hashlib.sha256((ROOT/file).read_bytes()).hexdigest()
    for g in ('0','1'):
        r=checks['b6-real-resume-gpu'+g];pin=pins['anchors'][g]
        assert r['checkpoint_sha256']==pin['sha256'] and r['steps_bit_exact']==2 and r['effective_batch']==128
        assert r['final_anchor_metadata_exact'] and r['post_restore_generations_numerically_exact'] and r['inherited_main_state_unchanged'] and r['main_added_updates']==0
        assert r['production_serializer'] and r['disk_resume_between_steps'] and r['opaque_CUDA_rng_preserved']
        assert r['script_sha256']==hashlib.sha256((ROOT/'b6_cpu_resume_check.py').read_bytes()).hexdigest()
        assert checks['b6-order-gpu'+g]['N']==pins['budget'][g]['N']
        anchor=checks['b6-final-anchor-gpu'+g]
        assert anchor['sha256']==pins['predecessors'][g]['sha256'] and anchor['final_anchor']==pins['finals'][g]['anchor']
    assert checks['b6-real-resume-gpu1']['C_overrides']==prod['C_overrides']
    assert checks['b6-inputs']['local_payload_deleted'] and checks['b6-inputs']['source_manifest_sha256']==pins['source_manifest_sha256']
    inputs=checks['b6-inputs'];free=inputs['free_quota_seconds'];assert free>=20100
    budget=dict(quota_free_seconds=free,quota_reset_utc=inputs['quota_reset_utc'],per_gpu=pins['budget'],
        expected_wall_seconds=max(b['expected_wall_seconds'] for b in pins['budget'].values()),
        hard_wall_seconds=19200,worker_deadline_seconds=18900,watchdog_seconds=19140,smoke_hard_seconds=720,
        remaining_after_production_and_smoke_seconds=free-19200-720,
        quota_basis='Kaggle T4x2 session wall, as B5; platform startup/publication outside measured first-cell wall needs extra reserve')
    schedule={g:dict(N=b['N'],first_lr=b['peak_lr']*(1-math.sqrt(1/b['N'])),
        penultimate_lr=b['peak_lr']*(1-math.sqrt((b['N']-1)/b['N'])),last_lr=0.) for g,b in pins['budget'].items()}
    rec=dict(passed=True,bundles=[prod,smoke],CPU_checks={n:True for n in names},frozen_math_SHA=True,
             production_smoke_module_parity=True,schedule=schedule,GPU_runs=0)
    write(ROOT.parent/'checks/b6-static.json',rec)
    base=dict(ready=True,CPU_ready=True,bundle_path=prod['bundle_path'],smoke_bundle_path=smoke['bundle_path'],
        owner='friend1',kernel=prod['kernel'],GPU_runs=previous.get('GPU_runs',0),pushes=previous.get('pushes',0),
        GPU_smoke_pending=True,smoke_passed=False,readiness='CPU/static passed; independent review and same-source GPU smoke pending',
        modules=prod['modules'],C_overrides=prod['C_overrides'],production_config_sha256=prod['config_sha256'],
        smoke_config_sha256=smoke['config_sha256'],N_by_gpu={g:b['N'] for g,b in pins['budget'].items()},
        expected_duration_seconds=budget['expected_wall_seconds'],budget=budget,anchors=pins['anchors'],
        source=pins['source'],source_manifest_sha256=pins['source_manifest_sha256'],CPU_checks={n:True for n in names},
        source_anchor_verification='Exact final-anchor metadata + SHA/bytes and numeric equality of both retained post-restore generations; pre-final PT files were pruned by B5',
        pre_final_blobs_available=False,
        launch_by='independent reviewer; production already authorized after review and successful smoke',
        launch_precondition='review + successful same-source smoke + fresh quota >= 19500s; inspect current target before push',
        launch_working_directory='work/kaggle',
        launch_command='KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle kernels push -p bundles/b6_friend1 --accelerator NvidiaTeslaT4 --timeout 19200',
        smoke_command='KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle kernels push -p bundles/b6_gpu_smoke_friend1 --accelerator NvidiaTeslaT4 --timeout 720',
        guarded_launch_command='KAGGLE_CONFIG_DIR=~/.kaggle-friend python b6_review_api.py push friend1/nlp-hw02-b6 --path bundles/b6_friend1',
        guarded_smoke_command='KAGGLE_CONFIG_DIR=~/.kaggle-friend python b6_review_api.py push friend1/nlp-hw02-b6-gpu-smoke --path bundles/b6_gpu_smoke_friend1')
    receipt=None
    if previous.get('GPU_smoke_receipt'):
        p=ROOT.parent/previous['GPU_smoke_receipt']
        if p.is_file():receipt=json.loads(p.read_text())
    write(handoff,merge_review(base,previous,receipt))
    print('B6 CPU/static passed; GPU launch belongs to independent reviewer')


if __name__=='__main__':main()
