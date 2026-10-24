"""CPU-only B7 readiness: exact B6 code/config delta, retained provenance and budget."""
import ast, hashlib, json
import nbformat
from build_l3 import ROOT, literals, write
from build_b7 import build
from b7_budget import calculate, HARD, WORKER, WATCHDOG
from b6_static_check import merge_review
from b5_sources import c_overrides


def inspect(smoke):
    path,cfg=build(smoke)
    folder=ROOT/'bundles'/('b7_gpu_smoke_friend1' if smoke else 'b7_friend1')
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    nb=nbformat.read(folder/meta['code_file'],as_version=4);nbformat.validate(nb)
    assert path.read_bytes()==(folder/meta['code_file']).read_bytes()
    codes=[c.source for c in nb.cells if c.cell_type=='code'];values=literals(codes[0])
    original=ROOT/'notebooks'/('nlp-hw02-b6-gpu-smoke.ipynb' if smoke else 'nlp-hw02-b6.ipynb')
    old=[c['source'] for c in json.loads(original.read_text())['cells'] if c['cell_type']=='code']
    prior=literals(old[0])['CONFIG']
    allowed={'session_kind','comparison_protocol','N_by_gpu','budget_seconds','worker_deadline_seconds','hard_shutdown_seconds','session_ceiling_seconds'}
    assert {k:v for k,v in cfg.items() if k not in allowed}=={k:v for k,v in prior.items() if k not in allowed}
    def strip(s):
        a=s.index('CONFIG=');b=s.index('WORKING=Path',a);return s[:a]+s[b:]
    assert strip(codes[0])==strip(old[0]) and codes[1:]==old[1:]
    overrides=next(ast.literal_eval(n.value) for n in ast.parse(codes[0]).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='C_OVERRIDES' for t in n.targets))
    assert overrides==c_overrides()
    for code in codes:compile(code,'B7 notebook','exec')
    for name,body in values['SOURCES'].items():
        compile(body,name,'exec');p=ROOT/'src'/name if (ROOT/'src'/name).exists() else ROOT/name
        assert body==p.read_text(),name
    identity=json.loads((folder/'source-identity.json').read_text())
    assert identity['modules']=={n:hashlib.sha256(s.encode()).hexdigest() for n,s in values['SOURCES'].items()}
    assert identity['C_overrides']=={n:hashlib.sha256(s.encode()).hexdigest() for n,s in overrides.items()}
    assert identity['config_sha256']==hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    assert cfg['port_sha256']==hashlib.sha256(json.dumps(dict(SOURCES=values['SOURCES'],C_OVERRIDES=overrides),sort_keys=True).encode()).hexdigest()
    pins=json.loads((ROOT/'b6-anchors.json').read_text());assert cfg['b6_pins']==pins and cfg['anchors']==pins['anchors']
    assert cfg['resume_source']=='friend1/nlp-hw02-b5/6'
    assert [pins['anchors'][g]['sha256'] for g in ('0','1')]==[
        '3b11360d5242e0e8b3fc1ac38254f0d4d544fba8bb62ee4c7bb034943e6cdc32',
        '69dff4405d2236097f48733adbfc9e3bfd5d20b501a3fcfd1dae62ec2ce63074']
    for g,pin in pins['anchors'].items():
        for n,h in pin['identity']['recipe']['implementation_sha256'].items():
            assert (identity['C_overrides'].get(n,identity['modules'][n]) if g=='1' else identity['modules'][n])==h
    slug='nlp-hw02-b7'+('-gpu-smoke' if smoke else '')
    assert meta['id']=='friend1/'+slug and meta['title']==slug
    assert meta['is_private'] is True and meta['enable_gpu'] is True and meta['enable_internet'] is True and not meta['enable_tpu']
    assert meta['machine_shape']=='NvidiaTeslaT4' and meta['dataset_sources']==[pins['starter_source']]
    assert meta['kernel_sources']==[pins['cache_source'],pins['source']]
    assert cfg['N_by_gpu']==({'0':200,'1':200} if smoke else calculate()['N_by_gpu'])
    assert cfg['session_ceiling_seconds']==(720 if smoke else HARD)
    assert cfg['worker_deadline_seconds']<cfg['hard_shutdown_seconds']<cfg['session_ceiling_seconds']
    assert cfg['main_seconds']==0. and cfg['final_q']==0. and cfg['save_midpoint'] and not cfg['live_sync_enabled']
    assert cfg['final_eval_seconds']==(90. if smoke else 1800.)
    assert 'require_secrets=False' in codes[0] and "MODULE_DIR/'b6_worker.py'" in codes[-1]
    assert 'from b6_resume import mount' in codes[-1] and 'build_order(' not in codes[-1]
    assert all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=='code')
    limits=json.loads((folder/'session-limits.json').read_text())
    assert limits['session_timeout_seconds']==cfg['session_ceiling_seconds'] and limits['watchdog_exit_seconds']==cfg['hard_shutdown_seconds']
    return dict(bundle_path=str(folder.relative_to(ROOT.parent)),kernel=meta['id'],**identity)


def main():
    path=ROOT.parent/'results/next-b7.json';previous=json.loads(path.read_text()) if path.exists() else {}
    write(path,dict(previous,ready=False,CPU_ready=False,readiness='CPU/static verification in progress'))
    prod=inspect(False);smoke=inspect(True)
    assert prod['modules']==smoke['modules'] and prod['C_overrides']==smoke['C_overrides']
    for name in ('b7-cpu','b7-regressions'):
        cpu=json.loads((ROOT.parent/'checks'/f'{name}.json').read_text())
        assert cpu['passed'] and not cpu['GPU']
        assert cpu['regressions'] is (name=='b7-regressions')
        assert cpu['script_sha256']==hashlib.sha256((ROOT/'b7_tests.py').read_bytes()).hexdigest()
        for n,h in cpu['scripts'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
        for n,h in cpu['modules'].items():
            if n in prod['modules']:assert prod['modules'][n]==h,('Stale CPU module',n)
    reused={}
    for name in ('b6-inputs','b6-real-resume-gpu0','b6-real-resume-gpu1','b6-final-anchor-gpu0','b6-final-anchor-gpu1'):
        p=ROOT.parent/'checks'/f'{name}.json';r=json.loads(p.read_text());assert r['passed'];reused[name]=hashlib.sha256(p.read_bytes()).hexdigest()
        for n,h in r.get('modules',{}).items():assert prod['modules'][n]==h,('Changed real-resume module',n)
        if 'real-resume' in name:
            g=name[-1];pin=json.loads((ROOT/'b6-anchors.json').read_text())['anchors'][g]
            assert r['checkpoint_sha256']==pin['sha256'] and r['steps_bit_exact']==2 and r['effective_batch']==128
            assert r['production_serializer'] and r['disk_resume_between_steps'] and r['opaque_CUDA_rng_preserved']
            assert r['script_sha256']==hashlib.sha256((ROOT/'b6_cpu_resume_check.py').read_bytes()).hexdigest()
            if g=='1':assert r['C_overrides']==prod['C_overrides']
    quota=json.loads((ROOT.parent/'checks/b7-quota.json').read_text());assert quota['free_seconds_conservative']>=HARD+720+600
    budget=calculate();budget.update(quota=quota,remaining_after_production_and_smoke_seconds=quota['free_seconds_conservative']-HARD-720,
        platform_overhead_reserve_seconds=600,quota_basis='Conservative min(user14.1h, read-only CLI remaining); fresh account quota required again by reviewer')
    pins=json.loads((ROOT/'b6-anchors.json').read_text())
    base=dict(ready=True,CPU_ready=True,readiness='CPU/static passed; independent review and same-source GPU smoke pending',
        owner='friend1',kernel=prod['kernel'],bundle_path=prod['bundle_path'],smoke_bundle_path=smoke['bundle_path'],
        GPU_runs=previous.get('GPU_runs',0),pushes=previous.get('pushes',0),GPU_smoke_pending=True,smoke_passed=False,
        modules=prod['modules'],C_overrides=prod['C_overrides'],production_config_sha256=prod['config_sha256'],smoke_config_sha256=smoke['config_sha256'],
        N_by_gpu=budget['N_by_gpu'],budget=budget,expected_duration_seconds=budget['expected_wall_seconds'],
        anchors=pins['anchors'],source=pins['source'],source_manifest_sha256=pins['source_manifest_sha256'],
        CPU_checks=dict(b7_full_N_and_resume=True,b6_regressions=True,static=True),reused_B6_receipts=reused,
        source_anchor_verification='Unchanged B6 source/code SHA; reuse verified B5 retained full states and B6 real batch128 CPU receipts; pre-final B5 PT pruned',
        pre_final_blobs_available=False,launch_by='independent reviewer; production already authorized after review and successful smoke',
        launch_precondition='review + successful same-source smoke + fresh quota >=40920s; inspect target before push',
        launch_working_directory='work/kaggle',
        launch_command='KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle kernels push -p bundles/b7_friend1 --accelerator NvidiaTeslaT4 --timeout 40800',
        smoke_command='KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle kernels push -p bundles/b7_gpu_smoke_friend1 --accelerator NvidiaTeslaT4 --timeout 720',
        guarded_launch_command='KAGGLE_CONFIG_DIR=~/.kaggle-friend python3 b7_review_api.py push friend1/nlp-hw02-b7 --path bundles/b7_friend1',
        guarded_smoke_command='KAGGLE_CONFIG_DIR=~/.kaggle-friend python3 b7_review_api.py push friend1/nlp-hw02-b7-gpu-smoke --path bundles/b7_gpu_smoke_friend1',
        smoke_audit_command='python3 b7_review_output.py <downloaded-smoke-output> --smoke --out ../checks/b7-review-gpu-smoke.json',
        inherited_output_names='b6 Worker/runtime/format, b6-final/b6-mid and b6-publication.json intentionally retained; identify B7 via kernel and session config')
    receipt=None
    if previous.get('GPU_smoke_receipt'):
        p=ROOT.parent/previous['GPU_smoke_receipt']
        if p.is_file():receipt=json.loads(p.read_text())
    write(ROOT.parent/'checks/b7-static.json',dict(passed=True,bundles=[prod,smoke],budget=budget,exact_B6_executable_code=True,reused_B6_receipts=reused,GPU_runs=0))
    write(path,merge_review(base,previous,receipt))
    print('B7 CPU/static passed; N33000/65800; GPU launch belongs to independent reviewer')


if __name__=='__main__':main()
