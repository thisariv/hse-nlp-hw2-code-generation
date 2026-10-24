"""Compile/source parity, independent mount pins, budgets and CPU-only reviewer handoff."""
import ast,hashlib,json
from pathlib import Path
import nbformat
from build_l3 import ROOT,literals,write
from build_b5 import build


def reviewed_handoff(base, previous, receipts):
    """Keep real launch history; GPU approval requires both source namespaces."""
    same=all(previous.get(k)==base[k] for k in ('modules','C_overrides'))
    for key in ('GPU_runs','pushes'):
        if key in previous:base[key]=previous[key]
    passed=next((r for r in reversed(receipts) if r.get('passed') is True
                 and all(r.get(k)==base[k] for k in ('modules','C_overrides'))),None)
    base.update(smoke_passed=bool(passed),GPU_smoke_pending=not bool(passed),
                readiness='Reviewed same-source GPU smoke passed' if passed else base['readiness'])
    if passed:base['GPU_smoke_receipt']=passed['receipt']
    if same:
        for key in ('production_url','production_version','pushed_at_utc','started_at_utc','start_time_basis',
                    'ETA_utc','ETA_moscow','status','status_checked_at_utc','smoke_url','smoke_status',
                    'smoke_pushed_at_utc','quota_before_production','reviewer_smoke_push_versions',
                    'external_smoke_version_observed','external_smoke_status_observed',
                    'reviewer_production_url','reviewer_production_pushed_at_utc','production_launch_origin',
                    'production_backend_timeout_unverified','production_backend_timeout_verified_for_version',
                    'production_runtime_ceiling_seconds','production_retry_blocked_by_running_version','observed_production_versions'):
            if key in previous:base[key]=previous[key]
    return base


def inspect(smoke):
    build(smoke)
    folder=ROOT/'bundles'/('b5_gpu_smoke_friend1' if smoke else 'b5_friend1')
    meta=json.loads((folder/'kernel-metadata.json').read_text());nb=nbformat.read(folder/meta['code_file'],as_version=4);nbformat.validate(nb)
    codes=[c.source for c in nb.cells if c.cell_type=='code'];v=literals(codes[0]);cfg=v['CONFIG'];sources=v['SOURCES']
    overrides=next(ast.literal_eval(n.value) for n in ast.parse(codes[0]).body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='C_OVERRIDES' for t in n.targets))
    from b5_sources import c_overrides
    assert overrides==c_overrides()
    for name,source in overrides.items():compile(source,name,'exec')
    c_hashes={n:hashlib.sha256(s.encode()).hexdigest() for n,s in overrides.items()}
    for c in codes:compile(c,'B5 notebook','exec')
    for n,s in sources.items():
        compile(s,n,'exec');p=ROOT/'src'/n if (ROOT/'src'/n).exists() else ROOT/n;assert s==p.read_text(),n
    modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}
    assert modules==json.loads((folder/'source-identity.json').read_text())['modules']
    assert c_hashes==json.loads((folder/'source-identity.json').read_text())['C_overrides']
    assert cfg['port_sha256']==hashlib.sha256(json.dumps(dict(SOURCES=sources,C_OVERRIDES=overrides),sort_keys=True).encode()).hexdigest()
    slug='nlp-hw02-b5-gpu-smoke' if smoke else 'nlp-hw02-b5'
    assert meta['id']=='friend1/'+slug and meta['title']==slug
    assert meta['is_private'] is True and meta['enable_gpu'] is True and meta['enable_internet'] is True
    assert meta['machine_shape']=='NvidiaTeslaT4' and meta['enable_tpu'] is False
    pins=json.loads((ROOT/'b5-anchors.json').read_text());transfer=dict(version=pins['transfer_version'],transfer_sha256=pins['transfer_sha256'])
    assert cfg['owner']=='friend1' and cfg['anchors']==pins['anchors'] and cfg['resume_sources']==pins['sources']
    assert cfg['anchors']['0']['sha256']=='f23fe1e36135e23b1ba1d393537ea501f25165a084b8462c16d8bc24843abc7a'
    assert cfg['anchors']['1']['sha256']=='48f2ba3fa4ffe6a722b45a084d3ee33a27cc4ed02b72433605c9cb1740f5a80b'
    assert cfg['anchors']['0']['identity']['order_sha256']!=cfg['anchors']['1']['identity']['order_sha256']
    assert cfg['transfer_sha256']==transfer['transfer_sha256']
    assert meta['dataset_sources']==[pins['starter_source'],pins['transfer_dataset']+'/'+str(transfer['version']),pins['sources']['1']]
    assert meta['kernel_sources']==[pins['cache_source']]
    assert cfg['main_seconds']==33840 and cfg['probe_offset_main_seconds']==16200
    assert cfg['probe_cooldown_seconds']==900 and cfg['final_cooldown_seconds']==3000 and cfg['final_q']==0.
    assert cfg['probe_eval_seconds']==300 and cfg['final_eval_seconds']==1800
    assert not cfg['live_sync_enabled'] and 'require_secrets=False' in codes[0]
    assert cfg['checkpoint_seconds']==900
    assert cfg['worker_deadline_seconds']<cfg['hard_shutdown_seconds']<cfg['session_ceiling_seconds']<=40800
    assert cfg['session_ceiling_seconds']==(720 if smoke else 40800)
    assert cfg['gpu_smoke'] is smoke and not cfg.get('max_updates')
    if smoke:assert cfg['smoke_updates']>=200
    assert 'from b1_session import launch' in codes[-1] and "MODULE_DIR/'b5_worker.py'" in codes[-1]
    assert 'build_order(' not in codes[-1] and 'restore_order(live' in codes[-1] and 'c_mount(live' in codes[-1]
    assert 'assert report' not in codes[-1] and "assert result['status']" not in sources['b5_worker.py']
    assert 'weighted-order-GPU' in sources['b5_worker.py'] and "cfg['resume_roots'][str(a.gpu)]" in sources['b5_worker.py']
    assert 'b5_session' not in sources['b5_worker.py']  # No erroneous common-start audit from B4.
    assert 'B6_uncooled_latest' in sources['b5_runtime.py']
    assert 'truncation' not in sources['b5_state.py'].split('def main_step')[1]
    assert "src-C' if a.gpu==1" in sources['b5_worker.py'] and 'MODULE_C_DIR=WORKING' in codes[0]
    for gpu,pin in pins['anchors'].items():
        for n,h in pin['identity']['recipe']['implementation_sha256'].items():
            assert (c_hashes.get(n,modules[n]) if gpu=='1' else modules[n])==h,(gpu,n)
    tree=ast.parse(codes[0]).body
    armed=next(n.lineno for n in tree if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WATCHDOG' for t in n.targets))
    imported=next(n.lineno for n in tree if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names));assert armed<imported
    limits=json.loads((folder/'session-limits.json').read_text());assert limits['session_timeout_seconds']==cfg['session_ceiling_seconds']
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in nb.cells if c.cell_type=='code')
    return dict(bundle_path=str(folder.relative_to(ROOT.parent)),kernel=meta['id'],modules=modules,C_overrides=c_hashes)


def main():
    prod=inspect(False);smoke=inspect(True);assert prod['modules']==smoke['modules'] and prod['C_overrides']==smoke['C_overrides']
    names=['b5-inputs','b5-transfer','b5-order-gpu0','b5-order-gpu1','b5-real-resume-gpu0','b5-real-resume-gpu1','b5-cpu','b5-regressions']
    receipts={n:json.loads((ROOT.parent/'checks'/f'{n}.json').read_text()) for n in names}
    assert all(r['passed'] for r in receipts.values())
    for name in ('b5-cpu','b5-real-resume-gpu0','b5-real-resume-gpu1'):
        assert all(prod['modules'][n]==h for n,h in receipts[name]['modules'].items()),'Stale CPU source'
    pins=json.loads((ROOT/'b5-anchors.json').read_text())
    assert receipts['b5-transfer']['version']==pins['transfer_version'] and receipts['b5-transfer']['transfer_sha256']==pins['transfer_sha256']
    for gpu in ('0','1'):
        rec=receipts['b5-real-resume-gpu'+gpu];pin=pins['anchors'][gpu]
        assert rec['checkpoint_sha256']==pin['sha256'] and rec['steps_bit_exact']==2 and rec['effective_batch']==128
        assert rec['disk_resume_between_steps'] and rec['production_serializer'] and rec['opaque_CUDA_rng_preserved']
        assert rec['script_sha256']==hashlib.sha256((ROOT/'b5_cpu_resume_check.py').read_bytes()).hexdigest()
    assert receipts['b5-real-resume-gpu1']['C_overrides']==prod['C_overrides']
    inputs=receipts['b5-inputs'];assert inputs['local_payload_deleted'] and receipts['b5-transfer']['local_payload_deleted']
    budget=dict(quota_free_seconds=inputs['free_quota_seconds'],quota_reset_utc=inputs['quota_reset_utc'],
       hard_wall_seconds=40800,main_cap_seconds_per_GPU=33840,probe_offsets_main_seconds=[16200],probe_cooldown_seconds=900,
       probe_eval_cap_seconds=300,final_cooldown_seconds=3000,final_eval_cap_seconds=1800,setup_cap_seconds=480,
       stable_period_seconds=900,smoke_hard_seconds=720,remaining_after_one_production_and_smoke_seconds=inputs['free_quota_seconds']-40800-720,
       nominal_compute_setup_eval_seconds=480+33840+900+300+3000+1800,
       basis='session wall from first code cell; main cap may shorten to preserve final/evaluation/I/O; platform startup/publication overhead excluded')
    rates={'0':38517/(91801.75519380573-59400.86158832819),'1':(209144-123842)/(79850.42887455746-46370.24584266532)}
    expected={}
    for gpu,rate in rates.items():
        pin=pins['anchors'][gpu];updates=int(rate*33840);pairs=updates*128
        expected[gpu]=dict(model=pin['identity']['model'],reference_updates_per_second=rate,added_updates=updates,
            added_pairs=pairs,total_pairs=pin['updates']*128+pairs,order_cursor=pin['cursor']+pairs,
            source='B4 plain throughput' if gpu=='0' else 'L3 C-long2x throughput',estimate=True)
    write(ROOT.parent/'checks/b5-static.json',dict(passed=True,bundles=[prod,smoke],CPU_checks={n:True for n in names},
         production_smoke_module_parity=True,frozen_math_SHA=True,GPU_runs=0))
    handoff=ROOT.parent/'results/next-b5.json'
    previous=json.loads(handoff.read_text()) if handoff.exists() else {}
    gpu_receipts=[]
    for path in sorted((ROOT.parent/'checks').glob('b5-review-gpu-smoke-v*.json')):
        receipt=json.loads(path.read_text());receipt['receipt']=str(path.relative_to(ROOT.parent));gpu_receipts.append(receipt)
    base=dict(bundle_path=prod['bundle_path'],smoke_bundle_path=smoke['bundle_path'],
        ready=True,CPU_ready=True,readiness='CPU/static passed; independent review and same-source GPU smoke pending',
        owner='friend1',kernel=prod['kernel'],GPU_runs=0,pushes=0,GPU_smoke_pending=True,
        launch_by='independent reviewer; user already authorized production after review and successful smoke',
        launch_precondition='review + successful same-source smoke + quota recheck >= 42000 seconds',
        budget=budget,expected_pairs=expected,anchors=pins['anchors'],transfer=receipts['b5-transfer'],
        CPU_checks={n:True for n in names},static_check='checks/b5-static.json',modules=prod['modules'],C_overrides=prod['C_overrides'],
        launch_working_directory='work/kaggle',
        smoke_command='KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle kernels push -p bundles/b5_gpu_smoke_friend1 --accelerator NvidiaTeslaT4 --timeout 720',
        launch_command='KAGGLE_CONFIG_DIR=~/.kaggle-friend kaggle kernels push -p bundles/b5_friend1 --accelerator NvidiaTeslaT4 --timeout 40800')
    write(handoff,reviewed_handoff(base,previous,gpu_receipts))
    print('B5 CPU/static passed; this check launches no GPU and preserves reviewer history')


if __name__=='__main__':main()
