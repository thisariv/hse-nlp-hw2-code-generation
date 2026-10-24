"""Compile/parity/metadata/budget/CPU gates. Writes CPU-ready handoff only."""
import ast,hashlib,json
from pathlib import Path
import nbformat
from build_l3 import ROOT,literals,write


def reviewed_handoff(base, previous, receipts):
    """A static rebuild cannot manufacture GPU success or erase a live launch."""
    same=previous.get('modules')==base['modules']
    for key in ('GPU_runs','pushes'):
        if key in previous:base[key]=previous[key]
    for key in ('production_url','production_version','pushed_at_utc',
                'started_at_utc','start_time_basis','ETA_utc','ETA_moscow','status','status_checked_at_utc',
                'smoke_url','smoke_status','smoke_pushed_at_utc','quota_before_production'):
        if same and key in previous:base[key]=previous[key]
    passed=next((r for r in reversed(receipts) if r.get('passed') is True and r.get('modules')==base['modules']),None)
    base.update(CPU_ready=True,ready=bool(passed),smoke_passed=bool(passed),GPU_smoke_pending=not bool(passed),
                readiness='Reviewed same-source GPU smoke passed' if passed else 'CPU/static passed; reviewer GPU smoke required')
    if passed:base['GPU_smoke_receipt']=passed.get('receipt')
    return base


def inspect(smoke):
    folder=ROOT/'bundles'/('b4_gpu_smoke_thisarivson' if smoke else 'b4_thisarivson')
    meta=json.loads((folder/'kernel-metadata.json').read_text());nb=nbformat.read(folder/meta['code_file'],as_version=4);nbformat.validate(nb)
    codes=[c.source for c in nb.cells if c.cell_type=='code']
    for c in codes:compile(c,'B4 notebook','exec')
    v=literals(codes[0]);cfg=v['CONFIG'];sources=v['SOURCES']
    modules={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}
    for n,s in sources.items():
        compile(s,n,'exec');p=ROOT/'src'/n if (ROOT/'src'/n).exists() else ROOT/n
        assert s==p.read_text(),n
    assert modules==json.loads((folder/'source-identity.json').read_text())['modules']
    assert cfg['port_sha256']==hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    slug='nlp-hw02-b4-gpu-smoke' if smoke else 'nlp-hw02-b4'
    assert meta['id']=='thisarivson/'+slug and meta['title']==slug
    assert meta['is_private'] is True and meta['enable_gpu'] is True and meta['enable_internet'] is True
    assert not meta['enable_tpu'] and meta['machine_shape']=='NvidiaTeslaT4'
    assert meta['dataset_sources']==['thisarivson/nlp-hw02-26b-inputs']
    assert meta['kernel_sources']==['thisarivson/nlp-hw02-b3','thisarivson/nlp-hw02-compact-cache-verified/2']
    assert cfg['owner']=='thisarivson' and not cfg['live_sync_enabled'] and 'require_secrets=False' in codes[0]
    pins=json.loads((ROOT/'b4-anchor.json').read_text());pin=pins['anchor']
    assert cfg['anchors']=={'0':pin,'1':pin}
    assert cfg['resume_source']==pins['source']=='thisarivson/nlp-hw02-b3/1'
    assert cfg['source_manifest_sha256']==pins['source_manifest_sha256']
    assert cfg['truncation_q_by_gpu']==[0.,.1]
    assert cfg['main_seconds']==32400 and cfg['probe_main_marks']==[pin['main_seconds']+10800.,pin['main_seconds']+21600.]
    assert cfg['probe_cooldown_seconds']==900 and cfg['final_cooldown_seconds']==3000
    assert cfg['probe_eval_seconds']==450 and cfg['final_eval_seconds']==1800
    assert cfg['session_ceiling_seconds']==(720 if smoke else 40800)
    if smoke:assert cfg['smoke_updates']>=200
    assert cfg['worker_deadline_seconds']<cfg['hard_shutdown_seconds']<cfg['session_ceiling_seconds']
    limits=json.loads((folder/'session-limits.json').read_text());assert limits['session_timeout_seconds']==cfg['session_ceiling_seconds']
    assert 'build_order(validated' not in codes[-1] and 'verify_mount' in codes[-1]
    assert "MODULE_DIR/'b4_worker.py'" in codes[-1] and 'from b4_session import launch' in codes[-1]
    assert 'assert report' not in codes[-1] and 'b4-publication.json' in codes[-1]
    assert "assert result['status']" not in sources['b4_worker.py']
    assert "Path(cfg['resume_root'])/'model-GPU1'" in sources['b4_worker.py']
    tree=ast.parse(codes[0]).body
    armed=next(n.lineno for n in tree if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WATCHDOG' for t in n.targets))
    imported=next(n.lineno for n in tree if isinstance(n,ast.Import) and any(t.name=='torch' for t in n.names));assert armed<imported
    for n,h in pin['identity']['recipe']['implementation_sha256'].items():assert modules[n]==h,n
    assert "from p1_math import truncation_mask" in sources['b4_math.py']
    assert "with patch.object(O,'update',two_pass_update)" in sources['b4_math.py']
    assert 'from b2_optim import update as retained_update' in sources['b4_math.py']
    assert 'save_on_cpu' not in sources['b4_math.py'] and 'autograd.Function' not in sources['b4_math.py']
    assert 'B5_uncooled_latest' in sources['b4_runtime.py']
    return dict(bundle=str(folder.relative_to(ROOT.parent)),kernel=meta['id'],hard_seconds=cfg['session_ceiling_seconds'],modules=modules)


def main():
    prod=inspect(False);smoke=inspect(True);assert prod['modules']==smoke['modules']
    checks={}
    for name in ('b4-inputs','b4-order','b4-real-resume','b4-cpu','b4-regressions'):
        rec=json.loads((ROOT.parent/'checks'/f'{name}.json').read_text());assert rec['passed'],name;checks[name]=True
    cpu=json.loads((ROOT.parent/'checks/b4-cpu.json').read_text())
    assert all(prod['modules'].get(n)==h for n,h in cpu['modules'].items()),'CPU receipt source changed'
    pins=json.loads((ROOT/'b4-anchor.json').read_text());pin=pins['anchor']
    real=json.loads((ROOT.parent/'checks/b4-real-resume.json').read_text())
    assert real['checkpoint_sha256']==pin['sha256'] and real['steps_bit_exact']==1 and real['effective_batch']==128
    inputs=json.loads((ROOT.parent/'checks/b4-inputs.json').read_text())
    assert inputs['stable_present_in_output'] and inputs['sha256']==pin['sha256'] and inputs['local_payload_deleted']
    write(ROOT.parent/'checks/b4-static.json',dict(passed=True,bundles=[prod,smoke],CPU_checks=checks,
          frozen_B2_implementation_SHA=True,production_smoke_module_parity=True,GPU_operations_performed_by_this_check=0))
    handoff=dict(bundle=prod['bundle'],smoke_bundle=smoke['bundle'],ready=False,
          readiness='CPU/static passed; reviewer GPU smoke and review still required before production',owner='thisarivson',
          kernel=prod['kernel'],source=pins['source'],anchor_sha256=pin['sha256'],CPU_checks=checks,static_check='checks/b4-static.json',
          GPU_smoke_pending=True,GPU_runs=0,pushes=0,launch_by='reviewer; user already authorized tonight',
          hard_wall_seconds=40800,main_cap_seconds_per_GPU=32400,probe_offsets_main_seconds=[10800,21600],
          probe_cooldown_seconds=900,final_cooldown_seconds=3000,quota_free_seconds_at_preparation=inputs['free_quota_seconds'],
          quota_reset_utc=inputs['quota_reset_utc'],smoke_hard_wall_seconds=720,
          launch_working_directory='work/kaggle',
          smoke_command='KAGGLE_CONFIG_DIR=~/.kaggle kaggle kernels push -p bundles/b4_gpu_smoke_thisarivson --accelerator NvidiaTeslaT4 --timeout 720',
          launch_command='KAGGLE_CONFIG_DIR=~/.kaggle kaggle kernels push -p bundles/b4_thisarivson --accelerator NvidiaTeslaT4 --timeout 40800',
          launch_precondition='Reviewer approval + successful same-source smoke; recheck free quota >= 42000 seconds',
          budget_basis='first code cell; queue/VM startup/output upload may add platform wall and billing time',
          modules=prod['modules'])
    path=ROOT.parent/'results/next-b4.json'
    previous=json.loads(path.read_text()) if path.exists() else {}
    receipts=[]
    for p in sorted((ROOT.parent/'checks').glob('b4-review-gpu-smoke-v*.json')):
        r=json.loads(p.read_text());r['receipt']=str(p.relative_to(ROOT.parent));receipts.append(r)
    handoff=reviewed_handoff(handoff,previous,receipts)
    write(path,handoff)
    print('B4 CPU/static passed; GPU smoke passed:',handoff['smoke_passed'])


if __name__=='__main__':main()
