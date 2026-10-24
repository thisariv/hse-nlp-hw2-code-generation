"""Compile notebooks/modules; verify frozen B2 pins, both private bundles and CPU gates."""
import ast, hashlib, json, re
from pathlib import Path
import nbformat
from build_l3 import ROOT, literals, write

def smoke_gate(modules, receipts):
    gpu=json.loads(receipts[-1].read_text()) if receipts else {}
    return bool(gpu.get('passed',False) and gpu.get('modules')==modules)


def inspect(smoke):
    folder=ROOT/'bundles'/('b3_gpu_smoke_thisarivson' if smoke else 'b3_thisarivson')
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    nb=nbformat.read(folder/meta['code_file'],as_version=4); nbformat.validate(nb)
    codes=[c.source for c in nb.cells if c.cell_type=='code']
    for code in codes: compile(code,'B3 notebook','exec')
    literal=literals(codes[0]); cfg=literal['CONFIG']; sources=literal['SOURCES']
    for n,s in sources.items(): compile(s,n,'exec')
    identities={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}
    assert identities==json.loads((folder/'source-identity.json').read_text())['modules']
    assert cfg['port_sha256']==hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    assert meta['id']=='thisarivson/'+re.sub('[^a-z0-9]+','-',meta['title'].lower()).strip('-')
    assert meta['id']==('thisarivson/nlp-hw02-b3-gpu-smoke' if smoke else 'thisarivson/nlp-hw02-b3')
    assert meta['is_private'] and meta['enable_internet'] and meta['enable_gpu'] and not meta['enable_tpu']
    assert meta['machine_shape']=='NvidiaTeslaT4' and cfg['owner']=='thisarivson'
    assert set(meta['dataset_sources'])=={'thisarivson/nlp-hw02-b2-states','thisarivson/nlp-hw02-26b-inputs'}
    assert meta['kernel_sources']==[cfg['cache_source']]==['thisarivson/nlp-hw02-compact-cache-verified/2']
    assert cfg['gpu_smoke']==smoke and cfg['session_ceiling_seconds']==(720 if smoke else 40800)
    assert cfg['worker_deadline_seconds']<cfg['hard_shutdown_seconds']<cfg['session_ceiling_seconds']
    assert cfg['live_sync_enabled'] is False and 'require_secrets=False' in codes[0]
    assert "'B2 resume runtime changed'" in codes[0] and "resume-runtime-gate.json" in codes[0]
    assert "MODULE_DIR/'b3_worker.py'" in codes[-1] and "assert CONFIG['resume_source']" in codes[-1]
    assert 'build_order(validated' not in codes[-1]
    assert 'assert report' not in codes[-1] and 'b3-publication.json' in codes[-1]
    assert "assert result['status']" not in sources['b3_worker.py']
    if smoke: assert cfg['smoke_updates']>=200
    limits=json.loads((folder/'session-limits.json').read_text())
    assert limits['session_timeout_seconds']==cfg['session_ceiling_seconds']
    tree=ast.parse(codes[0]).body
    armed=next(n.lineno for n in tree if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WATCHDOG' for t in n.targets))
    imported=next(n.lineno for n in tree if isinstance(n,ast.Import) and any(t.name=='torch' for t in n.names))
    assert armed<imported
    pins=json.loads((ROOT/'b3-anchors.json').read_text())
    assert cfg['anchors']==pins['anchors']
    for i,pin in pins['anchors'].items():
        assert pin['identity']['model']==('W-lr1000','W-lr600')[int(i)]
        for n,h in pin['identity']['recipe']['implementation_sha256'].items(): assert identities[n]==h,n
    assert cfg['main_seconds']==32400. and cfg['probe_main_marks']==[37800.,48600.]
    assert cfg['probe_cooldown_seconds']==900. and cfg['final_cooldown_seconds']==3000. and cfg['final_q']==0.
    receipt=json.loads((ROOT.parent/'checks/b3-transfer.json').read_text())
    assert receipt['passed'] and receipt['private'] and receipt['remote_SHA_verified'] and receipt['local_payload_deleted']
    assert cfg['transfer_sha256']==receipt['transfer_sha256']
    assert cfg['resume_source']==receipt['dataset']+'/'+str(receipt['version'])
    return dict(bundle=str(folder.relative_to(ROOT.parent)),id=meta['id'],hard_seconds=cfg['session_ceiling_seconds'],modules=identities)


def main():
    prod=inspect(False); smoke=inspect(True); assert prod['modules']==smoke['modules']
    access=json.loads((ROOT.parent/'checks/b3-input-access.json').read_text())
    assert access['passed'] and access['owner']=='thisarivson' and access['cache_private'] and access['dataset_private']
    checks={}
    for name in ('b3-cpu','b3-b2-regressions','b3-real-resume-gpu0','b3-real-resume-gpu1','b3-order'):
        row=json.loads((ROOT.parent/f'checks/{name}.json').read_text()); assert row['passed'],name; checks[name]=True
    pins=json.loads((ROOT/'b3-anchors.json').read_text())['anchors']
    for i in (0,1):
        row=json.loads((ROOT.parent/f'checks/b3-real-resume-gpu{i}.json').read_text())
        assert row['checkpoint_sha256']==pins[str(i)]['sha256'] and row['steps_bit_exact']==1 and row['effective_batch']==128
    write(ROOT.parent/'checks/b3-static.json',dict(passed=True,bundles=[prod,smoke],CPU_checks=checks,
                                                 frozen_B2_implementation_SHA=True,input_access=True,GPU_runs=0,pushes=0))
    smoke_receipts=sorted((ROOT.parent/'checks').glob('b3-review-gpu-smoke-v*.json'))
    passed=smoke_gate(prod['modules'],smoke_receipts)
    launch='KAGGLE_CONFIG_DIR=~/.kaggle kaggle kernels push -p bundles/b3_thisarivson --accelerator NvidiaTeslaT4 --timeout 40800'
    write(ROOT.parent/'results/next-b3.json',dict(bundle=prod['bundle'],smoke_bundle=smoke['bundle'],ready=bool(passed),
          owner='thisarivson',kernel='thisarivson/nlp-hw02-b3',earliest_launch_utc='2026-10-17T00:00:00Z',
          launch_by='user',production_GPU_runs=0,production_pushes=0,hard_wall_seconds=40800,main_cap_seconds_per_GPU=32400,
          probe_main_marks=[37800,48600],probe_cooldown_seconds=900,final_cooldown_seconds=3000,
          CPU_checks=checks,static_check='checks/b3-static.json',GPU_smoke_pending=not passed,smoke_passed=bool(passed),
          smoke_receipt=str(smoke_receipts[-1].relative_to(ROOT.parent)) if smoke_receipts else None,
          launch_command=launch,launch_working_directory='work/kaggle',
          ETA=dict(max_runtime_seconds=40800,basis='first code cell; queue/VM startup/output upload additional',
                   formula='actual first-cell start + up to 11h20m code; platform completion may be later',
                   platform_billing_cap_guaranteed=False,
                   example_start_utc='2026-10-17T05:00:00Z',example_finish_by_utc='2026-10-17T16:20:00Z',
                   example_start_msk='2026-10-17 08:00',example_finish_by_msk='2026-10-17 19:20'),
          launch_precondition='After 2026-10-17 00:00 UTC; verify free GPU quota >= 42000 seconds'))
    print('B3 CPU/static passed; smoke_passed='+str(bool(passed))+'; production push 0')


if __name__=='__main__': main()
