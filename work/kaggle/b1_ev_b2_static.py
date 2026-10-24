"""Compile and check assembled private kernels, frozen EV hashes and quota/recipe contracts."""
import ast,hashlib,json
from pathlib import Path
import nbformat
from build_l3 import ROOT,literals,write


def inspect(folder):
    m=json.loads((folder/'kernel-metadata.json').read_text());nb=nbformat.read(folder/m['code_file'],as_version=4);nbformat.validate(nb)
    codes=[c.source for c in nb.cells if c.cell_type=='code']
    for code in codes:compile(code,'notebook code','exec')
    values=literals(codes[0]);sources=values['SOURCES'];cfg=values['CONFIG']
    identities=json.loads((folder/'source-identity.json').read_text())['modules']
    assert {n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}==identities
    for n,s in sources.items():compile(s,n,'exec')
    assert m['is_private'] is True and m['enable_gpu'] is True and m['machine_shape']=='NvidiaTeslaT4'
    assert m['kernel_sources']==['thisarivson/nlp-hw02-compact-cache-verified/2']
    assert m['dataset_sources'][0]=='thisarivson/nlp-hw02-26b-inputs'
    nodes=ast.parse(codes[0]).body
    armed=next(n.lineno for n in nodes if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='WATCHDOG' for t in n.targets))
    imported=next(n.lineno for n in nodes if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
    assert armed<imported
    assert 'require_secrets=False' in codes[0] and cfg['live_sync_enabled'] is False
    limits=json.loads((folder/'session-limits.json').read_text())
    assert limits['session_timeout_seconds']==cfg['session_ceiling_seconds']
    return m,cfg,sources


def main():
    import sys
    sys.path.insert(0,str(ROOT/'src'))
    import b2_runtime as B
    prod,cfg,sources=inspect(ROOT/'bundles/b2_friend2')
    assert prod['id']=='friend2/nlp-hw02-b2' and not cfg['gpu_smoke']
    assert (cfg['main_seconds'],cfg['probe_cooldown_seconds'],cfg['probe_eval_seconds'])==(27000.,900.,600.)
    assert cfg['session_ceiling_seconds']==31500 and cfg['hard_shutdown_seconds']<31500
    assert cfg['worker_deadline_seconds']==31200
    quota=json.loads((ROOT.parent/'checks/b2-review-friend2-quota.json').read_text())
    assert quota['free_seconds']>=31500+1200
    assert B.PEAKS=={'W-lr1000':.001,'W-lr600':.0006} and B.PAIR_MARK==3100032
    assert all(s['qk_norm']=='layernorm' and s['d_model']==768 and s['encoder_layers']==s['decoder_layers']==6 for s in B.SPECS.values())
    assert B.new_state(32)['warmup_updates']==2000
    assert 'logsumexp' in sources['b2_optim.py'] and 'coefficient=1e-4' in sources['b2_optim.py']
    assert 'next_point_pairs' in sources['b2_runtime.py'] and 'from b2_optim import safe_update' in sources['b2_runtime.py']
    smoke,sc,ss=inspect(ROOT/'bundles/b2_gpu_smoke_friend1')
    assert smoke['id']=='friend1/nlp-hw02-b2-gpu-smoke' and sc['gpu_smoke'] and sc['session_ceiling_seconds']==900
    for n in B.IMPLEMENTATION:assert sources[n]==ss[n]
    access=json.loads((ROOT.parent/'checks/b2-smoke-inputs.json').read_text())
    assert access['owner']=='friend1' and all(r['accessible'] for r in access['inputs'])
    assert (ROOT/'src/qk_attention.py').read_bytes()==(ROOT.parent.parent/'submission/code/hw2_codegen/_qk_attention.py').read_bytes()
    ev,ec,es=inspect(ROOT/'bundles/b1_ev')
    assert ev['id']=='thisarivson/nlp-hw02-b1-ev' and ec['session_ceiling_seconds']==2700
    assert ec['resume_source']=='thisarivson/nlp-hw02-b1-ev-inputs'
    receipt=json.loads((ROOT.parent/'checks/b1-ev-transfer.json').read_text())
    assert receipt['passed'] and receipt['remote_SHA_verified'] and receipt['streaming_one_payload']
    assert ec['transfer_sha256']==receipt['transfer_sha256']
    pins=json.loads((ROOT.parent/'checks/b1-ev-runtime-pins.json').read_text())['implementation_sha256']
    assert all(hashlib.sha256(es[n].encode()).hexdigest()==h for n,h in pins.items())
    tree=ast.parse(es['b1_ev.py'])
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('main_step','run') for n in ast.walk(tree))
    assert 'cooldown_target_seconds=seconds' in es['b1_ev.py']
    assert json.loads((ROOT.parent/'checks/b1-ev-quota.json').read_text())['free_seconds']>=2700+300
    record=dict(passed=True,bundles=[prod['id'],smoke['id'],ev['id']],
                frozen_EV_runtime_SHA=True,EV_remote_SHA=True,B2_GPU_runs=0,B2_pushes=0,
                B2_hard_seconds=31500,B2_quota_margin_seconds=quota['free_seconds']-31500,
                B2_main_pair_mark=B.PAIR_MARK,EV_hard_seconds=2700)
    write(ROOT.parent/'checks/b1-ev-b2-static.json',record);print(json.dumps(record,indent=2))

if __name__=='__main__':main()
