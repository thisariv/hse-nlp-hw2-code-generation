#!/usr/bin/env python3
"""Offline S2 bundle, immutable input/memmap, model-only load and quota request checks."""
import ast,json,sys,tempfile,zipfile
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))

def main():
    import numpy as np,torch
    import build,kaggle_tools as K,s2_runtime as S,exp26_core as C,exp26_eval as E
    from kaggle_io import materialize_dataset,write,sha
    from kaggle_runtime import prepare_starter
    import exp26_protocol as P
    bundle=ROOT/'bundles/s2';meta=json.loads((bundle/'kernel-metadata.json').read_text())
    nb=json.loads((bundle/meta['code_file']).read_text());config=None;sources=None
    for cell in nb['cells']:
        if cell['cell_type']!='code':continue
        compile(cell['source'],'notebook-cell','exec')
        for node in ast.parse(cell['source']).body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
                if node.targets[0].id=='CONFIG':config=ast.literal_eval(node.value)
                if node.targets[0].id=='SOURCES':sources=ast.literal_eval(node.value)
        assert cell['outputs']==[] and cell['execution_count'] is None
    assert config['budget_seconds']==11700 and config['terminate_grace_seconds']==20
    assert config['hard_shutdown_seconds']==12000 and config['session_ceiling_seconds']==13200
    assert meta['id']=='thisarivson/nlp-hw02-kaggle-s2' and meta['is_private'] and meta['enable_gpu']
    assert meta['machine_shape']=='NvidiaTeslaT4' and meta['kernel_sources']==['thisarivson/nlp-hw02-kaggle-s1/1']
    first=next(c['source'] for c in nb['cells'] if c['cell_type']=='code')
    assert 'WATCHDOG=Watchdog(CONFIG).start()' in first and 'import torch' not in first
    identity=json.loads((ROOT/'source-identity.json').read_text())
    import hashlib
    for name,digest in identity['modules'].items():assert hashlib.sha256(sources[name].encode()).hexdigest()==digest
    for name,source in sources.items():compile(source,name,'exec')
    assert config['live_dataset']=='thisarivson/nlp-hw02-s2-live'
    assert config['live_interval_seconds']==900 and config['live_staging_limit_bytes']==450000000
    assert not any(k in config for k in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'))
    live_cell=next(c['source'] for c in nb['cells'] if c['cell_type']=='code' and c['source'].startswith('from s2_live_sync import'))
    assert 'WATCHDOG.live_sync=LIVE' in live_cell
    for name in ('session_runner.py','session_watchdog.py','s2_live_sync.py'):
        for node in ast.walk(ast.parse(sources[name])):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('wait','join'):
                assert node.args or any(k.arg=='timeout' for k in node.keywords),(name,node.lineno)
    table=json.loads((ROOT.parent/'results/26b-final-summary.json').read_text())['table']
    assert config['targets']==[{'A_point':r['point'],'seconds':r['train_seconds_total'],'actual_pairs':round(r['actual_M_pairs']*1e6)} for r in table]
    commands=[]
    with patch.object(K,'call',side_effect=lambda args:commands.append(list(map(str,args)))), \
         patch.object(K,'probe',return_value={'exists':True,'private':True}), \
         patch.object(K,'versioned'):
        # Source metadata read by push is supplied in an isolated mock staging directory.
        with tempfile.TemporaryDirectory(prefix='s2-push-offline-') as t:
            fake=Path(t);path=fake/'staging/source-metadata/thisarivson-nlp-hw02-kaggle-s1-1'
            write(path/'kernel-metadata.json',{'is_private':True})
            with patch.object(K,'ROOT',fake):K.push(bundle,run_gpu=True)
    assert len(commands)==1 and commands[0][-2:]==['--accelerator','NvidiaTeslaT4']
    assert commands[0][commands[0].index('--timeout')+1]=='13200'
    with tempfile.TemporaryDirectory(prefix='s2-input-check-') as t:
        base=Path(t);working=base/'working';working.mkdir();scratch=base/'scratch'
        mounted=base/'input/cache';mounted.mkdir(parents=True)
        write(mounted/'cache-manifest.json',{'fixture':True});np.save(mounted/'source.npy',np.array([1,4,2],dtype=np.int16))
        direct=materialize_dataset(mounted,scratch/'cache','cache-manifest.json')
        assert direct==mounted and not (scratch/'cache').exists()
        array=np.load(direct/'source.npy',mmap_mode='r',allow_pickle=False)
        assert isinstance(array,np.memmap) and not array.flags.writeable;del array
        packed=base/'input/packed';packed.mkdir()
        with zipfile.ZipFile(packed/'payload.zip','w') as archive:
            for p in mounted.iterdir():archive.write(p,'cache/'+p.name)
        unpacked=materialize_dataset(packed,scratch/'fallback','cache-manifest.json')
        assert unpacked.is_relative_to(scratch) and not unpacked.is_relative_to(working)
        assert sha(unpacked/'source.npy')==sha(mounted/'source.npy')
        prepared=prepare_starter(ROOT/'staging/26b-inputs',scratch/'starter','training-inputs',read_only=True)
        for name,digest in P.INPUT_FILES.items():assert sha(prepared/name)==digest
        assert prepare_starter(prepared,scratch/'unused','training-inputs',read_only=True)==prepared
        assert not (scratch/'unused').exists() and not list(working.iterdir())
        S.install();spec={'d_model':16,'heads':2,'encoder_layers':1,'decoder_layers':1,'swiglu_dim':32,'vocab':8192,'max_positions':64}
        model=C.SimpleTransformer(**spec,dropout=0.).eval();cp=base/'weights.pt'
        torch.save({'format':'nlp-hw02-model-only-v1','model':model.state_dict(),'optimizer_included':False},cp)
        loaded=E.load_model(cp,torch.device('cpu'),dropout=0.,spec=spec)
        assert E.state_digest(loaded.state_dict())==E.state_digest(model.state_dict())
        # Partial tmp weights/checkpoints cannot inflate the finalized output.
        (working/'abandoned.pt.tmp').write_bytes(b'partial')
        root=working/'gpu0-exp29';(root/'checkpoints').mkdir(parents=True)
        (root/'checkpoints/latest.pt').write_bytes(b'latest fixture')
        (root/'checkpoints/orphan.pt').write_bytes(b'orphan fixture')
        write(root/'latest.json',{'file':'checkpoints/latest.pt'})
        from session_runner import archive
        archive(working,'s2');assert not (working/'abandoned.pt.tmp').exists() and not list(working.glob('*.zip'))
        assert (root/'checkpoints/latest.pt').exists() and not (root/'checkpoints/orphan.pt').exists()
    counts={}
    import exp28_capacity as B
    for name,spec in [('M',B.M_SPEC),('B',B.SPEC)]:
        with torch.device('meta'):model=C.SimpleTransformer(**spec,dropout=0.)
        counts[name]=sum(p.numel() for p in model.parameters())
    assert counts=={'M':15766464,'B':31940096}
    total=sum(counts.values())
    result={'status':'passed','notebook_sha256':sha(bundle/meta['code_file']),'port_sha256':config['port_sha'],
        'frozen_exp26_modules_verified':len(identity['modules']),'embedded_sources_compile':len(sources),
        'first_cell_watchdog_before_install':True,'budget_seconds':11700,'worker_deadline_seconds':11400,
        'hard_watchdog_seconds':12000,'backend_session_timeout_request_seconds':13200,
        'push_command_mock_checked':True,'live_secret_env_only_and_no_config_credentials':True,
        'live_watchdog_hook_and_all_wait_timeouts':True,'expanded_input_readonly_memmap':True,'archive_fallback_outside_working':True,
        'real_starter_SHA_verified':True,'model_only_loader_exact':True,'incomplete_and_orphan_checkpoint_cleanup':True,
        'parameter_counts':counts,'tensor_storage_estimate':{
            'latest_and_final_plus_four_FP32_points_and_initial_per_model_bytes':total*(2*12+4*4+4),
            'both_latest_in_branch_with_embedded_anchor_bytes':total*(2*12+4*4+4+12),
            'scope':'FP32 model and AdamW first/second moment; excludes JSON, generators, serialization overhead and transient writes'},
        'network_calls':0,'GPU_runs':0,'unverified':['Kaggle acceptance of timeout and output collection after emergency exit','real CUDA D-state','full-size GPU gates, speed and total output size']}
    write(ROOT.parent/'checks/kaggle-s2-retry-static-cpu-checks.json',result)
    print('PASSED S2 bundle, first-cell watchdog, input memmap/scratch, model-only load and13200s push request; no network')

if __name__=='__main__':main()
