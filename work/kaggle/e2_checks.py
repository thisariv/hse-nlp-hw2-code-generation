"""Verify E2 embedded modules, mount fallback, and the actual CPU pool worker."""
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import torch

ROOT=Path(__file__).resolve().parent; WORK=ROOT.parent


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    bundle=ROOT/'bundles/e2_mine'; identity=read(bundle/'source-identity.json')
    nb=read(bundle/'nlp-hw02-e2-policy.ipynb')
    assert sha(bundle/'nlp-hw02-e2-policy.ipynb')==identity['notebook_sha256']
    init=''.join(nb['cells'][1]['source'])
    sources=ast.literal_eval(next(n.value for n in ast.parse(init).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='SOURCES' for t in n.targets)))
    assert {n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources.items()}==identity['source_sha256']
    for cell in nb['cells']:
        if cell['cell_type']=='code': compile(''.join(cell['source']), 'E2 cell', 'exec')
    import nbformat
    nbformat.validate(nbformat.read(bundle/'nlp-hw02-e2-policy.ipynb',as_version=4))
    metadata=read(bundle/'kernel-metadata.json')
    assert metadata['id']=='thisarivson/nlp-hw02-e2-policy'
    assert metadata['is_private'] and not metadata['enable_internet'] and not metadata['kernel_sources']
    assert all(ref.startswith('thisarivson/') for ref in metadata['dataset_sources'])
    assert 'friend2' not in json.dumps(nb)
    assert identity['config']['selection_sha256']==sha(WORK/'results/e2-selection.json')
    unchanged={n:d for n,d in identity['source_sha256'].items() if n!='e2_worker.py'}
    assert all(identity['e1_source_sha256'][n]==d for n,d in unchanged.items())
    config=identity['config']
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='e2-cpu-') as d:
        tmp=Path(d); modules=tmp/'src'; modules.mkdir()
        for name,s in sources.items(): (modules/name).write_text(s)
        sys.path.insert(0,str(modules))
        import final_core as C
        from final_session import weight_path
        from final_mounts import resolve_cache_mount
        specs=[]
        for i,spec in enumerate(config['models']):
            arch=dict(d_model=16 if i<2 else 24,heads=2,encoder_layers=1,decoder_layers=1,
                      swiglu_dim=32,vocab=8195,max_positions=512,qk_norm='layernorm' if i==2 else False)
            m=C.SimpleTransformer(**arch,dropout=0.).eval()
            with torch.no_grad(): m.head.weight.zero_(); m.head.bias.fill_(-1000); m.head.bias[2]=1000
            path=tmp/(str(i)+'.pt')
            torch.save(dict(format='nlp-hw02-model-only-v1',optimizer_included=False,model_spec=arch,model=m.state_dict()),path)
            specs.append(dict(spec,path=str(path),sha256=sha(path)))
            mount=tmp/'input'/spec['dataset'].split('/')[1]
            target=mount/spec['relative_path']; target.parent.mkdir(parents=True,exist_ok=True); target.touch()
            found=resolve_cache_mount(tmp/'input',spec['dataset'],spec['source_kind'])
            assert Path(weight_path(found,spec))==target
        rows=tmp/'heldout4k.json'; rows.write_text(json.dumps([{'question':'Compute 2 + 3.'},{'question':'What is 7 minus 2?'}]))
        refs=tmp/'refs.json'; refs.write_text(json.dumps({'rows':[{'status':'numeric','value':5}]*2}))
        token=WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json'
        cfg=dict(config,models=specs,cpu_fixture=True,fixture_rows=2,smoke_rows=1,cap=8,
                 single_lanes=32,output=str(tmp/'e2'),inputs={'heldout4k.json':str(rows),
                 'heldout4k-reference-execution.json':str(refs),'tokenizer.json':str(token)})
        p=tmp/'config.json'; p.write_text(json.dumps(cfg))
        run=subprocess.run([sys.executable,str(modules/'e2_worker.py'),str(p),'tune2048','smoke'],
            capture_output=True,text=True,timeout=60,env=dict(os.environ,E1_DEADLINE=str(time.monotonic()+55)))
        assert run.returncode==0,run.stderr+run.stdout
        folder=tmp/'e2/smoke/tune2048'; summary=read(folder/'summary.json')
        assert summary['status']=='complete' and summary['candidates_per_row']==75
        assert not summary['cross_LL'] and not summary['token_ensemble'] and not summary['soups']
        assert all(t['kind']=='generation' for t in summary['timings'])
        with gzip.open(folder/'pool-00000.jsonl.gz','rt') as f: record=json.loads(f.readline())
        assert len(record['candidates'])==1 and len(record['pools'])==9
        assert record['candidates'][0]['ll']=={}
        assert all(c['execution']['status']!='budget_not_evaluated' for c in record['candidates'])
        assert all(len(record['pools'][m+'/0.7'])==16 and len(record['pools'][m+'/1.0'])==8 for m in ('C','P','W'))
        result=dict(passed=True,notebook_sha256=identity['notebook_sha256'],unchanged_E1_modules=len(unchanged),
                    mount_paths=True,actual_worker=True,executor=True,gzip_roundtrip=True,
                    pools=9,candidates_per_row=75,no_LL_token_ensemble_soups=True,seconds=summary['seconds'])
    (WORK/'checks/e2-pipeline.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
