"""Pre-push CPU gates: five strict weight loads, E2 replay, decoder and cold Run All."""
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import warnings
from variants_common import *

sys.path.insert(0,str(SUBMISSION/'code'))
import torch
from hw2_codegen.pipeline import load_model
from hw2_codegen.vote import replay_policy, slot_temperatures, winner_fraction
from hw2_codegen.decode import Context, Options, decode
from hw2_codegen.dry_run import fixtures


def journal_samples(row, model, count, mixed=False):
    used=Counter();result=[]
    for slot in range(1,count+1):
        key=model+('/1.0' if mixed and slot%4==0 else '/0.7')
        entry=row['pools'][key][used[key]];used[key]+=1
        result.append(dict(row['candidates'][entry['u']],mean_logprob=entry['mean_logprob']))
    return result


def main():
    torch.set_num_threads(1)
    weight_checks=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for m in models().values():
            path=WORK/m['local_path'];assert sha(path)==m['sha256']
            payload=torch.load(path,map_location='cpu',weights_only=True)
            spec=payload['model_spec'];wide=m['name'].startswith('W')
            assert spec['encoder_layers']==spec['decoder_layers']==6
            assert spec['d_model']==(768 if wide else 512)
            assert bool(spec.get('qk_norm'))==wide
            assert all(t.dtype==torch.float16 for t in payload['model'].values() if t.is_floating_point())
            del payload
            model=load_model(path,torch.device('cpu'))
            weight_checks.append(dict(name=m['name'],sha256=m['sha256'],model_spec=spec,strict_load=True,fp16=True))
            del model
    cfgs=variants()
    assert len(cfgs)==5
    for n,c in enumerate(cfgs,1):
        assert c['variant']==f'v{n}' and c['sequence_budget']==c['greedy_sequence_budget']==640
        assert c['guards']==dict(generation=160.,execution=166.,verification=170.,supervisor=172.,greedy=130.)
        assert c['seed']==270042 and c['temperature']==.7
        assert c['release_cache_between_batches'] and c['cuda_allocator']=='expandable_segments:True,garbage_collection_threshold:0.8'
        assert [m['K1'] for m in c['models']]==[1,1,1]
        assert [m['K2'] for m in c['models']]==([2,8,8] if n==4 else [1,8,8])
    # Reproduce the sealed independent E2 oracle on every saved row.
    raw=WORK/read(WORK/'checks/e2-download.json')['output']
    with gzip.open(raw/'e2-comparison.jsonl.gz','rt') as f:expected={r['id']:r for r in map(json.loads,f)}
    specs=[dict(name=m,K1=1,K2=k2,K3=k3,mixed_temperature=m!='C') for m,k2,k3 in zip('CPW',[2,8,8],[2,16,8])]
    rows=early=third=0
    for split in ['tune2048','eval2048']:
        folder=raw/'e2/full'/split
        for entry in read(folder/'summary.json')['journal_manifest']:
            path=folder/entry['file'];assert sha(path)==entry['sha256']
            with gzip.open(path,'rt') as f:
                for row in map(json.loads,f):
                    e=row['pools']['C/g'][0];anchor=dict(row['candidates'][e['u']],mean_logprob=0.)
                    samples={m['name']:journal_samples(row,m['name'],m['K3'],m['mixed_temperature']) for m in specs}
                    winner,stop,expand,_=replay_policy(anchor,samples,specs,e2_primary=True)
                    exp=expected[row['id']]
                    assert (winner['code'],expand)==(exp['primary'],exp['stage3'])
                    ref=row['reference'];correct=ref['status']==winner['execution']['status']=='numeric' and math.isclose(ref['value'],winner['execution']['value'],rel_tol=1e-6,abs_tol=1e-6)
                    assert correct==exp['primary_correct']
                    early+=stop;third+=expand;rows+=1
    assert (rows,early,third)==(4096,2545,826)
    assert slot_temperatures(specs[1],2,8)==[.7,.7,1.,.7,.7,.7,1.]
    assert slot_temperatures(specs[1],9,16)==[.7,.7,.7,1.,.7,.7,.7,1.]
    # Scalar temperature path remains bit-exact when vector slots are equal,
    # including compaction. Untempered log probabilities are preserved.
    tiny_rows,tiny_models,tok,table,_=fixtures()
    model=next(iter(tiny_models.values()));src=torch.tensor([[4,5,6],[4,5,6]])
    opts=Options(eos_every=1)
    torch.manual_seed(13);a=decode(Context(model,src,32),k=4,temperature=.7,options=opts)
    torch.manual_seed(13);b=decode(Context(model,src,32),k=4,temperature=[.7]*4,options=opts)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
    # Exercise the complete notebook's ten NEW worker processes on tiny data.
    nb=read(BUNDLE/'variants.ipynb')
    import nbformat
    nbformat.validate(nbformat.from_dict(nb))
    for code in [c['source'] for c in nb['cells'] if c['cell_type']=='code']:compile(code,'variants.ipynb','exec')
    for path in (SUBMISSION/'code/hw2_codegen').glob('*.py'):compile(path.read_text(),str(path),'exec')
    with tempfile.TemporaryDirectory(prefix='hw2-variants-cpu-') as temp:
        script=Path(temp)/'run.py';script.write_text('\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code'))
        env=dict(os.environ,HW2_VARIANTS_DRY_RUN='1',HW2_VARIANTS_ROOT=temp,CUDA_VISIBLE_DEVICES='')
        subprocess.run([sys.executable,str(script)],env=env,check=True,timeout=180)
        summary=read(Path(temp)/'variants/summary.json')
        assert len(summary['variants'])==5
        for variant,r in summary['variants'].items():
            for split in ['test','tune']:
                assert r[split]['rows']==8 and not r[split]['guards']
                assert not r[split]['partial_rows']
                if variant=='v4':assert len(r[split]['third_rows'])==8
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(SUBMISSION/'tests'),'-p','test_*.py'],
        env=dict(os.environ,PYTHONPATH=str(SUBMISSION/'code')),check=True,timeout=100)
    identity=read(WORK/'checks/final-variants-build.json')
    assert sha(SUBMISSION/'inference.ipynb')==identity['submission_notebook_sha256']
    out=dict(passed=True,weights=weight_checks,variants=len(cfgs),e2_replay_rows=rows,
        e2_early_stops=early,e2_third_rows=third,e2_winner_code_correctness_stage3_exact=True,
        scalar_vector_temperature_exact=True,notebook_cpu_cold_processes=10,regression_tests=21,
        notebook_sha256=sha(BUNDLE/'variants.ipynb'),submission_notebook_unchanged=True)
    write(WORK/'checks/final-variants-cpu.json',out)
    print(json.dumps(out))

if __name__=='__main__':main()
