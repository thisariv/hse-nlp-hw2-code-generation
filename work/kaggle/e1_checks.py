"""CPU invariants + real pinned weights + actual embedded worker/executor."""
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
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent; WORK = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import torch
import final_core as C
import final_fast as Fast
import final_speed as Speed
from e1_decode import decode, teacher_score, soup_state, log_probs
from build_e1 import BUNDLE, NAME


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@torch.inference_mode()
def slow(models, src, cap):
    memories = [m.transformer.encoder(m.embed(src), src_key_padding_mask=src.eq(C.PAD)) for m in models]
    prefix = torch.full((len(src), 1), C.BOS, dtype=torch.long)
    done = torch.zeros(len(src), dtype=torch.bool)
    total = torch.zeros(len(src)); lengths = torch.zeros(len(src), dtype=torch.long)
    result = []
    for _ in range(cap):
        lps = [log_probs(m.head(m.decode_prefix(prefix, memory, src.eq(C.PAD))[:, -1]))
               for m, memory in zip(models, memories)]
        pooled = torch.stack(lps).mean(0)
        lp = torch.log_softmax(pooled, -1)
        pick = pooled.argmax(-1)
        total += torch.where(done, 0., lp.gather(-1, pick[:, None]).squeeze(-1))
        lengths += ~done
        token = pick.masked_fill(done, C.PAD)
        result.append(token); done |= token.eq(C.EOS)
        prefix = torch.cat([prefix, token[:, None]], -1)
        if bool(done.all()): break
    return torch.stack(result, -1)[:, None], total[:, None], lengths[:, None]


def unit():
    torch.manual_seed(419)
    src = torch.tensor([[1,5,6,2,0], [1,9,8,7,2], [1,7,2,0,0]])
    models = [C.SimpleTransformer(d_model=16 if i<2 else 24, heads=2, encoder_layers=1,
        decoder_layers=1, swiglu_dim=32, vocab=32, max_positions=32, dropout=0.,
        qk_norm='layernorm' if i==2 else False).eval() for i in range(3)]
    # Genuine full-prefix forward reference, including a different-width QK member.
    a = decode([Fast.Context(m, src, 12) for m in models], check_every=1, compact=False)
    b = slow(models, src, 12)
    assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
    assert torch.allclose(a[1], b[1], atol=2e-5, rtol=1e-6)
    # Three copies = single model, with byte/token equality and LP tolerance.
    m = models[0]
    three = decode([Fast.Context(m, src, 12) for _ in range(3)], check_every=1, compact=False)
    seq, mean, _ = Speed.decode(Fast.Context(m, src, 12), options=Speed.Options(eos_every=1))
    lengths = torch.tensor([[len(C.active_tokens(x))] for x in seq[:,0].tolist()])
    assert torch.equal(seq, three[0]) and torch.equal(lengths, three[2])
    assert torch.allclose(mean*lengths, three[1], atol=2e-5)
    # Teacher forcing: heterogeneous EOS lengths, padding and sum/mean identity.
    targets = [[5,6,2], [9,2], [2]]
    sums, means, n = teacher_score(m, src, targets)
    assert n.tolist() == [3,2,1] and torch.allclose(sums, means*n)
    for i,t in enumerate(targets):
        prefix = torch.tensor([[1]+t[:-1]])
        with torch.inference_mode(): lp = log_probs(m(src[i:i+1], prefix))
        expected = sum(lp[0,j,x] for j,x in enumerate(t))
        assert abs(sums[i]-expected) < 1e-5
    for alpha in (.3,.5,.7):
        state = soup_state(m.state_dict(), models[1].state_dict(), alpha)
        assert all(t.dtype == torch.float32 for t in state.values())
        for key,t in state.items():
            assert torch.equal(t, m.state_dict()[key].float()*alpha+models[1].state_dict()[key].float()*(1-alpha))
    for right in ({}, dict(m.state_dict(), **{'tokens.weight': torch.zeros(2,2)})):
        try: soup_state(m.state_dict(), right, .5)
        except AssertionError: pass
        else: raise AssertionError('Invalid soup accepted')
    # Staggered EOS, compaction of every member, output ID/order and PAD tail.
    class Context:
        capacity=7
        def __init__(self): self.src=torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8]])
    class Script:
        def __init__(self,c,k,share_cross=False):
            self.ids=c.src[:,0].repeat_interleave(k); self.t=0
        def step(self,tokens):
            out=torch.full((len(self.ids),16),-10.)
            stop=self.ids.remainder(4)+1
            out[torch.arange(len(self.ids)),torch.where(self.t+1>=stop,2,7)]=10.
            self.t+=1;return out
        def select(self,index): self.ids=self.ids[index]
    stats={}
    with patch('e1_decode.Decoder',Script):
        seq,lp,n=decode([Context() for _ in range(3)],k=2,check_every=1,stats=stats)
    assert stats['compactions'] and seq.shape == (8,2,4)
    for i in range(8):
        end=(i+1)%4+1
        assert n[i].tolist()==[end,end]
        assert seq[i,:,end-1].eq(2).all() and seq[i,:,end:].eq(0).all()
    # Non-EOS capacity termination keeps all positions, no invented EOS.
    with patch('e1_decode.Decoder',Script):
        c=Context();c.capacity=1
        seq,_,n=decode([c]*3,check_every=1)
        assert seq.shape==(8,1,1) and n.eq(1).all()
    return dict(passed=True,slow_prefix_rows=3,identical_copies=True,EOS_padding_compaction=True,
                teacher_forcing_padding=True,soup_FP32_keys_shapes=True)


def real():
    from final_representation import tokenizer
    from e1_worker import vocabulary
    inputs = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    tok,table=tokenizer(inputs/'tokenizer.json',30)
    paths=[WORK/r['local'] for r in read(WORK/'checks/e1-weights.json')['files']]
    specs=read(ROOT/'e1-config.json')['models']
    payloads=[];models=[]
    for p,spec in zip(paths,specs):
        assert sha(p)==spec['sha256']
        payload=torch.load(p,map_location='cpu',weights_only=True);payloads.append(payload)
        m=C.SimpleTransformer(**payload['model_spec'],dropout=0.).eval()
        m.load_state_dict(payload['model'],strict=True);models.append(m)
    vocab=vocabulary(payloads,tok,table)
    rows=read(inputs/'heldout4k.json')
    from final_split import fixed_split
    ids=fixed_split()['tune2048'][:3]
    src=torch.nn.utils.rnn.pad_sequence([torch.tensor([1]+C.encode(tok,rows[i]['question'])+[2])for i in ids],batch_first=True)
    a=decode([Fast.Context(m,src,256)for m in models],check_every=1,compact=False)
    b=slow(models,src,256)
    assert torch.equal(a[0],b[0]) and torch.allclose(a[1],b[1],atol=2e-4)
    assert a[0][:,0].eq(C.EOS).any(-1).all(),'Reference examples must finish, not just match a prefix'
    ctx=Fast.Context(models[0],src,256)
    one=decode([ctx],check_every=16)
    three=decode([Fast.Context(models[0],src,256)for _ in range(3)],check_every=16)
    assert torch.equal(one[0],three[0]) and torch.equal(one[2],three[2])
    assert torch.allclose(one[1],three[1],atol=2e-4)
    assert [C.decode_bytes(table,x)for x in one[0][:,0].tolist()]==[C.decode_bytes(table,x)for x in three[0][:,0].tolist()]
    targets=[C.active_tokens(x) for x in one[0][:,0].tolist()]
    assert all(t[-1]==2 for t in targets),'Real greedy failed to reach EOS within 256'
    sums,means,n=teacher_score(models[0],src,targets)
    err=float((sums-one[1][:,0]).abs().max())
    assert err<.003 and torch.equal(n,one[2][:,0]),err
    tf_errors={'C':err}
    canonical={}
    def canonical_check(name,model,target,generation_sums):
        encoded=[C.encode(tok,C.decode_bytes(table,t).decode('utf8',errors='replace'))+[2]for t in target]
        total,_,_=teacher_score(model,src,encoded)
        canonical[name]=dict(equal_token_paths=sum(a==b for a,b in zip(target,encoded)),rows=len(target),
            sum_LP_error_vs_generation=float((total-generation_sums[:,0]).abs().max()))
        assert canonical[name]['equal_token_paths']==len(target),name+' fixed greedy changed tokenization'
        assert canonical[name]['sum_LP_error_vs_generation']<.003,(name,canonical[name])
    canonical_check('C',models[0],targets,one[1])
    for name,model in zip(('P','W'),models[1:]):
        generated=decode([Fast.Context(model,src,256)],check_every=16)
        target=[C.active_tokens(x)for x in generated[0][:,0].tolist()]
        assert all(t[-1]==2 for t in target),name+' greedy did not reach EOS'
        total,_,length=teacher_score(model,src,target)
        difference=float((total-generated[1][:,0]).abs().max())
        assert difference<.003 and torch.equal(length,generated[2][:,0]),(name,difference)
        tf_errors[name]=difference
        canonical_check(name,model,target,generated[1])
    soup_errors={}
    for alpha in (.3,.5,.7):
        state=soup_state(payloads[0]['model'],payloads[1]['model'],alpha)
        models[1].load_state_dict(state,strict=True)
        soup_errors[str(alpha)]=len(state)
    return dict(passed=True,vocabulary=vocab,real_models=3,real_rows=ids,
                slow_cached_greedy_exact=True,slow_prefix_steps=int(a[0].shape[-1]),slow_reference_all_EOS=True,
                identical_three_greedy_bytes=True,
                generated_greedy_TF_sum_max_abs_error=err,TF_sum_errors_by_model=tf_errors,
                canonical_greedy_checks=canonical,soups=soup_errors)


def pipeline():
    nb=read(BUNDLE/NAME)
    first=next(c['source']for c in nb['cells']if c['cell_type']=='code')
    if isinstance(first,list):first=''.join(first)
    sources=ast.literal_eval(next(n.value for n in ast.parse(first).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name)and t.id=='SOURCES'for t in n.targets)))
    config=read(BUNDLE/'source-identity.json')['config']
    with tempfile.TemporaryDirectory(prefix='e1-cpu-')as d:
        tmp=Path(d);modules=tmp/'src';modules.mkdir()
        for name,s in sources.items():(modules/name).write_text(s)
        specs=[]
        for i,spec in enumerate(config['models']):
            arch=dict(d_model=16 if i<2 else 24,heads=2,encoder_layers=1,decoder_layers=1,
                      swiglu_dim=32,vocab=8195,max_positions=512,qk_norm='layernorm'if i==2 else False)
            m=C.SimpleTransformer(**arch,dropout=0.).eval()
            with torch.no_grad():m.head.weight.zero_();m.head.bias.fill_(-1000);m.head.bias[2]=1000
            p=tmp/(str(i)+'.pt')
            torch.save(dict(format='nlp-hw02-model-only-v1',optimizer_included=False,model_spec=arch,model=m.state_dict()),p)
            specs.append(dict(spec,path=str(p),sha256=sha(p)))
        rows=tmp/'heldout4k.json';rows.write_text(json.dumps([{'question':'Compute 2 + 3.'},{'question':'What is 7 minus 2?'}]))
        refs=tmp/'refs.json';refs.write_text(json.dumps({'rows':[{'status':'numeric','value':5}]*2}))
        token=WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json'
        cfg=dict(config,models=specs,cpu_fixture=True,fixture_rows=2,smoke_rows=1,cap=8,
            single_lanes=32,ensemble_lanes=16,output=str(tmp/'e1'),
            inputs={'heldout4k.json':str(rows),'heldout4k-reference-execution.json':str(refs),'tokenizer.json':str(token)})
        p=tmp/'config.json';p.write_text(json.dumps(cfg))
        r=subprocess.run([sys.executable,str(modules/'e1_worker.py'),str(p),'tune2048','smoke'],
                         capture_output=True,text=True,timeout=60,env=dict(os.environ,E1_DEADLINE=str(time.monotonic()+55)))
        assert r.returncode==0,r.stderr+r.stdout
        folder=tmp/'e1/smoke/tune2048';summary=read(folder/'summary.json')
        assert summary['status']=='complete'and summary['candidates_per_row']==87
        with gzip.open(folder/'pool-00000.jsonl.gz','rt')as f:record=json.loads(f.readline())
        assert len(record['candidates'])==1 and len(record['pools'])==14
        assert set(record['candidates'][0]['ll'])=={'C','P','W'}
        assert all(c['execution']['status']!='budget_not_evaluated'for c in record['candidates'])
        return dict(passed=True,embedded_worker=True,executor=True,gzip=True,unique_LL=True,
                    candidates_per_row=87,seconds=summary['seconds'])


def static():
    import nbformat
    nb=read(BUNDLE/NAME);nbformat.validate(nbformat.from_dict(nb))
    for c in nb['cells']:
        if c['cell_type']=='code':compile(''.join(c['source'])if isinstance(c['source'],list)else c['source'],'E1cell','exec')
    identity=read(BUNDLE/'source-identity.json')
    assert sha(BUNDLE/NAME)==identity['notebook_sha256']
    assert identity['frozen_v10_source_sha256']==read(ROOT/'final-v8-source-identity.json')['embedded_source_sha256']
    meta=read(BUNDLE/'kernel-metadata.json')
    assert meta['is_private']and not meta['enable_internet']and meta['machine_shape']=='NvidiaTeslaT4'
    assert meta['kernel_sources']==['friend1/nlp-hw02-b6/1','friend1/nlp-hw02-kaggle-p1/3']
    assert read(BUNDLE/'session-limits.json')['session_timeout_seconds']==4200
    return dict(passed=True,nbformat=True,private=True,internet=False,frozen_v10=True,timeout=4200)


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    results={}
    for name in ('unit','real','static','pipeline'):
        results[name]=globals()[name]();print(name,json.dumps(results[name]),flush=True)
    result=dict(passed=True,torch=torch.__version__,notebook_sha256=sha(BUNDLE/NAME),checks=results)
    (WORK/'checks/e1-cpu.json').write_text(json.dumps(result,indent=2)+'\n')
