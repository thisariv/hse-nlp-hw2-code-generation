#!/usr/bin/env python3
"""Real CPU tiny representation training/evaluation plus real B teacher-forced score."""
import json, math, pathlib, sys, tempfile, time
ROOT=pathlib.Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))

def main():
    import numpy as np, torch
    import exp26_core as C,exp26_eval as E,exp26_exec as N,s3_representation as R,s3_runtime as S,s3_score as B
    from exp26_common import read_json,write_json,sha_file
    from session_runner import archive
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tmp=pathlib.Path(tempfile.mkdtemp(prefix='nlp-s3-smoke-'));inp=tmp/'inputs';inp.mkdir()
    original_inputs=ROOT.parent/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    import shutil
    shutil.copyfile(original_inputs/'tokenizer.json',inp/'tokenizer.json')
    spec={'d_model':16,'heads':2,'encoder_layers':1,'decoder_layers':1,'swiglu_dim':32,'max_positions':512}
    torch.manual_seed(42);model=C.SimpleTransformer(**spec,dropout=0.);torch.save(model.state_dict(),inp/'initial.pt');del model
    fixtures=[('5 and 5', 'def solve() -> int:\n    result = 5.0 + 5\n    return result\n'),
              ('-5 and 0.5 and 1/2', 'def solve() -> int:\n    result = -5 + .5\n    return result\n'),
              ('2 and 3', 'def solve() -> int:\n    s = "# literal" # comment\n    result = 2 * 3\n    return result\n'),
              ('9', 'def solve() -> int:\n    x = 9 \\\n\n    result = x\n    return result\n'),
              ('5', 'def solve() -> int:\n    result = (\n        5 + 1\n    )\n    return result\n'),
              (' and '.join(map(str,range(25))), 'def solve():\n    return 19 + 24\n')]
    checks=[]
    for variant in (30,31):
        tok,tokenbytes=R.tokenizer(inp/'tokenizer.json',variant)
        fixture_codes=[]
        for q,c in fixtures:
            qq,text=R.transform(q,c,variant);encoded=C.encode(tok,text)
            assert C.decode_bytes(tokenbytes,encoded).decode()==text
            restored=R.restore(text,q,variant)
            fixture_codes.extend([c,restored])
        execution=N.evaluate_programs(fixture_codes,workers=1)
        for a,b in zip(execution['rows'][::2],execution['rows'][1::2]):
            assert a['status']==b['status']=='numeric' and a['value']==b['value']
        checks.append({'check':f'{variant} fixtures: signs/types/repeats/string/backslash/multiline/20limit','passed':True})
    rows=[{'question':f'Add {i} and 2','code':f'def solve() -> int:\n    result = {i} + 2\n    return result\n'} for i in range(12)]
    tok,_=C.init_tokenizer(inp/'tokenizer.json')
    class Cache:
        manifest_sha='tiny-original-cache'
        def __len__(self):return len(rows)
        def __getitem__(self,i):
            r=rows[int(i)];return np.array([1]+C.encode(tok,r['question'])+[2]),np.array(C.encode(tok,r['code']))
    order=np.array([5,1,8,0,3,7,11,2,10,4,9,6],dtype=np.uint32)
    sets={'heldout4k':rows[:2],'old1024':rows[2:4]}
    refs={k:{'rows':[{'status':'numeric','value':i+2} for i in (range(2) if k=='heldout4k' else range(2,4))]} for k in sets}
    baselines={tag:{'splits':{s:{'exec':{'eligible_indices':[0,1],'match_indices':[],'denominator':2}} for s in sets}} for tag in ('p000002','p000004')}
    working=tmp/'working';working.mkdir()
    for variant in (30,31):
        root=working/f'gpu{variant-30}-exp{variant}'
        report=S.run(inp,Cache(),order,{'file':{'sha256':'tiny-order'}},sets,refs,baselines,variant,torch.device('cpu'),root,
                     tmp/f'scratch{variant}',time.monotonic()+180,'tiny-code',smoke=True,base_spec=spec)
        assert report['status']=='complete' and not report['missing_points']
        assert report['points']['p000002']['actual_pairs']==6 and report['points']['p000004']['actual_pairs']==12
        assert report['points']['p000004']['compute_updates_total']==7
        p=torch.load(root/'checkpoints/latest.pt',map_location='cpu',weights_only=True)
        assert p['training']['phase']=='main' and p['training']['main_updates']==4 and p['training']['data_cursor']==8
        assert all(int(s['step'])==4 for s in p['optimizer']['state'].values())
        assert len(list((root/'weights').glob('*.pt')))==2 and len(list((root/'checkpoints').glob('*.pt')))==1
        expected=[R.transform(rows[int(i)]['question'],rows[int(i)]['code'],variant) for i in order]
        rep=S.PrefixCache(Cache(),order,inp/'tokenizer.json',variant,tmp/f'replay{variant}')
        rep.prepare(6,time.monotonic()+60,workers=2)
        rep.prepare(12,time.monotonic()+60,workers=2)
        rtok,rbytes=R.tokenizer(inp/'tokenizer.json',variant)
        assert all(C.decode_bytes(rbytes,rep[i][0]).decode()==q and C.decode_bytes(rbytes,rep[i][1]).decode()==c for i,(q,c) in enumerate(expected))
        checks.append({'check':f'{variant} real tiny updates, two cooldown branches, AdamW rollback, exact original batches, restored exec/CE','passed':True})
    # Independent likelihood oracle; EOS contributes, PAD does not, variable lengths are preserved.
    small=C.SimpleTransformer(**spec,dropout=0.).eval()
    sources=[[1,4,5,2],[1,6,2]];targets=[[5,6,2],[7,2]]
    actual=B.score_batch(small,sources,targets,torch.device('cpu'))
    for source,target,(total,mean,n) in zip(sources,targets,actual):
        with torch.inference_mode():
            logits=small(torch.tensor([source]),torch.tensor([[1]+target[:-1]]))
            expected=logits.log_softmax(-1)[0,torch.arange(len(target)),torch.tensor(target)].sum().item()
        assert abs(total-expected)<1e-5 and abs(mean-total/n)<1e-6 and n==len(target)
    tiny_weights=tmp/'score-tiny.pt'
    torch.save({'format':'nlp-hw02-model-only-v1','model':small.state_dict(),'model_spec':spec},tiny_weights)
    del small
    real_root=ROOT.parent/'results/kaggle-runs/nlp-hw02-kaggle-s2/2/gpu1-exp28'
    receipt=read_json(real_root/'points/t04/checkpoint.json');cp=real_root/receipt['file'];assert sha_file(cp)==receipt['sha256']
    payload=torch.load(cp,map_location='cpu',weights_only=True);real=C.SimpleTransformer(**payload['model_spec'],dropout=0.).eval()
    real.load_state_dict(payload['model']);del payload
    candidate_folder=ROOT.parent/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/gpu0-exp27/heldout4k/candidates'
    f=next(p for p in sorted(candidate_folder.glob('batch-*.json')) if '.sha.' not in p.name)
    row=read_json(f)[0];flat=B.flatten(row,tok);assert len(flat)==65 and [c['candidate_index'] for c in flat]==list(range(65))
    src=[1]+C.encode(tok,row['question'])+[2]
    scores=B.score_batch(real,[src,src],[flat[0]['ids'],flat[1]['ids']],torch.device('cpu'))
    assert all(math.isfinite(total) and math.isfinite(mean) and total<0 and mean<0 for total,mean,n in scores)
    assert all(parameter.grad is None for parameter in real.parameters())
    checks.append({'check':'real S2/2 B t04 SHA/spec, real S1 candidates, sum/mean/EOS/PAD independent oracle; no training','passed':True,'real_tiny_scores':scores})
    del real
    candidates=tmp/'candidates'
    tiny_sets={split:[{'question':row['question']},{'question':row['question']}] for split in ('heldout4k','old1024')}
    for split in tiny_sets:
        path=candidates/split/'candidates/batch-000000.json';write_json(path,[dict(row,id=1),dict(row,id=0)])
        write_json(path.with_suffix('.sha.json'),{'sha256':sha_file(path)})
    scored=B.run(tiny_weights,spec,candidates,tiny_sets,inp/'tokenizer.json',torch.device('cpu'),working/'gpu0-exp32',time.monotonic()+60,batch=8)
    assert scored['status']=='complete' and all(s['candidates']==130 for s in scored['splits'].values())
    for split in tiny_sets:
        scored_rows=read_json(working/'gpu0-exp32'/split/'batch-000000.json')
        assert [(r['id'],r['candidate_index']) for r in scored_rows]==[(i,c) for i in (0,1) for c in range(65)]
        assert all(abs(r['mean_logprob']*r['length_with_eos']-r['sum_logprob'])<1e-4 for r in scored_rows)
    checks.append({'check':'B scoring end-to-end tiny JSON/SHA/alignment/all65/dedup/partial receipts','passed':True})
    result=archive(working,'s3');assert result==working and not list(working.glob('*.zip'))
    checks.append({'check':'S3 compact individual output, latest retention, no ZIP','passed':True})
    report={'passed':True,'checks':checks,'roundtrip':'kaggle-s3-roundtrip.json','tiny_output_bytes':sum(p.stat().st_size for p in working.rglob('*') if p.is_file()),
        'not_verified':['T4 FP16/fused/SDPA memory and numerical behavior','full data preprocessing/scoring speed','GPU quality and equal-time alignment']}
    write_json(ROOT.parent/'checks/kaggle-s3-smoke.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
