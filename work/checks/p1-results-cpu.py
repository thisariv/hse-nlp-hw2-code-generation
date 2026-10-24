"""Meaningful CPU checks for reverse scoring, preserved votes and question CV."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import torch

WORK = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reviewer', WORK/'research/score-reviewer.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
torch.set_num_threads(1)
torch.manual_seed(42)
model = R.C.SimpleTransformer(d_model=16, heads=2, encoder_layers=1,
                              decoder_layers=1, swiglu_dim=24, vocab=32, dropout=0.).eval()
codes = [[4,5,6],[7]]; questions = [[8,9],[10]]
batch = R.score_batch(model,codes,questions)
single = [R.score_batch(model,[c],[q])[0] for c,q in zip(codes,questions)]
for q,b,s in zip(questions,batch,single):
    assert b['question_tokens']==len(q)+1  # question content + EOS; no BOS/PAD
    assert abs(b['sum_logprob']-s['sum_logprob'])<1e-5
    assert abs(b['mean_logprob']*b['question_tokens']-b['sum_logprob'])<1e-5

def candidate(value,score):
    return dict(execution=dict(status='numeric',value=value),mean_logprob=-1.,
                reviewer=dict(mean_logprob=score,sum_logprob=score*3))

a=candidate(1.,-1.);b=candidate(2.,-.9)
row=dict(id=0,reference=dict(status='numeric',value=2.),greedy=a,
         samples={'0.7':[a,copy.deepcopy(a),b]})
assert sorted(len(g) for g in R.components(row['samples']['0.7']))==[1,2]
assert R.select(row,'vote-reviewer',0)['execution']['value']==1.
assert R.select(row,'vote-reviewer',32)['execution']['value']==2.
changed=copy.deepcopy(row);changed['reference']=dict(status='numeric',value=999.)
assert R.select(changed,'vote-reviewer',32)==R.select(row,'vote-reviewer',32)
questions_fixture=[{'question':' A   B '},{'question':'a b'},{'question':'different'}]
folds=R.folds_for([{'id':i} for i in range(3)],questions_fixture)
assert sorted(folds[0]+folds[1])==[0,1,2]
assert any(0 in f and 1 in f for f in folds)

receipt=dict(passed=True,GPU=False,new_training_updates=0,checks=[
    'BOS/PAD not scored; EOS included; token count and mean/sum agree',
    'Variable-length padded batch agrees with single-example forward',
    'Duplicate candidates retain vote multiplicity',
    'Reviewer changes cluster selection; reference mutation cannot change selection',
    'Identical normalized questions remain together in CV folds'],
    scorer_sha256=R.sha(WORK/'research/score-reviewer.py'))
scored_path=WORK/'results/p1-reviewer/summary.json'
if scored_path.exists():
    scored=json.loads(scored_path.read_text());identity=scored['identity']
    assert identity['scorer_sha256']==receipt['scorer_sha256']
    count=0
    for p in sorted(scored_path.parent.glob('scores-[0-9]*.json')):
        if p.name.endswith('.sha.json'):continue
        assert R.sha(p)==json.loads(p.with_suffix('.sha.json').read_text())['sha256']
        values=json.loads(p.read_text());count+=len(values)
        for v in values:
            assert math.isfinite(v['sum_logprob']) and math.isfinite(v['mean_logprob'])
            # Saved mean is an FP32 division, so absolute error scales with NLL.
            assert v['question_tokens']>0 and math.isclose(v['sum_logprob'],
                v['mean_logprob']*v['question_tokens'],rel_tol=2*torch.finfo(torch.float32).eps,abs_tol=1e-6)
    assert count==identity['unique_pairs']==26291
    assert scored['scored_candidates']==34816 and len(identity['task_ids'])==2048
    assert scored['full_vote_replay']=={'C-main':{'rows':2048,'matches':1502},
                                        'C-long2x':{'rows':2048,'matches':1507}}
    report=json.loads((WORK/'results/p1-results.json').read_text())
    assert report['passed'] and report['reviewer_scoring']['sha256']==R.sha(scored_path)
    assert report['adaptive_replay']['L3-long adaptive2→8']['mean_stochastic_samples']==9.361328125
    payload=torch.load(WORK/'results/kaggle-runs/nlp-hw02-kaggle-p1/final/model-reviewer__weights__final-fp16.pt',
                       map_location='cpu',weights_only=True)
    assert all(t.dtype==torch.float16 and bool(torch.isfinite(t).all()) for t in payload['model'].values())
    receipt.update(real_scored_pairs=count,real_scored_candidates=34816,
                   score_summary_sha256=R.sha(scored_path),analysis_sha256=R.sha(WORK/'research/analyze-p1-results.py'))
    receipt['checks'].extend(['All immutable score chunks verified by SHA and finite sum/mean/token contract',
                             'Real L3 saved baseline vote replay and result references agree',
                             'All trained reviewer model tensors finite FP16; all2048 tasks scored'])
(WORK/'checks/p1-results-cpu.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('P1 scoring CPU checks passed')
