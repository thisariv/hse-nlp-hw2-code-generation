#!/usr/bin/env python3
"""CPU log P(question|represented_code), cluster selectors and two-fold beta tuning.

Every candidate retains its multiplicity. Gold enters only metrics/beta training.
Immutable scored chunks permit resuming a long CPU pass without rescoring forwards.
"""
import argparse
import copy
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'kaggle/src'))
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
import exp26_core as C
import exp27_decode as D
import l1_runtime as L
import s3_representation as S
from kaggle_io import write,sha
from exp26_common import json_hash

BETAS={'sum_logprob':(0.,.001,.0025,.005,.01,.025,.05,.1,.25,.5,1.,2.,4.,8.),
       'mean_logprob':(0.,.25,.5,1.,2.,4.,8.,16.,32.,64.,128.,256.)}
RULES=('vote-reviewer','cluster-exp','cluster-max','cluster-mean')


@torch.inference_mode()
def score_batch(model, codes, questions):
    """BOS/EOS on encoder; decoder BOS, labels question content + EOS. No PAD/BOS score."""
    src=pad_sequence([torch.tensor([C.BOS,*code,C.EOS]) for code in codes],batch_first=True,padding_value=C.PAD)
    tgt=pad_sequence([torch.tensor([C.BOS,*q]) for q in questions],batch_first=True,padding_value=C.PAD)
    labels=pad_sequence([torch.tensor([*q,C.EOS]) for q in questions],batch_first=True,padding_value=C.PAD)
    logits=model(src,tgt).float()
    nll=F.cross_entropy(logits.transpose(1,2),labels,ignore_index=C.PAD,reduction='none')
    counts=labels.ne(C.PAD).sum(1)
    sums=-nll.sum(1)
    assert bool(torch.isfinite(sums).all())
    return [dict(sum_logprob=float(s),mean_logprob=float(s/n),question_tokens=int(n))
            for s,n in zip(sums,counts)]


def normalize(question): return re.sub(r'\s+',' ',question.strip()).casefold()


def components(candidates):
    good=[c for c in candidates if c['execution']['status']=='numeric'
          and math.isfinite(c['execution']['value']) and c.get('mean_logprob') is not None]
    parent=list(range(len(good)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(good)):
        for j in range(i):
            if math.isclose(good[i]['execution']['value'],good[j]['execution']['value'],rel_tol=1e-6,abs_tol=1e-6):
                parent[root(i)]=root(j)
    groups={}
    for i,c in enumerate(good):groups.setdefault(root(i),[]).append(c)
    return list(groups.values())


def select(row, rule, beta=1., score='mean_logprob'):
    """No access to reference/correctness in selection."""
    groups=components(row['samples']['0.7'])
    if not groups:return row['greedy']
    greedy=row['greedy']['execution']
    def key(group):
        values=[c['reviewer'][score] for c in group]
        if rule=='vote-reviewer': value=math.log(len(group))+beta*max(values)
        elif rule=='cluster-exp':
            scaled=[beta*s for s in values];m=max(scaled)
            value=m+math.log(sum(math.exp(s-m) for s in scaled))
        elif rule=='cluster-max':value=max(values)
        elif rule=='cluster-mean':value=sum(values)/len(values)
        else:raise ValueError(rule)
        contains=greedy['status']=='numeric' and any(math.isclose(c['execution']['value'],greedy['value'],rel_tol=1e-6,abs_tol=1e-6) for c in group)
        return value,len(group),contains,sum(c['mean_logprob'] for c in group)/len(group)
    group=max(groups,key=key)
    return max(group,key=lambda c:(c['reviewer'][score],c['mean_logprob']))


def correct(row,candidate):
    a,b=row['reference'],candidate['execution']
    return a['status']==b['status']=='numeric' and math.isclose(a['value'],b['value'],rel_tol=1e-6,abs_tol=1e-6)


def paired_ci(baseline,variant,draws=10000):
    delta=np.asarray(variant,dtype=float)-np.asarray(baseline,dtype=float)
    rng=np.random.default_rng(2026101401);boots=[]
    for lo in range(0,draws,256):
        indices=rng.integers(len(delta),size=(min(256,draws-lo),len(delta)))
        boots.extend(delta[indices].mean(1).tolist())
    return dict(n=len(delta),gains=int((delta>0).sum()),losses=int((delta<0).sum()),
                delta_pp=100*float(delta.mean()),CI95_pp=(100*np.quantile(boots,[.025,.975])).tolist(),
                method='paired question-row bootstrap; 10000 draws, seed2026101401')


def folds_for(rows,questions):
    # Identical normalized questions stay in the same fold (no beta tuning leakage).
    groups={}
    for r in rows:groups.setdefault(normalize(questions[r['id']]['question']),[]).append(r['id'])
    halves=[[],[]]
    for q,ids in sorted(groups.items(),key=lambda item:hashlib.sha256(('reviewer-cv-v1:'+item[0]).encode()).digest()):
        halves[0 if len(halves[0])<=len(halves[1]) else 1].extend(ids)
    return [sorted(h) for h in halves]


def analyze(rows,questions):
    baseline=[correct(r,D.choose(r,dict(mode='vote',temperature=.7,k=len(r['samples']['0.7']),greedy_vote=False))) for r in rows]
    oracle=[any(correct(r,c) for c in r['samples']['0.7']) for r in rows]
    halves=folds_for(rows,questions);position={r['id']:i for i,r in enumerate(rows)}
    report=dict(rows=len(rows),vote_matches=sum(baseline),oracle_matches=sum(oracle),
                missed_oracle_ids=[r['id'] for r,b,o in zip(rows,baseline,oracle) if o and not b],
                vote_correct_ids=[r['id'] for r,b in zip(rows,baseline) if b],folds=halves,selectors={})
    all_predictions={}
    def metrics(pred):
        return dict(oof_matches=sum(pred),paired_vs_vote=paired_ci(baseline,pred),
                    repaired_missed_oracle=sum(o and not b and v for o,b,v in zip(oracle,baseline,pred)),
                    broken_vote=sum(b and not v for b,v in zip(baseline,pred)),
                    oof_correct_ids=[r['id'] for r,c in zip(rows,pred) if c])
    for score in ('sum_logprob','mean_logprob'):
        for rule in RULES:
            grid=BETAS[score] if rule in ('vote-reviewer','cluster-exp') else (1.,)
            predictions={beta:[correct(r,select(r,rule,beta,score)) for r in rows] for beta in grid}
            for beta,pred in predictions.items():all_predictions[(rule,score,beta)]=pred
            oof=[None]*len(rows);tuning=[]
            for held in (0,1):
                train=[position[i] for i in halves[1-held]];test=[position[i] for i in halves[held]]
                best=max(grid,key=lambda beta:(sum(predictions[beta][i] for i in train),-beta))
                for i in test:oof[i]=predictions[best][i]
                tuning.append(dict(train_fold=1-held,held_fold=held,beta=best,
                                   train_matches=sum(predictions[best][i] for i in train),
                                   held_matches=sum(predictions[best][i] for i in test)))
            assert all(v is not None for v in oof)
            report['selectors'][rule+'/'+score]=dict(**metrics(oof),tuning=tuning,
                full_tune_grid={str(b):sum(predictions[b]) for b in grid})
    # Select rule, normalization AND beta on the other half; no full-data winner.
    oof=[None]*len(rows);tuning=[]
    for held in (0,1):
        train=[position[i] for i in halves[1-held]];test=[position[i] for i in halves[held]]
        best=max(all_predictions,key=lambda key:sum(all_predictions[key][i] for i in train))
        for i in test:oof[i]=all_predictions[best][i]
        tuning.append(dict(train_fold=1-held,held_fold=held,rule=best[0],score=best[1],beta=best[2],
                           train_matches=sum(all_predictions[best][i] for i in train),
                           held_matches=sum(all_predictions[best][i] for i in test)))
    report['joint_cv']=dict(**metrics(oof),tuning=tuning,
                           policy='rule/normalization/beta all chosen on opposite half; ties follow predefined grid order')
    return report


def load_journals(folder,questions):
    models={};receipts=[]
    for name in ('C-main','C-long2x'):
        paths=sorted(folder.glob(f'model-{name}__points__l2-62280__evaluation__k8-tune2048__batch-*.json'))
        assert paths, 'L3 final K8 journals missing'
        rows=[r for p in paths for r in json.loads(p.read_text())]
        assert len({r['id'] for r in rows})==len(rows)
        for r in rows:
            q=questions[r['id']]['question']
            assert hashlib.sha256(q.encode()).hexdigest()==r['question_sha256']
            assert len(r['samples']['0.7'])==8
            assert all(isinstance(c['represented_code'],str) for c in r['samples']['0.7'])
        models[name]=sorted(rows,key=lambda r:r['id'])
        receipts.extend(dict(file=p.name,sha256=sha(p)) for p in paths)
    assert [r['id'] for r in models['C-main']]==[r['id'] for r in models['C-long2x']]
    path=folder/'model-C-main__points__l2-62280__evaluation__heldout4k-generations.json'
    greedy=json.loads(path.read_text());receipts.append(dict(file=path.name,sha256=sha(path)))
    for row in models['C-main']:
        g=greedy[row['id']]
        assert g['id']==row['id'] and g['generated_code']==row['greedy']['code']
        assert g['question']==questions[row['id']]['question']
        row['greedy']['represented_code']=g['represented_code']
    return models,receipts


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights',type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--journals',type=Path,default=WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final')
    p.add_argument('--inputs',type=Path,default=WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs')
    p.add_argument('--out',type=Path,default=WORK/'research/outputs/reviewer-scores');p.add_argument('--batch',type=int,default=16)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--max-tasks',type=int,default=0,help='0=all; random task subset before inspecting outcomes')
    p.add_argument('--subset-seed',type=int,default=2026101402)
    p.add_argument('--benchmark-only',action='store_true');a=p.parse_args()
    assert a.smoke or a.weights is not None
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.manual_seed(42)
    questions=json.loads((a.inputs/'heldout4k.json').read_text());tok,_=S.tokenizer(a.inputs/'tokenizer.json',30)
    models,receipts=load_journals(a.journals,questions)
    full_vote_replay={}
    for name,rows in models.items():
        replay=[correct(r,D.choose(r,dict(mode='vote',temperature=.7,k=8,greedy_vote=False))) for r in rows]
        assert replay==[r['correct'] for r in rows], 'Saved L3 voting baseline drift'
        full_vote_replay[name]=dict(rows=len(rows),matches=sum(replay))
    if a.smoke:
        models={name:rows[:8] for name,rows in models.items()}
        spec=dict(d_model=16,heads=2,encoder_layers=1,decoder_layers=1,swiglu_dim=24,vocab=8195)
        model=L.new_model(spec,torch.device('cpu')).eval();weight_sha='random-seed42-smoke'
    else:
        payload=torch.load(a.weights,map_location='cpu',weights_only=True)
        assert payload['format']=='nlp-hw02-model-only-v1' and payload['identity']['model']=='reviewer'
        spec=payload['model_spec'];model=C.SimpleTransformer(**spec,dropout=0.).eval()
        model.load_state_dict(payload['model'],strict=True);weight_sha=sha(a.weights);del payload
    if a.max_tasks and a.max_tasks < len(models['C-main']):
        ids=sorted(np.random.default_rng(a.subset_seed).choice(
            [r['id'] for r in models['C-main']],size=a.max_tasks,replace=False).tolist())
        models={name:[r for r in rows if r['id'] in set(ids)] for name,rows in models.items()}
    # All candidates are scored, including invalid/non-numeric ones. Selection filters later.
    jobs={}
    for name,rows in models.items():
        for row in rows:
            q=tuple(C.encode(tok,questions[row['id']]['question']))
            candidates=row['samples']['0.7']+([row['greedy']] if name=='C-main' else [])
            for candidate in candidates:
                code=tuple(C.encode(tok,candidate['represented_code']))
                jobs.setdefault((code,q),[]).append(candidate)
    ordered=sorted(jobs,key=lambda key:(len(key[0])+len(key[1]),key))
    identity=dict(weights_sha256=weight_sha,tokenizer_sha256=sha(a.inputs/'tokenizer.json'),
        journal_receipts_sha256=json_hash(receipts),smoke=a.smoke,unique_pairs=len(jobs),
        jobs_sha256=json_hash(ordered),batch=a.batch,precision='CPU FP32',BOS=1,EOS=2,PAD=0,
        task_ids=[r['id'] for r in models['C-main']],subset_seed=a.subset_seed,
        torch_version=torch.__version__,scorer_sha256=sha(Path(__file__)),
        direction='question|represented_code; same mode retained; no restore/recompact of candidates')
    a.out.mkdir(parents=True,exist_ok=True)
    if a.benchmark_only:
        # Stratified length positions, complete batches around quantiles; warmup excluded.
        batch=a.batch
        score_batch(model,[list(c) for c,q in ordered[:batch]],[list(q) for c,q in ordered[:batch]])
        timings=[]
        for fraction in (.05,.25,.5,.75,.95):
            start=min(len(ordered)-batch,int(fraction*len(ordered)));pairs=ordered[start:start+batch]
            begin=time.monotonic()
            score_batch(model,[list(c) for c,q in pairs],[list(q) for c,q in pairs])
            timings.append(dict(quantile=fraction,seconds=time.monotonic()-begin,pairs=len(pairs),
                                max_code_tokens=max(len(c) for c,q in pairs),max_question_tokens=max(len(q) for c,q in pairs)))
        estimate=sum(t['seconds']/t['pairs'] for t in timings)/len(timings)*len(ordered)
        result=dict(identity=identity,timings=timings,projected_forward_seconds=estimate,
                    policy='length-quantile pilot, excludes load/tokenization/analysis; timing estimate, not guarantee')
        write(a.out/'benchmark.json',result);print(json.dumps(dict(unique_pairs=len(ordered),timings=timings,projected_seconds=estimate)),flush=True)
        return
    if (a.out/'identity.json').exists():assert json.loads((a.out/'identity.json').read_text())==identity
    else:write(a.out/'identity.json',identity)
    begin=time.monotonic()
    for lo in range(0,len(ordered),256):
        part=ordered[lo:lo+256];path=a.out/f'scores-{lo:06d}.json'
        if path.exists():
            assert sha(path)==json.loads(path.with_suffix('.sha.json').read_text())['sha256']
            values=json.loads(path.read_text());assert len(values)==len(part)
        else:
            values=[]
            for start in range(0,len(part),a.batch):
                pairs=part[start:start+a.batch]
                values.extend(score_batch(model,[list(c) for c,q in pairs],[list(q) for c,q in pairs]))
            write(path,values);write(path.with_suffix('.sha.json'),dict(sha256=sha(path)))
        for key,value in zip(part,values):
            for candidate in jobs[key]:candidate['reviewer']=value
        write(a.out/'progress.json',dict(completed=lo+len(part),total=len(ordered),seconds=time.monotonic()-begin))
        print('SCORED',lo+len(part),'/',len(ordered),'seconds',round(time.monotonic()-begin,1),flush=True)
    # Compact cache: immutable score chunks + hashed input identity reconstruct all rows.
    source_models=dict(models)
    for k in (4,8):
        for greedy_vote in (False,True):
            mixed=[]
            for main,long in zip(source_models['C-main'],source_models['C-long2x']):
                row=dict(main);row['samples']={'0.7':([main['greedy']] if greedy_vote else [])+
                    main['samples']['0.7'][:k]+long['samples']['0.7'][:k]};mixed.append(row)
            models[f'mixed{k}+{k}'+('+greedy' if greedy_vote else '')]=mixed
    result=dict(passed=True,smoke=a.smoke,GPU=False,training_updates=0,identity=identity,
                scored_candidates=17*len(source_models['C-main']),
                full_vote_replay=full_vote_replay,
                seconds=time.monotonic()-begin,pools={name:analyze(rows,questions) for name,rows in models.items()},
                limitation='Two-fold tuning CI is descriptive; fixed trained weights, repeated tune, no fresh/test evaluation')
    write(a.out/'summary.json',result)
    if a.smoke:write(WORK/'checks/reviewer-score-cpu-smoke.json',result)
    print('REVIEWER CPU', 'smoke' if a.smoke else 'weights', 'passed;',result['scored_candidates'],'candidates')


if __name__=='__main__':main()
