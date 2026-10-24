"""Tune-only selection receipt first; frozen eval confirmation in a separate process."""
import argparse
import hashlib
import itertools
import json
import math
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

WORK=Path(__file__).resolve().parents[1];ROOT=WORK/'kaggle'
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from final_split import fixed_split
from final_choose import choose
from final_ensemble import replay_record
from tta_rules import apply

RULES=['dollar-words','ungroup','price-trim']
MIXES=[tuple(x)for k in range(1,4)for x in itertools.combinations(RULES,k)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def correct(ref,candidate):
    x=candidate['execution']
    return ref['status']==x['status']=='numeric' and math.isclose(ref['value'],x['value'],rel_tol=1e-6,abs_tol=1e-6)

def paired(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);d=b-a;n=len(d)
    values,counts=np.unique(d,return_counts=True)
    # Exactly the empirical question bootstrap of the mean, aggregated by value.
    draws=np.random.default_rng(2026101902).multinomial(n,counts/n,size=20000)
    boot=draws@values/n;ci=np.quantile(boot,[.025,.975])*100
    return dict(n=n,original_mean=float(a.mean()),mixed_mean=float(b.mean()),delta_pp=float(d.mean()*100),
                ci95_pp=ci.tolist(),gains=int((d>0).sum()),losses=int((d<0).sum()),
                original_sum=float(a.sum()),mixed_sum=float(b.sum()),bootstrap='paired tasks;20000;seed2026101902;percentile95')

def mixed(original,variants,rules,k):
    counters={'original':0,**{r:0 for r in rules}};result=[];sources=[]
    for j in range(k):
        branch='original' if j%2==0 else rules[(j//2)%len(rules)]
        if branch!='original' and variants[branch] is None:branch='original'
        pool=original if branch=='original' else variants[branch]
        result.append(pool[counters[branch]]);sources.append(branch);counters[branch]+=1
    return result,sources

def load(raw,split):
    summary=read(raw/'tta/summary.json');identity=read(ROOT/'tta-source-identity.json')
    assert summary['status']=='complete' and summary['K']==8 and summary['training'] is False
    assert summary['single_visible_T4'] and len(summary['models'])==3
    assert read(raw/'tta/config.json')==identity['config']
    for name,digest in identity['source_sha256'].items():assert sha(raw/'tta-src'/name)==digest,name
    assert read(WORK/'checks/tta-launch.json')['notebook_sha256']==identity['notebook_sha256']
    inputs=raw/'tta/inputs';rows=read(inputs/'heldout4k.json');refs=read(inputs/'heldout4k-reference-execution.json')['rows']
    # These two input hashes are fixed v8 identities, independent of generated output.
    assert sha(inputs/'heldout4k.json')=='28a3c0c9088c5b4a17d75d216e4abc4262cc9ca62884966c43e7ec09b6f040c2'
    assert sha(inputs/'heldout4k-reference-execution.json')=='3855680e9a37292aa2e351b8e8e4f7a5cbbe76444096d136470e4a9be8797111'
    ids=fixed_split()[split];journal={};receipts={}
    for spec,observed in zip(identity['config']['models'],summary['models']):
        name=spec['name'];assert observed['name']==name and observed['sha256']==spec['sha256'];journal[name]={}
        for branch in ['original']+RULES:
            expected=[i for i in ids if branch=='original' or apply(rows[i]['question'],branch)!=rows[i]['question']]
            folder=raw/'tta'/name/branch;assert read(folder/'summary.json')['complete']
            records=[]
            # Critically, tune action never opens eval journal files.
            for p in sorted(folder.glob(split+'-batch-*.json')):
                receipts[str(p.relative_to(raw))]=sha(p);records.extend(read(p))
            assert sorted(r['id']for r in records)==expected,(name,branch,len(records),len(expected))
            journal[name][branch]={r['id']:r for r in records}
            for r in records:
                i=r['id'];q=rows[i]['question'] if branch=='original' else apply(rows[i]['question'],branch)
                assert r['question_sha256']==hashlib.sha256(rows[i]['question'].encode()).hexdigest()
                assert r['input_sha256']==hashlib.sha256(q.encode()).hexdigest()
                assert r['reference']==refs[i] and r['variant']==branch and r['split']==split
                samples=r['samples']['0.7'];assert len(samples)==8
                assert [c['sample']for c in samples]==list(range(8))
                assert all(c['mean_logprob'] is not None and math.isfinite(c['mean_logprob'])for c in samples)
                assert all(c['execution']['status']!='budget_not_evaluated'for c in [r['greedy']]+samples)
    return ids,journal,identity['config']['models'],receipts

def pools(journal,name,i,rules,k):
    original=journal[name]['original'][i]['samples']['0.7']
    variants={r:journal[name][r][i]['samples']['0.7'] if i in journal[name][r] else None for r in rules}
    return mixed(original,variants,rules,k)

def primary(ids,journal,models,rules=None):
    bits=[];stops=[];counts=[];variant_counts=[]
    for i in ids:
        greedy=journal[models[0]['name']]['original'][i]['greedy'];samples={};sources={}
        for m in models:
            if rules:samples[m['name']],sources[m['name']]=pools(journal,m['name'],i,rules,m['K2'])
            else:
                samples[m['name']]=journal[m['name']]['original'][i]['samples']['0.7'][:m['K2']]
                sources[m['name']]=['original']*m['K2']
        win,stop,count=replay_record(greedy,samples,models)
        ref=journal[models[0]['name']]['original'][i]['reference'];bits.append(correct(ref,win));stops.append(stop);counts.append(count)
        variant_counts.append(sum(s!='original'for m in models for s in sources[m['name']][:(m['K1']if stop else m['K2'])]))
    return np.array(bits),stops,counts,variant_counts

def single(ids,journal,name,k,rules=None):
    bits=[]
    for i in ids:
        r=journal[name]['original'][i]
        samples=pools(journal,name,i,rules,k)[0]if rules else r['samples']['0.7'][:k]
        win=choose(dict(greedy=r['greedy'],samples={'0.7':samples}),dict(mode='vote',temperature=.7,k=k,greedy_vote=False))
        bits.append(correct(r['reference'],win))
    return np.array(bits)

def correlations(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if len(a)<2:return dict(n=len(a),within=None,cross=None,within_finite=0,cross_finite=0)
    def corr(x,y):
        x=x-x.mean();y=y-y.mean();den=float(np.linalg.norm(x)*np.linalg.norm(y))
        return float(x@y/den)if den>0 else None
    within=[corr(a[:,i],a[:,j])for i in range(8)for j in range(i)]
    cross=[corr(a[:,i],b[:,j])for i in range(8)for j in range(8)]
    w=[x for x in within if x is not None];c=[x for x in cross if x is not None]
    return dict(n=len(a),within=float(np.mean(w))if w else None,cross=float(np.mean(c))if c else None,
                within_finite=len(w),cross_finite=len(c),within_total=28,cross_total=64)

def diagnostics(ids,journal):
    result={}
    for name,branches in journal.items():
        result[name]={}
        for rule in RULES:
            g0=[];g1=[];p0=[];p1=[];e0=[];e1=[];mask=[]
            for i in ids:
                a=branches['original'][i];b=branches[rule].get(i,a);ref=a['reference'];mask.append(i in branches[rule])
                g0.append(correct(ref,a['greedy']));g1.append(correct(ref,b['greedy']))
                e0.append([not correct(ref,s)for s in a['samples']['0.7']]);e1.append([not correct(ref,s)for s in b['samples']['0.7']])
                p0.append(1-np.mean(e0[-1]));p1.append(1-np.mean(e1[-1]))
            mask=np.array(mask);e0=np.array(e0);e1=np.array(e1)
            result[name][rule]=dict(coverage=int(mask.sum()),all=dict(greedy=paired(g0,g1),per_sample=paired(p0,p1),error_correlation=correlations(e0,e1)),
                applicable=dict(greedy=paired(np.array(g0)[mask],np.array(g1)[mask]),per_sample=paired(np.array(p0)[mask],np.array(p1)[mask]),error_correlation=correlations(e0[mask],e1[mask])))
    return result

def run(raw,action):
    split='tune2048'if action=='tune'else'eval2048'
    selection_path=WORK/'results/tta-selection.json'
    # Freeze the selection before even loading eval journals.
    selection=read(selection_path)if action=='confirm'else None
    if selection:assert selection['raw_identity']==sha(raw/'tta/summary.json') and selection['selected_on']=='tune2048'
    ids,journal,models,receipts=load(raw,split)
    baseline,stops,counts,_=primary(ids,journal,models)
    rulesets=MIXES if action=='tune'else[tuple(selection['rules'])]
    rows=[]
    for rules in rulesets:
        bits,s,c,vc=primary(ids,journal,models,rules)
        assert s==stops and c==counts,'Sample budget/early-stop changed'
        rows.append(dict(rules=list(rules),**paired(baseline,bits),mean_samples=float(np.mean(c)),max_samples=max(c),
                         early_stops=sum(s),mean_variant_samples=float(np.mean(vc))))
    if action=='tune':
        chosen=max(enumerate(rows),key=lambda x:(x[1]['mixed_sum'],-x[0]))[1]
        selection=dict(selected_at_utc=datetime.now(timezone.utc).isoformat(),selected_on=split,rules=chosen['rules'],tune=chosen,
                       raw_identity=sha(raw/'tta/summary.json'),tune_journal_sha256=receipts,eval_journals_opened=False,
                       selection='max tune primary matches; declared order breaks ties')
        write(selection_path,selection)
    singles=[]
    for name in journal:
        for k in [4,8]:
            a=single(ids,journal,name,k)
            for rules in rulesets:singles.append(dict(model=name,K=k,rules=list(rules),**paired(a,single(ids,journal,name,k,rules))))
    result=dict(split=split,primary=rows,single_model=singles,diagnostics=diagnostics(ids,journal),journal_sha256=receipts,
                selected_rules=selection['rules'],original_matches=int(baseline.sum()),N=len(ids))
    if action=='confirm':
        row=rows[0];result['success']=row['delta_pp']>=.4 and row['ci95_pp'][0]>0
        result['selection_sha256']=sha(selection_path)
    write(WORK/f'results/tta-{action}.json',result)
    print(json.dumps(dict(split=split,primary=rows,selected_rules=selection['rules'],success=result.get('success')),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['tune','confirm']);p.add_argument('--raw',type=Path,required=True);a=p.parse_args();run(a.raw,a.action)
