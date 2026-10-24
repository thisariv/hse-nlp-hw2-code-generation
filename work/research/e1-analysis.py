"""E1 tune selection, then independent eval confirmation in a separate process."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

WORK=Path(__file__).resolve().parents[1];ROOT=WORK/'kaggle'
sys.path[:0]=[str(ROOT/'src'),str(WORK/'research')]
from final_split import fixed_split
from final_ensemble import replay_record
from e1_policies import grid,predict,materialize,costs

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def now():return datetime.now(timezone.utc).isoformat()
def prereg_hash():return hashlib.sha256((WORK/'notes/e1-ensembles.md').read_text().split('## CPU-проверки')[0].encode()).hexdigest()
def correct(ref,c):
    x=c['execution']
    return ref['status']==x['status']=='numeric'and math.isclose(ref['value'],x['value'],rel_tol=1e-6,abs_tol=1e-6)


def paired(a,b):
    a=np.asarray(a,dtype=np.int8);b=np.asarray(b,dtype=np.int8);d=b-a;n=len(d)
    values,counts=np.unique(d,return_counts=True)
    draws=np.random.default_rng(2026101904).multinomial(n,counts/n,size=20000)
    boot=draws@values/n*100
    return dict(n=n,baseline_matches=int(a.sum()),matches=int(b.sum()),accuracy_pp=float(b.mean()*100),
        delta_pp=float(d.mean()*100),ci95_pp=np.quantile(boot,[.025,.975]).tolist(),
        gains=int((d>0).sum()),losses=int((d<0).sum()),bootstrap='paired tasks;20000;seed2026101904;percentile95')


def load(raw,split):
    """Tune action never opens any eval pool or reference input file."""
    identity=read(ROOT/'e1-source-identity.json');launch=read(WORK/'checks/e1-launch.json')
    assert launch['preregistration_sha256']==prereg_hash(),'Preregistration changed'
    assert launch['grid_sha256']==sha(WORK/'research/e1_policies.py'),'Policy grid changed'
    assert launch['timing_sha256']==sha(ROOT/'e1-timing-reference.json')
    assert launch['notebook_sha256']==identity['notebook_sha256']
    assert read(raw/'e1/config.json')==identity['config']
    for name,digest in identity['source_sha256'].items():assert sha(raw/'e1-src'/name)==digest,name
    session=read(raw/'e1/summary.json')
    assert session['status']=='complete'and session['smoke_passed']and session['test_read']is False
    summary=read(raw/'e1/full'/split/'summary.json')
    assert summary['status']=='complete'and summary['rows']==2048 and summary['candidates_per_row']==87
    assert summary['input_sha256']['heldout4k.json']=='28a3c0c9088c5b4a17d75d216e4abc4262cc9ca62884966c43e7ec09b6f040c2'
    assert summary['input_sha256']['heldout4k-reference-execution.json']=='3855680e9a37292aa2e351b8e8e4f7a5cbbe76444096d136470e4a9be8797111'
    assert [x['sha256']for x in summary['models']]==[x['sha256']for x in identity['config']['models']]
    assert summary['executor']['unique_programs']==summary['executor']['completed_unique']
    assert summary['vocabulary']['token_table_sha256']=='eb0080fcc2bda85916b20a1e76c6ef57c2c3460e1e3995fd9339644fdb49e713'
    rows=[];receipts={}
    for entry in summary['journal_manifest']:
        p=raw/'e1/full'/split/entry['file'];assert p.parent==raw/'e1/full'/split
        assert sha(p)==entry['sha256'];receipts[str(p.relative_to(raw))]=entry['sha256']
        with gzip.open(p,'rt',encoding='utf8')as f:batch=[json.loads(line)for line in f]
        assert len(batch)==entry['rows'];rows.extend(batch)
    assert [r['id']for r in rows]==fixed_split()[split]
    expected={m+'/g':1 for m in ('C','P','W','E','S0.3','S0.5','S0.7')}
    expected.update({m+'/0.7':16 for m in ('C','P','W')});expected['E/0.7']=8
    expected.update({m+'/1.0':8 for m in ('C','P','W')})
    for r in rows:
        assert r['split']==split and {k:len(v)for k,v in r['pools'].items()}==expected
        assert len({c['represented_code']for c in r['candidates']})==len(r['candidates'])
        for c in r['candidates']:
            assert c['execution']['status']!='budget_not_evaluated'and set(c['ll'])=={'C','P','W'}
            for score in c['ll'].values():
                assert math.isfinite(score['sum'])and math.isfinite(score['mean'])
                assert abs(score['sum']-score['mean']*c['ll_length'])<.01
        for entries in r['pools'].values():
            for c in entries:assert 0<=c['u']<len(r['candidates'])and math.isfinite(c['mean_logprob'])
    return rows,summary,receipts


def time_model(summary):
    totals=defaultdict(lambda:[0.,0])
    ll=defaultdict(lambda:[0.,0])
    for row in summary['timings']:
        if row['kind']=='generation':
            key=(row['label'],row['K']);totals[key][0]+=row['seconds'];totals[key][1]+=row['sequences']
        elif row['kind']=='LL':
            ll[row['label']][0]+=row['seconds'];ll[row['label']][1]+=row['rows']
    rates={f'{name}:K{k}':seconds/n for (name,k),(seconds,n)in totals.items()}
    reference=read(ROOT/'e1-timing-reference.json')
    counts=reference['mean_samples_per_model']
    raw_ref=rates['C/g:K1']
    for short,long in [('C','C-long2x-B6'),('P','pilot-b'),('W','W-B6')]:
        raw_ref+=rates[short+'/0.7:K1']+(counts[long]-1)*rates[short+'/0.7:K15']
    calibration=reference['generation_seconds']/(1919*raw_ref)
    return dict(rates=rates,ll_seconds_per_unique=sum(s/n for s,n in ll.values()),
        generation_calibration=calibration,fixed_seconds=reference['fixed_seconds'],
        reference_sha256=sha(ROOT/'e1-timing-reference.json'),soup_setup_seconds=summary['soup_setup_seconds'])


def forecast(p,usage,scored,model,n):
    rates=model['rates'];raw=0.
    for row in usage:
        for pool,count in row.items():
            if pool.endswith('/0.7'):
                extra=7 if pool.startswith('E/')else 15
                raw+=rates[pool+':K1']+max(count-1,0)*rates[pool+f':K{extra}']
            elif pool.endswith('/1.0'):
                raw+=count*rates[pool+':K8']
            else:raw+=count*rates[pool+':K1']
    raw*=1919/n
    ll_seconds=sum(scored)*1919/n*model['ll_seconds_per_unique']
    soup=p.get('soup',p['anchor'])
    soup_seconds=model['soup_setup_seconds'][soup[1:].split('/')[0]]if soup.startswith('S')else 0.
    seconds=model['fixed_seconds']+raw*model['generation_calibration']+ll_seconds+soup_seconds
    return dict(test1919_seconds=seconds,under_155=seconds<=155.,under_140=seconds<=140.,
        uncalibrated_generation_seconds=raw,generation_calibration=model['generation_calibration'],
        fixed_seconds=model['fixed_seconds'],LL_seconds=ll_seconds,soup_seconds=soup_seconds,
        kind='tune-throughput calibrated to existing v10; forecast, not a cold test')


def evaluate(rows,p):
    bits=[];stops=[];usage=[];scored=[];winners=[]
    for r in rows:
        w,stop,used,score=predict(r,p)
        bits.append(correct(r['reference'],w));stops.append(stop);usage.append(costs(used));scored.append(len(score))
        winners.append((r['id'],w['code'],w['execution']))
    return bits,dict(early_stops=sum(stops),mean_generated_candidates=float(np.mean([sum(x.values())for x in usage])),
        mean_scored_unique=float(np.mean(scored)),winner_sha256=hashlib.sha256(json.dumps(winners,ensure_ascii=False).encode()).hexdigest()),usage,scored


def baseline_parity(rows):
    models=[dict(name=m,K1=1,K2=k)for m,k in zip(('C','P','W'),(1,5,6))]
    for r in rows:
        greedy=materialize(r,'C/g',1,True)[0]
        samples={m:materialize(r,m+'/0.7')for m in ('C','P','W')}
        a,stop,n=replay_record(greedy,samples,models)
        b,s,used,_=predict(r,grid()[0])
        assert (a['code'],a['execution'],stop,n)==(b['code'],b['execution'],s,len(used)-1)
    return dict(rows=len(rows),winner_code_execution_exact=True,early_stop_counts_exact=True)


def run(raw,action):
    selection_path=WORK/'results/e1-selection.json'
    if action=='confirm':
        selection=read(selection_path)
        seal=read(WORK/'checks/e1-selection.json')
        assert sha(selection_path)==seal['selection_sha256'],'Selection changed after sealing'
        assert selection['selected_on']=='tune2048'and selection['analyzer_sha256']==sha(Path(__file__))
        assert selection['raw_session_sha256']==sha(raw/'e1/summary.json')
        split='eval2048'
    else:
        assert not selection_path.exists(),'Tune selection already frozen; do not overwrite'
        selection=None;split='tune2048'
    rows,summary,receipts=load(raw,split)
    parity=baseline_parity(rows)
    baseline=evaluate(rows,grid()[0])[0]
    tm=time_model(summary)if action=='tune'else selection['time_model']
    results=[]
    for p in grid():
        bits,diagnostics,usage,scored=evaluate(rows,p)
        timing=forecast(p,usage,scored,tm,len(rows))if action=='tune'else selection['timings'][p['name']]
        results.append(dict(policy=p,**paired(baseline,bits),**diagnostics,timing=timing))
    result=dict(split=split,at_utc=now(),rows=results,baseline_parity=parity,journal_sha256=receipts,
                raw_session_sha256=sha(raw/'e1/summary.json'),analyzer_sha256=sha(Path(__file__)))
    # Collected individual greedies are diagnostics, never additional fitted policies.
    result['individual_greedy']={m:paired(baseline,[correct(r['reference'],materialize(r,m+'/g',1,True)[0])
        for r in rows])for m in ('C','P','W')}
    if action=='tune':
        def rank(r):return -r['matches'],r['timing']['test1919_seconds'],r['policy']['name']
        eligible=[r for r in results if r['timing']['under_155']]
        assert eligible,'No policy fits155s; inspect frozen time model without reading eval'
        primary=min(eligible,key=rank)
        family={}
        for f in sorted({r['policy']['family']for r in results}):
            group=[r for r in eligible if r['policy']['family']==f]
            family[f]=min(group,key=rank)['policy']['name']if group else None
        selection=dict(selected_on='tune2048',at_utc=now(),primary=primary['policy']['name'],
            family_best=family,time_model=tm,timings={r['policy']['name']:r['timing']for r in results},
            raw_session_sha256=result['raw_session_sha256'],preregistration_sha256=prereg_hash(),
            grid_sha256=sha(WORK/'research/e1_policies.py'),analyzer_sha256=sha(Path(__file__)),
            tune_journal_sha256=receipts,rule='max tune matches;time<=155;ties:min time,name;one primary')
        write(WORK/'results/e1-tune.json',result)
        write(selection_path,selection)
        write(WORK/'checks/e1-selection.json',dict(at_utc=now(),selection_sha256=sha(selection_path),
            primary=selection['primary'],eval_pool_opened=False,analyzer_sha256=sha(Path(__file__))))
        print(json.dumps(dict(selected=selection['primary'],family_best=family,selection_sha256=sha(selection_path))))
    else:
        primary=next(r for r in results if r['policy']['name']==selection['primary'])
        success=primary['delta_pp']>=.4 and primary['ci95_pp'][0]>0 and primary['timing']['under_155']
        result.update(primary=selection['primary'],primary_success=success,selection_sha256=sha(selection_path),
            decision='prepare v11 policy; cold verification still required'if success else'keep v10; E1 success criterion not met',
            family_best_frozen=selection['family_best'])
        write(WORK/'results/e1-confirm.json',result)
        print(json.dumps(dict(primary=primary['policy']['name'],success=success,eval=primary,decision=result['decision']),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['tune','confirm']);parser.add_argument('raw',type=Path)
    args=parser.parse_args();run(args.raw,args.action)
