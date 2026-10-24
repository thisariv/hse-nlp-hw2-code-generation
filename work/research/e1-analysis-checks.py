"""CPU-only selector, bootstrap and timing calibration checks; no eval files."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import random
import sys

WORK=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(WORK/'kaggle/src'),str(WORK/'research')]
spec=importlib.util.spec_from_file_location('e1_analysis',WORK/'research/e1-analysis.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
from e1_policies import grid,predict,ll_select


def main():
    rng=random.Random(4021)
    policies=grid();assert len(policies)==len({p['name']for p in policies})==33
    records=[]
    expected={m+'/g':1 for m in ('C','P','W','E','S0.3','S0.5','S0.7')}
    expected.update({m+'/0.7':16 for m in ('C','P','W')});expected['E/0.7']=8
    expected.update({m+'/1.0':8 for m in ('C','P','W')})
    for i in range(128):
        candidates=[]
        for j in range(10):
            # Includes nontransitive isclose chains and execution failures.
            value=[0.,.9e-6,1.8e-6,2.,3.][j%5]
            candidates.append(dict(code=str(j),represented_code=str(j),mean_logprob=-j-1.,length=j+1,
                execution=dict(status='numeric',value=value)if j<8 else dict(status='error'),
                ll={m:dict(sum=-j*3.-1,mean=(-j*3.-1)/(j+1))for m in ('C','P','W')}))
        pools={}
        for pool,n in expected.items():
            pools[pool]=[dict(u=3 if i%8==0 else rng.randrange(10),mean_logprob=-rng.random()*5,ended=True)for _ in range(n)]
        records.append(dict(id=i,candidates=candidates,pools=pools,reference=dict(status='numeric',value=2.)))
    parity=A.baseline_parity(records)
    for p in policies:
        bits,diagnostics,usage,scored=A.evaluate(records,p)
        assert len(bits)==128 and len(usage)==len(scored)==128
        if p['family']=='budget':
            for row,use in zip(records,usage):
                _,stop,_,_=predict(row,p)
                if stop:assert sum(use.values())==4
        # Exact task labels cannot influence a selector.
        r=records[1];a=predict(r,p)[0];r['reference']=dict(status='numeric',value=999.)
        assert predict(r,p)[0]==a
    def candidate(uid,value,score):
        return dict(uid=uid,code=str(uid),mean_logprob=-1.,execution=dict(status='numeric',value=value),
                    ll={m:dict(sum=score,mean=score)for m in ('C','P','W')})
    anchor=candidate(0,1.,-20.);a=candidate(1,1.,-20.);b=candidate(2,2.,-1.);c=candidate(3,2.,-1.)
    assert ll_select(anchor,[a,b,c],'tie-sum')['execution']['value']==2.
    assert ll_select(anchor,[a,b,c],'weight',.25)['execution']['value']==2.
    assert A.paired([0]*100,[1]*100)['ci95_pp']==[100.,100.]
    assert A.paired([1,0],[0,1])['delta_pp']==0.
    timing=[]
    for pool,n in expected.items():
        ks=[1,7 if pool.startswith('E/')else 15]if pool.endswith('/0.7')else[8 if pool.endswith('/1.0')else 1]
        for k in ks:timing.append(dict(kind='generation',label=pool,K=k,seconds=k*10.,sequences=k*2048))
    for m in ('C','P','W'):timing.append(dict(kind='LL',label=m,seconds=1.,rows=100))
    summary=dict(timings=timing,soup_setup_seconds={str(x):1. for x in (.3,.5,.7)})
    tm=A.time_model(summary)
    usage=[Counter({'C/g':1,'C/0.7':1,'P/0.7':5 if i<735 else 1,'W/0.7':6 if i<735 else 1})for i in range(1919)]
    reference=A.read(WORK/'kaggle/e1-timing-reference.json')
    seconds=A.forecast(grid()[0],usage,[0]*1919,tm,1919)['test1919_seconds']
    assert abs(seconds-reference['full_pipeline_seconds'])<1e-8,(seconds,reference['full_pipeline_seconds'])
    result=dict(passed=True,policies=33,synthetic_rows=128,baseline_parity=parity,
        LL_weight_and_tie=True,task_bootstrap=True,v10_time_reconstruction_seconds=seconds,eval_read=False)
    A.write(WORK/'checks/e1-analysis-cpu.json',result);print(json.dumps(result))


if __name__=='__main__':main()
