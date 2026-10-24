"""Verify the interrupted, sealed E2 selection without altering it or its inputs."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys

WORK=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(WORK/'research'),str(WORK/'kaggle/src')]
spec=importlib.util.spec_from_file_location('e2_analysis',WORK/'research/e2-analysis.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
sha=A.sha;read=A.read

def main():
    sealed=read(WORK/'results/e2-selection.json')
    receipt=read(WORK/'checks/e2-selection.json')
    assert sha(WORK/'results/e2-selection.json')==receipt['selection_sha256']
    for key,path in [('analyzer_sha256','research/e2-analysis.py'),
                     ('policy_sha256','research/e2_policies.py'),
                     ('e1_analyzer_sha256','research/e1-analysis.py')]:
        assert sealed[key]==sha(WORK/path)
    prereg=read(WORK/'checks/e2-preregistration.json')
    assert hashlib.sha256(prereg['text'].encode()).hexdigest()==sealed['preregistration_sha256']==prereg['sha256']
    assert prereg['at_utc']<sealed['at_utc'] and not prereg['E2_accuracy_analyzed']
    raw=WORK/'results/kaggle-runs/nlp-hw02-e1-ensembles/4'
    assert sha(raw/'e1/summary.json')==sealed['raw_session_sha256']
    rows=[];manifests={}
    for split in ('tune2048','eval2048'):
        rr,ss,mm=A.E1.load(raw,split);rows+=rr;manifests.update(mm)
    assert manifests==sealed['journal_sha256']
    assert len(rows)==len({r['id'] for r in rows})==len({r['question_sha256'] for r in rows})==4096
    saved=read(WORK/'results/e2-phase1.json');control=[];winners=[]
    for row in rows:
        w,stop,third,used,f=A.predict(row,A.baseline())
        old,oldstop,oldused,_=A.E1.predict(row,dict(anchor='C/g',limits=[1,8,8]))
        assert (w['code'],w['execution'],stop,used)==(old['code'],old['execution'],oldstop,A.E1.costs(oldused))
        control.append(A.E1.correct(row['reference'],w))
    tm=sealed['time_model']
    for p,expected in zip(A.grid(),saved['rows'],strict=True):
        assert p==expected['policy']
        bits=[];usage=Counter();early=third_count=0;winners=[]
        for row in rows:
            w,stop,third,used,f=A.predict(row,p)
            bits.append(A.E1.correct(row['reference'],w));usage.update(used)
            early+=stop;third_count+=third;winners.append((row['id'],w['code'],w['execution']))
        assert A.paired(control,bits)=={k:expected[k] for k in A.paired(control,bits)}
        assert early==expected['early_stops'] and third_count==expected['third_rows']
        assert {k:v/4096 for k,v in usage.items()}==expected['mean_usage']
        assert hashlib.sha256(json.dumps(winners,ensure_ascii=False).encode()).hexdigest()==expected['winner_sha256']
        forecast=tm['fixed_seconds']+A.raw_extra(expected['mean_usage'],tm['rates'])*1919*tm['calibration']+expected['extra_barrier_seconds']
        assert forecast==expected['forecast_seconds']
    eligible=[r for r in saved['rows'] if r['forecast_seconds']<=150]
    assert min(eligible,key=lambda r:(-r['matches'],r['forecast_seconds'],r['policy']['name']))==sealed['result']
    result=dict(passed=True,at_utc=A.now(),selection_sha256=receipt['selection_sha256'],
        preregistration_precedes_selection=True,policies_verified=252,rows=4096,
        baseline_production_parity_rows=4096,immutable_selection_preserved=True,
        independent_tasks=False,primary=sealed['primary'],forecast_seconds=sealed['result']['forecast_seconds'])
    A.write(WORK/'checks/e2-resume-verification.json',result);print(json.dumps(result))

if __name__=='__main__':main()
