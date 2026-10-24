"""Evaluate only sealed E2 primary versus v11 on the complete fresh GPU pool."""
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

WORK=Path(__file__).resolve().parents[1];ROOT=WORK/'kaggle'
sys.path[:0]=[str(WORK/'research'),str(ROOT/'src')]
spec=importlib.util.spec_from_file_location('e2_analysis',WORK/'research/e2-analysis.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
sha=A.sha;read=A.read;write=A.write
from final_split import fixed_split

def main(raw):
    target=WORK/'results/e2-confirm.json'
    assert not target.exists(),'Confirmation already written; never select another primary'
    seal=read(WORK/'results/e2-selection.json');selection_sha=sha(WORK/'results/e2-selection.json')
    assert selection_sha==read(WORK/'checks/e2-selection.json')['selection_sha256']
    assert sha(WORK/'research/e2_policies.py')==seal['policy_sha256']
    identity=read(ROOT/'e2-source-identity.json');config=identity['config']
    assert config['selection_sha256']==selection_sha
    assert read(raw/'e2/config.json')==config
    launch=read(WORK/'checks/e2-launch.json');design=read(WORK/'checks/e2-phase2-design.json')
    assert seal['at_utc']<launch['requested_at_utc']
    assert launch['notebook_sha256']==identity['notebook_sha256']
    assert launch['phase2_design_sha256']==sha(WORK/'checks/e2-phase2-design.json')
    assert design['fresh_sampling_on_shared_tasks']
    assert read(WORK/'checks/e2-download.json')['status']=='COMPLETE'
    for name,digest in identity['source_sha256'].items():assert sha(raw/'e2-src'/name)==digest
    for name in ('final_adaptive.py','final_ensemble.py','final_numeric.py','final_split.py'):
        assert sha(ROOT/'src'/name)==identity['source_sha256'][name],'Selector/split differs from frozen runtime'
    session=read(raw/'e2/summary.json')
    assert session['status']=='complete' and session['smoke_passed'] and session['test_read'] is False
    records=[];manifests={};summaries=[]
    expected={m+s:k for m in ('C','P','W') for s,k in (('/g',1),('/0.7',16),('/1.0',8))}
    for split in ('tune2048','eval2048'):
        folder=raw/'e2/full'/split;summary=read(folder/'summary.json');summaries.append(summary)
        assert summary==session['splits']['full/'+split]
        assert summary['status']=='complete' and summary['rows']==2048 and summary['candidates_per_row']==75
        assert summary['seed']==config['seed']!=read(ROOT/'e1-config.json')['seed']
        assert summary['input_sha256']==config['input_sha256']
        assert [m['sha256'] for m in summary['models']]==[m['sha256'] for m in config['models']]
        assert summary['vocabulary']['token_table_sha256']=='eb0080fcc2bda85916b20a1e76c6ef57c2c3460e1e3995fd9339644fdb49e713'
        assert summary['executor']['unique_programs']==summary['executor']['completed_unique']
        assert summary['cross_LL'] is summary['token_ensemble'] is summary['soups'] is False
        rows=[]
        for entry in summary['journal_manifest']:
            p=folder/entry['file'];assert p.parent==folder and sha(p)==entry['sha256']
            with gzip.open(p,'rt',encoding='utf8') as f:batch=[json.loads(line) for line in f]
            assert len(batch)==entry['rows'];rows+=batch;manifests[str(p.relative_to(raw))]=entry['sha256']
        assert [r['id'] for r in rows]==fixed_split()[split]
        for r in rows:
            assert r['split']==split and {k:len(v) for k,v in r['pools'].items()}==expected
            assert len({c['represented_code'] for c in r['candidates']})==len(r['candidates'])
            for c in r['candidates']:
                assert c['ll']=={} and c['execution']['status']!='budget_not_evaluated'
            for pool in r['pools'].values():
                assert all(0<=c['u']<len(r['candidates']) and math.isfinite(c['mean_logprob']) for c in pool)
        # Assert shared questions/references without selecting anything on the new labels.
        original=WORK/'results/kaggle-runs/nlp-hw02-e1-ensembles/4/e1/full'/split
        old={}
        for p in sorted(original.glob('pool-*.jsonl.gz')):
            with gzip.open(p,'rt',encoding='utf8') as f:
                for line in f:
                    r=json.loads(line);old[r['id']]=(r['question_sha256'],r['reference'])
        assert all(old[r['id']]==(r['question_sha256'],r['reference']) for r in rows)
        records+=rows
    assert len(records)==len({r['id'] for r in records})==len({r['question_sha256'] for r in records})==4096
    base=[];primary=[];usage=Counter();early=third=0;winners=[]
    for row in records:
        bw,bs,bt,bu,bf=A.predict(row,A.baseline())
        pw,ps,pt,pu,pf=A.predict(row,seal['primary'])
        assert bs==ps and not bt
        base.append(A.E1.correct(row['reference'],bw));primary.append(A.E1.correct(row['reference'],pw))
        usage.update(pu);early+=ps;third+=pt
        winners.append(dict(id=row['id'],baseline=bw['code'],primary=pw['code'],
            baseline_correct=base[-1],primary_correct=primary[-1],stage3=pt))
    stats=A.paired(base,primary);forecast=seal['result']['forecast_seconds']
    passed=stats['delta_pp']>=.3 and stats['ci95_pp'][0]>0 and forecast<=150
    out=dict(at_utc=A.now(),primary=seal['primary'],selection_sha256=selection_sha,
        **stats,baseline_accuracy=100*sum(base)/4096,primary_accuracy=100*sum(primary)/4096,
        forecast_seconds=forecast,forecast_frozen_before_phase2=True,
        criterion_passed=passed,independent_tasks=False,fresh_sampling=True,
        decision='candidate_v12_shared_tasks_only' if passed else 'retain_v11',
        limitation='Same heldout4k tasks used for selection in phase1; CI does not remove task-selection bias or capture repeated sampling variability.',
        early_stops=early,third_rows=third,mean_usage={k:v/4096 for k,v in usage.items()},
        session_seconds=session['seconds'],journal_sha256=manifests,notebook_sha256=identity['notebook_sha256'],
        winner_sha256=hashlib.sha256(json.dumps(winners,ensure_ascii=False).encode()).hexdigest(),
        OOM=[s['OOM'] for s in summaries],
        peak_reserved_bytes=[s['peak_reserved_bytes'] for s in summaries])
    with gzip.open(raw/'e2-comparison.jsonl.gz','wt',encoding='utf8') as f:
        for row in winners:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write(target,out)
    print(json.dumps({k:v for k,v in out.items() if k not in ('journal_sha256',)},ensure_ascii=False))

if __name__=='__main__':
    assert len(sys.argv)==2,'Pass complete downloaded E2 run directory'
    main(Path(sys.argv[1]))
