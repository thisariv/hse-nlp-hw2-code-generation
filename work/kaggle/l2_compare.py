#!/usr/bin/env python3
"""CPU paired greedy report and fixed C-main K4 + C-long2x K4 voting."""
import argparse, json, math, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import write
from l2_sampler import fixed_split

def paired(a,b,ids):
    gains=sum(not a[i]['correct'] and b[i]['correct'] for i in ids)
    losses=sum(a[i]['correct'] and not b[i]['correct'] for i in ids); n=len(ids)
    delta=(gains-losses)/n if n else 0.; variance=(gains+losses)/n-delta**2 if n else 0.
    half=1.96*math.sqrt(max(0.,variance)/max(1,n))
    return {'n':n,'gains':gains,'losses':losses,'delta_pp':100*delta,'paired_95CI_pp':[100*(delta-half),100*(delta+half)]}

def compare(working):
    working=Path(working);result={'points':{}}
    for a in sorted((working/'model-C-main/points').glob('*/complete.json')):
        b=working/'model-C-long2x/points'/a.parent.name/'complete.json'
        if not b.exists():continue
        aa=json.loads(a.read_text());bb=json.loads(b.read_text());ta=aa['training'];tb=bb['training']
        if abs(ta['l2_main_seconds']-tb['l2_main_seconds'])>5:continue
        if ta['cooldown_target_seconds']!=tb['cooldown_target_seconds']:continue
        pa=a.parent/'evaluation';pb=b.parent/'evaluation'
        x=json.loads((pa/'heldout4k-per-example.json').read_text());y=json.loads((pb/'heldout4k-per-example.json').read_text())
        assert [(r['id'],r['question_sha256']) for r in x]==[(r['id'],r['question_sha256']) for r in y]
        split=fixed_split()
        subsets={**split,'heldout4k':list(range(len(x))),'ge90':[r['id'] for r in x if r['original_bpe_length']>=90],
                 'lt60':[r['id'] for r in x if r['original_bpe_length']<60]}
        subsets['tune-ge90']=[i for i in split['tune2048'] if i<len(x) and x[i]['original_bpe_length']>=90]
        subsets['tune-lt60']=[i for i in split['tune2048'] if i<len(x) and x[i]['original_bpe_length']<60]
        result['points'][a.parent.name]={k:paired(x,y,[i for i in v if i<len(x)]) for k,v in subsets.items()}
        olda=pa/'old991-per-example.json';oldb=pb/'old991-per-example.json'
        if olda.exists() and oldb.exists():
            ox=json.loads(olda.read_text());oy=json.loads(oldb.read_text())
            assert [(r['id'],r['question_sha256']) for r in ox]==[(r['id'],r['question_sha256']) for r in oy]
            result['points'][a.parent.name]['old991']=paired(ox,oy,list(range(len(ox))))
        # Choosing recipes is a human tune-only decision; eval stored for descriptive paired audit.
    return result

def mixed(a,b):
    import exp27_decode as D
    import l2_eval as E
    def load(root):
        rows=[r for p in sorted(Path(root).glob('batch-*.json')) for r in json.loads(p.read_text())]
        assert len({r['id'] for r in rows})==len(rows)
        return {r['id']:r for r in rows}
    x,y=load(a),load(b);ids=sorted(set(x)&set(y));output=[]
    for i in ids:
        assert x[i]['question_sha256']==y[i]['question_sha256'] and x[i]['reference']==y[i]['reference']
        pool=x[i]['samples']['0.7'][:4]+y[i]['samples']['0.7'][:4]
        row=dict(x[i],samples={'0.7':[c for c in pool if c['mean_logprob'] is not None]})
        chosen=D.choose(row,{'mode':'vote','temperature':.7,'k':8,'greedy_vote':False})
        output.append({'id':i,'execution':chosen['execution'],'correct':E.matches(row['reference'],chosen['execution'])})
    return {'rows':len(ids),'complete_tune2048':len(ids)==2048,'mixed_matches':sum(r['correct'] for r in output),
            'C_main_K8_matches':sum(x[i]['correct'] for i in ids),'C_long_K8_matches':sum(y[i]['correct'] for i in ids),
            'per_example':output,'GPU':False,'production_timer_established':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--working');p.add_argument('--main-k8');p.add_argument('--long-k8');p.add_argument('--out',required=True);a=p.parse_args()
    write(a.out,mixed(a.main_k8,a.long_k8) if a.main_k8 else compare(a.working))
