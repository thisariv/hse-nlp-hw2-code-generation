"""CPU retrieval substitution, similarity, skeleton and saved A error diagnostics."""
import time
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import ks_2samp
from tokenizers import Tokenizer
from common import *

def distribution(x):
    return dict(n=len(x),mean=float(np.mean(x)),quantiles={str(p):float(np.quantile(x,p)) for p in [0,.1,.25,.5,.75,.9,.95,1]},
                ge={str(p):float(np.mean(x>=p)) for p in [.5,.7,.8,.9,.95,.99,.999]})

def score_flags(flags,lengths):
    return dict(matches=int(flags.sum()),n=len(flags),accuracy=float(flags.mean()),
                by_reference_length={k:dict(n=int((np.array(lengths)==k).sum()),
                    matches=int(flags[np.array(lengths)==k].sum()),
                    accuracy=float(flags[np.array(lengths)==k].mean())) for k in ['<60','60-90','90-120','>=120']})

def main():
    start=time.monotonic();meta=read(LOCAL/'retrieval-index.json');sizes=meta['index_sizes']
    sets={s:read(INPUT/(s+'.json')) for s in ['heldout4k','old1024','test']}
    queries=[r['question'] for rr in sets.values() for r in rr]
    zs={n:np.load(LOCAL/f'retrieval-{n}.npz') for n in sizes}
    required=np.unique(np.concatenate([z['ids'].flatten() for z in zs.values()]));required=required[required<2**62]
    cp=LOCAL/'neighbors.json'
    if cp.exists():neighbors={int(k):v for k,v in read(cp).items()}
    else:
        neighbors={}
        for shard in range(60):
            lo,hi=np.searchsorted(required,[shard*200000,(shard+1)*200000])
            if lo==hi:continue
            table=pq.read_table(RAW/f'part-{shard:05d}.parquet').to_pylist()
            for i in required[lo:hi]:neighbors[int(i)]=table[int(i-shard*200000)]
        write(cp,neighbors)
    print('neighbors',len(neighbors),flush=True)
    tok=Tokenizer.from_file(str(INPUT/'tokenizer.json'))
    lengths={s:[bucket(len(e.ids)) for e in tok.encode_batch([r['code'] for r in rr])] for s,rr in sets.items() if s!='test'}
    refs={s:read(INPUT/(s+'-reference-execution.json'))['rows'] for s in ['heldout4k','old1024']}
    refsk={s:[skeleton(r['code']) for r in rr] for s,rr in sets.items() if s!='test'}
    neighborsk={i:skeleton(r['code']) for i,r in neighbors.items()}
    allcodes=[];grids={};reason_counts={}
    for n,z in zs.items():
        for rule in ['order','role']:
            grid=[];reasons=Counter()
            for j,q in enumerate(queries[:5120]):
                row=[]
                for rid in z['ids'][j]:
                    r=neighbors[int(rid)]
                    code,reason=substitute(r['code'],r['question'],q,rule)
                    row.append(code);reasons[reason]+=1
                    if code is not None:allcodes.append(code)
                grid.append(row)
            grids[(n,rule)]=grid;reason_counts[f'{n}-{rule}']=dict(reasons)
    allcodes=list(dict.fromkeys(allcodes));xp=LOCAL/'retrieval-execution.json'
    if xp.exists():exmap=read(xp)
    else:
        print('execute unique candidates',len(allcodes),flush=True)
        ex=execute(allcodes);exmap=dict(zip(allcodes,ex['rows']));write(xp,exmap)
    report=dict(index=meta,substitution_reasons=reason_counts,unique_programs_executed=len(allcodes),by_index={})
    refsizes=[len(sets[s]) for s in sets]
    gen=read(S1/'gpu0-a26/points/p050000/evaluation/heldout4k-generations.json')
    gx=read(S1/'gpu0-a26/points/p050000/evaluation/heldout4k-execution.json')['rows']
    correct=np.array([same(x,y) for x,y in zip(gx,refs['heldout4k'])]);assert correct.sum()==2538
    errors=~correct
    for n,z in zs.items():
        result={};skflags={}
        for s,lo,hi in [('heldout4k',0,4096),('old1024',4096,5120)]:
            eligible=np.array([r['status']=='numeric' for r in refs[s]])
            sk=np.array([[neighborsk[int(i)]==refsk[s][j] for i in z['ids'][lo+j]] for j in range(hi-lo)])
            skflags[s]=sk;result[s]={}
            for rule in ['order','role']:
                grid=grids[(n,rule)][lo:hi]
                flags=np.array([[code is not None and same(exmap[code],refs[s][j]) for code in rr] for j,rr in enumerate(grid)])
                result[s][rule]={}
                for k in [1,5,20]:
                    f=flags[:,:k].any(axis=1)
                    result[s][rule][f'top{k}']=score_flags(f[eligible],np.array(lengths[s])[eligible])
                    if s=='heldout4k':
                        result[s][rule][f'top{k}']['A_union_oracle_matches']=int((f|correct).sum())
                        result[s][rule][f'top{k}']['A_errors_recovered_oracle']=int((f&errors).sum())
            result[s]['skeleton']={f'top{k}':score_flags(sk[:,:k].any(axis=1)[eligible],np.array(lengths[s])[eligible]) for k in [1,5,20]}
        result['similarity']={s:distribution(z['scores'][lo:hi,0]) for s,lo,hi in [('heldout4k',0,4096),('old1024',4096,5120),('test',5120,7039)]}
        ks=ks_2samp(z['scores'][:4096,0],z['scores'][5120:,0]);result['test_vs_heldout_ks']=dict(statistic=float(ks.statistic),pvalue=float(ks.pvalue))
        result['A_error_neighbor_skeleton']={f'top{k}':dict(n=int(errors.sum()),matches=int(skflags['heldout4k'][errors,:k].any(axis=1).sum()),
                    by_reference_length=score_flags(skflags['heldout4k'][errors,:k].any(axis=1),np.array(lengths['heldout4k'])[errors])['by_reference_length']) for k in [1,5,20]}
        report['by_index'][str(n)]=result
    # Literal diagnostics are overlapping, plus a declared exclusive priority.
    tags=[];exclusive=[];literal_notes=[]
    for i,(r,g,x) in enumerate(zip(sets['heldout4k'],gen,gx)):
        if correct[i]:tags.append([]);exclusive.append('correct');continue
        refvals=literals(r['code']);predvals=literals(g['generated_code']);info=number_info(r['question'],g['generated_code'])
        qvals={a[0] for a in question_numbers(r['question'])};tt=[]
        if x['status']!='numeric':tt.append('runtime_or_non_numeric')
        if info.get('counts',{}).get('other',0):tt.append('number_not_from_question_or_known_constant')
        if predvals is not None:
            requiredq={v for v in refvals if v in qvals}
            if requiredq-set(predvals):tt.append('used_question_number_missing')
            if set(predvals)==set(refvals):tt.append('same_literal_set_wrong_program')
        if not tt:tt.append('other_numeric_or_structure')
        tags.append(tt);exclusive.append(tt[0])
    report['A_errors']=dict(total=4096,correct=2538,errors=int(errors.sum()),
        statuses=dict(Counter(x['status'] for i,x in enumerate(gx) if errors[i])),
        overlapping_tags={k:dict(n=sum(k in t for t in tags),fraction_of_errors=sum(k in t for t in tags)/errors.sum()) for k in sorted({v for tt in tags for v in tt})},
        exclusive_priority_counts=dict(Counter(x for x in exclusive if x!='correct')),
        by_reference_length={k:dict(n=sum(l==k for l in lengths['heldout4k']),errors=sum(l==k and errors[i] for i,l in enumerate(lengths['heldout4k'])),
            exclusive=dict(Counter(exclusive[i] for i,l in enumerate(lengths['heldout4k']) if l==k and errors[i]))) for k in ['<60','60-90','90-120','>=120']},
        caveat='missing counts only digit numbers actually used by reference; irrelevant question numbers excluded. Outside-number classification discounts explicit words/percent and heuristic constants. Tags overlap.')
    # JSON cannot serialize NumPy ints; normalize the small aggregate tree.
    report=json.loads(json.dumps(report,default=lambda x:x.item() if isinstance(x,np.generic) else x))
    report['seconds']=time.monotonic()-start;write(LOCAL/'retrieval-and-errors.json',report)
    write(LOCAL/'A-error-labels.json',dict(tags=tags,exclusive=exclusive))
    print('finished seconds',report['seconds'],flush=True)

if __name__=='__main__':main()
