"""Execute every repeated-question reference in full train, CPU only, resumable."""
import hashlib
import time
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from common import *

def summarize(groups, values, statuses, raw_rows):
    order=np.argsort(groups,kind='stable');gg=groups[order];vv=values[order];ss=statuses[order]
    bounds=np.r_[0,np.flatnonzero(gg[1:]!=gg[:-1])+1,len(gg)]
    stats=Counter();modal_sum=0;group_rates=[];pair_agree=pair_total=0;finite_rows=0
    finite_rows_ge2=modal_sum_ge2=0;group_pair_rates=[]
    for a,b in zip(bounds[:-1],bounds[1:]):
        v=vv[a:b][ss[a:b]==0];n=len(v);finite_rows+=n
        stats['duplicate_groups']+=1
        if n!=b-a:stats['groups_with_non_numeric']+=1
        if n<2:
            stats['less_than_two_finite']+=1
            modal_sum+=n
            continue
        counts=[];reps=[]
        for x in v:
            for i,r in enumerate(reps):
                if math.isclose(x,r,rel_tol=1e-6,abs_tol=1e-6):
                    counts[i]+=1;break
            else:reps.append(float(x));counts.append(1)
        m=max(counts);modal_sum+=m;group_rates.append(m/n)
        finite_rows_ge2+=n;modal_sum_ge2+=m
        stats['at_least_two_finite']+=1
        stats['unanimous_finite']+=len(counts)==1
        stats['conflicting_finite']+=len(counts)>1
        pair_agree+=sum(c*(c-1)//2 for c in counts);pair_total+=n*(n-1)//2
        group_pair_rates.append(sum(c*(c-1)//2 for c in counts)/(n*(n-1)//2))
    return dict(counts=dict(stats),finite_duplicate_rows=finite_rows,
                pair_agreement=pair_agree/pair_total if pair_total else None,pairs=pair_total,
                modal_duplicate_row_ceiling=modal_sum/finite_rows,
                finite_rows_ge2=finite_rows_ge2,modal_rows_ge2_ceiling=modal_sum_ge2/finite_rows_ge2,
                pair_agreement_question_macro=float(np.mean(group_pair_rates)),
                modal_group_mean=float(np.mean(group_rates)),
                observed_conflict_loss_rows=finite_rows-modal_sum,
                all_train_row_optimistic_ceiling=1-(finite_rows-modal_sum)/raw_rows,
                caveat='all-train ceiling assumes singleton and nonnumeric rows perfectly predictable; duplicate ceiling is conditional on repeated questions and numeric references')

def main():
    start=time.monotonic()
    membership={k:np.load(LOCAL/(k+'-duplicate-members.npz')) for k in ['exact','normal']}
    union=membership['normal']['row'];statuses=np.full(len(union),255,np.uint8);values=np.zeros(len(union))
    names=['numeric'];offset=0
    # Verify text equality, rather than assume truncated hashes are collision free.
    seen={k:{} for k in membership}
    exactpos=np.searchsorted(union,membership['exact']['row']);assert np.array_equal(union[exactpos],membership['exact']['row'])
    for shard in range(60):
        p=RAW/f'part-{shard:05d}.parquet';n=pq.ParquetFile(p).metadata.num_rows
        lo,hi=np.searchsorted(union,[offset,offset+n]);ids=union[lo:hi]-offset
        table=pq.read_table(p).to_pylist();selected=[table[int(i)] for i in ids]
        for key,m in membership.items():
            aa,bb=np.searchsorted(m['row'],[offset,offset+n])
            for rid,g in zip(m['row'][aa:bb],m['group'][aa:bb]):
                q=table[int(rid-offset)]['question'];q=q if key=='exact' else ' '.join(q.lower().split())
                if int(g) in seen[key]:assert seen[key][int(g)]==q
                else:seen[key][int(g)]=q
        cp=LOCAL/f'noise-execution-{shard:02d}.json'
        if cp.exists():ex=read(cp)
        else:
            ex=execute([r['code'] for r in selected]);write(cp,ex)
        assert len(ex['rows'])==len(ids)
        for j,x in enumerate(ex['rows'],lo):
            if x['status'] not in names:names.append(x['status'])
            statuses[j]=names.index(x['status']);values[j]=x.get('value',0)
        offset+=n
        print('executed',shard,'duplicates',len(ids),'statuses',ex['summary']['statuses'],flush=True)
    report=dict(raw_train_rows=offset,executed_duplicate_rows=len(union),status_names=names,
                statuses={names[i]:int((statuses==i).sum()) for i in range(len(names))},hash_text_equality_verified=True)
    for key,m in membership.items():
        pos=np.searchsorted(union,m['row'])
        report[key]=summarize(m['group'],values[pos],statuses[pos],offset)
    np.savez(LOCAL/'noise-values.npz',rows=union,values=values,statuses=statuses)
    report['seconds']=time.monotonic()-start
    write(LOCAL/'noise.json',report)
    print(report,flush=True)

if __name__=='__main__':main()
