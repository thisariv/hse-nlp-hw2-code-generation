"""Stronger role baseline: monotone lexical alignment, tolerates extra numbers."""
import time
from collections import Counter
import numpy as np
from tokenizers import Tokenizer
from common import *
from retrieval import score_flags

def main():
    start=time.monotonic();meta=read(LOCAL/'retrieval-index.json');neighbors={int(k):v for k,v in read(LOCAL/'neighbors.json').items()}
    sets={s:read(INPUT/(s+'.json')) for s in ['heldout4k','old1024']}
    queries=[r['question'] for rr in sets.values() for r in rr]
    grids={};reasons={};allcodes=[]
    for n in meta['index_sizes']:
        z=np.load(LOCAL/f'retrieval-{n}.npz');grid=[];cnt=Counter()
        for j,q in enumerate(queries):
            row=[]
            for rid in z['ids'][j]:
                r=neighbors[int(rid)];code,reason=substitute(r['code'],r['question'],q,'aligned_role');cnt[reason]+=1
                row.append(code)
                if code is not None:allcodes.append(code)
            grid.append(row)
        grids[n]=grid;reasons[n]=dict(cnt)
        print('aligned',n,dict(cnt),flush=True)
    exmap=read(LOCAL/'retrieval-execution.json');extra=list(set(allcodes)-exmap.keys());xp=LOCAL/'aligned-execution.json'
    if xp.exists():exmap.update(read(xp))
    else:
        print('execute new',len(extra),flush=True);ex=execute(extra);d=dict(zip(extra,ex['rows']));write(xp,d);exmap.update(d)
    tok=Tokenizer.from_file(str(INPUT/'tokenizer.json'));refs={s:read(INPUT/(s+'-reference-execution.json'))['rows'] for s in sets}
    base={s:read(S1/f'gpu0-a26/points/p050000/evaluation/{s}-execution.json')['rows'] for s in sets}
    report=dict(rule='SequenceMatcher monotone alignment of word tokens with numbers masked; requires all source code-used question numeric values aligned; repeated-value conflicts abstain; no order fallback',
                reasons=reasons,new_programs_executed=len(extra),by_index={})
    fullflags=None
    for n,grid in grids.items():
        rr={}
        for s,lo,hi in [('heldout4k',0,4096),('old1024',4096,5120)]:
            eligible=np.array([r['status']=='numeric' for r in refs[s]])
            lengths=np.array([bucket(len(e.ids)) for e in tok.encode_batch([r['code'] for r in sets[s]])])
            flags=np.array([[c is not None and same(exmap[c],refs[s][j]) for c in row] for j,row in enumerate(grid[lo:hi])])
            greedy=np.array([same(a,b) for a,b in zip(refs[s],base[s])]);rr[s]={}
            for k in [1,5,20]:
                f=flags[:,:k].any(axis=1);rr[s][f'top{k}']=score_flags(f[eligible],lengths[eligible]);rr[s][f'top{k}']['A_union_oracle_matches']=int((f|greedy)[eligible].sum())
            if n==meta['index_sizes'][-1] and s=='heldout4k':fullflags=flags
        report['by_index'][str(n)]=rr
    np.savez(LOCAL/'aligned-match-flags.npz',flags=fullflags)
    report['seconds']=time.monotonic()-start;write(LOCAL/'alignment.json',report)

if __name__=='__main__':main()
