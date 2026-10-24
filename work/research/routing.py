"""Gold-free inference routing rules, threshold chosen on old991, assessed on4k."""
import time
from collections import Counter
import numpy as np
from common import *

def main():
    start=time.monotonic();meta=read(LOCAL/'retrieval-index.json');n=meta['index_sizes'][-1]
    z=np.load(LOCAL/f'retrieval-{n}.npz');neighbors={int(k):v for k,v in read(LOCAL/'neighbors.json').items()}
    exmap=read(LOCAL/'retrieval-execution.json');sampler=np.load(LOCAL/'sampling-oracle-flags.npz')
    results={};retrflags={}
    for s,lo,hi in [('heldout4k',0,4096),('old1024',4096,5120)]:
        rows=read(INPUT/(s+'.json'));ref=read(INPUT/(s+'-reference-execution.json'))['rows']
        greedy=read(S1/f'gpu0-a26/points/p050000/evaluation/{s}-execution.json')['rows']
        eligible=np.array([r['status']=='numeric' for r in ref]);base=np.array([same(a,b) for a,b in zip(ref,greedy)])
        top20=np.zeros(hi-lo,bool);top1=[];majority=[]
        for j,r in enumerate(rows):
            outs=[]
            for rid in z['ids'][lo+j]:
                nr=neighbors[int(rid)];code,reason=substitute(nr['code'],nr['question'],r['question'],'order')
                out=exmap[code] if code is not None else {'status':'abstain'}
                outs.append(out)
            top20[j]=any(same(x,ref[j]) for x in outs);top1.append(outs[0])
            good=[x for x in outs[:5] if x['status']=='numeric'];clusters=[]
            for x in good:
                for c in clusters:
                    if same(x,c[0]):c.append(x);break
                else:clusters.append([x])
            clusters.sort(key=len,reverse=True)
            majority.append(clusters[0][0] if clusters and len(clusters[0])>=3 else {'status':'abstain'})
        retrflags[s]=top20
        variants={'keep_greedy':dict(matches=int(base[eligible].sum()),n=int(eligible.sum()),
                    accuracy=float(base[eligible].mean()),routed=0,gains=0,losses=0)}
        for rule,oo in [('top1',top1),('top5_agree3',majority)]:
            for threshold in [.9,.95,.97,.99,.999]:
                use=np.array([z['scores'][lo+j,0]>=threshold and x['status']=='numeric' for j,x in enumerate(oo)])
                f=np.array([same(x,ref[j]) for j,x in enumerate(oo)])
                routed=np.where(use,f,base);gains=int((use&f&~base&eligible).sum());losses=int((use&~f&base&eligible).sum())
                variants[f'{rule}-{threshold}']=dict(matches=int(routed[eligible].sum()),n=int(eligible.sum()),
                    accuracy=float(routed[eligible].mean()),routed=int((use&eligible).sum()),gains=gains,losses=losses)
        results[s]=dict(base_matches=int(base[eligible].sum()),variants=variants)
    # Use secondary only for selecting a threshold. Primary is an assessment.
    chosen=max(results['old1024']['variants'],key=lambda k:(results['old1024']['variants'][k]['matches'],-results['old1024']['variants'][k]['routed']))
    nontrivial=max((k for k in results['old1024']['variants'] if k!='keep_greedy'),key=lambda k:results['old1024']['variants'][k]['matches'])
    oracle={}
    for k in ['16','32','65']:
        f=sampler[k]|retrflags['heldout4k'];oracle[k]=dict(matches=int(f.sum()),accuracy=float(f.mean()),n=4096)
    write(LOCAL/'routing.json',dict(index_size=n,selection_set='old991',selected_rule=chosen,
        primary=results['heldout4k']['variants'][chosen],secondary=results['old1024']['variants'][chosen],
        best_nontrivial=dict(rule=nontrivial,primary=results['heldout4k']['variants'][nontrivial],secondary=results['old1024']['variants'][nontrivial]),
        sweep=results,sampling_plus_retrieval_top20_oracle=oracle,
        caveat='post-hoc threshold sweep; old991 and4k reused. No reference participates in inference rule itself. Fresh untouched.',
        seconds=time.monotonic()-start))

if __name__=='__main__':main()
