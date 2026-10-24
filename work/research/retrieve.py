"""Exact sparse TF-IDF top20 against nested protocol26a finite train indices."""
import pickle
import time
import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
from common import *

def cache_shard(shard):
    if shard<42:
        return WORK/'kaggle/staging/drive-merged/26-token-cache/shards'/f'part-{shard:05d}'
    return WORK/'results/kaggle-runs/nlp-hw02-26c-cpu-cache/2/26c-token-cache/shards'/f'part-{shard:05d}'

def merge(scores,ids,new_scores,new_ids):
    a=np.concatenate((scores,new_scores),axis=1);b=np.concatenate((ids,new_ids),axis=1)
    order=np.lexsort((b,-a),axis=1)[:,:20]
    return np.take_along_axis(a,order,axis=1),np.take_along_axis(b,order,axis=1)

def main():
    start=time.monotonic()
    sets={k:read(INPUT/(k+'.json')) for k in ['heldout4k','old1024','test']}
    # Secondary questions included so transfer can be checked without a new index.
    queries=[r['question'] for rows in sets.values() for r in rows]
    protected=set(q for g in read(INPUT/'protected-questions.json')['groups'].values() for q in g)
    vp=LOCAL/'tfidf.pkl'
    if vp.exists():vec=pickle.loads(vp.read_bytes())
    else:
        sample=read(LOCAL/'sample-60k.json')
        vec=TfidfVectorizer(lowercase=False, stop_words='english',max_features=50000,
                            min_df=2,max_df=.3,sublinear_tf=True,dtype=np.float32)
        vec.fit([masked(r['question']) for r in sample]);vp.write_bytes(pickle.dumps(vec))
    Q=vec.transform([masked(q) for q in queries]).tocsr()
    scores=np.zeros((len(queries),20),np.float32)
    ids=np.full((len(queries),20),2**62,np.int64)
    total=0;segments=[]
    for shard in range(60):
        p=RAW/f'part-{shard:05d}.parquet'
        allq=pq.read_table(p,columns=['question']).column('question').to_pylist()
        rowids=np.load(cache_shard(shard)/'provenance-rows.npy').astype(np.int64)
        assert not any(allq[i] in protected for i in rowids)
        lo=0
        while lo<len(rowids):
            hi=len(rowids)
            for boundary in [200000,1000000]:
                if total<boundary<total+hi-lo:hi=lo+boundary-total;break
            chosen=rowids[lo:hi]
            key=f'{shard:02d}-{lo}-{hi}'
            cp=LOCAL/('nearest-'+key+'.npz')
            tick=time.monotonic()
            if cp.exists():
                z=np.load(cp);ss,ii=z['scores'],z['ids']
            else:
                X=vec.transform([masked(allq[i]) for i in chosen]).tocsr()
                result=sp_matmul_topn(Q,X.T.tocsr(),top_n=20,threshold=0.,sort=True,n_threads=4)
                ss=np.zeros((len(queries),20),np.float32);ii=np.full((len(queries),20),2**62,np.int64)
                for j in range(len(queries)):
                    a,b=result.indptr[j:j+2];nn=b-a
                    ss[j,:nn]=result.data[a:b];ii[j,:nn]=shard*200000+chosen[result.indices[a:b]]
                np.savez(cp,scores=ss,ids=ii)
            scores,ids=merge(scores,ids,ss,ii)
            total+=len(chosen);lo=hi
            segments.append(dict(key=key,rows=len(chosen),seconds=time.monotonic()-tick))
            print(key,'total',total,'seconds',round(segments[-1]['seconds'],1),flush=True)
            if total in (200000,1000000):
                np.savez(LOCAL/f'retrieval-{total}.npz',scores=scores,ids=ids)
        del allq
    np.savez(LOCAL/f'retrieval-{total}.npz',scores=scores,ids=ids)
    report=dict(index_sizes=[200000,1000000,total],raw_index_rows=total,vocab=len(vec.vocabulary_),
                train_selection='all 60 finite-reference caches from26a+26c, exact protected exclusions and SHA42:q hash heldout; nested shard,row prefixes',
                vectorizer='word unigrams; english stopwords; numeric masking; sublinear TF; L2; min_df2 max_df.3; train-only stratified60k IDF fixed across sizes',
                queries={k:len(v) for k,v in sets.items()},segments=segments,seconds=time.monotonic()-start)
    report['tie_policy']='sp_matmul_topn arbitrary membership at equal-score shard cutoff; merge retained candidates by decreasing score then increasing shard,row'
    write(LOCAL/'retrieval-index.json',report)

if __name__=='__main__':main()
