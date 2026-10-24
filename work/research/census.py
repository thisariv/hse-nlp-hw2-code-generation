"""Full pinned-train duplicate census; hash grouping then exact text verification."""
import hashlib
import time
from collections import Counter
import numpy as np
import pyarrow.parquet as pq
from common import LOCAL, RAW, WORK, read, write

def main():
    started=time.monotonic()
    inventory=read(WORK/'checks/26-pinned-train-shard-inventory.json')
    records=[]
    for spec in inventory['files']:
        p=RAW/('part-%05d.parquet'%spec['shard'])
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        assert p.stat().st_size==spec['bytes'] and digest==spec['sha256']
        out=LOCAL/('hashes-%02d.npy'%spec['shard'])
        if not out.exists():
            q=pq.read_table(p, columns=['question']).column('question').to_pylist()
            a=np.empty(len(q), dtype=[('exact','V16'),('normal','V16')])
            for i,s in enumerate(q):
                a[i]=(hashlib.sha256(s.encode()).digest()[:16],
                      hashlib.sha256(' '.join(s.lower().split()).encode()).digest()[:16])
            np.save(out,a)
        records.append(dict(shard=spec['shard'],rows=len(np.load(out,mmap_mode='r')),sha256=digest))
        print('hashed', spec['shard'],records[-1]['rows'],flush=True)
    a=np.concatenate([np.load(LOCAL/('hashes-%02d.npy'%r['shard'])) for r in records])
    report=dict(revision=inventory['revision'],shards=records,raw_rows=len(a),hash_bits=128)
    for key in ('exact','normal'):
        idx=np.argsort(a[key],kind='stable')
        h=a[key][idx]
        starts=np.r_[0,np.flatnonzero(h[1:]!=h[:-1])+1,len(h)]
        sizes=np.diff(starts)
        dupe=sizes>=2
        # Save only group membership; avoid all train texts/codes in RAM.
        counts=Counter(map(int,sizes))
        selected=np.concatenate([idx[starts[i]:starts[i+1]] for i in np.flatnonzero(dupe)])
        gids=np.repeat(np.flatnonzero(dupe),sizes[dupe])
        order=np.argsort(selected)
        np.savez(LOCAL/(key+'-duplicate-members.npz'),row=selected[order],group=gids[order],counts=sizes[dupe])
        report[key]=dict(unique_questions=len(sizes),duplicate_groups=int(dupe.sum()),
                         duplicate_rows=int(sizes[dupe].sum()), multiplicity=dict(counts))
        print(key,report[key],flush=True)
    report['seconds']=time.monotonic()-started
    write(LOCAL/'census.json',report)

if __name__=='__main__':main()
