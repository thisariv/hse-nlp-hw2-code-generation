"""Read the previously executed S1 candidate grid: no new model inference."""
import numpy as np
from common import *

def main():
    root=S1/'gpu0-exp27/heldout4k'
    candidates=[]
    for p in sorted((root/'candidates').glob('batch-*.json')):
        if '.sha.' not in p.name:candidates.extend(read(p))
    candidates.sort(key=lambda r:r['id'])
    assert [r['id'] for r in candidates]==list(range(4096))
    execution=[]
    for p in sorted((root/'execution').glob('chunk-*.json')):execution.extend(read(p))
    assert len(execution)==4096*65
    refs=read(INPUT/'heldout4k-reference-execution.json')['rows']
    offset=0;flags={str(k):np.zeros(4096,dtype=bool) for k in [1,4,8,16,32,65]}
    for i,r in enumerate(candidates):
        row=execution[offset:offset+65];offset+=65
        codes=[r['greedy']]+[c for group in r['samples'].values() for c in group]
        assert len(codes)==65
        # exp27 execute_candidates flattens rows in original id order.
        match=[same(x,refs[i]) for x in row]
        flags['1'][i]=match[0]
        for k in [4,8,16,32]:flags[str(k)][i]=any(match[1:1+k]) or match[0]
        flags['65'][i]=any(match)
    assert flags['1'].sum()==2538
    write(LOCAL/'sampling-oracle.json',dict(source='S1 saved execution chunks; T0.7 first32, T0.5 next16, T1.0 next16',
         includes_greedy=True,oracle={k:dict(matches=int(f.sum()),n=4096,accuracy=float(f.mean())) for k,f in flags.items()}))
    np.savez(LOCAL/'sampling-oracle-flags.npz',**flags)

if __name__=='__main__':main()
