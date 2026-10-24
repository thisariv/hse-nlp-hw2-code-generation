"""Frozen, label-free E1 policy grid and exact execution-based selectors."""
import math
from collections import Counter
from final_adaptive import numeric, same, select


def grid():
    policies = [dict(name='baseline',family='baseline',anchor='C/g',limits=[1,5,6])]
    policies += [dict(name='token-greedy',family='token',anchor='E/g',kind='greedy')]
    for adaptive in (False,True):
        for k in (4,8):
            policies.append(dict(name=f'token-vote-{k}-'+('adaptive'if adaptive else'fixed'),
                family='token',anchor='E/g',kind='token',K=k,adaptive=adaptive))
    for k in (4,8):
        policies.append(dict(name=f'token-mix-{k}',family='token',anchor='E/g',kind='mix',K=k,limits=[1,5,6]))
    for lam in (.25,.5,1.):
        policies.append(dict(name=f'll-weight-{lam}',family='LL',anchor='C/g',limits=[1,5,6],LL='weight',lam=lam))
    for mode in ('tie-sum','tie-mean','cluster-sum','cluster-logmass'):
        policies.append(dict(name='ll-'+mode,family='LL',anchor='C/g',limits=[1,5,6],LL=mode))
    for alpha in (.3,.5,.7):
        anchor=f'S{alpha}/g'
        policies.append(dict(name=f'soup-{alpha}-greedy',family='soup',anchor=anchor,kind='greedy'))
        policies.append(dict(name=f'soup-{alpha}-anchor',family='soup',anchor=anchor,limits=[1,5,6]))
        policies.append(dict(name=f'soup-{alpha}-replace-P',family='soup',anchor='C/g',kind='replace',soup=anchor,limits=[1,1,6]))
    for limits in ([1,8,8],[1,12,12],[1,16,16],[4,8,8],[8,8,8],[16,16,16]):
        policies.append(dict(name='budget-'+'-'.join(map(str,limits)),family='budget',anchor='C/g',limits=limits))
    for limits in ([1,5,6],[1,8,8],[1,16,16]):
        policies.append(dict(name='temperature-'+'-'.join(map(str,limits)),family='temperature',anchor='C/g',limits=limits,mixed_T=True))
    return policies


def materialize(row, pool, count=None, greedy=False):
    refs=row['pools'][pool] if count is None else row['pools'][pool][:count]
    return [dict(row['candidates'][r['u']],mean_logprob=0. if greedy else r['mean_logprob'],
                 uid=r['u'],pool=pool,lane=j) for j,r in enumerate(refs)]


def components(candidates):
    remaining=[c for c in candidates if numeric(c)];groups=[]
    while remaining:
        group=[remaining.pop(0)]
        while True:
            nearby=[c for c in remaining if any(same(c['execution']['value'],v['execution']['value'])for v in group)]
            if not nearby:break
            group.extend(nearby);identities={id(c)for c in nearby}
            remaining=[c for c in remaining if id(c)not in identities]
        groups.append(group)
    return groups


def ll_select(anchor,samples,mode,lam=0.):
    allc=[anchor]+samples;groups=components(allc)
    if not groups:return anchor
    def score(c,mean=False):return sum(c['ll'][m]['mean'if mean else'sum']for m in ('C','P','W'))
    unique={c['uid']:score(c)for c in allc}
    center=sum(unique.values())/len(unique)
    scale=max(max(abs(v-center)for v in unique.values()),1e-12)
    z={k:(v-center)/scale for k,v in unique.items()}
    def key(group):
        contains=any(c is anchor for c in group)
        avg=sum(c['mean_logprob']for c in group)/len(group)
        total=sum(score(c)for c in group)
        if mode=='weight':return sum(1+lam*z[c['uid']]for c in group),contains,avg
        if mode=='tie-sum':return len(group),total,contains,avg
        if mode=='tie-mean':return len(group),sum(score(c,True)for c in group),contains,avg
        if mode=='cluster-sum':return total,contains,avg
        # Probability mass in log space; preserves multiplicity of sampled paths.
        mx=max(score(c)/3 for c in group)
        mass=mx+math.log(sum(math.exp(score(c)/3-mx)for c in group))
        return mass,contains,avg
    group=max(groups,key=key)
    return anchor if any(c is anchor for c in group)else max(group,key=lambda c:c['mean_logprob'])


def predict(row,p):
    anchor=materialize(row,p['anchor'],1,True)[0]
    used=[anchor];scored=[]
    if p.get('kind')=='greedy':return anchor,True,used,scored
    def samples(counts):
        out=[]
        for m,count in zip(('C','P','W'),counts):
            if p.get('kind')=='replace'and m=='P':
                out+=materialize(row,p['soup'],1)
            elif p.get('mixed_T'):
                counters={.7:0,1.:0}
                for j in range(count):
                    t=.7 if j%2==0 else 1.
                    pool=m+('/0.7'if t==.7 else'/1.0')
                    c=materialize(row,pool,counters[t]+1)[-1]
                    out.append(c);counters[t]+=1
            else:out+=materialize(row,m+'/0.7',count)
        return out
    kind=p.get('kind')
    if kind=='token':
        first=materialize(row,'E/0.7',2 if p['adaptive']else p['K'])
    else:
        first=samples([1,1,1])
        if kind=='mix':first+=materialize(row,'E/0.7',1)
    stop=all(numeric(c)and same(c['execution']['value'],anchor['execution']['value'])for c in [anchor]+first)
    if kind=='token':
        pool=first if stop or not p['adaptive']else materialize(row,'E/0.7',p['K'])
    else:
        pool=first if stop else samples(p['limits'])
        if kind=='mix'and not stop:pool+=materialize(row,'E/0.7',p['K'])
    used=[anchor]+pool
    if p.get('LL')and not stop:
        winner=ll_select(anchor,pool,p['LL'],p.get('lam',0.))
        scored=list({c['uid']for c in used})
    else:
        winner=select(anchor,pool,[],k=len(pool))
    return winner,stop,used,scored


def costs(used):
    """Actual generated sequences; each greedy source counted once per row."""
    return Counter(c['pool']for c in used)
