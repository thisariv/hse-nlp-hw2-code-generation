"""E2 preregistered grid; selectors never receive labels or correctness."""
from collections import Counter
from e1_policies import materialize, components
from final_adaptive import numeric, same, select


def grid():
    out = []
    for c in (1, 2, 3, 4):
        for k in (8, 12, 16):
            for t in (0, 4, 2):
                out.append(dict(name=f'base-c{c}-p{k}-w{k}-t{t}', family='base',
                                C=c, K=k, hot_every=t))
    for c in (1, 2, 3, 4):
        for t in (0, 4, 2):
            for q in (.5, .6, .7):
                for p, w in ((4, 0), (0, 4), (4, 4), (8, 0), (0, 8), (8, 8)):
                    out.append(dict(name=f'third-c{c}-t{t}-q{q}-p{p}-w{w}',
                                    family='third', C=c, K=8, hot_every=t,
                                    threshold=q, extra=[p, w]))
    assert len(out) == len({p['name'] for p in out}) == 252
    return out


def baseline():
    return grid()[0]


def slots(row, model, count, hot_every):
    if model == 'C' or not hot_every:
        return materialize(row, model+'/0.7', count)
    used = Counter(); out = []
    for j in range(1, count+1):
        pool = model+('/1.0' if j % hot_every == 0 else '/0.7')
        # Preserve the actual saved lane, rather than a prefix-local index.
        out.append(materialize(row, pool, used[pool]+1)[-1])
        used[pool] += 1
    return out


def winner_and_fraction(anchor, samples):
    winner = select(anchor, samples, [], k=len(samples))
    groups = components([anchor]+samples)
    group = next((g for g in groups if any(c is winner for c in g)), [])
    return winner, len(group)/(len(samples)+1)


def predict(row, policy):
    anchor = materialize(row, 'C/g', 1, True)[0]
    first = [slots(row, m, 1, 0)[0] for m in ('C', 'P', 'W')]
    stop = all(numeric(c) and same(c['execution']['value'], anchor['execution']['value'])
               for c in [anchor]+first)
    if stop:
        return select(anchor, first, [], k=3), True, False, Counter(c['pool'] for c in [anchor]+first), 1.
    counts = [policy['C'], policy['K'], policy['K']]
    samples = sum((slots(row, m, n, policy['hot_every'])
                   for m, n in zip(('C', 'P', 'W'), counts)), [])
    winner, fraction = winner_and_fraction(anchor, samples)
    third = 'threshold' in policy and fraction < policy['threshold']
    if third:
        counts[1] += policy['extra'][0]; counts[2] += policy['extra'][1]
        samples = sum((slots(row, m, n, policy['hot_every'])
                       for m, n in zip(('C', 'P', 'W'), counts)), [])
        winner, _ = winner_and_fraction(anchor, samples)
    return winner, False, third, Counter(c['pool'] for c in [anchor]+samples), fraction
