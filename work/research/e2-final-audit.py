"""Independent union-find replay of the two sealed policies; no policy search."""
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK/'kaggle/src'))
from final_ensemble import replay_record


def read(path):
    return json.loads(path.read_text())


def samples(row, model, count, mixed=False):
    used = Counter()
    out = []
    for slot in range(1, count+1):
        pool = model+('/1.0' if mixed and slot % 4 == 0 else '/0.7')
        entry = row['pools'][pool][used[pool]]
        used[pool] += 1
        out.append(dict(row['candidates'][entry['u']], mean_logprob=entry['mean_logprob']))
    return out


def numeric(c):
    x = c['execution']
    return x['status'] == 'numeric' and math.isfinite(x['value'])


def vote(votes):
    good = [i for i, c in enumerate(votes) if numeric(c)]
    if not good:
        return votes[0], 0.
    parent = {i: i for i in good}

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i in good:
        for j in good:
            if j >= i:
                break
            if math.isclose(votes[i]['execution']['value'], votes[j]['execution']['value'],
                            rel_tol=1e-6, abs_tol=1e-6):
                parent[root(i)] = root(j)
    groups = {}
    for i in good:
        groups.setdefault(root(i), []).append(i)
    group = max(groups.values(), key=lambda g: (
        len(g), 0 in g, sum(votes[i]['mean_logprob'] for i in g)/len(g)))
    winner = 0 if 0 in group else max(group, key=lambda i: votes[i]['mean_logprob'])
    return votes[winner], len(group)/len(votes)


def main():
    confirmation = read(WORK/'results/e2-confirm.json')
    assert confirmation['primary']['name'] == 'third-c2-t4-q0.5-p8-w0'
    download = read(WORK/'checks/e2-download.json')
    raw = WORK/download['output']
    with gzip.open(raw/'e2-comparison.jsonl.gz', 'rt') as f:
        expected = {r['id']: r for r in map(json.loads, f)}
    bits = Counter()
    early = third = n = 0
    for split in ('tune2048', 'eval2048'):
        folder = raw/'e2/full'/split
        for entry in read(folder/'summary.json')['journal_manifest']:
            path = folder/entry['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
            with gzip.open(path, 'rt') as f:
                for row in map(json.loads, f):
                    anchor_entry = row['pools']['C/g'][0]
                    anchor = dict(row['candidates'][anchor_entry['u']], mean_logprob=0.)
                    first = [anchor]+[samples(row, m, 1)[0] for m in ('C', 'P', 'W')]
                    stop = all(numeric(c) and math.isclose(c['execution']['value'],
                        anchor['execution']['value'], rel_tol=1e-6, abs_tol=1e-6) for c in first)
                    base = first if stop else [anchor]+sum((samples(row, m, k)
                        for m, k in zip(('C', 'P', 'W'), (1, 8, 8))), [])
                    bw, _ = vote(base)
                    production, ps, _ = replay_record(anchor,
                        {m: samples(row, m, 8) for m in ('C', 'P', 'W')},
                        [dict(name=m, K1=1, K2=k) for m, k in zip(('C', 'P', 'W'), (1, 8, 8))])
                    assert (production['code'], production['execution'], ps) == (bw['code'], bw['execution'], stop)
                    primary = first if stop else [anchor]+samples(row, 'C', 2)+samples(row, 'P', 8, True)+samples(row, 'W', 8, True)
                    pw, fraction = vote(primary)
                    expand = not stop and fraction < .5
                    if expand:
                        pw, _ = vote([anchor]+samples(row, 'C', 2)+samples(row, 'P', 16, True)+samples(row, 'W', 8, True))
                    exp = expected[row['id']]
                    assert (bw['code'], pw['code'], expand) == (exp['baseline'], exp['primary'], exp['stage3'])
                    ref = row['reference']
                    correct = lambda c: ref['status'] == 'numeric' and numeric(c) and math.isclose(
                        ref['value'], c['execution']['value'], rel_tol=1e-6, abs_tol=1e-6)
                    a, b = correct(bw), correct(pw)
                    assert (a, b) == (exp['baseline_correct'], exp['primary_correct'])
                    bits[(a, b)] += 1
                    early += stop
                    third += expand
                    n += 1
    assert n == len(expected) == 4096
    assert (early, third) == (confirmation['early_stops'], confirmation['third_rows'])
    assert (bits[(False, True)], bits[(True, False)]) == (confirmation['gains'], confirmation['losses'])
    result = dict(passed=True, at_utc=datetime.now(timezone.utc).isoformat(), rows=n,
        baseline_production_parity=n, primary_independent_union_find_parity=n,
        early_stops=early, third_rows=third, gains=bits[(False, True)], losses=bits[(True, False)],
        selection_sha256=confirmation['selection_sha256'],
        confirmation_sha256=hashlib.sha256((WORK/'results/e2-confirm.json').read_bytes()).hexdigest(),
        independent_tasks=False, decision=confirmation['decision'])
    (WORK/'checks/e2-final.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
