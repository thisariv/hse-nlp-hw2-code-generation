"""E2 invariants on controlled executions, independent of generation/labels."""
from copy import deepcopy
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WORK/'research'), str(WORK/'kaggle/src')]
from e2_policies import grid, baseline, slots, predict, winner_and_fraction
from e1_policies import predict as e1_predict, costs


def candidate(value, index):
    return dict(code=f'def solve(): return {value}', represented_code=str(value),
                execution=dict(status='numeric', value=value), mean_logprob=-index-.1)


def fixture():
    row = dict(id=0, reference=dict(status='numeric', value=0), candidates=[], pools={})
    for m in ('C', 'P', 'W'):
        for pool, n in (('/g', 1), ('/0.7', 16), ('/1.0', 8)):
            refs = []
            for j in range(n):
                u = len(row['candidates']); row['candidates'].append(candidate(0, j))
                refs.append(dict(u=u, mean_logprob=-j-.1))
            row['pools'][m+pool] = refs
    return row


def main():
    assert len(grid()) == 252
    r = fixture(); p = baseline()
    w, stop, third, used, fraction = predict(r, p)
    assert stop and not third and sum(used.values()) == 4 and fraction == 1
    # Force a stage-one failure: failed candidates count in the denominator.
    r['candidates'][r['pools']['P/0.7'][0]['u']]['execution'] = dict(status='error', value=None)
    w, stop, third, used, fraction = predict(r, p)
    assert not stop and not third and sum(used.values()) == 18 and fraction == 17/18
    old, oldstop, oldused, _ = e1_predict(r, dict(anchor='C/g', limits=[1, 8, 8]))
    assert (w['code'], w['execution'], stop, used) == (old['code'], old['execution'], oldstop, costs(oldused))
    # A changed gold cannot affect prediction, scheduling, or counts.
    other = deepcopy(r); other['reference'] = dict(status='numeric', value=999)
    assert predict(r, p) == predict(other, p)
    for hot in (0, 4, 2):
        s = slots(r, 'P', 16, hot)
        expected_hot = 0 if hot == 0 else 16//hot
        assert sum(c['pool'] == 'P/1.0' for c in s) == expected_hot
        assert len({(c['pool'], c['lane']) for c in s}) == 16
        assert s[0]['pool'] == 'P/0.7'
        assert slots(r, 'P', 8, hot) == s[:8]
    # Connected components, including a nontransitive isclose chain.
    anchor = candidate(1., 0)
    chain = [candidate(1.+.9e-6, 1), candidate(1.+1.8e-6, 2)]
    w, f = winner_and_fraction(anchor, chain)
    assert w is anchor and f == 1.
    # Strict threshold equality does not trigger; an infinitesimal increase does.
    r = fixture()
    for j, ref in enumerate(r['pools']['W/0.7']):
        r['candidates'][ref['u']]['execution']['value'] = 1 if j < 8 else 0
    _, _, _, _, fraction = predict(r, p)
    assert fraction == 10/18
    eq = dict(p, threshold=fraction, extra=[8, 8])
    gt = dict(eq, threshold=fraction+1e-12)
    assert predict(r, eq)[2] is False
    result = predict(r, gt)
    assert result[2] is True and sum(result[3].values()) == 34
    # Every grid policy consumes unique available pool occurrences.
    for policy in grid():
        result = predict(r, policy)
        assert result[3]['C/0.7'] <= 4
        assert all(n <= (8 if k.endswith('/1.0') else 16) for k, n in result[3].items())
    receipt = dict(passed=True, grid=252, baseline_parity=True, failed_vote_denominator=True,
                   label_independence=True, temperature_prefix_unique=True,
                   nontransitive_components=True, strict_threshold=True, pool_bounds=True)
    (WORK/'checks/e2-cpu.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
