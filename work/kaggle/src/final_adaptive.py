"""CPU-only adaptive decisions and the frozen v3 vote extended to arbitrary K.

K1: stage-one samples PER model. K2: total samples PER escalated model.
Unanimity is relative to greedy (isclose is intentionally not transitive here).
"""
import math
from final_choose import choose


def parameters(config):
    k1, k2 = config.get('K1', 2), config.get('K2', 4)
    threshold = config.get('unanimity_threshold', 1.)
    if (isinstance(k1, bool) or not isinstance(k1, int) or k1 < 1 or
        isinstance(k2, bool) or not isinstance(k2, int) or k2 < k1):
        raise ValueError('Require integer 1 <= K1 <= K2 (per model)')
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('Require 0 < unanimity_threshold <= 1')
    return k1, k2, float(threshold)


def numeric(candidate):
    execution = candidate.get('execution', {})
    value = execution.get('value')
    return (execution.get('status') == 'numeric' and isinstance(value, (int, float))
            and math.isfinite(value))


def same(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def question_batch(sequence_budget, k):
    """K lanes per question, including greedy K=1; never exceed the lane budget."""
    if (isinstance(sequence_budget, bool) or not isinstance(sequence_budget, int) or
        isinstance(k, bool) or not isinstance(k, int) or k < 1 or sequence_budget < k):
        raise ValueError('Require integer sequence_budget >= K >= 1')
    return sequence_budget//k


def stage1_leader_fraction(record, k1=2):
    """Largest numeric vote component / all stage-one candidates (failures included).

    Use the frozen vote's connected-component semantics; failed programs supply
    no agreement. This is scheduling evidence, never a gold-dependent decision.
    """
    if stage1_decision(record, k1) is None:
        raise ValueError('Priority requires completed stage-one executions')
    candidates = [record['greedy']]+record['main'][:k1]+record['long'][:k1]
    remaining = [c for c in candidates if numeric(c)]
    largest = 0
    while remaining:
        component = [remaining.pop(0)]
        while True:
            adjacent = [c for c in remaining if any(same(c['execution']['value'],
                        v['execution']['value']) for v in component)]
            if not adjacent:
                break
            component.extend(adjacent)
            identities = {id(c) for c in adjacent}
            remaining = [c for c in remaining if id(c) not in identities]
        largest = max(largest, len(component))
    return largest/len(candidates)


def stage1_decision(record, k1=2, threshold=1.):
    """None: await execution; True: stop; False: escalate. No targets/scores."""
    if record['greedy'] is None or any(len(record[key]) < k1 for key in ('main', 'long')):
        return None
    candidates = [record['greedy']] + record['main'][:k1] + record['long'][:k1]
    if any(c.get('execution', {}).get('status', 'budget_not_evaluated') == 'budget_not_evaluated'
           for c in candidates):
        return None
    # A failed or nonfinite program never supplies evidence for early stopping.
    if not all(numeric(c) for c in candidates):
        return False
    value = candidates[0]['execution']['value']
    return sum(same(c['execution']['value'], value) for c in candidates)/len(candidates) >= threshold


def select(greedy, main, long, k=8, question=None, scorer=None, score_weight=0., deadline=None):
    """count -> greedy -> mean LP -> stable order; greedy code in winning component.

    Optional score(question, represented_code) adds weight * max(score) to the
    cluster's vote count. Disabled callers never invoke/import a reviewer.
    """
    samples = [c for c in main[:k]+long[:k] if c.get('mean_logprob') is not None
               and math.isfinite(c['mean_logprob'])]
    winner = choose({'greedy': greedy, 'samples': {'0.7': samples}},
                    {'mode': 'vote', 'temperature': .7, 'k': len(samples), 'greedy_vote': True})
    if not numeric(winner):
        return greedy
    good = [c for c in [greedy]+samples if numeric(c)]
    components = []
    remaining = good[:]
    while remaining:
        component = [remaining.pop(0)]
        while True:
            adjacent = [c for c in remaining if any(same(c['execution']['value'],
                        v['execution']['value']) for v in component)]
            if not adjacent:
                break
            component.extend(adjacent)
            identities = {id(c) for c in adjacent}
            remaining = [c for c in remaining if id(c) not in identities]
        components.append(component)
    if scorer is not None and score_weight:
        if question is None or not math.isfinite(score_weight):
            raise ValueError('Scorer requires question and finite weight')
        cache = {}
        def key(component):
            for c in component:
                if deadline is not None:
                    from final_core import check_deadline
                    check_deadline(deadline)
                represented = c['represented_code']
                if represented not in cache:
                    cache[represented] = float(scorer(question, represented))
                    if not math.isfinite(cache[represented]):
                        raise ValueError('Nonfinite reviewer score')
            return (len(component)+score_weight*max(cache[c['represented_code']] for c in component),
                    any(c is greedy for c in component),
                    sum(c['mean_logprob'] for c in component)/len(component))
        component = max(components, key=key)
        winner = max(component, key=lambda c: c['mean_logprob'])
    else:
        component = next(group for group in components if any(c is winner for c in group))
    return greedy if any(c is greedy for c in component) else winner


def select_available(record, k=8, **kwargs):
    available = {key: [c for c in record[key] if
                 c.get('execution', {}).get('status', 'budget_not_evaluated') != 'budget_not_evaluated']
                 for key in ('main', 'long')}
    count = len(available['main'])+len(available['long'])
    return select(record['greedy'], available['main'], available['long'], k, **kwargs), count


def replay_record(greedy, main, long, k1=2, k2=8, threshold=1.):
    """The same stage-one decision and selector as the live worker, in journal order."""
    record = {'greedy': greedy, 'main': main[:k1], 'long': long[:k1]}
    decision = stage1_decision(record, k1, threshold)
    if decision is None:
        raise ValueError('Offline replay needs completed stage-one execution')
    if not decision:
        record.update(main=main[:k2], long=long[:k2])
    winner, count = select_available(record, k2)
    return winner, decision, count
