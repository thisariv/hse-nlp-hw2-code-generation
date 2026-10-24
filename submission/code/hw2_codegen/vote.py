"""Числовые ответы: граф isclose(1e-6), число голосов → greedy → mean LP → порядок.

isclose нетранзитивен: кластер — связная компонента, а не округлённое значение.
Одинаковые программы экономят исполнение, но каждая sample сохраняет свой голос.
"""

import math


def numeric(candidate):
    execution = candidate.get("execution", {})
    value = execution.get("value")
    return (
        execution.get("status") == "numeric"
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def same(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def select(greedy, main, long, k=8):
    """Вернуть сам объект кандидата; greedy представляет свою winning component."""
    samples = [
        c
        for c in main[:k] + long[:k]
        if c.get("mean_logprob") is not None and math.isfinite(c["mean_logprob"])
    ]
    good = [c for c in [greedy] + samples if numeric(c)]
    if not good:
        return greedy
    parent = list(range(len(good)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(good)):
        for j in range(i):
            if same(good[i]["execution"]["value"], good[j]["execution"]["value"]):
                parent[root(i)] = root(j)
    groups = {}
    for i, candidate in enumerate(good):
        groups.setdefault(root(i), []).append(candidate)
    winning = max(
        groups.values(),
        key=lambda group: (
            len(group),
            any(c is greedy for c in group),
            sum(c["mean_logprob"] for c in group) / len(group),
        ),
    )
    if any(c is greedy for c in winning):
        return greedy
    return max(winning, key=lambda c: c["mean_logprob"])


def candidates(record, models, stage=None):
    return [record["greedy"]] + [
        c for m in models for c in record["samples"][m["name"]][: m.get(stage or "K3", m["K2"])]
    ]


def stage1_decision(record, models, threshold=1.0):
    if record["greedy"] is None or any(
        len(record["samples"][m["name"]]) < m["K1"] for m in models
    ):
        return None
    first = candidates(record, models, "K1")
    if any(
        c.get("execution", {}).get("status", "budget_not_evaluated")
        == "budget_not_evaluated"
        for c in first
    ):
        return None
    if not all(numeric(c) for c in first):
        return False
    return (
        sum(
            same(c["execution"]["value"], first[0]["execution"]["value"]) for c in first
        )
        / len(first)
        >= threshold
    )


def stage1_leader_fraction(record, models):
    if stage1_decision(record, models) is None:
        raise ValueError("Priority requires completed stage-one executions")
    first = candidates(record, models, "K1")
    remaining = [c for c in first if numeric(c)]
    largest = 0
    while remaining:
        component = [remaining.pop(0)]
        while True:
            adjacent = [
                c
                for c in remaining
                if any(
                    same(c["execution"]["value"], v["execution"]["value"])
                    for v in component
                )
            ]
            if not adjacent:
                break
            component.extend(adjacent)
            identities = {id(c) for c in adjacent}
            remaining = [c for c in remaining if id(c) not in identities]
        largest = max(largest, len(component))
    return largest / len(first)


def select_available(record, models, **kwargs):
    available = [
        c
        for c in candidates(record, models)[1:]
        if c.get("execution", {}).get("status", "budget_not_evaluated")
        != "budget_not_evaluated"
    ]
    return (
        select(record["greedy"], available, [], k=len(available), **kwargs),
        len(available),
    )


def replay_record(greedy, samples, models, threshold=1.0):
    record = {
        "greedy": greedy,
        "samples": {m["name"]: samples[m["name"]][: m["K1"]] for m in models},
    }
    decision = stage1_decision(record, models, threshold)
    if decision is None:
        raise ValueError("Offline replay requires completed stage-one executions")
    if not decision:
        record["samples"] = {m["name"]: samples[m["name"]][: m["K2"]] for m in models}
    winner, count = select_available(record, models)
    return (winner, decision, count)


def winner_fraction(record, models):
    """Largest voting numeric component / ALL stage-two votes, failures included."""
    votes = candidates(record, models, "K2")
    remaining = [c for i, c in enumerate(votes) if numeric(c) and
                 (i == 0 or (c.get("mean_logprob") is not None and math.isfinite(c["mean_logprob"])))]
    largest = 0
    while remaining:
        component = [remaining.pop(0)]
        while True:
            adjacent = [c for c in remaining if any(same(c["execution"]["value"], v["execution"]["value"]) for v in component)]
            if not adjacent:
                break
            component.extend(adjacent)
            identities = {id(c) for c in adjacent}
            remaining = [c for c in remaining if id(c) not in identities]
        largest = max(largest, len(component))
    return largest / len(votes)


def slot_temperatures(spec, first, last, default=.7):
    """Slots are one-based; E2 mixes every fourth P/W sample at T=1."""
    return [1.0 if spec.get("mixed_temperature") and slot % 4 == 0 else default
            for slot in range(first, last + 1)]


def replay_policy(greedy, samples, models, *, e2_primary=False, threshold=1.0):
    """Production policy on a completed journal, including conditional E2 stage3."""
    record = dict(greedy=greedy, samples={m["name"]: samples[m["name"]][:m["K1"]] for m in models})
    stop = stage1_decision(record, models, threshold)
    if stop is None:
        raise ValueError("Offline replay requires completed stage-one executions")
    third = False
    if not stop:
        record["samples"] = {m["name"]: samples[m["name"]][:m["K2"]] for m in models}
        third = e2_primary and winner_fraction(record, models) < .5
        if third:
            record["samples"] = {m["name"]: samples[m["name"]][:m.get("K3", m["K2"])] for m in models}
    winner, count = select_available(record, models)
    return winner, stop, third, count
