"""Frozen mixed vote. final_choose is generated verbatim from exp27_decode.choose."""
import json
import math
from pathlib import Path
from final_choose import choose

CONFIG = {'mode': 'vote', 'temperature': .7, 'k': 8, 'greedy_vote': True}


def select(greedy, main, long):
    # Deduplication only caches execution: repeated samples still cast votes.
    samples = [c for c in main[:4] + long[:4] if c.get('mean_logprob') is not None
               and math.isfinite(c['mean_logprob'])]
    row = {'greedy': greedy, 'samples': {'0.7': samples}}
    winner = choose(row, CONFIG)
    if winner['execution']['status'] != 'numeric':
        return greedy
    # Code representative: greedy anywhere in the winning connected component,
    # otherwise greatest mean LP, stable order. Do not replace graph connectivity
    # by a direct isclose comparison (isclose is non-transitive).
    good = [c for c in [greedy] + samples if c['execution']['status'] == 'numeric']
    component = [winner]
    remaining = [c for c in good if c is not winner]
    while True:
        adjacent = [c for c in remaining if any(math.isclose(
            c['execution']['value'], v['execution']['value'], rel_tol=1e-6, abs_tol=1e-6)
            for v in component)]
        if not adjacent:
            break
        component.extend(adjacent)
        adjacent_ids = {id(c) for c in adjacent}
        remaining = [c for c in remaining if id(c) not in adjacent_ids]
    if any(c is greedy for c in component):
        return greedy
    return winner


def select_available(record):
    """A missing model/batch never discards already executed samples or their votes."""
    available = {key: [c for c in record[key]
                       if c['execution']['status'] != 'budget_not_evaluated']
                 for key in ('main', 'long')}
    count = len(available['main']) + len(available['long'])
    return select(record['greedy'], available['main'], available['long']), count


def write_answers(path, codes):
    with Path(path).open('w', encoding='utf8') as stream:
        for code in codes:
            if not isinstance(code, str):
                raise TypeError('Answer code must be a string')
            stream.write(json.dumps({'code': code}, ensure_ascii=False) + '\n')


def validate_answers(path, expected):
    with Path(path).open(encoding='utf8') as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == expected
    assert all(set(r) == {'code'} and isinstance(r['code'], str) for r in rows)
    return rows


def verify_winners(winners, deadline, workers=None):
    """Fresh service: compare the actual chosen CODE against its voted execution."""
    from final_executor import Executor
    executor = Executor(workers)
    eligible = [i for i, w in enumerate(winners) if w['execution']['status'] == 'numeric']
    items = [{'code': winners[i]['code']} for i in eligible]
    try:
        executor.submit(items)
        complete = executor.drain(deadline)
        executor.attach(items)
        mismatches = [i for i, item in zip(eligible, items) if
                      item['execution']['status'] != 'numeric' or not math.isclose(
                          item['execution']['value'], winners[i]['execution']['value'],
                          rel_tol=1e-6, abs_tol=1e-6)]
        return {'checked_rows': len(eligible), 'complete': complete,
                'mismatch_rows': mismatches, 'execution': executor.summary()}
    finally:
        executor.close()
