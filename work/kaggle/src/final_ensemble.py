"""Ordered 2–4 model ensemble; K2 is the TOTAL sample limit for each model.

Decisions use executed programs only. Flattening preserves model/lane order and
delegates voting/code selection to the frozen v3–v6 implementation.
"""
import math
from pathlib import Path
import re
from final_adaptive import numeric, same, select

DEFAULT_MODELS = [
    {'name': 'C-main', 'filename': 'model-C-main__weights__l2-62280-fp16.pt',
     'sha256': '8c11b1038855422b6912e243049a554317978f848edb412ed48aebbe4cb6b886',
     'dataset': 'thisarivson/nlp-hw02-l3-live'},
    {'name': 'C-long2x', 'filename': 'model-C-long2x__weights__l2-62280-fp16.pt',
     'sha256': 'd10bf0018d0b14fd6cc3c8b9c7121f127b5cec97057544fd63e5824253a55057',
     'dataset': 'thisarivson/nlp-hw02-l3-live'},
    {'name': 'pilot-b', 'filename': 'model-pilot-b__weights__long-truncation-fp16.pt',
     'sha256': '8439037720dda16dda6b73fe676860d5498bbcba4cacebc0d709bbb22db464a5',
     'dataset': 'thisarivson/nlp-hw02-p1-weights'},
]


def parameters(config):
    models = config.get('models')
    if not isinstance(models, list) or not 2 <= len(models) <= 4:
        raise ValueError('Require an ordered list of 2–4 models')
    normalized = []
    for model in models:
        if not isinstance(model, dict):
            raise ValueError('Require model objects with name/filename/sha256/K1/K2')
        name, filename, digest = (model.get(k) for k in ('name', 'filename', 'sha256'))
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name == 'greedy':
            raise ValueError('Invalid model name')
        if (not isinstance(filename, str) or Path(filename).name != filename or
                not filename.endswith('.pt') or '\\' in filename):
            raise ValueError('Require a weight basename ending in .pt')
        if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise ValueError('Every model requires a pinned SHA256')
        k1, k2 = model.get('K1', config.get('K1', 1)), model.get('K2', config.get('K2', 4))
        if (isinstance(k1, bool) or not isinstance(k1, int) or k1 < 1 or
                isinstance(k2, bool) or not isinstance(k2, int) or k2 < k1):
            raise ValueError('Require integer 1 <= K1 <= K2 for each model')
        if 'dataset' in model and (not isinstance(model['dataset'], str) or
                not re.fullmatch(r'[A-Za-z0-9_-]+/[A-Za-z0-9_-]+', model['dataset'])):
            raise ValueError('Invalid model dataset')
        normalized.append(dict(model, K1=k1, K2=k2))
    if len({m['name'] for m in normalized}) != len(models):
        raise ValueError('Model names must be unique')
    threshold = config.get('unanimity_threshold', 1.)
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or
            not math.isfinite(threshold) or not 0 < threshold <= 1):
        raise ValueError('Require 0 < unanimity_threshold <= 1')
    return normalized, float(threshold)


def candidates(record, models, stage='K2'):
    return [record['greedy']]+[c for m in models for c in record['samples'][m['name']][:m[stage]]]


def stage1_decision(record, models, threshold=1.):
    if record['greedy'] is None or any(len(record['samples'][m['name']]) < m['K1'] for m in models):
        return None
    first = candidates(record, models, 'K1')
    if any(c.get('execution', {}).get('status', 'budget_not_evaluated') == 'budget_not_evaluated' for c in first):
        return None
    if not all(numeric(c) for c in first):
        return False
    return sum(same(c['execution']['value'], first[0]['execution']['value']) for c in first)/len(first) >= threshold


def stage1_leader_fraction(record, models):
    if stage1_decision(record, models) is None:
        raise ValueError('Priority requires completed stage-one executions')
    first = candidates(record, models, 'K1')
    remaining = [c for c in first if numeric(c)]
    largest = 0
    while remaining:
        component = [remaining.pop(0)]
        while True:
            adjacent = [c for c in remaining if any(same(c['execution']['value'], v['execution']['value']) for v in component)]
            if not adjacent:
                break
            component.extend(adjacent)
            identities = {id(c) for c in adjacent}
            remaining = [c for c in remaining if id(c) not in identities]
        largest = max(largest, len(component))
    return largest/len(first)


def select_available(record, models, **kwargs):
    available = [c for c in candidates(record, models)[1:] if
                 c.get('execution', {}).get('status', 'budget_not_evaluated') != 'budget_not_evaluated']
    return select(record['greedy'], available, [], k=len(available), **kwargs), len(available)


def replay_record(greedy, samples, models, threshold=1.):
    record = {'greedy': greedy, 'samples': {m['name']: samples[m['name']][:m['K1']] for m in models}}
    decision = stage1_decision(record, models, threshold)
    if decision is None:
        raise ValueError('Offline replay requires completed stage-one executions')
    if not decision:
        record['samples'] = {m['name']: samples[m['name']][:m['K2']] for m in models}
    winner, count = select_available(record, models)
    return winner, decision, count
