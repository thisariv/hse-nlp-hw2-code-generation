"""Preregistered CPU selection study. Scorers accept no reference or gold feature."""
import ast
import hashlib
import importlib.metadata
import itertools
import json
import math
import platform
import time
from collections import Counter
from functools import lru_cache

import numpy as np
from scipy.stats import binomtest
from tokenizers import Tokenizer

from common import CONSTANTS, INPUT, LOCAL, S1, WORK, bucket, question_numbers, read, write
from exp26_common import json_hash, sha_file
from exp26_fast_exec import run_chunks

OUT = WORK / 'research/outputs/candidate-selection'
TAUS = [.05, .1, .2, .5, 1., 2.]
ALPHAS = [0., .25, .5, 1., 2., 4., 8., 16., 32.]
VOTE_WEIGHTS = [0., .5, 1., 2., 4., 8.]
BETAS = [0., .25, .5, 1., 2., 4.]
DELTAS = [0., .25, .5, 1., 2.]
FAMILIES = ['vote', 'probability-sum', 'probability-sum-dedup',
            'probability-mean', 'probability-mean-dedup', 'vote-max-lp', 'vote-mean-lp',
            'greedy-weight', 'temperature-mix', 'temperature-balanced',
            'number-coverage', 'number-full', 'number-novel', 'number-channel',
            'answer-integer', 'answer-round', 'probability-number',
            'probability-prior', 'probability-number-prior']


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def logsumexp(values):
    # These vectors contain 1–65 elements. Avoid SciPy array-dispatch overhead
    # inside hundreds of thousands of small feature calculations.
    m = max(values)
    return float(m + math.log(sum(math.exp(x-m) for x in values)))


@lru_cache(maxsize=None)
def numeric_literals(code):
    """Signed literals, without double counting the child of unary +/- nodes."""
    tree = ast.parse(code)
    values = set()
    def visit(node):
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
                and isinstance(node.operand, ast.Constant)
                and type(node.operand.value) in (int, float)):
            v = float(node.operand.value)
            values.add(-v if isinstance(node.op, ast.USub) else v)
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            values.add(float(node.value))
        else:
            for child in ast.iter_child_nodes(node):
                visit(child)
    visit(tree)
    return frozenset(values)


def number_features(question, code):
    q = {x[0] for x in question_numbers(question)}
    vals = numeric_literals(code)
    coverage = sum(any(near(x, y) for y in vals) for x in q) / len(q) if q else 0.
    novel = sum(not any(near(x, y) for y in q | CONSTANTS) for x in vals) / len(vals) if vals else 0.
    return coverage, float(bool(q) and coverage == 1.), novel


def answer_features(value):
    integer = float(abs(value - round(value)) <= 1e-6)
    multiple10 = float(value != 0 and abs(value - 10 * round(value / 10)) <= 1e-6)
    return integer, integer + multiple10


def load_split(split):
    root = S1 / 'gpu0-exp27' / split
    rows, receipts = [], []
    for p in sorted((root / 'candidates').glob('batch-*.json')):
        if '.sha.' in p.name:
            continue
        digest = sha_file(p)
        assert digest == read(p.with_suffix('.sha.json'))['sha256']
        receipts.append([p.name, digest])
        rows.extend(read(p))
    rows.sort(key=lambda r: r['id'])
    original = read(INPUT / (split + '.json'))
    assert [r['id'] for r in rows] == list(range(len(original)))
    assert all(r['question'] == x['question'] for r, x in zip(rows, original))
    flat = []
    for r in rows:
        assert list(r['samples']) == ['0.7', '0.5', '1.0']
        assert [len(v) for v in r['samples'].values()] == [32, 16, 16]
        rr = [dict(r['greedy'], temperature='greedy')]
        rr += [dict(c, temperature=t) for t, cc in r['samples'].items() for c in cc]
        flat.extend(rr)
        r['flat'] = rr
    codes = [c['code'] for c in flat]
    identity = dict(codes_sha256=json_hash(codes), candidate_count=len(codes))
    if (root / 'execution/identity.json').exists():
        assert read(root / 'execution/identity.json') == identity
    saved = {}
    execution_receipts = []
    for p in sorted((root / 'execution').glob('chunk-*.json')):
        digest = sha_file(p)
        assert digest == read(p.with_suffix('.sha'))['sha256']
        saved[int(p.stem.split('-')[1])] = read(p)
        execution_receipts.append([p.name, digest])
    sizes = [len(codes[i:i+2000]) for i in range(0, len(codes), 2000)]
    assert all(len(v) == sizes[i] for i, v in saved.items())
    missing = len(saved) != len(sizes)
    if missing:
        result = run_chunks(codes, workers=4, chunk_rows=2000, saved_chunks=saved,
                            wall_limit=1., cpu_limit=2.)
        execution = result['rows']
        write(OUT / (split + '-reexecution.json'), result)
    else:
        execution = [e for i in range(len(sizes)) for e in saved[i]]
    assert len(flat) == len(execution)
    for c, e in zip(flat, execution):
        assert math.isfinite(c['mean_logprob']) and c['length'] > 0
        c['execution'] = e
    return rows, original, dict(rows=len(rows), candidates=len(flat), reexecuted_missing=missing,
        candidate_files=len(receipts), candidate_receipts_sha256=json_hash(receipts),
        execution_files=len(execution_receipts), execution_receipts_sha256=json_hash(execution_receipts),
        execution_identity=identity, input_sha256=sha_file(INPUT / (split + '.json')),
        reference_sha256=sha_file(INPUT / (split + '-reference-execution.json')),
        execution_statuses=dict(Counter(e['status'] for e in execution)))


def indices(kind, k):
    if kind == 'all':
        return list(range(65))
    if kind == 'mixed':
        return [0] + list(range(1, 1+k//2)) + list(range(33, 33+k//4)) + list(range(49, 49+k//4))
    return ([0] if kind == 'pure' else []) + list(range(1, k+1))


def clusters(row, ids):
    good = [row['flat'][i] for i in ids if row['flat'][i]['execution']['status'] == 'numeric']
    parent = list(range(len(good)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(good):
        for j, b in enumerate(good[:i]):
            if near(a['execution']['value'], b['execution']['value']):
                parent[root(i)] = root(j)
    groups = {}
    for i, c in enumerate(good):
        groups.setdefault(root(i), []).append(c)
    greedy = row['flat'][0]['execution']
    def key(cc):
        has_greedy = greedy['status'] == 'numeric' and any(near(c['execution']['value'], greedy['value']) for c in cc)
        return len(cc), has_greedy, sum(c['mean_logprob'] for c in cc)/len(cc)
    return sorted(groups.values(), key=key, reverse=True)


def pack(rows, kind, k):
    """Feature matrix contains only question, candidate code, probability and execution."""
    begin = time.perf_counter()
    ids = indices(kind, k)
    groups = [clusters(r, ids) for r in rows]
    shape = len(rows), max(1, max(map(len, groups)))
    names = ['n', 'unique', 'greedy', 't07', 't05', 't10', 'lpmax', 'lpmean',
             'coverage', 'full', 'novel', 'integer', 'round', 'value', 'sum', 'sum-dedup']
    names += [f'mean-{t}' for t in TAUS] + [f'mean-dedup-{t}' for t in TAUS]
    a = {name: np.zeros(shape, dtype=np.float64) for name in names}
    a['valid'] = np.zeros(shape, dtype=bool)
    nontransitive = 0
    for i, (r, gg) in enumerate(zip(rows, groups)):
        for j, cc in enumerate(gg):
            a['valid'][i, j] = True
            unique = {}
            for c in cc:
                if c['code'] not in unique or c['mean_logprob'] > unique[c['code']]['mean_logprob']:
                    unique[c['code']] = c
            dedup = list(unique.values())
            lp = np.array([c['mean_logprob'] for c in cc])
            length = np.array([c['length'] for c in cc])
            dlp = np.array([c['mean_logprob'] for c in dedup])
            dl = np.array([c['length'] for c in dedup])
            winner = max(cc, key=lambda c: c['mean_logprob'])
            v = winner['execution']['value']
            a['n'][i, j], a['unique'][i, j] = len(cc), len(dedup)
            a['lpmax'][i, j], a['lpmean'][i, j] = lp.max(), lp.mean()
            a['value'][i, j] = v
            temp = Counter(c['temperature'] for c in cc)
            for name, t in [('greedy', 'greedy'), ('t07', '0.7'), ('t05', '0.5'), ('t10', '1.0')]:
                a[name][i, j] = temp[t]
            ff = np.array([c['number_features'] for c in cc]).mean(axis=0)
            a['coverage'][i, j], a['full'][i, j], a['novel'][i, j] = ff
            a['integer'][i, j], a['round'][i, j] = answer_features(v)
            a['sum'][i, j], a['sum-dedup'][i, j] = logsumexp(lp*length), logsumexp(dlp*dl)
            for t in TAUS:
                a[f'mean-{t}'][i, j], a[f'mean-dedup-{t}'][i, j] = logsumexp(lp/t), logsumexp(dlp/t)
            nontransitive += any(not near(x['execution']['value'], v) for x in cc)
    a['kind'], a['k'] = kind, k
    a['pool_indices'] = ids
    a['build_seconds'] = time.perf_counter()-begin
    a['nontransitive_components'] = int(nontransitive)
    return a


def configs(family):
    if family in ['vote', 'probability-sum', 'probability-sum-dedup', 'temperature-balanced']:
        return [{}]
    if family in ['probability-mean', 'probability-mean-dedup']:
        return [dict(tau=t) for t in TAUS]
    if family in ['vote-max-lp', 'vote-mean-lp']:
        return [dict(alpha=x) for x in ALPHAS]
    if family == 'greedy-weight':
        return [dict(g=x) for x in [0., .5, 1., 2., 4., 8., 16.]]
    if family == 'temperature-mix':
        return [dict(lam=x, g=g) for x, g in itertools.product([.25, .5, 1., 2., 4.], [0., 1., 2., 4., 8.])]
    if family.startswith('number-'):
        return [dict(beta=x) for x in VOTE_WEIGHTS]
    if family.startswith('answer-'):
        return [dict(delta=x) for x in VOTE_WEIGHTS]
    if family == 'probability-number':
        return [dict(tau=t, beta=b) for t, b in itertools.product(TAUS, BETAS)]
    if family == 'probability-prior':
        return [dict(tau=t, delta=d) for t, d in itertools.product(TAUS, DELTAS)]
    if family == 'probability-number-prior':
        return [dict(tau=t, beta=b, delta=d) for t, b, d in itertools.product(TAUS, BETAS, DELTAS)]
    raise ValueError(family)


def complexity(cfg):
    defaults = dict(tau=1., g=1., lam=1., alpha=0., beta=0., delta=0.)
    deviations = [abs(math.log2(v)) if key in ['tau', 'lam'] else abs(v-defaults[key])
                  for key, v in cfg.items()]
    return sum(x != 0 for x in deviations), sum(deviations)


def score(a, family, cfg):
    n = a['n']
    if family == 'vote':
        s = n
    elif family == 'probability-sum':
        s = a['sum']
    elif family == 'probability-sum-dedup':
        s = a['sum-dedup']
    elif family == 'probability-mean':
        s = a[f"mean-{cfg['tau']}"]
    elif family == 'probability-mean-dedup':
        s = a[f"mean-dedup-{cfg['tau']}"]
    elif family == 'vote-max-lp':
        s = n + cfg['alpha']*a['lpmax']
    elif family == 'vote-mean-lp':
        s = n + cfg['alpha']*a['lpmean']
    elif family == 'greedy-weight':
        s = n + (cfg['g']-1)*a['greedy']
    elif family == 'temperature-mix':
        s = a['t07'] + cfg['lam']*a['t05'] + a['t10']/cfg['lam'] + cfg['g']*a['greedy']
    elif family == 'temperature-balanced':
        counts = Counter('greedy' if i == 0 else 't07' if i <= 32 else 't05' if i <= 48 else 't10'
                         for i in a['pool_indices'])
        s = a['greedy'].copy()
        for name in ['t07', 't05', 't10']:
            if counts[name]:
                s += a[name]/counts[name]
    elif family.startswith('number-'):
        f = {'number-coverage': a['coverage'], 'number-full': a['full'],
             'number-novel': -a['novel'], 'number-channel': a['coverage']-a['novel']}[family]
        s = n + cfg['beta']*f
    elif family.startswith('answer-'):
        s = n + cfg['delta']*a['integer' if family == 'answer-integer' else 'round']
    elif family.startswith('probability-'):
        s = (a[f"mean-{cfg['tau']}"] + cfg.get('beta', 0)*(a['coverage']-a['novel'])
             + cfg.get('delta', 0)*a['round'])
    else:
        raise ValueError(family)
    return np.where(a['valid'], s, -np.inf)


def correct_matrix(a, refs):
    out = np.zeros(a['valid'].shape, dtype=bool)
    for i, ref in enumerate(refs):
        if ref['status'] == 'numeric':
            tolerance = np.maximum(1e-6, 1e-6*np.maximum(np.abs(a['value'][i]), abs(ref['value'])))
            out[i] = a['valid'][i] & (np.abs(a['value'][i]-ref['value']) <= tolerance)
    return out


def choose(a, family, cfg):
    return score(a, family, cfg).argmax(axis=1)


def flags(labels, picks):
    return labels[np.arange(len(labels)), picks]


def tune(a, labels, mask, family):
    search = []
    for order, cfg in enumerate(configs(family)):
        matched = int(flags(labels, choose(a, family, cfg))[mask].sum())
        search.append(dict(parameters=cfg, matches=matched, complexity=list(complexity(cfg)), order=order))
    best = min(search, key=lambda x: (-x['matches'], *x['complexity'], x['order']))
    return best['parameters'], search


def metrics(f, base, mask, lengths):
    f, base, lengths = f[mask], base[mask], lengths[mask]
    n = len(f)
    gains, losses = int((f & ~base).sum()), int((base & ~f).sum())
    paired = f.astype(float)-base.astype(float)
    delta = float(paired.mean())
    se = float(paired.std(ddof=1)/math.sqrt(n))
    return dict(n=n, matches=int(f.sum()), accuracy=float(f.mean()),
        delta_vote_pp=100*delta, gains=gains, losses=losses,
        mcnemar_exact_p=float(binomtest(min(gains, losses), gains+losses, .5).pvalue) if gains+losses else 1.,
        delta_paired_normal_95ci_pp=[100*(delta-1.96*se), 100*(delta+1.96*se)],
        by_reference_length={b: dict(n=int((lengths == b).sum()), matches=int(f[lengths == b].sum()),
            accuracy=float(f[lengths == b].mean()) if (lengths == b).any() else None,
            delta_vote_pp=float(100*paired[lengths == b].mean()) if (lengths == b).any() else None)
            for b in ['<60', '60-90', '90-120', '>=120']})


def potential(a, labels, picks, scores, rows, refs, mask):
    idx = np.flatnonzero(mask)
    correct = labels[idx]
    current = flags(labels, picks)[idx]
    any_cluster = correct.any(axis=1)
    repeated = (correct & (a['n'][idx] >= 2)).any(axis=1)
    repeated_unique = (correct & (a['unique'][idx] >= 2)).any(axis=1)
    oracle_candidate = np.array([any(c['execution']['status'] == 'numeric' and
        near(c['execution']['value'], refs[i]['value']) for c in [rows[i]['flat'][j] for j in a['pool_indices']]) for i in idx])
    ranks, repeat_ranks = [], []
    for i in idx:
        order = np.argsort(-scores[i], kind='stable')
        hits = np.flatnonzero(labels[i, order])
        rhits = np.flatnonzero(labels[i, order] & (a['n'][i, order] >= 2))
        ranks.append(int(hits[0]+1) if len(hits) else 0)
        repeat_ranks.append(int(rhits[0]+1) if len(rhits) else 0)
    ranks = np.array(ranks)
    def hist(rr):
        return {name: int(m.sum()) for name, m in [('1', rr == 1), ('2', rr == 2), ('3', rr == 3),
            ('4-5', (rr >= 4) & (rr <= 5)), ('6-10', (rr >= 6) & (rr <= 10)), ('>10', rr > 10), ('absent', rr == 0)]}
    def summary(f):
        return dict(matches=int(f.sum()), accuracy=float(f.mean()), delta_best_pp=float(100*(f.mean()-current.mean())))
    return dict(n=len(idx), selected_matches=int(current.sum()),
        any_candidate_oracle=summary(oracle_candidate), cluster_representative_oracle=summary(any_cluster),
        repeated_cluster_oracle=summary(repeated), repeated_distinct_programs_oracle=summary(repeated_unique),
        repair_repeated_keep_other_predictions=summary(current | repeated),
        repair_repeated_distinct_keep_other_predictions=summary(current | repeated_unique),
        correct_rank_histogram=hist(ranks), correct_rank_histogram_on_errors=hist(ranks[~current]),
        repeated_correct_rank_histogram=hist(np.array(repeat_ranks)),
        cumulative_top_rank={str(k): int(((ranks > 0) & (ranks <= k)).sum()) for k in [1, 2, 3, 5, 10]},
        correct_cluster_available_but_only_singletons=int((any_cluster & ~repeated).sum()),
        mean_clusters=float(a['valid'][idx].sum(axis=1).mean()))


def train_prior():
    sample = read(LOCAL / 'sample-60k.json')
    execution = read(LOCAL / 'normalization-execution.json')
    finite = [r for r in sample if execution[r['code']]['status'] == 'numeric']
    ns = Counter(r['shard'] for r in finite)
    populations = {}
    for s in range(60):
        root = WORK / ('kaggle/staging/drive-merged/26-token-cache/shards' if s < 42 else
            'results/kaggle-runs/nlp-hw02-26c-cpu-cache/2/26c-token-cache/shards')
        populations[f'part-{s:05d}.parquet'] = len(np.load(root / f'part-{s:05d}/provenance-rows.npy', mmap_mode='r'))
    ff = np.array([answer_features(execution[r['code']]['value']) for r in finite])
    weights = np.array([populations[r['shard']]/ns[r['shard']] for r in finite])
    return dict(sample_finite=len(finite), finite_population=sum(populations.values()),
        integer_count=int(ff[:, 0].sum()), nonzero_multiple10_count=int((ff[:, 1]-ff[:, 0]).sum()),
        integer_fraction_unweighted=float(ff[:, 0].mean()),
        nonzero_multiple10_fraction_unweighted=float((ff[:, 1]-ff[:, 0]).mean()),
        integer_fraction_weighted=float(np.average(ff[:, 0], weights=weights)),
        nonzero_multiple10_fraction_weighted=float(np.average(ff[:, 1]-ff[:, 0], weights=weights)),
        sample_sha256=sha_file(LOCAL / 'sample-60k.json'), sample_seed=20261012)


def direct_vote(row, ids):
    """Independent reference implementation of the historical connected-component vote."""
    good = [row['flat'][i] for i in ids if row['flat'][i]['execution']['status'] == 'numeric']
    remaining = set(range(len(good)))
    gg = []
    while remaining:
        todo = [min(remaining)]
        component = []
        while todo:
            i = todo.pop()
            if i not in remaining:
                continue
            remaining.remove(i)
            component.append(i)
            todo.extend(j for j in sorted(remaining) if near(good[i]['execution']['value'], good[j]['execution']['value']))
        gg.append([good[i] for i in sorted(component)])
    if not gg:
        return None
    greedy = row['flat'][0]['execution']
    def key(cc):
        return (len(cc), greedy['status'] == 'numeric' and any(near(c['execution']['value'], greedy['value']) for c in cc),
                sum(c['mean_logprob'] for c in cc)/len(cc))
    return max(max(gg, key=key), key=lambda c: c['mean_logprob'])['execution']['value']


def method_checks():
    assert numeric_literals('x = -3; y = +2; z = 2.5') == {-3., 2., 2.5}
    assert number_features('Use 3 and 5', 'x = 3; y = 5') == (1., 1., 0.)
    assert number_features('Use 3 and 5', 'x = 3; y = 19') == (.5, 0., .5)
    assert number_features('Nothing', 'x = 1') == (0., 0., 0.)
    assert answer_features(30.) == (1., 2.) and answer_features(0.) == (1., 1.)
    assert answer_features(2.25) == (0., 0.)
    # numpy.isclose uses additive/asymmetric tolerances; the study uses math.isclose.
    a = dict(valid=np.ones((1, 2), bool), value=np.array([[1.0000015, 1.0000005]]))
    assert correct_matrix(a, [dict(status='numeric', value=1.)]).tolist() == [[False, True]]
    # Known paired test and no-change identity, without touching study labels.
    dummy = np.array(['<60']*4)
    m = metrics(np.array([1, 1, 0, 0], bool), np.array([0, 0, 0, 0], bool), np.ones(4, bool), dummy)
    assert m['gains'] == 2 and m['losses'] == 0 and m['mcnemar_exact_p'] == .5


def main():
    begin = time.perf_counter()
    method_checks()
    preregistration = OUT / 'preregistration.json'
    if not preregistration.exists():
        # Replaying a committed study must preserve its original registration.
        registered = read(WORK / 'results/candidate-selection-cpu.json')['protocol']
        OUT.mkdir(parents=True, exist_ok=True)
        write(preregistration, registered)
    else:
        registered = read(preregistration)
    protocol = (WORK / 'notes/candidate-selection-cpu.md').read_text().split('## Результаты')[0]
    assert hashlib.sha256(protocol.encode()).hexdigest() == registered['protocol_sha256']
    study = dict(protocol=registered, python=platform.python_version(), packages={p: importlib.metadata.version(p)
        for p in ['numpy', 'scipy', 'tokenizers']}, cpu_only=True, trained_models=0, gpu_calls=0, kaggle_calls=0,
        sources={}, train_prior=train_prior(), search={}, results={}, winners={}, reranker_potential={})
    rows, originals, refs = {}, {}, {}
    for split in ['heldout4k', 'old1024']:
        rows[split], originals[split], study['sources'][split] = load_split(split)
        refs[split] = read(INPUT / (split + '-reference-execution.json'))['rows']
        print('loaded', split, len(rows[split]), flush=True)
    ordered = sorted(range(4096), key=lambda i: hashlib.sha256(f'candidate-selection-v1:heldout4k:{i}'.encode()).digest())
    tune_ids, eval_ids = sorted(ordered[:2048]), sorted(ordered[2048:])
    tm, em = np.zeros(4096, bool), np.zeros(4096, bool)
    tm[tune_ids], em[eval_ids] = True, True
    om = np.array([r['status'] == 'numeric' for r in refs['old1024']])
    assert tm.sum() == em.sum() == 2048 and not (tm & em).any() and om.sum() == 991
    study['split'] = dict(tune_ids=tune_ids, eval_ids=eval_ids, tune_ids_sha256=json_hash(tune_ids),
                         eval_ids_sha256=json_hash(eval_ids), old_eligible_ids=np.flatnonzero(om).tolist())
    # References are kept outside all feature matrices/scorers.
    feature_begin = time.perf_counter()
    for split in rows:
        for r in rows[split]:
            memo = {}
            for c in r['flat']:
                if c['execution']['status'] == 'numeric':
                    if c['code'] not in memo:
                        memo[c['code']] = number_features(r['question'], c['code'])
                    c['number_features'] = memo[c['code']]
    feature_seconds = time.perf_counter()-feature_begin
    print('number_features_seconds', round(feature_seconds, 2), flush=True)
    tok = Tokenizer.from_file(str(INPUT / 'tokenizer.json'))
    lengths = {s: np.array([bucket(len(x.ids)) for x in tok.encode_batch([r['code'] for r in originals[s]],
        add_special_tokens=False)]) for s in rows}
    layouts = [('pure', k) for k in [8, 16, 32]] + [('all', 64)] + [('mixed', k) for k in [8, 16, 32]]
    packs, labels, baseline = {}, {}, {}
    for kind, k in layouts + [('nogreedy', k) for k in [8, 16, 32]]:
        name = f'{kind}{k}'
        packs[name] = {s: pack(rows[s], kind, k) for s in rows}
        labels[name] = {s: correct_matrix(a, refs[s]) for s, a in packs[name].items()}
        baseline[name] = {s: flags(labels[name][s], choose(a, 'vote', {})) for s, a in packs[name].items()}
        print('packed', name, flush=True)
    # Frozen historical controls validate the data ordering and voting semantics.
    assert baseline['pure16']['heldout4k'].sum() == 2703
    assert baseline['nogreedy16']['heldout4k'].sum() == 2707
    for name, expected in [('pure16', 3314), ('pure32', 3468), ('all64', 3605)]:
        assert labels[name]['heldout4k'].any(axis=1).sum() == expected
    checks = 0
    for name in packs:
        for s in rows:
            for i in range(len(rows[s])):
                expected = direct_vote(rows[s][i], packs[name][s]['pool_indices'])
                a = packs[name][s]
                actual = a['value'][i, 0] if a['valid'][i, 0] else None
                assert actual == expected, (name, s, i)
                checks += 1
    # All parameter choices are frozen before the evaluation reporting loop.
    chosen = {}
    for anchor in ['pure16', 'mixed16', 'all64']:
        study['search'][anchor] = {}
        for family in FAMILIES:
            if family in ['temperature-mix', 'temperature-balanced'] and anchor == 'pure16':
                continue
            cfg, search = tune(packs[anchor]['heldout4k'], labels[anchor]['heldout4k'], tm, family)
            chosen[(anchor, family)] = cfg
            study['search'][anchor][family] = search
        winner = min(study['search'][anchor], key=lambda f: (
            -max(x['matches'] for x in study['search'][anchor][f]), *complexity(chosen[(anchor, f)]), FAMILIES.index(f)))
        study['winners'][anchor] = dict(family=winner, parameters=chosen[(anchor, winner)],
            tune_matches=max(x['matches'] for x in study['search'][anchor][winner]))
    write(OUT / 'frozen-choices.json', study['winners'])
    report_begin = time.perf_counter()
    chosen_predictions = {}
    for name, pp in packs.items():
        anchor = 'all64' if name == 'all64' else 'mixed16' if name.startswith('mixed') else 'pure16'
        study['results'][name] = {}
        for family in FAMILIES:
            if name.startswith('nogreedy') and family != 'vote':
                continue
            if (anchor, family) not in chosen:
                continue
            cfg = chosen[(anchor, family)]
            ff = {s: flags(labels[name][s], choose(a, family, cfg)) for s, a in pp.items()}
            result = dict(parameters=cfg, parameter_anchor=anchor, free_parameters=len(cfg),
                k_samples=pp['heldout4k']['k'], candidate_count=len(pp['heldout4k']['pool_indices']),
                tune_matches=int(ff['heldout4k'][tm].sum()),
                eval=metrics(ff['heldout4k'], baseline[name]['heldout4k'], em, lengths['heldout4k']),
                old991=metrics(ff['old1024'], baseline[name]['old1024'], om, lengths['old1024']))
            if name.startswith('mixed'):
                pure = 'pure' + name.removeprefix('mixed')
                result['eval_against_pure_vote'] = metrics(ff['heldout4k'], baseline[pure]['heldout4k'], em, lengths['heldout4k'])
                result['old991_against_pure_vote'] = metrics(ff['old1024'], baseline[pure]['old1024'], om, lengths['old1024'])
            study['results'][name][family] = result
        family = study['winners'][anchor]['family'] if not name.startswith('nogreedy') else 'vote'
        cfg = chosen[(anchor, family)]
        if not name.startswith('nogreedy'):
            study['reranker_potential'][name] = {sname: potential(pp[s], labels[name][s], choose(pp[s], family, cfg),
                score(pp[s], family, cfg), rows[s], refs[s], mask) for sname, s, mask in
                [('eval', 'heldout4k', em), ('old991', 'old1024', om)]}
        for s in pp:
            chosen_predictions[f'{name}-{s}'] = choose(pp[s], family, cfg)
    study['QA'] = dict(method_fixtures='passed', direct_vote_comparisons=checks,
        historical_vote16_greedy=2703, historical_vote16_no_greedy=2707,
        historical_oracles=[3314, 3468, 3605], exact_split=True, old991_eligible=True,
        reference_not_in_score_arguments=True, protocol_unchanged=True,
        nontransitive_components={name: {s: a['nontransitive_components'] for s, a in pp.items()} for name, pp in packs.items()})
    study['timing'] = dict(total_cpu_seconds=time.perf_counter()-begin, number_feature_seconds=feature_seconds,
        report_and_oracle_seconds=time.perf_counter()-report_begin,
        feature_build_seconds={name: {s: a['build_seconds'] for s, a in pp.items()} for name, pp in packs.items()},
        note='Research includes many configurations and independent QA, not a production timer; execution reused.')
    main_rule = study['winners']['pure16']['family']
    main_result = study['results']['pure16'][main_rule]
    study['decision'] = dict(selection='pure16 tune winner; not replaced after inspecting eval/old',
        rule=main_rule, parameters=main_result['parameters'],
        observed_gain_eval_pp=main_result['eval']['delta_vote_pp'],
        observed_gain_old991_pp=main_result['old991']['delta_vote_pp'],
        expected_gain_interpretation='small 0–0.5 pp effect; zero/negative remains plausible; not a leaderboard forecast',
        production='keep voting as control; benchmark fast sampler k8 first, k16 only with full T4 pipeline <=180s',
        gpu_next='teacher-forcing B mean_logprob on unique A programs, one A/B mixing weight tuned only on tune',
        noisy_channel='code->question uses the same pairs but question targets: gray area; clarify with instructor before training',
        learned_verifier='oracle only; correctness targets for generated candidates are additional targets prohibited by task')
    study['script_sha256'] = sha_file(WORK / 'research/candidate-selection.py')
    np.savez_compressed(OUT / 'selected-clusters.npz', **chosen_predictions)
    write(WORK / 'results/candidate-selection-cpu.json', study)
    print('WINNERS', json.dumps(study['winners']), flush=True)
    print('CPU_SECONDS', study['timing']['total_cpu_seconds'], flush=True)


if __name__ == '__main__':
    main()
