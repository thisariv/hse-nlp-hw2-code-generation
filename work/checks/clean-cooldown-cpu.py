"""Boundary/tie checks and real order reproducibility through the L2 loader."""
import ast
import hashlib
import json
import math
import runpy
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

WORK = Path(__file__).resolve().parents[1]
module = runpy.run_path(str(WORK / 'research/clean-cooldown.py'))
select, normal, sha = (module[k] for k in ['select', 'normal', 'sha'])
report = json.loads((WORK / 'results/clean-cooldown-analysis.json').read_text())

# Answer agreement is sufficient; textual program identity is irrelevant.
fixtures = [
    ([2., 2., 3.], [True, True, False], [True, True, False]),
    ([2., 3.], [False, False], [True, True]),
    ([2., 2., 3., 4.], [False]*4, [True, True, False, False]),
    ([2., 2., 3., 3., 4.], [False]*5, [True, True, True, True, False]),
    ([2., 2. + 1e-7], [True, True], [True, True]),
]
for values, strict, conservative in fixtures:
    _, _, s, c = select(values)
    assert s.tolist() == strict and c.tolist() == conservative
assert normal('  Three\nAPPLES  ') == 'three apples'
assert normal('3 apples') != normal('4 apples')

cache = WORK / 'results/kaggle-runs/nlp-hw02-compact-cache-verified/2/compact-cache'
manifest = json.loads((cache / 'compact-manifest.json').read_text())
assert sha(cache / 'compact-manifest.json') == report['cache_manifest_sha256']
source, target = [], []
for rec in manifest['shards']:
    folder = cache / rec['folder']
    source.append(np.diff(np.load(folder / 'source-offsets.npy', mmap_mode='r')).astype(np.uint16))
    target.append(np.diff(np.load(folder / 'target-offsets.npy', mmap_mode='r')).astype(np.uint16))
source, target = np.concatenate(source), np.concatenate(target)
receipts = {}
for rule, info in report['orders'].items():
    folder = WORK / info['folder']
    meta = json.loads((folder / 'order.json').read_text())
    path = folder / 'order.npy'
    assert sha(path) == meta['file']['sha256']
    assert sha(folder / 'order.json') == info['manifest_sha256']
    loaded = np.load(path, mmap_mode='r', allow_pickle=False)
    assert loaded.dtype == np.uint32 and loaded.ndim == 1
    retained = np.sort(loaded)
    assert np.all(retained[1:] > retained[:-1]) and retained[-1] < len(source)
    assert len(retained) == meta['slots'] == report['rules'].get(rule, report['control'])['rows']
    assert not np.isin(retained, report['protected']['removed_cache_indices']).any()
    # Reproduce every slot in RAM with the documented seed/bucketing recipe.
    expected = retained.copy()
    rng = np.random.default_rng(meta['seed'])
    rng.shuffle(expected)
    for lo in range(0, len(expected), meta['chunk']):
        part = expected[lo:lo + meta['chunk']]
        ns, nt = source[part].astype(np.int32), target[part].astype(np.int32)
        part = part[np.lexsort((nt, ns, ns + nt + 1))]
        full = len(part) // meta['batch']
        if full:
            part[:full * meta['batch']] = part[:full * meta['batch']].reshape(
                full, meta['batch'])[rng.permutation(full)].reshape(-1)
        expected[lo:lo + len(part)] = part
    assert np.array_equal(loaded, expected)
    assert hashlib.sha256(retained.astype('<u4').tobytes()).hexdigest() == meta['selected_cache_indices_sha256']
    loader = module['load_function'](WORK / 'kaggle/src/l2_sampler.py', 'load_order',
                                     {'Path': Path, 'np': np, 'json': json, 'sha': sha})
    class CacheRows:
        def __len__(self):
            return len(source)
    checked, _ = loader(folder, CacheRows(), meta['file']['sha256'])
    assert np.array_equal(checked, expected)
    # Real wrap-around semantics of the existing training sampler.
    take = module['load_function'](WORK / 'kaggle/src/l1_runtime.py', 'take', {'np': np})
    for cursor in [0, len(loaded)-17, len(loaded)+3]:
        assert np.array_equal(take(loaded, cursor, 128),
                              loaded[(np.arange(128)+cursor) % len(loaded)])
    receipts[rule] = {'slots': len(loaded), 'sha256': sha(path),
                      'deterministic_replay': True, 'unique_in_bounds': True,
                      'unmodified_L2_loader': True, 'epoch_wraparound': True}
    del expected, retained, checked

assert report['protected']['exact_overlaps'] == report['protected']['normalized_overlaps'] == 0
assert report['execution_reuse']['missing_numeric_receipts'] == report['execution_reuse']['new_program_executions'] == 0
assert report['census']['singleton_rows'] + report['census']['duplicate_rows'] == report['cache_rows']
assert report['census']['conflicting_groups'] + report['census']['unanimous_groups'] == report['census']['duplicate_groups']
assert sum(v['rows'] for v in report['lengths']['original_BPE']['bins'].values()) == report['cache_rows']
for info in report['rules'].values():
    assert info['rows'] + info['dropped_rows'] == report['cache_rows']
    assert sum(v['rows'] for v in info['original_BPE']['bins'].values()) == info['rows']

# Rejoin the saved raw-source proxy to both L2 models, independently of orders.
# This verifies the observational analysis; it cannot create a train overlap.
proxy_path = WORK / 'results/clean-cooldown/heldout-source-proxy.json'
proxy = json.loads(proxy_path.read_text())
heldout = json.loads((module['INPUT'] / 'heldout4k.json').read_text())
labels = proxy['labels']
assert len(labels) == len(heldout) == 4096
assert dict(Counter(labels)) == report['heldout']['label_counts']
assert all((label == 'insufficient') == (n < 2)
           for label, n in zip(labels, proxy['numeric_peer_counts']))
assert report['heldout']['actual_cache_join_labels'] == {'absent': 4096}
splits = {'heldout4k': list(range(4096)), **module['load_function'](
    WORK / 'kaggle/src/l2_sampler.py', 'fixed_split', {'hashlib': hashlib})()}
assert not set(splits['tune2048']) & set(splits['eval2048'])
assert sorted(splits['tune2048'] + splits['eval2048']) == splits['heldout4k']
l2_receipts = {}
presentations = {}
for model, evidence in report['heldout']['models'].items():
    module['verify'](WORK / evidence['input']['file'], evidence['input'])
    rows = json.loads((WORK / evidence['input']['file']).read_text())
    assert len(rows) == 4096
    for i, (row, source) in enumerate(zip(rows, heldout)):
        assert row['id'] == i
        assert row['question_sha256'] == hashlib.sha256(source['question'].encode()).hexdigest()
        good = row['execution']['status'] == row['reference']['status'] == 'numeric' and math.isclose(
            row['execution']['value'], row['reference']['value'], rel_tol=1e-6, abs_tol=1e-6)
        assert row['correct'] == good
    for split, ids in splits.items():
        stored = evidence['subsets'][split]
        for label in ['conflicting', 'unanimous', 'insufficient']:
            selected = [i for i in ids if labels[i] == label]
            assert stored[label]['n'] == len(selected)
            assert stored[label]['errors'] == sum(not rows[i]['correct'] for i in selected)
        # Replay the question-cluster bootstrap, rather than a row-independent CI.
        grouped = defaultdict(list)
        for label in ['conflicting', 'unanimous']:
            for i in ids:
                if labels[i] == label:
                    grouped[normal(heldout[i]['question'])].append(i)
        vectors = np.array([[sum(labels[i] == label and not rows[i]['correct'] for i in ii)
                             if field == 'errors' else sum(labels[i] == label for i in ii)
                             for label in ['conflicting', 'unanimous']
                             for field in ['errors', 'n']] for ii in grouped.values()])
        rng = np.random.default_rng(module['SEED'])
        deltas = []
        for _ in range(4000):
            e1, n1, e0, n0 = vectors[rng.integers(len(vectors), size=len(vectors))].sum(axis=0)
            if n1 and n0:
                deltas.append(100 * (e1/n1 - e0/n0))
        assert np.allclose(np.quantile(deltas, [.025, .975]),
                           stored['question_cluster_bootstrap95_pp'], rtol=0, atol=1e-12)
    l2_receipts[model] = {'input': evidence['input'], 'IDs_SHA_correctness_rejoined': True,
                          'all_split_counts_and_cluster_CIs_replayed': True,
                          'heldout4k': {k: v for k, v in evidence['subsets']['heldout4k'].items()
                                        if k != 'by_original_length'}}
    training_path = WORK / ('results/kaggle-runs/nlp-hw02-kaggle-l2/1/'
                           f'model-{model}/points/l2-28800/training.json')
    training = json.loads(training_path.read_text())
    n = training['cooldown_updates'] * 128
    presentations[model] = {'input': module['record'](training_path), 'attempted_presentations': n,
                             'expected_R1_excluded_at_uniform_sampling': n *
                                 report['rules']['strict-two-thirds']['consensus_removed_rows'] / report['control']['rows'],
                             'expected_arbitrary_nonmodal_at_uniform_sampling': n *
                                 report['census']['nonmodal_row_fraction_all_train']}

# Sensitivity scenarios for L4, not measured effects or a training-variance CI.
z = NormalDist().inv_cdf(1 - .025 / 2)  # two models, family alpha .05
z80 = NormalDist().inv_cdf(.8)
planning = {'n_eval': 2048, 'CI_level': .975, 'power': .8,
            'approximation': 'normal paired binary difference near zero; independent questions; fixed trained weights',
            'scenarios': [{'discordance_fraction': q,
                           'CI_halfwidth_pp': 100*z*math.sqrt(q/2048),
                           'minimum_effect_80pct_power_pp': 100*(z+z80)*math.sqrt(q/2048),
                           'n_for_half_pp_80pct_power': math.ceil(q*((z+z80)/.005)**2)}
                          for q in [.005, .01, .05, .10]],
            'L2_uniform_sampling_scale': presentations}
for filename in ['research/clean-cooldown.py', 'checks/clean-cooldown-cpu.py']:
    ast.parse((WORK / filename).read_text())
receipt = {'status': 'passed', 'fixtures': len(fixtures), 'orders': receipts,
           'full_compact_SHA': '360/360 verified during analysis',
           'full_raw_SHA': '60/60 verified during analysis',
           'original_length_SHA_alignment': '60/60 verified during analysis',
           'protected_exact_normalized_hash_heldout_overlaps': 0,
           'L2_raw_proxy': {'input': module['record'](proxy_path), 'labels': dict(Counter(labels)),
                            'models': l2_receipts, 'direct_train_link_available': False},
           'L4_sensitivity': planning,
           'new_executions_GPU_Kaggle': 0, 'temporary_arrays_written': 0}
(WORK / 'checks/clean-cooldown-cpu.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
