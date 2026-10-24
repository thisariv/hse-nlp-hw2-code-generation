"""CPU audit of saved B5 output and K8 prefixes; no network/model/program execution.

Run with NumPy/Matplotlib and a C++17 compiler. B3's selector and two independent
oracles validate the reported choices. Tune search intervals are descriptive.
"""
import csv
import ctypes
from functools import lru_cache
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('b4_analysis', HERE/'analyze-b4-results.py')
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
B, np, plt = A.B, A.np, A.plt
WORK, RUNS, IDS = B.WORK, B.RUNS, B.IDS
RAW = RUNS/'nlp-hw02-b5/1'  # Local download label, NOT the production version.
NAMES = ['C-main', 'C-long2x', 'C-long2x-B5', 'pilot-b', 'W-B4', 'W-B5']
INPUTS, RECORDS = {}, {}
SPLITS = ['heldout4k', 'old991']
B.read = lru_cache(maxsize=None)(B.read)


def read(path):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(WORK))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def lines(path):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(WORK))] = hashlib.sha256(raw).hexdigest()
    return [json.loads(line) for line in raw.splitlines()]


@lru_cache(maxsize=None)
def bootstrap(n, gained, lost):
    rng = np.random.default_rng(2026101901)
    draws = rng.multinomial(n, [gained/n, lost/n, 1-(gained+lost)/n], size=20000)
    return np.quantile(100*(draws[:, 0]-draws[:, 1])/n, [.025, .975]).tolist()


def paired(a, b, boot=True):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    assert a.shape == b.shape and a.ndim == 1
    n, g, l = len(a), int(np.sum(a & ~b)), int(np.sum(b & ~a))
    mean = (g-l)/n
    se = math.sqrt(((g+l)-n*mean**2)/(n-1)/n)
    out = dict(n=n, gained=g, lost=l, delta_pp=100*mean,
               CI95_pp=[100*(mean-1.96*se), 100*(mean+1.96*se)])
    if boot:
        out['bootstrap_CI95_pp'] = bootstrap(n, g, l)
    return out


def verified_rows(path, refs):
    rows = read(path)
    assert len(rows) == len(refs)
    for i, (x, r) in enumerate(zip(rows, refs)):
        assert x['id'] == i
        assert all(x[k] == r[k] for k in ['question_sha256', 'reference', 'original_bpe_length'])
        assert x['correct'] == B.correct(x['reference'], x['execution'])
    return rows


def audit():
    session = read(RAW/'session-summary.json')
    assert 'COMPLETE' in (RAW/'kernel-status.txt').read_text()
    assert session['status'] == 'complete' and session['exit_codes'] == [0, 0]
    assert not session['unreaped_worker_pids']
    checked, missing = [], []
    for rec in read(RAW/'b1-output-manifest.json')['files']:
        p = RAW/rec['file']
        if not p.exists():
            missing.append(rec)
            continue
        raw = p.read_bytes()
        assert len(raw) == rec['bytes'] and hashlib.sha256(raw).hexdigest() == rec['sha256'], p
        checked.append(rec['file'])
        INPUTS[str(p.relative_to(WORK))] = rec['sha256']
    prior = read(WORK/'results/b4-results.json')
    # Bind all inherited inputs used below to the previous audited snapshot.
    for name, digest in prior['inputs_sha256'].items():
        p = WORK/name
        if p.exists() and (name.startswith('results/kaggle-runs/') or name == 'results/b3-results.json'):
            assert hashlib.sha256(p.read_bytes()).hexdigest() == digest, name
    refs = {s: read(RUNS/f'nlp-hw02-kaggle-l2/1/model-C-main/points/l2-10800/evaluation/{s}-per-example.json') for s in SPLITS}
    points = []

    def point(row, getter):
        row = dict(row, scores={}, CE={})
        metrics = read(getter('evaluation/metrics.json'))
        for split in SPLITS:
            rr = verified_rows(getter(f'evaluation/{split}-per-example.json'), refs[split])
            assert sum(x['correct'] for x in rr) == metrics['splits'][split]['exec']['matches']
            RECORDS[row['model'], row['tag'], split] = rr
            row['scores'][split] = B.scores(rr)
            row['CE'][split] = metrics['splits'][split]['CE']['CE']
            if split == 'heldout4k':
                row['bins'] = [B.scores([x for x in rr if lo <= x['original_bpe_length'] < hi])
                               for lo, hi in [(0, 60), (60, 90), (90, 120), (120, math.inf)]]
                row['scores']['tune2048'] = B.scores([rr[i] for i in IDS])
                row['scores']['eval2048'] = B.scores([x for x in rr if x['id'] not in set(IDS)])
        points.append(row)
        return row

    for old in prior['points']:
        if old['model'] != 'W600':
            continue
        tag = old['tag']
        run, gpu = ('b4', 0) if tag.startswith('b4') else ('b3', 1) if tag.startswith('b3') else ('b2', 1)
        root = RUNS/f'nlp-hw02-{run}/1/model-GPU{gpu}/points'/tag
        point(old, lambda suffix, root=root: root/suffix)
    for old in prior['C']:
        if old['model'] not in ['C (shared L1)', 'C-long2x', 'C-main']:
            continue
        name, tag = old['model'], old['tag']
        if tag == 't009000':
            # L1 stores generations/executions, not a later per-example format.
            root = RUNS/'nlp-hw02-kaggle-l2/1/l1-reference-live-v7'
            row = dict(old, scores={}, CE=dict(old['CE']))
            for split in SPLITS:
                gens = read(root/f'model-C__points__t009000__evaluation__{split}-generations.json')
                executions = read(root/f'model-C__points__t009000__evaluation__{split}-execution.json')['rows']
                assert len(gens) == len(executions) == len(refs[split])
                rr = []
                for gen, ex, ref in zip(gens, executions, refs[split]):
                    assert hashlib.sha256(gen['question'].encode()).hexdigest() == ref['question_sha256']
                    rr.append(dict(ref, execution=ex, correct=B.correct(ref['reference'], ex)))
                RECORDS[name, tag, split] = rr
                row['scores'][split] = B.scores(rr)
                assert row['scores'][split] == old['scores'][split]
                if split == 'heldout4k':
                    row['bins'] = [B.scores([x for x in rr if lo <= x['original_bpe_length'] < hi])
                                   for lo, hi in [(0, 60), (60, 90), (90, 120), (120, math.inf)]]
                    row['scores']['tune2048'] = B.scores([rr[i] for i in IDS])
                    row['scores']['eval2048'] = B.scores([x for x in rr if x['id'] not in set(IDS)])
            points.append(row)
            continue
        if tag in ['l2-45540', 'l2-62280']:
            root = RUNS/('nlp-hw02-kaggle-l3/mid' if tag == 'l2-45540' else 'nlp-hw02-kaggle-l3/final')
            prefix = f'model-{name}__points__{tag}__'
            getter = lambda suffix, root=root, prefix=prefix: root/(prefix+suffix.replace('/', '__'))
        else:
            root = RUNS/f'nlp-hw02-kaggle-l2/1/model-{name}/points/{tag}'
            getter = lambda suffix, root=root: root/suffix
        point(old, getter)

    stability, finals = {}, {}
    for gpu, name, lr, origin_amp in [(0, 'W600', .0006, 40), (1, 'C-long2x', .0015, 84)]:
        root = RAW/f'model-GPU{gpu}'
        summary = read(root/'summary.json')
        state, origin = summary['state'], read(root/'b5-origin.json')['anchor']
        assert summary['status'] == 'complete' and summary['final_complete'] and summary['main_target_reached']
        assert not summary['missed_points'] and not summary['budget_stopped_main']
        assert origin['sha256'] == ('f23fe1e36135e23b1ba1d393537ea501f25165a084b8462c16d8bc24843abc7a' if gpu == 0 else
                                    '48f2ba3fa4ffe6a722b45a084d3ee33a27cc4ed02b72433605c9cb1740f5a80b')
        assert read(root/'start-fingerprints.json')['source_sha256'] == origin['sha256']
        latest = read(root/'latest.json')
        assert latest == summary['B6_uncooled_latest']
        assert all(latest[k] == state[k] for k in ['updates', 'cursor', 'main_seconds'])
        assert state['cursor']-128*state['updates'] == (0 if gpu == 0 else -6035840)
        assert state['last_lr'] == lr and state.get('lr_factor', 1) == 1
        assert state['nonfinite_skips'] == state['stability']['total'] == state['stability']['escalations'] == 0
        assert not state['stability']['history']
        text = (RAW/f'gpu{gpu}-process.log').read_text()
        hits = re.findall(r'B5 (?:W600|C-long2x) (\d+) ([\d.]+) pairs (\d+) LR ([\d.]+)', text)
        assert hits and all(float(h[3]) == lr for h in hits)
        assert all(int(b[0])-int(a[0]) == 100 for a, b in zip(hits, hits[1:]))
        assert all(float(b[1]) > float(a[1]) for a, b in zip(hits, hits[1:]))
        assert not re.search(r'Traceback|WORKER FAILED|POINT INCOMPLETE|\b(?:NaN|nan|inf)\b', text)
        attempted = state['updates']-origin['updates']
        seconds = state['main_seconds']-origin['main_seconds']
        assert state['cursor']-origin['cursor'] == 128*attempted == 128*summary['stream']['attempted_batches']
        logs = lines(root/'updates.jsonl')
        assert len(logs) == attempted
        elapsed = origin['main_seconds']
        for j, x in enumerate(logs, 1):
            assert x['updates'] == origin['updates']+j and x['main_pairs'] == origin['cursor']+128*j
            assert x['lr'] == lr and x['nonfinite_skip'] == 0 and not x['OOM_retries']
            assert all(math.isfinite(x[k]) for k in ['CE']+(['z_loss'] if gpu == 0 else []))
            assert x['AMP_skip'] or math.isfinite(x['gradient_norm'])
            elapsed += x['seconds']
            assert math.isclose(elapsed, x['main_seconds'], abs_tol=1e-7)
        assert elapsed == state['main_seconds'] and logs[-1]['stream_sha256'] == summary['stream']['sha256']
        assert sum(x['AMP_skip'] for x in logs) == state['AMP_skips']-origin_amp
        st = dict(updates_B5=attempted, pairs_B5=128*attempted, main_s_B5=seconds,
                  seconds_per_update=seconds/attempted, updates_per_s=attempted/seconds,
                  AMP_B5=state['AMP_skips']-origin_amp, AMP_cumulative=state['AMP_skips'],
                  nonfinite=0, escalations=0, lr=lr, progress_log_points=len(hits), latest=latest,
                  cooldowns=[], step_logs_available=True,
                  last10000_seconds_per_update=float(np.mean([x['seconds'] for x in logs[-10000:]])),
                  sampled_attention_abs_logit_max=max((x['sampled_attention_abs_logit_max'] for x in logs if x.get('sampled_attention_abs_logit_max') is not None), default=None))
        for tag in summary['B5_points']:
            p = root/'points'/tag
            t, m, c, ck = [read(p/f) for f in ['training.json', 'evaluation/metrics.json', 'complete.json', 'checkpoint.json']]
            assert t == c['training'] and m == c['metrics']
            assert c['weights_sha256'] == ck['sha256'] == m['weights_sha256']
            anchor = t['anchor']
            assert t['main_pairs'] == anchor['cursor'] and t['q'] == 0 and t['start_lr'] == lr
            row = dict(model=name, tag=tag, main_s=t['main_seconds'], pairs=128*anchor['updates'],
                       order_cursor=anchor['cursor'], updates=anchor['updates'], cooldown_s=t['cooldown_seconds'],
                       cooldown_updates=t['completed_updates'], weights_sha256=ck['sha256'])
            row = point(row, lambda suffix, p=p: p/suffix)
            cl = lines(p/'cooldown-updates.jsonl')
            assert len(cl) == t['completed_updates']
            elapsed = 0.
            for j, x in enumerate(cl, 1):
                assert x['branch_update'] == j and math.isclose(x['lr'], lr*(1-min(1., elapsed/t['cooldown_target_seconds'])), abs_tol=1e-15)
                assert not x['OOM_retries'] and x['nonfinite_skip'] == 0
                assert math.isfinite(x['CE']) and (x['AMP_skip'] or math.isfinite(x['gradient_norm']))
                elapsed += x['seconds']
                assert elapsed == x['cooldown_seconds']
            assert elapsed == t['cooldown_seconds'] and cl[-1]['lr'] == t['endpoint_lr']
            st['cooldowns'].append(dict(tag=tag, seconds=t['cooldown_seconds'], updates=t['completed_updates'],
                                        seconds_per_update=t['cooldown_seconds']/t['completed_updates'], endpoint_lr=t['endpoint_lr'],
                                        AMP=sum(x['AMP_skip'] for x in cl), nonfinite=0, OOM=0))
            if t['final']:
                k = m['K8_tune2048']
                assert k['complete'] and k['rows'] == k['target_rows'] == 2048 and k['K'] == 8
                assert k['seed'] == 270042 and k['T'] == .7 and not k['OOM_attempts']
                assert k['weights_sha256'] == ck['sha256'] and m['test']['rows'] == 1919
                row['K8'], row['test'] = k, m['test']
                finals['W-B5' if gpu == 0 else 'C-long2x-B5'] = p
        stability[name] = st
    return dict(session=session, producer_checked=checked, producer_missing=missing, points=points, stability=stability), finals


def comparisons_and_curves(result):
    points, comparisons = result['points'], {}
    def compare(a, b, label):
        assert (a['cooldown_s'] < 1000) == (b['cooldown_s'] < 1000)
        for split in SPLITS:
            aa, bb = [RECORDS[p['model'], p['tag'], split] for p in [a, b]]
            groups = {'all': list(range(len(aa)))}
            if split == 'heldout4k':
                groups.update(tune2048=IDS, eval2048=[i for i in range(4096) if i not in set(IDS)])
                groups.update({f'bin{lo}-{hi}': [i for i, x in enumerate(aa) if lo <= x['original_bpe_length'] < hi]
                               for lo, hi in [(0, 60), (60, 90), (90, 120), (120, math.inf)]})
            for group, ids in groups.items():
                comparisons[f'{label}/{split}/{group}'] = paired([aa[i]['correct'] for i in ids], [bb[i]['correct'] for i in ids])
    for name in ['W600', 'C-long2x']:
        for short in [True, False]:
            seq = sorted([p for p in points if p['model'] in [name]+(['C (shared L1)'] if name == 'C-long2x' else []) and
                          (p['cooldown_s'] < 1000) == short], key=lambda p: p['main_s'])
            for a, b in zip(seq, seq[1:]):
                compare(b, a, f'{name} {b["tag"]} minus {a["tag"]}')
    fw, fc = [next(p for p in points if p['model'] == n and p['tag'].startswith('b5-final')) for n in ['W600', 'C-long2x']]
    compare(fw, fc, 'W minus C B5 final50')
    result['comparisons'] = comparisons
    # Same cooldown, log interpolation only inside observed endpoints. No CI.
    estimates = []
    for p in points:
        if p['model'] not in ['W600', 'C-long2x']:
            continue
        other = 'W600' if p['model'] == 'C-long2x' else 'C-long2x'
        seq = [q for q in points if q['model'] == other and (q['cooldown_s'] < 1000) == (p['cooldown_s'] < 1000)]
        if other == 'C-long2x' and p['cooldown_s'] < 1000:
            seq += [q for q in points if q['model'] == 'C (shared L1)']
        for axis in ['pairs', 'main_s']:
            est = A.interpolate(seq, p[axis], axis)
            if est:
                estimates.append(dict(model=p['model'], tag=p['tag'], control=other, axis=axis,
                                      estimate=est, delta_pp=p['scores']['heldout4k']['accuracy_pp']-est['estimate_pp']))
    result['interpolations'] = estimates
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for ax, axis, div, title in [(axes[0, 0], 'pairs', 1e6, 'Main pairs, million'), (axes[0, 1], 'main_s', 3600, 'Main update time, hours')]:
        for name, color in [('W600', '#7744aa'), ('C-long2x', '#208667'), ('C-main', '#2470a0')]:
            for short, style in [(True, '-o'), (False, ':D')]:
                seq = sorted([p for p in points if p['model'] in [name]+(['C (shared L1)'] if name.startswith('C') else []) and
                              (p['cooldown_s'] < 1000) == short], key=lambda p: p[axis])
                ax.plot([p[axis]/div for p in seq], [p['scores']['heldout4k']['accuracy_pp'] for p in seq], style,
                        color=color, label=name+(' / CD15' if short else ' / CD50'))
        ax.set(xlabel=title, ylabel='Greedy heldout4096, %'); ax.grid(alpha=.2)
    axes[0, 1].legend(fontsize=7)
    for name, color in [('W600', '#7744aa'), ('C-long2x', '#208667')]:
        for short, style in [(True, '-o'), (False, ':D')]:
            seq = sorted([p for p in points if p['model'] in [name]+(['C (shared L1)'] if name == 'C-long2x' else []) and
                          (p['cooldown_s'] < 1000) == short], key=lambda p: p['main_s'])
            for ax, split in [(axes[1, 0], 'old991'), (axes[1, 1], 'heldout4k')]:
                vals = [p['scores'][split]['accuracy_pp'] if split == 'old991' else p['CE'][split] for p in seq]
                ax.plot([p['main_s']/3600 for p in seq], vals, style, color=color)
    axes[1, 0].set(xlabel='Main update time, hours', ylabel='Greedy old991, %')
    axes[1, 1].set(xlabel='Main update time, hours', ylabel='Heldout compact token CE')
    for ax in axes[1]: ax.grid(alpha=.2)
    fig.savefig(WORK/'results/b5-learning-curve.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
    for j, (ax, title) in enumerate(zip(axes.flat, ['<60 BPE (n=2013)', '60-90 BPE (n=1457)', '90-120 BPE (n=488)', '>=120 BPE (n=138)'])):
        for name, color in [('W600', '#7744aa'), ('C-long2x', '#208667')]:
            for short, style in [(True, '-o'), (False, ':D')]:
                seq = sorted([p for p in points if p['model'] in [name]+(['C (shared L1)'] if name == 'C-long2x' else []) and
                              (p['cooldown_s'] < 1000) == short], key=lambda p: p['main_s'])
                ax.plot([p['main_s']/3600 for p in seq], [p['bins'][j]['accuracy_pp'] for p in seq], style,
                        color=color, label=name+(' / CD15' if short else ' / CD50'))
        ax.set(title=title, xlabel='Main update time, hours', ylabel='Greedy accuracy, %'); ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(WORK/'results/b5-length-curves.png', dpi=180); plt.close(fig)


class FastReplay(B.Replay):
    """Reuse the K1=1 early-stop result within each family; same vote function."""
    def __init__(self, journals, anchor, names):
        super().__init__(journals, anchor, names)
        first = self.mask([1]*len(names))
        self.initial = np.zeros(len(IDS), dtype=bool)
        self.pending = []
        for i, row in enumerate(self.rows):
            if first & row[2] == first and first & row[1][0] == first:
                self.initial[i] = B.correct(row[3], self.vote(row, first)['execution'])
            else:
                self.pending.append((i, row))
        self.stopped = len(IDS)-len(self.pending)

    def bits(self, ks):
        mask, bits = self.mask(ks), self.initial.copy()
        for i, row in self.pending:
            bits[i] = B.correct(row[3], self.vote(row, mask)['execution'])
        return bits


def native_replay(sim, configurations, library):
    """Batch the same bitset vote in C++; Python/production validate winners."""
    n, c = len(sim.rows), len(sim.rows[0][0])
    adjacency = np.asarray([r[1] for r in sim.rows], dtype=np.uint64)
    valid = np.asarray([r[2] for r in sim.rows], dtype=np.uint64)
    lp = np.asarray([[x.get('mean_logprob') or 0. for x in r[0]] for r in sim.rows], dtype=np.float64)
    correct = np.asarray([[B.correct(r[3], x['execution']) for x in r[0]] for r in sim.rows], dtype=np.uint8)
    masks = np.asarray([sim.mask(ks) for ks in configurations], dtype=np.uint64)
    output = np.zeros((len(configurations), n), dtype=np.uint8)
    stopped = ctypes.c_int()
    arrays = [adjacency, valid, lp, correct, masks]
    library.replay_many(n, c, len(configurations), *[ctypes.c_void_p(x.ctypes.data) for x in arrays],
                        ctypes.c_uint64(sim.mask([1]*len(sim.names))), ctypes.c_void_p(output.ctypes.data), ctypes.byref(stopped))
    assert stopped.value == sim.stopped
    # Compare full per-task bits for several masks in every family, including
    # sparse K1, maximum total K and arbitrary prefixes/bridge removal.
    for j in sorted(set([0, len(configurations)//2, len(configurations)-1])):
        assert np.array_equal(output[j], sim.bits(configurations[j]))
    return output


def native_fixtures(library):
    # Nontransitive isclose: removing a bridge must split the component, while
    # duplicate votes remain distinct. Also exercise an empty valid pool.
    values = [100., 100.00009, 100.00018, 100.00018]
    adjacency = np.asarray([[sum(1 << k for k, v in enumerate(values) if B.same(x, v)) for x in values]], dtype=np.uint64)
    valid = np.asarray([15], dtype=np.uint64)
    lp = np.asarray([[0., -1., -2., -3.]], dtype=np.float64)
    correct = np.asarray([[0, 1, 1, 1]], dtype=np.uint8)
    masks = np.asarray([15, 13, 0], dtype=np.uint64)
    output = np.zeros((3, 1), dtype=np.uint8)
    stopped = ctypes.c_int()
    library.replay_many(1, 4, 3, *[ctypes.c_void_p(x.ctypes.data) for x in [adjacency, valid, lp, correct, masks]],
                        ctypes.c_uint64(16), ctypes.c_void_p(output.ctypes.data), ctypes.byref(stopped))
    assert output[:, 0].tolist() == [0, 1, 0] and stopped.value == 0
    return ['native bridge removal', 'native duplicate multiplicity', 'native empty-pool greedy fallback']


@lru_cache(maxsize=None)
def base_timer(n, maximum_extra, stopped):
    return B.timing_estimate([dict(name=str(i), K1=1, K2=maximum_extra+1 if i == 0 else 1) for i in range(n)], stopped)


def timing(names, anchor, ks, stopped, ratios):
    # Algebraically identical to B3/B4's timer; cache only the repeated inputs.
    base = base_timer(len(names), max(ks)-1, stopped)
    runs = {}
    for label, t in base['runs'].items():
        old = B.read(RUNS/f'nlp-hw02-kaggle-l3/final/inference-v5/final-inference-l3-v5/{label}/timing.json')
        p = old['phases_seconds']
        mean = sum(p[f'stage1_{n}_sampling_restore_submit'] for n in ['main', 'long'])/sum(b['stage'] == 1 for b in old['generation_batches'])/base['sample_speedup']
        wave = 1024//(max(ks)-1) if max(ks) > 1 else 0
        first = math.ceil(t['rows']/1024)*sum(ratios[n]['sample'] for n in names)*mean
        second = math.ceil(t['escalated_rows_proxy']/wave)*sum(ratios[n]['sample'] for n, k in zip(names, ks) if k > 1) if wave else 0
        greedy = p['greedy_generation_restore_submit']/1.6565*ratios[anchor]['greedy']
        fixed = t['fixed_seconds']+t['extra_model_load_warmup_reserve_seconds']+2+4*sum(n.startswith('W-') for n in names)
        variable = greedy+first+second*base['worst_old_batch_seconds']*1.10/base['sample_speedup']
        runs[label] = dict(nominal_s=fixed+greedy+first+second*mean, conservative_s=fixed+variable,
                           stress15_s=fixed+1.15*variable, escalated_rows=t['escalated_rows_proxy'], stage2_wave_questions=wave)
    return dict(runs=runs, conservative_s=max(t['conservative_s'] for t in runs.values()), stress15_s=max(t['stress15_s'] for t in runs.values()))


def ensembles(finals, library):
    journals, receipts = B.load_all()
    finals['W-B4'] = RUNS/'nlp-hw02-b4/1/model-GPU0/points/b4-final-091801'
    for name, root in finals.items():
        files = sorted((root/'evaluation/k8-tune2048').glob('batch-*.json'))
        assert len(files) == 64
        rr = [r for f in files for r in read(f)]
        assert len(rr) == 2048 and sorted(r['id'] for r in rr) == IDS
        assert sum(r['correct'] for r in rr) == read(root/'evaluation/metrics.json')['K8_tune2048']['matches']
        journals[name] = {r['id']: r for r in rr}
    for i in IDS:
        for name in NAMES:
            r, ref = journals[name][i], journals['C-main'][i]
            assert r['question_sha256'] == ref['question_sha256'] and r['reference'] == ref['reference']
            assert len(r['samples']['0.7']) == 8
            win = B.choose(r, dict(mode='vote', temperature=.7, k=8, greedy_vote=False))
            assert B.correct(r['reference'], win['execution']) == r['correct']
    ratios = {n: dict(sample=1., greedy=1.) for n in NAMES}
    for n in ['W-B4', 'W-B5']:
        ratios[n] = read(WORK/'results/b3-results.json')['ensemble']['ratios']['W-lr600']
    baselines = {}
    baseline_costs = {}
    parity = 0
    for key, names, anchor, ks, expected in [('v7', ['C-main', 'C-long2x', 'pilot-b'], 'C-main', [3, 5, 4], 1535),
                                            ('v8', ['C-long2x', 'pilot-b', 'W-B4'], 'C-long2x', [1, 8, 1], 1541)]:
        sim = B.Replay(journals, anchor, names)
        bits, stop, _ = sim.evaluate(ks, [1]*3)
        assert sum(bits) == expected
        parity += sim.validate(ks, [1]*3)
        baselines[key] = np.asarray(bits, dtype=bool)
        baseline_costs[key] = sum((1+(2048-stop)/2048*(k-1))*ratios[n]['sample'] for n, k in zip(names, ks))
    def summarize(names, anchor, ks, bits, stopped, boot=False):
        prob = (2048-stopped)/2048
        means = [1+prob*(k-1) for k in ks]
        return dict(models=list(names), anchor=anchor, K1=[1]*len(names), K2=list(ks), matches=int(np.sum(bits)),
                    accuracy_pp=100*float(np.mean(bits)), stopped=stopped, max_samples=sum(ks), mean_samples=sum(means),
                    mean_C_sample_units=sum(v*ratios[n]['sample'] for n, v in zip(names, means)),
                    timing=timing(tuple(names), anchor, tuple(ks), stopped, ratios),
                    **{f'paired_vs_{key}': paired(bits, val, boot) for key, val in baselines.items()})
    grid, static = [], []
    for name in NAMES:
        sim = B.Replay(journals, name, [name])
        for greedy in [False, True]:
            bits, _, _ = sim.evaluate([8], greedy=greedy)
            static.append(dict(model=name, K=8, greedy_vote=greedy, matches=sum(bits),
                               **{f'paired_vs_{key}': paired(bits, val) for key, val in baselines.items()}))
    for count in [2, 3, 4]:
        for family in itertools.combinations(NAMES, count):
            # Match B4's canonical first-C anchor, plus both anchors for pairs.
            for anchor in (family if count == 2 else family[:1]):
                names = (anchor,)+tuple(n for n in family if n != anchor)
                sim = FastReplay(journals, anchor, list(names))
                configurations = [ks for ks in itertools.product(range(1, 9), repeat=count) if sum(ks) <= 14]
                bitrows = native_replay(sim, configurations, library)
                for ks, bits in zip(configurations, bitrows):
                    grid.append(summarize(names, anchor, ks, bits, sim.stopped))
            print('replayed', family, 'configs', len(grid), flush=True)
    eligible = [r for r in grid if r['timing']['conservative_s'] <= 140]
    selectors = {
        'all_140': lambda r: True,
        'stress140': lambda r: r['timing']['stress15_s'] <= 140,
        'at_v8_mean_cost': lambda r: r['mean_C_sample_units'] <= baseline_costs['v8'] and r['max_samples'] <= 12,
        'at_v7_mean_cost': lambda r: r['mean_C_sample_units'] <= 6.5859375 and r['max_samples'] <= 12,
        'new_only_replacements': lambda r: 'C-long2x' not in r['models'] and 'W-B4' not in r['models'],
        'both_new': lambda r: all(n in r['models'] for n in ['C-long2x-B5', 'W-B5']) and all(n not in r['models'] for n in ['C-long2x', 'W-B4']),
        'contains_new': lambda r: any(n in r['models'] for n in ['C-long2x-B5', 'W-B5'])}
    best = {key: max(filter(pred, eligible), key=lambda r: (r['matches'], -r['timing']['conservative_s'], -r['mean_C_sample_units']))
            for key, pred in selectors.items()}
    replacements = []
    for c, w in itertools.product(['C-long2x', 'C-long2x-B5'], ['W-B4', 'W-B5']):
        names, ks = [c, 'pilot-b', w], [1, 8, 1]
        sim = FastReplay(journals, c, names)
        replacements.append(summarize(names, c, ks, sim.bits(ks), sim.stopped, True))
    # Verify every distinct reported best/replacement, including timing parity.
    selected = {json.dumps([r['models'], r['anchor'], r['K2']]): r for r in list(best.values())+replacements}
    for r in selected.values():
        sim = FastReplay(journals, r['anchor'], r['models'])
        bits, stops, _ = sim.evaluate(r['K2'], r['K1'])
        assert np.array_equal(bits, sim.bits(r['K2'])) and stops == sim.stopped and sum(bits) == r['matches']
        parity += sim.validate(r['K2'], r['K1'])
        assert r['timing'] == B.timing(r['models'], r['anchor'], r['K2'], r['stopped'], ratios)
        for key, base in baselines.items(): r[f'paired_vs_{key}'] = paired(bits, base)
    path = WORK/'results/b5-ensemble-grid.csv'
    with path.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['models', 'anchor', 'K2', 'matches', 'stopped', 'max_samples', 'mean_samples', 'mean_C_units',
                         'delta_v8_pp', 'normal_v8_lo', 'normal_v8_hi', 'gained_v8', 'lost_v8',
                         'delta_v7_pp', 'normal_v7_lo', 'normal_v7_hi', 'gained_v7', 'lost_v7', 'conservative_s', 'stress15_s'])
        for r in grid:
            a, b = r['paired_vs_v8'], r['paired_vs_v7']
            writer.writerow(['+'.join(r['models']), r['anchor'], '/'.join(map(str, r['K2'])), r['matches'], r['stopped'],
                             r['max_samples'], r['mean_samples'], r['mean_C_sample_units'],
                             a['delta_pp'], *a['CI95_pp'], a['gained'], a['lost'], b['delta_pp'], *b['CI95_pp'], b['gained'], b['lost'],
                             r['timing']['conservative_s'], r['timing']['stress15_s']])
    return dict(best=best, v8_replacements=replacements, single_K8=static, ratios=ratios, journal_sha256=receipts,
                grid_configurations=len(grid), eligible_140=len(eligible), grid_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                production_independent_parity_rows=parity, graph_fixtures=B.graph_fixtures(),
                native_fixture_checks=native_fixtures(library), native_python_bit_comparisons=65*3*2048,
                baselines=dict(v7=1535, v8=1541, GPU_v8_fresh_samples_not_comparator=1532),
                baseline_costs=baseline_costs,
                search='All 2-4 model subsets of six, canonical first-C anchor, both pair anchors; K1=1, K2=1..8, sum<=14; no K tuning on fresh',
                CI='normal in full grid; selected comparisons also 20000 paired multinomial bootstrap seed2026101901; pointwise, no search correction')


def b6_plan(result):
    out = {}
    for name, st in result['stability'].items():
        final = st['cooldowns'][-1]
        # 3h40 update-time target; round N down to whole hundreds, reserve 40m
        # fixed plus 10% update overhead inside a 5h total cap.
        update_s = max(st['seconds_per_update'], final['seconds_per_update'])
        n = 100*math.floor(13200/update_s/100)
        out[name] = dict(N=n, update_seconds_reference=update_s, cooldown_pairs=128*n,
                         cooldown_update_s=n*update_s, total_with_10pct_and_2400s_reserve=n*update_s*1.10+2400,
                         anchor=st['latest'], peak_lr=st['lr'], q=0,
                         schedule='lr(j)=peak*(1-sqrt(j/N)), j=1..N; final attempted update lr=0; no warmup/main',
                         fallback='Keep existing B5 final50 and v8; do not relabel interrupted cooldown as final')
    result['B6_proposal'] = out


def main():
    result, finals = audit()
    print('audited', len(result['producer_checked']), 'producer files', flush=True)
    comparisons_and_curves(result)
    b6_plan(result)
    # Write the audit first so failures in a long exploratory replay lose no work.
    B.write(WORK/'results/b5-results.json', result)
    with tempfile.TemporaryDirectory(prefix='nlp-b5-replay-') as tmp:
        library_path = Path(tmp)/'replay.dylib'
        subprocess.run(['c++', '-O3', '-shared', '-fPIC', '-std=c++17', str(HERE/'b5-replay.cpp'), '-o', str(library_path)], check=True)
        library = ctypes.CDLL(str(library_path))
        library.replay_many.restype = None
        result['ensemble'] = ensembles(finals, library)
    result.update(status='complete', GPU=False, API=False, new_forward=False, new_execution=False, new_sampling=False,
                  provenance=dict(kernel='friend1/nlp-hw02-b5', version_from_user=6, start_UTC_from_user='2026-10-18 18:14',
                                  local_directory='results/kaggle-runs/nlp-hw02-b5/1',
                                  version_limit='local label 1 is a download name; v6/start/versions1-5 cancelled supplied by user, no version-specific API check'),
                  limits=['Per-update logs audited; recorded stream hashes are checked for endpoint consistency, orders/full states not locally replayed',
                          'No full/FP16 tensors downloaded: no tensor finiteness or actual CUDA restore check',
                          'Same cooldown duration does not equate main time, tokens or cooldown updates; interpolation has no CI',
                          'Repeated heldout/tune; search-selected pointwise CI; no independent quality gate or new cold timer',
                          'Timing same B4 proxy, not a measured upper bound; two resident W models especially require a memory gate'])
    result['inputs_sha256'] = {**INPUTS, **B.INPUTS}
    for p in [Path(__file__), HERE/'b5-replay.cpp', HERE/'analyze-b4-results.py', HERE/'analyze-b3-results.py']:
        result['inputs_sha256'][str(p.relative_to(WORK))] = hashlib.sha256(p.read_bytes()).hexdigest()
    B.write(WORK/'results/b5-results.json', result)
    B.write(WORK/'checks/b5-results-cpu.json', dict(passed=True, producer_files=len(result['producer_checked']),
            producer_missing=len(result['producer_missing']), K8_journals=384, K8_samples=98304,
            replay_configurations=result['ensemble']['grid_configurations'],
            parity_rows=result['ensemble']['production_independent_parity_rows'], no_GPU=True, no_API=True,
            no_forward=True, no_new_execution=True, no_sampling=True, per_update_logs_audited=True,
            native_python_bit_comparisons=result['ensemble']['native_python_bit_comparisons'],
            native_fixture_checks=result['ensemble']['native_fixture_checks'],
            main_updates=sum(s['updates_B5'] for s in result['stability'].values()),
            cooldown_updates=sum(c['updates'] for s in result['stability'].values() for c in s['cooldowns'])))
    print(json.dumps(result['ensemble']['best'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
