"""CPU FP32 train/heldout token CE and teacher-forced argmax accuracy.

Read-only inputs, no execution/generation/training. Run from any directory.
Uniform original train rows, fixed seed; identical examples for every model.
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'kaggle/src'))
import exp26_core as C
import s3_representation as R
from exp26_order import collate
from kaggle_io import sha
from l1_cache import CompactCache

SEED = 2026101503
CPU = torch.device('cpu')
INPUTS = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
CACHE = WORK / 'results/kaggle-runs/nlp-hw02-compact-cache-verified/2/compact-cache'
RESULT = WORK / 'results/overfit-check.json'
ROWS = WORK / 'results/overfit-check-rows.npz'
MODELS = {
    'L3-long': 'nlp-hw02-kaggle-l3/final/model-C-long2x__weights__l2-62280-fp16.pt',
    'L3-main': 'nlp-hw02-kaggle-l3/final/model-C-main__weights__l2-62280-fp16.pt',
    'P1-A': 'nlp-hw02-kaggle-p1/final/model-pilot-a__weights__long-fp16.pt',
    'P1-B': 'nlp-hw02-kaggle-p1/final/model-pilot-b__weights__long-truncation-fp16.pt',
    'R1-control': 'nlp-hw02-kaggle-r1/1/model-GPU1/weights/final-fp16.pt',
    'R1-deep': 'nlp-hw02-kaggle-r1/1/model-GPU0/weights/final-fp16.pt',
}


def write_result(report, arrays):
    tmp = RESULT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(RESULT)
    with ROWS.with_suffix('.npz.tmp').open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    ROWS.with_suffix('.npz.tmp').replace(ROWS)


def hash_array(strings):
    return np.asarray([hashlib.sha256(q.encode()).digest() for q in strings], dtype='S32').view('V32')


def prepare():
    started = time.monotonic()
    held = json.loads((INPUTS / 'heldout4k.json').read_text())
    groups = json.loads((INPUTS / 'protected-questions.json').read_text())['groups']
    assert [r['question'] for r in held] == groups['heldout4k'] and len(held) == 4096
    # Fresh/test strings only enter exclusion hashes; their labels/metrics are never read.
    protected = hash_array(set(q for values in groups.values() for q in values))
    heldhash = hash_array(r['question'] for r in held)
    cache = CompactCache(CACHE, verify=True)
    forbidden, duplicates = [], []
    for k, part in enumerate(cache.parts):
        hashes = np.asarray(part['question-sha256.npy']).copy().view('V32').reshape(-1)
        forbidden.extend((np.flatnonzero(np.isin(hashes, protected)) + cache.starts[k]).tolist())
        duplicates.extend((np.flatnonzero(np.isin(hashes, heldhash)) + cache.starts[k]).tolist())
    eligible = np.ones(len(cache), dtype=bool)
    eligible[forbidden] = False
    rng = np.random.default_rng(SEED)
    ids = rng.choice(np.flatnonzero(eligible), 4096, replace=False)
    dupids = rng.choice(duplicates, min(4096, len(duplicates)), replace=False).astype(np.int64)
    train = [(np.array(cache[int(i)][0], dtype=np.int64), np.array(cache[int(i)][1], dtype=np.int64)) for i in ids]
    tok, _ = R.tokenizer(INPUTS / 'tokenizer.json', 30)
    assert sha(INPUTS / 'tokenizer.json') == cache.manifest['original_tokenizer_sha256']
    assert sha(CACHE / 'tokenizer.json') == cache.manifest['tokenizer']['sha256']
    encoded = [([C.BOS, *C.encode(tok, r['question']), C.EOS], C.encode(tok, R.compact(r['code']))) for r in held]
    # Verify actual source token hashes for every sampled row (not only cached hash receipts).
    sampled_hashes = []
    protected_bytes = {v.tobytes() for v in protected}
    _, table = R.tokenizer(INPUTS / 'tokenizer.json', 30)
    for i, (source, _) in zip(ids, train):
        q = C.decode_bytes(table, [int(v) for v in source if v >= 3]).decode('utf8')
        digest = hashlib.sha256(q.encode()).digest()
        k = int(np.searchsorted(cache.starts, i, side='right') - 1)
        j = int(i - cache.starts[k])
        assert digest == cache.parts[k]['question-sha256.npy'][j].tobytes()
        assert digest not in protected_bytes
        sampled_hashes.append(digest.hex())
    sets = {'train4096': train, 'heldout4k': encoded}
    if len(dupids):
        sets['heldout-question-train-duplicates'] = [cache[int(i)] for i in dupids]
    meta = dict(seed=SEED, train_rows=len(cache), sampled_rows=len(ids), train_ids=ids.tolist(),
                train_ids_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
                train_questions_sha256=sampled_hashes, protected_overlap_rows=len(forbidden),
                heldout_question_train_duplicate_rows=len(duplicates), duplicate_sample_rows=len(dupids),
                duplicate_ids=dupids.tolist(), duplicate_definition='exact UTF-8 question SHA256',
                cache_manifest_sha256=cache.manifest_sha, cache_all_file_SHA_verified=True,
                heldout_sha256=sha(INPUTS / 'heldout4k.json'),
                protected_sha256=sha(INPUTS / 'protected-questions.json'),
                tokenizer_sha256=sha(CACHE / 'tokenizer.json'),
                source_check='decoded sampled source tokens match cached question SHA256; no protected overlap',
                selection='uniform original row identities without replacement; no long weighting or loss selection',
                seconds=time.monotonic()-started)
    print('DATA', {k: v for k, v in meta.items() if k in ('train_rows', 'protected_overlap_rows', 'heldout_question_train_duplicate_rows', 'seconds')}, flush=True)
    return sets, meta


def load(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['format'] == 'nlp-hw02-model-only-v1' and payload['optimizer_included'] is False
    assert all(t.dtype == torch.float16 for t in payload['model'].values())
    model = C.SimpleTransformer(**payload['model_spec'], dropout=0.).float().eval()
    model.load_state_dict(payload['model'], strict=True)
    assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in model.parameters())
    identity = payload['identity']
    recipe = identity['recipe']
    meta = dict(weights_sha256=sha(path), model_spec=payload['model_spec'],
                parameters=sum(p.numel() for p in model.parameters()),
                point=payload['point'], dropout=recipe['dropout'], weight_decay=recipe['weight_decay'],
                r1=recipe.get('r1'), cache_manifest_sha256=identity['cache_manifest_sha256'])
    return model, meta


@torch.inference_mode()
def batch_stats(model, pairs, indices):
    source, target, labels, counts = collate(pairs, indices, CPU)
    logits = model(source, target).float()
    losses = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=C.PAD, reduction='none')
    mask = labels.ne(C.PAD)
    pred = logits.argmax(-1)
    sums = losses.sum(1, dtype=torch.float64)
    tokens = mask.sum(1)
    correct = ((pred == labels) & mask).sum(1)
    assert int(tokens.sum()) == counts['target_tokens'] and bool(torch.isfinite(sums).all())
    illegal = int(((pred == C.BOS) | (pred == C.PAD))[mask].sum())
    return np.column_stack((sums.numpy(), tokens.numpy(), correct.numpy())), illegal


def aggregate(values, lengths):
    sums, counts, correct = values.T
    out = dict(rows=len(values), CE=float(sums.sum()/counts.sum()), tokens=int(counts.sum()),
               token_accuracy=float(correct.sum()/counts.sum()), correct_tokens=int(correct.sum()))
    out['lengths'] = {key: np.quantile(v, [0, .25, .5, .75, .9, .99, 1]).tolist() for key, v in lengths.items()}
    out['compact_target_bins'] = {}
    for label, lo, hi in [('<53', 0, 53), ('53-79', 53, 79), ('79-105', 79, 105), ('>=105', 105, 100000)]:
        mask = (lengths['target'] >= lo) & (lengths['target'] < hi)
        out['compact_target_bins'][label] = dict(rows=int(mask.sum()), tokens=int(counts[mask].sum()),
            CE=float(sums[mask].sum()/counts[mask].sum()),
            token_accuracy=float(correct[mask].sum()/counts[mask].sum()))
    return out


@torch.inference_mode()
def evaluate(model, pairs, deadline, batch, label):
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0])+len(pairs[i][1]))
    values = np.zeros((len(pairs), 3), dtype=np.float64)
    started = last = time.monotonic()
    illegal = 0
    for lo in range(0, len(order), batch):
        if time.monotonic() > deadline:
            raise TimeoutError('CPU diagnosis budget exhausted; completed splits already saved')
        indices = order[lo:lo+batch]
        values[indices], n = batch_stats(model, pairs, indices)
        illegal += n
        if time.monotonic()-last > 40:
            print('PROGRESS', label, lo+len(indices), '/', len(pairs), 'seconds', round(time.monotonic()-started, 1), flush=True)
            last = time.monotonic()
    lengths = {'source': np.asarray([len(s) for s, t in pairs]), 'target': np.asarray([len(t) for s, t in pairs])}
    out = aggregate(values, lengths)
    out.update(seconds=time.monotonic()-started, illegal_argmax_tokens=illegal)
    print('DONE', label, 'CE', out['CE'], 'accuracy', out['token_accuracy'], 'seconds', out['seconds'], flush=True)
    return out, values


def bootstrap_gap(train, held, draws=2000):
    rng = np.random.default_rng(SEED+1)
    gaps = []
    for _ in range(draws):
        a = train[rng.integers(len(train), size=len(train)), :2].sum(0)
        b = held[rng.integers(len(held), size=len(held)), :2].sum(0)
        gaps.append(b[0]/b[1]-a[0]/a[1])
    return np.quantile(gaps, [.025, .975]).tolist()


def summarize(report, arrays, draws=2000):
    """Paired temporal differences on the same examples; no model forwards."""
    report['data']['sample_unique_questions'] = len(set(report['data']['train_questions_sha256']))
    comparisons = [('L2-long', 'L3-long'), ('L3-long', 'P1-A'), ('P1-A', 'P1-B'),
                   ('L3-long', 'R1-control'), ('L3-long', 'R1-deep'), ('R1-control', 'R1-deep')]
    report['paired_comparisons'] = {}
    for before, after in comparisons:
        if any(name+'__'+split not in arrays for name in (before, after) for split in ('train4096', 'heldout4k')):
            continue
        rng = np.random.default_rng(SEED+2)
        deltas, samples = {}, {}
        for split in ('train4096', 'heldout4k'):
            a, b = arrays[before+'__'+split], arrays[after+'__'+split]
            assert np.array_equal(a[:, 1], b[:, 1])
            delta = float((b[:, 0]-a[:, 0]).sum()/a[:, 1].sum())
            boots = []
            for _ in range(draws):
                ids = rng.integers(len(a), size=len(a))
                boots.append(float((b[ids, 0]-a[ids, 0]).sum()/a[ids, 1].sum()))
            samples[split] = np.asarray(boots)
            deltas[split] = dict(CE_delta=delta, CI95=np.quantile(boots, [.025, .975]).tolist())
        gap_samples = samples['heldout4k']-samples['train4096']
        deltas['gap_delta'] = dict(value=deltas['heldout4k']['CE_delta']-deltas['train4096']['CE_delta'],
                                  CI95=np.quantile(gap_samples, [.025, .975]).tolist())
        report['paired_comparisons'][before+' -> '+after] = deltas
    report['bootstrap'] = dict(draws=draws, seed_gap=SEED+1, seed_paired=SEED+2,
        unit='question row; independent train/heldout resampling, paired model differences',
        limitations='conditional on checkpoints/split; no seed, training, or repeated-selection uncertainty')
    metrics = {
        'L3-long': 'nlp-hw02-kaggle-l3/final/model-C-long2x__points__l2-62280__evaluation__metrics.json',
        'L3-main': 'nlp-hw02-kaggle-l3/final/model-C-main__points__l2-62280__evaluation__metrics.json',
        'P1-A': 'nlp-hw02-kaggle-p1/final/model-pilot-a__points__long__evaluation__metrics.json',
        'P1-B': 'nlp-hw02-kaggle-p1/final/model-pilot-b__points__long-truncation__evaluation__metrics.json',
        'R1-control': 'nlp-hw02-kaggle-r1/1/model-GPU1/summary.json',
        'R1-deep': 'nlp-hw02-kaggle-r1/1/model-GPU0/summary.json',
    }
    report['GPU_saved_metric_checks'] = {}
    for name, relative in metrics.items():
        path = WORK / 'results/kaggle-runs' / relative
        if name not in report['models'] or 'heldout4k' not in report['models'][name].get('splits', {}) or not path.exists():
            continue
        saved = json.loads(path.read_text())
        if name.startswith('R1-'):
            saved = saved['metrics']
        model = report['models'][name]
        assert saved['weights_sha256'] == model['weights_sha256']
        gpu, cpu = saved['splits']['heldout4k']['CE'], model['splits']['heldout4k']
        assert gpu['tokens'] == cpu['tokens'] == 215035
        assert abs(gpu['CE']-cpu['CE']) < .001
        report['GPU_saved_metric_checks'][name] = dict(file=relative, sha256=sha(path), weights_SHA_match=True,
            tokens=gpu['tokens'], saved_GPU_CE=gpu['CE'], CPU_minus_GPU_CE=cpu['CE']-gpu['CE'])
    report['script_sha256'] = sha(Path(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-seconds', type=int, default=3000)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    if args.summarize_only:
        report = json.loads(RESULT.read_text())
        with np.load(ROWS, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        summarize(report, arrays)
        write_result(report, arrays)
        return
    started = time.monotonic()
    deadline = started+args.max_seconds
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    sets, data = prepare()
    report = dict(format='nlp-hw02-overfit-check-v1', data=data, models={}, benchmark=[],
                  protocol=dict(device='cpu', parameters='FP16 checkpoint -> FP32 strict load',
                      forward='FP32, eval(), inference_mode, no autocast',
                      CE='sum unsmoothed FP32 token NLL / all non-PAD target tokens, including EOS; float64 reduction',
                      accuracy='raw-logit argmax per position under teacher forcing; gold prefix; EOS included, PAD ignored',
                      representation=30, batch=args.batch, torch=torch.__version__, execution_calls=0))
    arrays = {}
    paths = {k: WORK / 'results/kaggle-runs' / v for k, v in MODELS.items()}
    trash = Path.home() / '.Trash/cleanup-2026-10-14'
    try:
        early = sorted(trash.rglob('model-C-long2x__weights__l2-28800-fp16.pt'))
        for i, p in enumerate(early[:1]):
            paths = {f'L2-long': p, **paths}
        report['L2_search'] = dict(available=bool(early), reason='found read-only checkpoint' if early else 'specified cleanup directory absent, inaccessible, or no matching FP16 checkpoint')
    except OSError as error:
        report['L2_search'] = dict(available=False, reason=type(error).__name__)
    for name, path in paths.items():
        if not path.exists():
            report['models'][name] = dict(skipped='checkpoint absent')
            continue
        model, meta = load(path)
        if not report['benchmark']:
            pairs = sets['train4096']
            order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0])+len(pairs[i][1]))
            indices = order[len(order)//2:len(order)//2+args.batch]
            choices = [args.threads] if args.threads else [1, 2, 4]
            for threads in choices:
                torch.set_num_threads(threads)
                batch_stats(model, pairs, indices)
                mark = time.monotonic()
                for _ in range(3):
                    batch_stats(model, pairs, indices)
                report['benchmark'].append(dict(threads=threads, batch_seconds=(time.monotonic()-mark)/3))
            best = min(report['benchmark'], key=lambda r: r['batch_seconds'])['threads']
            torch.set_num_threads(best)
            report['protocol']['threads'] = best
            print('BENCHMARK', report['benchmark'], 'selected', best, flush=True)
            # Validate batching/padding against single-example metric computation on real weights.
            chosen = [order[0], order[len(order)//2], order[-1]]
            multi, _ = batch_stats(model, pairs, chosen)
            single = np.concatenate([batch_stats(model, pairs, [i])[0] for i in chosen])
            assert np.allclose(multi[:, 0], single[:, 0], atol=2e-3, rtol=3e-5)
            assert np.array_equal(multi[:, 1:], single[:, 1:])
            report['batch_single_check'] = dict(rows=3, max_abs_NLL_difference=float(np.max(abs(multi[:, 0]-single[:, 0]))),
                                               token_counts_and_argmax_matches=True)
        meta['splits'] = {}
        report['models'][name] = meta
        for split, pairs in sets.items():
            out, values = evaluate(model, pairs, deadline, args.batch, name+'/'+split)
            meta['splits'][split] = out
            arrays[name+'__'+split] = values
            write_result(report, arrays)
        a, b = meta['splits']['train4096'], meta['splits']['heldout4k']
        meta['gap'] = b['CE']-a['CE']
        meta['gap_CI95_row_bootstrap'] = bootstrap_gap(arrays[name+'__train4096'], arrays[name+'__heldout4k'])
        meta['heldout_token_accuracy_minus_train_pp'] = 100*(b['token_accuracy']-a['token_accuracy'])
        bins = b['compact_target_bins']
        meta['train_CE_reweighted_to_heldout_target_bin_token_shares'] = sum(
            a['compact_target_bins'][key]['CE']*v['tokens']/b['tokens'] for key, v in bins.items())
        meta['weights'] = str(path.relative_to(WORK)) if path.is_relative_to(WORK) else 'read-only cleanup L2 checkpoint'
        report['wall_seconds'] = time.monotonic()-started
        write_result(report, arrays)
        del model
        gc.collect()
    report['complete'] = True
    report['wall_seconds'] = time.monotonic()-started
    summarize(report, arrays)
    write_result(report, arrays)
    print('COMPLETE seconds', report['wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
