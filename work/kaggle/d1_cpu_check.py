"""CPU-only parity on real compact rows and both pinned L3 FP16 artifacts."""
import ast
import json
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent; WORK = ROOT.parent


def main():
    nb = json.loads((ROOT/'notebooks/nlp-hw02-kaggle-d1.ipynb').read_text())
    startup = next(c['source'] for c in nb['cells'] if c['cell_type'] == 'code')
    sources = ast.literal_eval(next(n.value for n in ast.parse(startup).body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SOURCES'))
    with tempfile.TemporaryDirectory(prefix='nlp-d1-real-cpu-') as tmp:
        root = Path(tmp)
        for name, source in sources.items(): (root/name).write_text(source)
        sys.path.insert(0, str(root))
        from l1_cache import CompactCache
        import exp26_core as Training
        import final_core as Final
        from exp26_order import collate
        from final_representation import tokenizer
        from d1_loss import row_nll
        from d1_generate import generation_wave
        from d1_data import WEIGHTS, length_bins
        from kaggle_io import sha, write
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        cache = CompactCache(WORK/'results/kaggle-runs/nlp-hw02-compact-cache-verified/2/compact-cache', verify=False)
        bins = length_bins(cache.target_lengths)
        ids = np.concatenate([np.flatnonzero(bins == b)[:4] for b in range(4)]).astype(np.uint32)
        seed_path = ROOT/'staging/d1-live/seed-evidence.npz'
        with np.load(seed_path, allow_pickle=False) as z:
            evidence = {k:z[k] for k in ('row_id', 'reference')}
        # A known cached reference and a singleton exercise reuse and execution.
        singleton = int(np.flatnonzero(~np.isin(np.arange(1000), evidence['row_id']))[0])
        generate_ids = np.asarray([evidence['row_id'][0], singleton], dtype=np.uint32)
        tokpath = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json'
        assert sha(tokpath) == cache.manifest['original_tokenizer_sha256']
        tok, table = tokenizer(tokpath, 30); del tok
        report = dict(passed=False, GPU=False, push=False, actual_cache_rows=len(cache), models={})
        for name, expected_sha in WEIGHTS.items():
            path = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final'/f'model-{name}__weights__l2-62280-fp16.pt'
            assert sha(path) == expected_sha
            p = torch.load(path, map_location='cpu', weights_only=True)
            model = Final.SimpleTransformer(**p['model_spec'], dropout=0.).eval()
            model.load_state_dict(p['model'], strict=True)
            train_model = Training.SimpleTransformer(**p['model_spec'], dropout=0.).eval()
            train_model.load_state_dict(p['model'], strict=True); del p
            started = time.monotonic()
            with torch.inference_mode():
                nn, ll = row_nll(model, cache, ids, torch.device('cpu'))
                src, tgt, labels, counts = collate(cache, ids, torch.device('cpu'))
                logits = train_model(src, tgt)
                ordinary = F.cross_entropy(logits.reshape(-1, train_model.vocab), labels.reshape(-1), ignore_index=0)
                per_row = F.cross_entropy(logits.float().reshape(-1, train_model.vocab), labels.reshape(-1),
                                          ignore_index=0, reduction='none').reshape_as(labels).sum(1)
            np.testing.assert_allclose(nn, per_row.numpy(), rtol=1e-6, atol=1e-5)
            assert abs(float(nn.sum(dtype=np.float64)/ll.sum())-float(ordinary)) < 1e-6
            assert int(ll.sum()) == counts['target_tokens']
            del train_model, logits, src, tgt, labels
            generation = generation_wave(model, cache, generate_ids, table, torch.device('cpu'), root/f'{name}.npz',
                lambda:None, time.monotonic()+240, evidence=evidence, batch=2, cap=512)
            with np.load(root/f'{name}.npz') as z:
                assert np.all(z['generation_status'] == 'eos')
                assert np.all(z['reference_status'] == 'numeric')
                statuses = z['status'].tolist()
                assert 'numeric' in statuses and 'budget_not_evaluated' not in statuses, statuses
                assert generation['execution']['error'] is None
                assert set(z['reference_source']) == {'receipt', 'executed'}
                answers = [float(v) if np.isfinite(v) else None for v in z['answer']]
                references = z['reference'].tolist()
            report['models'][name] = dict(weight_sha256=expected_sha, loss_rows=len(ids),
                row_ids=ids.tolist(), sum_NLL=float(nn.sum(dtype=np.float64)), tokens=int(ll.sum()),
                token_CE=float(ordinary), max_row_NLL_error=float(np.max(np.abs(nn-per_row.numpy()))),
                generation_rows=generate_ids.tolist(), generated_answers=answers, references=references,
                EOS_complete=True, execution_statuses=statuses, cached_and_executed_reference=True, generation=generation,
                seconds=time.monotonic()-started)
            del model
        report['passed'] = True
        write(WORK/'checks/kaggle-d1-real-cpu.json', report)
        print('D1 REAL CPU PASSED:', {m:round(r['seconds'], 2) for m,r in report['models'].items()})


if __name__ == '__main__': main()
