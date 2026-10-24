"""Static V4 assembly checks and real L3 FP16 CPU loads/vote/code reexecution."""
import ast
import gc
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
FINAL = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final'
BUNDLE = ROOT/'bundles/final_inference_l3'


def read(p): return json.loads(p.read_text())


def main():
    ready = FINAL/'next-inference.json'
    ready.write_text(json.dumps({'bundle_path': str(BUNDLE.resolve()), 'ready': False})+'\n')
    nb = read(BUNDLE/'nlp-hw02-final-inference-l3.ipynb')
    nbformat.validate(nbformat.from_dict(nb))
    codes = [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            assert not c['outputs'] and c['execution_count'] is None
            compile(c['source'], 'v4 cell', 'exec')
    sources = ast.literal_eval(next(n.value for n in ast.parse(codes[0]).body if isinstance(n, ast.Assign)
                                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SOURCES'))
    identity = read(ROOT/'final-l3-source-identity.json')
    digest = lambda s: hashlib.sha256(s.encode()).hexdigest()
    for name, source in sources.items():
        compile(source, name, 'exec')
        assert digest(source) == identity['embedded_source_sha256'][name]
        assert digest((ROOT/'src'/name).read_text()) == identity['base_source_sha256'][name]
        if name not in ['final_session.py', 'final_inference.py']:
            assert source == (ROOT/'src'/name).read_text()
    base_worker = (ROOT/'src/final_inference.py').read_text()
    final = read(WORK/'results/l3-final-results.json')
    expected = base_worker.replace('3806a5827004fd88f533a7307abc0c1e387faafc5672f4224392b363c4d2ad3d', final['receipts']['C-main']['FP16']['sha256'])
    expected = expected.replace('6227ee2fc8dd12e186137be229fef2b1ceba659848c35bedd90cec6502f41548', final['receipts']['C-long2x']['FP16']['sha256']).replace('1492', '1511')
    assert sources['final_inference.py'] == expected
    session = sources['final_session.py']
    assert 'for number in range(1, 2):' in session and "launch('tune', rows_path, 2048" in session
    assert 'elapsed >= 660.' in session and 'elapsed >= 690.' in session
    assert "'budget_seconds': 720" in session and "'budget_seconds': 180." in session
    assert 'def find_input' in codes[-1] and "find_input('nlp-hw02-l3-live')" in codes[-1]
    meta, limits = read(BUNDLE/'kernel-metadata.json'), read(BUNDLE/'session-limits.json')
    assert meta['id'] == 'friend1/nlp-hw02-final-inference-l3' and meta['is_private'] and not meta['enable_internet']
    assert meta['enable_gpu'] and not meta['enable_tpu'] and meta['machine_shape'] == 'NvidiaTeslaT4'
    assert meta['dataset_sources'] == ['friend1/nlp-hw02-l3-live', 'thisarivson/nlp-hw02-26b-inputs']
    assert limits['session_timeout_seconds'] == 720 and limits['cold_test_runs'] == limits['tune_runs'] == 1
    access = read(WORK/'checks/kaggle-l4-source-access.json')
    assert access['passed'] and access['datasets']['friend1/nlp-hw02-l3-live']['private']
    manifest = read(FINAL/'recover-manifest.json')
    with tempfile.TemporaryDirectory(prefix='final-l3-cpu-') as scratch:
        scratch = Path(scratch)
        for name, source in sources.items(): (scratch/name).write_text(source)
        sys.path.insert(0, str(scratch))
        import torch
        import final_inference as I
        import final_session as S
        from final_vote import select, verify_winners, write_answers, validate_answers
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        tokenizer_path = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json'
        assert I.sha(tokenizer_path) == S.INPUT_SHA['tokenizer.json']
        tok, table = I.tokenizer(tokenizer_path, 30)
        test = read(WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/test.json')[:2]
        encoded = [[I.C.BOS]+I.C.encode(tok, r['question'])+[I.C.EOS] for r in test]
        src = torch.nn.utils.rnn.pad_sequence([torch.tensor(r) for r in encoded], batch_first=True, padding_value=I.C.PAD)
        weight_checks = {}
        for name in ['C-main', 'C-long2x']:
            filename = f'model-{name}__weights__l2-62280-fp16.pt'
            file = FINAL/filename
            remote = next(r for r in access['files'] if r['name'] == filename)
            entry = next(r for r in manifest['files'] if r['original_path'] == f'model-{name}/weights/l2-62280-fp16.pt')
            assert remote['bytes'] == entry['bytes'] == file.stat().st_size
            assert I.sha(file) == entry['sha256'] == I.WEIGHT_SHA[name]
            p = torch.load(file, map_location='cpu', weights_only=True)
            assert len(p['model']) == 212
            assert all(not t.is_floating_point() or (t.dtype == torch.float16 and bool(torch.isfinite(t).all())) for t in p['model'].values())
            del p; gc.collect()
            model = I.load(file, torch.device('cpu'))
            with torch.no_grad():
                greedy = I.Fast.decode(I.Fast.Context(model, src, 16), compute_logprob=False)
                torch.manual_seed(270042)
                sample = I.Fast.decode(I.Fast.Context(model, src, 16), k=4, temperature=.7)
            assert greedy[0].shape[:2] == (2, 1) and sample[0].shape[:2] == (2, 4)
            assert torch.isfinite(sample[1]).all()
            weight_checks[name] = {'SHA256': I.sha(file), 'bytes': file.stat().st_size, 'finite_FP16_tensors': 212,
                                   'strict_model_load': True, 'CPU_greedy_rows': 2, 'CPU_K4_rows': 2, 'cap': 16}
            del model, sample, greedy; gc.collect()
        # Saved candidates: actual v3 selector/code representative, no new sampling/gold-dependent selection.
        heldout = read(WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/heldout4k.json')
        journals, baseline = S.stored_tune(FINAL, heldout)
        assert baseline['matches'] == 1511
        winners = [select(journals['C-main'][i]['greedy'], journals['C-main'][i]['samples']['0.7'][:4],
                          journals['C-long2x'][i]['samples']['0.7'][:4]) for i in S.fixed_split()['tune2048'][:16]]
        verification = verify_winners(winners, time.monotonic()+45, workers=2)
        assert verification['complete'] and not verification['mismatch_rows']
        write_answers(scratch/'answers.jsonl', [w['code'] for w in winners])
        validate_answers(scratch/'answers.jsonl', 16)
    receipt = {'passed': True, 'nbformat': True, 'metadata': meta, 'limits': limits, 'remote_L3_live': access['source'],
               'base_V3_sources_unchanged': True, 'inference_worker_only_weight_SHA_and_quality_metadata_changed': True,
               'unchanged_decoder_executor_vote_sources': True, 'weights': weight_checks, 'stored_tune': baseline,
               'selected_code_reexecution': verification, 'GPU': False, 'push': False,
               'GPU_timing_quality_gate': 'pending observer run', 'CPU_scope': 'real FP16 strict loads, 2-row cap16 greedy/K4 each, 16 saved mixed winners'}
    (WORK/'checks/final-inference-l3-cpu.json').write_text(json.dumps(receipt, indent=2)+'\n')
    ready.write_text(json.dumps({'bundle_path': str(BUNDLE.resolve()), 'ready': True,
          'kernel_ref': meta['id'], 'session_timeout_seconds': 720,
          'checks': str((WORK/'checks/final-inference-l3-cpu.json').resolve()), 'GPU_gate': 'pending'}, indent=2)+'\n')
    print('V4 CPU/STATIC PASSED; next-inference ready=true; GPU/push=0')


if __name__ == '__main__': main()
