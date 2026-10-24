"""CPU correctness/cache/statics gates for decode-profile; never uses Kaggle credentials."""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import nbformat
import torch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import final_core as C
import final_fast as Fast
import final_speed as Speed
import final_profile as P
from build_final_profile import BUNDLE, READY, NAME, FILES


def equal(a, b, cap):
    return torch.equal(P.canonical(a, cap), P.canonical(b, cap))


def tiny():
    torch.manual_seed(42)
    return C.SimpleTransformer(d_model=16, heads=2, encoder_layers=1, decoder_layers=2,
        swiglu_dim=32, vocab=32, max_positions=64, dropout=0.).eval()


@torch.inference_mode()
def cache_tests():
    model = tiny()
    src = torch.tensor([[1, 5, 2, 0], [1, 6, 9, 2], [1, 8, 2, 0]])
    ctx = Fast.Context(model, src, 20)
    a, b = Fast.Decoder(ctx, 1), Speed.Decoder(ctx, 1, share_cross=True)
    assert all(x.data_ptr() == y.data_ptr() for p, q in zip(b.cross, ctx.cross) for x, y in zip(p, q))
    ptrs = [(k.data_ptr(), v.data_ptr()) for k, v in b.self_kv]
    cross_ptrs = [(k.data_ptr(), v.data_ptr()) for k, v in b.cross]
    for token in (torch.tensor([1, 1, 1]), torch.tensor([5, 7, 8]), torch.tensor([2, 9, 4])):
        assert torch.equal(a.step(token), b.step(token))
        assert ptrs == [(k.data_ptr(), v.data_ptr()) for k, v in b.self_kv]
        assert cross_ptrs == [(k.data_ptr(), v.data_ptr()) for k, v in b.cross]
    assert b.t == 3
    assert torch.equal(b.valid[:, :3], torch.ones(3, 3, dtype=torch.bool))
    prefix = [[t[:, :, :3].clone() for t in pair] for pair in b.self_kv]
    index = torch.tensor([2, 0])  # Deliberately reorder to test mapping, not just prefix truncation.
    b.select(index)
    assert b.t == 3 and b.n == 2
    for pair, old in zip(b.self_kv, prefix):
        for new, prior in zip(pair, old):
            assert torch.equal(new[:, :, :3], prior[index])
    token = torch.tensor([10, 11, 12])
    assert torch.allclose(a.step(token)[index], b.step(token[index]), atol=1e-6, rtol=1e-6)
    assert b.t == 4
    # PAD is invalid in the per-position mask; caches still keep a stable address.
    b.step(torch.tensor([0, 2]))
    assert b.valid[:, 4].tolist() == [False, True]
    repeated = Speed.Decoder(ctx, 4, share_cross=True)
    assert repeated.cross[0][0].shape[0] == 12
    assert torch.equal(repeated.cross[0][0], ctx.cross[0][0].repeat_interleave(4, 0))
    try:
        Speed.Options(cuda_graphs=True)
        raise AssertionError('Graph refusal must be explicit')
    except NotImplementedError:
        pass
    return {'passed': True, 'stable_self_and_cross_pointers': True, 'initialized_prefix_only': True,
            'reorder_and_continue': True, 'PAD_mask': True, 'K4_cross_repeat': True}


def decoder_tests():
    model = tiny()
    src = torch.tensor([[1, 5, 2, 0], [1, 6, 9, 2], [1, 8, 2, 0]])
    report = {}
    baseline = Fast.decode(Fast.Context(model, src, 48), compute_logprob=False)[0]
    for name, flags in P.VARIANTS.items():
        if name in ('baseline', 'workers2', 'nice10'):
            continue
        candidate = Speed.decode(Fast.Context(model, src, 48), compute_logprob=False,
                                 options=Speed.Options(**flags))[0]
        assert equal(baseline, candidate, 48), name
        report[name] = True
    # Exercise actual compaction, EOS exactly once, PAD tails, staggered finish,
    # retained logprob sums/lengths and question/lane mapping. Real KV path is above.
    class Scripted(Speed.Decoder):
        def __init__(self, ctx, k, *_):
            super().__init__(ctx, k)
            self.identity = torch.arange(self.n)
        def step(self, tokens):
            super().step(tokens)
            logits = torch.full((self.n, 32), -1000.)
            picks = torch.where(self.t >= self.identity+1, C.EOS, 5)
            logits.scatter_(1, picks[:, None], 1000.)
            return logits
        def select(self, index):
            super().select(index)
            self.identity = self.identity[index]
    stats = {}
    with patch.object(Speed, 'Decoder', Scripted):
        a = Speed.decode(Fast.Context(model, src, 16), 4, options=Speed.Options(eos_every=2))
        b = Speed.decode(Fast.Context(model, src, 16), 4,
            options=Speed.Options(eos_every=2, compact_done=True), stats=stats)
    assert equal(a[0], b[0], 16) and torch.equal(a[1], b[1])
    assert len(stats['compactions']) >= 2
    for i, seq in enumerate(P.canonical(b[0], 16).reshape(12, 16)):
        assert seq.tolist() == [5]*i+[C.EOS]+[C.PAD]*(15-i)
    # CPU non-async fallback, capacity boundary (no EOS), and original sampler RNG.
    torch.manual_seed(123)
    a = Fast.decode(Fast.Context(model, src, 13), 4, .7)
    torch.manual_seed(123)
    b = Speed.decode(Fast.Context(model, src, 13), 4, .7)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # Categorical law of the race, including exactly zero probability classes.
    torch.manual_seed(777)
    probabilities = torch.tensor([[0., .1, .2, .7]]).expand(40000, -1).contiguous()
    picks = Speed.race_pick(probabilities, torch.empty_like(probabilities))
    frequency = torch.bincount(picks, minlength=4).float()/len(picks)
    assert frequency[0] == 0 and torch.allclose(frequency, probabilities[0], atol=.01, rtol=0)
    return {'passed': True, 'greedy_tiny_FP32': report, 'staggered_EOS_K4_compactions': stats,
            'sampling_baseline_RNG_exact_without_flags': True, 'race_probabilities': frequency.tolist(),
            'no_EOS_capacity_boundary': True}


@torch.inference_mode()
def real_cpu():
    from final_representation import tokenizer
    path = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final/model-C-main__weights__l2-62280-fp16.pt'
    assert P.sha(path) == P.WEIGHT_SHA
    payload = torch.load(path, map_location='cpu', weights_only=True)
    model = C.SimpleTransformer(**payload['model_spec'], dropout=0.).eval()
    model.load_state_dict(payload['model'], strict=True)
    del payload
    inputs = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    tok, _ = tokenizer(inputs/'tokenizer.json', 30)
    rows = json.loads((inputs/'test.json').read_text())
    encoded = [[C.BOS]+C.encode(tok, r['question'])+[C.EOS] for r in rows]
    order = sorted(range(len(rows)), key=lambda i: (len(encoded[i]), i))
    # Four real questions spanning lengths, full512 capacity; all arithmetic FP32.
    selected = [order[i] for i in [0, 639, 1278, 1918]]
    src = torch.nn.utils.rnn.pad_sequence([torch.tensor(encoded[i]) for i in selected],
                                        batch_first=True, padding_value=C.PAD)
    begin = time.monotonic()
    baseline = Fast.decode(Fast.Context(model, src, 512), compute_logprob=False)[0]
    result = {}
    for name in ['share_cross', 'async_eos', 'compact_done', 'race_sampler', 'lean_greedy', 'all']:
        stats = {}
        candidate = Speed.decode(Fast.Context(model, src, 512), compute_logprob=False,
                                 options=Speed.Options(**P.VARIANTS[name]), stats=stats)[0]
        assert equal(baseline, candidate, 512), name
        result[name] = {'greedy_bit_exact': True, **stats}
    return {'passed': True, 'real_L3_weights': True, 'weight_sha256': P.WEIGHT_SHA,
            'dtype': 'float32', 'test_row_ids': selected, 'capacity': 512,
            'baseline_steps': baseline.shape[-1], 'variants': result, 'seconds': time.monotonic()-begin}


def static():
    notebook = json.loads((BUNDLE/NAME).read_text())
    nbformat.validate(nbformat.from_dict(notebook))
    codes = [c['source'] for c in notebook['cells'] if c['cell_type'] == 'code']
    for c in notebook['cells']:
        if c['cell_type'] == 'code':
            compile(c['source'], c['id'], 'exec')
            assert not c['outputs'] and c['execution_count'] is None
    sources = ast.literal_eval(next(n.value for n in ast.parse(codes[0]).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SOURCES'))
    identity = json.loads((BUNDLE/'source-identity.json').read_text())
    assert set(sources) == set(FILES)
    for name, source in sources.items():
        assert source == (ROOT/'src'/name).read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == identity['source_sha256'][name]
        compile(source, name, 'exec')
        assert 'p1_' not in source and '.kaggle' not in source
    assert hashlib.sha256((BUNDLE/NAME).read_bytes()).hexdigest() == identity['notebook_sha256']
    assert (ROOT/'notebooks'/NAME).read_bytes() == (BUNDLE/NAME).read_bytes()
    metadata = json.loads((BUNDLE/'kernel-metadata.json').read_text())
    assert metadata['id'] == 'thisarivson/nlp-hw02-decode-profile'
    assert re.sub('[^a-z0-9]+', '-', metadata['title'].lower()).strip('-') == metadata['id'].split('/')[1]
    assert metadata['is_private'] and not metadata['enable_internet'] and metadata['enable_gpu']
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['dataset_sources'] == ['thisarivson/nlp-hw02-l3-live', 'thisarivson/nlp-hw02-26b-inputs']
    limits = json.loads((BUNDLE/'session-limits.json').read_text())
    assert limits['session_timeout_seconds'] == 720 and limits['watchdog_stop_seconds'] == 690
    assert 'SESSION_START+650' in codes[-1] and '>= 690' in codes[0]
    assert 'while child.poll() is None:' in codes[-1] and 'kill_children()' in codes[-1]
    assert "SCOPE = "+repr(limits['scope']) in codes[-1]
    # Ensure frozen production code did not change, including every P1 tracked file.
    frozen = [f'kaggle/src/{n}' for n in FILES[:7]]
    status = subprocess.run(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.splitlines()
    assert not any('/p1_' in p or '/p1-' in p or p.endswith('/build_p1.py') for p in status)
    assert not any(p.endswith('/'+n) for p in status for n in frozen)
    # A synthetic trace validates overlap-union calculation and sync attribution.
    fake = {'traceEvents': [
        {'name': 'decode_window', 'ph': 'X', 'ts': 0, 'dur': 100, 'tid': 1},
        {'name': 'k', 'ph': 'X', 'ts': 10, 'dur': 30, 'cat': 'kernel'},
        {'name': 'k', 'ph': 'X', 'ts': 20, 'dur': 30, 'cat': 'kernel'},
        {'name': 'cudaStreamSynchronize', 'ph': 'X', 'ts': 50, 'dur': 10, 'cat': 'cuda_runtime'}]}
    summary = P.trace_summary(fake)
    assert summary['kernel_busy_fraction'] == .4 and summary['sync_and_memcpy_api_fraction'] == .1
    return {'passed': True, 'embedded_sources': len(sources), 'nbformat_and_compile': True,
            'identity': True, 'private_offline_T4_metadata': True, 'watchdog': True,
            'frozen_production_and_P1_unchanged': True, 'trace_interval_union': True,
            'scope': limits['scope']}


def load_smoke():
    with tempfile.TemporaryDirectory(prefix='final-profile-load-') as tmp:
        folder = Path(tmp)
        with P.Load(folder, 1, 5) as load:
            load.alive()
            info = json.loads((folder/'ready.json').read_text())
            assert info['workers'] == 1 and (info['nice_applied'] or info['nice_error'])
            time.sleep(.2)
        assert load.child.poll() == 0
        summary = json.loads((folder/'load-summary.json').read_text())
        assert summary['completed_unique'] > 0 and summary['error'] is None
    return {'passed': True, 'real_executor_service': True, 'nice_applied': info['nice_applied'],
            'nice_error': info['nice_error'], 'clean_exit': True}


def main():
    READY.write_text(json.dumps({'bundle': str(BUNDLE.resolve()), 'ready': False})+'\n')
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    report = {'torch': str(torch.__version__), 'GPU': False, 'push': False}
    for name, fn in [('cache', cache_tests), ('decoder', decoder_tests), ('real_CPU_FP32', real_cpu),
                     ('load_smoke', load_smoke), ('static', static)]:
        report[name] = fn()
        print(name, 'passed', flush=True)
    report.update(passed=True, seconds=time.monotonic()-started,
                  GPU_runtime_gate='pending Torch2.11.0+cu128/T4; local Torch version differs')
    receipt = WORK/'checks/final-profile-cpu.json'
    receipt.write_text(json.dumps(report, indent=2)+'\n')
    READY.write_text(json.dumps({'bundle': str(BUNDLE.resolve()), 'ready': True,
        'kernel': 'thisarivson/nlp-hw02-decode-profile', 'checks': str(receipt.resolve()),
        'notebook_sha256': json.loads((BUNDLE/'source-identity.json').read_text())['notebook_sha256'],
        'GPU_gate': 'pending', 'push': False, 'scope': report['static']['scope'],
        'readiness': 'bundle upload only; neither 2x speedup nor GPU bit-exact is established'}, indent=2)+'\n')
    print('CPU/static gates passed; ready:true; GPU/push=0', flush=True)


if __name__ == '__main__':
    main()
