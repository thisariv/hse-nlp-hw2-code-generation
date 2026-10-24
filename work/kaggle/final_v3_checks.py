"""V3 guard/OOM regressions and relative CPU decoder benchmark; no GPU/network."""
import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))


def guard_child(oom):
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    import final_inference as inference
    state = {'oom': oom, 'draws': []}
    def decode(ctx, k=1, temperature=None, **kwargs):
        if temperature is not None:
            state['draws'].append(float(torch.rand(())))
            if state['oom']:
                state['oom'] = False
                raise torch.cuda.OutOfMemoryError('Injected after consuming RNG')
        return torch.zeros(len(ctx.src), k, 1, dtype=torch.long), torch.zeros(len(ctx.src), k), 0
    def pack(ids, means, table, greedy=False):
        return [[{'code': 'def solve():\n    return '+('1' if greedy else '2'),
                  'mean_logprob': 0. if greedy else -1.} for _ in range(ids.shape[1])]
                for _ in range(ids.shape[0])]
    with tempfile.TemporaryDirectory(prefix='final-partial-guard-') as folder:
        folder = Path(folder)
        rows = folder/'rows.json'
        rows.write_text(json.dumps([{'question': q} for q in ['a', 'bb', 'ccc']]))
        cfg = {'rows': str(rows), 'tokenizer': 'fixture', 'models': {'C-main': 'a', 'C-long2x': 'b'},
               'output': str(folder/'output'), 'verify_weights': False, 'device': 'cpu',
               'workers': 1, 'cap': 8, 'budget_seconds': 8., 'greedy_batch': 2, 'sample_batch': 2,
               'cpu_test_expire_after_main_batches': 1}
        with patch.object(inference, 'load', return_value=object()), \
             patch.object(inference, 'tokenizer', return_value=(None, {})), \
             patch.object(inference.C, 'encode', side_effect=lambda tok, question: [3+len(question)]), \
             patch.object(inference.Fast, 'Context', side_effect=lambda model, src, cap: SimpleNamespace(src=src)), \
             patch.object(inference.Fast, 'decode', side_effect=decode), \
             patch.object(inference, 'pack', side_effect=pack):
            timing = inference.run(cfg)  # Real execution, vote, verification, files and timer.
        answers = [row['code'] for row in map(json.loads, (folder/'output/answers.jsonl').read_text().splitlines())]
        sampled = 1 if oom else 2
        assert answers == ['def solve():\n    return 2']*sampled+['def solve():\n    return 1']*(3-sampled)
        guard = timing['deadline_guard']
        assert guard['triggered'] and guard['partial_vote_rows'] == list(range(sampled))
        assert guard['fallback_rows'] == list(range(sampled, 3))
        assert guard['no_executed_sample_count'] == 3-sampled
        assert timing['winner_reexecution']['complete'] and not timing['winner_reexecution']['mismatch_rows']
        assert timing['full_pipeline_seconds'] < 8.
        assert (folder/'output/answers.jsonl').read_bytes() != (folder/'output/greedy-fallback.jsonl').read_bytes()
        if oom:
            assert len(timing['OOM_attempts']) == 1 and state['draws'][0] == state['draws'][1]
            assert timing['generation_batches'][-1]['questions'] == 1
        return {'small_budget_seconds': 8., 'seconds': timing['full_pipeline_seconds'],
                'partial_vote_count': sampled, 'greedy_only_count': 3-sampled,
                'partial_sample_winners_preserved': True, 'independent_reexecution_completed': True,
                'OOM_halves_batch_and_restores_CPU_RNG': bool(oom), 'GPU_decoder': 'mock; executor real'}


def available_vote():
    from final_vote import select_available
    def candidate(value, status='numeric'):
        return {'code': str(value), 'mean_logprob': -1.,
                'execution': {'status': status, 'value': float(value)}}
    greedy = candidate(1)
    row = {'greedy': greedy, 'main': [candidate(2)]*3+[candidate(99, 'budget_not_evaluated')], 'long': []}
    winner, count = select_available(row)
    assert count == 3 and winner['code'] == '2'
    winner, count = select_available(dict(row, main=[candidate(3, 'budget_not_evaluated')]))
    assert count == 0 and winner is greedy
    return {'missing_long_and_unexecuted_samples_do_not_discard_votes': True,
            'no_executed_samples_returns_greedy': True}


def decoder_benchmark():
    import torch
    import final_core as C
    import final_fast as new
    import exp27b_fast as old
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = C.SimpleTransformer(d_model=32, heads=4, encoder_layers=1, decoder_layers=2,
                                swiglu_dim=48, vocab=8195, max_positions=128, dropout=0.).eval()
    src = torch.tensor([[1]+list(range(3, 25))+[2]]*1024)
    report = {'device': 'CPU FP32', 'MPS_available': torch.backends.mps.is_available(),
              'scope': 'tiny random model, 1024 total lanes, cap48; relative only, not a T4 forecast',
              'model_width': 32, 'decoder_layers': 2, 'vocab': 8195, 'cases': {}}
    for name, k, temperature in [('greedy', 1, None), ('sampling', 4, .7)]:
        rows = 1024//k
        samples = {}
        for version, module, batch in [('v2', old, 256//k), ('v3', new, 1024//k)]:
            module.decode(module.Context(model, src[:2], 16), k, temperature)
            times = []
            for _ in range(2):
                torch.manual_seed(270042)
                started = time.monotonic()
                for lo in range(0, rows, batch):
                    ctx = module.Context(model, src[lo:lo+batch], 48)
                    extra = {'compute_logprob': temperature is not None} if version == 'v3' else {}
                    ids, lp, _ = module.decode(ctx, k, temperature, **extra)
                    assert ids.shape[:2] == (len(ctx.src), k) and torch.isfinite(lp).all()
                times.append(time.monotonic()-started)
            samples[version] = {'seconds': times, 'median_seconds': statistics.median(times), 'lanes_per_batch': batch*k}
        samples['speedup'] = samples['v2']['median_seconds']/samples['v3']['median_seconds']
        report['cases'][name] = samples
    # Early all-EOS exit and the greedy LP-free optimization preserve tokens exactly.
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.bias[C.EOS] = 10.
    a = old.decode(old.Context(model, src[:2], 64))[0]
    b, lp, _ = new.decode(new.Context(model, src[:2], 64), compute_logprob=False)
    assert torch.equal(a, b) and b.shape[-1] == 16 and torch.count_nonzero(b[:, :, 1:]) == 0
    assert torch.count_nonzero(lp) == 0
    report['EOS_stops_at_first_check_and_greedy_tokens_exact'] = True
    return report


def main():
    report = {'vote': available_vote(), 'guards': [], 'GPU_used': False, 'Kaggle_push': False}
    for case in ('plain', 'oom'):
        child = subprocess.run([sys.executable, __file__, '--guard-child', case],
                               capture_output=True, text=True, timeout=15.)
        assert child.returncode == 0, child.stderr
        report['guards'].append(json.loads(child.stdout))
    report['relative_decoder_benchmark'] = decoder_benchmark()
    (ROOT.parent/'checks/final-inference-v3.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guard-child', choices=['plain', 'oom'])
    args = parser.parse_args()
    if args.guard_child:
        print(json.dumps(guard_child(args.guard_child == 'oom')))
    else:
        main()
