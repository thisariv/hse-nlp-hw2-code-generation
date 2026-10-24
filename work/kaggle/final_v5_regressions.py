"""Run existing v3/v4 CPU checks without modifying frozen sources or receipts."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


def v4():
    from unittest.mock import patch
    import final_l3_checks as checks
    write = Path.write_text
    blocked = {checks.FINAL/'next-inference.json', ROOT.parent/'checks/final-inference-l3-cpu.json'}
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in blocked}
    def guarded(path, text, *args, **kwargs):
        return len(text) if path in blocked else write(path, text, *args, **kwargs)
    with patch.object(Path, 'write_text', guarded):
        checks.main()
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == d for p, d in digests.items())
    return {'passed': True, 'unchanged_v4_static_real_FP16_loads_and_reexecution': True,
            'v4_receipts_unchanged': True, 'GPU': False}


def real_decoder():
    """Exercise actual v5 tokenizer/decoder/packing/executor at the selected K."""
    import torch
    import final_inference_v5 as I
    from unittest.mock import patch
    work = ROOT.parent
    tokenizer = work/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json'
    tok, _ = I.tokenizer(tokenizer, 30)
    model = I.C.SimpleTransformer(d_model=16, heads=2, encoder_layers=1, decoder_layers=1,
                                 swiglu_dim=32, vocab=tok.get_vocab_size(), max_positions=512, dropout=0.).eval()
    # Empty EOS programs force escalation, giving deterministic coverage of both
    # stages without pretending random tiny weights measure quality or T4 speed.
    with torch.no_grad():
        model.head.weight.zero_(); model.head.bias.fill_(-1000.); model.head.bias[I.C.EOS] = 1000.
    selected = json.loads((work/'checks/final-inference-v5-offline.json').read_text())['selected_K2']
    with tempfile.TemporaryDirectory(prefix='final-v5-real-') as directory:
        directory = Path(directory)
        rows = directory/'rows.json'
        rows.write_text(json.dumps([{'question': 'Compute 2 + 3.'}, {'question': 'What is 7 times 6?'}]))
        cfg = {'rows': str(rows), 'tokenizer': str(tokenizer), 'models': {'C-main': 'a', 'C-long2x': 'b'},
               'output': str(directory/'out'), 'device': 'cpu', 'verify_weights': False, 'workers': 1,
               'K1': 2, 'K2': selected, 'cap': 16, 'sequence_budget': 8, 'greedy_sequence_budget': 2,
               'budget_seconds': 30.}
        begin = time.monotonic()
        with patch.object(I, 'load', return_value=model):
            timing = I.run(cfg)
        assert not timing['deadline_guard']['triggered']
        assert timing['adaptive']['escalated_rows'] == [0, 1]
        assert timing['deadline_guard']['executed_sample_counts'] == [2*selected]*2
        assert any(b['stage'] == 2 for b in timing['generation_batches'])
        assert timing['winner_reexecution']['complete'] and not timing['winner_reexecution']['mismatch_rows']
    return {'passed': True, 'real_tiny_CPU_decoder': True, 'real_tokenizer_pack_execution': True,
            'K1': 2, 'K2': selected, 'cap': 16, 'seconds': time.monotonic()-begin, 'GPU': False}


def main():
    import final_inference_checks as checks
    import final_review_checks as review
    report = {name: getattr(checks, name)() for name in
              ['unit', 'sampler_checks', 'static', 'suite_checks', 'supervisor_checks', 'process_smoke']}
    report['frozen_L2_journals'] = checks.journals()[0]
    report['security'] = review.regression_checks()
    report['v3_guard'] = {}
    for case in ['plain', 'oom']:
        child = subprocess.run([sys.executable, str(ROOT/'final_v3_checks.py'), '--guard-child', case],
                               capture_output=True, text=True, timeout=20.)
        assert child.returncode == 0, child.stderr
        report['v3_guard'][case] = json.loads(child.stdout)
    report['passed'] = True
    (ROOT.parent/'checks/final-inference-v5-regressions.json').write_text(json.dumps(report, indent=2)+'\n')
    child = subprocess.run([sys.executable, __file__, '--v4-child'], capture_output=True, text=True, timeout=60.)
    assert child.returncode == 0, child.stderr
    child = subprocess.run([sys.executable, __file__, '--real-decoder-child'], capture_output=True, text=True, timeout=35.)
    assert child.returncode == 0, child.stderr
    print('V3/V4 CPU regressions passed; frozen receipts unchanged')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v4-child', action='store_true')
    parser.add_argument('--real-decoder-child', action='store_true')
    args = parser.parse_args()
    if args.v4_child:
        report = v4()
        (ROOT.parent/'checks/final-inference-v5-v4-regression.json').write_text(json.dumps(report, indent=2)+'\n')
    elif args.real_decoder_child:
        report = real_decoder()
        (ROOT.parent/'checks/final-inference-v5-real-decoder.json').write_text(json.dumps(report, indent=2)+'\n')
    else:
        main()
