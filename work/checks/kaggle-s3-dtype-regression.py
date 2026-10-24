"""CPU reproduction of S3 CUDA dtype transition and eval->training regression."""
import json
import sys
import tempfile
import time
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'kaggle/src'))


def main():
    import torch
    import exp26_core as C
    import s3_runtime as S
    import s3_representation as R
    from exp26_common import write_json
    torch.set_num_threads(1)
    tokpath = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json'
    spec = dict(d_model=16, heads=2, encoder_layers=1, decoder_layers=1, swiglu_dim=32, max_positions=512)
    directory = Path(tempfile.mkdtemp(prefix='nlp-s3-dtype-regression-'))
    checks = []
    for variant in [30, 31]:
        tok, tokenbytes = R.tokenizer(tokpath, variant)
        model = C.SimpleTransformer(**spec, vocab=tok.get_vocab_size()+3, dropout=0.)
        src, tgt = torch.tensor([[1, 5, 2]]), torch.tensor([[1, 6, 2]])
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        model(src, tgt).float().square().mean().backward(); opt.step(); opt.zero_grad(set_to_none=True)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        # Reproduce precisely the CUDA-only part the original float-only smoke missed.
        with torch.inference_mode(): model.half(); model.float(); model.load_state_dict(state)
        assert any(torch.is_inference(p) for p in model.parameters())
        try:
            model(src, tgt).float().square().mean().backward()
        except RuntimeError as error:
            assert 'Inference tensors cannot be saved for backward' in str(error)
            reproduced = str(error).split('.')[0]
        else: raise AssertionError('Original dtype bug was not reproduced')
        # Reload the saved state into a fresh model, as a future resume would do.
        fresh = C.SimpleTransformer(**spec, vocab=tok.get_vocab_size()+3, dropout=0.)
        fresh.load_state_dict(state)
        opt = torch.optim.AdamW(fresh.parameters(), lr=.001)
        identities = [id(p) for p in fresh.parameters()]
        with torch.no_grad(): fresh.half(); fresh.float()
        assert identities == [id(p) for p in fresh.parameters()]
        assert not any(torch.is_inference(p) for p in fresh.parameters())
        # Invoke the actual patched evaluate, not just a context-manager toy.
        rows = [dict(question='Add 2 and 3', code='def solve() -> int:\n    return 5\n')]
        sets = {s: rows for s in ['heldout4k', 'old1024']}
        refs = {s: dict(rows=[dict(status='numeric', value=5.)]) for s in sets}
        base = dict(splits={s: dict(exec=dict(eligible_indices=[0], match_indices=[], denominator=1)) for s in sets})
        fresh.half()  # Force conversion within evaluate even on CPU.
        metrics = S.evaluate(fresh, sets, refs, tokpath, tok, tokenbytes, variant, torch.device('cpu'),
            directory / str(variant), base, time.monotonic()+90, smoke=True)
        assert identities == [id(p) for p in fresh.parameters()]
        assert not any(torch.is_inference(p) for p in fresh.parameters())
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            fresh(src, tgt).float().square().mean().backward(); opt.step()
        assert all(int(v['step']) == 2 for v in opt.state.values())
        checks.append(dict(variant=variant, original_error_reproduced=reproduced,
            actual_evaluate_then_two_AdamW_updates=True, parameter_identity_preserved=True,
            inference_parameters_after_fix=0, evaluated_splits=list(metrics['splits'])))
    report = dict(status='passed', device='CPU', checks=checks,
        old_smoke_gap='FP32->FP32 on CPU did not exercise CUDA half/float conversion',
        CUDA_verified=False, full_training_updates=0, push=False)
    write_json(WORK / 'checks/kaggle-s3-dtype-regression.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
