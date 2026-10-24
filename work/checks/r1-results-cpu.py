"""Inspect real R1 stable/FP16 payloads on CPU, without forwards or training."""
import gc
import hashlib
import json
from pathlib import Path

import torch

WORK = Path(__file__).resolve().parents[1]
RAW = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-r1/1'


def read(p):
    return json.loads(p.read_text())


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    result = dict(passed=True, GPU=False, model_forwards=0, training_updates=0, models={})
    for gpu, name in [('0', 'deep'), ('1', 'control')]:
        root = RAW / f'model-GPU{gpu}'
        latest = read(root / 'latest.json')
        training = read(root / 'points/final/training.json')
        fp16 = read(root / 'points/final/checkpoint.json')
        for receipt in [latest, fp16]:
            p = root / receipt['file']
            assert p.stat().st_size == receipt['bytes'] and digest(p) == receipt['sha256']
        full = torch.load(root / latest['file'], map_location='cpu', weights_only=False)
        assert full['format'] == 'nlp-hw02-r1-full-v1'
        state = full['state']
        for key in ['updates', 'cursor', 'main_seconds']:
            assert state[key] == latest[key]
        assert state['r1']['added_updates'] == training['main_updates']
        assert state['last_lr'] == .0015 and state['micro_batch'] == 64
        assert state['r1']['anchor_updates'] == 209144 and state['r1']['anchor_cursor'] == 20734592
        assert state['stability']['total'] == state['stability']['escalations'] == state['nonfinite_skips'] == 0
        assert not state['stability']['history']
        assert all(torch.isfinite(t).all().item() for t in full['model'].values())
        for s in full['optimizer']['state'].values():
            assert all(torch.isfinite(t).all().item() for t in s.values() if torch.is_tensor(t))
            assert (s['exp_avg_sq'] >= 0).all().item()
        info = dict(stable_sha256=latest['sha256'], model_spec=full['model_spec'],
                    stable_updates=state['updates'], stable_cursor=state['cursor'],
                    main_wall_s_excluding_final_save=state['r1']['main_wall_seconds'],
                    main_AMP_skips=state['AMP_skips'] - 84,
                    cumulative_AMP_skips=state['AMP_skips'], nonfinite_skips=state['nonfinite_skips'],
                    escalations=state['stability']['escalations'],
                    parameter_groups=len(full['optimizer']['param_groups']),
                    finite_model_tensors=len(full['model']), finite_optimizer_states=len(full['optimizer']['state']),
                    finite_nonnegative_second_moments=True, stable_has_no_terminal_cooldown=True,
                    scaler=full['scaler'])
        del full, state, s
        gc.collect()
        cooled = torch.load(root / fp16['file'], map_location='cpu', weights_only=False)
        assert cooled['model_spec'] == info['model_spec']
        assert all(t.dtype == torch.float16 and torch.isfinite(t).all().item() for t in cooled['model'].values())
        info.update(FP16_sha256=fp16['sha256'], finite_FP16_tensors=len(cooled['model']))
        result['models'][name] = info
        del cooled
        gc.collect()
    target = WORK / 'checks/r1-results-cpu.json'
    result['source_sha256'] = digest(Path(__file__))
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
