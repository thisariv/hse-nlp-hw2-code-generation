"""One independent B2 process per visible T4; no credentials."""
import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gpu', type=int, required=True); p.add_argument('--config', required=True)
    a = p.parse_args(); cfg = json.loads(Path(a.config).read_text()); root = Path(cfg['working_root'])
    sys.path.insert(0, str(root/'src'))
    import torch
    import exp26_core as C
    import b2_runtime as B
    import b1_eval as E
    from b1_data import verify_order
    from l1_cache import CompactCache
    from exp26_input import validate_training_inputs
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    device = torch.device('cuda:0')
    assert torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0)
    assert C.fused_adamw_available(device)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cache = CompactCache(cfg['compact_cache'], verify=False)
    assert cache.manifest_sha == cfg['cache_manifest_sha256']
    order, meta = verify_order(root/'weighted-order', cache)
    name = ('W-lr1000', 'W-lr600')[a.gpu]; model_root = root/f'model-GPU{a.gpu}'
    inputs = Path(cfg['starter_dataset']); full, _, _ = validate_training_inputs(inputs)
    refs = {k:json.loads((inputs/(k+'-reference-execution.json')).read_text()) for k in full}
    eligible = [i for i, r in enumerate(refs['old1024']['rows']) if r['status'] == 'numeric']; assert len(eligible) == 991
    sets = dict(heldout4k=full['heldout4k'], old991=[full['old1024'][i] for i in eligible])
    refs = dict(heldout4k=refs['heldout4k'], old991=dict(rows=[refs['old1024']['rows'][i] for i in eligible]))
    if cfg['gpu_smoke']:
        sets = {k:v[:64 if k == 'heldout4k' else 16] for k, v in sets.items()}
        refs = {k:dict(rows=v['rows'][:len(sets[k])]) for k, v in refs.items()}
        from b2_gpu_smoke import exercise
        exercise(name, cache, order, meta, model_root, device, cfg, inputs, sets, refs)
        return
    def evaluate(point, out, deadline, final):
        return E.evaluate(point, out, inputs, sets, refs, device, deadline, final)
    resume = Path(cfg['resume_root'])/f'model-GPU{a.gpu}' if cfg.get('resume_root') else None
    result = B.run(name, cache, order, meta, model_root, device, cfg, evaluate, resume=resume)
    assert result['status'] == 'complete', 'B2 incomplete; stable retained'


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        import traceback
        traceback.print_exc()
        print('B2 WORKER FAILED', type(error).__name__, flush=True); sys.exit(1)
