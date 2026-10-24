"""D1 worker: one visible GPU, one pinned L3 model, no authentication environment."""
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['working_root']); sys.path.insert(0, str(root/'src'))
    import numpy as np
    import torch
    import final_core as C
    from final_representation import tokenizer
    from l1_cache import CompactCache
    from kaggle_io import sha, write
    from d1_data import MODELS, WEIGHTS, ranked_candidates, choose_budget, save_npz, publish, verify_receipt
    from d1_loss import score_all, parity_check, merge_shards
    from d1_generate import generation_wave
    name = MODELS[args.gpu]; out = root/'d1'/name; out.mkdir(parents=True, exist_ok=True)
    start = cfg['session_start_monotonic']
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    device = torch.device('cuda:0')
    assert torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
    assert str(torch.__version__) == '2.11.0+cu128' and torch.version.cuda == '12.8'
    assert not any(os.environ.get(s) for s in ('KAGGLE_KEY', 'KAGGLE_API_TOKEN', 'KAGGLE_USER_SECRETS_TOKEN'))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True); torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    cache = CompactCache(cfg['compact_cache'], verify=False)
    assert cache.manifest_sha == cfg['cache_manifest_sha256']
    def load_model(which):
        path = Path(cfg['models'][which]); assert sha(path) == WEIGHTS[which]
        p = torch.load(path, map_location='cpu', weights_only=True)
        assert p['format'] == 'nlp-hw02-model-only-v1' and p['optimizer_included'] is False
        assert p['point'] == 'l2-62280' and p['model_spec']['vocab'] == 8195
        result = C.SimpleTransformer(**p['model_spec'], dropout=0.).eval().half()
        result.load_state_dict(p['model'], strict=True)
        return result.to(device)
    # Both GPUs score C-long2x; C-main is loaded only when generation begins.
    model = load_model('C-long2x')
    def terminated(*_): raise TimeoutError('D1 graceful stop')
    signal.signal(signal.SIGTERM, terminated)
    write(out/'identity.json', dict(model=name, weight_sha256=WEIGHTS[name],
          cache_manifest_sha256=cache.manifest_sha, tokenizer_sha256=sha(cfg['tokenizer']),
          point='l2-62280', training=False))
    tok, table = tokenizer(cfg['tokenizer'], 30)
    del tok

    def check(until, dependency=False):
        if time.monotonic() >= until or (root/'stop-request.json').exists():
            raise TimeoutError('D1 phase deadline or stop request')
        if dependency and any(root.glob('gpu*-failure.json')):
            raise RuntimeError('D1 dependency worker failed')

    def wait(path, until):
        while not path.exists():
            check(until, dependency=True); time.sleep(.25)
        return json.loads(path.read_text())

    score_ids = np.load(root/'d1/scored-row-ids.npy', allow_pickle=False)
    if cfg['gpu_smoke']:
        write(out/'loss-parity.json', parity_check(model, cache, score_ids[:16], device))
    loss_until = start+cfg['loss_stop_seconds']
    common = root/'d1'; shard_root = common/f'loss-shards/gpu{args.gpu}'
    loss = score_all(model, cache, score_ids[args.gpu::2], shard_root, device, lambda: check(loss_until),
                     batch=cfg['loss_batch'], token_slots=cfg['loss_token_slots'], block=cfg['loss_block'], compact=True)
    publish(shard_root/'done.receipt.json', [], phase='loss-shard-done', **{'summary': loss})
    write(out/'summary.json', dict(status='loss-complete', loss=loss))
    candidate_path = common/'candidates.npz'
    for gpu in (0, 1):
        wait(common/f'loss-shards/gpu{gpu}/done.receipt.json', loss_until+120)
    if args.gpu == 0:
        merged = merge_shards(common, score_ids, cache)
        ce = np.load(common/'C-long2x/loss/ce.npy', mmap_mode='r', allow_pickle=False)
        scored = score_ids[np.isfinite(ce[score_ids])]
        if not len(scored): raise RuntimeError('No scored rows')
        ranked, _ = ranked_candidates(ce[scored], ce[scored], cache.target_lengths[scored], cfg['fraction'])
        save_npz(candidate_path, **{f'bin{b}': scored[ii] for b, ii in enumerate(ranked)})
        publish(common/'candidates.receipt.json', [candidate_path], phase='candidate-ranking',
                fraction=cfg['fraction'], compact_content_edges=[53, 79, 105], rows=sum(map(len, ranked)),
                loss_complete=merged['complete'], scored_rows=len(scored), score_model='C-long2x')
    wait(common/'candidates.receipt.json', loss_until+120)
    verify_receipt(common/'candidates.receipt.json')
    with np.load(candidate_path, allow_pickle=False) as saved:
        ranked = [saved[f'bin{b}'] for b in range(4)]
    if args.gpu == 0:
        del model
        import gc
        gc.collect(); torch.cuda.empty_cache()
        model = load_model(name)
    evidence = None
    if args.gpu == 0:
        with np.load(common/'seed-evidence.npz', allow_pickle=False) as seed:
            evidence = {k: seed[k] for k in ('row_id', 'reference')}
        if cfg['gpu_smoke']:
            # Independently execute two original references/bin before the wave, testing reuse
            # of genuine numeric answers without attaching a fourth smoke input dataset.
            from d1_generate import reference_for, Executor
            jobs = []; rows = []
            for ii in ranked:
                for row in ii[:2]:
                    code, value, _ = reference_for(cache, int(row), table, evidence)
                    if code is not None: jobs.append({'code':code}); rows.append(int(row))
            executor = Executor(2)
            try:
                executor.submit(jobs); executor.drain(min(start+cfg['execution_stop_seconds'], time.monotonic()+30))
                executor.attach(jobs)
                reused = sorted((r,j['execution']['value']) for r,j in zip(rows,jobs)
                                if j['execution']['status'] == 'numeric')
                evidence = dict(row_id=np.asarray([r for r,v in reused], np.uint32),
                                reference=np.asarray([v for r,v in reused], np.float64))
                write(out/'smoke-reference-cache.json', dict(real_original_references=True, reused=len(reused)))
            finally: executor.close()
    pilot = []; pilot_ids = []
    for b, ids in enumerate(ranked):
        take = ids[:cfg['pilot_per_bin']]; pilot_ids.extend(take.tolist())
        if len(take):
            metrics = generation_wave(model, cache, take, table, device, out/f'pilot-bin-{b}.npz',
                lambda: check(start+cfg['generation_stop_seconds']),
                min(start+cfg['execution_stop_seconds'], time.monotonic()+cfg['wave_timeout_seconds']),
                evidence=evidence, batch=cfg['generation_batch'], cap=cfg['cap'],
                generation_deadline=start+cfg['generation_stop_seconds'])
            pilot.append(dict(rows=len(take), seconds=metrics['seconds']))
        else: pilot.append(dict(rows=0, seconds=0.))
    write(out/'pilot.json', dict(bins=pilot, model=name))
    selection_receipt = common/'selection.receipt.json'
    if args.gpu == 0:
        pilots = [wait(common/m/'pilot.json', start+cfg['generation_stop_seconds']) for m in MODELS]
        selected, budget = choose_budget(ranked, pilots, start+cfg['generation_stop_seconds']-time.monotonic(),
                                         safety=cfg['timing_safety'], tail=cfg['pilot_tail_reserve'])
        save_npz(common/'selection.npz', row_id=selected)
        publish(selection_receipt, [common/'selection.npz'], phase='selection', budget=budget,
                cache_manifest_sha256=cache.manifest_sha, score_precision='C-long2x persisted FP16 CE')
    selection = wait(selection_receipt, start+cfg['generation_stop_seconds'])
    verify_receipt(selection_receipt)
    with np.load(common/'selection.npz', allow_pickle=False) as saved: selected = saved['row_id']
    assert len(np.unique(selected)) == len(selected)
    remaining = selected[~np.isin(selected, np.asarray(pilot_ids, dtype=np.uint32))]
    metrics = []
    for lo in range(0, len(remaining), cfg['wave_rows']):
        try: check(start+cfg['generation_stop_seconds'])
        except TimeoutError: break
        metrics.append(generation_wave(model, cache, remaining[lo:lo+cfg['wave_rows']], table, device,
            out/f'generation-{lo:08d}.npz', lambda: check(start+cfg['generation_stop_seconds']),
            min(start+cfg['execution_stop_seconds'], time.monotonic()+cfg['wave_timeout_seconds']),
            evidence=evidence, batch=cfg['generation_batch'], cap=cfg['cap'],
            generation_deadline=start+cfg['generation_stop_seconds']))
        print('D1 GREEDY', name, min(lo+cfg['wave_rows'], len(remaining)), '/', len(remaining), flush=True)
    pilot_complete = all(verify_receipt(out/f'pilot-bin-{b}.receipt.json')['complete']
                         for b in range(4) if len(ranked[b]))
    generated_rows = len(pilot_ids)+sum(m['rows'] for m in metrics)
    full_loss = (common/'C-long2x/loss/complete.receipt.json').exists()
    write(out/'summary.json', dict(status='complete' if full_loss and generated_rows == len(selected) and
          pilot_complete and all(m['complete'] for m in metrics) else 'partial',
          loss=loss, selected_rows=len(selected), pilot_rows=len(pilot_ids), generated_rows=generated_rows,
          budget=selection['budget'], generation_waves=len(metrics), weight_sha256=WEIGHTS[name],
          wall_seconds=time.monotonic()-start))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        # Never stringify exceptions: API credential values/URLs must not reach logs.
        print('D1 WORKER FAILED', type(error).__name__, flush=True)
        sys.exit(1)
