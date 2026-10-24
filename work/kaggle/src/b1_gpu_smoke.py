"""Reviewer-run only: actual E/W probe, representative steps, resume, v7 speed, short branches."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
import b1_runtime as B
import b1_eval as E
import exp26_core as C
import exp26_train as T
import l1_runtime as L
import l2_runtime as R
from l2_gpu_smoke import digest
from kaggle_io import write


def exercise(name, cache, order, meta, root, device, cfg, inputs, sets, refs):
    root = Path(root); root.mkdir(parents=True, exist_ok=True); spec = B.SPECS[name]
    report = dict(passed=False, model=name, spec=spec, parameter_count=B.parameter_count(spec))
    write(root/'gpu-smoke.json', report)
    m = B.new_model(spec, device, cfg['dropout'])
    micro, attempts = B.probe(m, cache, device, .001, name)
    report['memory_probe'] = dict(micro_batch=micro, attempts=attempts)
    write(root/'gpu-smoke.json', report)
    print('B1 SMOKE',name,'micro',micro,attempts,flush=True)
    # Replay exactly the eight original synthetic length batches for an equal-work timer.
    import shape_bench
    rng = T.cpu(T.rng_state())
    report['shape_bench_replay'] = shape_bench.train_bench(m, cfg['shape_bench_lengths'], device,
                                                        cfg['deadline_monotonic']-120.)
    T.restore_rng(rng, device)
    baseline = {'E':246.7, 'W':229.4}[name]
    report['shape_bench_ratio'] = report['shape_bench_replay']['pairs_per_s']/baseline
    write(root/'gpu-smoke.json', report)
    o = L.optimizer(m, .001, False); s = torch.amp.GradScaler('cuda', init_scale=1024)
    state = B.new_state(micro); ident = B.identity_for(name, spec, cache, meta, cfg['dropout'])
    seconds = []; target_tokens = []; torch.cuda.reset_peak_memory_stats(device)
    for j in range(256):
        # Representative weighted order positions; this probe never feeds production state.
        if j < 36: state['cursor'] = ((j*len(order)//36)//128)*128
        elif j == 36: state['cursor'] = state['updates']*128
        log, escalate = B.main_step(m, o, s, state, cache, order, device)
        assert not escalate and not log['nonfinite_skip']
        if 4 <= j < 36: seconds.append(log['seconds']); target_tokens.append(log['tokens'])
    state['cursor'] = state['updates']*128
    report['main_attempts'] = state['updates']
    report['AMP_skips'] = state['AMP_skips']
    report['nonfinite_skips'] = state['nonfinite_skips']
    assert state['updates'] == 256 and state['nonfinite_skips'] == 0
    report.update(seconds_per_step=float(np.mean(seconds)), training_pairs_per_second=128/float(np.mean(seconds)),
                  target_tokens_per_second=sum(target_tokens)/sum(seconds), step_seconds=seconds,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device), peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    write(root/'gpu-smoke.json', report)
    print('B1 SMOKE',name,'256 steps',report['training_pairs_per_second'],'pairs/s',flush=True)
    B.save_full(root, m, o, s, state, spec, ident)
    B.main_step(m, o, s, state, cache, order, device)
    expected = digest((m.state_dict(), o.state_dict(), s.state_dict(), T.rng_state()))
    del m, o, s; R.release(device)
    m, o, s, state = B.restore(B.read_full(root, ident), device)
    B.main_step(m, o, s, state, cache, order, device)
    assert expected == digest((m.state_dict(), o.state_dict(), s.state_dict(), T.rng_state()))
    report['CUDA_next_update_bit_exact'] = True
    # Exercise peak LR after warmup without contaminating the production state.
    state.update(updates=1000, cursor=128000)
    for _ in range(16):
        peak_log, escalate = B.main_step(m, o, s, state, cache, order, device)
        assert peak_log['lr'] == .001 and not escalate and not peak_log['nonfinite_skip']
    report['peak_lr_attempts'] = 16
    report['peak_lr_AMP_skips'] = state['AMP_skips']
    write(root/'gpu-smoke.json', report)
    del m, o, s; R.release(device)
    m, o, s, state = B.restore(B.read_full(root, ident), device)
    # Overflow is an ordinary AMP skip, with model/moments unchanged.
    # Warmup advances group LR on a skipped attempt; parameters/moments must stay exact.
    before = digest((m.state_dict(), o.state_dict()['state'])); scale = s.get_scale()
    hook = next(m.parameters()).register_hook(lambda g:torch.full_like(g, float('inf')))
    try: log, _ = B.main_step(m, o, s, state, cache, order, device)
    finally: hook.remove()
    assert log['AMP_skip'] and before == digest((m.state_dict(), o.state_dict()['state'])) and s.get_scale() == scale/2
    report['AMP_overflow_skip'] = True
    report['AMP_overflow_detail'] = dict(skipped=log['AMP_skip'], scale_before=scale, scale_after=s.get_scale(),
                                       group_lr_after=o.param_groups[0]['lr'], weight_moments_unchanged=True)
    write(root/'gpu-smoke.json', report)
    point = L.save_point(root, 'v7', m, spec, ident, {})
    del m, o, s; R.release(device)
    import final_inference_v7 as V
    import s3_representation as S
    m = V.load(point, device); tok, _ = S.tokenizer(Path(inputs)/'tokenizer.json', 30)
    rows = sets['heldout4k'][:32]; src = C.source_batch(tok, [r['question'] for r in rows], device)
    with torch.no_grad():
        C.sync(device); begin = time.perf_counter()
        ctx = V.Fast.Context(m, src, 128)
        tokens, _, _ = V.Speed.decode(ctx, k=1, deadline=min(cfg['deadline_monotonic']-120, time.monotonic()+120),
                                       options=V.DECODE_OPTIONS)
        C.sync(device); seconds_gen = time.perf_counter()-begin
    report['v7_generation'] = dict(seconds=seconds_gen, rows=len(rows), sequences_per_second=len(rows)/seconds_gen,
                                  cap=128, full180s_gate=False, decoder='actual v7 Speed.decode',
                                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    del m, src, ctx, tokens; R.release(device)
    # Fresh short run uses production code, with two branches and a serialized session boundary.
    short = dict(cfg, main_seconds=1.e8, max_updates=12, probe_cooldown_seconds=3., probe_eval_seconds=120.,
                 final_cooldown_seconds=3., final_eval_seconds=180., final_session=False, final_q=0.,
                 io_reserve_seconds=15.)
    def evaluate(p, out, deadline, final): return E.evaluate(p, out, inputs, sets, refs, device, deadline, final, True)
    # Force an early probe via a valid saved starting state, then continue from it.
    p = B.read_full(root, ident); p['state']['next_point_main_seconds'] = p['state']['main_seconds']
    m, o, s, state = B.restore(p, device); B.save_full(root, m, o, s, state, spec, ident)
    for _ in range(12): B.main_step(m, o, s, state, cache, order, device)
    control = digest((m.state_dict(), o.state_dict(), s.state_dict(), T.rng_state()))
    del m, o, s, p; R.release(device)
    result = B.run(name, cache, order, meta, root, device, short, evaluate, resume=root)
    assert result['status'] == 'partial; main target or evaluation incomplete'
    assert result['state']['updates'] == 268
    assert any(p['complete'] and not p['training']['final'] for p in result['state']['points'])
    # Compare the actual branch/restore/continuation to the no-evaluation control.
    m, o, s, state = B.restore(B.read_full(root, ident), device)
    assert control == digest((m.state_dict(), o.state_dict(), s.state_dict(), T.rng_state()))
    report['eval_restore_next_12_updates_bit_exact'] = True
    del m, o, s; R.release(device)
    end = B.run(name, cache, order, meta, root, device,
                dict(short, main_seconds=.000001, max_updates=1), evaluate, resume=root)
    assert end['status'] == 'complete' and end['end_main_complete']
    assert end['state']['updates'] == 269
    result = end
    stable = B.read_full(root, ident); del stable
    report.update(passed=True, status='complete', stable_bytes=(root/json.loads((root/'latest.json').read_text())['file']).stat().st_size, result=result)
    write(root/'summary.json', report); write(root/'gpu-smoke.json', report)
