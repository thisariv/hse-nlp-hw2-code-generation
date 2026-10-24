"""Bounded waves of v6 greedy + overlapped CPU execution; no generated targets."""
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import final_core as C
import final_fast as Fast
import final_speed as Speed
from final_representation import restore_compact
from final_executor_v6 import Executor
from d1_data import save_npz, publish, verify_receipt

OPTIONS = Speed.Options(share_cross=True, async_eos=True, eos_every=32,
                        compact_done=True, race_sampler=True, lean_greedy=True)


def decode_code(tokens, table):
    active = C.active_tokens(tokens)
    represented = C.decode_bytes(table, active).decode('utf8', errors='replace')
    try:
        return restore_compact(represented), ('eos' if C.EOS in active else 'token_cap'), len(active)
    except (ValueError, IndexError):
        return '', 'restore_error', len(active)


def reference_for(cache, row, table, evidence):
    pos = int(np.searchsorted(evidence['row_id'], row))
    if pos < len(evidence['row_id']) and int(evidence['row_id'][pos]) == row:
        return None, {'status': 'numeric', 'value': float(evidence['reference'][pos])}, 'receipt'
    text = C.decode_bytes(table, cache[row][1]).decode('utf8')
    try:
        return restore_compact(text), None, 'executed'
    except (ValueError, IndexError):
        return None, {'status': 'reference_restore_error'}, 'restore_error'


@torch.inference_mode()
def generation_wave(model, cache, ids, table, device, root, check, execution_deadline,
                    evidence=None, batch=256, cap=512, executor_factory=Executor, generation_deadline=None):
    """At most one wave of codes/results resident. Submit after every GPU batch.

    Two CPU workers per GPU = four total, including main's missing references.
    No per-program timeout can grow the queue beyond the current wave.
    """
    root = Path(root); root.parent.mkdir(parents=True, exist_ok=True)
    receipt = root.with_suffix('.receipt.json')
    ids = np.asarray(ids, dtype=np.uint32)
    if receipt.exists():
        meta = verify_receipt(receipt)
        with np.load(root, allow_pickle=False) as old: assert np.array_equal(old['row_id'], ids)
        return meta
    started = time.monotonic(); model.eval()
    order = sorted(map(int, ids), key=lambda i: (int(cache.source_lengths[i]), i))
    candidates = {}; references = {}; sources = {}; gen_status = {}; gen_length = {}; offset = 0; oom = 0
    executor = executor_factory(2); interrupted = None; gpu_seconds = 0.
    try:
        while offset < len(order):
            try: check()
            except TimeoutError: interrupted = 'phase deadline'; break
            take = order[offset:offset+batch]
            src = ctx = sequences = means = None
            try:
                src = nn.utils.rnn.pad_sequence([torch.as_tensor(np.asarray(cache[i][0]).copy(), dtype=torch.long)
                    for i in take], batch_first=True, padding_value=C.PAD).to(device)
                ctx = Fast.Context(model, src, cap)
                gpu_started = time.monotonic()
                sequences, means, _ = Speed.decode(ctx, deadline=generation_deadline or execution_deadline,
                    compute_logprob=False, options=OPTIONS)
                if device.type == 'cuda': torch.cuda.synchronize(device)
                gpu_seconds += time.monotonic()-gpu_started
                jobs = []
                for i, tokens in zip(take, sequences[:, 0].tolist()):
                    code, status, length = decode_code(tokens, table)
                    gen_status[i], gen_length[i] = status, length
                    candidates[i] = {'code': code}
                    if status != 'restore_error': jobs.append(candidates[i])
                    else: candidates[i]['execution'] = {'status': 'restore_error'}
                    if evidence is not None:
                        refcode, result, origin = reference_for(cache, i, table, evidence)
                        sources[i] = origin; references[i] = {'code': refcode}
                        if result is None: jobs.append(references[i])
                        else: references[i]['execution'] = result
                executor.submit(jobs)
                offset += len(take)
            except TimeoutError:
                interrupted = 'phase deadline'; break
            except torch.cuda.OutOfMemoryError:
                oom += 1; batch //= 2
                if batch < 1: raise
            except Exception as error:
                interrupted = type(error).__name__; break
            finally:
                del src, ctx, sequences, means
            if oom and device.type == 'cuda':
                import gc
                gc.collect(); torch.cuda.empty_cache()
        # Publish every completed GPU batch even if a later batch hit its guard.
        ids = np.asarray([i for i in ids if int(i) in gen_status], dtype=np.uint32)
        executor.drain(execution_deadline)
        for c in [*candidates.values(), *references.values()]:
            if 'execution' not in c:
                try: executor.attach([c])
                except KeyError: c['execution'] = {'status': 'budget_not_evaluated'}
        def values(items):
            out = np.asarray([items[int(i)]['execution'].get('value', np.nan) for i in ids], dtype=np.float64)
            status = np.asarray([items[int(i)]['execution']['status'] for i in ids], dtype='U48')
            # A service crash/nonfinite answer cannot masquerade as a numeric disagreement.
            status[(status == 'numeric') & ~np.isfinite(out)] = 'nonfinite'
            return out, status
        answer, status = values(candidates)
        arrays = dict(row_id=ids, answer=answer, status=status,
                      generation_status=np.asarray([gen_status[int(i)] for i in ids], dtype='U16'),
                      generation_length=np.asarray([gen_length[int(i)] for i in ids], dtype=np.uint16))
        if evidence is not None:
            arrays['reference'], arrays['reference_status'] = values(references)
            arrays['reference_source'] = np.asarray([sources[int(i)] for i in ids], dtype='U16')
        execution = executor.summary()
        save_npz(root, **arrays)
        meta = dict(phase='generation', rows=len(ids), seconds=time.monotonic()-started,
                    execution=execution, OOM_retries=oom, cap=cap, effective_batch=batch,
                    complete=interrupted is None and len(ids) == len(order) and
                        bool(np.all(status != 'budget_not_evaluated')) and
                        (evidence is None or bool(np.all(arrays['reference_status'] != 'budget_not_evaluated'))),
                    interruption=interrupted, target_rows=len(order), GPU_seconds=gpu_seconds,
                    GPU_rows_per_second=len(ids)/max(gpu_seconds, 1e-9),
                    pipeline_rows_per_second=len(ids)/max(time.monotonic()-started, 1e-9),
                    reference_reused=sum(s == 'receipt' for s in sources.values()),
                    reference_executed=sum(s == 'executed' for s in sources.values()))
        publish(receipt, [root], **meta)
        return meta
    finally:
        executor.close()
