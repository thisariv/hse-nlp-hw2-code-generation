"""Cold mixed C-main/C-long2x inference on one GPU; no training or targets in selection."""
import os
import time
PROCESS_START = float(os.environ.get('FINAL_PROCESS_START', time.monotonic()))
import sys
import signal
# Arm before Torch import: session watchdog can kill the supervisor and its entire tree.
if __name__ == '__main__' and sys.platform.startswith('linux'):
    import ctypes
    _parent = os.getppid()
    signal.signal(signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
    assert ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0, 0, 0) == 0
    if _parent == 1 or os.getppid() != _parent:
        os.killpg(os.getpgrp(), signal.SIGKILL)
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
import final_core as C
import final_fast as Fast
from final_representation import tokenizer, restore_compact
from final_executor import Executor
from final_vote import select_available, write_answers, verify_winners

WEIGHT_SHA = {
    'C-main': '3806a5827004fd88f533a7307abc0c1e387faafc5672f4224392b363c4d2ad3d',
    'C-long2x': '6227ee2fc8dd12e186137be229fef2b1ceba659848c35bedd90cec6502f41548'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def load(path, device):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['format'] == 'nlp-hw02-model-only-v1' and payload['optimizer_included'] is False
    model = C.SimpleTransformer(**payload['model_spec'], dropout=0.).eval()
    if device.type == 'cuda':
        model.half()
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device)


def pack(ids, lps, table, greedy=False):
    groups = []
    for sequences, means in zip(ids.tolist(), lps.tolist()):
        group = []
        for tokens, lp in zip(sequences, means):
            active = C.active_tokens(tokens)
            represented = C.decode_bytes(table, active).decode('utf8', errors='replace')
            try:
                code, error = restore_compact(represented), None
            except (ValueError, IndexError):
                code, error = '', 'invalid representation'
            group.append({'code': code, 'mean_logprob': 0. if greedy else (lp if math.isfinite(lp) else None),
                          'represented_code': represented, 'restore_error': error, 'length': len(active)})
        groups.append(group)
    return groups


def run(config):
    phases = {'imports_and_launcher': time.monotonic()-PROCESS_START}
    out = Path(config['output']); out.mkdir(parents=True, exist_ok=True)
    budget = float(config.get('budget_seconds', 180.))
    # GPU/CPU work stops well before the external hard stop, leaving voting/write reserve.
    work_deadline = PROCESS_START + budget - min(40., budget*.23)
    execution_deadline = PROCESS_START + budget - min(28., budget*.16)
    verification_deadline = PROCESS_START + budget - min(10., budget*.06)
    greedy_deadline = PROCESS_START + budget - min(50., budget*.28)
    started = time.monotonic()
    rows = json.loads(Path(config['rows']).read_text())
    phases['read_input'] = time.monotonic()-started
    assert rows and all(isinstance(r['question'], str) for r in rows)
    if config.get('expected_rows') is not None:
        assert len(rows) == config['expected_rows']
    device = torch.device(config.get('device', 'cuda:0'))
    if device.type == 'cpu' and config.get('cpu_test_expire_sampling'):
        work_deadline = PROCESS_START  # Deterministic CPU-only guard fixture.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    # The decoder writes each KV slot before reading it; avoid filling unused slots.
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == 'cuda':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.device_count() == 1
        assert 'T4' in torch.cuda.get_device_name(0)
        assert str(torch.__version__) == '2.11.0+cu128' and torch.version.cuda == '12.8', 'Pinned L2 runtime required'
    started = time.monotonic()
    if config.get('verify_weights', True):
        for name, path in config['models'].items():
            assert sha(path) == WEIGHT_SHA[name], name
    if config.get('tokenizer_sha'):
        assert sha(config['tokenizer']) == config['tokenizer_sha']
    models = {name: load(path, device) for name, path in config['models'].items()}
    tok, table = tokenizer(config['tokenizer'], 30)
    encoded = [[C.BOS]+C.encode(tok, r['question'])+[C.EOS] for r in rows]
    phases['load_models_tokenizer_encode'] = time.monotonic()-started
    started = time.monotonic()
    # No stochastic warmup; seed is reset separately for each model afterwards.
    for model in models.values():
        src = torch.tensor([encoded[0]], device=device)
        ctx = Fast.Context(model, src, 16)
        Fast.decode(ctx)
        del ctx, src
    C.sync(device)
    phases['warmup'] = time.monotonic()-started
    executor = Executor(config.get('workers'))
    records = [{'greedy': None, 'main': [], 'long': []} for _ in rows]
    guard_events = []
    cap = int(config.get('cap', 512))
    restore_seconds = 0.
    attempts = []
    batch_stats = []

    def generate(name, order, batch, k, temperature, target, deadline):
        nonlocal restore_seconds
        offset = 0
        while offset < len(order):
            C.check_deadline(deadline)
            selected = order[offset:offset+batch]
            src = ctx = seq = means = None
            rng_cpu = torch.get_rng_state() if temperature is not None else None
            rng_gpu = torch.cuda.get_rng_state(device) if temperature is not None and device.type == 'cuda' else None
            had_oom = False
            try:
                src = nn.utils.rnn.pad_sequence([torch.tensor(encoded[i]) for i in selected],
                    batch_first=True, padding_value=C.PAD).to(device)
                ctx = Fast.Context(models[name], src, cap)
                seq, means, _ = Fast.decode(ctx, k, temperature, deadline=deadline,
                                           compute_logprob=temperature is not None)
                begin = time.monotonic()
                groups = pack(seq, means, table, greedy=temperature is None)
                restore_seconds += time.monotonic()-begin
                for i, group in zip(selected, groups):
                    records[i][target] = group[0] if target == 'greedy' else group[:4]
                executor.submit([c for group in groups for c in group[:(1 if target == 'greedy' else 4)]])
                batch_stats.append({'phase': target, 'questions': len(selected), 'sequences': len(selected)*k,
                                    'source_tokens': src.shape[1], 'decode_steps': seq.shape[-1]})
                offset += len(selected)
                if (device.type == 'cpu' and target == 'main' and
                    config.get('cpu_test_expire_after_main_batches') == sum(s['phase'] == 'main' for s in batch_stats)):
                    raise C.BudgetExpired('CPU partial-candidate guard fixture')
            except torch.cuda.OutOfMemoryError:
                had_oom = True
                if rng_cpu is not None:
                    torch.set_rng_state(rng_cpu)
                if rng_gpu is not None:
                    torch.cuda.set_rng_state(rng_gpu, device)
                attempts.append({'model': name, 'offset': offset, 'question_batch': batch, 'K': k})
                batch //= 2
                if batch < 1:
                    raise
            finally:
                del src, ctx, seq, means
                if had_oom:
                    # Only OOM requires collection. Normal batches reuse the CUDA allocator.
                    import gc
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

    try:
        started = time.monotonic()
        greedy_order = sorted(range(len(rows)), key=lambda i: (len(encoded[i]), i))
        generate('C-main', greedy_order, config.get('greedy_batch', 1024), 1, None, 'greedy', greedy_deadline)
        phases['greedy_generation_restore_submit'] = time.monotonic()-started
        # Completed before sampling: the independent supervisor can publish it after a hard kill.
        started = time.monotonic()
        write_answers(out/'greedy-fallback.jsonl', [r['greedy']['code'] for r in records])
        write(out/'greedy-ready.json', {'rows': len(rows), 'seconds': time.monotonic()-PROCESS_START})
        phases['greedy_checkpoint_write'] = time.monotonic()-started
        replay = config.get('l2_replay', False)
        # Length buckets reduce source padding; writeback always uses the original row ID.
        # Only the diagnostic L2 replay retains its original order and batch32/K8 RNG lanes.
        order = list(range(len(rows))) if replay else greedy_order
        for name, target in [('C-main', 'main'), ('C-long2x', 'long')]:
            started = time.monotonic()
            torch.manual_seed(270042)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(270042)
            try:
                generate(name, order, 32 if replay else config.get('sample_batch', 256),
                         8 if replay else 4, .7, target, work_deadline)
            except C.BudgetExpired:
                guard_events.append({'phase': target, 'elapsed_seconds': time.monotonic()-PROCESS_START})
            phases[target+'_sampling_restore_submit'] = time.monotonic()-started
            if guard_events:
                break
        started = time.monotonic()
        drained = executor.drain(execution_deadline)
        phases['execution_tail_wait'] = time.monotonic()-started
        if not drained:
            guard_events.append({'phase': 'execution', 'elapsed_seconds': time.monotonic()-PROCESS_START})
        candidates = [c for r in records for c in [r['greedy']]+r['main']+r['long']]
        executor.attach(candidates)
        execution = executor.summary()
    finally:
        started = time.monotonic()
        executor.close()
        phases['executor_shutdown'] = time.monotonic()-started
    started = time.monotonic()
    winners = []
    fallback = []
    partial_vote_rows = []
    executed_sample_counts = []
    for i, r in enumerate(records):
        winner, count = select_available(r)
        winners.append(winner)
        executed_sample_counts.append(count)
        if count == 0:
            fallback.append(i)
        elif count < 8:
            partial_vote_rows.append(i)
    phases['vote'] = time.monotonic()-started
    started = time.monotonic()
    verification = verify_winners(winners, verification_deadline, config.get('workers'))
    for i in verification['mismatch_rows']:
        winners[i] = records[i]['greedy']
        if i not in fallback:
            fallback.append(i)
    if verification['mismatch_rows']:
        guard_events.append({'phase': 'winner_reexecution', 'rows': verification['mismatch_rows'],
                             'elapsed_seconds': time.monotonic()-PROCESS_START})
    phases['winner_reexecution'] = time.monotonic()-started
    started = time.monotonic()
    write_answers(out/'answers.jsonl', [w['code'] for w in winners])
    closed = time.monotonic()
    phases['answer_write'] = closed-started
    phases['bookkeeping_and_executor_start'] = closed-PROCESS_START-sum(phases.values())
    timing = {'full_pipeline_seconds': closed-PROCESS_START, 'rows': len(rows), 'phases_seconds': phases,
              'restore_seconds_in_generation': restore_seconds, 'execution': execution,
              'deadline_guard': {'triggered': bool(guard_events), 'events': guard_events,
                                 'fallback_rows': sorted(fallback), 'fallback_count': len(fallback)},
              'winner_reexecution': verification,
              'no_numeric_winner_rows': [i for i, w in enumerate(winners) if w['execution']['status'] != 'numeric'],
              'input_sha256': sha(config['rows']),
              'question_order_sha256': hashlib.sha256(json.dumps(
                  [r['question'] for r in rows], ensure_ascii=False).encode()).hexdigest(),
              'answers_sha256': sha(out/'answers.jsonl'),
              'OOM_attempts': attempts, 'single_gpu': device.type == 'cuda', 'seed': 270042,
              'sample_K': 8 if config.get('l2_replay') else 4, 'used_K_per_model': 4,
              'greedy_batch': config.get('greedy_batch', 1024),
              'sampling_batch': 32 if config.get('l2_replay') else config.get('sample_batch', 256),
              'length_bucketing': not config.get('l2_replay', False), 'generation_batches': batch_stats,
              'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'deterministic_algorithms': True, 'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
              'timer': 'launcher process start through answers file close; imports/load/warmup/exec/vote/write included'}
    timing['deadline_guard'].update(partial_vote_rows=partial_vote_rows,
        partial_vote_count=len(partial_vote_rows), no_executed_sample_count=executed_sample_counts.count(0),
        executed_sample_counts=executed_sample_counts,
        generation_stop_seconds=work_deadline-PROCESS_START, execution_stop_seconds=execution_deadline-PROCESS_START,
        verification_stop_seconds=verification_deadline-PROCESS_START)
    write(out/'timing.json', timing)
    # Diagnostic outputs and gold scoring are outside the answer timer, no gold is passed to select.
    if config.get('reference'):
        reference = json.loads(Path(config['reference']).read_text())
        assert len(reference) == len(winners)
        matches = [i for i, (ref, win) in enumerate(zip(reference, winners)) if
                   ref['status'] == win['execution']['status'] == 'numeric' and math.isclose(
                       ref['value'], win['execution']['value'], rel_tol=1e-6, abs_tol=1e-6)]
        write(out/'quality.json', {'matches': len(matches), 'n': len(winners), 'accuracy_pp': 100*len(matches)/len(winners),
                                 'match_indices': matches, 'expected_matches': 1492,
                                 'delta_pp': 100*(len(matches)-1492)/len(winners),
                                 'tolerance_pp': 1.,
                                 'passed': abs(100*(len(matches)-1492)/len(winners)) <= 1. and not guard_events})
        import gzip
        with gzip.open(out/'candidate-journal.json.gz', 'wt', encoding='utf8') as stream:
            json.dump(records, stream, ensure_ascii=False)
    return timing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    run(json.loads(Path(parser.parse_args().config).read_text()))
