"""V6: adaptive mixed L3 inference on one T4; no targets in decisions/selection."""
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
import final_speed as Speed
from dataclasses import asdict

DECODE_OPTIONS = Speed.Options(share_cross=True, async_eos=True, eos_every=32,
    compact_done=True, race_sampler=True, lean_greedy=True)
from final_representation import tokenizer, restore_compact
from final_executor_v6 import Executor, verify_winners
from final_vote import write_answers
from final_adaptive import parameters, question_batch, stage1_decision, stage1_leader_fraction, select_available

WEIGHT_SHA = {
    'C-main': '8c11b1038855422b6912e243049a554317978f848edb412ed48aebbe4cb6b886',
    'C-long2x': 'd10bf0018d0b14fd6cc3c8b9c7121f127b5cec97057544fd63e5824253a55057'}


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
    k1, k2, unanimity_threshold = parameters(config)
    lanes = config.get('sequence_budget', config.get('sample_lanes', 1024))
    stage1_batch = question_batch(lanes, k1)
    extra = k2-k1
    extra_batch = question_batch(lanes, extra) if extra else 0
    greedy_lanes = config.get('greedy_sequence_budget', config.get('greedy_batch', lanes))
    greedy_batch = question_batch(greedy_lanes, 1)
    scorer = None
    if config.get('scorer_enabled', False):
        import importlib
        module, attribute = config['scorer_callable'].split(':')
        scorer = getattr(importlib.import_module(module), attribute)
    phases = {'imports_and_launcher': time.monotonic()-PROCESS_START}
    out = Path(config['output']); out.mkdir(parents=True, exist_ok=True)
    budget = min(180., float(config.get('budget_seconds', 180.)))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('Require a positive finite budget (maximum 180 seconds)')
    # GPU/CPU work stops well before the external hard stop, leaving voting/write reserve.
    work_deadline = PROCESS_START + budget - min(20., budget*.12)
    execution_deadline = PROCESS_START + budget - min(14., budget*.08)
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
    model_hashes = {name: sha(path) for name, path in config['models'].items() if Path(path).is_file()}
    if config.get('verify_weights', True):
        for name in config['models']:
            assert model_hashes[name] == config.get('weight_sha256', WEIGHT_SHA)[name], name
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
        Speed.decode(ctx, options=DECODE_OPTIONS)
        del ctx, src
    C.sync(device)
    phases['warmup'] = time.monotonic()-started
    generation_workers = 2
    tail_workers = max(2, min(4, os.cpu_count() or 1))
    executor = Executor(generation_workers)
    records = [{'greedy': None, 'main': [], 'long': []} for _ in rows]
    guard_events = []
    cap = int(config.get('cap', 512))
    restore_seconds = 0.
    attempts = []
    batch_stats = []
    stopped_rows, escalated_rows = [], []
    stage2_waves = []
    model_rng = {}
    effective_batches = {}
    stage2_order = []
    leader_fractions = {}

    def generate(name, order, batch, k, temperature, target, deadline, stage=0):
        nonlocal restore_seconds
        batch = min(batch, effective_batches.get((name, k), batch))
        offset = 0
        while offset < len(order):
            C.check_deadline(deadline)
            selected = order[offset:offset+batch]
            src = ctx = seq = means = None
            rng_cpu = torch.get_rng_state() if temperature is not None else None
            rng_gpu = torch.cuda.get_rng_state(device) if temperature is not None and device.type == 'cuda' else None
            had_oom = False
            batch_started = time.monotonic()
            try:
                src = nn.utils.rnn.pad_sequence([torch.tensor(encoded[i]) for i in selected],
                    batch_first=True, padding_value=C.PAD).to(device)
                ctx = Fast.Context(models[name], src, cap)
                decode_stats = {}
                seq, means, _ = Speed.decode(ctx, k, temperature, deadline=deadline,
                    compute_logprob=temperature is not None, options=DECODE_OPTIONS, stats=decode_stats)
                begin = time.monotonic()
                groups = pack(seq, means, table, greedy=temperature is None)
                restore_seconds += time.monotonic()-begin
                for i, group in zip(selected, groups):
                    if target == 'greedy':
                        records[i][target] = group[0]
                    else:
                        records[i][target].extend(group)
                executor.submit([c for group in groups for c in group])
                batch_stats.append({'phase': target, 'stage': stage, 'questions': len(selected), 'sequences': len(selected)*k,
                                    'row_ids': selected, 'source_tokens': src.shape[1], 'decode_steps': seq.shape[-1],
                                    'decoder': decode_stats, 'wall_seconds': time.monotonic()-batch_started})
                offset += len(selected)
                if (device.type == 'cpu' and target != 'greedy' and
                    config.get('cpu_test_expire_after_sample_batches') ==
                    sum(s['phase'] != 'greedy' for s in batch_stats)):
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
                effective_batches[name, k] = batch
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
        generate('C-main', greedy_order, greedy_batch, 1, None, 'greedy', greedy_deadline)
        phases['greedy_generation_restore_submit'] = time.monotonic()-started
        # Completed before sampling: the independent supervisor can publish it after a hard kill.
        started = time.monotonic()
        write_answers(out/'greedy-fallback.jsonl', [r['greedy']['code'] for r in records])
        write(out/'greedy-ready.json', {'rows': len(rows), 'seconds': time.monotonic()-PROCESS_START})
        phases['greedy_checkpoint_write'] = time.monotonic()-started
        order = greedy_order
        for name, target in [('C-main', 'main'), ('C-long2x', 'long')]:
            started = time.monotonic()
            torch.manual_seed(270042)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(270042)
            try:
                generate(name, order, stage1_batch, k1, .7, target, work_deadline, stage=1)
            except C.BudgetExpired:
                guard_events.append({'phase': target, 'elapsed_seconds': time.monotonic()-PROCESS_START})
            model_rng[name] = (torch.get_rng_state(),
                              torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
            phases['stage1_'+target+'_sampling_restore_submit'] = time.monotonic()-started
            if guard_events:
                break
        # Global priority requires all stage-one evidence, including the slowest
        # CPU tasks. Execution still overlaps greedy and both stage-one GPU phases.
        pending = set(order)
        phases['stage1_decision_wait'] = 0.
        phases['stage2_sampling_restore_submit'] = 0.
        try:
            if not guard_events:
                begin = time.monotonic()
                try:
                    if not executor.drain(work_deadline):
                        raise C.BudgetExpired('Stage-one executor unavailable')
                    for i in order:
                        C.check_deadline(work_deadline)
                        record = records[i]
                        executor.attach([record['greedy']]+record['main'][:k1]+record['long'][:k1])
                        decision = stage1_decision(record, k1, unanimity_threshold)
                        if decision is None:
                            raise C.BudgetExpired('Incomplete stage-one execution')
                        pending.remove(i)
                        if decision:
                            stopped_rows.append(i)
                        else:
                            escalated_rows.append(i)
                            leader_fractions[i] = stage1_leader_fraction(record, k1)
                    stage2_order = sorted(escalated_rows, key=lambda i: (leader_fractions[i], len(encoded[i]), i))
                finally:
                    phases['stage1_decision_wait'] += time.monotonic()-begin
                # Complete both models for a priority batch before touching the
                # next one, so a guard cannot spend everything on only C-main.
                for offset in range(0, len(stage2_order), extra_batch) if extra else []:
                    C.check_deadline(work_deadline)
                    ready = stage2_order[offset:offset+extra_batch]
                    stage2_waves.append({'questions': len(ready), 'pending_stage1_rows': 0,
                        'row_ids': ready, 'leader_fraction_min': leader_fractions[ready[0]],
                        'leader_fraction_max': leader_fractions[ready[-1]]})
                    begin = time.monotonic()
                    try:
                        for name, target in [('C-main', 'main'), ('C-long2x', 'long')]:
                            cpu_rng, cuda_rng = model_rng[name]
                            torch.set_rng_state(cpu_rng)
                            if cuda_rng is not None:
                                torch.cuda.set_rng_state(cuda_rng, device)
                            generate(name, ready, extra_batch, extra, .7, target, work_deadline, stage=2)
                            model_rng[name] = (torch.get_rng_state(),
                                torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)
                    finally:
                        phases['stage2_sampling_restore_submit'] += time.monotonic()-begin
        except C.BudgetExpired:
            guard_events.append({'phase': 'adaptive_stage2', 'elapsed_seconds': time.monotonic()-PROCESS_START})
        started = time.monotonic()
        # Preserve the same queue/cache and in-flight tasks; only add idle slots.
        executor.resize(tail_workers)
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
        try:
            winner, count = select_available(r, k2, question=rows[i]['question'], scorer=scorer,
                score_weight=float(config.get('scorer_weight', 0.)), deadline=verification_deadline)
        except C.BudgetExpired:
            winner, count = select_available(r, k2)
            guard_events.append({'phase': 'scorer', 'row': i})
        winners.append(winner)
        executed_sample_counts.append(count)
        if count == 0:
            fallback.append(i)
        elif count < (2*k1 if i in stopped_rows else 2*k2):
            partial_vote_rows.append(i)
    phases['vote'] = time.monotonic()-started
    started = time.monotonic()
    verification = verify_winners(winners, verification_deadline, tail_workers)
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
              'adaptive': {'K1': k1, 'K2': k2, 'unanimity_threshold': unanimity_threshold,
                           'stopped_rows': sorted(stopped_rows), 'escalated_rows': sorted(escalated_rows),
                           'undecided_rows': sorted(pending), 'stage2_waves': stage2_waves,
                           'stage2_order': stage2_order, 'stage2_priority': 'stage1_leader_fraction_then_length_then_id',
                           'stage1_leader_fractions': leader_fractions,
                           'mean_sample_count': sum(len(r['main'])+len(r['long']) for r in records)/len(rows),
                           'scorer_enabled': config.get('scorer_enabled', False)},
              'greedy_batch': greedy_batch, 'greedy_sequence_budget': greedy_lanes,
              'sample_lanes': lanes, 'sequence_budget': lanes,
              'length_bucketing': True, 'generation_batches': batch_stats,
              'decoder_options': asdict(DECODE_OPTIONS),
              'executor_policy': {'generation_workers': generation_workers, 'tail_workers': tail_workers,
                                  'nice_increment': 10, 'resize_preserves_queue_and_cache': True},
              'model_sha256': model_hashes,
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
                                 'match_indices': matches, 'expected_matches': config.get('expected_matches', 1511),
                                 'delta_pp': 100*(len(matches)-config.get('expected_matches', 1511))/len(winners),
                                 'tolerance_pp': 1.,
                                 'passed': abs(100*(len(matches)-config.get('expected_matches', 1511))/len(winners)) <= 1. and not guard_events})
        import gzip
        with gzip.open(out/'candidate-journal.json.gz', 'wt', encoding='utf8') as stream:
            json.dump(records, stream, ensure_ascii=False)
    return timing


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    run(json.loads(Path(parser.parse_args().config).read_text()))
