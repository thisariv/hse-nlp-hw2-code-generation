"""One bounded T4 session: paired decoder timings, full greedy gate, short traces."""
import argparse
import csv
from dataclasses import asdict
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile
import torch
from torch import nn
import final_core as C
import final_fast as Fast
import final_speed as Speed
from final_representation import tokenizer

INPUT_SHA = {'tokenizer.json': '261f25b5d4be006a6a00e41e0eafbe522580741da627ba670f6b9805a2659b7d',
             'test.json': '20cacb334e3fc81291433d2173d9085b17a15e1834bd84aea67b350403182503'}
WEIGHT_SHA = '8c11b1038855422b6912e243049a554317978f848edb412ed48aebbe4cb6b886'
VARIANTS = {
    'baseline': {},
    'share_cross': {'share_cross': True},
    'async_eos': {'async_eos': True, 'eos_every': 32},
    'compact_done': {'compact_done': True},
    'race_sampler': {'race_sampler': True},
    'lean_greedy': {'lean_greedy': True},
    'workers2': {},
    'nice10': {},
    'all': {'share_cross': True, 'async_eos': True, 'eos_every': 32,
            'compact_done': True, 'race_sampler': True, 'lean_greedy': True}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def unique(root, name):
    paths = list(Path(root).rglob(name))
    assert len(paths) == 1, f'Missing or ambiguous {name}: {paths}'
    return paths[0]


def prepare(root, target):
    result = {}
    for name, expected in INPUT_SHA.items():
        direct = list(Path(root).rglob(name))
        if direct:
            assert len(direct) == 1
            path = direct[0]
        else:
            archive = unique(root, '26b-training-inputs-d0-initial-and-evaluation.zip')
            with zipfile.ZipFile(archive) as stream:
                names = [p for p in stream.namelist() if Path(p).name == name]
                assert len(names) == 1
                path = target/name
                path.write_bytes(stream.read(names[0]))
        assert sha(path) == expected, name
        result[name] = path
    return result


def load_model(path, device):
    assert sha(path) == WEIGHT_SHA, 'Require frozen L3 C-main l2-62280'
    payload = torch.load(path, weights_only=True, map_location='cpu')
    assert payload['format'] == 'nlp-hw02-model-only-v1' and not payload['optimizer_included']
    model = C.SimpleTransformer(**payload['model_spec'], dropout=0.).eval().half()
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device)


class Load:
    """Separate Torch-free driver; service owns and cleans its worker process group."""
    def __init__(self, directory, workers, nice):
        self.directory, self.workers, self.nice = Path(directory), workers, nice
        self.child = None
        self.info = {'nice': 0, 'nice_applied': True}
        self.log = None

    def __enter__(self):
        if not self.workers:
            return self
        self.directory.mkdir(parents=True, exist_ok=True)
        for name in ('ready.json', 'load-summary.json'):
            (self.directory/name).unlink(missing_ok=True)
        self.log = (self.directory/'driver.log').open('w')
        self.child = subprocess.Popen([sys.executable, str(Path(__file__).with_name('final_profile_load.py')),
            '--workers', str(self.workers), '--nice', str(self.nice), '--directory', str(self.directory)],
            start_new_session=True, stdout=self.log, stderr=subprocess.STDOUT)
        until = time.monotonic()+30
        try:
            while not (self.directory/'ready.json').exists():
                if self.child.poll() is not None or time.monotonic() >= until:
                    raise RuntimeError('Execution load startup failed')
                time.sleep(.02)
            self.info = json.loads((self.directory/'ready.json').read_text())
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def alive(self):
        if self.child is not None and self.child.poll() is not None:
            raise RuntimeError('Execution load stopped during timing')

    def __exit__(self, *_):
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=2.)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait(timeout=2.)
        if self.log is not None:
            self.log.close()


def canonical(ids, cap):
    result = torch.full((*ids.shape[:-1], cap), C.PAD, dtype=torch.long)
    result[..., :ids.shape[-1]] = ids
    return result


@torch.inference_mode()
def measure(model, encoded, order, variant, k, device, deadline, cap=512, sequence_budget=1024):
    """Same order, capacity and initial batch shapes for every paired comparison."""
    options = Speed.Options(**VARIANTS[variant])
    question_batch = sequence_budget//k
    predictions, batches = {}, []
    torch.manual_seed(270042)
    torch.cuda.manual_seed_all(270042)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    gpu0, gpu1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    gpu0.record()
    begin, cpu = time.monotonic(), time.process_time()
    for offset in range(0, len(order), question_batch):
        C.check_deadline(deadline)
        selected = order[offset:offset+question_batch]
        b = time.monotonic()
        src = nn.utils.rnn.pad_sequence([torch.tensor(encoded[i]) for i in selected],
                    batch_first=True, padding_value=C.PAD).to(device)
        context = Fast.Context(model, src, cap)
        stats = {}
        if variant in ('baseline', 'workers2', 'nice10'):
            ids, lps, _ = Fast.decode(context, k, None if k == 1 else .7,
                        deadline=deadline, compute_logprob=k != 1)
            stats['steps'] = ids.shape[-1]
        else:
            ids, lps, _ = Speed.decode(context, k, None if k == 1 else .7,
                        deadline=deadline, compute_logprob=k != 1, options=options, stats=stats)
        if k == 1:
            predictions.update(zip(selected, canonical(ids, cap)[:, 0]))
        batches.append({'questions': len(selected), 'sequences': len(selected)*k,
                        'source_tokens': src.shape[1], 'seconds': time.monotonic()-b, **stats})
        del src, context, ids, lps
    gpu1.record()
    torch.cuda.synchronize(device)
    seconds = time.monotonic()-begin
    row = {'variant': variant, 'mode': 'greedy' if k == 1 else 'K4', 'questions': len(order),
           'sequences': len(order)*k, 'seconds': seconds, 'sequences_per_second': len(order)*k/seconds,
           'parent_cpu_seconds': time.process_time()-cpu, 'gpu_stream_span_seconds': gpu0.elapsed_time(gpu1)/1000,
           'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
           'peak_reserved_bytes': torch.cuda.max_memory_reserved(device), 'cap': cap,
           'sequence_budget': sequence_budget, 'options': asdict(options), 'batches': batches}
    # Event span includes GPU idle between host launches; it is NOT summed kernel time.
    if predictions:
        values = torch.stack([predictions[i] for i in order])
        row['greedy_token_sha256'] = hashlib.sha256(values.numpy().tobytes()).hexdigest()
    return row, predictions


def union_us(intervals):
    total, right = 0., float('-inf')
    for lo, hi in sorted(intervals):
        total += max(0., hi-max(lo, right))
        right = max(right, hi)
    return total


def trace_summary(trace):
    events = trace['traceEvents']
    window = next(e for e in events if e.get('name') == 'decode_window' and e.get('ph') == 'X')
    lo, hi = window['ts'], window['ts']+window['dur']
    def intervals(predicate):
        return [(max(lo, e['ts']), min(hi, e['ts']+e['dur'])) for e in events
            if e.get('ph') == 'X' and 'dur' in e and e['ts'] < hi and e['ts']+e['dur'] > lo and predicate(e)]
    kernels = intervals(lambda e: e.get('cat') == 'kernel')
    ops = intervals(lambda e: e.get('cat') == 'cpu_op' and e.get('tid') == window['tid'])
    syncs = intervals(lambda e: e.get('cat') == 'cuda_runtime' and
                     ('Synchronize' in e.get('name', '') or 'Memcpy' in e.get('name', '')))
    launches = intervals(lambda e: e.get('cat') == 'cuda_runtime' and 'Launch' in e.get('name', ''))
    span = hi-lo
    return {'window_us': span, 'kernel_count': len(kernels), 'kernel_busy_fraction': union_us(kernels)/span,
            'gpu_idle_fraction_upper_bound_for_host_overhead': 1-union_us(kernels)/span,
            'cpu_operator_fraction': union_us(ops)/span, 'sync_and_memcpy_api_fraction': union_us(syncs)/span,
            'launch_api_fraction': union_us(launches)/span,
            'host_outside_torch_ops_fraction': 1-union_us(ops)/span,
            'fractions_overlap': True, 'idle_is_not_a_direct_measure_of_python_cost': True,
            'sync_event_count': len(syncs)}


def profile(model, encoded, order, variant, device, output, deadline):
    C.check_deadline(deadline)
    selected = order[:256]  # K4 = 1024 lanes, same as timed sampling batches.
    src = nn.utils.rnn.pad_sequence([torch.tensor(encoded[i]) for i in selected],
                batch_first=True, padding_value=C.PAD).to(device)
    context = Fast.Context(model, src, 32)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, profile_memory=False, with_stack=False) as prof:
        with torch.profiler.record_function('decode_window'):
            if variant == 'baseline':
                Fast.decode(context, 4, .7)
            else:
                Speed.decode(context, 4, .7, options=Speed.Options(**VARIANTS[variant]))
    path = output.with_suffix('.json')
    prof.export_chrome_trace(str(path))
    trace = json.loads(path.read_text())
    summary = trace_summary(trace)
    summary['operators'] = [{'name': e.key, 'calls': e.count, 'self_cpu_us': e.self_cpu_time_total,
        'self_device_us': getattr(e, 'self_device_time_total', 0.)}
        for e in sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)[:25]]
    with gzip.open(output.with_suffix('.json.gz'), 'wt', compresslevel=6) as stream:
        json.dump(trace, stream, separators=(',', ':'))
    path.unlink()
    write(output.with_suffix('.summary.json'), summary)
    return summary


def stratified(order, count):
    if count >= len(order):
        return list(order)
    return [order[(j*(len(order)-1))//(count-1)] for j in range(count)]


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'running', 'training': False, 'voting': False, 'model': 'C-main',
              'rows': [], 'profiles': {}, 'greedy_checks': [], 'GPU_gate_passed': False,
              'matrix_scope': args.scope, 'ablation_questions': args.ablation_questions,
              'cuda_graphs': {'supported': False, 'reason': Speed.GRAPH_REFUSAL},
              'preallocated_kv': 'already in baseline; identity alias, no redundant timed run',
              'torch_threads': 'already 1 in baseline; identity alias',
              'RNG': 'seed 270042 per run; compaction/race_sampler may change stream, categorical law unchanged',
              'load': 'sustained synthetic numeric programs through real final_executor, not measured candidate mix',
              'timer': 'generation only incl. input transfer/context/encode/output D2H; excludes load start/stop, voting, restore',
              'session_start': args.start}
    def save():
        report['session_seconds'] = time.monotonic()-args.start
        full_checks = [r for r in report['greedy_checks'] if r['rows'] == 1919 and r['variant'] == 'all']
        report['full_greedy_bit_exact'] = (not any(r['mismatch_rows'] for r in full_checks)
                                          if len(full_checks) == 2 else None)
        comparisons = []
        for row in report['rows']:
            if row['variant'] == 'all' and row['scope'] == 'test1919':
                control = next(b for b in report['rows'] if b['variant'] == 'baseline' and
                    b['scope'] == row['scope'] and b['mode'] == row['mode'] and
                    bool(b['load_workers']) == bool(row['load_workers']))
                comparisons.append({'mode': row['mode'], 'loaded': bool(row['load_workers']),
                                    'speedup': control['seconds']/row['seconds']})
        report['full_speedups'] = comparisons
        report['target_2x_all_generation'] = (all(r['speedup'] >= 2 for r in comparisons)
                                            if len(comparisons) == 4 else None)
        write(out/'summary.json', report)
        if report['rows']:
            fields = ['variant', 'mode', 'scope', 'load_workers', 'nice', 'questions', 'sequences',
                'nice_requested_increment', 'nice_applied',
                'seconds', 'sequences_per_second', 'parent_cpu_seconds', 'gpu_stream_span_seconds',
                'peak_allocated_bytes', 'peak_reserved_bytes', 'greedy_mismatches']
            with (out/'timings.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(report['rows'])
    save()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda:0')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    assert torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0)
    assert str(torch.__version__) == '2.11.0+cu128' and torch.version.cuda == '12.8'
    report['runtime'] = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                         'device': torch.cuda.get_device_name(0), 'cpu_count': os.cpu_count(),
                         'affinity': sorted(os.sched_getaffinity(0)), 'torch_threads': torch.get_num_threads()}
    inputs_dir = out/'inputs'
    inputs_dir.mkdir(exist_ok=True)
    inputs = prepare(args.inputs, inputs_dir)
    rows = json.loads(inputs['test.json'].read_text())
    assert len(rows) == 1919
    tok, _ = tokenizer(inputs['tokenizer.json'], 30)
    encoded = [[C.BOS]+C.encode(tok, r['question'])+[C.EOS] for r in rows]
    order = sorted(range(1919), key=lambda i: (len(encoded[i]), i))
    subset = stratified(order, args.ablation_questions)
    write(out/'row-ids.json', {'full_sorted': order, 'ablation_length_stratified': subset})
    model = load_model(unique(args.live, 'model-C-main__weights__l2-62280-fp16.pt'), device)
    # Warm cuBLAS/SDPA for both initial batch shapes, no reported cold-start speedup.
    for k in (1, 4):
        measure(model, encoded, order[:1024//k], 'baseline', k, device, args.deadline, cap=8)
    report['setup_seconds'] = time.monotonic()-args.start
    save()
    references = {}

    def one(variant, k, load, selected, scope):
        load.alive()
        row, predictions = measure(model, encoded, selected, variant, k, device, args.deadline)
        row.update(scope=scope, load_workers=load.workers, nice=load.info['nice'],
                   nice_requested_increment=load.nice, nice_applied=load.info['nice_applied'])
        if k == 1:
            # Subset batches have different GEMM shapes from full1919: use their
            # own paired control, never misattribute shape rounding to a flag.
            reference = references.setdefault(scope, {})
            missing = [i for i in selected if i not in reference]
            if missing:
                assert variant == 'baseline', 'Control must precede each greedy comparison'
                reference.update(predictions)
            mismatches = [i for i in selected if not torch.equal(predictions[i], reference[i])]
            row['greedy_mismatches'] = len(mismatches)
            report['greedy_checks'].append({'variant': variant, 'load_workers': load.workers,
                'nice': load.nice, 'rows': len(selected), 'mismatch_rows': mismatches})
        else:
            row['greedy_mismatches'] = None
        report['rows'].append(row)
        load.alive()
        save()
        print(json.dumps({key: row[key] for key in ['variant', 'mode', 'scope', 'load_workers',
                          'seconds', 'sequences_per_second', 'greedy_mismatches']}), flush=True)

    try:
        # Mandatory full-data control/all gate first, both with and without sustained execution.
        for loaded in (False, True):
            for variant in ('baseline', 'all'):
                workers = (2 if variant == 'all' else 4) if loaded else 0
                nice = 10 if loaded and variant == 'all' else 0
                with Load(out/f'load-full-{variant}-{workers}', workers, nice) as load:
                    for k in (1, 4):
                        one(variant, k, load, order, 'test1919')
                    report['profiles'][f'{variant}-workers{workers}'] = profile(
                        model, encoded, order, variant, device, out/f'trace-{variant}-workers{workers}', args.deadline)
                    save()
        # Reuse mandatory rows for baseline/all; single flags share a paired ablation control.
        selected = order if args.scope == 'full' else subset
        for loaded in (False, True):
            for variant in VARIANTS:
                if args.scope == 'full' and variant in ('baseline', 'all'):
                    continue
                # workers/nice have no effect with execution off: explicit aliases below.
                if not loaded and variant in ('workers2', 'nice10'):
                    continue
                workers = (2 if variant in ('workers2', 'all') else 4) if loaded else 0
                nice = 10 if loaded and variant in ('nice10', 'all') else 0
                with Load(out/f'load-ablation-{variant}-{workers}', workers, nice) as load:
                    for k in (1, 4):
                        one(variant, k, load, selected, 'test1919' if args.scope == 'full' else 'test1919-subset')
        report['identity_aliases'] = [
            {'variant': 'preallocated_kv', 'same_as': 'baseline', 'reason': 'self-KV index writes and one-time cross projection already present'},
            {'variant': 'torch_threads1', 'same_as': 'baseline', 'reason': 'v5 already has one intra/inter-op thread'},
            {'variant': 'workers2/nice10 without load', 'same_as': 'baseline', 'reason': 'no executor exists'}]
        report['matrix_complete'] = True
        full_checks = [r for r in report['greedy_checks'] if r['rows'] == 1919 and r['variant'] == 'all']
        report['GPU_gate_passed'] = (len(full_checks) == 2 and
                                    not any(r['mismatch_rows'] for r in report['greedy_checks']))
        report['status'] = 'complete' if report['GPU_gate_passed'] else 'greedy_mismatch'
        report['nice_supported'] = all(r['nice_applied'] for r in report['rows']
                                       if r['nice_requested_increment'])
    except C.BudgetExpired:
        report.update(status='budget_stop_partial', matrix_complete=False, GPU_gate_passed=False)
    except BaseException as error:
        report.update(status='error', error=repr(error), matrix_complete=False, GPU_gate_passed=False)
        raise
    finally:
        save()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--live', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start', type=float, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    parser.add_argument('--scope', choices=['full', 'paired-subset'], default='paired-subset')
    parser.add_argument('--ablation-questions', type=int, default=192)
    args = parser.parse_args()
    assert 2 <= args.ablation_questions <= 1919
    try:
        run(args)
    except BaseException as error:
        # Cover setup/runtime assertions too, before the main measurement try block.
        path = args.output/'summary.json'
        report = json.loads(path.read_text()) if path.exists() else {}
        report.update(status='error', error=repr(error), GPU_gate_passed=False,
                      matrix_complete=False, session_seconds=time.monotonic()-args.start)
        write(path, report)
        raise
