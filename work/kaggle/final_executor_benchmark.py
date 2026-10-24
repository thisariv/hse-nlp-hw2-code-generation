"""CPU-only v2/v3 benchmark: same 2000 unique K8 programs, exact result bytes.

python final_executor_benchmark.py [--programs 2000] [--repeats 3]
Baseline sources are read from pinned Git commit 2cd5208; no working files replaced.
"""
import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path.insert(0, str(ROOT/'src'))
from final_executor import Executor


def corpus(n):
    raw = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l2/1'
    codes = []
    for model in ('C-main', 'C-long2x'):
        for path in sorted((raw/f'model-{model}/points/l2-28800/evaluation/k8-tune2048').glob('batch-*.json')):
            for row in json.loads(path.read_text()):
                codes.extend(sample['code'] for sample in row['samples']['0.7'])
    codes = list(dict.fromkeys(codes))
    random.Random(270042).shuffle(codes)
    # Include the rare SymPy path as well as ordinary arithmetic and invalid programs.
    symbolic = [code for code in codes if 'sympy' in code]
    selected = (symbolic + [code for code in codes if code not in symbolic])[:n]
    assert len(selected) == n >= 2000
    return selected


def run(service, codes):
    started = time.monotonic()
    executor = Executor(4, service_path=service)
    try:
        items = [{'code': code} for code in codes]
        executor.submit(items)  # All work queued now; service wall no longer includes GPU wait.
        assert executor.drain(started+120.), executor.summary()
        executor.attach(items)
        seconds = time.monotonic()-started
        results = json.dumps([item['execution'] for item in items], sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode()
        return {'seconds': seconds, 'programs_per_second': len(codes)/seconds,
                'result_sha256': hashlib.sha256(results).hexdigest(),
                'execution': executor.summary()}, results
    finally:
        executor.close()


def benchmark(programs=2000, repeats=3):
    codes = corpus(programs)
    report = {'CPU_only': True, 'GPU_used': False, 'workers': 4, 'programs': len(codes),
              'corpus': 'both L2 l2-28800 K8 tune journals; unique shuffled seed270042; SymPy included',
              'corpus_sha256': hashlib.sha256(json.dumps(codes).encode()).hexdigest(),
              'sympy_programs': sum('sympy' in code for code in codes),
              'baseline_commit': '2cd5208', 'v2': [], 'v3': [], 'baseline_source_sha256': {}}
    with tempfile.TemporaryDirectory(prefix='final-executor-v2-') as folder:
        folder = Path(folder)
        for name in ('final_execute.py', 'final_numeric.py'):
            git_path = (ROOT/'src'/name).relative_to(ROOT.parents[5])
            source = subprocess.check_output(['git', 'show', '2cd5208:'+str(git_path)], cwd=ROOT)
            (folder/name).write_bytes(source)
            report['baseline_source_sha256'][name] = hashlib.sha256(source).hexdigest()
        expected = None
        for repetition in range(repeats):
            # Alternate order; baseline and candidate never compete for CPU with each other.
            for name in (('v2', 'v3') if repetition % 2 == 0 else ('v3', 'v2')):
                row, results = run(folder/'final_execute.py' if name == 'v2' else None, codes)
                if expected is None:
                    expected = results
                assert results == expected, name+' result bytes changed'
                report[name].append(row)
                print(name, repetition+1, round(row['seconds'], 3), 's', flush=True)
    for name in ('v2', 'v3'):
        report[name+'_median_seconds'] = statistics.median(row['seconds'] for row in report[name])
        report[name+'_programs_per_second'] = len(codes)/report[name+'_median_seconds']
    report['speedup'] = report['v2_median_seconds']/report['v3_median_seconds']
    report['bit_identical_result_bytes'] = True
    report['result_mismatches'] = 0
    report['target_150_programs_per_second_CPU_host'] = report['v3_programs_per_second'] >= 150.
    report['platform'] = sys.platform
    report['T4_4vCPU_GPU_overlap_verified'] = False
    target = WORK/'checks/final-executor-v3-benchmark.json'
    target.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--programs', type=int, default=2000)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    result = benchmark(args.programs, args.repeats)
    print(json.dumps({key: value for key, value in result.items() if key not in ('v2', 'v3')}, indent=2))
