"""Small independent session watchdog and single-launch benchmark suite (no Kaggle API/secrets)."""
import gzip
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from final_split import fixed_split
from final_vote import select, validate_answers

INPUT_SHA = {
    'tokenizer.json': '261f25b5d4be006a6a00e41e0eafbe522580741da627ba670f6b9805a2659b7d',
    'test.json': '20cacb334e3fc81291433d2173d9085b17a15e1834bd84aea67b350403182503',
    'heldout4k.json': '28a3c0c9088c5b4a17d75d216e4abc4262cc9ca62884966c43e7ec09b6f040c2',
    'heldout4k-reference-execution.json': '3855680e9a37292aa2e351b8e8e4f7a5cbbe76444096d136470e4a9be8797111'}


def write(path, value):
    temp = Path(path).with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


class Watchdog:
    def __init__(self, output, start):
        self.output, self.start = Path(output), start
        self.children = set()
        self.done = threading.Event()
        self.triggered = False

    def kill(self):
        for pid in list(self.children):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def arm(self):
        def watch():
            while not self.done.wait(.1):
                elapsed = time.monotonic()-self.start
                if elapsed >= 840. and not self.triggered:
                    self.triggered = True
                    self.kill()
                    write(self.output/'watchdog.json', {'triggered': True, 'elapsed_seconds': elapsed})
                if elapsed >= 870.:
                    self.kill()
                    os._exit(0)
        threading.Thread(target=watch, daemon=True, name='final-session-watchdog').start()
        return self

    def finish(self):
        self.kill()
        self.done.set()


def prepare_inputs(root, target, benchmark=True):
    """Extract only 4 small JSONs from starter ZIP; never unpack initial/control/train weights."""
    root, target = Path(root), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in INPUT_SHA.items():
        if not benchmark and name not in ('tokenizer.json', 'test.json'):
            continue
        direct = list(root.rglob(name))
        if len(direct) == 1:
            path = direct[0]
        elif not direct:
            archives = list(root.rglob('26b-training-inputs-d0-initial-and-evaluation.zip'))
            assert len(archives) == 1, 'Expected direct starter files or one known starter ZIP'
            with zipfile.ZipFile(archives[0]) as archive:
                members = [m for m in archive.namelist() if Path(m).name == name]
                assert len(members) == 1
                path = target/name
                with path.open('wb') as stream:
                    stream.write(archive.read(members[0]))
        else:
            raise ValueError('Ambiguous input '+name)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
        yield name, str(path)


def unique(root, name):
    paths = list(Path(root).rglob(name))
    assert len(paths) == 1, 'Missing or ambiguous '+name
    return str(paths[0])


def stored_tune(live, heldout):
    """Check attached L2 first4 baseline independently, including question identities."""
    ids = fixed_split()['tune2048']
    models = {}
    for name in ['C-main', 'C-long2x']:
        pattern = f'model-{name}__points__l2-28800__evaluation__k8-tune2048__batch-*.json*'
        files = sorted(Path(live).rglob(pattern))
        assert len(files) == 64, 'Expected all L2 K8 journals: '+name
        rows = []
        for path in files:
            data = gzip.decompress(path.read_bytes()) if path.name.endswith('.gz') else path.read_bytes()
            rows.extend(json.loads(data))
        index = {r['id']: r for r in rows}
        assert len(rows) == len(index) == 2048 and sorted(index) == ids
        for i in ids:
            assert index[i]['question_sha256'] == hashlib.sha256(heldout[i]['question'].encode()).hexdigest()
        models[name] = index
    matches = 0
    for i in ids:
        main, long = models['C-main'][i], models['C-long2x'][i]
        assert main['reference'] == long['reference']
        win = select(main['greedy'], main['samples']['0.7'][:4], long['samples']['0.7'][:4])
        ref = main['reference']
        matches += ref['status'] == win['execution']['status'] == 'numeric' and math.isclose(
            ref['value'], win['execution']['value'], rel_tol=1e-6, abs_tol=1e-6)
    assert matches == 1492
    return models, {'matches': matches, 'n': 2048, 'accuracy_pp': 100*matches/2048}


def diagnose(records, journals, ids):
    result = {}
    for name, target in [('C-main', 'main'), ('C-long2x', 'long')]:
        code_diff = sum(records[j][target][s]['code'] != journals[name][i]['samples']['0.7'][s]['code']
                        for j, i in enumerate(ids) for s in range(min(4, len(records[j][target]))))
        complete = sum(len(r[target]) == 4 for r in records)
        result[name] = {'complete_rows': complete, 'first4_code_mismatches': code_diff}
    result['greedy_code_mismatches'] = sum(r['greedy']['code'] != journals['C-main'][i]['greedy']['code']
                                         for r, i in zip(records, ids))
    return result


def suite(inputs_root, live_root, output, module_dir, watchdog, benchmark=True):
    output = Path(output)
    report = {'status': 'running', 'test': [], 'GPU_runs': 0, 'training': False,
              'budget_seconds': 900, 'internet': False, 'single_gpu': True}
    write(output/'summary.json', report)
    try:
        inputs = dict(prepare_inputs(inputs_root, output/'inputs', benchmark=benchmark))
        models = {name: unique(live_root, f'model-{name}__weights__l2-28800-fp16.pt')
                  for name in ['C-main', 'C-long2x']}
        heldout = json.loads(Path(inputs['heldout4k.json']).read_text()) if benchmark else None
        ids = fixed_split()['tune2048']
        rows_path, ref_path = output/'tune2048.json', output/'tune2048-reference.json'
        if benchmark:
            references = json.loads(Path(inputs['heldout4k-reference-execution.json']).read_text())
            refs = references['rows'] if isinstance(references, dict) else references
            assert len(heldout) == len(refs) == 4096
            write(rows_path, [{'question': heldout[i]['question']} for i in ids])
            write(ref_path, [refs[i] for i in ids])
            journals, report['stored_tune'] = stored_tune(live_root, heldout)
            for i in ids:
                assert journals['C-main'][i]['reference'] == refs[i]
        base = {'models': models, 'tokenizer': inputs['tokenizer.json'],
                'tokenizer_sha': INPUT_SHA['tokenizer.json'], 'workers': os.cpu_count() or 1,
                'budget_seconds': 180., 'sample_batch': 256, 'greedy_batch': 1024}
        def launch(name, rows, expected, **extra):
            assert not watchdog.triggered and time.monotonic()-watchdog.start < 660.
            destination = output/name
            destination.mkdir(exist_ok=True)
            config = dict(base, rows=str(rows), output=str(destination), expected_rows=expected, **extra)
            config_path = destination/'config.json'; write(config_path, config)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0')
            log_path = destination/'supervisor.log'
            with log_path.open('w') as log:
                # Timestamp before creation includes Python interpreter startup and all imports.
                env['FINAL_PROCESS_START'] = str(time.monotonic())
                child = subprocess.Popen([sys.executable, str(Path(module_dir)/'final_runner.py'), str(config_path)],
                                         start_new_session=True, env=env, stdout=log, stderr=log)
                watchdog.children.add(child.pid)
                try:
                    until = min(watchdog.start+840., time.monotonic()+config['budget_seconds']+5.)
                    next_heartbeat = time.monotonic()+30.
                    while child.poll() is None and time.monotonic() < until and not watchdog.triggered:
                        if time.monotonic() >= next_heartbeat:
                            print(name, 'running; session_s', round(time.monotonic()-watchdog.start), flush=True)
                            next_heartbeat += 30.
                        time.sleep(.1)
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=1.)
                    assert child.returncode == 0, name+' failed; inspect small worker/supervisor logs'
                finally:
                    watchdog.children.discard(child.pid)
            validate_answers(destination/'answers.jsonl', expected)
            report['GPU_runs'] += 1
            return json.loads((destination/'summary.json').read_text())
        for number in range(1, 3 if benchmark else 2):
            row = launch(f'test-{number}', inputs['test.json'], 1919)
            report['test'].append(row)
            if number == 1:
                # The published answer is always the FIRST cold test run, never picked by heldout score.
                (output/'answers.jsonl').write_bytes((output/'test-1/answers.jsonl').read_bytes())
            write(output/'summary.json', report)
            print('test', number, row['status'], round(row['full_pipeline_seconds'], 3), flush=True)
        if not benchmark:
            report['status'] = report['test'][0]['status']
            return report
        report['tune'] = launch('tune', rows_path, 2048, reference=str(ref_path))
        quality_path = output/'tune/quality.json'
        report['quality'] = json.loads(quality_path.read_text()) if quality_path.exists() else {'passed': False, 'reason': 'supervisor fallback'}
        if (output/'tune/candidate-journal.json.gz').exists():
            records = json.loads(gzip.decompress((output/'tune/candidate-journal.json.gz').read_bytes()))
            report['K4_diagnostic'] = diagnose(records, journals, ids)
        write(output/'summary.json', report)
        digests = [hashlib.sha256((output/f'test-{i}/answers.jsonl').read_bytes()).hexdigest()
                   for i in range(1, 3)]
        report['answers_sha256'] = digests
        report['deterministic_answers'] = len(set(digests)) == 1
        report['gates'] = {'test_all_under_180': all(r['under_180_seconds'] for r in report['test']),
                           'test_all_under_120': all(r['target_120_seconds'] for r in report['test']),
                           'no_test_fallback': all(r['status'] == 'complete' and
                               r['deadline_guard']['fallback_count'] == 0 for r in report['test']),
                           'deterministic_answers': report['deterministic_answers'],
                           'quality_within_1pp': report['quality']['passed']}
        report['status'] = 'passed' if all(report['gates'].values()) else 'review_required'
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['session_seconds'] = time.monotonic()-watchdog.start
        write(output/'summary.json', report)
        watchdog.finish()
    return report
