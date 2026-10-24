"""V8: one cold test and tune, ordered dataset/model config and independent watchdog (no Kaggle API/secrets)."""
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
from final_vote import validate_answers
from final_ensemble import parameters

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
                if elapsed >= 660. and not self.triggered:
                    self.triggered = True
                    self.kill()
                    write(self.output/'watchdog.json', {'triggered': True, 'elapsed_seconds': elapsed})
                if elapsed >= 690.:
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


def weight_path(root, model):
    if 'relative_path' not in model:
        return unique(root, model['filename'])
    relative = Path(model['relative_path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    path = Path(root)/relative
    assert path.is_file() and path.name == model['filename'], 'Missing pinned model '+str(relative)
    return str(path)


def suite(inputs_root, weights_roots, output, module_dir, watchdog, benchmark=True, ensemble_config=None):
    output = Path(output)
    report = {'status': 'running', 'test': [], 'GPU_runs': 0, 'training': False,
              'budget_seconds': 720, 'internet': False, 'single_gpu': True}
    write(output/'summary.json', report)
    try:
        inputs = dict(prepare_inputs(inputs_root, output/'inputs', benchmark=benchmark))
        assert ensemble_config is not None, 'Pass the reviewed ensemble config'
        model_specs, _ = parameters(ensemble_config)
        models = [dict(m, path=weight_path(weights_roots[m['dataset']], m)) for m in model_specs]
        heldout = json.loads(Path(inputs['heldout4k.json']).read_text()) if benchmark else None
        ids = fixed_split()['tune2048']
        rows_path, ref_path = output/'tune2048.json', output/'tune2048-reference.json'
        if benchmark:
            references = json.loads(Path(inputs['heldout4k-reference-execution.json']).read_text())
            refs = references['rows'] if isinstance(references, dict) else references
            assert len(heldout) == len(refs) == 4096
            write(rows_path, [{'question': heldout[i]['question']} for i in ids])
            write(ref_path, [refs[i] for i in ids])
            report['stored_tune'] = {'matches': ensemble_config['expected_matches'], 'n': 2048,
                                     'origin': 'local verified K8 replay; new RNG samples can differ'}
        base = dict(ensemble_config, models=models, tokenizer=inputs['tokenizer.json'],
                    tokenizer_sha=INPUT_SHA['tokenizer.json'], workers=2, verify_weights=True,
                    budget_seconds=180., greedy_batch=1024)
        def launch(name, rows, expected, **extra):
            assert not watchdog.triggered and time.monotonic()-watchdog.start < 480.
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
                    until = min(watchdog.start+660., time.monotonic()+config['budget_seconds']+5.)
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
        for number in range(1, 2):
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
        write(output/'summary.json', report)
        digests = [hashlib.sha256((output/f'test-{i}/answers.jsonl').read_bytes()).hexdigest()
                   for i in range(1, 2)]
        report['answers_sha256'] = digests
        report['deterministic_answers'] = None  # One cold test; repeatability not measured
        report['gates'] = {'test_all_under_180': all(r['under_180_seconds'] for r in report['test']),
                           'test_with_40s_margin': all(r['full_pipeline_seconds'] <= 140. for r in report['test']),
                           'tune_under_180_complete': report['tune']['under_180_seconds'] and report['tune']['status'] == 'complete',
                           'resident_models_and_T4_memory': all(
                               len(r.get('model_order', [])) == len(models) and
                               r.get('peak_reserved_bytes', float('inf')) < r.get('total_device_memory_bytes', 0) and
                               r.get('peak_allocated_bytes', float('inf'))+256*1024**2 < r.get('total_device_memory_bytes', 0)
                               for r in report['test']+[report['tune']]),
                           'no_test_fallback': all(r['status'] == 'complete' and
                               r['deadline_guard']['fallback_count'] == 0 for r in report['test']),
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
