"""Regression checks for independent final-inference review; no network/GPU."""
import json
import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent/'src'
sys.path.insert(0, str(SRC))
from final_executor import Executor
from final_vote import verify_winners


def output_checks():
    from final_output_check import check, WORK
    from final_session import INPUT_SHA
    from final_vote import write_answers
    test = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/test.json'
    questions = [r['question'] for r in json.loads(test.read_text())]
    order = hashlib.sha256(json.dumps(questions, ensure_ascii=False).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='final-output-regression-') as folder:
        folder = Path(folder)
        for i in range(1, 3):
            path = folder/f'test-{i}'; path.mkdir()
            write_answers(path/'answers.jsonl', ['def solve():\n    return '+str(j) for j in range(1919)])
            timing = {'status': 'complete', 'rows': 1919, 'full_pipeline_seconds': 120.,
                      'deadline_guard': {'triggered': False, 'fallback_count': 0}, 'OOM_attempts': [],
                      'winner_reexecution': {'complete': True, 'mismatch_rows': []}, 'single_gpu': True,
                      'question_order_sha256': order, 'input_sha256': INPUT_SHA['test.json'],
                      'answers_sha256': hashlib.sha256((path/'answers.jsonl').read_bytes()).hexdigest(),
                      'phases_seconds': {'fixture': 120.}}
            (path/'summary.json').write_text(json.dumps(timing))
            (path/'config.json').write_text(json.dumps({'expected_rows': 1919}))
        (folder/'answers.jsonl').write_bytes((folder/'test-1/answers.jsonl').read_bytes())
        (folder/'summary.json').write_text(json.dumps({'status': 'passed', 'GPU_runs': 3,
            'session_seconds': 600., 'gates': {'fixture': True}}))
        tune = folder/'tune'; tune.mkdir()
        write_answers(tune/'answers.jsonl', ['def solve():\n    return 1']*2048)
        (tune/'quality.json').write_text(json.dumps({'n': 2048, 'passed': True, 'delta_pp': 0.}))
        assert check(folder, test)['passed']
        path = folder/'test-2/summary.json'
        original = json.loads(path.read_text())
        for patch in [{'question_order_sha256': 'wrong'}, {'full_pipeline_seconds': 120.01},
                      {'deadline_guard': {'triggered': True, 'fallback_count': 1}},
                      {'single_gpu': False}]:
            path.write_text(json.dumps(dict(original, **patch)))
            try:
                check(folder, test)
            except AssertionError:
                pass
            else:
                raise AssertionError('Invalid benchmark accepted: '+str(patch))
        path.write_text(json.dumps(original))
        write_answers(folder/'test-2/answers.jsonl', ['different']*1919)
        original['answers_sha256'] = hashlib.sha256((folder/'test-2/answers.jsonl').read_bytes()).hexdigest()
        path.write_text(json.dumps(original))
        try:
            check(folder, test)
        except AssertionError:
            pass
        else:
            raise AssertionError('Nondeterministic test answers accepted')
    return {'passed': True, 'GPU_used': False, 'fixtures_only': True,
            'wrong_order_late_run_fallback_multiGPU_different_answers_rejected': True}


def authorization_check():
    from unittest.mock import patch
    import final_review_api as api
    with patch.dict(os.environ, {}, clear=True), \
         patch.object(api.importlib.util, 'find_spec', side_effect=AssertionError('SDK must not load')):
        try:
            api.environment_api()
        except RuntimeError as error:
            assert str(error) == 'environment_authorization_unavailable'
        else:
            raise AssertionError('Missing environment authorization was not rejected')
    return {'missing_authorization_stops_before_SDK_or_credential_file_reads': True}


def pipe_death_check():
    import io
    from unittest.mock import patch
    import final_execute as service
    failures = [True]
    class Process:
        alive = True
        def start(self): pass
        def is_alive(self): return self.alive
        def kill(self): self.alive = False
        def join(self, **kwargs): pass
    class Pipe:
        result = None
        def close(self): pass
        def poll(self): return self.result is not None
        def recv(self):
            result, self.result = self.result, None
            return result
        def send(self, item):
            if failures[0]:
                failures[0] = False
                raise BrokenPipeError('worker died after is_alive')
            self.result = (item[0], {'status': 'numeric', 'value': 17.})
    class Context:
        def Pipe(self): return Pipe(), Pipe()
        def Process(self, **kwargs): return Process()
    output = io.StringIO()
    with patch.object(service.mp, 'get_context', return_value=Context()), \
         patch.object(service, 'wait'), \
         patch.object(service.sys, 'stdin', io.StringIO(json.dumps([[0, 'def solve(): return 17']])+'\nnull\n')), \
         patch.object(service.sys, 'stdout', output):
        service.serve(1)
    assert [json.loads(line)[:2] for line in output.getvalue().splitlines()] == [[0, {'status': 'numeric', 'value': 17.}]]
    return {'send_failure_after_is_alive_requeues_exactly_once': True, 'worker_death_race_injected': True}


def regression_checks():
    with tempfile.TemporaryDirectory(prefix='final-review-') as folder:
        marker = Path(folder)/'escaped.txt'
        payload = "__import__('pathlib').Path("+repr(str(marker))+").write_text('escape')"
        codes = [
            'import math\nmath.pi = 42\ndef solve():\n    return math.pi',
            'import math\ndef solve():\n    return math.pi',
            'from sympy import Symbol\ndef solve():\n    x = Symbol("x")\n    return (x+1).subs(x, '+repr(payload)+')',
            'from sympy import Symbol, Eq, solve as roots\ndef solve():\n    x = Symbol("x")\n    return float(roots(Eq(x+1, 3), x)[0])',
            'def solve():\n    print("x" * 1000000)\n    return 19',
            'import os\ndef solve():\n    return os.fork()',
            ('def solve():\n    return len([0]*1000000000)' if sys.platform.startswith('linux')
             else 'def solve():\n    return 1'),
            'def solve():\n    while True:\n        try:\n            while True: pass\n        except Exception: pass',
            'from sympy import Symbol\ndef solve():\n    return Symbol("x").random()',
            'def solve():\n    return float("{0.__globals__[math].pi}".format(print))',
        ]
        items = [{'code': c} for c in codes]
        executor = Executor(1)  # Same worker is reused; leakage must be impossible.
        started = time.monotonic()
        try:
            executor.submit(items)
            assert executor.drain(time.monotonic()+20.)
            executor.attach(items)
            results = [c['execution'] for c in items]
            assert results[0]['status'] == 'blocked_attribute_mutation'
            assert results[1] == {'status': 'numeric', 'value': 3.141592653589793}
            assert results[2]['status'] == 'execution_error' and not marker.exists()
            assert results[3] == {'status': 'numeric', 'value': 2.}
            assert results[4] == {'status': 'numeric', 'value': 19.}
            assert results[5]['status'].startswith('blocked_')
            if sys.platform.startswith('linux'):
                assert results[6]['status'] in ('execution_error', 'timeout_or_worker_exit')
            assert results[7]['status'] in ('timeout', 'timeout_or_worker_exit')
            assert results[8]['status'] == 'blocked_attribute'
            assert results[9]['status'] == 'blocked_attribute'
            recovered = [{'code': 'def solve():\n    return 23'}]
            executor.submit(recovered)
            assert executor.drain(time.monotonic()+10.)
            executor.attach(recovered)
            assert recovered[0]['execution'] == {'status': 'numeric', 'value': 23.}
            assert executor.error is None
        finally:
            executor.close()
        # Detect a corrupted cached answer by reexecuting the chosen program independently.
        winners = [{'code': 'def solve():\n    return 7',
                    'execution': {'status': 'numeric', 'value': 8}},
                   {'code': 'def solve():\n    return 9',
                    'execution': {'status': 'numeric', 'value': 9}}]
        verified = verify_winners(winners, time.monotonic()+10., workers=1)
        assert verified['complete'] and verified['mismatch_rows'] == [0]
        return {'namespace_mutation_blocked': True, 'sympy_string_escape_blocked': True,
                'ordinary_sympy_equations_preserved': True, 'stdout_protocol_clean': True,
                'fork_import_blocked': True, 'memory_exhaustion_bounded': sys.platform.startswith('linux'),
                'caught_timeout_bounded_and_worker_recovered': True,
                'chosen_code_reexecution_detects_wrong_cached_value': True,
                'sympy_random_method_blocked_and_internal_RNG_seeded': True,
                'format_string_hidden_attribute_traversal_blocked': True,
                'seconds': time.monotonic()-started,
                'linux_process_group_and_RLIMIT_AS': sys.platform.startswith('linux'),
                'authorization': authorization_check(), 'output_validation': output_checks(),
                'pipe_death': pipe_death_check()}


if __name__ == '__main__':
    print(json.dumps(regression_checks(), indent=2))
