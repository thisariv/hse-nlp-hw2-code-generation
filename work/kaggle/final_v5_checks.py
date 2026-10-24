"""V5 CPU regressions, bundle audit and readiness. Never starts CUDA or Kaggle."""
import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
SRC = ROOT/'src'
FINAL = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final'
BUNDLE = ROOT/'bundles/final_inference_l3_v5'
sys.path.insert(0, str(SRC))


def read(path):
    return json.loads(Path(path).read_text())


def unit():
    from final_adaptive import parameters, question_batch, stage1_decision, stage1_leader_fraction, select, select_available
    from final_vote import select as old
    def c(v, lp=-1., status='numeric'):
        return {'code': str(v), 'represented_code': '<|raw|>'+str(v), 'mean_logprob': lp,
                'execution': {'status': status, 'value': v}}
    for cfg in [{'K1': 0}, {'K1': 1.5}, {'K2': True}, {'K1': 3, 'K2': 2},
                {'unanimity_threshold': 0}, {'unanimity_threshold': float('nan')}]:
        try:
            parameters(cfg)
        except ValueError:
            pass
        else:
            raise AssertionError(cfg)
    g = c(1., 0.)
    assert question_batch(1024, 1) == 1024 and question_batch(1024, 2) == 512
    assert question_batch(1024, 4) == 256 and question_batch(1024, 6) == 170
    for lanes, k in [(1, 2), (True, 1), (1024., 2), (1024, 0)]:
        try:
            question_batch(lanes, k)
        except ValueError:
            pass
        else:
            raise AssertionError((lanes, k))
    r = {'greedy': g, 'main': [c(1.)]*2, 'long': [c(1.)]*2}
    assert stage1_decision(r) is True
    assert stage1_decision(dict(r, long=[c(1.), c(2.)])) is False
    assert stage1_decision(dict(r, long=[c(1.), c(2.)]), threshold=.8) is True
    for status in ['syntax_error', 'timeout', 'non_number']:
        assert stage1_decision(dict(r, main=[c(1.), c(1., status=status)])) is False
    assert stage1_decision(dict(r, main=[c(1.), c(float('inf'))])) is False
    assert stage1_decision(dict(r, main=[c(1., status='budget_not_evaluated')]*2)) is None
    assert stage1_decision(dict(r, long=[])) is None
    assert stage1_leader_fraction(r) == 1.
    assert stage1_leader_fraction(dict(r, long=[c(2.), c(3.)])) == .6
    assert stage1_leader_fraction({'greedy': c(0.), 'main': [c(.9e-6), c(1.8e-6)],
                                  'long': [c(2.), c(3., status='timeout')]}) == .6
    assert stage1_leader_fraction({'greedy': c(1.), 'main': [c(2.), c(3.)],
                                  'long': [c(4.), c(5.)]}) == .2
    assert stage1_leader_fraction({'greedy': c(0., status='timeout'),
                                  'main': [c(2., status='timeout')]*2,
                                  'long': [c(3., status='timeout')]*2}) == 0.
    assert select(g, [c(2.)]*8, [c(1.)]*4, k=8)['execution']['value'] == 2.
    assert select(g, [c(2.)]*4, [c(1.)]*4, k=4) is g
    bad = c(0., status='syntax_error')
    assert select(bad, [c(2., -2.)], [c(3., -1.)], k=8)['execution']['value'] == 3.
    assert select(g, [c(1., 1.)], [], k=8) is g
    chain = [c(0., -.1), c(.9e-6, -.2), c(1.8e-6, -.3)]
    assert select(bad, chain, [c(20.)]*2)['execution']['value'] == 0.
    assert select(c(0., -100.), chain[1:], [])['code'] == '0.0'
    winner, count = select_available(dict(r, main=[c(2.)]*6, long=[c(2., status='budget_not_evaluated')]))
    assert count == 6 and winner['execution']['value'] == 2.
    for main, long in [([c(2.)]*4, [c(1.)]*4), (chain, [c(20.)]*2)]:
        assert select(g, main, long, 4) == old(g, main, long)
    called = []
    def scorer(question, represented):
        called.append((question, represented))
        return 20. if represented == '<|raw|>3.0' else 0.
    assert select(g, [c(2.)]*4, [c(3.)], scorer=scorer, score_weight=0.)['execution']['value'] == 2.
    assert not called
    assert select(g, [c(2.)]*4, [c(3.)], question='q', scorer=scorer, score_weight=1.)['execution']['value'] == 3.
    assert called and all(q == 'q' and code.startswith('<|raw|>') for q, code in called)
    return {'parameters_stop_nonfinite_pending_graph_LP_duplicates_partial': True,
            'sequence_budget_and_numeric_leader_fraction': True,
            'scorer_disabled_and_represented_code_interface': True}


def fixture(case):
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    import final_inference_v5 as I
    priority_case = case in ['priority', 'priority_guard']
    state = {'oom': case == 'oom', 'draws': [], 'stage2_started': False, 'drained': False}
    def decode(ctx, k=1, temperature=None, **kwargs):
        if temperature is not None:
            state['draws'].append((k, float(torch.rand(()))))
            if k == 4:
                state['stage2_started'] = True
                if state['oom']:
                    state['oom'] = False
                    raise torch.cuda.OutOfMemoryError('Injected stage2 OOM after random draw')
        value = 1 if temperature is None else (7 if k == 4 else (2 if ctx.model == 'a' else 3))
        def token(row, lane):
            if int(row[1]) == 3 or temperature is None:
                return 1
            if priority_case and k == 2:
                if int(row[1]) == 4:
                    return 1 if ctx.model == 'a' else 3
                return (2 if ctx.model == 'a' else 4)+lane
            return value
        seq = torch.tensor([[[token(row, lane)] for lane in range(k)] for row in ctx.src])
        return seq, torch.full((len(ctx.src), k), -1.), 0
    def pack(ids, lps, table, greedy=False):
        return [[{'code': 'def solve():\n    return '+str(tokens[0]),
                  'represented_code': '<|raw|>def solve():\n    return '+str(tokens[0]),
                  'mean_logprob': 0. if greedy else -1.} for tokens in group] for group in ids.tolist()]
    class DelayedExecutor:
        """Slowest stage1 row becomes ready only on the global drain barrier."""
        def __init__(self, workers):
            self.codes, self.results = {}, {}
            self.error = None
            self.child = SimpleNamespace(poll=lambda: None)
        def submit(self, candidates):
            for c in candidates:
                self.codes[c['code']] = len(self.codes)
        def attach(self, candidates):
            for c in candidates:
                val = int(c['code'].split()[-1])
                pending = val == 5 and not state['drained']
                c['execution'] = {'status': 'budget_not_evaluated'} if pending else {'status': 'numeric', 'value': float(val)}
        def drain(self, deadline):
            state['drained'] = True
            return True
        def summary(self):
            return {'mock': True}
        def close(self):
            pass
    with tempfile.TemporaryDirectory(prefix='final-v5-fixture-') as directory:
        directory = Path(directory)
        rows = directory/'rows.json'
        rows.write_text(json.dumps([{'question': q} for q in ['a', 'bb', 'ccc']]))
        cfg = {'rows': str(rows), 'tokenizer': 'fixture', 'models': {'C-main': 'a', 'C-long2x': 'b'},
               'output': str(directory/'out'), 'verify_weights': False, 'device': 'cpu', 'workers': 1,
               'cap': 8, 'budget_seconds': 12., 'greedy_sequence_budget': 3,
               'sequence_budget': 4 if priority_case else 8,
               'K1': 2, 'K2': 6, 'unanimity_threshold': 1.}
        if case in ['guard1', 'guard2']:
            cfg['cpu_test_expire_after_sample_batches'] = 1 if case == 'guard1' else 3
        if case == 'priority_guard':
            cfg['cpu_test_expire_after_sample_batches'] = 6
        patches = [patch.object(I, 'load', side_effect=lambda path, device: path),
                   patch.object(I, 'tokenizer', return_value=(None, {})),
                   patch.object(I.C, 'encode', side_effect=lambda tok, q: [2+len(q)]),
                   patch.object(I.Fast, 'Context', side_effect=lambda model, src, cap: SimpleNamespace(src=src, model=model)),
                   patch.object(I.Fast, 'decode', side_effect=decode), patch.object(I, 'pack', side_effect=pack)]
        if case == 'priority':
            patches.append(patch.object(I, 'Executor', DelayedExecutor))
        from contextlib import ExitStack
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            timing = I.run(cfg)
        answers = [json.loads(line)['code'] for line in (directory/'out/answers.jsonl').read_text().splitlines()]
        assert timing['full_pipeline_seconds'] < 12.
        assert answers[0].endswith('return 1')
        assert not timing['winner_reexecution']['mismatch_rows'] and timing['winner_reexecution']['complete']
        if case == 'priority_guard':
            assert answers[1].endswith('return 1') and answers[2].endswith('return 7')
            assert timing['deadline_guard']['partial_vote_rows'] == [1]
            assert timing['deadline_guard']['executed_sample_counts'] == [4, 4, 12]
        elif case == 'guard1':
            assert all(a.endswith('return 2') for a in answers[1:])
            assert timing['deadline_guard']['triggered'] and timing['deadline_guard']['partial_vote_count'] == 3
        else:
            assert all(a.endswith('return 7') for a in answers[1:])
            assert timing['adaptive']['stopped_rows'] == [0]
            assert timing['adaptive']['escalated_rows'] == [1, 2]
            assert timing['deadline_guard']['triggered'] == (case == 'guard2')
        if case == 'guard2':
            assert timing['deadline_guard']['partial_vote_rows'] == [1, 2]
            assert timing['deadline_guard']['executed_sample_counts'] == [4, 8, 8]
        if case == 'oom':
            assert len(timing['OOM_attempts']) == 1
            draws = [v for k, v in state['draws'] if k == 4]
            assert draws[0] == draws[1]
        if priority_case:
            assert timing['adaptive']['stage2_order'] == [2, 1]
            assert timing['adaptive']['stage1_leader_fractions'] == {1: .6, 2: .2}
            second = [b for b in timing['generation_batches'] if b['stage'] == 2]
            assert [(b['phase'], b['row_ids']) for b in second[:2]] == [('main', [2]), ('long', [2])]
        assert all(b['sequences'] <= cfg['sequence_budget'] for b in timing['generation_batches'] if b['stage'])
        return {'case': case, 'seconds': timing['full_pipeline_seconds'],
                'real_executor': case != 'priority', 'real_winner_reexecution': True,
                'adaptive': timing['adaptive'], 'partial': timing['deadline_guard']['partial_vote_count']}


def bundle_static():
    import nbformat
    metadata = read(BUNDLE/'kernel-metadata.json')
    notebook = read(BUNDLE/metadata['code_file'])
    nbformat.validate(nbformat.from_dict(notebook))
    assert (BUNDLE/metadata['code_file']).read_bytes() == (ROOT/'notebooks'/metadata['code_file']).read_bytes()
    assert metadata['id'] == 'friend1/nlp-hw02-adaptive-inference-l3-v5'
    import re
    assert re.sub(r'[^a-z0-9]+', '-', metadata['title'].lower()).strip('-') == metadata['id'].split('/')[1]
    assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_internet']
    assert metadata['machine_shape'] == 'NvidiaTeslaT4' and not metadata['enable_tpu']
    assert metadata['dataset_sources'] == ['friend1/nlp-hw02-l3-live', 'thisarivson/nlp-hw02-26b-inputs']
    code = [c['source'] for c in notebook['cells'] if c['cell_type'] == 'code']
    for c in notebook['cells']:
        if c['cell_type'] == 'code':
            assert not c['outputs'] and c['execution_count'] is None
            compile(c['source'], 'v5 cell', 'exec')
    sources = ast.literal_eval(next(n.value for n in ast.parse(code[0]).body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SOURCES'))
    identity = read(ROOT/'final-v5-source-identity.json')
    changed = {'final_inference.py', 'final_session.py', 'final_runner.py', 'final_adaptive.py'}
    for name, source in sources.items():
        compile(source, name, 'exec')
        assert hashlib.sha256(source.encode()).hexdigest() == identity['embedded_source_sha256'][name]
        assert '/Users/' not in source and '~/.kaggle' not in source and 'KAGGLE_KEY' not in source
        if name not in changed:
            assert hashlib.sha256(source.encode()).hexdigest() == identity['base_v4_sha256'][name]
    assert sources['final_adaptive.py'] == (SRC/'final_adaptive.py').read_text()
    assert sources['final_inference.py'] == (SRC/'final_inference_v5.py').read_text()
    worker = sources['final_inference.py']
    final = read(WORK/'results/l3-final-results.json')
    for name in ['C-main', 'C-long2x']:
        path = FINAL/f'model-{name}__weights__l2-62280-fp16.pt'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == final['receipts'][name]['FP16']['sha256'] and digest in worker
    session = sources['final_session.py']
    assert 'for number in range(1, 2):' in session and "launch('tune', rows_path, 2048" in session
    assert 'elapsed >= 660.' in session and 'elapsed >= 690.' in session
    assert "'budget_seconds': 720" in session and "'budget_seconds': 180." in session
    adaptive = identity['adaptive_config']
    assert "'scorer_enabled': False" in session and f"'K1': 2, 'K2': {adaptive['K2']}" in session
    assert adaptive['sequence_budget'] == adaptive['greedy_sequence_budget'] == 1024
    assert 'min(20., budget*.12)' in worker and 'min(14., budget*.08)' in worker
    assert 'min(10., budget*.06)' in worker  # 160/166/170 < supervisor172 <180.
    assert 'test_with_25s_margin' in session and 'target_155_seconds' in sources['final_runner.py']
    limits = read(BUNDLE/'session-limits.json')
    assert limits['session_timeout_seconds'] == 720 and limits['cold_test_runs'] == limits['tune_runs'] == 1
    return sources, {'passed': True, 'nbformat': True, 'private': True, 'internet': False,
                     'one_T4': True, 'L3_weights_SHA': True, 'frozen_decoder_executor': True, 'limits': limits}


def bundle_suite(sources):
    from unittest.mock import patch
    import importlib.util
    with tempfile.TemporaryDirectory(prefix='final-v5-suite-') as folder:
        folder = Path(folder)
        module = folder/'session_v5.py'
        module.write_text(sources['final_session.py'])
        spec = importlib.util.spec_from_file_location('suite_v5_check', module)
        session = importlib.util.module_from_spec(spec); spec.loader.exec_module(session)
        calls = []
        class FakeChild:
            pid, returncode = 99999999, 0
            def __init__(self, args, **kwargs):
                from final_vote import write_answers
                cfg = read(args[-1]); out = Path(cfg['output']); calls.append(cfg)
                adaptive = read(ROOT/'final-v5-source-identity.json')['adaptive_config']
                assert all(cfg[k] == v for k, v in adaptive.items())
                write_answers(out/'answers.jsonl', ['def solve():\n    return 1']*cfg['expected_rows'])
                session.write(out/'summary.json', {'status': 'complete', 'full_pipeline_seconds': 140.,
                    'under_180_seconds': True, 'target_155_seconds': True, 'deadline_guard': {'fallback_count': 0}})
                if 'reference' in cfg:
                    session.write(out/'quality.json', {'matches': adaptive['expected_matches'], 'passed': True})
            def poll(self):
                return 0
            def wait(self, **kwargs):
                return 0
        output = folder/'out'; output.mkdir()
        with patch.object(session.subprocess, 'Popen', FakeChild):
            report = session.suite(WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs', FINAL,
                                   output, SRC, session.Watchdog(output, time.monotonic()))
        assert report['status'] == 'passed' and report['GPU_runs'] == 2
        assert [Path(c['output']).name for c in calls] == ['test-1', 'tune']
        assert 'reference' not in calls[0] and 'reference' in calls[1]
        assert report['stored_tune']['matches'] == read(ROOT/'final-v5-source-identity.json')['adaptive_config']['expected_matches']
        assert (output/'answers.jsonl').read_bytes() == (output/'test-1/answers.jsonl').read_bytes()
        return {'one_test_then_tune': True, 'config_and_stored_replay': True,
                'no_gold_in_test': True, 'GPU_launches': 0, 'launches_mocked': True}


def main():
    ready = FINAL/'next-inference-v5.json'
    ready.write_text(json.dumps({'bundle': str(BUNDLE.resolve()), 'ready': False})+'\n')
    report = {'passed': False, 'unit': unit(), 'fixtures': [], 'GPU': False, 'push': False}
    for case in ['complete', 'guard1', 'guard2', 'oom', 'priority', 'priority_guard']:
        child = subprocess.run([sys.executable, __file__, '--fixture', case], capture_output=True, text=True, timeout=25.)
        assert child.returncode == 0, case+': '+child.stderr
        report['fixtures'].append(json.loads(child.stdout))
    sources, report['static'] = bundle_static()
    report['suite'] = bundle_suite(sources)
    offline = read(WORK/'checks/final-inference-v5-offline.json')
    k2 = offline['selected_K2']
    selected = next(r for r in offline['rows'] if r['K2'] == k2)
    assert offline['passed'] and selected['timing']['fits_15_second_margin']
    for filename in ['final-inference-v5-regressions.json', 'final-inference-v5-v4-regression.json',
                     'final-inference-v5-real-decoder.json']:
        assert read(WORK/'checks'/filename)['passed'], 'Run final_v5_regressions.py first'
    report['v3_v4_regressions'] = True
    # Independently execute all numeric chosen programs, not just the journal values.
    from final_v5_analysis import load_journals
    from final_adaptive import replay_record
    from final_vote import verify_winners
    from final_split import fixed_split
    models, _ = load_journals()
    winners = [replay_record(models['C-main'][i]['greedy'], models['C-main'][i]['samples']['0.7'],
               models['C-long2x'][i]['samples']['0.7'], k2=k2)[0] for i in fixed_split()['tune2048']]
    verification = verify_winners(winners, time.monotonic()+60., workers=4)
    assert verification['complete'] and not verification['mismatch_rows']
    report['all_winners_reexecution'] = verification
    report['passed'] = True
    report['exact_7471_with_frozen_vote'] = False
    report['frozen_vote_preserved'] = True
    checks = WORK/'checks/final-inference-v5-cpu.json'
    checks.write_text(json.dumps(report, indent=2)+'\n')
    # Current task authorizes readiness after technical gates; keep frozen voting.
    ready.write_text(json.dumps({'bundle': str(BUNDLE.resolve()), 'ready': True,
        'checks': str(checks.resolve()), 'GPU_gate': 'pending', 'push': False,
        'reason': 'CPU/static/offline/regressions passed; measured-v5 timing estimate; frozen v3/v4 vote',
        'kernel': 'friend1/nlp-hw02-adaptive-inference-l3-v5', 'K1': 2, 'K2': k2}, indent=2)+'\n')
    print(json.dumps({'CPU_static_passed': True, 'ready': True,
                      'K1': 2, 'K2': k2, 'winner_mismatches': 0}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', choices=['complete', 'guard1', 'guard2', 'oom', 'priority', 'priority_guard'])
    args = parser.parse_args()
    if args.fixture:
        print(json.dumps(fixture(args.fixture)))
    else:
        main()
