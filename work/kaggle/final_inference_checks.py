"""CPU tests, frozen tune replay and static notebook audit. Does not push or use GPU."""
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
SRC = ROOT/'src'
sys.path.insert(0, str(SRC))
from final_choose import choose
from final_vote import select, write_answers, validate_answers, CONFIG
from final_executor import Executor
from final_split import fixed_split
from final_representation import restore_compact


def read(path):
    return json.loads(path.read_text())


def candidate(value, lp=-1., code='sample'):
    return {'code': code, 'mean_logprob': lp,
            'execution': {'status': 'numeric', 'value': float(value)}}


def unit():
    g = candidate(1, 0, 'greedy')
    assert select(g, [candidate(2), candidate(2)], [candidate(2), candidate(1)])['code'] == 'sample'
    assert select(g, [candidate(2), candidate(2)], [candidate(1)]) is g
    bad = dict(g, execution={'status': 'syntax_error'})
    assert select(bad, [], []) is bad
    assert select(bad, [candidate(3, -2, 'lower'), candidate(3, -1, 'best')], [])['code'] == 'best'
    # Tie count → greedy, mean LP → order, and code greedy even with a lower LP.
    assert select(g, [candidate(1, 1, 'higher')], []) is g
    assert select(bad, [candidate(2, -1, 'first'), candidate(3, -1, 'second')], [])['code'] == 'first'
    assert select(bad, [candidate(2, -2), candidate(3, -1, 'lp')], [])['code'] == 'lp'
    chain = [candidate(0, -.1, 'a'), candidate(.9e-6, -.2, 'b'), candidate(1.8e-6, -.3, 'c')]
    assert select(bad, chain, [candidate(20), candidate(20)])['code'] == 'a'
    g_chain = candidate(0, -100, 'greedy')
    assert select(g_chain, [candidate(.9e-6, -.1), candidate(1.8e-6, -.2)], []) is g_chain
    # A cached repeated program is still several votes.
    assert select(g, [candidate(2)]*4, [candidate(1)]*2)['execution']['value'] == 2
    assert restore_compact('<|raw|>def solve():\n    return 3') == 'def solve():\n    return 3'
    assert restore_compact('<|header|>    return 2\n') == 'def solve() -> int:\n    return 2\n'
    assert restore_compact('<|header_tail|>    x = 1\n    x + 2') == 'def solve() -> int:\n    x = 1\n    result = x + 2\n    return result\n'
    for text in ['missing', '<|header_tail|>', '<|header_tail|>3']:
        try:
            restore_compact(text)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid compact accepted')
    with tempfile.TemporaryDirectory(prefix='final-format-') as folder:
        path = Path(folder)/'answers.jsonl'
        codes = ['print("строка")\n', '', 'def solve():\n    return 2']
        write_answers(path, codes)
        assert [r['code'] for r in validate_answers(path, 3)] == codes
    return {'vote_restore_format': 'passed', 'cases': 18}


def journals():
    raw = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l2/1'
    models = {}
    restored = 0
    for name in ['C-main', 'C-long2x']:
        folder = raw/f'model-{name}/points/l2-28800/evaluation/k8-tune2048'
        rows = [r for path in sorted(folder.glob('batch-*.json')) for r in read(path)]
        models[name] = {r['id']: r for r in rows}
        assert len(rows) == 2048 and sorted(models[name]) == fixed_split()['tune2048']
        for row in rows:
            for sample in row['samples']['0.7']:
                try:
                    code = restore_compact(sample['represented_code'])
                except (ValueError, IndexError):
                    code = ''
                assert code == sample['code']
                restored += 1
    ids = fixed_split()['tune2048']
    executions = []
    matches = 0
    codes = []
    candidates = []
    for i in ids:
        a, b = models['C-main'][i], models['C-long2x'][i]
        assert a['reference'] == b['reference'] and a['question_sha256'] == b['question_sha256']
        main, long = a['samples']['0.7'][:4], b['samples']['0.7'][:4]
        exact = choose({'greedy': a['greedy'], 'samples': {'0.7': main+long}}, CONFIG)
        win = select(a['greedy'], main, long)
        assert win['execution'] == exact['execution'], i
        assert win['code'] == exact['code'], i
        candidates.extend([a['greedy']]+main+long)
        executions.append(hashlib.sha256(json.dumps(win['execution'], sort_keys=True).encode()).hexdigest())
        codes.append(win['code'])
        ref = a['reference']
        matches += ref['status'] == win['execution']['status'] == 'numeric' and math.isclose(
            ref['value'], win['execution']['value'], rel_tol=1e-6, abs_tol=1e-6)
    saved = next(r for r in read(WORK/'results/l2-results.json')['voting_tune2048']['rows']
                 if r['name'] == 'mixed K4+K4 +greedy')
    digest = hashlib.sha256(''.join(executions).encode()).hexdigest()
    assert digest == saved['chosen_execution_sha256'] and matches == saved['matches'] == 1492
    return {'rows': 2048, 'matches': matches, 'accuracy_pp': 100*matches/2048,
            'execution_mismatches': 0, 'code_mismatches': 0, 'restored_samples': restored,
            'chosen_execution_sha256': digest,
            'chosen_code_sha256': hashlib.sha256(json.dumps(codes, ensure_ascii=False).encode()).hexdigest(),
            'candidate_count': len(candidates), 'unique_programs': len({c['code'] for c in candidates})}, candidates


def execution_checks(candidates):
    import final_numeric as numeric
    test_codes = ['def solve():\n    return 3', 'def solve():\n    return True', 'bad python :',
                  'import os\ndef solve():\n    return 1', 'def solve():\n    return 1/0',
                  'def solve():\n    return float("nan")']
    items = [{'code': c} for c in test_codes+test_codes]
    executor = Executor(workers=2)
    try:
        executor.submit(items[:6]); executor.submit(items[6:])
        assert executor.drain(time.monotonic()+15)
        executor.attach(items)
        assert len(executor.codes) == 6 and executor.requests == 12
        for item in items:
            expected = numeric.execute_task((item['code'], 3., 2.))
            assert item['execution'] == expected
        loop = [{'code': 'def solve():\n    while True:\n        pass'}]
        executor.submit(loop)
        assert executor.drain(time.monotonic()+10.)
        executor.attach(loop)
        assert loop[0]['execution']['status'] in ('timeout', 'timeout_or_worker_exit')
    finally:
        executor.close()
    # Same numeric values/statuses on ALL unique mixed programs, independent of dedup/overlap.
    unique = {}
    conflicts = []
    for c in candidates:
        if c['code'] in unique and unique[c['code']]['execution'] != c['execution']:
            conflicts.append(c['code'])
        else:
            unique[c['code']] = c
    executor = Executor()  # os.cpu_count, just like Kaggle
    started = time.monotonic()
    try:
        items = [{'code': code} for code in unique]
        # Incremental batches mirror the GPU service submission, without waiting between batches.
        for lo in range(0, len(items), 256):
            executor.submit(items[lo:lo+256])
        assert executor.drain(time.monotonic()+300), 'Full CPU execution audit incomplete: '+json.dumps(executor.summary())
        executor.attach(items)
        mismatches = [{'code_sha256': hashlib.sha256(c['code'].encode()).hexdigest(),
                       'saved': unique[c['code']]['execution'], 'new': c['execution']}
                      for c in items if c['execution'] != unique[c['code']]['execution']]
        # Only nonnumeric timeout outcomes can vary with host scheduling. Numeric equivalence is mandatory.
        numeric_mismatches = [m for m in mismatches if m['saved']['status'] == 'numeric' or m['new']['status'] == 'numeric']
        assert not numeric_mismatches, numeric_mismatches[:3]
        return {'small_equivalence': True, 'unique_programs': len(items), 'workers': executor.workers,
                'numeric_mismatches': len(numeric_mismatches), 'status_mismatches': mismatches,
                'saved_duplicate_execution_conflicts': len(conflicts), 'seconds': time.monotonic()-started,
                'infinite_loop_bounded_and_worker_recovered': True}
    finally:
        executor.close()


def sampler_checks():
    import torch
    import exp26_core as original_core
    import exp27b_fast as original_fast
    import final_core as core
    import final_fast as fast
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = core.SimpleTransformer(d_model=16, heads=2, encoder_layers=1, decoder_layers=1,
                                  swiglu_dim=24, vocab=32, max_positions=64, dropout=0.).eval()
    src = torch.tensor([[1, 5, 2, 0], [1, 7, 8, 2]])
    for k, temperature in [(1, None), (4, .7), (8, .7)]:
        torch.manual_seed(270042)
        ctx = original_fast.Context(model, src, 32)
        old_ids, old_lp, _ = original_fast.decode(ctx, k, temperature)
        torch.manual_seed(270042)
        ctx = fast.Context(model, src, 32)
        new_ids, new_lp, _ = fast.decode(ctx, k, temperature)
        assert torch.equal(old_ids, new_ids) and torch.equal(old_lp, new_lp)
        if temperature is None:
            no_lp_ids, no_lp, _ = fast.decode(fast.Context(model, src, 32), compute_logprob=False)
            assert torch.equal(old_ids, no_lp_ids) and torch.count_nonzero(no_lp) == 0
    # Compare physical k1 against the existing full model/CachedDecoder mathematics.
    with torch.inference_mode():
        physical = fast.Decoder(fast.Context(model, src, 32), 1)
        cached = original_core.CachedDecoder(model, src, 32)
        for tokens in [torch.tensor([1, 1]), torch.tensor([5, 7]), torch.tensor([2, 2])]:
            assert torch.allclose(physical.step(tokens), cached.step(tokens), atol=1e-6, rtol=1e-6)
    return {'original_physical_sampler_token_and_LP_identical': True, 'K': [1, 4, 8],
            'greedy_without_LP_tokens_identical': True,
            'cached_logits_allclose_1e_6': True, 'device': 'CPU FP32 tiny; CUDA not tested'}


def process_smoke():
    import torch
    from final_core import SimpleTransformer
    inputs = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    with tempfile.TemporaryDirectory(prefix='final-process-') as folder:
        folder = Path(folder)
        spec = {'d_model': 16, 'heads': 2, 'encoder_layers': 1, 'decoder_layers': 1,
                'swiglu_dim': 24, 'vocab': 8195, 'max_positions': 2048}
        model = SimpleTransformer(**spec, dropout=0.)
        path = folder/'tiny.pt'
        torch.save({'format': 'nlp-hw02-model-only-v1', 'optimizer_included': False,
                    'model_spec': spec, 'model': model.state_dict()}, path)
        rows = folder/'rows.json'; rows.write_text(json.dumps([{'question': q} for q in ['2+2?', '3*5?', '7-2?']]))
        config = {'rows': str(rows), 'tokenizer': str(inputs/'tokenizer.json'),
                  'models': {'C-main': str(path), 'C-long2x': str(path)}, 'expected_rows': 3,
                  'output': str(folder/'run'), 'verify_weights': False, 'device': 'cpu',
                  'workers': 2, 'cap': 8, 'budget_seconds': 40., 'greedy_batch': 2, 'sample_batch': 2}
        cp = folder/'config.json'; cp.write_text(json.dumps(config))
        env = dict(os.environ, FINAL_PROCESS_START=str(time.monotonic()))
        child = subprocess.run([sys.executable, str(SRC/'final_runner.py'), str(cp)],
                               capture_output=True, text=True, timeout=45, env=env)
        if child.returncode:
            raise AssertionError((folder/'run/worker.log').read_text()+child.stderr)
        report = read(folder/'run/summary.json')
        assert report['status'] == 'complete' and report['rows'] == 3
        assert report['deadline_guard']['fallback_count'] == 0
        assert report['winner_reexecution']['mismatch_rows'] == []
        validate_answers(folder/'run/answers.jsonl', 3)
        assert abs(sum(report['phases_seconds'].values())-report['full_pipeline_seconds']) < .1
        stale = subprocess.run([sys.executable, str(SRC/'final_runner.py'), str(cp)],
                               capture_output=True, text=True, timeout=5)
        assert stale.returncode != 0 and 'Fresh cold run' in stale.stderr
        config.update(output=str(folder/'guard'), cpu_test_expire_sampling=True)
        cp.write_text(json.dumps(config))
        child = subprocess.run([sys.executable, str(SRC/'final_runner.py'), str(cp)],
                               capture_output=True, text=True, timeout=45,
                               env=dict(os.environ, FINAL_PROCESS_START=str(time.monotonic())))
        assert child.returncode == 0, child.stderr
        guard = read(folder/'guard/summary.json')
        assert guard['status'] == 'guarded' and guard['deadline_guard']['fallback_rows'] == [0, 1, 2]
        assert (folder/'guard/answers.jsonl').read_bytes() == (folder/'guard/greedy-fallback.jsonl').read_bytes()
        return {'passed': True, 'cold_subprocess_seconds': report['full_pipeline_seconds'],
                'all_phases_in_timer': True, 'sample_candidates': report['execution']['requested_candidates'],
                'deadline_fallback_greedy_exact': True, 'guard_seconds': guard['full_pipeline_seconds'],
                'stale_output_rejected': True}


def static():
    notebook = read(ROOT/'notebooks/nlp-hw02-final-inference.ipynb')
    assert notebook['nbformat'] == 4
    embedded = None
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            assert cell['outputs'] == [] and cell['execution_count'] is None
            tree = ast.parse(source); compile(tree, f'cell-{i}', 'exec')
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in node.targets):
                    embedded = ast.literal_eval(node.value)
    assert embedded
    for name, source in embedded.items():
        assert source == (SRC/name).read_text()
        compile(source, name, 'exec')
        assert '/Users/' not in source and 'kaggle.json' not in source and 'KAGGLE_KEY' not in source
    for target, source, definitions in [
        ('final_choose.py', 'exp27_decode.py', ['choose']),
        ('final_fast.py', 'exp27b_fast.py', ['Context', 'Decoder']),
        ('final_representation.py', 's3_representation.py', ['restore_compact', 'tokenizer']),
        ('final_split.py', 'l2_sampler.py', ['fixed_split']),
        ('final_core.py', 'exp26_core.py', ['SwiGLUInput', 'replace_ffn_with_swiglu', 'SimpleTransformer',
                                         'encode', 'decode_bytes', 'active_tokens', 'BudgetExpired', 'check_deadline', 'sync'])]:
        a, b = ast.parse((SRC/target).read_text()), ast.parse((SRC/source).read_text())
        for definition in definitions:
            nodes = [next(n for n in t.body if getattr(n, 'name', None) == definition) for t in [a, b]]
            assert ast.dump(nodes[0]) == ast.dump(nodes[1]), definition
    bundle = ROOT/'bundles/final_inference'
    metadata = read(bundle/'kernel-metadata.json')
    assert metadata['id'] == 'thisarivson/nlp-hw02-final-inference'
    assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_internet']
    assert set(metadata['dataset_sources']) == {'thisarivson/nlp-hw02-l2-live', 'thisarivson/nlp-hw02-26b-inputs'}
    assert read(bundle/'session-limits.json')['session_timeout_seconds'] == 900
    assert (bundle/metadata['code_file']).read_bytes() == (ROOT/'notebooks'/metadata['code_file']).read_bytes()
    try:
        import nbformat
    except ImportError:
        nbformat_validation = 'unavailable; structural/AST checks passed'
    else:
        nbformat.validate(nbformat.read(ROOT/'notebooks'/metadata['code_file'], as_version=4))
        nbformat_validation = 'passed'
    return {'passed': True, 'code_cells': sum(c['cell_type'] == 'code' for c in notebook['cells']),
            'embedded_modules': len(embedded), 'frozen_function_AST_equivalence': True,
            'private': True, 'GPU_count_in_worker': 1, 'internet': False, 'nbformat': nbformat_validation,
            'notebook_bytes': (ROOT/'notebooks'/metadata['code_file']).stat().st_size}


def suite_checks():
    import gzip
    import zipfile
    from unittest.mock import patch
    import final_session as session
    inputs = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    live = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l2/1/live-v11'
    with tempfile.TemporaryDirectory(prefix='final-suite-') as folder:
        folder = Path(folder)
        archive_root = folder/'archive'; archive_root.mkdir()
        with zipfile.ZipFile(archive_root/'26b-training-inputs-d0-initial-and-evaluation.zip', 'w') as archive:
            for name in session.INPUT_SHA:
                archive.write(inputs/name, 'starter/'+name)
            archive.writestr('starter/initial.pt', 'unused large-weight fixture')
        prepared = dict(session.prepare_inputs(archive_root, folder/'prepared'))
        assert set(prepared) == set(session.INPUT_SHA)
        assert sorted(p.name for p in (folder/'prepared').iterdir()) == sorted(session.INPUT_SHA)
        launch_history = []
        match_count = 1492
        identical = True
        class FakeChild:
            returncode = 0
            pid = 99999999
            def __init__(self, args, **kwargs):
                cfg = read(Path(args[-1])); out = Path(cfg['output'])
                launch_history.append((out.name, cfg, kwargs['env']['FINAL_PROCESS_START']))
                # Distinct test files demonstrate that suite publishes first run exactly.
                write_answers(out/'answers.jsonl', [('test' if identical and out.name.startswith('test-') else out.name)+'-'+str(i)
                    for i in range(cfg['expected_rows'])])
                summary = {'status': 'complete', 'rows': cfg['expected_rows'], 'full_pipeline_seconds': 120.,
                           'under_180_seconds': True, 'target_150_seconds': True, 'target_120_seconds': True,
                           'deadline_guard': {'fallback_count': 0}}
                session.write(out/'summary.json', summary)
                if cfg.get('reference'):
                    matches = 1492 if cfg.get('l2_replay') else match_count
                    session.write(out/'quality.json', {'matches': matches, 'passed': abs(100*(matches-1492)/2048) <= 1.})
            def poll(self):
                return 0
            def wait(self, **kwargs):
                return 0
        with patch.object(session.subprocess, 'Popen', FakeChild):
            out = folder/'suite'; out.mkdir()
            report = session.suite(inputs, live, out, SRC, session.Watchdog(out, time.monotonic()))
            assert report['status'] == 'passed' and report['GPU_runs'] == 3
            assert [name for name, _, _ in launch_history] == ['test-1', 'test-2', 'tune']
            assert len({marker for _, _, marker in launch_history}) == 3
            assert (out/'answers.jsonl').read_bytes() == (out/'test-1/answers.jsonl').read_bytes()
            launch_history.clear(); match_count = 1470
            out = folder/'quality-fail'; out.mkdir()
            report = session.suite(inputs, live, out, SRC, session.Watchdog(out, time.monotonic()))
            assert report['status'] == 'review_required' and report['GPU_runs'] == 3
            assert launch_history[-1][0] == 'tune'
            identical = False; match_count = 1492; launch_history.clear()
            out = folder/'nondeterministic'; out.mkdir()
            report = session.suite(inputs, live, out, SRC, session.Watchdog(out, time.monotonic()))
            assert report['status'] == 'review_required' and not report['gates']['deterministic_answers']
            assert (out/'answers.jsonl').read_bytes() == (out/'test-1/answers.jsonl').read_bytes()
            launch_history.clear()
            out = folder/'submission'; out.mkdir()
            report = session.suite(inputs, live, out, SRC, session.Watchdog(out, time.monotonic()), benchmark=False)
            assert report['status'] == 'complete' and report['GPU_runs'] == 1
    return {'passed': True, 'selective_ZIP_without_initial_weights': True,
            'two_cold_test_then_tune': True, 'first_answers_preserved': True,
            'quality_fail_does_not_spend_extra_GPU_run': True, 'single_test_submission_mode': True,
            'GPU_launches_in_this_check': 0, 'launches_mocked': True}


def supervisor_checks():
    from unittest.mock import patch
    import final_runner as runner
    with tempfile.TemporaryDirectory(prefix='final-hard-stop-') as folder:
        folder = Path(folder)
        out = folder/'hard'; out.mkdir()
        cfg = folder/'config.json'; cfg.write_text(json.dumps({'output': str(out), 'budget_seconds': 180.}))
        publish_greedy = True
        class HungChild:
            pid = 99999999
            returncode = None
            def __init__(self, args, **kwargs):
                if publish_greedy:
                    write_answers(out/'greedy-fallback.jsonl', ['first', 'second'])
                    (out/'greedy-ready.json').write_text(json.dumps({'rows': 2}))
            def poll(self):
                return None
            def wait(self, **kwargs):
                self.returncode = -9
                return -9
        with patch.object(runner.subprocess, 'Popen', HungChild), \
             patch.object(runner.os, 'killpg') as kill, \
             patch.object(runner.time, 'monotonic', return_value=175.):
            report = runner.supervise(cfg, start=0.)
            assert kill.called and report['status'] == 'guarded' and report['hard_stop']
            assert report['full_pipeline_seconds'] == 175.
            assert report['deadline_guard']['fallback_count'] == 2
            assert (out/'answers.jsonl').read_bytes() == (out/'greedy-fallback.jsonl').read_bytes()
            out = folder/'pre-greedy'; out.mkdir(); publish_greedy = False
            cfg.write_text(json.dumps({'output': str(out), 'budget_seconds': 180.}))
            try:
                runner.supervise(cfg, start=0.)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Pre-greedy startup hang must fail')
            assert not (out/'answers.jsonl').exists()
            assert read(out/'summary.json')['status'] == 'failed_before_greedy'
    return {'external_hard_stop_publishes_exact_greedy': True,
            'startup_before_greedy_fails_without_fabrication': True, 'process_and_clock_mocked': True}


def main():
    report = {'GPU_used': False, 'Kaggle_push': False, 'new_training': False}
    from final_review_checks import regression_checks
    report['review_regressions'] = regression_checks()
    report['unit'] = unit()
    report['offline_vote'], candidates = journals()
    report['static'] = static()
    report['suite'] = suite_checks()
    report['supervisor'] = supervisor_checks()
    report['sampler'] = sampler_checks()
    report['execution'] = execution_checks(candidates)
    report['process_smoke'] = process_smoke()
    target = WORK/'checks/final-inference-cpu.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
