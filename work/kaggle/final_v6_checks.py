"""V6 local gates and fail-closed bundle readiness, no GPU or Kaggle calls."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from build_final_inference_v6 import BUNDLE, READY, NAME, build
from final_v6_analysis import FINAL, read


def fixture(case):
    """Run identical v5 guard/priority/OOM assertions against the v6 production worker."""
    import final_v5_checks as v5
    source = inspect.getsource(v5.fixture)
    source = source.replace('import final_inference_v5 as I', 'import final_inference_v6 as I')
    source = source.replace("patch.object(I.Fast, 'decode'", "patch.object(I.Speed, 'decode'")
    source = source.replace('        def close(self):', '        def resize(self, workers):\n            pass\n        def close(self):')
    namespace = dict(v5.__dict__)
    exec(compile(source, 'v5-fixture-against-v6', 'exec'), namespace)
    return namespace['fixture'](case)


def real_pipeline():
    import final_v5_regressions as v5
    source = inspect.getsource(v5.real_decoder).replace('import final_inference_v5 as I', 'import final_inference_v6 as I')
    source = source.replace('final-inference-v5-offline.json', 'final-inference-v6-offline.json')
    ns = dict(v5.__dict__)
    exec(compile(source, 'real-v6-pipeline', 'exec'), ns)
    return ns['real_decoder']()


def decoder():
    import torch
    import final_profile_checks as checks
    import final_inference_v6 as inference
    import final_profile as profile
    assert asdict(inference.DECODE_OPTIONS) == asdict(profile.Speed.Options(**profile.VARIANTS['all']))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    report = {'cache': checks.cache_tests(), 'tiny_decoder': checks.decoder_tests()}
    source = inspect.getsource(checks.real_cpu)
    source = source.replace("['share_cross', 'async_eos', 'compact_done', 'race_sampler', 'lean_greedy', 'all']", "['all']")
    ns = dict(checks.__dict__)
    exec(compile(source, 'real-v6-CPU-subset', 'exec'), ns)
    report['real_L3_FP32_subset'] = ns['real_cpu']()
    # With a degenerate categorical distribution all samplers must choose the
    # same tokens, including completed lanes and the untempered mean LP.
    from final_profile_checks import tiny, equal
    import final_fast as Fast
    import final_speed as Speed
    import final_core as C
    model = tiny()
    with torch.no_grad():
        model.head.weight.zero_(); model.head.bias.fill_(-1000.); model.head.bias[C.EOS] = 1000.
    src = torch.tensor([[1, 5, 2], [1, 8, 2]])
    a = Fast.decode(Fast.Context(model, src, 48), 6, .7)
    b = Speed.decode(Fast.Context(model, src, 48), 6, .7, options=inference.DECODE_OPTIONS)
    assert equal(a[0], b[0], 48) and torch.equal(a[1], b[1])
    report.update(passed=True, production_all_options_identical=True, K6_one_hot_sampling_tokens_LP_exact=True,
                  GPU=False, async_CUDA_branch='covered by existing T4 profile; CPU uses synchronous fallback')
    return report


def executor():
    from final_executor_v6 import Executor
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    e = Executor(2)
    jobs = [{'code': f'def solve():\n    x = {i}\n    for j in range(25000):\n        x += j\n    return x'} for i in range(32)]
    try:
        e.submit(jobs[:16]); e.resize(4); e.submit(jobs[16:]+jobs[:4])
        assert e.drain(time.monotonic()+25.)
        e.attach(jobs)
        assert all(j['execution'] == {'status': 'numeric', 'value': float(i+24999*25000//2)} for i, j in enumerate(jobs))
        report = e.summary()
        assert report['worker_history'] == [2, 4] and report['service_state']['workers'] == 4
        assert report['unique_programs'] == report['completed_unique'] == 32 and report['deduplicated'] == 4
        assert report['service_state']['nice_requested_increment'] == 10
        assert report['service_state']['nice_applied'] or report['service_state']['nice_error']
        assert os.getpriority(os.PRIO_PROCESS, 0) == priority
        assert not report['error']
        return {'passed': True, 'pending_queue_and_cache_preserved': True, 'parent_priority_unchanged': True, **report}
    finally:
        e.close()


def static():
    import nbformat
    metadata = read(BUNDLE/'kernel-metadata.json'); notebook = read(BUNDLE/NAME)
    nbformat.validate(nbformat.from_dict(notebook))
    assert (ROOT/'notebooks'/NAME).read_bytes() == (BUNDLE/NAME).read_bytes()
    codes = [c['source'] for c in notebook['cells'] if c['cell_type'] == 'code']
    for c in notebook['cells']:
        if c['cell_type'] == 'code':
            compile(c['source'], c['id'], 'exec'); assert not c['outputs'] and c['execution_count'] is None
    sources = ast.literal_eval(next(n.value for n in ast.parse(codes[0]).body if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'SOURCES'))
    identity = read(ROOT/'final-v6-source-identity.json')
    assert identity == read(BUNDLE/'source-identity.json')
    assert identity['notebook_sha256'] == hashlib.sha256((BUNDLE/NAME).read_bytes()).hexdigest()
    changed = {'final_inference.py', 'final_session.py', 'final_adaptive.py'}
    for name, source in sources.items():
        compile(source, name, 'exec')
        assert hashlib.sha256(source.encode()).hexdigest() == identity['embedded_source_sha256'][name]
        assert not any(token in source for token in ['/Users/', 'KAGGLE_KEY', '~/.kaggle', 'p1_'])
        if name in identity['base_v5_sha256'] and name not in changed:
            assert identity['base_v5_sha256'][name] == identity['embedded_source_sha256'][name]
    for n in ['final_speed.py', 'final_adaptive.py', 'final_execute_v6.py', 'final_executor_v6.py']:
        assert sources[n] == (ROOT/'src'/n).read_text()
    assert sources['final_inference.py'] == (ROOT/'src/final_inference_v6.py').read_text()
    assert metadata['id'] == 'thisarivson/nlp-hw02-adaptive-inference-l3-v6'
    assert re.sub('[^a-z0-9]+', '-', metadata['title'].lower()).strip('-') == metadata['id'].split('/')[1]
    assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_tpu'] and not metadata['enable_internet']
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['dataset_sources'] == ['thisarivson/nlp-hw02-l3-live', 'thisarivson/nlp-hw02-26b-inputs']
    session = sources['final_session.py']; worker = sources['final_inference.py']
    assert 'for number in range(1, 2):' in session and "launch('tune', rows_path, 2048" in session
    assert "'budget_seconds': 720" in session and "'budget_seconds': 180." in session
    assert 'elapsed >= 660.' in session and 'elapsed >= 690.' in session
    assert "< 480." in session and '>= start+budget-min(8.' in sources['final_runner.py']
    assert 'min(20., budget*.12)' in worker and 'min(14., budget*.08)' in worker and 'min(10., budget*.06)' in worker
    assert 'executor.resize(tail_workers)' in worker and 'Speed.decode(ctx, k' in worker
    assert "generation_workers = 2" in worker and 'min(4, os.cpu_count() or 1)' in worker
    assert "test_with_30s_margin" in session
    limits = read(BUNDLE/'session-limits.json')
    assert limits['cold_test_runs'] == limits['tune_runs'] == 1 and limits['session_timeout_seconds'] == 720
    assert limits['adaptive']['K2'] == read(WORK/'checks/final-inference-v6-offline.json')['selected_K2']
    for name, expected in identity['weights']['sha256'].items():
        path = FINAL/identity['weights']['filenames'][name]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    status = subprocess.run(['git', 'diff', '--name-only', 'HEAD'], capture_output=True, text=True, check=True).stdout
    assert not any('/p1_' in p or '/p1-' in p or p.endswith('/build_p1.py') for p in status.splitlines())
    return sources, {'passed': True, 'nbformat_compile_identity_frozen_base': True,
        'private_offline_T4_account_mounts_title': True, 'budgets_watchdog_one_test_one_tune': True,
        'L3_weight_SHA': True, 'P1_unchanged': True}


def suite(sources, custom=False):
    with tempfile.TemporaryDirectory(prefix='v6-suite-') as folder:
        folder = Path(folder); module = folder/'suite_v6.py'; module.write_text(sources['final_session.py'])
        spec = importlib.util.spec_from_file_location('suite_v6', module)
        session = importlib.util.module_from_spec(spec); spec.loader.exec_module(session)
        calls = []; config = read(ROOT/'final-v6-source-identity.json')['adaptive_config']
        identity = read(ROOT/'final-v6-source-identity.json')
        class Child:
            pid, returncode = 99999999, 0
            def __init__(self, args, **kwargs):
                from final_vote import write_answers
                cfg = read(Path(args[-1])); calls.append(cfg); out = Path(cfg['output'])
                assert all(cfg[k] == v for k, v in config.items()) and cfg['workers'] == 2
                assert cfg['weight_sha256'] == identity['weights']['sha256']
                assert cfg['verify_weights'] == bool(identity['weights']['sha256'])
                write_answers(out/'answers.jsonl', ['def solve():\n    return 1']*cfg['expected_rows'])
                session.write(out/'summary.json', {'status': 'complete', 'full_pipeline_seconds': 126.,
                    'under_180_seconds': True, 'target_150_seconds': True, 'deadline_guard': {'fallback_count': 0}})
                if 'reference' in cfg:
                    session.write(out/'quality.json', {'matches': config['expected_matches'], 'passed': True})
            def poll(self): return 0
            def wait(self, **kwargs): return 0
        output = folder/'out'; output.mkdir()
        live = FINAL
        if custom:
            live = folder/'weights'; live.mkdir()
            for name in identity['weights']['filenames'].values(): (live/name).touch()
        with patch.object(session.subprocess, 'Popen', Child):
            report = session.suite(WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs', live,
                                   output, ROOT/'src', session.Watchdog(output, time.monotonic()))
        assert report['status'] == 'passed' and report['GPU_runs'] == 2
        assert [Path(c['output']).name for c in calls] == ['test-1', 'tune']
        assert 'reference' not in calls[0] and 'reference' in calls[1]
        assert (output/'answers.jsonl').read_bytes() == (output/'test-1/answers.jsonl').read_bytes()
        return {'passed': True, 'one_cold_test_then_tune_mocked': True, 'no_gold_in_test': True,
                'custom_pair_without_L3_journals': custom}


def custom_builder():
    # Build through the public CLI parameter API into temporary targets only.
    import build_final_inference_v6 as builder
    with tempfile.TemporaryDirectory(prefix='v6-custom-build-') as tmp:
        tmp = Path(tmp); root = tmp/'kaggle'; (root/'notebooks').mkdir(parents=True)
        for name in ['src', 'final-v5-source-identity.json']:
            (root/name).symlink_to(ROOT/name, target_is_directory=(name == 'src'))
        (root/'notebooks/nlp-hw02-final-inference-l3-v5.ipynb').symlink_to(ROOT/'notebooks/nlp-hw02-final-inference-l3-v5.ipynb')
        with patch.object(builder, 'ROOT', root), patch.object(builder, 'BUNDLE', tmp/'bundle'), patch.object(builder, 'READY', tmp/'ready.json'):
            _, sources = builder.build('example/alternate-pair', 'branch-a-fp16.pt', 'branch-b-fp16.pt')
        metadata = read(tmp/'bundle/kernel-metadata.json'); identity = read(root/'final-v6-source-identity.json')
        assert metadata['dataset_sources'][0] == 'example/alternate-pair'
        assert "LIVE_ROOT=find_input('alternate-pair')" in read(tmp/'bundle'/NAME)['cells'][2]['source']
        assert not identity['weights']['sha256'] and not identity['weights']['L3_journal_diagnostic']
        # suite() reads identity from ROOT; point it at this temporary build for the mock.
        with patch.dict(globals(), ROOT=root):
            # Source modules are still on sys.path; suite's module_dir is only passed to mock.
            result = suite(sources, custom=True)
        return result


def child(mode, value=None):
    args = [sys.executable, __file__, '--child', mode]+(['--case', value] if value else [])
    r = subprocess.run(args, capture_output=True, text=True, timeout=180.)
    assert r.returncode == 0, f'{mode}/{value}: {r.stderr}\n{r.stdout}'
    return json.loads(r.stdout)


def main():
    READY.write_text(json.dumps({'bundle': str(BUNDLE.resolve()), 'ready': False})+'\n')
    report = {'GPU': False, 'push': False, 'fixtures': []}
    assert read(WORK/'checks/final-inference-v6-regressions.json')['passed']
    report['v3_v4_v5_regressions'] = True
    for case in ['complete', 'guard1', 'guard2', 'oom', 'priority', 'priority_guard']:
        report['fixtures'].append(child('fixture', case))
    for mode in ['real_pipeline', 'decoder', 'executor']:
        report[mode] = child(mode); print(mode, 'passed', flush=True)
    sources, report['static'] = static()
    report['suite'] = suite(sources)
    report['custom_weights_builder'] = custom_builder()
    from final_v5_analysis import load_journals
    from final_adaptive import replay_record
    from final_split import fixed_split
    from final_executor_v6 import verify_winners
    models, _ = load_journals()
    offline = read(WORK/'checks/final-inference-v6-offline.json'); k = offline['selected_K2']
    assert offline['passed']
    winners = [replay_record(models['C-main'][i]['greedy'], models['C-main'][i]['samples']['0.7'],
        models['C-long2x'][i]['samples']['0.7'], k2=k)[0] for i in fixed_split()['tune2048']]
    report['all_winners_reexecution'] = verify_winners(winners, time.monotonic()+60., 4)
    assert report['all_winners_reexecution']['complete'] and not report['all_winners_reexecution']['mismatch_rows']
    report.update(passed=True, K1=2, K2=k, known_FP16_greedy_difference_accepted=offline['known_FP16_shape_difference'])
    receipt = WORK/'checks/final-inference-v6-cpu.json'
    receipt.write_text(json.dumps(report, indent=2)+'\n')
    READY.write_text(json.dumps({'bundle': str(BUNDLE.resolve()), 'ready': True,
        'checks': str(receipt.resolve()), 'kernel': 'thisarivson/nlp-hw02-adaptive-inference-l3-v6',
        'K1': 2, 'K2': k, 'GPU_gate': 'pending full v6 cold test/tune', 'push': False,
        'notebook_sha256': read(ROOT/'final-v6-source-identity.json')['notebook_sha256'],
        'readiness': 'local CPU/offline/regressions/static gates passed; no push'}, indent=2)+'\n')
    print('V6 gates passed; ready:true; no push', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child', choices=['fixture', 'real_pipeline', 'decoder', 'executor'])
    parser.add_argument('--case')
    args = parser.parse_args()
    if args.child: print(json.dumps(fixture(args.case) if args.child == 'fixture' else globals()[args.child]()))
    else: main()
