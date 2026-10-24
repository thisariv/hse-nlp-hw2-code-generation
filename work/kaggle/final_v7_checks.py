"""V7 local CPU/replay/memory-budget/static gates. No GPU or kernel push.

ready:true means locally verified and ready for the single cold T4 gate. GPU
peak memory, real new-sample quality and full cold runtime remain pending.
"""
import argparse
import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from build_final_inference_v7 import BUNDLE, READY, NAME, default_config
from final_v7_analysis import load_all, specs, winners_for, FINAL, P1, PROFILE
from final_ensemble import parameters, stage1_decision, stage1_leader_fraction, select_available, replay_record
from final_split import fixed_split


def read(path):
    return json.loads(Path(path).read_text())


def unit():
    config = default_config(); models, _ = parameters(config)
    def c(value, lp=-1., status='numeric', code=None):
        return {'code': code or f'def solve():\n    return {value}', 'mean_logprob': lp,
                'represented_code': f'<|raw|>def solve():\n    return {value}',
                'execution': {'status': status, 'value': value}}
    for bad in [[], models[:1], models+[models[0], models[0]], dict(a=1),
                [dict(models[0], name='greedy'), models[1]], [dict(models[0], filename='../a.pt'), models[1]],
                [dict(models[0], sha256='x'), models[1]], [dict(models[0], K1=True), models[1]],
                [dict(models[0], K2=0), models[1]], [dict(models[0], dataset=3), models[1]],
                [models[0], models[0]]]:
        try: parameters({'models': bad})
        except ValueError: pass
        else: raise AssertionError(bad)
    for t in [0, True, float('nan'), 2]:
        try: parameters(dict(config, unanimity_threshold=t))
        except ValueError: pass
        else: raise AssertionError(t)
    g = c(1., 0.); record = {'greedy': g, 'samples': {m['name']: [c(1.)] for m in models}}
    assert stage1_decision(record, models) is True and stage1_leader_fraction(record, models) == 1.
    last = models[-1]['name']; record['samples'][last] = [c(2.)]
    assert stage1_decision(record, models) is False and stage1_leader_fraction(record, models) == .75
    assert stage1_decision(record, models, .75) is True
    record['samples'][last] = [c(0., status='budget_not_evaluated')]
    assert stage1_decision(record, models) is None
    record['samples'][last] = []
    assert stage1_decision(record, models) is None
    for value, status in [(2., 'timeout'), (float('inf'), 'numeric'), (float('nan'), 'numeric')]:
        record['samples'][last] = [c(value, status=status)]
        assert stage1_decision(record, models) is False
    # All failures stay in the denominator; agreement uses graph-isclose components.
    record = {'greedy': c(0.), 'samples': {models[0]['name']: [c(.9e-6)],
              models[1]['name']: [c(1.8e-6)], last: [c(99., status='timeout')]}}
    assert stage1_leader_fraction(record, models) == .75
    assert stage1_decision(record, models) is False
    winner, count = select_available(record, models)
    assert winner is record['greedy'] and count == 3
    # Mean LP and stable MODEL order break tied nongreedy components.
    record = {'greedy': c(0., status='timeout'), 'samples': {models[0]['name']: [c(2., -.2)],
              models[1]['name']: [c(3., -.1)], last: [c(9., status='budget_not_evaluated')]}}
    assert select_available(record, models)[0]['execution']['value'] == 3.
    record['samples'][models[0]['name']][0]['mean_logprob'] = -.1
    assert select_available(record, models)[0]['execution']['value'] == 2.
    record['samples'][models[0]['name']][0]['mean_logprob'] = None
    assert select_available(record, models)[0]['execution']['value'] == 3.
    # Available executions count even when their LP excludes them from voting.
    assert select_available(record, models)[1] == 2
    called = []
    select_available(record, models, scorer=lambda *a: called.append(a), score_weight=0.)
    assert not called
    return {'passed': True, 'ordered_2_to_4_config_validation': True,
            'pending_failures_transitivity_LP_order_partial_greedy_code': True, 'disabled_scorer_not_called': True}


def fixture(case):
    import torch
    import final_inference_v7 as I
    n = 2 if case == 'complete2' else 4 if case == 'complete4' else 3
    config = default_config()
    model_specs = config['models'][:n]
    if n == 4:
        model_specs.append(dict(model_specs[-1], name='extra', K2=2))
    if case == 'zero_extra':
        model_specs[0] = dict(model_specs[0], K2=1)
    names = [m['name'] for m in model_specs]
    state = {'oom': case == 'oom', 'draws': [], 'drained': False}
    priority = case in ['priority', 'priority_guard']
    def load(path, device):
        return SimpleNamespace(name=path, width=1, state_dict=lambda: {},
                               transformer=SimpleNamespace(decoder=SimpleNamespace(layers=[])))
    def decode(ctx, k=1, temperature=None, **kwargs):
        if temperature is not None:
            state['draws'].append((ctx.model.name, k, float(torch.rand(()))))
            if k == 2 and state['oom']:
                state['oom'] = False
                raise torch.cuda.OutOfMemoryError('Injected after random draw')
        idx = names.index(ctx.model.name)
        def token(row):
            if temperature is None or int(row[1]) == 3: return 1
            if k > 1: return 7
            if int(row[1]) == 4: return 3 if idx == n-1 else 1
            return 2+idx
        seq = torch.tensor([[[token(row)] for _ in range(k)] for row in ctx.src])
        return seq, torch.full((len(ctx.src), k), -1.), 0
    def pack(ids, lps, table, greedy=False):
        return [[{'code': f'def solve():\n    return {tokens[0]}',
                  'represented_code': f'<|raw|>def solve():\n    return {tokens[0]}',
                  'mean_logprob': 0. if greedy else -1.} for tokens in group] for group in ids.tolist()]
    class DelayedExecutor:
        def __init__(self, workers): pass
        def submit(self, candidates): pass
        def attach(self, candidates):
            for c in candidates:
                value = float(c['code'].split()[-1])
                c['execution'] = {'status': 'budget_not_evaluated'} if value == 4 and not state['drained'] else {'status': 'numeric', 'value': value}
        def drain(self, deadline): state['drained'] = True; return True
        def resize(self, workers): pass
        def summary(self): return {'mock': True}
        def close(self): pass
    with tempfile.TemporaryDirectory(prefix='v7-fixture-') as tmp:
        tmp = Path(tmp); rows = tmp/'rows.json'
        rows.write_text(json.dumps([{'question': q} for q in ['a', 'bb', 'ccc']]))
        cfg = dict(config, models=[dict(m, path=m['name']) for m in model_specs], rows=str(rows),
                   tokenizer='fixture', output=str(tmp/'out'), verify_weights=False, device='cpu',
                   cap=8, budget_seconds=20., greedy_sequence_budget=3, sequence_budget=4 if priority else 12)
        if case == 'guard1': cfg['cpu_test_expire_after_sample_batches'] = 1
        if case == 'guard2': cfg['cpu_test_expire_after_sample_batches'] = n+1
        if case == 'priority_guard': cfg['cpu_test_expire_after_sample_batches'] = 2*n+1
        patches = [patch.object(I, 'load', side_effect=load), patch.object(I, 'tokenizer', return_value=(None, {})),
                   patch.object(I.C, 'encode', side_effect=lambda tok, q: [2+len(q)]),
                   patch.object(I.Fast, 'Context', side_effect=lambda model, src, cap: SimpleNamespace(src=src, model=model)),
                   patch.object(I.Speed, 'decode', side_effect=decode), patch.object(I, 'pack', side_effect=pack)]
        if case == 'priority': patches.append(patch.object(I, 'Executor', DelayedExecutor))
        with ExitStack() as stack:
            for p in patches: stack.enter_context(p)
            timing = I.run(cfg)
        answers = [json.loads(line)['code'] for line in (tmp/'out/answers.jsonl').read_text().splitlines()]
        assert answers[0].endswith('return 1')
        assert timing['full_pipeline_seconds'] < 20.
        assert timing['winner_reexecution']['complete'] and not timing['winner_reexecution']['mismatch_rows']
        assert timing['model_order'] == names
        assert timing['deadline_guard']['triggered'] == (case in ['guard1', 'guard2', 'priority_guard'])
        if case == 'guard1':
            assert timing['deadline_guard']['executed_sample_counts'] == [1, 1, 1]
            assert timing['deadline_guard']['partial_vote_rows'] == [0, 1, 2]
            assert answers[2].endswith('return 1')  # count tie -> greedy
        elif case == 'guard2':
            assert timing['deadline_guard']['executed_sample_counts'] == [3, 5, 5]
            assert timing['deadline_guard']['partial_vote_rows'] == [1, 2]
        elif case == 'priority_guard':
            assert answers[1].endswith('return 1') and answers[2].endswith('return 7')
            assert timing['deadline_guard']['executed_sample_counts'] == [3, 5, 12]
            assert timing['deadline_guard']['partial_vote_rows'] == [1]
        else:
            assert all(a.endswith('return 7') for a in answers[1:])
            assert timing['adaptive']['stopped_rows'] == [0]
            assert timing['adaptive']['escalated_rows'] == [1, 2]
            assert timing['deadline_guard']['executed_sample_counts'] == [n]+[sum(m['K2'] for m in model_specs)]*2
            assert not timing['deadline_guard']['partial_vote_rows']
        if priority:
            assert timing['adaptive']['stage2_order'] == [2, 1]
            assert timing['adaptive']['stage1_leader_fractions'] == {1: .75, 2: .25}
            second = [b for b in timing['generation_batches'] if b['stage'] == 2]
            assert [(b['phase'], b['row_ids']) for b in second[:3]] == [(name, [2]) for name in names]
        if case == 'oom':
            assert len(timing['OOM_attempts']) == 1
            draws = [v for name, k, v in state['draws'] if name == names[0] and k == 2]
            assert draws[0] == draws[1]
        assert all(b['sequences'] <= cfg['sequence_budget'] for b in timing['generation_batches'] if b['stage'])
        return {'passed': True, 'case': case, 'N': n, 'seconds': timing['full_pipeline_seconds'],
                'real_executor': case != 'priority', 'real_winner_reexecution': True,
                'adaptive': timing['adaptive'], 'partial_vote_count': timing['deadline_guard']['partial_vote_count']}


def real_pipeline():
    import torch
    import final_inference_v7 as I
    tokenizer = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json'
    tok, _ = I.tokenizer(tokenizer, 30)
    model = I.C.SimpleTransformer(d_model=16, heads=2, encoder_layers=1, decoder_layers=1,
                                 swiglu_dim=32, vocab=tok.get_vocab_size(), max_positions=512, dropout=0.).eval()
    with torch.no_grad():
        model.head.weight.zero_(); model.head.bias.fill_(-1000.); model.head.bias[I.C.EOS] = 1000.
    with tempfile.TemporaryDirectory(prefix='v7-real-') as tmp:
        tmp = Path(tmp); rows = tmp/'rows.json'
        rows.write_text(json.dumps([{'question': 'Compute 2 + 3.'}, {'question': 'What is 7 times 6?'}]))
        cfg = dict(default_config(), rows=str(rows), tokenizer=str(tokenizer), output=str(tmp/'out'),
                   device='cpu', verify_weights=False, cap=16, sequence_budget=8, greedy_sequence_budget=2,
                   budget_seconds=30.)
        cfg['models'] = [dict(m, path=m['name']) for m in cfg['models']]
        with patch.object(I, 'load', return_value=model): timing = I.run(cfg)
        assert not timing['deadline_guard']['triggered'] and timing['adaptive']['escalated_rows'] == [0, 1]
        assert timing['deadline_guard']['executed_sample_counts'] == [12, 12]
        assert timing['winner_reexecution']['complete'] and not timing['winner_reexecution']['mismatch_rows']
        assert all(b['sequences'] <= 8 for b in timing['generation_batches'] if b['stage'])
        return {'passed': True, 'real_tokenizer_decoder_pack_executor': True, 'N': 3,
                'K1': [m['K1'] for m in cfg['models']], 'K2': [m['K2'] for m in cfg['models']], 'GPU': False}


def cold_process():
    """Execute embedded launcher -> supervisor -> actual worker and real load/SHA."""
    import torch
    import final_core as C
    from final_vote import validate_answers
    sources, _ = static()
    model = C.SimpleTransformer(d_model=16, heads=2, encoder_layers=1, decoder_layers=1,
                               swiglu_dim=32, vocab=8195, max_positions=512, dropout=0.).eval()
    with torch.no_grad():
        model.head.weight.zero_(); model.head.bias.fill_(-1000.); model.head.bias[C.EOS] = 1000.
    with tempfile.TemporaryDirectory(prefix='v7-cold-') as tmp:
        tmp = Path(tmp); module_dir = tmp/'src'; module_dir.mkdir()
        for name, source in sources.items(): (module_dir/name).write_text(source)
        rows = tmp/'rows.json'; rows.write_text(json.dumps([{'question': 'Compute 2 + 3.'}]))
        weight = tmp/'tiny.pt'
        torch.save({'format': 'nlp-hw02-model-only-v1', 'optimizer_included': False,
                    'model_spec': {'d_model': 16, 'heads': 2, 'encoder_layers': 1, 'decoder_layers': 1,
                                   'swiglu_dim': 32, 'vocab': 8195, 'max_positions': 512}, 'model': model.half().state_dict()}, weight)
        digest = hashlib.sha256(weight.read_bytes()).hexdigest()
        cfg = dict(default_config(), rows=str(rows), tokenizer=str(WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json'),
                   output=str(tmp/'out'), device='cpu', expected_rows=1, cap=16,
                   sequence_budget=8, greedy_sequence_budget=2, budget_seconds=35.)
        cfg['models'] = [dict(m, path=str(weight), filename=weight.name, sha256=digest) for m in cfg['models']]
        config_path = tmp/'config.json'; config_path.write_text(json.dumps(cfg))
        r = subprocess.run([sys.executable, str(module_dir/'final_runner.py'), str(config_path)],
                           env=dict(os.environ, FINAL_PROCESS_START=str(time.monotonic())),
                           capture_output=True, text=True, timeout=40.)
        assert r.returncode == 0, (tmp/'out/worker.log').read_text()+r.stderr
        report = read(tmp/'out/summary.json')
        assert report['status'] == 'complete' and report['rows'] == 1
        assert report['adaptive']['mean_sample_count'] == 12 and len(report['model_sha256']) == 3
        assert all(s == digest for s in report['model_sha256'].values())
        assert report['winner_reexecution']['complete'] and not report['winner_reexecution']['mismatch_rows']
        assert abs(sum(report['phases_seconds'].values())-report['full_pipeline_seconds']) < .1
        validate_answers(tmp/'out/answers.jsonl', 1)
        stale = subprocess.run([sys.executable, str(module_dir/'final_runner.py'), str(config_path)], capture_output=True, text=True, timeout=5.)
        assert stale.returncode != 0 and 'Fresh cold run' in stale.stderr
        return {'passed': True, 'embedded_real_supervisor_worker_load_SHA_N3': True,
                'full_timer_phases_and_stale_rejection': True, 'seconds': report['full_pipeline_seconds'], 'GPU': False}


def memory():
    """Real strict FP16 payload audit plus measured one/two-model T4 peaks.

    No simultaneous decoder caches: N adds resident parameters only. Estimate
    keeps the larger historical reserved peak plus additional model bytes and
    an extra512MiB allocator reserve. The cold run records the actual device peak.
    """
    import torch
    import final_inference_v7 as I
    report = {}; model_bytes = 0
    for m in default_config()['models']:
        folder = P1 if m['name'] == 'pilot-b' else FINAL
        path = folder/m['filename']
        assert I.sha(path) == m['sha256']
        payload = torch.load(path, map_location='cpu', weights_only=True)
        assert payload['dtype'] == 'FP16' and payload['optimizer_included'] is False
        assert all(t.dtype == torch.float16 for t in payload['model'].values())
        model = I.load(path, torch.device('cpu'))  # strict real architecture/state_dict
        spec = payload['model_spec']; del model
        resident = sum(t.numel()*t.element_size() for t in payload['model'].values())
        model_bytes += resident
        report[m['name']] = {'sha256': m['sha256'], 'spec': spec, 'FP16_resident_bytes': resident}
        del payload
    assert all(r['spec'] == next(iter(report.values()))['spec'] for r in report.values())
    profile = read(PROFILE)
    peak_one = max(r['peak_reserved_bytes'] for r in profile['rows'] if r['scope'] == 'test1919' and r['variant'] == 'all')
    peak_two = max(read(FINAL/f'inference-v5/final-inference-l3-v5/{s}/timing.json')['peak_reserved_bytes'] for s in ['test-1', 'tune'])
    one_bytes = next(iter(report.values()))['FP16_resident_bytes']
    projected = max(peak_one+model_bytes-one_bytes, peak_two+model_bytes-2*one_bytes)
    reserve = 512*1024**2
    # Conservative available T4 tensor/allocator budget; device-reported total checked live.
    t4_budget = 14*1024**3
    assert projected+reserve < t4_budget
    d = next(iter(report.values()))['spec']; lanes=1024; cap=512
    self_kv = 2*d['decoder_layers']*lanes*cap*d['d_model']*2
    return {'passed': True, 'model_only_SHA_strict_load_FP16_specs': report, 'resident_model_bytes': model_bytes,
            'self_KV_1024_lanes_cap512_bytes': self_kv, 'measured_all_one_model_reserved_bytes': peak_one,
            'measured_v5_two_model_reserved_bytes': peak_two, 'projected_reserved_bytes': projected,
            'allocator_extra_reserve_bytes': reserve, 'conservative_T4_budget_bytes': t4_budget,
            'headroom_after_reserve_bytes': t4_budget-projected-reserve,
            'GPU_measured_three_models': False, 'sequential_model_generation_one_cache_at_a_time': True,
            'limitations': 'measured T4 profile/v5 + resident parameters; N3 peak requires cold run; OOM halves batches'}


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
    identity = read(ROOT/'final-v7-source-identity.json')
    assert identity == read(BUNDLE/'source-identity.json')
    assert identity['notebook_sha256'] == hashlib.sha256((BUNDLE/NAME).read_bytes()).hexdigest()
    assert identity['ensemble_config'] == read(ROOT/'final-v7-config.json') == default_config()
    for name, source in sources.items():
        compile(source, name, 'exec')
        assert hashlib.sha256(source.encode()).hexdigest() == identity['embedded_source_sha256'][name]
        assert not any(s in source for s in ['/Users/', '~/.kaggle', 'KAGGLE_KEY', 'd1_'])
        if name in identity['base_v6_sha256'] and name not in ['final_inference.py', 'final_session.py', 'final_core.py', 'final_fast.py']:
            assert identity['base_v6_sha256'][name] == identity['embedded_source_sha256'][name]
    for name in ['qk_attention.py', 'final_core.py', 'final_fast.py', 'final_ensemble.py', 'final_speed.py', 'final_adaptive.py', 'final_execute_v6.py', 'final_executor_v6.py']:
        assert sources[name] == (ROOT/'src'/name).read_text()
    assert sources['final_inference.py'] == (ROOT/'src/final_inference_v7.py').read_text()
    assert sources['final_session.py'] == (ROOT/'src/final_session_v7.py').read_text()
    assert metadata['id'] == 'thisarivson/nlp-hw02-ensemble-inference-v7'
    assert re.sub('[^a-z0-9]+', '-', metadata['title'].lower()).strip('-') == metadata['id'].split('/')[1]
    assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_tpu'] and not metadata['enable_internet']
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    assert metadata['dataset_sources'] == ['thisarivson/nlp-hw02-l3-live', 'thisarivson/nlp-hw02-p1-weights', 'thisarivson/nlp-hw02-26b-inputs']
    session, worker = sources['final_session.py'], sources['final_inference.py']
    assert 'for number in range(1, 2):' in session and "launch('tune', rows_path, 2048" in session
    assert "'budget_seconds': 720" in session and 'budget_seconds=180.' in session
    assert 'elapsed >= 660.' in session and 'elapsed >= 690.' in session
    assert '< 480.' in session and '>= start+budget-min(8.' in sources['final_runner.py']
    assert 'min(20., budget*.12)' in worker and 'min(14., budget*.08)' in worker and 'min(10., budget*.06)' in worker
    assert 'executor.resize(tail_workers)' in worker and 'Speed.decode(ctx, k' in worker
    assert 'generation_workers = 2' in worker and 'min(4, os.cpu_count() or 1)' in worker
    assert 'test_with_40s_margin' in session and 'resident_models_and_T4_memory' in session
    limits = read(BUNDLE/'session-limits.json')
    assert limits['cold_test_runs'] == limits['tune_runs'] == 1 and limits['session_timeout_seconds'] == 720
    return sources, {'passed': True, 'notebook_identity_frozen_v6_compile': True,
                     'private_offline_single_T4_account_mounts_title': True, 'one_test_one_tune_watchdog_guard': True}


def suite(sources):
    with tempfile.TemporaryDirectory(prefix='v7-suite-') as tmp:
        tmp = Path(tmp); module = tmp/'suite_v7.py'; module.write_text(sources['final_session.py'])
        spec = importlib.util.spec_from_file_location('suite_v7', module)
        session = importlib.util.module_from_spec(spec); spec.loader.exec_module(session)
        calls = []; config = default_config()
        class Child:
            pid, returncode = 99999999, 0
            def __init__(self, args, **kwargs):
                from final_vote import write_answers
                cfg = read(args[-1]); calls.append(cfg); out = Path(cfg['output'])
                assert [m['name'] for m in cfg['models']] == [m['name'] for m in config['models']]
                assert [m['K2'] for m in cfg['models']] == [3, 5, 4]
                assert cfg['workers'] == 2 and cfg['verify_weights']
                write_answers(out/'answers.jsonl', ['def solve():\n    return 1']*cfg['expected_rows'])
                session.write(out/'summary.json', {'status': 'complete', 'full_pipeline_seconds': 127.,
                    'under_180_seconds': True, 'target_150_seconds': True, 'deadline_guard': {'fallback_count': 0},
                    'model_order': [m['name'] for m in cfg['models']], 'peak_reserved_bytes': 13*1024**3,
                    'peak_allocated_bytes': 11*1024**3, 'total_device_memory_bytes': 15*1024**3})
                if 'reference' in cfg: session.write(out/'quality.json', {'matches': config['expected_matches'], 'passed': True})
            def poll(self): return 0
            def wait(self, **kwargs): return 0
        output = tmp/'out'; output.mkdir()
        weights = {'thisarivson/nlp-hw02-l3-live': FINAL, 'thisarivson/nlp-hw02-p1-weights': P1}
        with patch.object(session.subprocess, 'Popen', Child):
            report = session.suite(WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs', weights,
                                   output, ROOT/'src', session.Watchdog(output, time.monotonic()), ensemble_config=config)
        assert report['status'] == 'passed' and report['GPU_runs'] == 2
        assert [Path(c['output']).name for c in calls] == ['test-1', 'tune']
        assert 'reference' not in calls[0] and 'reference' in calls[1]
        assert (output/'answers.jsonl').read_bytes() == (output/'test-1/answers.jsonl').read_bytes()
        return {'passed': True, 'mock_one_cold_test_then_tune_all_mounts': True, 'no_gold_in_test': True,
                'first_answer_published': True, 'N3_memory_gate': True}


def custom_builder():
    import build_final_inference_v7 as builder
    reports = []
    with tempfile.TemporaryDirectory(prefix='v7-custom-builder-') as tmp:
        tmp = Path(tmp); root = tmp/'kaggle'; (root/'notebooks').mkdir(parents=True)
        (root/'src').symlink_to(ROOT/'src', target_is_directory=True)
        (root/'bundles').mkdir()
        (root/'bundles/final_inference_l3_v6_mine').symlink_to(ROOT/'bundles/final_inference_l3_v6_mine', target_is_directory=True)
        for n in [2, 4]:
            config = default_config(); config['models'] = config['models'][:n]
            if n == 4: config['models'].append(dict(config['models'][-1], name='fourth', filename='fourth.pt', dataset='example/fourth-weights', K1=1, K2=2))
            with patch.object(builder, 'ROOT', root), patch.object(builder, 'BUNDLE', tmp/f'bundle{n}'), patch.object(builder, 'READY', tmp/'ready.json'):
                _, sources = builder.build(config)
            identity = read(tmp/f'bundle{n}/source-identity.json')
            assert identity['ensemble_config'] == config
            assert len(identity['ensemble_config']['models']) == n
            assert not read(tmp/'ready.json')['ready']
            for name, source in sources.items(): compile(source, name, 'exec')
            reports.append({'N': n, 'passed': True})
    return {'passed': True, 'ordered_model_list_and_custom_dataset': reports}


def offline():
    from final_adaptive import replay_record as v6_replay
    from final_executor_v6 import verify_winners
    journals, _ = load_all()
    report = read(WORK/'checks/final-inference-v7-offline.json'); selected = report['selected']
    assert report['passed'] and selected['max_samples'] <= 12 and selected['timing']['fits_140']
    assert selected['mean_samples'] <= report['baseline_v6']['mean_samples']
    model_specs = specs(selected['K1'], selected['K2'])
    winners, correct, stopped, counts = winners_for(journals, model_specs)
    assert hashlib.sha256(json.dumps(winners, ensure_ascii=False).encode()).hexdigest() == selected['winner_code_execution_sha256']
    assert sum(correct) == selected['matches'] and sum(stopped) == selected['stopped']
    pair = specs([2, 2, 1], [6, 6, 1])[:2]
    for i in fixed_split()['tune2048']:
        a, b = journals['C-main'][i], journals['C-long2x'][i]
        new = replay_record(a['greedy'], {'C-main': a['samples']['0.7'], 'C-long2x': b['samples']['0.7']}, pair)
        old = v6_replay(a['greedy'], a['samples']['0.7'], b['samples']['0.7'], k1=2, k2=6)
        assert new == old
    reexecution = verify_winners([{'code': w['code'], 'execution': w['execution']} for w in winners], time.monotonic()+60., 4)
    assert reexecution['complete'] and not reexecution['mismatch_rows']
    return {'passed': True, 'N2_v6_replay_code_execution_count_stop_exact_2048': True,
            'N3_independent_union_find_vote_priority_code_execution_exact_2048': True,
            'matches': sum(correct), 'all_winners_reexecution': reexecution}


def child(mode, case=None):
    r = subprocess.run([sys.executable, __file__, '--child', mode]+(['--case', case] if case else []),
                       capture_output=True, text=True, timeout=180.)
    assert r.returncode == 0, f'{mode}/{case}: {r.stderr}\n{r.stdout}'
    return json.loads(r.stdout)


def main():
    READY.write_text(json.dumps({'bundle': str(BUNDLE.resolve()), 'ready': False})+'\n')
    assert read(WORK/'checks/final-inference-v7-regressions.json')['passed']
    report = {'GPU': False, 'push': False, 'v3_to_v6_regressions': True, 'unit': unit(), 'fixtures': []}
    for case in ['complete', 'complete2', 'complete4', 'guard1', 'guard2', 'oom', 'priority', 'priority_guard', 'zero_extra']:
        report['fixtures'].append(child('fixture', case))
        print(case, 'passed', flush=True)
    for mode in ['real_pipeline', 'cold_process', 'memory', 'offline']:
        report[mode] = child(mode); print(mode, 'passed', flush=True)
    sources, report['static'] = static()
    report['suite'] = suite(sources)
    report['custom_builder'] = custom_builder()
    dataset = read(WORK/'checks/final-inference-v7-dataset.json')
    assert dataset['prepared'] and dataset['id'] == 'thisarivson/nlp-hw02-p1-weights' and dataset['private']
    report['dataset'] = dataset
    report.update(passed=True, config=default_config(), D1_files_edited=False)
    receipt = WORK/'checks/final-inference-v7-cpu.json'
    receipt.write_text(json.dumps(report, indent=2)+'\n')
    READY.write_text(json.dumps({'bundle': str(BUNDLE.resolve()), 'ready': True,
        'checks': str(receipt.resolve()), 'kernel': 'thisarivson/nlp-hw02-ensemble-inference-v7',
        'ensemble_config': default_config(), 'GPU_gate': 'pending one cold test1919 + tune2048, N3 actual peak/runtime/quality',
        'dataset_created': dataset.get('created', False), 'push': False,
        'notebook_sha256': read(ROOT/'final-v7-source-identity.json')['notebook_sha256'],
        'readiness': 'local CPU/offline/v3–v6/static and projected T4 memory gates passed; no push'}, indent=2)+'\n')
    print('V7 gates passed; ready:true; no kernel push', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--child', choices=['fixture', 'real_pipeline', 'cold_process', 'memory', 'offline'])
    parser.add_argument('--case'); args = parser.parse_args()
    if args.child: print(json.dumps(fixture(args.case) if args.child == 'fixture' else globals()[args.child]()))
    else: main()
