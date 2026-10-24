"""CPU gates for frozen v8: production replay, real W load/greedy, v7 and mounts."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from build_final_inference_v8 import BUNDLE, NAME, READY, default_config
from final_ensemble import replay_record
from final_split import fixed_split
RAW = WORK/'results/kaggle-runs/nlp-hw02-b4/1'
POINT = RAW/'model-GPU0/points/b4-final-091801'
INPUTS = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'


def read(path):
    return json.loads(Path(path).read_text())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def offline():
    from final_v7_analysis import load_all
    journals, receipts = load_all()
    saved = read(WORK/'results/b4-results.json')
    paths = sorted((POINT/'evaluation/k8-tune2048').glob('batch-*.json'))
    assert len(paths) == 64
    records = []
    hashes = {}
    for p in paths:
        h = digest(p.read_bytes())
        assert saved['inputs_sha256'][str(p.relative_to(WORK))] == h
        hashes[str(p.relative_to(WORK))] = h
        records.extend(read(p))
    journals['W600'] = {r['id']: r for r in records}
    ids = fixed_split()['tune2048']
    assert len(records) == 2048 and sorted(journals['W600']) == ids
    module_spec = importlib.util.spec_from_file_location('b4_replay', WORK/'research/analyze-b4-results.py')
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    config = default_config(); models = config['models']; names = [m['name'] for m in models]
    ks = [m['K2'] for m in models]; k1 = [m['K1'] for m in models]
    replay = module.B.Replay(journals, names[0], names)
    bits, stops, expected = replay.evaluate(ks, k1)
    winners = []; production_matches = production_stops = 0
    for i, want, bit in zip(ids, expected, bits):
        anchor = journals[names[0]][i]
        for name in names:
            row = journals[name][i]
            assert row['question_sha256'] == anchor['question_sha256']
            assert row['reference'] == anchor['reference'] and len(row['samples']['0.7']) == 8
        win, stop, count = replay_record(anchor['greedy'],
            {n: journals[n][i]['samples']['0.7'] for n in names}, models)
        assert (win['code'], win['execution']) == (want['code'], want['execution']), i
        assert module.B.correct(anchor['reference'], win['execution']) == bit
        production_matches += bit; production_stops += stop
        winners.append({'id': i, 'code': win['code'], 'execution': win['execution']})
    assert production_matches == sum(bits) == config['expected_matches'] == 1541
    assert production_stops == stops == 1228
    assert names == saved['ensemble']['best']['same_weighted_mean_budget']['models']
    return dict(passed=True, rows=2048, matches=production_matches, accuracy_pp=100*production_matches/2048,
        early_stops=stops, winner_code_execution_mismatches=0,
        winner_sha256=digest(json.dumps(winners, ensure_ascii=False, sort_keys=True).encode()),
        journal_sha256={**receipts, 'W600': hashes}, comparison='analyze-b4-results.py -> B.Replay.evaluate')


def weights():
    import torch
    import final_inference_v7 as I
    from final_session_v8 import weight_path
    from final_v7_analysis import FINAL, P1
    torch.set_num_threads(1)
    roots = {'thisarivson/nlp-hw02-l3-live': FINAL,
             'thisarivson/nlp-hw02-p1-weights': P1, 'thisarivson/nlp-hw02-b4': RAW}
    report = {}
    for m in default_config()['models']:
        path = Path(weight_path(roots[m['dataset']], m)); assert I.sha(path) == m['sha256']
        payload = torch.load(path, map_location='cpu', weights_only=True)
        assert payload['format'] == 'nlp-hw02-model-only-v1' and payload['optimizer_included'] is False
        assert all(t.dtype == torch.float16 and torch.isfinite(t).all() for t in payload['model'].values())
        model = I.load(path, torch.device('cpu'))
        state = model.state_dict()
        assert state.keys() == payload['model'].keys()
        assert all(state[k].shape == v.shape and torch.equal(state[k], v.float()) for k, v in payload['model'].items())
        spec = payload['model_spec']
        report[m['name']] = dict(spec=spec, keys=len(state), sha256=m['sha256'],
            parameter_count=sum(p.numel() for p in model.parameters()),
            FP16_resident_bytes=sum(t.numel()*t.element_size() for t in payload['model'].values()),
            QK_norm_keys=sum('q_norm' in k or 'k_norm' in k for k in state))
        if m['name'] == 'W600':
            assert spec == dict(d_model=768, heads=12, encoder_layers=6, decoder_layers=6,
                                swiglu_dim=2048, vocab=8195, qk_norm='layernorm')
            assert report[m['name']]['QK_norm_keys'] == 72
        del model, payload, state
    # Only one model owns decoder caches at a time. W K1 can use 1024 lanes;
    # live OOM retry halves that model's batch and restores RNG (frozen v7).
    self_kv = 2*6*1024*512*768*2
    return dict(passed=True, models=report, mixed_resident_bytes=sum(r['FP16_resident_bytes'] for r in report.values()),
                W_self_KV_1024_cap512_bytes=self_kv, actual_GPU_peak='measured by cold run')


def greedy():
    import torch
    import final_core as C
    import final_fast as Fast
    import final_speed as Speed
    from final_inference_v7 import load, DECODE_OPTIONS
    from final_representation import tokenizer
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    m = default_config()['models'][-1]
    model = load(RAW/m['relative_path'], torch.device('cpu'))
    tok, table = tokenizer(INPUTS/'tokenizer.json', 30)
    heldout = read(INPUTS/'heldout4k.json')
    saved = read(POINT/'evaluation/heldout4k-generations.json')
    selected = [0, 1, 100, 1000]
    src = torch.nn.utils.rnn.pad_sequence([torch.tensor([C.BOS]+C.encode(tok, heldout[i]['question'])+[C.EOS])
        for i in selected], batch_first=True, padding_value=C.PAD)
    begin = time.monotonic()
    ids, _, _ = Speed.decode(Fast.Context(model, src, 512), compute_logprob=False, options=DECODE_OPTIONS)
    baseline, _, _ = Fast.decode(Fast.Context(model, src, 512), compute_logprob=False)
    actual = [C.active_tokens(row[0]) for row in ids.tolist()]
    assert actual == [C.active_tokens(row[0]) for row in baseline.tolist()]
    rows = [dict(id=i, CPU_baseline_all_exact=True, B4_GPU_FP16_exact=tokens==saved[i]['generated_token_ids'],
                 CPU_tokens=len(tokens), saved_tokens=len(saved[i]['generated_token_ids'])) for i,tokens in zip(selected,actual)]
    # Real W QK math: cached self/cross Q,K must match full prefix logits.
    with torch.inference_mode():
        d = Fast.Decoder(Fast.Context(model, src, 8), 1)
        prefix = torch.full((len(src),1), C.BOS, dtype=torch.long)
        max_diff = 0.
        for _ in range(4):
            cached = d.step(prefix[:,-1]); full = model(src, prefix)[:,-1]
            max_diff = max(max_diff, (cached-full).abs().max().item())
            assert torch.allclose(cached, full, atol=2e-4, rtol=2e-4)
            prefix = torch.cat([prefix,cached.argmax(-1)[:,None]],dim=1)
    return dict(passed=True, rows=rows, CPU_FP32_baseline_all_exact=True,
        saved_B4_FP16_matches=sum(r['B4_GPU_FP16_exact'] for r in rows),
        cached_full_prefix_max_abs_diff=max_diff, seconds=time.monotonic()-begin,
        limitation='B4 journals use CUDA FP16 batch256, CPU uses FP32 batch4. FP16 rounding and compaction change shapes; token identity is measured, not guaranteed.')


def static():
    import nbformat
    from final_mounts import find_input, resolve_cache_mount
    from final_session_v8 import weight_path
    notebook = read(BUNDLE/NAME); nbformat.validate(nbformat.from_dict(notebook))
    assert (ROOT/'notebooks'/NAME).read_bytes() == (BUNDLE/NAME).read_bytes()
    codes = [c['source'] for c in notebook['cells'] if c['cell_type']=='code']
    sources = ast.literal_eval(next(n.value for n in ast.parse(codes[0]).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SOURCES'))
    identity = read(BUNDLE/'source-identity.json')
    for code in codes: compile(code,'v8-cell','exec')
    for name, source in sources.items():
        compile(source,name,'exec'); assert digest(source.encode()) == identity['embedded_source_sha256'][name]
        assert not any(s in source for s in ['/Users/', '~/.kaggle', 'KAGGLE_KEY'])
    assert sources['final_inference.py'] == (ROOT/'src/final_inference_v7.py').read_text()
    # Freeze every v7 module except the mount-aware session; all optimizations/guards/vote remain identical.
    old_nb = read(ROOT/'notebooks/nlp-hw02-ensemble-inference-v7.ipynb')
    old_code = next(c['source'] for c in old_nb['cells'] if c['cell_type']=='code')
    old = ast.literal_eval(next(n.value for n in ast.parse(old_code).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SOURCES'))
    assert all(sources[n] == source for n,source in old.items() if n!='final_session.py')
    meta = read(BUNDLE/'kernel-metadata.json')
    assert meta['id']=='thisarivson/nlp-hw02-ensemble-inference-v8' and meta['is_private']
    assert meta['enable_gpu'] and not meta['enable_internet'] and meta['machine_shape']=='NvidiaTeslaT4'
    assert meta['kernel_sources']==['thisarivson/nlp-hw02-b4']
    assert meta['dataset_sources']==['thisarivson/nlp-hw02-l3-live','thisarivson/nlp-hw02-p1-weights','thisarivson/nlp-hw02-26b-inputs']
    assert identity['notebook_sha256']==digest((BUNDLE/NAME).read_bytes())
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        for kind in ['datasets','notebooks']:
            p=root/kind/'thisarivson'/'model-output';p.mkdir(parents=True)
            assert resolve_cache_mount(root,'thisarivson/model-output',kind)==p
        flat=root/'flat';flat.mkdir(); assert find_input('flat',root)==flat
        rel=Path(default_config()['models'][-1]['relative_path']);p=root/rel;p.parent.mkdir(parents=True);p.touch()
        assert weight_path(root,default_config()['models'][-1])==str(p)
    return dict(passed=True, nbformat=True, v7_worker_decoder_selector_executor_supervisor_unchanged=True,
                QK_norm_already_supported=True, private=True, single_T4=True, kernel_output_mounts=True,
                guard_generation_execution_verification_supervisor=[160,166,170,172], notebook_sha256=identity['notebook_sha256'])


def pipeline():
    import torch
    from final_core import SimpleTransformer, EOS
    from final_vote import validate_answers
    static()
    notebook=read(BUNDLE/NAME)
    code=next(c['source'] for c in notebook['cells'] if c['cell_type']=='code')
    sources=ast.literal_eval(next(n.value for n in ast.parse(code).body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SOURCES'))
    with tempfile.TemporaryDirectory(prefix='v8-mixed-cold-') as tmp:
        tmp=Path(tmp);modules=tmp/'src';modules.mkdir()
        for name,source in sources.items(): (modules/name).write_text(source)
        models=[]
        for i,m in enumerate(default_config()['models']):
            spec=dict(d_model=24 if i==2 else 16,heads=2,encoder_layers=1,decoder_layers=1,
                      swiglu_dim=32,vocab=8195,max_positions=512,qk_norm='layernorm' if i==2 else False)
            model=SimpleTransformer(**spec,dropout=0.).eval()
            with torch.no_grad():
                model.head.weight.zero_();model.head.bias.fill_(-1000.);model.head.bias[EOS]=1000.
            weight=tmp/(m['name']+'.pt')
            torch.save(dict(format='nlp-hw02-model-only-v1',optimizer_included=False,
                            model_spec=spec,model=model.half().state_dict()),weight)
            models.append(dict(m,path=str(weight),sha256=digest(weight.read_bytes())))
        rows=tmp/'rows.json';rows.write_text(json.dumps([dict(question='Compute 2 + 3.'),dict(question='What is 7 minus 2?')]))
        cfg=dict(default_config(),models=models,rows=str(rows),tokenizer=str(INPUTS/'tokenizer.json'),
                 output=str(tmp/'out'),device='cpu',expected_rows=2,cap=16,
                 sequence_budget=14,greedy_sequence_budget=2,budget_seconds=40.)
        config=tmp/'config.json';config.write_text(json.dumps(cfg))
        r=subprocess.run([sys.executable,str(modules/'final_runner.py'),str(config)],capture_output=True,text=True,
                         timeout=45,env=dict(os.environ,FINAL_PROCESS_START=str(time.monotonic())))
        assert r.returncode==0,(tmp/'out/worker.log').read_text()+r.stderr
        report=read(tmp/'out/summary.json');validate_answers(tmp/'out/answers.jsonl',2)
        assert report['status']=='complete' and report['adaptive']['mean_sample_count']==10
        assert report['model_specs']['W600']['width']==24 and report['model_specs']['C-long2x']['width']==16
        assert report['deadline_guard']['executed_sample_counts']==[10,10]
        assert all(b['phase']=='pilot-b' for b in report['generation_batches'] if b['stage']==2)
        assert report['winner_reexecution']['complete'] and not report['winner_reexecution']['mismatch_rows']
        assert abs(sum(report['phases_seconds'].values())-report['full_pipeline_seconds'])<.1
        return dict(passed=True,embedded_cold_mixed_width_QK_SHA_load=True,stage2_only_P1_B=True,
                    K2=[1,8,1],seconds=report['full_pipeline_seconds'])


def regressions():
    modes=[('fixture',c) for c in ['complete','complete2','complete4','guard1','guard2','oom','priority','priority_guard','zero_extra']]
    modes += [(mode,None)for mode in ['real_pipeline','cold_process','offline']]
    results=[]
    for mode,case in modes:
        args=[sys.executable,str(ROOT/'final_v7_checks.py'),'--child',mode]+(['--case',case]if case else [])
        r=subprocess.run(args,capture_output=True,text=True,timeout=200)
        assert r.returncode==0, f'{mode}/{case}: {r.stderr}\n{r.stdout}'
        results.append(dict(mode=mode,case=case,result=json.loads(r.stdout)))
        print('v7 regression',mode,case,'passed',flush=True)
    return dict(passed=True,checks=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['offline','weights','greedy','static','pipeline','regressions']);args=parser.parse_args()
    result=globals()[args.mode]()
    target=WORK/f'checks/final-inference-v8-{args.mode}.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['journal_sha256','checks']},ensure_ascii=False,indent=2))
