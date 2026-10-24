"""Validate frozen v9 output and publish code-only JSONL; compare executed v8 answers."""
import argparse, ast, gzip, hashlib, json, math, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from build_final_inference_v9 import default_config
from final_vote import validate_answers
from final_ensemble import select_available

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(a,b):
    return a['status']==b['status'] and (a['status']!='numeric'or math.isclose(a['value'],b['value'],rel_tol=1e-6,abs_tol=1e-6))

def execute(rows):
    from final_executor_v6 import Executor
    executor=Executor(4)
    candidates=[dict(code=r['code'])for r in rows]
    try:
        executor.submit(candidates);assert executor.drain(time.monotonic()+180),'Execution incomplete'
        executor.attach(candidates);return [r['execution']for r in candidates],executor.summary()
    finally:executor.close()

def check(root):
    root=Path(root);summary=read(root/'summary.json');config=default_config()
    reports={n:read(root/n/'summary.json')for n in ['test-1','tune']}
    quality=read(root/'tune/quality.json');answers={n:validate_answers(root/n/'answers.jsonl',count)for n,count in [('test-1',1919),('tune',2048)]}
    assert (root/'answers.jsonl').read_bytes()==(root/'test-1/answers.jsonl').read_bytes()
    expected_hashes={m['name']:m['sha256']for m in config['models']}
    for name,n in [('test-1',1919),('tune',2048)]:
        report=reports[name];assert report['status']=='complete' and report['rows']==n
        assert report['model_order']==list(expected_hashes) and report['model_sha256']==expected_hashes
        assert report['single_gpu'] and report['deadline_guard']['generation_stop_seconds']==160.
        assert not report['deadline_guard']['triggered']
        assert report['deadline_guard']['fallback_count']==report['deadline_guard']['partial_vote_count']==0
        assert report['winner_reexecution']['complete'] and not report['winner_reexecution']['mismatch_rows']
        assert all(b['sequences']<=config['sequence_budget']for b in report['generation_batches'])
        assert report['peak_reserved_bytes']<report['total_device_memory_bytes']
        assert report['peak_allocated_bytes']+256*1024**2<report['total_device_memory_bytes']
        assert report['adaptive']['K1']=={m['name']:m['K1']for m in config['models']}
        assert report['adaptive']['K2']=={m['name']:m['K2']for m in config['models']}
        assert {b['phase']for b in report['generation_batches']if b['stage']==2}=={'pilot-b','W-B5'}
        cfg=read(root/name/'config.json');assert cfg['expected_rows']==n and ('reference'in cfg)==(name=='tune')
        assert report['execution']['service_state']['nice']>=10
    inputs=WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    test_source=read(inputs/'test.json');test=reports['test-1'];tune=reports['tune']
    assert test['input_sha256']==sha(inputs/'test.json')
    assert test['question_order_sha256']==hashlib.sha256(json.dumps([r['question']for r in test_source],ensure_ascii=False).encode()).hexdigest()
    with gzip.open(root/'tune/candidate-journal.json.gz','rt')as f:journal=json.load(f)
    references=read(root/'tune2048-reference.json');assert len(journal)==len(references)==2048
    matches=0
    for i,(row,ref)in enumerate(zip(journal,references)):
        winner,_=select_available(row,config['models']);assert winner['code']==answers['tune'][i]['code']
        matches+=ref['status']=='numeric'and same(ref,winner['execution'])
    assert matches==quality['matches'] and quality['n']==2048
    previous=validate_answers(WORK/'submissions/final-ensemble-v8.jsonl',1919)
    changed=[i for i,(a,b)in enumerate(zip(answers['test-1'],previous))if a['code']!=b['code']]
    executions,execution_receipt=execute(answers['test-1']+previous)
    current,old=executions[:1919],executions[1919:]
    different=[i for i,(a,b)in enumerate(zip(current,old))if not same(a,b)]
    nonnumeric=[i for i,(a,b)in enumerate(zip(current,old))if a['status']!='numeric'or b['status']!='numeric']
    assert not nonnumeric,('Nonnumeric comparisons require review',nonnumeric)
    destination=WORK/'submissions/final-ensemble-v9.jsonl'
    destination.write_bytes((root/'answers.jsonl').read_bytes());validate_answers(destination,1919)
    identity=read(ROOT/'final-v9-source-identity.json');modules=root/'src'
    assert {n:sha(modules/n)for n in identity['embedded_source_sha256']}==identity['embedded_source_sha256']
    result=dict(passed=True,test1919_seconds=test['full_pipeline_seconds'],test_target_140_passed=test['full_pipeline_seconds']<=140,
        tune_seconds=tune['full_pipeline_seconds'],tune_matches=matches,tune_n=2048,tune_accuracy_pp=100*matches/2048,
        offline_matches=1547,v8_offline_matches=1541,delta_fresh_vs_offline_matches=matches-1547,
        changed_code_vs_v8=len(changed),changed_code_vs_v8_rows=changed,changed_executed_answer_vs_v8=len(different),changed_executed_answer_vs_v8_rows=different,
        execution_comparison=dict(method='independent fresh bounded production executor for both files, numeric isclose rel/abs 1e-6',nonnumeric_rows=nonnumeric,receipt=execution_receipt),
        submission=str(destination.relative_to(WORK)),answers_sha256=sha(destination),rows=1919,code_only=True,test_order_verified=True,
        peak_memory_bytes={n:{k:r[k]for k in ['peak_reserved_bytes','peak_allocated_bytes','total_device_memory_bytes']}for n,r in reports.items()},
        OOM_attempts={n:r['OOM_attempts']for n,r in reports.items()},session_seconds=summary['session_seconds'],summary_status=summary['status'],gates=summary['gates'],all_20_remote_sources_exact=True)
    (WORK/'checks/final-inference-v9-output.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (WORK/'results/final-inference-v9.json').write_text(json.dumps({k:v for k,v in result.items()if not k.endswith('_rows')},ensure_ascii=False,indent=2)+'\n')
    return result

def check_interactive(root, answer_path):
    """Validate an externally collected execution of the byte-identical notebook.

    This mode does not pretend that missing remote files were downloaded. A
    runtime candidate replay is separately marked and runs if its raw files exist.
    """
    import nbformat
    from final_split import fixed_split
    root,answer_path=Path(root).resolve(),Path(answer_path).resolve()
    report=read(root/'summary.json');config=default_config()
    assert report['status']=='passed' and all(report['gates'].values())
    assert report['GPU_runs']==2 and report['training']is False
    assert len(report['test'])==1 and report['budget_seconds']==720
    assert report['internet']is False and report['single_gpu']is True
    test,tune=report['test'][0],report['tune'];quality=report['quality']
    assert test==read(root/'test-1-summary.json')
    assert all(test[k]==v for k,v in read(root/'test-1-timing.json').items())
    nb=read(root/'nlp-hw02-ensemble-inference-v9.ipynb')
    nbformat.validate(nbformat.from_dict(nb))
    local=read(ROOT/'notebooks/nlp-hw02-ensemble-inference-v9.ipynb')
    def codes(nb):return [''.join(c['source'])if isinstance(c['source'],list)else c['source']for c in nb['cells']if c['cell_type']=='code']
    assert codes(nb)==codes(local),'Executed notebook code differs from frozen v9'
    assert [c['execution_count']for c in nb['cells']if c['cell_type']=='code']==[1,2]
    identity=read(ROOT/'final-v9-source-identity.json')
    sources=ast.literal_eval(next(n.value for n in ast.parse(codes(nb)[0]).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SOURCES'))
    assert {n:hashlib.sha256(v.encode()).hexdigest()for n,v in sources.items()}==identity['embedded_source_sha256']
    for name in ['final_ensemble.py','final_adaptive.py','final_vote.py']:
        assert sha(ROOT/'src'/name)==identity['embedded_source_sha256'][name]
    hashes={m['name']:m['sha256']for m in config['models']}
    for name,r,n in [('test',test,1919),('tune',tune,2048)]:
        assert r['status']=='complete' and r['rows']==n and r['under_180_seconds']
        assert r['single_gpu'] and r['model_order']==list(hashes) and r['model_sha256']==hashes
        assert r['seed']==270042 and r['deterministic_algorithms']
        assert not r['deadline_guard']['triggered']
        assert r['deadline_guard']['fallback_count']==r['deadline_guard']['partial_vote_count']==0
        assert r['deadline_guard']['generation_stop_seconds']==160.
        assert r['winner_reexecution']['complete'] and not r['winner_reexecution']['mismatch_rows']
        assert r['winner_reexecution']['checked_rows']==n
        assert r['adaptive']['K1']=={m['name']:m['K1']for m in config['models']}
        assert r['adaptive']['K2']=={m['name']:m['K2']for m in config['models']}
        assert not r['adaptive']['scorer_enabled'] and not r['adaptive']['undecided_rows']
        stopped=set(r['adaptive']['stopped_rows'])
        assert all(c==sum(m['K1'if i in stopped else'K2']for m in config['models'])for i,c in enumerate(r['deadline_guard']['executed_sample_counts']))
        assert r['execution']['requested_candidates']==n+sum(r['deadline_guard']['executed_sample_counts'])
        assert r['execution']['completed_unique']==r['execution']['unique_programs'] and r['execution']['error']is None
        assert r['execution']['service_state']['nice']>=10
        assert all(b['sequences']<=config['sequence_budget']for b in r['generation_batches'])
        assert {b['phase']for b in r['generation_batches']if b['stage']==2}=={'pilot-b','W-B5'}
        assert r['peak_reserved_bytes']<r['total_device_memory_bytes']
        assert r['peak_allocated_bytes']+256*1024**2<r['total_device_memory_bytes']
        assert abs(sum(r['phases_seconds'].values())-r['full_pipeline_seconds'])<.1
    assert test['full_pipeline_seconds']<=140.
    inputs=WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    original_test=read(inputs/'test.json');heldout=read(inputs/'heldout4k.json');ids=fixed_split()['tune2048']
    tune_rows=[dict(question=heldout[i]['question'])for i in ids]
    assert test['input_sha256']==sha(inputs/'test.json')
    assert tune['input_sha256']==hashlib.sha256((json.dumps(tune_rows,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest()
    for r,rows in [(test,original_test),(tune,tune_rows)]:
        assert r['question_order_sha256']==hashlib.sha256(json.dumps([v['question']for v in rows],ensure_ascii=False).encode()).hexdigest()
    answers=validate_answers(answer_path,1919)
    assert test['answers_sha256']==sha(answer_path)==report['answers_sha256'][0]
    assert quality['n']==2048 and quality['expected_matches']==config['expected_matches']
    assert quality['matches']==len(set(quality['match_indices']))==len(quality['match_indices'])
    assert all(0<=i<2048 for i in quality['match_indices'])
    assert quality['accuracy_pp']==100*quality['matches']/2048
    runtime_replay=False
    if (root/'tune/candidate-journal.json.gz').exists() and (root/'tune/answers.jsonl').exists():
        with gzip.open(root/'tune/candidate-journal.json.gz','rt')as f:journal=json.load(f)
        tune_answers=validate_answers(root/'tune/answers.jsonl',2048)
        refs=read(inputs/'heldout4k-reference-execution.json');refs=refs['rows']if isinstance(refs,dict)else refs
        assert len(journal)==2048 and sha(root/'tune/answers.jsonl')==tune['answers_sha256']
        matched=[]
        for i,row in enumerate(journal):
            winner,_=select_available(row,config['models']);assert winner['code']==tune_answers[i]['code']
            ref=refs[ids[i]]
            if ref['status']=='numeric'and same(ref,winner['execution']):matched.append(i)
        assert matched==quality['match_indices'];runtime_replay=True
    previous_path=WORK/'submissions/final-ensemble-v8.jsonl'
    previous=validate_answers(previous_path,1919)
    comp_path=WORK/'checks/final-inference-v9-comparison.json'
    comparison=read(comp_path)if comp_path.exists()else{}
    if comparison.get('v9_sha256')!=sha(answer_path)or comparison.get('v8_sha256')!=sha(previous_path):
        values,execution=execute(answers+previous);current,old=values[:1919],values[1919:]
        changed_code=[i for i,(a,b)in enumerate(zip(answers,previous))if a['code']!=b['code']]
        changed_answer=[i for i,(a,b)in enumerate(zip(current,old))if not same(a,b)]
        nonnumeric=[i for i,(a,b)in enumerate(zip(current,old))if a['status']!='numeric'or b['status']!='numeric']
        comparison=dict(rows=1919,v9_sha256=sha(answer_path),v8_sha256=sha(previous_path),
            changed_code_vs_v8=len(changed_code),changed_code_vs_v8_rows=changed_code,
            changed_executed_answer_vs_v8=len(changed_answer),changed_executed_answer_vs_v8_rows=changed_answer,
            nonnumeric_rows=nonnumeric,execution=execution,method='fresh production bounded executor; numerical isclose rel_tol=abs_tol=1e-6')
        comp_path.write_text(json.dumps(comparison,indent=2)+'\n')
    assert not comparison['nonnumeric_rows']
    assert comparison['execution']['completed_unique']==comparison['execution']['unique_programs']
    assert comparison['changed_code_vs_v8']==len(comparison['changed_code_vs_v8_rows'])
    assert comparison['changed_executed_answer_vs_v8']==len(comparison['changed_executed_answer_vs_v8_rows'])
    destination=WORK/'submissions/final-ensemble-v9.jsonl'
    if destination.exists():assert destination.read_bytes()==answer_path.read_bytes(),'Keep a different existing submission intact'
    else:
        with destination.open('xb')as f:f.write(answer_path.read_bytes())
    validate_answers(destination,1919)
    result=dict(passed=True,status='passed',origin='externally collected interactive execution of byte-identical private v9 notebook',
        own_push_calls=1,test1919_seconds=test['full_pipeline_seconds'],test_target_140_passed=True,
        tune_seconds=tune['full_pipeline_seconds'],tune_matches=quality['matches'],tune_n=2048,tune_accuracy_pp=quality['accuracy_pp'],
        offline_matches=1547,v8_offline_matches=1541,delta_fresh_vs_offline_matches=quality['matches']-1547,
        changed_code_vs_v8=comparison['changed_code_vs_v8'],changed_executed_answer_vs_v8=comparison['changed_executed_answer_vs_v8'],
        submission=str(destination.relative_to(WORK)),answers_sha256=sha(destination),rows=1919,code_only=True,test_order_verified=True,tune_order_verified=True,
        all_20_executed_embedded_sources_exact=True,notebook_code_cells_exact=True,notebook_execution_counts=[1,2],
        quality=quality,verified_model_sha256=hashes,requested_account='friend1',interactive_session_metadata_available=False,
        winner_reexecution_checked_rows={'test':1919,'tune':2048},winner_reexecution_mismatches=0,
        runtime_replay_verified=runtime_replay,
        runtime_replay_limitation=None if runtime_replay else'Raw tune answers/candidate-journal were not supplied; quality is from executed notebook report, match indices/count/accuracy validated.',
        peak_memory_bytes={name:{k:r[k]for k in ['peak_reserved_bytes','peak_allocated_bytes','total_device_memory_bytes']}for name,r in [('test',test),('tune',tune)]},
        OOM_attempts={name:r['OOM_attempts']for name,r in [('test',test),('tune',tune)]},
        early_stops={name:len(r['adaptive']['stopped_rows'])for name,r in [('test',test),('tune',tune)]},
        mean_samples={name:r['adaptive']['mean_sample_count']for name,r in [('test',test),('tune',tune)]},
        session_seconds=report['session_seconds'],gates=report['gates'],raw_summary_sha256=sha(root/'summary.json'),
        executed_notebook_sha256=sha(root/'nlp-hw02-ensemble-inference-v9.ipynb'),raw_artifacts=str(root.relative_to(WORK)))
    (WORK/'checks/final-inference-v9-output.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (WORK/'results/final-inference-v9.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--interactive',action='store_true')
    p.add_argument('--answers',type=Path,default=WORK/'submissions/final-ensemble-v9-interactive.jsonl')
    a=p.parse_args();r=check_interactive(a.output,a.answers)if a.interactive else check(a.output)
    print(json.dumps({k:v for k,v in r.items()if not k.endswith('_rows')},ensure_ascii=False,indent=2))
