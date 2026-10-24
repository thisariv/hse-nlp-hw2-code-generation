"""Validate frozen v10 output and publish code-only JSONL; compare executed v8 answers."""
import argparse, ast, gzip, hashlib, json, math, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from build_final_inference_v10 import default_config
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
    download=read(WORK/'checks/final-inference-v10-output-download.json');remote=read(WORK/'checks/final-inference-v10-remote.json')
    assert download['files']>0 and download['ref']==remote['ref']=='friend1/nlp-hw02-ensemble-inference-v10'
    assert download['version']==remote['version'] and remote['all_code_cells_exact'] and remote['private']
    assert root.resolve().is_relative_to((WORK/download['output']).resolve())
    reports={n:read(root/n/'summary.json')for n in ['test-1','tune']}
    quality=read(root/'tune/quality.json');answers={n:validate_answers(root/n/'answers.jsonl',count)for n,count in [('test-1',1919),('tune',2048)]}
    assert (root/'answers.jsonl').read_bytes()==(root/'test-1/answers.jsonl').read_bytes()
    assert summary['status']in('passed','review_required') and summary['GPU_runs']==2 and len(summary['test'])==1
    assert summary['training']is False and summary['internet']is False and summary['single_gpu']
    assert summary['budget_seconds']==720
    expected_hashes={m['name']:m['sha256']for m in config['models']}
    for name,n in [('test-1',1919),('tune',2048)]:
        report=reports[name];assert report['status']=='complete' and report['rows']==n
        assert report['model_order']==list(expected_hashes) and report['model_sha256']==expected_hashes
        assert report['seed']==270042 and report['deterministic_algorithms']
        assert report['winner_reexecution']['checked_rows']==n
        assert not report['adaptive']['scorer_enabled'] and not report['adaptive']['undecided_rows']
        stopped=set(report['adaptive']['stopped_rows'])
        assert all(c==sum(m['K1'if i in stopped else'K2']for m in config['models'])for i,c in enumerate(report['deadline_guard']['executed_sample_counts']))
        assert report['execution']['completed_unique']==report['execution']['unique_programs'] and report['execution']['error']is None
        assert report['single_gpu'] and report['deadline_guard']['generation_stop_seconds']==160.
        assert not report['deadline_guard']['triggered']
        assert report['deadline_guard']['fallback_count']==report['deadline_guard']['partial_vote_count']==0
        assert report['winner_reexecution']['complete'] and not report['winner_reexecution']['mismatch_rows']
        assert all(b['sequences']<=config['sequence_budget']for b in report['generation_batches'])
        assert report['peak_reserved_bytes']<report['total_device_memory_bytes']
        assert report['peak_allocated_bytes']+256*1024**2<report['total_device_memory_bytes']
        assert report['adaptive']['K1']=={m['name']:m['K1']for m in config['models']}
        assert report['adaptive']['K2']=={m['name']:m['K2']for m in config['models']}
        assert {b['phase']for b in report['generation_batches']if b['stage']==2}=={'pilot-b','W-B6'}
        cfg=read(root/name/'config.json');assert cfg['expected_rows']==n and ('reference'in cfg)==(name=='tune')
        assert report['execution']['service_state']['nice']>=10
    inputs=WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    test_source=read(inputs/'test.json');test=reports['test-1'];tune=reports['tune']
    assert test==summary['test'][0] and tune==summary['tune'] and quality==summary['quality']
    from final_split import fixed_split
    ids=fixed_split()['tune2048'];heldout=read(inputs/'heldout4k.json')
    tune_rows=[dict(question=heldout[i]['question'])for i in ids]
    assert tune['input_sha256']==hashlib.sha256((json.dumps(tune_rows,ensure_ascii=False,indent=2)+'\n').encode()).hexdigest()
    assert tune['question_order_sha256']==hashlib.sha256(json.dumps([r['question']for r in tune_rows],ensure_ascii=False).encode()).hexdigest()
    assert test['input_sha256']==sha(inputs/'test.json')
    assert test['question_order_sha256']==hashlib.sha256(json.dumps([r['question']for r in test_source],ensure_ascii=False).encode()).hexdigest()
    with gzip.open(root/'tune/candidate-journal.json.gz','rt')as f:journal=json.load(f)
    references=read(root/'tune2048-reference.json');assert len(journal)==len(references)==2048
    refs=read(inputs/'heldout4k-reference-execution.json');refs=refs['rows']if isinstance(refs,dict)else refs
    assert references==[refs[i]for i in ids]
    matches=0;match_indices=[]
    for i,(row,ref)in enumerate(zip(journal,references)):
        winner,_=select_available(row,config['models']);assert winner['code']==answers['tune'][i]['code']
        hit=ref['status']=='numeric'and same(ref,winner['execution']);matches+=hit
        if hit:match_indices.append(i)
    assert matches==quality['matches'] and quality['n']==2048
    assert quality['expected_matches']==1553 and quality['match_indices']==match_indices
    assert quality['accuracy_pp']==100*matches/2048
    previous=validate_answers(WORK/'submissions/final-ensemble-v8.jsonl',1919)
    changed=[i for i,(a,b)in enumerate(zip(answers['test-1'],previous))if a['code']!=b['code']]
    executions,execution_receipt=execute(answers['test-1']+previous)
    current,old=executions[:1919],executions[1919:]
    different=[i for i,(a,b)in enumerate(zip(current,old))if not same(a,b)]
    nonnumeric=[i for i,(a,b)in enumerate(zip(current,old))if a['status']!='numeric'or b['status']!='numeric']
    assert not nonnumeric,('Nonnumeric comparisons require review',nonnumeric)
    destination=WORK/'submissions/final-ensemble-v10.jsonl'
    identity=read(ROOT/'final-v10-source-identity.json');modules=root/'src'
    assert {n:sha(modules/n)for n in identity['embedded_source_sha256']}==identity['embedded_source_sha256']
    if destination.exists():assert destination.read_bytes()==(root/'answers.jsonl').read_bytes()
    else:destination.write_bytes((root/'answers.jsonl').read_bytes())
    validate_answers(destination,1919)
    result=dict(passed=True,test1919_seconds=test['full_pipeline_seconds'],test_target_140_passed=test['full_pipeline_seconds']<=140,
        tune_seconds=tune['full_pipeline_seconds'],tune_matches=matches,tune_n=2048,tune_accuracy_pp=100*matches/2048,
        v8_fresh_matches=1532,delta_fresh_vs_v8_matches=matches-1532,delta_fresh_vs_v8_pp=100*(matches-1532)/2048,
        runtime_replay_verified=True,early_stops={n:len(r['adaptive']['stopped_rows'])for n,r in reports.items()},
        mean_samples={n:r['adaptive']['mean_sample_count']for n,r in reports.items()},
        offline_matches=1553,v8_offline_matches=1541,delta_fresh_vs_offline_matches=matches-1553,
        changed_code_vs_v8=len(changed),changed_code_vs_v8_rows=changed,changed_executed_answer_vs_v8=len(different),changed_executed_answer_vs_v8_rows=different,
        execution_comparison=dict(method='independent fresh bounded production executor for both files, numeric isclose rel/abs 1e-6',nonnumeric_rows=nonnumeric,receipt=execution_receipt),
        origin='private API T4 run; version-bound output and exact code cells',remote_ref=download['ref'],remote_version=download['version'],
        verified_model_sha256=expected_hashes,tune_order_verified=True,requested_account='friend1',own_push_calls=1,
        submission=str(destination.relative_to(WORK)),answers_sha256=sha(destination),rows=1919,code_only=True,test_order_verified=True,
        peak_memory_bytes={n:{k:r[k]for k in ['peak_reserved_bytes','peak_allocated_bytes','total_device_memory_bytes']}for n,r in reports.items()},
        OOM_attempts={n:r['OOM_attempts']for n,r in reports.items()},session_seconds=summary['session_seconds'],summary_status=summary['status'],gates=summary['gates'],all_20_remote_sources_exact=True)
    comparison=dict(rows=1919,v10_sha256=sha(destination),v8_sha256=sha(WORK/'submissions/final-ensemble-v8.jsonl'),
        changed_code_vs_v8=len(changed),changed_code_vs_v8_rows=changed,changed_executed_answer_vs_v8=len(different),
        changed_executed_answer_vs_v8_rows=different,execution=result['execution_comparison'])
    (WORK/'checks/final-inference-v10-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    (WORK/'checks/final-inference-v10-output.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (WORK/'results/final-inference-v10.json').write_text(json.dumps({k:v for k,v in result.items()if not k.endswith('_rows')},ensure_ascii=False,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    r=check(p.parse_args().output)
    print(json.dumps({k:v for k,v in r.items()if not k.endswith('_rows')},ensure_ascii=False,indent=2))
