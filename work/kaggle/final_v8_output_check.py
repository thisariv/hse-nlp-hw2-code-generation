"""Validate the first cold v8 output, publish code-only test JSONL and compare v7."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from build_final_inference_v8 import default_config

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from final_vote import validate_answers


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(root):
    root=Path(root)
    summary=read(root/'summary.json')
    test=read(root/'test-1/summary.json');tune=read(root/'tune/summary.json')
    quality=read(root/'tune/quality.json');config=default_config()
    test_rows=validate_answers(root/'test-1/answers.jsonl',1919)
    tune_rows=validate_answers(root/'tune/answers.jsonl',2048)
    assert (root/'answers.jsonl').read_bytes()==(root/'test-1/answers.jsonl').read_bytes()
    expected_hashes={m['name']:m['sha256']for m in config['models']}
    for name,report,n in [('test-1',test,1919),('tune',tune,2048)]:
        assert report['rows']==n and report['status']=='complete'
        assert report['model_order']==list(expected_hashes) and report['model_sha256']==expected_hashes
        assert report['single_gpu'] and report['deadline_guard']['generation_stop_seconds']==160.
        assert not report['deadline_guard']['triggered']
        assert report['deadline_guard']['fallback_count']==report['deadline_guard']['partial_vote_count']==0
        assert report['winner_reexecution']['complete'] and not report['winner_reexecution']['mismatch_rows']
        assert all(b['sequences']<=config['sequence_budget'] for b in report['generation_batches'])
        assert report['peak_reserved_bytes']<report['total_device_memory_bytes']
        assert report['peak_allocated_bytes']+256*1024**2<report['total_device_memory_bytes']
        assert report['adaptive']['K1']=={m['name']:1 for m in config['models']}
        assert report['adaptive']['K2']=={m['name']:m['K2']for m in config['models']}
        assert all(b['phase']=='pilot-b'for b in report['generation_batches']if b['stage']==2)
        cfg=read(root/name/'config.json')
        # Test and tune both execute through the fresh child supervisor.
        assert cfg['expected_rows']==n and ('reference'in cfg)==(name=='tune')
    inputs=WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    test_source=read(inputs/'test.json')
    assert test['input_sha256']==sha(inputs/'test.json')
    assert test['question_order_sha256']==hashlib.sha256(json.dumps(
        [r['question']for r in test_source],ensure_ascii=False).encode()).hexdigest()
    with gzip.open(root/'tune/candidate-journal.json.gz','rt')as stream:journal=json.load(stream)
    assert len(journal)==2048
    references=read(root/'tune2048-reference.json')
    from final_ensemble import select_available
    import math
    matches=0
    for i,(row,ref)in enumerate(zip(journal,references)):
        winner,_=select_available(row,config['models'])
        assert winner['code']==tune_rows[i]['code']
        result=winner['execution']
        matches+=ref['status']==result['status']=='numeric'and math.isclose(ref['value'],result['value'],rel_tol=1e-6,abs_tol=1e-6)
    assert matches==quality['matches'] and quality['n']==2048
    destination=WORK/'submissions/final-ensemble-v8.jsonl'
    destination.write_bytes((root/'answers.jsonl').read_bytes());validate_answers(destination,1919)
    previous=validate_answers(WORK/'submissions/final-ensemble3-v7.jsonl',1919)
    changed=[i for i,(a,b)in enumerate(zip(test_rows,previous))if a['code']!=b['code']]
    result=dict(passed=True,test1919_seconds=test['full_pipeline_seconds'],
        test_target_140_passed=test['full_pipeline_seconds']<=140.,
        tune_seconds=tune['full_pipeline_seconds'],tune_matches=matches,tune_n=2048,
        tune_accuracy_pp=100*matches/2048,offline_matches=1541,delta_offline_matches=matches-1541,
        changed_vs_v7=len(changed),changed_vs_v7_rows=changed,
        submission=str(destination.relative_to(WORK)),answers_sha256=sha(destination),
        test_order_verified=True,code_only=True,rows=1919,
        peak_memory_bytes={name:{k:report[k]for k in ['peak_reserved_bytes','peak_allocated_bytes','total_device_memory_bytes']}
            for name,report in [('test',test),('tune',tune)]},
        OOM_attempts={name:report['OOM_attempts']for name,report in [('test',test),('tune',tune)]},
        session_seconds=summary['session_seconds'],summary_status=summary['status'])
    (WORK/'checks/final-inference-v8-output.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    result=check(args.output)
    print(json.dumps({k:v for k,v in result.items()if k!='changed_vs_v7_rows'},ensure_ascii=False,indent=2))
