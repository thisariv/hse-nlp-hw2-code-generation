"""Torch/NumPy-free entrypoint for bounded program execution under RLIMIT_AS."""
import argparse, ast, json, time
from collections import Counter
from pathlib import Path
from kaggle_io import write
from exp26_fast_exec import run_chunks
from exp26_exec import compare_exec

def run(root,workers=4):
    root=Path(root);started=time.monotonic();samples=[]
    for p in sorted((root/'shards').glob('*/roundtrip-sample.json')):samples.extend(json.loads(p.read_text()))
    assert samples,'No original/restored samples'
    ast_mismatches=[i for i,s in enumerate(samples) if ast.dump(ast.parse(s['original']))!=ast.dump(ast.parse(s['restored']))]
    original=run_chunks([s['original'] for s in samples],workers=workers,wall_limit=3.,cpu_limit=2.)
    write(root/'roundtrip-original-execution.json',original)
    restored=run_chunks([s['restored'] for s in samples],workers=workers,wall_limit=3.,cpu_limit=2.)
    write(root/'roundtrip-restored-execution.json',restored)
    score=compare_exec(original,restored)
    matched=set(score['match_indices'])
    bad=[{'sample':i,'shard':samples[i]['shard'],'local_row':samples[i]['local_row'],
          'original':original['rows'][i],'restored':restored['rows'][i]} for i in range(len(samples))
         if i not in matched]
    result={'rows':len(samples),'matches':score['matches'],'numeric_original':score['denominator'],
            'answer_mismatches':score['denominator']-score['matches'],'AST_mismatches':ast_mismatches,
            'original_statuses':dict(Counter(r['status'] for r in original['rows'])),
            'restored_statuses':dict(Counter(r['status'] for r in restored['rows'])),
            'failures':bad,'score':score,'seconds':time.monotonic()-started,
            'executor_entrypoint':'l1_roundtrip.py: no NumPy/Torch imports; wall3s/cpu2s'}
    write(root/'roundtrip-report.json',result)
    print('ROUNDTRIP',json.dumps({k:result[k] for k in ('rows','matches','numeric_original','answer_mismatches','original_statuses','restored_statuses','seconds')}),flush=True)
    assert not ast_mismatches and result['matches']==result['rows'],'Round-trip failed; diagnostics saved'
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    run(a.root,a.workers)
