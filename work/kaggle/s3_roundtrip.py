#!/usr/bin/env python3
"""CPU execution gate over the existing stratified train sample, before any push."""
import json, sys, time, math
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'src'))
from s3_representation import transform, restore, tokenizer
from exp26_fast_exec import run_chunks
from exp26_common import read_json, write_json, sha_file, json_hash

def main():
    work = ROOT.parent
    sample = work/'research/outputs/data-ceiling/sample-60k.json'
    original = read_json(work/'research/outputs/data-ceiling/normalization-execution.json')
    rows = [r for r in read_json(sample) if original[r['code']]['status'] == 'numeric']
    assert len(rows) >= 50000
    report = {'sample_sha256': sha_file(sample), 'rows': len(rows), 'variants': {},
              'reference_execution': 'previous independently verified original execution; same frozen executor'}
    tokpath = work/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json'
    for variant in (30, 31):
        start = time.monotonic()
        tok, tokenbytes = tokenizer(tokpath, variant)
        codes, modes, original_tokens, transformed_tokens = [], Counter(), 0, 0
        for lo in range(0, len(rows),2000):
            part = rows[lo:lo+2000]
            texts = [transform(r['question'],r['code'],variant) for r in part]
            encoded = tok.encode_batch([c for q,c in texts], add_special_tokens=False)
            for r,(q,text),encoding in zip(part,texts,encoded):
                decoded = b''.join(tokenbytes[i+3] for i in encoding.ids).decode()
                assert decoded == text
                codes.append(restore(decoded,r['question'],variant))
                if variant == 30:modes[text.split('|>')[0]+'|>'] += 1
            original_tokens += sum(len(e.ids) for e in tok.encode_batch([r['code'] for r in part],add_special_tokens=False))
            transformed_tokens += sum(len(e.ids) for e in encoded)
        transform_seconds = time.monotonic()-start
        result = run_chunks(codes,workers=4,chunk_rows=2000,wall_limit=1.,cpu_limit=2.)
        mismatches=[]
        for i,(r,out) in enumerate(zip(rows,result['rows'])):
            expected = original[r['code']]
            if out['status'] != 'numeric' or not math.isclose(out['value'],expected['value'],rel_tol=1e-6,abs_tol=1e-6):
                mismatches.append({'sample_index':i,'shard':r['shard'],'row':r['row'],'status':out['status'],
                                   'expected':expected.get('value'),'actual':out.get('value')})
        report['variants'][str(variant)] = {'rows':len(rows),'mismatches':len(mismatches),'examples':mismatches[:20],
            'transform_and_token_roundtrip_seconds':transform_seconds,'executor':result['summary'],
            'restored_codes_sha256':json_hash(codes),'vocab':tok.get_vocab_size()+3,'modes':dict(modes),
            'original_content_tokens':original_tokens,'represented_content_tokens':transformed_tokens}
        write_json(work/'checks/kaggle-s3-roundtrip.json',report)
        print(variant, len(rows), 'mismatches',len(mismatches),'transform_seconds',round(transform_seconds,2),flush=True)
        assert not mismatches
    report['passed']=True
    write_json(work/'checks/kaggle-s3-roundtrip.json',report)

if __name__=='__main__':main()
