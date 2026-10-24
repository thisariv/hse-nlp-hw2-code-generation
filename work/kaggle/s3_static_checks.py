#!/usr/bin/env python3
"""Verify full initial A tensors, all pinned A candidates, and the actual assembled source."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))

def main():
    import torch
    import s3_runtime as S,exp26_protocol as P
    from exp26_common import read_json,write_json,sha_file
    from exp26_eval import state_digest
    from exp26_core import init_tokenizer
    from s3_score import flatten
    from build_s3 import main as build
    build();torch.set_num_threads(1)
    w=ROOT.parent;checks=[];inp=w/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs'
    for variant in (30,31):
        model,opt,scaler,state,meta,cfg,spec=S.initialize(inp,torch.device('cpu'),variant,smoke=False)
        old={k:v[:8192] if k in ('tokens.weight','head.weight','head.bias') else v for k,v in model.state_dict().items()}
        assert state_digest(old)==P.NORMALIZED_INITIAL_TENSORS
        checks.append({'check':f'full A {variant}: exact normalized original tensor digest; fresh special embeddings',
            'passed':True,'vocab':spec['vocab'],'parameters':sum(p.numel() for p in model.parameters()),
            'new_embedding_std':float(model.tokens.weight[8192:].detach().std())})
        del model,opt,scaler
    nb=read_json(ROOT/'notebooks/nlp-hw02-kaggle-s3.ipynb')
    for cell in nb['cells']:
        if cell['cell_type']=='code':compile(cell['source'],'s3 notebook','exec')
    meta=read_json(ROOT/'bundles/s3/kernel-metadata.json');limits=read_json(ROOT/'bundles/s3/session-limits.json')
    assert meta['is_private'] is True and meta['enable_internet'] is True and meta['machine_shape']=='NvidiaTeslaT4'
    assert meta['kernel_sources']==['thisarivson/nlp-hw02-kaggle-s1/1','thisarivson/nlp-hw02-kaggle-s2/2']
    assert limits=={'session_timeout_seconds':6300,'budget_seconds':5400,'watchdog_exit_seconds':6000}
    sources=ast.literal_eval(ast.parse(nb['cells'][3]['source']).body[0].value)
    for name,digest in read_json(ROOT/'source-identity.json')['modules'].items():assert hashlib.sha256(sources[name].encode()).hexdigest()==digest
    for name,source in sources.items():compile(source,name,'exec')
    assert 'WATCHDOG=Watchdog(CONFIG).start()' in nb['cells'][1]['source']
    checks.append({'check':'frozen26b unchanged; all notebook/worker cells compile; watchdog before install; private T4/pinned sources/90-105min limits','passed':True})
    tok,_=init_tokenizer(inp/'tokenizer.json')
    for split in ('heldout4k','old1024'):
        rows=read_json(inp/(split+'.json'));seen=set();count=files=0
        for f in sorted((w/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/gpu0-exp27'/split/'candidates').glob('batch-*.json')):
            if '.sha.' in f.name:continue
            assert sha_file(f)==read_json(f.with_suffix('.sha.json'))['sha256'];files+=1
            for r in read_json(f):
                i=r['id'];assert i not in seen and r['question']==rows[i]['question'];seen.add(i)
                cc=flatten(r,tok);assert len(cc)==65 and [c['candidate_index'] for c in cc]==list(range(65));count+=len(cc)
        assert seen==set(range(len(rows))) and count==65*len(rows)
        checks.append({'check':split+' full S1 SHA/id/question/all65 coverage','passed':True,'questions':len(rows),'candidates':count,'files':files})
    for f in ('kaggle-s3-smoke.json','kaggle-s3-roundtrip.json'):assert read_json(w/'checks'/f)['passed']
    report={'passed':True,'checks':checks,'notebook_sha256':sha_file(ROOT/'notebooks/nlp-hw02-kaggle-s3.ipynb'),'metadata':meta,'limits':limits}
    write_json(w/'checks/kaggle-s3-static.json',report);print(json.dumps(checks,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
