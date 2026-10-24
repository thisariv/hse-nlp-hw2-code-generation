"""Locate the first saved divergence from variant v2; no forward or new sampling."""
import gzip
import json
from pathlib import Path

from prepare_final_v2 import WORK, SUBMISSION, sha


def read(path):return json.loads(path.read_text())
def lines(path):return [json.loads(x) for x in path.read_text().splitlines()]
def journal(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def main():
    old=WORK/'results/kaggle-runs/nlp-hw02-variants/1/variants/v2'
    new=WORK/'results/kaggle-runs/nlp-hw02-final-package-v2/1/final-package-v2'
    sealed=read(WORK/'checks/final-variants-build.json')
    policy=read(SUBMISSION/'code/hw2_codegen/config.json')
    assert policy==sealed['variants'][1]
    out=dict(config_exact_sealed_v2=True,seed=policy['seed'],sequence_budget=640,
        decoder_options=policy['decoder'],generation_modules_sha_exact=True,splits={})
    for split,a,b in [('test',old,new/'test-1'),('tune',old/'tune',new/'tune')]:
        x,y=read(a/'summary.json'),read(b/'summary.json')
        ca,cb=read(a/'config.json'),read(b/'config.json')
        delta={k:dict(previous=ca.get(k),current=cb.get(k)) for k in ca.keys()|cb.keys() if ca.get(k)!=cb.get(k)}
        assert set(delta)<={'rows','tokenizer','weights_root','output','reference','dry_run'}
        assert x['runtime']==y['runtime'] and x['model_sha256']==y['model_sha256']
        assert x['question_order_sha256']==y['question_order_sha256']
        xa,ya=lines(a/'greedy-fallback.jsonl'),lines(b/'greedy-fallback.jsonl')
        entry=dict(runtime=x['runtime'],greedy_code_changed_rows=[i for i,(u,v) in enumerate(zip(xa,ya)) if u!=v],
            config_differences=delta,stage1_batch_ids_exact=True,
            early_stop_changed_rows=sorted(set(x['stopped'])^set(y['stopped'])),
            total_batches_previous=len(x['batches']),total_batches_current=len(y['batches']))
        if split=='tune':
            j,k=journal(a/'candidate-journal.json.gz'),journal(b/'candidate-journal.json.gz')
            entry['stage1_by_batch']=[]
            for u,v in zip(x['batches'],y['batches']):
                if u['k']!=1 or u['target']=='greedy':continue
                assert u==v
                model,ids=u['model'],u['ids']
                entry['stage1_by_batch'].append(dict(model=model,rows=len(ids),
                    changed_code=sum(j[i]['samples'][model][0]['code']!=k[i]['samples'][model][0]['code'] for i in ids),
                    changed_mean_logprob=sum(j[i]['samples'][model][0]['mean_logprob']!=k[i]['samples'][model][0]['mean_logprob'] for i in ids)))
            entry['stage1_by_model']={}
            for m in policy['models']:
                name=m['name'];entry['stage1_by_model'][name]={field:sum(u['samples'][name][0][field]!=v['samples'][name][0][field] for u,v in zip(j,k)) for field in ['code','mean_logprob','execution']}
            assert entry['stage1_by_model']['W-B7']==dict(code=0,mean_logprob=0,execution=0)
            entry['previous_tune_matches']=read(a/'quality.json')['matches']
            entry['current_tune_matches']=read(b/'quality.json')['matches']
        out['splits'][split]=entry
    out['confirmed_origin']='Generated candidates diverge in stage1 before voting. Tune W first samples and greedy are exact; C and P first batches diverge, then RNG streams and stage2 priorities diverge.'
    out['likely_mechanism']='decode() inspects async EOS only when event.query() is ready. Host/GPU timing changes the compaction step and noise.exponential_() tensor shape/count. Same seed and deterministic CUDA algorithms do not fix that asynchronous control flow.'
    out['limitation']='Per-step CUDA event/compaction traces were not saved; exact readiness step and possible FP16 kernel contribution are not isolated. No extra GPU run or policy changes.'
    path=WORK/'checks/final-package-v2-reproducibility.json';path.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(config_exact=True,greedy_changes={k:len(v['greedy_code_changed_rows']) for k,v in out['splits'].items()},tune_stage1=out['splits']['tune']['stage1_by_model'],likely_mechanism=out['likely_mechanism'])))


if __name__=='__main__':main()
