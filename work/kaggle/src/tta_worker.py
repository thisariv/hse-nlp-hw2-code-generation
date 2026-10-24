"""Bounded fresh TTA pool on one visible T4, frozen v8 model/decode/execution."""
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

def write(p,x):
    p=Path(p); tmp=p.with_suffix('.tmp'); tmp.write_text(json.dumps(x,ensure_ascii=False)+'\n'); tmp.replace(p)

def main(config):
    import torch
    from torch import nn
    import final_core as C
    import final_fast as Fast
    import final_speed as Speed
    from final_inference import load, pack, DECODE_OPTIONS
    from final_representation import tokenizer
    from final_session import prepare_inputs, weight_path
    from final_mounts import find_input, resolve_cache_mount
    from final_executor_v6 import Executor
    from final_split import fixed_split
    from tta_rules import apply
    start=time.monotonic(); deadline=float(os.environ['TTA_DEADLINE'])-25
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    fixture=config.get('cpu_fixture',False)
    if not fixture: assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
    device=torch.device('cpu' if fixture else 'cuda:0'); out=Path(config['output']); out.mkdir(parents=True,exist_ok=True)
    inputs=config['inputs'] if fixture else dict(prepare_inputs(find_input('nlp-hw02-26b-inputs'),out/'inputs'))
    rows=json.loads(Path(inputs['heldout4k.json']).read_text())
    refs=json.loads(Path(inputs['heldout4k-reference-execution.json']).read_text())['rows']
    tok,table=tokenizer(inputs['tokenizer.json'],30)
    branches=['original']+config['rules']; splits=fixed_split(len(rows)); eligible={}
    for branch in branches:
        eligible[branch]=[i for i,r in enumerate(rows) if branch=='original' or apply(r['question'],branch)!=r['question']]
    report=dict(status='running',K=config['K'],T=.7,seed=config['seed'],single_visible_T4=not fixture,
                training=False,branches=branches,coverage={k:len(v) for k,v in eligible.items()},models=[],
                decoder_options=vars(DECODE_OPTIONS),input_sha256={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in inputs.items()})
    write(out/'summary.json',report)
    executor=Executor(2)
    try:
        for model_index,spec in enumerate(config['models']):
            if fixture: path=spec['path']
            else:
                root=resolve_cache_mount(Path('/kaggle/input'),spec['dataset'],spec.get('source_kind','datasets'))
                path=weight_path(root,spec)
            actual=hashlib.sha256(Path(path).read_bytes()).hexdigest(); assert actual==spec['sha256']
            model=load(path,device); model_report=dict(name=spec['name'],sha256=actual,branches=[])
            print('MODEL',spec['name'],'loaded',round(time.monotonic()-start,1),flush=True)
            for branch_index,branch in enumerate(branches):
                ids=eligible[branch]; folder=out/spec['name']/branch; folder.mkdir(parents=True,exist_ok=True)
                branch_start=time.monotonic(); completed=0; attempts=[]; batch=config['batch']
                # Separate deterministic streams, same seed policy for every model/branch.
                seed=config['seed']+1009*model_index+100003*branch_index
                torch.manual_seed(seed)
                if not fixture: torch.cuda.manual_seed_all(seed)
                for split,split_ids in splits.items():
                    applicable=set(ids); selected_ids=[i for i in split_ids if i in applicable]; lo=0
                    while lo<len(selected_ids):
                        C.check_deadline(deadline)
                        chosen=selected_ids[lo:lo+batch]; questions=[rows[i]['question'] if branch=='original' else apply(rows[i]['question'],branch) for i in chosen]
                        rng=torch.get_rng_state(); cuda_rng=None if fixture else torch.cuda.get_rng_state()
                        src=ctx=seq=lps=gseq=glps=None
                        try:
                            encoded=[[C.BOS]+C.encode(tok,q)+[C.EOS] for q in questions]
                            src=nn.utils.rnn.pad_sequence([torch.tensor(x) for x in encoded],batch_first=True,padding_value=C.PAD).to(device)
                            ctx=Fast.Context(model,src,config.get('cap',512))
                            gseq,glps,_=Speed.decode(ctx,k=1,temperature=None,deadline=deadline,options=DECODE_OPTIONS,compute_logprob=False)
                            greedies=[g[0] for g in pack(gseq,glps,table,greedy=True)]
                            seq,lps,_=Speed.decode(ctx,k=config['K'],temperature=.7,deadline=deadline,options=DECODE_OPTIONS)
                            groups=pack(seq,lps,table)
                            candidates=[c for g,s in zip(greedies,groups) for c in [g]+s]
                            executor.submit(candidates); assert executor.drain(deadline), 'Execution incomplete'
                            executor.attach(candidates)
                            records=[]
                            for i,q,g,s in zip(chosen,questions,greedies,groups):
                                for j,c in enumerate(s):
                                    c.update(sample=j,sampling_status='complete' if c['mean_logprob'] is not None else 'nonfinite_logprob')
                                records.append(dict(id=i,split=split,question_sha256=hashlib.sha256(rows[i]['question'].encode()).hexdigest(),
                                    input_sha256=hashlib.sha256(q.encode()).hexdigest(),variant=branch,greedy=g,reference=refs[i],samples={'0.7':s}))
                            write(folder/f'{split}-batch-{lo:05d}.json',records)
                            lo+=len(chosen); completed+=len(chosen)
                            print('BATCH',spec['name'],branch,split,lo,'session_s',round(time.monotonic()-start,1),flush=True)
                        except torch.cuda.OutOfMemoryError:
                            torch.set_rng_state(rng); torch.cuda.set_rng_state(cuda_rng); attempts.append(dict(split=split,offset=lo,batch=batch)); batch//=2
                            if batch<1: raise
                        finally:
                            del src,ctx,seq,lps,gseq,glps
                            gc.collect(); torch.cuda.empty_cache()
                br=dict(variant=branch,complete=completed==len(ids),rows=completed,K=config['K'],T=.7,seed=seed,
                        seconds=time.monotonic()-branch_start,OOM=attempts)
                write(folder/'summary.json',br); model_report['branches'].append(br); write(out/'summary.json',report)
            del model; gc.collect(); torch.cuda.empty_cache()
            report['models'].append(model_report); write(out/'summary.json',report)
        report['status']='complete'
    finally:
        report['executor']=executor.summary(); executor.close(); report['seconds']=time.monotonic()-start
        write(out/'summary.json',report)
    print('TTA COMPLETE',report['status'],round(report['seconds'],2),flush=True)

if __name__=='__main__': main(json.loads(Path(sys.argv[1]).read_text()))
