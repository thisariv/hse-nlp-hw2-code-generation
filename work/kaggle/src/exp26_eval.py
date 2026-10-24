"""Identical25 model/greedy/CE/exec math; new point names and paired comparisons."""
import gc,hashlib,json,math,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import exp26_core as core
import exp26_exec as numeric
import exp26_protocol as P
from exp26_common import json_hash,write_json,read_json

def state_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        a=v.detach().cpu().contiguous().numpy();h.update(k.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def load_model(path, device, dropout=.1, scaled_tokens=False, spec=None):
    model=core.SimpleTransformer(**(spec or core.MODEL_SPEC),dropout=dropout)
    payload=torch.load(path,map_location='cpu',weights_only=True)
    state=payload['model'] if payload.get('format')in ('nlp-hw02-training-v1','nlp-hw02-wsd-v1')else payload
    model.load_state_dict(state,strict=True)
    del payload
    if scaled_tokens:
        with torch.no_grad():model.tokens.weight.div_(math.sqrt(model.width))
    del state
    return model.to(device)

@torch.inference_mode()
def ce_one(model_path,rows,tokenizer,device,spec=None,batch=32):
    started=time.perf_counter();model=load_model(model_path,device,spec=spec).eval();total=torch.zeros((),device=device);denominator=0
    cached=[([core.BOS]+core.encode(tokenizer,r['question'])+[core.EOS],core.encode(tokenizer,r['code'])) for r in rows]
    order=sorted(range(len(rows)),key=lambda i:len(cached[i][0])+len(cached[i][1]))
    try:
        for lo in range(0,len(order),batch):
            src,tgt,labels,counts=core.collate_cache(cached,order[lo:lo+batch],device)
            logits=model(src,tgt)
            total+=F.cross_entropy(logits.reshape(-1,model.vocab),labels.reshape(-1),ignore_index=core.PAD,reduction='sum')
            denominator+=counts['target_tokens']
            del src,tgt,labels,logits
        value=float(total.cpu())/denominator
        if not math.isfinite(value):raise FloatingPointError('CE nonfinite')
        return {'CE':value,'useful_target_tokens':denominator,'rows':len(rows),'precision':'FP32','batch':batch,
                'seconds':time.perf_counter()-started}
    finally:
        del model;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()

@torch.inference_mode()
def generate_one(model_path,rows,tokenizer_path,device,outdir,name,spec=None,batch=512,cap=512,half=True):
    """Full timer begins before loading model tensors; ends at answers-file close."""
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter();model=load_model(model_path,device,spec=spec).eval()
    tokenizer,tokenbytes=core.init_tokenizer(tokenizer_path)
    if half:model.half()
    try:
        core.generate_preallocated(model,core.source_batch(tokenizer,[rows[0]['question']],device),cap=min(8,cap))
        core.sync(device);generation_start=time.perf_counter()
        encoded=[[core.BOS]+core.encode(tokenizer,r['question'])+[core.EOS] for r in rows]
        order=sorted(range(len(rows)),key=lambda i:(len(encoded[i]),i));ordered=[];steps_sum=row_steps_sum=0
        for lo in range(0,len(order),batch):
            indices=order[lo:lo+batch]
            src=nn.utils.rnn.pad_sequence([torch.tensor(encoded[i]) for i in indices],batch_first=True,padding_value=core.PAD).to(device)
            tokens,steps=core.generate_preallocated(model,src,cap=cap)
            steps_sum+=steps;row_steps_sum+=steps*len(indices)
            for i,ids in zip(indices,tokens.cpu().tolist()):
                active=core.active_tokens(ids)
                ordered.append({'id':i,'question':rows[i]['question'],'generated_token_ids':active,
                    'generated_code':core.decode_bytes(tokenbytes,active).decode('utf-8',errors='replace'),
                    'eos_emitted':core.EOS in active})
        examples=core.restore_order(ordered,order,len(rows));core.sync(device)
        generation_seconds=time.perf_counter()-generation_start
        outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
        answers=outdir/f'{name}-answers.jsonl'
        with answers.open('w',encoding='utf-8') as f:
            for e in examples:f.write(json.dumps({'id':e['id'],'question':e['question'],'code':e['generated_code']},ensure_ascii=False)+'\n')
        full_seconds=time.perf_counter()-start
        core.write_json(outdir/f'{name}-generations.json',examples)
        timing={'full_seconds':full_seconds,'generation_seconds':generation_seconds,'rows':len(rows),
                'decode_steps_sum':steps_sum,'decode_row_steps_sum':row_steps_sum,
                'answers_sha256':core.sha_file(answers),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None,
                'timer':'model state load,restore,tokenizer,FP16/H2D,warmup,sorted greedy/order restore/answers write; download/cache/exec/CE/JSON dump outside'}
        timing['model_sha256']=core.sha_file(model_path)
        timing['tokenizer_sha256']=core.sha_file(tokenizer_path)
        return examples,timing
    finally:
        del model;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()

def paired_mcnemar(score,control):
    assert score['eligible_indices']==control['eligible_indices']
    s,c=set(score['match_indices']),set(control['match_indices']);g,l=len(s-c),len(c-s);n=g+l
    p=min(1.,(2*sum(math.comb(n,i) for i in range(min(g,l)+1)))/(1<<n)) if n else 1.
    return {'gains':g,'losses':l,'discordant':n,'exact_two_sided_p':p,
            'delta_percentage_points':100*(g-l)/score['denominator']}

def length_accuracy(rows,tokenizer,score):
    lengths=[len(core.encode(tokenizer,r['code'])) for r in rows];eligible=set(score['eligible_indices']);matched=set(score['match_indices'])
    groups={k:[] for k in ['<60','60–90','90–120','≥120']}
    for i,n in enumerate(lengths):
        if i in eligible:groups['<60' if n<60 else '60–90' if n<90 else '90–120' if n<120 else '≥120'].append(i)
    return {k:{'denominator':len(ids),'matches':sum(i in matched for i in ids),
               'accuracy':sum(i in matched for i in ids)/len(ids) if ids else None} for k,ids in groups.items()}

def evaluate_point(model_path,sets,refs,tokenizer_path,device,out,point,D0=None,previous=None,spec=None,smoke=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);digest=core.sha_file(model_path)
 identity={'model_sha256':digest,'protocol_sha256':json_hash(P.TRAIN_PROTOCOL),'tokenizer_sha256':core.sha_file(tokenizer_path),'sets_sha256':json_hash(sets),'torch':str(torch.__version__),'CUDA':torch.version.cuda,'smoke':smoke}
 receipt=out/'metrics.json'
 if receipt.exists():
  m=read_json(receipt);assert m['identity']==identity;return m
 tokenizer,_=core.init_tokenizer(tokenizer_path);m={'point':point,'identity':identity,'splits':{}}
 for split,rows in sets.items():
  gen_path=out/(split+'-generations.json');time_path=out/(split+'-generation-time.json')
  if gen_path.exists()and time_path.exists():
   gens=read_json(gen_path);timing=read_json(time_path)
   assert len(gens)==len(rows)and [g['question']for g in gens]==[r['question']for r in rows]
   assert timing['model_sha256']==digest and timing['tokenizer_sha256']==identity['tokenizer_sha256']
   assert timing['answers_sha256']==core.sha_file(out/(split+'-answers.jsonl'))
  else:
   gens,timing=generate_one(model_path,rows,tokenizer_path,device,out,split,spec=spec,batch=2 if smoke else 512,cap=8 if smoke else 512,half=True)
   write_json(time_path,timing)
  execution=numeric.evaluate_programs([g['generated_code']for g in gens]);assert execution['summary']['complete']
  write_json(out/(split+'-execution.json'),execution);score=numeric.compare_exec(refs[split],execution)
  if not smoke:assert score['denominator']==(4096 if split=='heldout4k'else 991)
  ce=ce_one(model_path,rows,tokenizer,device,spec=spec,batch=2 if smoke else 32)
  row={'exec':score,'CE':ce,'length_accuracy':length_accuracy(rows,tokenizer,score),'generation_time':timing}
  if D0 is not None:row['McNemar_against_D0']=paired_mcnemar(score,D0['splits'][split]['exec'])
  if previous is not None:row['McNemar_against_previous']=paired_mcnemar(score,previous['splits'][split]['exec'])
  m['splits'][split]=row;write_json(out/'evaluation-progress.json',m)
 write_json(receipt,m);return m

def verify_D0(m):
 expected={'heldout4k':(1677,.4764168630129111),'old1024':(422,.4863326656808216)}
 for split,(matches,ce)in expected.items():
  s=m['splits'][split];assert abs(s['exec']['matches']-matches)<=3 and abs(s['CE']['CE']-ce)<=1e-4,'D0 reproduction failed; stop before training'
 return {'status':'passed','expected4k':1677,'expected991':422,'match_tolerance':3,'CE_abs_tolerance':1e-4}
