"""Independent FP16 4D greedy, restored exec, token CE and fresh full-test timer."""
import gc, json, os, signal, subprocess, sys, tempfile, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer
import exp26_core as C, exp26_exec as N, exp26_eval as E
import exp27b_fast as Fast
import s3_representation as R
from exp26_fast_exec import run_chunks
from exp26_order import collate
from kaggle_io import sha, write

def load(path,device):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    assert payload['format']=='nlp-hw02-model-only-v1' and payload['optimizer_included'] is False
    model=C.SimpleTransformer(**payload['model_spec'],dropout=0.).eval()
    if device.type=='cuda':model.half()
    model.to(device);model.load_state_dict(payload['model']);del payload
    return model

@torch.no_grad()
def generate(model,rows,tok,table,device,deadline,smoke=False,batch=256):
    encoded=[[C.BOS]+C.encode(tok,r['question'])+[C.EOS] for r in rows]
    order=sorted(range(len(rows)),key=lambda i:(len(encoded[i]),i));examples=[None]*len(rows)
    lo=0; cap=8 if smoke else 512;attempts=[];C.sync(device);started=time.perf_counter()
    while lo<len(order):
        C.check_deadline(deadline); indices=order[lo:lo+batch]
        src=ctx=ids=lps=None
        try:
            src=nn.utils.rnn.pad_sequence([torch.tensor(encoded[i]) for i in indices],batch_first=True,padding_value=C.PAD).to(device)
            ctx=Fast.Context(model,src,cap);ids,lps,_=Fast.decode(ctx,deadline=deadline)
            for i,seq in zip(indices,ids[:,0].tolist()):
                active=C.active_tokens(seq);text=C.decode_bytes(table,active).decode('utf8',errors='replace')
                try:code=R.restore_compact(text);error=None
                except (ValueError,IndexError):code='';error='invalid representation'
                examples[i]={'id':i,'question':rows[i]['question'],'generated_token_ids':active,
                             'represented_code':text,'generated_code':code,'restore_error':error,'eos_emitted':C.EOS in active}
            lo+=len(indices)
        except torch.cuda.OutOfMemoryError:
            attempts.append({'lo':lo,'batch':batch,'status':'OOM'});batch//=2
            if batch<8:raise
        finally:
            del src,ctx,ids,lps
            if attempts and device.type=='cuda':gc.collect();torch.cuda.empty_cache()
    C.sync(device)
    return examples,{'seconds':time.perf_counter()-started,'batch':batch,'OOM_attempts':attempts,
                     'SDPA_dimensions':4,'precision':'FP16' if device.type=='cuda' else 'FP32 tiny smoke',
                     'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None}

def execute(examples,smoke=False,deadline=None):
    # The entire executor, including pool joins/retries, has a parent wall limit.
    C.check_deadline(deadline)
    with tempfile.TemporaryDirectory(prefix='l1-exec-') as folder:
        source=Path(folder)/'codes.json';target=Path(folder)/'result.json'
        write(source,[r['generated_code'] for r in examples])
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('l1_execute.py')),
            str(source),str(target),'1' if smoke else '2'],env=env,start_new_session=True,
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        until=min(time.monotonic()+900.,deadline) if deadline else time.monotonic()+900.
        try:
            while child.poll() is None:
                if time.monotonic()>=until:raise C.BudgetExpired('Execution wall budget exhausted')
                time.sleep(.05)
            if child.returncode:raise RuntimeError('Bounded execution child failed')
            return json.loads(target.read_text())
        finally:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            try:child.wait(timeout=.2)
            except subprocess.TimeoutExpired:pass

@torch.no_grad()
def ce(model,rows,tok,device,deadline,smoke=False):
    cached=[([1]+C.encode(tok,r['question'])+[2],C.encode(tok,R.compact(r['code']))) for r in rows]
    order=sorted(range(len(cached)),key=lambda i:len(cached[i][0])+len(cached[i][1]))
    total=0.;tokens=0;started=time.perf_counter()
    lo=0;batch=2 if smoke else 16;retries=[]
    while lo<len(order):
        C.check_deadline(deadline);src=tgt=labels=logits=None
        try:
            indices=order[lo:lo+batch];src,tgt,labels,counts=collate(cached,indices,device)
            logits=model(src,tgt).float()
            loss=float(F.cross_entropy(logits.reshape(-1,model.vocab),labels.reshape(-1),ignore_index=0,reduction='sum'))
            if not torch.isfinite(torch.tensor(loss)):raise FloatingPointError('Nonfinite evaluation CE')
            total+=loss;tokens+=counts['target_tokens'];lo+=len(indices)
        except torch.cuda.OutOfMemoryError:
            if batch==1:raise
            retries.append(batch);batch=max(1,batch//2)
        finally:
            del src,tgt,labels,logits
            if retries and device.type=='cuda':gc.collect();torch.cuda.empty_cache()
    return {'CE':total/tokens,'tokens':tokens,'seconds':time.perf_counter()-started,
            'forward':'FP16 evaluated weights on GPU; FP32 softmax/sum','representation':30,'comparable_to_raw_CE':False,
            'batch':batch,'OOM_retries':retries}

def evaluate(point,out,inputs,sets,refs,device,deadline,smoke=False,test_rows=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);inputs=Path(inputs)
    tok,table=R.tokenizer(inputs/'tokenizer.json',30);original=Tokenizer.from_file(str(inputs/'tokenizer.json'))
    result={'weights_sha256':sha(point),'splits':{},'restoration':'exact S3 compact30/mode-token code',
            'decoder':'exp27b_fast physical 4D greedy FP16','fresh_evaluated':0}
    model=load(point,device)
    try:
        for split,rows in sets.items():
            examples,timing=generate(model,rows,tok,table,device,deadline,smoke,batch=2 if smoke else 256)
            write(out/(split+'-generations.json'),examples)
            execution=execute(examples,smoke,deadline);write(out/(split+'-execution.json'),execution)
            score=N.compare_exec(refs[split],execution)
            if not smoke and test_rows is None:assert score['denominator']==(4096 if split=='heldout4k' else 991)
            result['splits'][split]={'exec':score,'CE':ce(model,rows,tok,device,deadline,smoke),
                                    'length_accuracy':E.length_accuracy(rows,original,score),'generation':timing,
                                    'restore_errors':sum(r['restore_error'] is not None for r in examples)}
            write(out/'partial-metrics.json',result)
    finally:
        del model;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    # Entire test timer is fresh: file read, weights/tokenizer load, generation/restore,
    # bounded execution and final answers close. No previous generations reused.
    C.sync(device);started=time.perf_counter()
    test=json.loads((inputs/'test.json').read_text())
    if test_rows is not None:test=test[:test_rows]
    model=load(point,device);tok,table=R.tokenizer(inputs/'tokenizer.json',30)
    try:
        examples,timing=generate(model,test,tok,table,device,deadline,smoke,batch=2 if smoke else 256)
        execution=execute(examples,smoke,deadline)
        answers=out/'test-answers.jsonl'
        with answers.open('w') as f:
            for row in examples:f.write(json.dumps({'id':row['id'],'question':row['question'],'code':row['generated_code']},ensure_ascii=False)+'\n')
        C.sync(device);elapsed=time.perf_counter()-started
        write(out/'test-generations.json',examples);write(out/'test-execution.json',execution)
        result['test']={'rows':len(test),'full_pipeline_seconds':elapsed,'generation':timing,
                        'execution':execution['summary'],'answers_sha256':sha(answers),'cache_reuse':False,
                        'timer':'test read + model/tokenizer load + FP16 4D greedy + restore + exec + answers close',
                        'accuracy':None}
        write(out/'metrics.json',result);return result
    finally:
        del model;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
