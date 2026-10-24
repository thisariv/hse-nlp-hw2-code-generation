"""Physical candidate cross-KV, 4D SDPA and fresh full pipeline timings."""
import gc,math,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import exp26_core as C
import exp26_eval as E
import exp26_exec as N
import exp27_decode as D
from exp26_common import read_json,write_json,json_hash,sha_file
from exp26_fast_exec import run_chunks

def config(k=0):
    return {'name':'greedy' if not k else f'vote-t0.7-k{k}-g0','mode':'vote' if k else 'greedy',
            'temperature':.7 if k else None,'k':k,'greedy_vote':False}

class Context:
    @torch.inference_mode()
    def __init__(self,model,src,capacity):
        self.model,self.src,self.capacity=model,src,capacity
        self.width=model.width;self.heads=model.transformer.decoder.layers[0].self_attn.num_heads;self.dim=self.width//self.heads
        memory=model.transformer.encoder(model.embed(src),src_key_padding_mask=src.eq(C.PAD))
        self.allowed=src.ne(C.PAD)[:,None,None,:];self.positions=model.positions(torch.arange(capacity,device=src.device));self.cross=[]
        for layer in model.transformer.decoder.layers:
            a=layer.multihead_attn;k,v=F.linear(memory,a.in_proj_weight[self.width:],a.in_proj_bias[self.width:]).chunk(2,-1)
            self.cross.append(tuple(t.reshape(len(src),-1,self.heads,self.dim).transpose(1,2).contiguous() for t in (k,v)))

class Decoder:
    @torch.inference_mode()
    def __init__(self,context,k):
        self.c,self.k,self.t=context,k,0;self.n=len(context.src)*k
        self.allowed=context.allowed.repeat_interleave(k,dim=0)
        self.cross=[tuple(t.repeat_interleave(k,dim=0).contiguous() for t in pair) for pair in context.cross]
        self.cross_bytes=sum(t.numel()*t.element_size() for pair in self.cross for t in pair)
        shape=(self.n,context.heads,context.capacity,context.dim)
        self.self_kv=[(context.positions.new_empty(shape),context.positions.new_empty(shape)) for _ in context.cross]
        self.valid=torch.zeros(self.n,context.capacity,dtype=torch.bool,device=context.src.device)
    def heads(self,x):return x.reshape(self.n,1,self.c.heads,self.c.dim).transpose(1,2)
    def attend(self,a,q,k,v,mask):
        assert q.ndim==k.ndim==v.ndim==4
        y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,is_causal=False)
        return a.out_proj(y.transpose(1,2).contiguous().reshape(self.n,1,self.c.width))
    @torch.inference_mode()
    def step(self,tokens):
        c=self.c;t=self.t;x=c.model.tokens(tokens[:,None])*math.sqrt(c.width)+c.positions[t][None,None,:]
        self.valid[:,t]=tokens.ne(C.PAD)
        for layer,(sk,sv),(ck,cv) in zip(c.model.transformer.decoder.layers,self.self_kv,self.cross):
            a=layer.self_attn;q,k,v=F.linear(layer.norm1(x),a.in_proj_weight,a.in_proj_bias).chunk(3,-1)
            sk[:,:,t:t+1]=self.heads(k);sv[:,:,t:t+1]=self.heads(v)
            x=x+layer.dropout1(self.attend(a,self.heads(q),sk[:,:,:t+1],sv[:,:,:t+1],self.valid[:,None,None,:t+1]))
            a=layer.multihead_attn;q=F.linear(layer.norm2(x),a.in_proj_weight[:c.width],a.in_proj_bias[:c.width])
            x=x+layer.dropout2(self.attend(a,self.heads(q),ck,cv,self.allowed))
            x=x+layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(x))))))
        if c.model.transformer.decoder.norm is not None:x=c.model.transformer.decoder.norm(x)
        self.t+=1;return c.model.head(x[:,0])

@torch.inference_mode()
def decode(context,k=1,temperature=None,deadline=None,check_every=16):
    d=Decoder(context,k);device=context.src.device
    token=torch.full((d.n,),C.BOS,device=device,dtype=torch.long);done=torch.zeros(d.n,dtype=torch.bool,device=device)
    output=torch.full((d.n,context.capacity),C.PAD,device=device,dtype=torch.long)
    sums=torch.zeros(d.n,device=device);lengths=torch.zeros(d.n,dtype=torch.long,device=device)
    for t in range(context.capacity):
        if t%check_every==0:C.check_deadline(deadline)
        logits=d.step(token).float();logits[:,[C.PAD,C.BOS]]=-torch.inf
        pick=logits.argmax(-1) if temperature is None else torch.multinomial(F.softmax(logits/temperature,-1),1).squeeze(-1)
        logprob=F.log_softmax(logits,-1).gather(-1,pick[:,None]).squeeze(-1)
        sums.add_(torch.where(done,0.,logprob));lengths.add_((~done).long())
        token=pick.masked_fill(done,C.PAD);output[:,t]=token;done|=token.eq(C.EOS)
        if (t+1)%check_every==0 and bool(done.all().cpu()):break
    return output[:,:d.t].reshape(len(context.src),k,d.t).cpu(),(sums/lengths.clamp_min(1)).reshape(-1,k).cpu(),d.cross_bytes

def pack(ids,lps,tokenbytes):
    result=D.candidates(ids,lps,tokenbytes)
    for group,seqs in zip(result,ids.tolist()):
        for c,seq in zip(group,seqs):c['token_ids']=C.active_tokens(seq)
    return result

def generate(model_path,rows,tokenizer_path,device,out,k=0,batch_rows=2048,cap=512,spec=None,deadline=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    identity={'model':sha_file(model_path),'rows':json_hash(rows),'tokenizer':sha_file(tokenizer_path),
              'k':k,'batch_rows':batch_rows,'cap':cap,'spec':spec,'decoder':'physical4D-v1','torch':str(torch.__version__)}
    if (out/'identity.json').exists():assert read_json(out/'identity.json')==identity
    else:write_json(out/'identity.json',identity)
    started=time.perf_counter();model=E.load_model(model_path,device,spec=spec).eval()
    if device.type=='cuda':model.half();torch.cuda.reset_peak_memory_stats(device)
    tokenizer,tokenbytes=C.init_tokenizer(tokenizer_path)
    encoded=[[C.BOS]+C.encode(tokenizer,r['question'])+[C.EOS] for r in rows]
    order=sorted(range(len(rows)),key=lambda i:(len(encoded[i]),i));qbatch=max(1,batch_rows//max(1,k)) if k else min(512,batch_rows);allrows=[None]*len(rows)
    load_seconds=time.perf_counter()-started;new=resumed=0;cross_bytes=0;decode_seconds=0.
    try:
        for lo in range(0,len(order),qbatch):
            C.check_deadline(deadline);indices=order[lo:lo+qbatch];file=out/f'batch-{lo:06d}.json'
            if file.exists():
                assert sha_file(file)==read_json(file.with_suffix('.sha.json'))['sha256'];part=read_json(file)
                assert [r['id'] for r in part]==indices;resumed+=1
            else:
                src=nn.utils.rnn.pad_sequence([torch.tensor(encoded[i]) for i in indices],batch_first=True,padding_value=C.PAD).to(device)
                C.sync(device);begin=time.perf_counter();ctx=Context(model,src,cap)
                ids,lps,_=decode(ctx,deadline=deadline);greedy=pack(ids,lps,tokenbytes)
                part=[{'id':i,'question':rows[i]['question'],'greedy':greedy[j][0],'samples':{}} for j,i in enumerate(indices)]
                if k:
                    seed=270042+lo+7000;torch.manual_seed(seed)
                    if device.type=='cuda':torch.cuda.manual_seed_all(seed)
                    ids,lps,memory=decode(ctx,k,.7,deadline);samples=pack(ids,lps,tokenbytes);cross_bytes=max(cross_bytes,memory)
                    for r,g in zip(part,samples):
                        for c in g:c.pop('token_ids') # greedy tokens retained for correctness gate
                        r['samples']['0.7']=g
                C.sync(device);decode_seconds+=time.perf_counter()-begin
                del ctx,src,ids,lps
                write_json(file,part);write_json(file.with_suffix('.sha.json'),{'sha256':sha_file(file)});new+=1
            for r in part:allrows[r['id']]=r
            if lo%(qbatch*8)==0:print('FAST SAMPLE',lo+len(indices),'/',len(rows),'k',k,'batch',batch_rows,flush=True)
        C.sync(device)
        return allrows,{'load_generate_write_seconds':time.perf_counter()-started,'load_seconds':load_seconds,'generation_seconds':decode_seconds,
            'batch_candidate_rows':batch_rows,'question_batch':qbatch,'new_batches':new,'resumed_batches':resumed,
            'cross_KV':'physical repeat_interleave; all SDPA4D','cross_KV_peak_bytes':cross_bytes,'check_every':16,
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None}
    finally:
        del model;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()

def execute(rows,out,deadline=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    flat=[c for r in rows for c in [r['greedy']]+r['samples'].get('0.7',[])];codes=[c['code'] for c in flat]
    identity={'codes':json_hash(codes),'wall':3.,'cpu':2.}
    if (out/'identity.json').exists():assert read_json(out/'identity.json')==identity
    else:write_json(out/'identity.json',identity)
    saved={}
    for p in out.glob('chunk-*.json'):
        assert sha_file(p)==read_json(p.with_suffix('.sha.json'))['sha256'];saved[int(p.stem.split('-')[1])]=read_json(p)
    def commit(i,value):
        p=out/f'chunk-{i:06d}.json';write_json(p,value);write_json(p.with_suffix('.sha.json'),{'sha256':sha_file(p)});C.check_deadline(deadline)
    result=run_chunks(codes,workers=2,chunk_rows=2000,wall_limit=3.,cpu_limit=2.,saved_chunks=saved,on_chunk=commit)
    for c,e in zip(flat,result['rows']):c['execution']=e
    result['summary'].update(timeouts=sum('timeout' in r['status'] for r in result['rows']),complete=True)
    result['summary']['timeout_limits']={kind:sum(r.get('timeout_limit')==kind for r in result['rows']) for kind in ('wall','cpu')}
    write_json(out/'summary.json',result['summary']);return result['summary']

def probe_batch(model_path,rows,tokenizer_path,device,out,k=16,spec=None,cap=512,choices=(2048,1024),deadline=None):
    out=Path(out);p=out/'batch-choice.json'
    if p.exists():return read_json(p)
    tok,_=C.init_tokenizer(tokenizer_path)
    longest=sorted(rows,key=lambda r:len(C.encode(tok,r['question'])),reverse=True);attempts=[]
    for size in choices:
        sample=longest[:max(1,size//max(1,k))]
        try:
            _,timing=generate(model_path,sample,tokenizer_path,device,out/f'probe-{size}',k,size,cap,spec,deadline)
            attempts.append({'batch':size,'status':'passed',**timing})
            report={'batch_rows':size,'attempts':attempts,'probe':'longest inputs, full cap; 2048 then OOM fallback1024'}
            write_json(p,report);return report
        except torch.cuda.OutOfMemoryError:
            gc.collect();torch.cuda.empty_cache();attempts.append({'batch':size,'status':'OOM'});write_json(out/'batch-attempts.json',attempts)
    raise RuntimeError('Sampler OOM at both candidate batches')

def time_test(model_path,test,tokenizer_path,device,out,cfg,spec=None,batch_rows=2048,cap=512,deadline=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);p=out/'timing.json'
    if p.exists():
        report=read_json(p);assert report['model_sha256']==sha_file(model_path) and report['config']==cfg
        assert report['answers_sha256']==sha_file(out/'test-answers.jsonl');return report
    # Every incomplete attempt is retained. Timing restarts with entirely fresh candidates.
    attempt=out/f'attempt-{len(list(out.glob("attempt-*"))):03d}';attempt.mkdir()
    C.sync(device);started=time.perf_counter();rows=read_json(test) if isinstance(test,(str,Path)) else test
    generated,gen=generate(model_path,rows,tokenizer_path,device,attempt/'candidates',cfg['k'],batch_rows,cap,spec,deadline)
    execution=execute(generated,attempt/'execution',deadline);begin=time.perf_counter()
    winners=[D.choose(r,cfg) for r in generated];vote_seconds=time.perf_counter()-begin;begin=time.perf_counter()
    with (out/'test-answers.jsonl').open('w',encoding='utf8') as f:
        for i,(row,c) in enumerate(zip(rows,winners)):f.write(__import__('json').dumps({'id':i,'question':row['question'],'code':c['code']},ensure_ascii=False)+'\n')
    C.sync(device);write_seconds=time.perf_counter()-begin;full=time.perf_counter()-started
    report={'config':cfg,'model_sha256':sha_file(model_path),'tokenizer_sha256':sha_file(tokenizer_path),'rows':len(rows),'complete':True,
        'answers_sha256':sha_file(out/'test-answers.jsonl'),'full_pipeline_seconds':full,'executed_candidates':len(rows)*(1+cfg['k']),
        'components':{'generation':gen,'execution':execution,'voting_seconds':vote_seconds,'answer_write_seconds':write_seconds},
        'timer':'test read, model/tokenizer load, generation/candidate journal, execution, voting, answers close; hashes/report outside',
        'attempt':attempt.name,'cache_reuse':False}
    write_json(p,report);return report

def select_k(timings):
    eligible=[r for r in timings if r['config']['k'] and r['complete'] and r['full_pipeline_seconds']<=120.]
    return max(eligible,key=lambda r:r['config']['k'])['config'] if eligible else config()

def run(model_path,sets,refs,inputs,device,out,control_out,deadline):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);summary=out/'27b-summary.json'
    if summary.exists() and read_json(summary).get('status')=='complete':return read_json(summary)
    report={'experiment':'27b','status':'partial','model_sha256':sha_file(model_path),'test':[]};write_json(summary,report)
    probe=probe_batch(model_path,sets['heldout4k'],Path(inputs)/'tokenizer.json',device,out/'memory',deadline=deadline)
    report['batch_probe']=probe;batch=probe['batch_rows']
    rows,timing=generate(model_path,sets['heldout4k'],Path(inputs)/'tokenizer.json',device,out/'heldout4k/candidates',16,batch,deadline=deadline)
    controls=read_json(Path(control_out)/'heldout4k-generations.json')
    mismatch=[i for i,(r,c) in enumerate(zip(rows,controls)) if r['greedy']['token_ids']!=c['generated_token_ids']]
    report['greedy_gate']={'token_identical':not mismatch,'mismatch_indices':mismatch,'rows':len(rows)}
    write_json(summary,report)
    if mismatch:raise AssertionError('Fast greedy token identity failed; see27b-summary')
    ex=execute(rows,out/'heldout4k/execution',deadline);wins=[D.choose(r,config(16)) for r in rows]
    score=N.compare_exec(refs['heldout4k'],[w['execution'] for w in wins]);matches=score['matches']
    report['heldout4k']={'generation':timing,'execution':ex,'exec':score,'quality_gate':abs(matches-2707)<=45,'expected_matches':2707,'tolerance':45}
    write_json(summary,report)
    if abs(matches-2707)>45:raise AssertionError('Fast k16 quality outside registered tolerance')
    for k in (0,8,16):
        C.check_deadline(deadline)
        timed=time_test(model_path,Path(inputs)/'test.json',Path(inputs)/'tokenizer.json',device,out/'test'/config(k)['name'],config(k),batch_rows=batch,deadline=deadline)
        report['test'].append(timed);write_json(summary,report)
    report.update(status='complete',decision={'config':select_k(report['test']),'rule':'largest k8/k16 with fresh fulltest<=120s',
        'voting_available':select_k(report['test'])['k']>0},fresh_encoded=0,fresh_evaluated=0)
    write_json(summary,report);return report
