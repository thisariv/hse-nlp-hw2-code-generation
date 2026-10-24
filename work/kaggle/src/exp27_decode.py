"""Sampling with shared encoder/cross KV, numeric voting, complete pipeline timers."""
import gc,json,math,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import exp26_core as C
import exp26_eval as E
import exp26_exec as N
from exp26_fast_exec import run_chunks
from exp26_common import read_json,write_json,json_hash,sha_file

TEMPERATURES={0.7:32,0.5:16,1.0:16}

def configurations():
    configs=[{'name':'greedy','mode':'greedy','k':0,'temperature':None,'greedy_vote':False}]
    for t,n in TEMPERATURES.items():
        for k in (4,8,16,32):
            if k>n: continue
            for g in (False,True):
                configs.append({'name':f'vote-t{t}-k{k}-g{int(g)}','mode':'vote','temperature':t,'k':k,'greedy_vote':g})
            if t==0.7:
                configs.append({'name':f'fallback-k{k}','mode':'fallback','temperature':t,'k':k,'greedy_vote':False})
    return configs

def choose(row,cfg):
    greedy=row['greedy']
    if cfg['mode']=='greedy': return greedy
    samples=row['samples'][str(cfg['temperature'])][:cfg['k']]
    if cfg['mode']=='fallback':
        return greedy if greedy['execution']['status']=='numeric' else next((s for s in samples if s['execution']['status']=='numeric'),greedy)
    votes=([greedy] if cfg['greedy_vote'] else [])+samples
    good=[v for v in votes if v['execution']['status']=='numeric']
    if not good:return greedy
    # isclose is not transitive: explicitly use connected components of its pairwise graph.
    parent=list(range(len(good)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(good)):
        for j in range(i):
            if math.isclose(good[i]['execution']['value'],good[j]['execution']['value'],rel_tol=1e-6,abs_tol=1e-6):
                parent[root(i)]=root(j)
    clusters={}
    for i,v in enumerate(good):clusters.setdefault(root(i),[]).append(v)
    def key(vs):
        contains=greedy['execution']['status']=='numeric' and any(math.isclose(v['execution']['value'],greedy['execution']['value'],rel_tol=1e-6,abs_tol=1e-6) for v in vs)
        return len(vs),contains,sum(v['mean_logprob'] for v in vs)/len(vs)
    winning=max(clusters.values(),key=key)
    return max(winning,key=lambda v:v['mean_logprob'])

class EncoderContext:
    @torch.inference_mode()
    def __init__(self,model,src,capacity):
        self.model,self.src,self.capacity=model,src,capacity
        self.width=model.width;self.heads=model.transformer.decoder.layers[0].self_attn.num_heads
        self.dim=self.width//self.heads
        memory=model.transformer.encoder(model.embed(src),src_key_padding_mask=src.eq(C.PAD))
        self.allowed=src.ne(C.PAD)[:,None,None,None,:]
        self.positions=model.positions(torch.arange(capacity,device=src.device))
        self.cross=[]
        for layer in model.transformer.decoder.layers:
            att=layer.multihead_attn
            k,v=F.linear(memory,att.in_proj_weight[self.width:],att.in_proj_bias[self.width:]).chunk(2,-1)
            shape=(len(src),-1,self.heads,self.dim)
            self.cross.append((k.reshape(shape).transpose(1,2)[:,None],v.reshape(shape).transpose(1,2)[:,None]))
        # [question,1,head,source,dim], stride0 broadcast to lanes at SDPA; no repeat/contiguous.

class SampleDecoder:
    def __init__(self,context,k):
        self.c=context;self.k=k;self.t=0;self.b=len(context.src)
        self.valid=torch.zeros(self.b,k,context.capacity,dtype=torch.bool,device=context.src.device)
        shape=(self.b,k,context.heads,context.capacity,context.dim)
        self.self_kv=[(context.positions.new_empty(shape),context.positions.new_empty(shape)) for _ in context.cross]
    def heads(self,x):return x.reshape(self.b,self.k,1,self.c.heads,self.c.dim).transpose(2,3)
    def attend(self,att,q,k,v,mask):
        # k=1 keeps the original4-D SDPA route; sampling uses5-D broadcasting.
        if self.k==1:
            y=F.scaled_dot_product_attention(q[:,0],k[:,0],v[:,0],attn_mask=mask[:,0],dropout_p=0.)[:,None]
        else:y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.)
        y=y.transpose(2,3).contiguous().reshape(self.b,self.k,1,self.c.width)
        return att.out_proj(y)
    @torch.inference_mode()
    def step(self,tokens):
        c=self.c;t=self.t
        x=c.model.tokens(tokens[:,:,None])*math.sqrt(c.width)+c.positions[t][None,None,None,:]
        self.valid[:,:,t]=tokens.ne(C.PAD)
        for layer,(sk,sv),(ck,cv) in zip(c.model.transformer.decoder.layers,self.self_kv,c.cross):
            a=layer.self_attn;q,k,v=F.linear(layer.norm1(x),a.in_proj_weight,a.in_proj_bias).chunk(3,-1)
            sk[:,:,:,t:t+1]=self.heads(k);sv[:,:,:,t:t+1]=self.heads(v)
            x=x+layer.dropout1(self.attend(a,self.heads(q),sk[:,:,:,:t+1],sv[:,:,:,:t+1],self.valid[:,:,None,None,:t+1]))
            a=layer.multihead_attn;q=F.linear(layer.norm2(x),a.in_proj_weight[:c.width],a.in_proj_bias[:c.width])
            x=x+layer.dropout2(self.attend(a,self.heads(q),ck,cv,c.allowed))
            x=x+layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm3(x))))))
        if c.model.transformer.decoder.norm is not None:x=c.model.transformer.decoder.norm(x)
        self.t+=1;return c.model.head(x[:,:,0])

@torch.inference_mode()
def decode_context(context,k,temperature,deadline=None,forced=None):
    d=SampleDecoder(context,k);device=context.src.device
    token=torch.full((d.b,k),C.BOS,device=device,dtype=torch.long);done=torch.zeros_like(token,dtype=torch.bool)
    out=torch.zeros(d.b,k,context.capacity,device=device,dtype=torch.long)
    sums=torch.zeros(d.b,k,device=device);lengths=torch.zeros(d.b,k,device=device,dtype=torch.long)
    for t in range(context.capacity):
        if t%16==0:C.check_deadline(deadline)
        logits=d.step(token).float();logits[:,:,[C.PAD,C.BOS]]=-torch.inf
        if forced is not None:pick=forced[:,:,t]
        elif temperature is None:pick=logits.argmax(-1)
        else:pick=torch.multinomial(F.softmax(logits.reshape(-1,logits.shape[-1])/temperature,-1),1).reshape(d.b,k)
        logprob=F.log_softmax(logits,-1).gather(-1,pick[:,:,None]).squeeze(-1)
        sums+=torch.where(done,0.,logprob);lengths+=(~done).long()
        token=pick.masked_fill(done,C.PAD);out[:,:,t]=token;done|=token.eq(C.EOS)
        if (t+1)%16==0 and bool(done.all()):break
    return out[:,:,:d.t].cpu(),(sums/lengths.clamp_min(1)).cpu()

def candidates(ids,logprobs,tokenbytes):
    return [[{'code':C.decode_bytes(tokenbytes,C.active_tokens(seq)).decode('utf8',errors='replace'),
              'length':len(C.active_tokens(seq)), 'mean_logprob':float(lp)} for seq,lp in zip(group,lps)]
            for group,lps in zip(ids.tolist(),logprobs.tolist())]

def generate_grid(model_path,rows,tokenizer_path,device,out,spec=None,grid=None,batch_rows=512,cap=512,deadline=None,controls=None):
    """One encoder pass/question for greedy + all temperatures; completed question batches resume."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True);grid=grid or TEMPERATURES
    identity={'model':sha_file(model_path),'tokenizer':sha_file(tokenizer_path),'rows':json_hash(rows),
              'grid':{str(k):v for k,v in grid.items()},'batch_rows':batch_rows,'cap':cap,'spec':spec,
              'controls':json_hash(controls) if controls is not None else None,'runtime':str(torch.__version__)}
    meta=out/'candidate-identity.json'
    if meta.exists():assert read_json(meta)==identity,'Sampling input/config changed'
    else:write_json(meta,identity)
    start=time.perf_counter();model=E.load_model(model_path,device,spec=spec).eval()
    if device.type=='cuda':model.half();torch.cuda.reset_peak_memory_stats(device)
    tokenizer,tokenbytes=C.init_tokenizer(tokenizer_path)
    encoded=[[C.BOS]+C.encode(tokenizer,r['question'])+[C.EOS] for r in rows]
    order=sorted(range(len(rows)),key=lambda i:(len(encoded[i]),i));qbatch=max(1,batch_rows//max(grid.values()))
    allrows=[None]*len(rows);new_batches=0;resumed_batches=0
    try:
        for lo in range(0,len(order),qbatch):
            C.check_deadline(deadline);indices=order[lo:lo+qbatch];file=out/f'batch-{lo:06d}.json'
            if file.exists():
                part=read_json(file);assert [r['id'] for r in part]==indices
                receipt=read_json(file.with_suffix('.sha.json'));assert sha_file(file)==receipt['sha256']
                resumed_batches+=1
            else:
                src=nn.utils.rnn.pad_sequence([torch.tensor(encoded[i]) for i in indices],batch_first=True,padding_value=C.PAD).to(device)
                ctx=EncoderContext(model,src,cap)
                forced=None
                if controls is not None:
                    forced=torch.zeros(len(indices),1,cap,dtype=torch.long,device=device)
                    for j,i in enumerate(indices):
                        active=controls[i]['generated_token_ids'][:cap];forced[j,0,:len(active)]=torch.tensor(active,device=device)
                gids,glp=decode_context(ctx,1,None,deadline,forced=forced);greedy=candidates(gids,glp,tokenbytes)
                part=[{'id':i,'question':rows[i]['question'],'greedy':greedy[j][0],'samples':{}} for j,i in enumerate(indices)]
                if controls is not None:
                    for r in part:assert r['greedy']['code']==controls[r['id']]['generated_code'],'Forced greedy replay differs'
                for t,k in grid.items():
                    torch.manual_seed(270042+lo+int(t*10000))
                    if device.type=='cuda':torch.cuda.manual_seed_all(270042+lo+int(t*10000))
                    ids,lps=decode_context(ctx,k,t,deadline);generated=candidates(ids,lps,tokenbytes)
                    for r,g in zip(part,generated):r['samples'][str(t)]=g
                    del ids,lps
                del ctx,src,gids,glp
                write_json(file,part);write_json(file.with_suffix('.sha.json'),{'sha256':sha_file(file)})
                new_batches+=1
            for r in part:allrows[r['id']]=r
            if lo%(qbatch*16)==0:print('SAMPLING',len([r for r in allrows if r is not None]),'/',len(rows),'seconds',round(time.perf_counter()-start,1),flush=True)
        C.sync(device)
        return allrows,{'load_generate_seconds':time.perf_counter()-start,'new_batches':new_batches,'resumed_batches':resumed_batches,
                       'batch_candidate_rows':batch_rows,'question_batch':qbatch,'cross_KV':'shared5D broadcast; no physical expansion',
                       'peak_reserved_bytes':torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None}
    finally:
        del model;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()

def execute_candidates(rows,out,deadline=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    flat=[c for r in rows for c in [r['greedy']]+[c for group in r['samples'].values() for c in group]]
    codes=[c['code'] for c in flat];identity=json_hash(codes);path=out/'identity.json'
    if path.exists():assert read_json(path)['codes_sha256']==identity
    else:write_json(path,{'codes_sha256':identity,'candidate_count':len(codes)})
    saved={}
    for p in out.glob('chunk-*.json'):
        saved[int(p.stem.split('-')[1])]=read_json(p)
        assert sha_file(p)==read_json(p.with_suffix('.sha'))['sha256']
    def commit(i,value):
        p=out/f'chunk-{i:06d}.json';write_json(p,value);write_json(p.with_suffix('.sha'),{'sha256':sha_file(p)})
        C.check_deadline(deadline)
    results=run_chunks(codes,workers=2,chunk_rows=2000,saved_chunks=saved,on_chunk=commit)
    for c,e in zip(flat,results['rows']):c['execution']=e
    extra={'candidate_count':len(flat),'reused_candidates':sum(len(v) for v in saved.values()),
           'committed_distinct_program_calls':sum(len(set(codes[i:i+2000])) for i in range(0,len(codes),2000)),
           'new_committed_distinct_program_calls':sum(len(set(codes[i:i+2000])) for i in range(0,len(codes),2000) if i//2000 not in saved),
           'failed_chunk_extra_calls_unknown':results['summary']['chunk_retries']>0}
    results['summary'].update(extra);write_json(out/'summary.json',results['summary'])
    return results['summary']

def score_configs(rows,reference,original,tokenizer,greedy_score,configs=None):
    result=[]
    for cfg in configs or configurations():
        winners=[choose(r,cfg) for r in rows];score=N.compare_exec(reference,[w['execution'] for w in winners])
        result.append({'config':cfg,'exec':score,'McNemar_against_greedy':E.paired_mcnemar(score,greedy_score),
                       'length_accuracy':E.length_accuracy(original,tokenizer,score),
                       'executed_candidates_used':len(rows)*(1+cfg['k']),
                       'mean_winning_length':sum(w['length'] for w in winners)/len(winners)})
    return result

def time_test(model_path,test,tokenizer_path,device,out,cfg,spec=None,batch_rows=512,cap=512,deadline=None):
    """No cache reuse. Timer includes input read, checkpoint/tokenizer load, generation, exec, vote, answers."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if list((out/'candidates').glob('batch-*.json')):raise ValueError('Fresh timing requires empty test candidates')
    C.sync(device);start=time.perf_counter()
    test=read_json(test) if isinstance(test,(Path,str)) else test
    if cfg['mode']=='greedy':
        gens,timing=E.generate_one(model_path,test,tokenizer_path,device,out,'test',spec=spec,batch=512,cap=cap,half=device.type=='cuda')
        programs=[g['generated_code'] for g in gens];count=0
    else:
        grid={cfg['temperature']:cfg['k']}
        generated,gen_timing=generate_grid(model_path,test,tokenizer_path,device,out/'candidates',spec,grid,batch_rows,cap,deadline)
        execution=execute_candidates(generated,out/'execution',deadline)
        programs=[choose(r,cfg)['code'] for r in generated];count=len(test)*(1+cfg['k'])
        timing=dict(gen_timing,execution=execution)
    path=out/'test-answers.jsonl'
    with path.open('w') as f:
        for i,(row,code) in enumerate(zip(test,programs)):
            f.write(json.dumps({'id':i,'question':row['question'],'code':code},ensure_ascii=False)+'\n')
    C.sync(device);full=time.perf_counter()-start
    report={'config':cfg,'full_pipeline_seconds':full,'rows':len(test),'executed_candidates':count,
            'answers_sha256':sha_file(path),'model_sha256':sha_file(model_path),'tokenizer_sha256':sha_file(tokenizer_path),'components':timing,'complete':True,
            'timer':'test read, checkpoint/tokenizer load, encoder/decode, candidate execution, voting, final answer write; SHA/report outside'}
    write_json(out/'timing.json',report);return report

def select_config(tested):
    eligible=[r for r in tested[1:] if r['timing']['full_pipeline_seconds']<=120 and r['McNemar_against_greedy']['delta_percentage_points']>=1.5]
    chosen=max(eligible,key=lambda r:(r['exec']['matches'],-r['timing']['full_pipeline_seconds'])) if eligible else tested[0]
    return bool(eligible),chosen

def run_experiment(model_path,sets,refs,inputs,device,out,control_metrics,control_out,spec=None,deadline=None,batch_rows=512):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);tokenizer,_=C.init_tokenizer(Path(inputs)/'tokenizer.json');splits={}
    start=time.perf_counter()
    for split,original in sets.items():
        controls=read_json(Path(control_out)/(split+'-generations.json'))
        generated,timing=generate_grid(model_path,original,Path(inputs)/'tokenizer.json',device,out/split/'candidates',spec,
                                       batch_rows=batch_rows,deadline=deadline,controls=controls)
        execution=execute_candidates(generated,out/split/'execution',deadline)
        configs=score_configs(generated,refs[split],original,tokenizer,control_metrics['splits'][split]['exec'])
        assert configs[0]['exec']['match_indices']==control_metrics['splits'][split]['exec']['match_indices'],'Greedy execution replay gate'
        splits[split]={'configs':configs,'generation':timing,'execution':execution,'candidate_count':len(original)*65}
        write_json(out/'progress.json',{'experiment':27,'status':'partial','splits':splits})
    primary=splits['heldout4k']['configs'];ranked=sorted(primary[1:],key=lambda r:(-r['exec']['matches'],r['config']['k'],r['config']['name']))[:3]
    tested=[]
    # Mandatory greedy baseline plus2–3 best alternatives; each gets a fresh full timer.
    for row in [primary[0]]+ranked:
        C.check_deadline(deadline);cfg=row['config'];dest=out/'test'/cfg['name']
        p=dest/'timing.json'
        timing=read_json(p) if p.exists() else time_test(model_path,Path(inputs)/'test.json',Path(inputs)/'tokenizer.json',device,dest,cfg,spec,batch_rows,deadline=deadline)
        tested.append(dict(row,timing=timing));write_json(out/'timed-configs.json',tested)
    accepted,chosen=select_config(tested)
    report={'experiment':27,'status':'complete','model_sha256':sha_file(model_path),'splits':splits,'test':tested,
            'decision':{'accepted':accepted,'config':chosen['config'],'delta_pp':chosen['McNemar_against_greedy']['delta_percentage_points'],
                        'full_test_seconds':chosen['timing']['full_pipeline_seconds'],'selection':'max exec4k among top3 measured alternatives with fulltest<=120s and gain>=1.5pp'},
            'wall_seconds':time.perf_counter()-start,'logprob':'mean untempered model logprob including EOS, PAD/BOS masked',
            'clustering':'pairwise isclose1e-6 connected components; tie count, greedy membership, mean candidate logprob',
            'fresh_encoded':0,'fresh_evaluated':0}
    write_json(out/'27-summary.json',report);return report
