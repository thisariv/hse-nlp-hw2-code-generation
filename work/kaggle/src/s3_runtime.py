"""Fixed-exposure 26b trajectory, new token representations, optional equal-time side branches."""
import copy, gc, math, multiprocessing, time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch import nn
import exp26_core as C, exp26_train as T, exp26_eval as E, exp26_exec as N, exp26_protocol as P
from exp26_common import read_json, write_json, sha_file, json_hash
import s3_representation as R

class PrefixCache:
    """Same original order; transform only required rows, outside all training timers."""
    def __init__(self, original, order, tokpath, variant, scratch):
        self.original,self.original_order,self.tokpath,self.variant=original,order,tokpath,variant
        self.root=Path(scratch);self.root.mkdir(parents=True,exist_ok=True)
        _,self.oldbytes=C.init_tokenizer(tokpath)
        self.parts=[];self.starts=[0];self.preparation_seconds=0.
        self.manifest_sha=original.manifest_sha
    def __len__(self):return self.starts[-1]
    def __getitem__(self,index):
        k=int(np.searchsorted(self.starts,int(index),side='right')-1);j=int(index)-self.starts[k]
        s,t,so,to=self.parts[k]
        return s[so[j]:so[j+1]],t[to[j]:to[j+1]]
    def prepare(self,end,deadline,workers=2):
        end=min(end,len(self.original_order));start=time.monotonic();pending=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=R.init_encoder,initargs=(str(self.tokpath),self.variant)) as pool:
            cursor=len(self)
            while cursor<end or pending:
                C.check_deadline(deadline)
                while cursor<end and len(pending)<workers*2:
                    # Four mmap files per block;64k keeps even the full7M cache below normal fd limits.
                    hi=min(cursor+65536,end);rows=[]
                    for i in self.original_order[cursor:hi]:
                        s,t=self.original[int(i)]
                        rows.append((C.decode_bytes(self.oldbytes,s).decode(),C.decode_bytes(self.oldbytes,t).decode()))
                    pending.append((hi,pool.submit(R.encode_block,rows)));cursor=hi
                hi,future=pending.pop(0)
                # Bounded wait; watchdog still owns the entire process group, including children.
                arrays=future.result(timeout=max(.1,min(300.,deadline-time.monotonic())))
                folder=self.root/f'{len(self.parts):04d}';folder.mkdir()
                loaded=[]
                for name,a in zip(('source','target','source-offsets','target-offsets'),arrays):
                    np.save(folder/(name+'.npy'),a,allow_pickle=False)
                    loaded.append(np.load(folder/(name+'.npy'),mmap_mode='r',allow_pickle=False))
                self.parts.append(loaded);self.starts.append(hi)
                if len(self.parts)%2==0 or hi==end:print('PREP',self.variant,hi,'seconds',round(time.monotonic()-start,1),flush=True)
        self.preparation_seconds+=time.monotonic()-start

def initialize(inputs, device, variant, smoke=False, base_spec=None):
    cfg=copy.deepcopy(P.TRAIN_PROTOCOL);cfg['experiment']=variant
    tok,_=R.tokenizer(Path(inputs)/'tokenizer.json',variant)
    base_spec=base_spec or C.MODEL_SPEC
    base,opt,scaler,state,meta=T.initialize(Path(inputs)/'initial.pt',device,cfg,base_spec,smoke)
    original=T.cpu(base.state_dict());del base,opt
    spec=dict(base_spec,vocab=tok.get_vocab_size()+3)
    model=C.SimpleTransformer(**spec,dropout=0.).to(device)
    state_dict=model.state_dict()
    with torch.no_grad():
        for name,value in original.items():
            if name in ('tokens.weight','head.weight','head.bias'):state_dict[name][:8192].copy_(value.to(device))
            else:state_dict[name].copy_(value.to(device))
        generator=torch.Generator(device='cpu').manual_seed(P.SEED)
        model.tokens.weight[8192:].copy_(torch.randn((spec['vocab']-8192,model.width),generator=generator).to(device)/math.sqrt(model.width))
    assert all(torch.equal(model.state_dict()[k][:8192].cpu() if k in ('tokens.weight','head.weight','head.bias')
                           else model.state_dict()[k].cpu(),v) for k,v in original.items())
    # Identical seed at first training update; enlarged model construction is outside the training RNG trajectory.
    import random
    random.seed(P.SEED);np.random.seed(P.SEED);torch.manual_seed(P.SEED)
    if device.type=='cuda':torch.cuda.manual_seed_all(P.SEED)
    model.train();opt=torch.optim.AdamW(model.parameters(),lr=cfg['peak_lr'],weight_decay=cfg['weight_decay'],**({'fused':True} if not smoke else {}))
    return model,opt,scaler,state,meta,cfg,spec

def payload(model,opt,scaler,state,meta,cfg,spec):
    return {'format':'nlp-hw02-wsd-v1','model':T.cpu(model.state_dict()),'optimizer':T.cpu(opt.state_dict()),
            'scaler':scaler.state_dict(),'scaler_enabled':scaler.is_enabled(),'training':copy.deepcopy(state),
            'experiment_history':copy.deepcopy(meta),'rng':T.cpu(T.rng_state()),'scheduler':T.schedule_state(state,cfg),
            'config':cfg,'model_spec':spec}

def restore_payload(p,model,opt,scaler,state):
    model.load_state_dict(p['model']);opt.load_state_dict(p['optimizer']);scaler.load_state_dict(p['scaler'])
    state.clear();state.update(copy.deepcopy(p['training']));T.restore_rng(p['rng'],next(model.parameters()).device)
    model.train()

def save_latest(root,model,opt,scaler,state,meta,cfg,spec,identity):
    path=Path(root)/'checkpoints/latest.pt';path.parent.mkdir(parents=True,exist_ok=True)
    p=payload(model,opt,scaler,state,meta,cfg,spec);p['identity']=identity
    tmp=path.with_suffix('.pt.tmp');torch.save(p,tmp);tmp.replace(path)
    write_json(Path(root)/'latest.json',{'file':'checkpoints/latest.pt','sha256':sha_file(path),'bytes':path.stat().st_size,
        'main_updates':state['main_updates'],'branch_updates':state['branch_updates'],'data_cursor':state['data_cursor'],
        'phase':state['phase'],'model_tensor_sha256':E.state_digest(p['model'])})

@torch.no_grad()
def evaluate(model,sets,refs,base_tokenizer_path,tok,tokenbytes,variant,device,out,baseline,deadline,smoke=False):
    # This model resumes training. FP32->FP16->FP32 conversions under inference_mode
    # make parameter storage inference tensors, which the next backward cannot save.
    out=Path(out);out.mkdir(parents=True,exist_ok=True);model.eval()
    original_tok,_=C.init_tokenizer(base_tokenizer_path)
    original_weights=T.cpu(model.state_dict())
    result={'variant':variant,'CE_comparable_to_26b':False,'CE_note':'own representation/tokenization; do not compare CE numerically to raw 26b', 'splits':{}}
    for split,rows in sets.items():
        C.check_deadline(deadline)
        represented=[R.transform(r['question'],r['code'],variant) for r in rows]
        encoded=[[C.BOS]+C.encode(tok,q)+[C.EOS] for q,c in represented]
        order=sorted(range(len(rows)),key=lambda i:(len(encoded[i]),i));examples=[None]*len(rows)
        model.half() if device.type=='cuda' else model.float();C.sync(device);started=time.perf_counter()
        for lo in range(0,len(order),2 if smoke else 512):
            indices=order[lo:lo+(2 if smoke else 512)]
            src=nn.utils.rnn.pad_sequence([torch.tensor(encoded[i]) for i in indices],batch_first=True,padding_value=C.PAD).to(device)
            tokens,_=C.generate_preallocated(model,src,cap=8 if smoke else 512,deadline=deadline)
            for i,ids in zip(indices,tokens.cpu().tolist()):
                active=C.active_tokens(ids);text=C.decode_bytes(tokenbytes,active).decode('utf-8',errors='replace')
                try:code=R.restore(text,rows[i]['question'],variant);error=None
                except (ValueError,IndexError):code='';error='invalid representation'
                examples[i]={'id':i,'question':rows[i]['question'],'generated_token_ids':active,'represented_code':text,
                             'generated_code':code,'restore_error':error,'eos_emitted':C.EOS in active}
        C.sync(device);generation_seconds=time.perf_counter()-started
        write_json(out/(split+'-generations.json'),examples)
        from exp26_fast_exec import run_chunks
        execution=run_chunks([g['generated_code'] for g in examples],workers=2,wall_limit=3.,cpu_limit=2.)
        write_json(out/(split+'-execution.json'),execution)
        score=N.compare_exec(refs[split],execution)
        if not smoke:assert score['denominator']==(4096 if split=='heldout4k' else 991)
        model.float();model.load_state_dict(original_weights);total=torch.zeros((),device=device);denominator=0
        cached=[(s,C.encode(tok,c)) for s,(q,c) in zip(encoded,represented)]
        ce_order=sorted(range(len(rows)),key=lambda i:len(cached[i][0])+len(cached[i][1]))
        for lo in range(0,len(rows),2 if smoke else 32):
            C.check_deadline(deadline)
            src,tgt,labels,counts=C.collate_cache(cached,ce_order[lo:lo+(2 if smoke else 32)],device)
            logits=model(src,tgt);total+=nn.functional.cross_entropy(logits.reshape(-1,model.vocab),labels.reshape(-1),ignore_index=C.PAD,reduction='sum')
            denominator+=counts['target_tokens']
        result['splits'][split]={'exec':score,'CE':{'CE':float(total.cpu())/denominator,'tokens':denominator,'precision':'FP32','comparable_to_26b':False},
            'length_accuracy':E.length_accuracy(rows,original_tok,score),'reference_length_basis':'original tokenizer22/raw reference',
            'McNemar_against_26b':E.paired_mcnemar(score,baseline['splits'][split]['exec']),
            'generation_seconds':generation_seconds,'mean_represented_length':sum(len(g['generated_token_ids']) for g in examples)/len(rows)}
        write_json(out/'metrics.json',result)
    model.float().train();return result

def train(model,opt,scaler,state,meta,cfg,cache,device,until,deadline,root,spec,identity):
    order=np.arange(len(cache),dtype=np.uint32)
    field='branch_updates' if state['phase']=='branch' else 'main_updates'
    while state[field]<until:
        C.check_deadline(deadline)
        log=T.train_chunk(model,opt,scaler,state,meta,cfg,cache,order,device,min(25,until-state[field]))
        if meta['compute_updates']%500==0:
            write_json(Path(root)/'progress.json',dict(log,train_seconds=meta['train_seconds'],data_cursor=state['data_cursor'],status='partial'))
            print('TRAIN',cfg['experiment'],state['phase'],state['main_updates'],state['branch_updates'],round(meta['train_seconds'],2),flush=True)
        if meta['compute_updates']%2000==0:save_latest(root,model,opt,scaler,state,meta,cfg,spec,identity)

def run(inputs,cache,original_order,order_meta,sets,refs,baselines,variant,device,root,scratch,deadline,port_sha,smoke=False,base_spec=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    model,opt,scaler,state,meta,cfg,spec=initialize(inputs,device,variant,smoke,base_spec)
    tok,tokenbytes=R.tokenizer(Path(inputs)/'tokenizer.json',variant);tok.save(str(root/'tokenizer.json'))
    repcache=PrefixCache(cache,original_order,Path(inputs)/'tokenizer.json',variant,Path(scratch)/'tokens')
    fixed=[(6250,1250),(12500,2500)] if not smoke else [(2,1),(4,2)]
    if smoke:cfg.update(batch=2,warmup_updates=1)
    identity={'code_sha256':port_sha,'original_cache_manifest_sha256':cache.manifest_sha,'order_sha256':order_meta['file']['sha256'],
              'original_tokenizer_sha256':sha_file(Path(inputs)/'tokenizer.json'),'tokenizer_sha256':sha_file(root/'tokenizer.json'),
              'representation':variant,'seed':42,'protocol':cfg,'smoke':smoke}
    summary={'experiment':variant,'status':'partial','points':{},'missing_points':[f'p{o:06d}' for o,n in fixed],
             'equal_time':{},'identity':identity,'model_spec':spec,'train_recipe':'26b exact batches/order/LR/AMP/fused; expanded special vocabulary only',
             'acceptance':'heldout +1pp and old >=-1pp at1.92M;30 additionally equal-time +1pp; otherwise neutral/reject'}
    write_json(root/f'{variant}-summary.json',summary)
    repcache.prepare(cfg['batch']*sum(fixed[-1]),deadline,workers=1 if smoke else 2)
    write_json(root/'representation-preparation.json',{'rows':len(repcache),'seconds':repcache.preparation_seconds,'included_in_training_timer':False,
               'prefix_original_order_sha256':__import__('hashlib').sha256(np.asarray(original_order[:len(repcache)],dtype=np.uint32).tobytes()).hexdigest()})
    anchors=[]
    try:
        for origin,length in fixed:
            tag=f'p{origin:06d}'
            train(model,opt,scaler,state,meta,cfg,repcache,device,origin,deadline,root,spec,identity)
            anchor=payload(model,opt,scaler,state,meta,cfg,spec)
            anchor_path=Path(scratch)/(tag+'-anchor.pt');torch.save(anchor,anchor_path);del anchor
            anchors.append((origin,length,anchor_path))
            state.update(phase='branch',branch_origin=origin,branch_updates=0,branch_length=length)
            train(model,opt,scaler,state,meta,cfg,repcache,device,length,deadline,root,spec,identity)
            weights=root/'weights'/(tag+'.pt');weights.parent.mkdir(exist_ok=True)
            torch.save({'format':'nlp-hw02-model-only-v1','model':T.cpu(model.state_dict()),'model_spec':spec,'identity':identity},weights)
            training={'main_updates':origin,'cooldown_updates':length,'actual_pairs':state['data_cursor'],
                'train_seconds_total':meta['train_seconds'],'compute_updates_total':meta['compute_updates'],
                'pairs_per_second_all_compute':meta['all_counts']['pairs']/meta['train_seconds'],'AMP_skips':meta['AMP_skips'],
                'counts':copy.deepcopy(meta['all_counts']),'weights_sha256':sha_file(weights)}
            write_json(root/'points'/tag/'training.json',training)
            metrics=evaluate(model,sets,refs,Path(inputs)/'tokenizer.json',tok,tokenbytes,variant,device,
                             root/'points'/tag/'evaluation',baselines[tag],deadline,smoke)
            summary['points'][tag]=dict(training,metrics=metrics);summary['missing_points'].remove(tag)
            write_json(root/f'{variant}-summary.json',summary)
            anchor=torch.load(anchor_path,map_location='cpu',weights_only=True)
            history=copy.deepcopy(meta);restore_payload(anchor,model,opt,scaler,state);meta=history;del anchor
            save_latest(root,model,opt,scaler,state,meta,cfg,spec,identity)
        # Extra branches do not enter fixed-exposure compute history or change its trajectory.
        # For compact targets, independently replay each stable anchor to the control's train budget.
        if variant==30 and not smoke:
            fixed_history=copy.deepcopy(meta)
            for origin,length,path in anchors:
                tag=f'p{origin:06d}';target=baselines[tag]['train_seconds_total']
                p=torch.load(path,map_location='cpu',weights_only=True)
                point_seconds=summary['points'][tag]['train_seconds_total']
                if abs(point_seconds-target)<=max(5.,.02*target):
                    summary['equal_time'][tag]={'status':'aligned fixed point','seconds':point_seconds,'target_seconds':target,'metrics':summary['points'][tag]['metrics']}
                    continue
                if point_seconds>target:
                    summary['equal_time'][tag]={'status':'missing; fixed point exceeds control time; no interpolation','target_seconds':target}
                    continue
                restore_payload(p,model,opt,scaler,state);meta=copy.deepcopy(p['experiment_history']);del p
                step_seconds=point_seconds/summary['points'][tag]['compute_updates_total']
                remaining=(target-meta['train_seconds'])/step_seconds
                extra=max(0,int((remaining-origin*.2)/1.2));new_origin=origin+extra;new_length=max(1,round(new_origin*.2))
                repcache.prepare((new_origin+new_length)*cfg['batch'],deadline)
                train(model,opt,scaler,state,meta,cfg,repcache,device,new_origin,deadline,root,spec,identity)
                state.update(phase='branch',branch_origin=new_origin,branch_updates=0,branch_length=new_length)
                train(model,opt,scaler,state,meta,cfg,repcache,device,new_length,deadline,root,spec,identity)
                actual=meta['train_seconds'];aligned=abs(actual-target)<=max(5.,.02*target)
                weights=root/'weights'/('time-'+tag+'.pt')
                torch.save({'format':'nlp-hw02-model-only-v1','model':T.cpu(model.state_dict()),'model_spec':spec,'identity':identity},weights)
                # Measure even a misaligned receipt, and label it rather than inventing equal-time accuracy.
                metrics=evaluate(model,sets,refs,Path(inputs)/'tokenizer.json',tok,tokenbytes,variant,device,
                    root/'equal-time'/tag/'evaluation',baselines[tag],deadline,False)
                summary['equal_time'][tag]={'status':'aligned' if aligned else 'misaligned; comparison pending',
                    'seconds':actual,'target_seconds':target,'main_updates':new_origin,'cooldown_updates':new_length,
                    'actual_pairs':state['data_cursor'],'metrics':metrics,'tolerance_seconds':max(5.,.02*target),'weights_sha256':sha_file(weights)}
                write_json(root/f'{variant}-summary.json',summary)
            meta=fixed_history
        if not smoke:
            final=summary['points']['p012500']['metrics']['splits']
            fixed_ok=final['heldout4k']['exec']['matches']>=2251 and final['old1024']['exec']['matches']>=528
            timepoint=summary['equal_time'].get('p012500',{})
            time_aligned=timepoint.get('status') in ('aligned','aligned fixed point')
            time_ok=time_aligned and timepoint['metrics']['splits']['heldout4k']['exec']['matches']>=2251 and timepoint['metrics']['splits']['old1024']['exec']['matches']>=528
            summary['decision']={'fixed_exposure_gate':fixed_ok,'equal_time_gate':time_ok if variant==30 else 'not required',
                'status':'neutral_or_reject' if not fixed_ok else 'accepted' if variant==31 or time_ok else 'pending_equal_time' if not time_aligned else 'neutral_or_reject'}
        summary['status']='complete';write_json(root/f'{variant}-summary.json',summary)
        return summary
    finally:
        del model,opt,scaler;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
