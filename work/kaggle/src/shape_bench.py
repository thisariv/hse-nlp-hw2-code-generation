"""One offline T4 run. Dummy optimizer updates are timings, not model training.

Frozen v6 decoder is used verbatim; only EOS logits are replaced by a shared
length schedule. Every shape sees identical padding and forced completion times.
"""
import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import final_core as C
import final_fast as Fast
import final_speed as Speed
from shape_specs import SHAPES, name, spec, parameter_formula


def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def clean():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def peak():
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else {}


def reset_peak():
    sync()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()


def tokens(lengths, device):
    n = len(lengths); width = max(lengths)
    x = torch.randint(3,8195,(n,width),device=device)
    mask = torch.arange(width,device=device)[None,:] >= torch.tensor(lengths,device=device)[:,None]
    return x.masked_fill(mask,0)


def batch(lengths, device):
    src = tokens([a for a,b in lengths],device)
    labels = tokens([b for a,b in lengths],device)
    for i,(_,b) in enumerate(lengths): labels[i,b-1] = C.EOS
    tgt = torch.zeros_like(labels); tgt[:,0] = C.BOS; tgt[:,1:] = labels[:,:-1]
    return src,tgt,labels


def parameter_counts(model):
    out = {k:sum(p.numel() for p in module.parameters()) for k,module in
           [('encoder',model.transformer.encoder),('decoder',model.transformer.decoder),
            ('head',model.head)]}
    out['embeddings'] = model.tokens.weight.numel()+model.positions.weight.numel()
    out['total'] = sum(p.numel() for p in model.parameters())
    assert out['total'] == sum(out[k] for k in ('encoder','decoder','head','embeddings'))
    return out


def update(model, opt, scaler, lengths, micro, device, deadline=None):
    opt.zero_grad(set_to_none=True)
    denom = sum(b for a,b in lengths)
    for lo in range(0,len(lengths),micro):
        C.check_deadline(deadline)
        src,tgt,labels = batch(lengths[lo:lo+micro],device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            logits = model(src,tgt)
            loss = F.cross_entropy(logits.flatten(0,1), labels.flatten(),ignore_index=0,reduction='sum')/denom
        scaler.scale(loss).backward()
        del src,tgt,labels,logits,loss
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    # A random-init overflow must not make an artificially cheap Adam timing.
    if not bool(torch.isfinite(norm)): raise FloatingPointError('Nonfinite benchmark gradient')
    scaler.step(opt); scaler.update()


def train_bench(model, data, device, deadline):
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    attempts = []; micro = None
    for candidate in (64,32,16,8,4,2,1):
        C.check_deadline(deadline)
        opt = scaler = None
        try:
            model.load_state_dict(initial); model.train(); clean(); reset_peak()
            opt = torch.optim.AdamW(model.parameters(),lr=1e-6,weight_decay=.01,fused=device.type=='cuda')
            scaler = torch.amp.GradScaler(device.type,enabled=device.type=='cuda',init_scale=1.)
            started = time.perf_counter()
            update(model,opt,scaler,data['probe_batch'],candidate,device,deadline); sync()
            passed = device.type!='cuda' or torch.cuda.max_memory_reserved()<.93*torch.cuda.get_device_properties(0).total_memory
            attempts.append(dict(micro=candidate,status='passed' if passed else 'headroom',
                                 seconds=time.perf_counter()-started,**peak()))
            if passed: micro=candidate; break
        except torch.cuda.OutOfMemoryError:
            attempts.append(dict(micro=candidate,status='OOM'))
        finally:
            model.zero_grad(set_to_none=True); opt=None; scaler=None; clean()
    if micro is None: raise RuntimeError('No safe training micro-batch')
    model.load_state_dict(initial)
    opt = torch.optim.AdamW(model.parameters(),lr=1e-6,weight_decay=.01,fused=device.type=='cuda')
    scaler = torch.amp.GradScaler(device.type,enabled=device.type=='cuda',init_scale=1.)
    # Complete first update outside measurement initializes Adam state/backend.
    update(model,opt,scaler,data['train_batches'][0],micro,device,deadline); sync()
    rows = []; reset_peak()
    for lengths in data['train_batches']:
        C.check_deadline(deadline); sync(); started=time.perf_counter()
        update(model,opt,scaler,lengths,micro,device,deadline); sync()
        rows.append(dict(seconds=time.perf_counter()-started, pairs=len(lengths),
                         source_tokens=sum(a for a,b in lengths),target_tokens=sum(b for a,b in lengths)))
    result = dict(micro=micro,effective_batch=128,probe=attempts,batches=rows,
                  pairs_per_s=sum(r['pairs'] for r in rows)/sum(r['seconds'] for r in rows),
                  seconds_per_effective128=float(np.mean([r['seconds'] for r in rows])),
                  **peak(),timer='CPU length collation/random ID creation+H2D+AMP forward/backward+clip+fused AdamW; synced wall')
    model.zero_grad(set_to_none=True); opt=None; scaler=None; clean()
    model.load_state_dict(initial)  # Every decode uses the initial random model.
    return result


class ForcedDecoder(Speed.Decoder):
    @torch.inference_mode()
    def __init__(self, context, k, share_cross=False):
        super().__init__(context,k,share_cross)
        self.ends=context.forced_lengths.repeat_interleave(k)

    @torch.inference_mode()
    def step(self, ids):
        logits=super().step(ids)
        # No natural EOS: at each row's predetermined last step EOS is certain.
        logits[:,C.EOS]=-torch.inf
        force=self.t>=self.ends
        logits.masked_fill_(force[:,None],-torch.inf)
        logits[:,C.EOS]=torch.where(force,torch.zeros_like(logits[:,C.EOS]),logits[:,C.EOS])
        return logits

    @torch.inference_mode()
    def select(self,index):
        super().select(index); self.ends=self.ends.index_select(0,index)


OPTIONS=Speed.Options(share_cross=True,async_eos=True,eos_every=32,compact_done=True,
                      race_sampler=True,lean_greedy=True)


@torch.inference_mode()
def forced(model, lengths, k, temperature, device, deadline):
    src=tokens([a for a,b in lengths],device)
    context=Fast.Context(model,src,512)
    context.forced_lengths=torch.tensor([b for a,b in lengths],device=device)
    old=Speed.Decoder; Speed.Decoder=ForcedDecoder
    try:
        stats={}
        ids,lp,_=Speed.decode(context,k=k,temperature=temperature,compute_logprob=temperature is not None,
                              options=OPTIONS,deadline=deadline,stats=stats)
        ends=context.forced_lengths.cpu().repeat_interleave(k).tolist()
        flat=ids.flatten(0,1)
        assert all(int(flat[i,b-1])==C.EOS and not bool((flat[i,:b-1]==C.EOS).any()) for i,b in enumerate(ends))
        return stats
    finally: Speed.Decoder=old


def generation_bench(model,data,device,deadline):
    model.half().eval(); out={}
    lengths=data['generation1024']
    # Short warmup includes both argmax and race/logprob kernels.
    for temp in (None,.7): forced(model,[[a,min(b,32)] for a,b in lengths[:32]],1,temp,device,deadline)
    for label,k,temp in [('greedy',1,None),('sample',4,.7)]:
        # 1024 greedy questions; sampling 256 questions x4 lanes =1024 sequences.
        chosen=lengths if k==1 else [lengths[i] for i in np.linspace(0,1023,256).round().astype(int)]
        oom=[]; lanes=1024
        while True:
            try:
                clean(); reset_peak(); sync(); start=time.perf_counter(); stats=[]
                for lo in range(0,len(chosen),lanes//k):
                    C.check_deadline(deadline)
                    stats.append(forced(model,chosen[lo:lo+lanes//k],k,temp,device,deadline))
                sync(); seconds=time.perf_counter()-start
                out[label]=dict(seconds=seconds,sequences=1024,questions=len(chosen),K=k,
                    seq_per_s=1024/seconds,lanes=lanes,OOM_attempts=oom,decode_stats=stats,**peak(),
                    timer='source IDs/H2D+encoder+cross-KV+forced v6 decode+D2H; no program execution')
                break
            except torch.cuda.OutOfMemoryError:
                oom.append(lanes); clean(); lanes//=2
                if lanes<64: raise
    return out


@torch.inference_mode()
def encoder_bench(model,data,device,deadline):
    # Prepared IDs are outside this timer, matching encoder compute rather than tokenizer.
    parts=[tokens(data['test1919_source_lengths'][lo:lo+256],device) for lo in range(0,1919,256)]
    model.eval()
    for src in parts[:1]: model.transformer.encoder(model.embed(src),src_key_padding_mask=src.eq(0))
    rows=[]; reset_peak()
    for _ in range(3):
        C.check_deadline(deadline); sync(); start=time.perf_counter()
        for src in parts: model.transformer.encoder(model.embed(src),src_key_padding_mask=src.eq(0))
        sync(); rows.append(time.perf_counter()-start)
    return dict(seconds=float(np.median(rows)),repeats=rows,questions=1919,batch=256,**peak())


def write(output,result):
    output.mkdir(parents=True,exist_ok=True)
    tmp=output/'summary.tmp'; tmp.write_text(json.dumps(result,indent=2,allow_nan=False)); tmp.replace(output/'summary.json')
    fields=['shape','params','pairs_per_s','seconds_per_effective128','micro','greedy_seq_per_s','sample_seq_per_s','encoder1919_s','status']
    with (output/'timings.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for row in result['rows']:
            tr=row.get('training',{}); gen=row.get('generation',{})
            writer.writerow(dict(shape=row['shape'],params=row['parameters']['total'],
                pairs_per_s=tr.get('pairs_per_s'),seconds_per_effective128=tr.get('seconds_per_effective128'),
                micro=tr.get('micro'),greedy_seq_per_s=gen.get('greedy',{}).get('seq_per_s'),
                sample_seq_per_s=gen.get('sample',{}).get('seq_per_s'),
                encoder1919_s=row.get('encoder',{}).get('seconds'),status=row['status']))


def run(data_path,output,start,smoke=False):
    device=torch.device('cpu' if smoke else 'cuda:0')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if not smoke:
        assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
        assert torch.__version__=='2.11.0+cu128' and torch.version.cuda=='12.8'
    data=json.loads(Path(data_path).read_text()); output=Path(output)
    deadline=start+1100
    result=dict(format='nlp-hw02-shape-bench-v1',status='running',rows=[],
        device=str(device),torch=torch.__version__,cuda=torch.version.cuda,random_init=True,
        no_weights_saved=True,data_provenance=data['provenance'],session_limit_s=1200,
        length_source='verified cache v2 train; joint L3 C-main test output/source lengths',
        module_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in
            ['final_core.py','final_fast.py','final_decode.py','final_speed.py','shape_specs.py','shape_bench.py']},
        lengths_payload_sha256=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        no_end_to_end_timer=True)
    write(output,result)
    for shape in SHAPES:
        if time.monotonic()>deadline-60: break
        row=dict(shape=name(shape),spec=spec(shape),parameters=parameter_formula(shape),status='running')
        result['rows'].append(row); write(output,result); model=None
        try:
            torch.manual_seed(42)
            model=C.SimpleTransformer(**spec(shape)).to(device)
            with torch.no_grad(): model.tokens.weight.div_(math.sqrt(model.width))
            assert parameter_counts(model)==row['parameters']
            row['training']=train_bench(model,data,device,deadline)
            row['generation']=generation_bench(model,data,device,deadline)
            row['encoder']=encoder_bench(model,data,device,deadline)
            row['status']='complete'
        except Exception as error:
            row['status']='failed'; row['error']=type(error).__name__+': '+str(error)[:300]
        finally:
            model=None; clean(); result['wall_s']=time.monotonic()-start; write(output,result)
        print(row['shape'],row['status'],row.get('training',{}).get('pairs_per_s'),flush=True)
    result['status']='complete' if len(result['rows'])==len(SHAPES) and all(r['status']=='complete' for r in result['rows']) else 'partial'
    result['wall_s']=time.monotonic()-start; write(output,result)
    return result
