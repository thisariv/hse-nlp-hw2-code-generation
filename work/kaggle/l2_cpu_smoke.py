#!/usr/bin/env python3
"""Real L1 C audit with CPU-only finite-math comparison; no downloads or GPU launch."""
import argparse, contextlib, copy, gc, hashlib, json, sys, time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import numpy as np,torch
import l1_runtime as L,l2_runtime as Run,exp26_core as C,exp26_train as T
from l1_cache import CompactCache
from l2_resume import verify_anchor,cpu_restore,initial_state,branch_identity
from l2_sampler import build_order
from kaggle_io import write,sha
from tokenizers import Tokenizer

def digest(value):
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            a=v.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--anchor-root',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--order-out',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);device=torch.device('cpu');started=time.monotonic()
    expected=json.loads((ROOT/'l2-anchor.json').read_text());file,anchor=verify_anchor(a.anchor_root,expected,check_environment=False)
    initial={'model':digest(anchor['model']),'optimizer':digest(anchor['optimizer']),'rng':digest(anchor['rng']),
             'scaler':copy.deepcopy(anchor['scaler']),'history':digest(anchor['state']['points'])}
    cache=CompactCache(a.cache,verify=True);assert cache.manifest_sha==anchor['identity']['cache_manifest_sha256']
    original=Tokenizer.from_file(str(a.inputs/'tokenizer.json'));heldout=json.loads((a.inputs/'heldout4k.json').read_text())
    lengths=[len(original.encode(r['code'],add_special_tokens=False).ids) for r in heldout]
    weighted,meta=build_order(cache,lengths,a.order_out)
    order=np.load(a.cache/'order.npy',mmap_mode='r',allow_pickle=False)
    assert sha(a.cache/'order.npy')==anchor['identity']['order_sha256']
    long_identity=branch_identity(anchor['identity'],meta);long_state=initial_state(anchor,'C-long2x')
    assert long_state['cursor']==0 and long_state['updates']==47155 and long_state['points']==anchor['state']['points']
    # CPU numerical comparison uses real weights/Adam/scaler and actual next128 rows.
    # CUDA AMP kernels are unavailable here: FP32 forward, real enabled scaler and CPU AdamW.
    del anchor;gc.collect();runs={}
    for label in ('L1-reference','C-main','C-long2x'):
        payload=torch.load(file,map_location='cpu',weights_only=True)
        model=C.SimpleTransformer(**payload['model_spec'],dropout=0.)
        opt=L.optimizer(model,.0015,True);state=cpu_restore(payload,model,opt)
        scaler=torch.amp.GradScaler('cpu',enabled=True);scaler.load_state_dict(payload['scaler'])
        check_opt=copy.deepcopy(opt.state_dict())
        for g in check_opt['param_groups']:g['fused']=True
        assert digest(model.state_dict())==initial['model'] and digest(check_opt)==initial['optimizer']
        assert scaler.state_dict()==initial['scaler']
        rng=T.cpu(T.rng_state());assert digest(rng['torch_cpu'])==digest(payload['rng']['torch_cpu'])
        assert rng['python']==payload['rng']['python'] and rng['numpy']==payload['rng']['numpy']
        assert digest(payload['state']['points'])==initial['history']
        ids=L.take(weighted,0,128) if label=='C-long2x' else L.take(order,state['cursor'],128)
        del payload,check_opt;gc.collect()
        update=L.update if label=='L1-reference' else Run.safe_update
        with patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
            if label=='L1-reference':log=update(model,opt,scaler,cache,ids,device,state['micro_batch'],.0015)
            else:log,_=update(model,opt,scaler,cache,ids,device,state['micro_batch'],.0015)
        assert log['AMP_skip']==0 and not log.get('nonfinite_skip',0)
        runs[label]={'model_sha256':digest(model.state_dict()),'optimizer_sha256':digest(opt.state_dict()),
                     'rng_sha256':digest(T.cpu(T.rng_state())),'scaler':scaler.state_dict(),'seconds':log['seconds'],
                     'indices_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),'effective_batch':128,
                     'micro_batch':state['micro_batch'],'CE':log['CE']}
        print('CPU REAL RESUME',label,'seconds',round(log['seconds'],2),flush=True)
        del model,opt,scaler;gc.collect()
    for key in ('model_sha256','optimizer_sha256','rng_sha256','scaler','indices_sha256','CE'):
        assert runs['L1-reference'][key]==runs['C-main'][key], 'C-main numerical mismatch '+key
    assert runs['C-long2x']['indices_sha256']!=runs['C-main']['indices_sha256']
    write(a.out,{'passed':True,'anchor_sha256':expected['sha256'],'updates':expected['updates'],'cursor':expected['cursor'],
                 'main_seconds':expected['main_seconds'],'initial_state_exact':initial,'runs':runs,'weighted_order':meta,
                 'long_recipe_identity':long_identity,'main_bit_exact_CPU_continuation':True,'fork_initial_state_exact':True,
                 'GPU':False,'limitation':'CPU FP32 forward + restored enabled CPU scaler; fused CUDA/FP16 numerical identity requires reviewer GPU smoke',
                 'new_checkpoint_download_bytes':0,'existing_checkpoint_reused':True,'D_downloaded':False,
                 'temporary_checkpoint_copies':0,'seconds':time.monotonic()-started})

if __name__=='__main__':main()
