#!/usr/bin/env python3
"""Two actual batch128 CPU steps, L2 finite reference versus the L3 production R.run."""
import argparse, contextlib, copy, gc, hashlib, json, sys, tempfile, time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import torch
import l1_runtime as L,l2_runtime as R,exp26_core as C,exp26_train as T
from l1_cache import CompactCache
from l2_sampler import load_order
from l3_resume import verify_anchor,phase_recipe
from l2_resume import cpu_restore
from l2_cpu_smoke import digest
from kaggle_io import write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['C-main','C-long2x'],required=True)
    p.add_argument('--session',choices=['l3','l4'],default='l3')
    for n in ('anchor-root','cache','weighted-order','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1);device=torch.device('cpu')
    pin=json.loads((ROOT/f'{a.session}-anchors.json').read_text())['anchors'][a.model]
    expected_points=5 if a.session=='l4' else 3
    file,anchor=verify_anchor(a.anchor_root,pin,check_environment=False,expected_points=expected_points);started=time.monotonic()
    initial={k:digest(anchor[k]) for k in ('model','optimizer','scaler','rng','state')}
    cache=CompactCache(a.cache,verify=False);assert cache.manifest_sha==pin['identity']['cache_manifest_sha256']
    order,_=load_order(a.cache if a.model=='C-main' else a.weighted_order,cache,pin['identity']['order_sha256'])
    def numerical(m,o,s):
        return {k:digest(v) for k,v in {'model':m.state_dict(),'optimizer':o.state_dict(),
                                      'scaler':s.state_dict(),'rng':T.cpu(T.rng_state())}.items()}
    reference=[];actual=[];restores=[]
    # Reference: exact restored CUDA Adam/scaler states; CPU forward/Adam adaptation only.
    m=C.SimpleTransformer(**anchor['model_spec'],dropout=0.)
    o=L.optimizer(m,.0015,True);state=cpu_restore(anchor,m,o)
    s=torch.amp.GradScaler('cpu',enabled=True);s.load_state_dict(anchor['scaler'])
    with patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
        for step in range(2):
            ids=L.take(order,state['cursor']+step*128,128)
            log,micro=L.resilient_update(m,o,s,cache,ids,device,state['micro_batch'],state['last_lr'])
            assert not log['AMP_skip'];reference.append(dict(numerical(m,o,s),indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
            print('CPU L2 reference',a.model,step+1,round(log['seconds'],2),flush=True)
    # CPU Adam reuses loaded moment tensors; re-read the sealed file for the second run.
    del m,o,s,anchor;gc.collect()
    _,anchor=verify_anchor(a.anchor_root,pin,check_environment=False,expected_points=expected_points)
    old_restore=L.restore_full;old_scaler=torch.amp.GradScaler;old_update=R.safe_update
    def restore(payload,m,o,s,d):
        result=old_restore(payload,m,o,s,d)
        assert digest(result)==initial['state'], 'L3 reset inherited cursor/history'
        restores.append({'state_exact':True,'model_exact':digest(m.state_dict())==initial['model'],
                         'optimizer_exact':digest(o.state_dict())==initial['optimizer'],
                         'scaler_exact':digest(s.state_dict())==initial['scaler']})
        assert all(restores[-1].values())
        for g in o.param_groups:g['fused']=False
        return result
    def update(m,o,s,cache,ids,*args):
        log,micro=old_update(m,o,s,cache,ids,*args)
        assert not log['AMP_skip'] and not log['nonfinite_skip']
        actual.append(dict(numerical(m,o,s),indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
        print('CPU L3 production run',a.model,len(actual),round(log['seconds'],2),flush=True)
        return log,micro
    with tempfile.TemporaryDirectory(prefix='nlp-l3-cpu-') as scratch:
        with patch.object(L,'restore_full',restore),patch.object(torch.amp,'GradScaler',
             lambda *args,**kw:old_scaler('cpu',enabled=True,init_scale=1024)),patch.object(R,'safe_update',update),\
             patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
            result=R.run(a.model,anchor,pin['identity'],cache,order,scratch,device,time.monotonic()+1200,
                         lambda *args:None,main_targets=(pin['l2_main_seconds']+1e9,),cooldown_seconds=(1.,),
                         eval_seconds=(1.,),smoke=True,max_updates=2)
        assert actual==reference and len(actual)==2, 'CPU continuation is not bit-identical'
        st=result['state']
        assert st['updates']==pin['updates']+2 and st['cursor']==pin['cursor']+256
        assert st['last_lr']==.0015 and st['l2_points']==anchor['state']['l2_points']
        rec=json.loads((Path(scratch)/'latest.json').read_text())
        saved=R.read_full(Path(scratch)/rec['file'],pin['identity'],check_environment=False)
        assert saved['state']==st and saved['identity']==anchor['identity']
        assert digest(saved['model'])==actual[-1]['model'] and digest(saved['optimizer'])==actual[-1]['optimizer']
        del saved
    report={'passed':True,'session':a.session,'model':a.model,'anchor_sha256':pin['sha256'],'anchor_bytes':pin['bytes'],
            'updates':pin['updates'],'cursor':pin['cursor'],'main_seconds':pin['main_seconds'],
            'initial_full_state_digests':initial,'restored_initial_state':restores,'steps':actual,
            'steps_bit_exact':2,'effective_batch':128,'micro_batch':anchor['state']['micro_batch'],
            'identity_history_cursor_preserved':True,'temporary_runtime_checkpoints_removed':True,
            'GPU':False,'limitation':'CPU FP32/no autocast, enabled CPU GradScaler restored, fused Adam disabled; CUDA/FP16/fused requires GPU smoke',
            'seconds':time.monotonic()-started}
    del anchor;gc.collect()
    file.unlink();report['downloaded_anchor_removed']=not file.exists();write(a.out,report)
    print(a.session.upper(),'REAL CPU RESUME PASSED',a.model,flush=True)

if __name__=='__main__':main()
