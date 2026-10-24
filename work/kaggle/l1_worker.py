"""One independent visible GPU per long-running model; no credentials in workers."""
import argparse, json, sys, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True,type=int);p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());working=Path(cfg['working_root']);sys.path.insert(0,str(working/'src'))
    import numpy as np, torch
    import l1_runtime as L, l1_eval as E
    from l1_cache import CompactCache
    from exp26_input import validate_training_inputs
    from kaggle_io import sha, write
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    smoke=cfg.get('smoke',False);gpu_smoke=cfg.get('gpu_smoke',False);device=torch.device('cpu' if smoke else 'cuda:0')
    if not smoke:assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
    if not smoke:
        import exp26_core as C
        assert C.fused_adamw_available(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    inputs=Path(cfg['starter_dataset']);cache=CompactCache(cfg['compact_cache'],verify=False)
    assert cache.manifest_sha==cfg['cache_manifest_sha256']
    order=np.load(Path(cfg['compact_cache'])/'order.npy',mmap_mode='r',allow_pickle=False)
    order_meta=json.loads((Path(cfg['compact_cache'])/'order.json').read_text())
    assert sha(Path(cfg['compact_cache'])/'order.npy')==order_meta['file']['sha256'] and len(order)==len(cache)
    if smoke:
        sets=json.loads((inputs/'smoke-sets.json').read_text());refs=json.loads((inputs/'smoke-refs.json').read_text())
    else:
        full,_,_=validate_training_inputs(inputs)
        refs={s:json.loads((inputs/(s+'-reference-execution.json')).read_text()) for s in full}
        eligible=[i for i,r in enumerate(refs['old1024']['rows']) if r['status']=='numeric']
        assert len(eligible)==991
        sets={'heldout4k':full['heldout4k'],'old991':[full['old1024'][i] for i in eligible]}
        refs={'heldout4k':refs['heldout4k'],'old991':{'rows':[refs['old1024']['rows'][i] for i in eligible]}}
        if gpu_smoke:
            sets={s:rows[:64 if s=='heldout4k' else 16] for s,rows in sets.items()}
            refs={s:{'rows':r['rows'][:len(sets[s])]} for s,r in refs.items()}
        write(working/f'gpu{a.gpu}-old991-index.json',{'original_indices':eligible})
    name='C' if a.gpu==0 else 'D';root=working/f'model-{name}'
    resume=None
    if cfg.get('resume_root'):
        candidate=Path(cfg['resume_root'])/f'model-{name}'
        if (candidate/'latest.json').exists():resume=candidate
        else:raise FileNotFoundError('Requested resume is missing; refuse fresh restart')
    evaluate=lambda point,out,deadline:E.evaluate(point,out,inputs,sets,refs,device,deadline,smoke,test_rows=256 if gpu_smoke else None)
    recipe=cfg.get('gpu_smoke_recipe',{}) if gpu_smoke else cfg.get('smoke_recipe',{})
    report=L.run(name,cache,order,order_meta['file']['sha256'],root,Path(cfg['temp_root'])/name,device,
          cfg['deadline_monotonic'],evaluate,smoke=smoke,spec=cfg.get('smoke_spec'),resume=resume,
          **recipe)
    if gpu_smoke:
        before=report['state'];assert before['updates']==cfg['smoke_main_updates']
        assert len(before['points'])==1 and before['branch_attempted_updates']>0
        receipt=json.loads((root/'latest.json').read_text())
        payload=L.read_full(root/receipt['file'],report['identity'])
        assert payload['state']==before
        # CUDA regression: fail after one scaled backward, restore the scaler/RNG,
        # then compare the retried update to a clean update at the returned micro-batch.
        import exp26_train as T
        m=C.SimpleTransformer(**payload['model_spec'],dropout=0.).to(device)
        o=L.optimizer(m,L.PEAK_LR[name],False);s=torch.amp.GradScaler('cuda',init_scale=1024)
        L.restore_full(payload,m,o,s,device);original_collate=L.collate;calls=[0]
        def injected_collate(*args):
            calls[0]+=1
            if calls[0]==2:raise torch.cuda.OutOfMemoryError('CUDA smoke injected after scaled backward')
            return original_collate(*args)
        L.collate=injected_collate
        try:
            retried,retry_micro=L.resilient_update(m,o,s,cache,L.take(order,before['cursor'],128),device,
                         before['micro_batch'],L.lr_at(before['updates']+1,L.PEAK_LR[name],1000))
        finally:L.collate=original_collate
        assert len(retried['OOM_retries'])==1 and retry_micro<before['micro_batch']
        retry_weights={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};retry_rng=T.cpu(T.rng_state())
        retry_scaler=s.state_dict();L.restore_full(payload,m,o,s,device)
        clean=L.update(m,o,s,cache,L.take(order,before['cursor'],128),device,retry_micro,
                       L.lr_at(before['updates']+1,L.PEAK_LR[name],1000))
        difference=max(float((v.detach().cpu()-retry_weights[k]).abs().max()) for k,v in m.state_dict().items())
        assert difference<=1e-6 and s.state_dict()==retry_scaler
        assert torch.equal(T.rng_state()['torch_cuda'][0].cpu(),retry_rng['torch_cuda'][0])
        cuda_oom={'passed':True,'injected_after_scaled_backward':True,'retry_micro_batch':retry_micro,
                  'max_weight_difference_from_clean_update':difference,'weight_tolerance':1e-6,
                  'scaler_and_CUDA_RNG_exact':True}
        del m,o,s,payload,retry_weights,retry_rng
        import gc
        gc.collect();torch.cuda.empty_cache()
        peak=torch.cuda.max_memory_reserved(device)
        recipe=dict(recipe,max_updates=cfg['smoke_resume_updates'])
        continued=L.run(name,cache,order,order_meta['file']['sha256'],root,Path(cfg['temp_root'])/name,device,
              cfg['deadline_monotonic'],evaluate,resume=root,**recipe)
        after=continued['state'];assert after['updates']==before['updates']+cfg['smoke_resume_updates']
        assert after['cursor']==before['cursor']+cfg['smoke_resume_updates']*128
        assert after['points']==before['points'] and after['AMP_skips']>=before['AMP_skips']
        write(root/'gpu-smoke.json',{'passed':True,'device':str(device),'GPU':torch.cuda.get_device_name(0),
              'phases':['full longest-row memory probe','main train','cooldown branch','CUDA FP16 greedy/CE',
                        'heldout restored execution','test256 greedy timer','full anchor restore','train after eval',
                        'full state resume into new model/Adam/scaler','train after resume'],
              'main_updates_before_resume':before['updates'],'main_updates_after_resume':after['updates'],
              'cursor_before_resume':before['cursor'],'cursor_after_resume':after['cursor'],
              'micro_batch':after['micro_batch'],'AMP_skips':after['AMP_skips'],
              'peak_reserved_bytes':max(peak,torch.cuda.max_memory_reserved(device)),
              'parameter_count':continued['parameter_count'],'points':after['points'],'CUDA_late_OOM_regression':cuda_oom})
        print('GPU SMOKE PASSED',name,'micro',after['micro_batch'],'peak',max(peak,torch.cuda.max_memory_reserved(device)),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        print('WORKER FAILED:',type(e).__name__,flush=True)
        # Credentials have been stripped from worker env; failure details remain local/public code only.
        import traceback
        traceback.print_exc();sys.exit(1)
