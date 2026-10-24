"""Independent W/C fixed-N cooldown workers, with the frozen B4/B5 evaluation contract."""
import argparse,json,sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=(0,1),required=True);p.add_argument('--config',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());root=Path(cfg['working_root']);cfg['gpu']=str(a.gpu)
    sys.path.insert(0,str(root/('src-C' if a.gpu==1 else 'src')))
    import torch
    import exp26_core as C
    import b6_runtime as G
    import b1_eval as E
    import l2_eval as Q
    from b5_resume import verify_order
    from exp26_input import validate_training_inputs
    from l1_cache import CompactCache
    from kaggle_io import write
    torch.set_num_threads(1);torch.set_num_interop_threads(1);device=torch.device('cuda:0')
    assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0) and C.fused_adamw_available(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=CompactCache(cfg['compact_cache'],verify=False);assert cache.manifest_sha==cfg['cache_manifest_sha256']
    pin=cfg['anchors'][str(a.gpu)];name=('W600','C-long2x')[a.gpu]
    order,meta=verify_order(root/f'weighted-order-GPU{a.gpu}',cache,pin)
    inputs=Path(cfg['starter_dataset']);full,_,_=validate_training_inputs(inputs)
    refs={k:json.loads((inputs/(k+'-reference-execution.json')).read_text()) for k in full}
    eligible=[i for i,r in enumerate(refs['old1024']['rows']) if r['status']=='numeric'];assert len(eligible)==991
    sets=dict(heldout4k=full['heldout4k'],old991=[full['old1024'][i] for i in eligible])
    refs=dict(heldout4k=refs['heldout4k'],old991=dict(rows=[refs['old1024']['rows'][i] for i in eligible]))
    if cfg['gpu_smoke']:
        sets={k:v[:64 if k=='heldout4k' else 16] for k,v in sets.items()}
        refs={k:dict(rows=v['rows'][:len(sets[k])]) for k,v in refs.items()}
    def evaluate(point,out,deadline,final):
        metrics=E.evaluate(point,out,inputs,sets,refs,device,deadline,True,cfg['gpu_smoke'])
        rows=json.loads((Path(out)/'heldout4k-per-example.json').read_text())
        metrics['heldout_length_bins']=Q.scores(rows,list(range(len(rows))))
        vote=metrics['K8_tune2048'];vote['model']=name
        vote['offline_mixed']='B6 q0 fixed-N cooldown; join B4/B5 journals by id/question SHA; no online selection'
        write(Path(out)/'k8-tune2048/summary.json',vote);write(Path(out)/'metrics.json',metrics)
        return metrics
    model_root=root/f'model-GPU{a.gpu}';resume=Path(cfg['resume_roots'][str(a.gpu)])
    if cfg['gpu_smoke']:
        from b6_gpu_smoke import exercise
        result=exercise(name,cache,order,meta,model_root,device,cfg,evaluate,resume,pin)
    else:result=G.run(name,cache,order,meta,model_root,device,cfg,evaluate,resume,pin)
    print('B6 RESULT',name,result['status'],'final_complete',result['final_complete'],flush=True)


if __name__=='__main__':main()
