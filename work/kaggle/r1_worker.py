"""Independent R1 branch, exactly one visible GPU, no credentials."""
import argparse
import json
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--config',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());root=Path(cfg['working_root'])
    sys.path.insert(0,str(root/'src'))
    import torch
    import exp26_core as C
    import l2_eval as E
    from l1_cache import CompactCache
    from l2_sampler import load_order
    from l3_resume import verify_anchor
    from r1_data import filtered_order
    from r1_runtime import run
    from exp26_input import validate_training_inputs
    from kaggle_io import write
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device=torch.device('cuda:0')
    assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
    assert C.fused_adamw_available(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=CompactCache(cfg['compact_cache'],verify=False);assert cache.manifest_sha==cfg['cache_manifest_sha256']
    branch=cfg['branches'][a.gpu];pin=cfg['resume_anchors'][branch['anchor']]
    _,anchor=verify_anchor(cfg['anchor_root'],pin,expected_points=5)
    if branch['data']=='filtered':
        order,meta=filtered_order(branch['order_path'],cache,branch['order_sha256'],branch['order_seed'],
                                   root/f'order-gpu{a.gpu}')
    else:
        order_root=cfg['weighted_order_root'] if branch['anchor']=='C-long2x' else cfg['compact_cache']
        order,meta=load_order(order_root,cache,pin['identity']['order_sha256'])
        if branch['anchor']=='C-long2x':assert meta==anchor['identity']['recipe']['sampler']
    inputs=Path(cfg['starter_dataset']);full,_,_=validate_training_inputs(inputs)
    refs={s:json.loads((inputs/(s+'-reference-execution.json')).read_text()) for s in full}
    eligible=[i for i,r in enumerate(refs['old1024']['rows']) if r['status']=='numeric'];assert len(eligible)==991
    sets={'heldout4k':full['heldout4k'],'old991':[full['old1024'][i] for i in eligible]}
    refs={'heldout4k':refs['heldout4k'],'old991':{'rows':[refs['old1024']['rows'][i] for i in eligible]}}
    if cfg['gpu_smoke']:
        sets={s:rows[:64 if s=='heldout4k' else 16] for s,rows in sets.items()}
        refs={s:{'rows':r['rows'][:len(sets[s])]} for s,r in refs.items()}
        from r1_gpu_smoke import probe
        probe(anchor,branch,cache,order,root/f'smoke-probe-gpu{a.gpu}',device)
    def evaluate(point,out,deadline):
        return E.evaluate(point,out,inputs,sets,refs,device,deadline,True,smoke=cfg['gpu_smoke'],
                          vote_cap_seconds=cfg['vote_cap_seconds'],test_rows=64 if cfg['gpu_smoke'] else None)
    result=run(anchor,branch,cache,order,meta,root/f'model-GPU{a.gpu}',device,cfg,evaluate)
    if cfg['gpu_smoke']:
        from r1_gpu_smoke import v7
        v7(root/f'model-GPU{a.gpu}/weights/final-fp16.pt',cache,order,root/f'model-GPU{a.gpu}',device)
        write(root/f'r1-smoke-gpu{a.gpu}.json',dict(passed=result['status']=='complete',GPU=True,result=result))
    assert result['status']=='complete', 'R1 branch incomplete; stable state retained'


if __name__=='__main__':
    try:main()
    except BaseException as error:
        print('R1 WORKER FAILED',type(error).__name__,flush=True);sys.exit(1)
