"""Two isolated EV processes; data setup is identical to B1."""
import argparse
import json
import sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--config',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());root=Path(cfg['working_root'])
    sys.path.insert(0,str(root/'src'))
    import torch
    from l1_cache import CompactCache
    from b1_data import verify_order
    from exp26_input import validate_training_inputs
    from b1_ev import run
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device=torch.device('cuda:0');assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=CompactCache(cfg['compact_cache'],verify=False);order,_=verify_order(root/'weighted-order',cache)
    inputs=Path(cfg['starter_dataset']);full,_,_=validate_training_inputs(inputs)
    refs={k:json.loads((inputs/(k+'-reference-execution.json')).read_text()) for k in full}
    eligible=[i for i,r in enumerate(refs['old1024']['rows']) if r['status']=='numeric'];assert len(eligible)==991
    sets=dict(heldout4k=full['heldout4k'],old991=[full['old1024'][i] for i in eligible])
    refs=dict(heldout4k=refs['heldout4k'],old991=dict(rows=[refs['old1024']['rows'][i] for i in eligible]))
    run(a.gpu,cfg,cache,order,root/f'model-GPU{a.gpu}',inputs,sets,refs,device)

if __name__=='__main__':main()
