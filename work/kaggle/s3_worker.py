"""GPU0 compact A then B scoring; GPU1 numeric placeholders on original A target."""
import argparse, json, os, sys, time, traceback
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());working=Path(cfg['working_root']);sys.path.insert(0,str(working/'src'))
    import numpy as np, torch
    import exp26_core as C, exp26_exec as N, s3_runtime as S, s3_score as B
    from exp26_common import read_json, write_json, sha_file
    from exp26_input import validate_training_inputs
    from exp26_cache import TokenCache
    from kaggle_io import resolve_dataset
    from kaggle_runtime import CACHE_SHA, ORDER_SHA
    from s2_exec import install
    install()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device=torch.device('cuda:0');assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert C.fused_adamw_available(device)
    inputs=Path(cfg['starter_dataset']);sets,_,protected=validate_training_inputs(inputs)
    cache=TokenCache(cfg['cache_dataset'],protected);assert cache.manifest_sha==CACHE_SHA
    run=Path(cfg['resume_dataset']);assert sha_file(run/'order.npy')==ORDER_SHA
    order=np.load(run/'order.npy',mmap_mode='r',allow_pickle=False);order_meta=read_json(run/'order.json')
    assert len(order)==len(cache) and [r['shard'] for r in cache.manifest['shards']]==list(range(42))
    refs={s:read_json(inputs/(s+'-reference-execution.json')) for s in sets}
    baselines={}
    for origin,expected in ((6250,1888),(12500,2210)):
        tag=f'p{origin:06d}';point=run/'points'/tag
        baseline=read_json(point/'evaluation/metrics.json')
        for split,rows in sets.items():
            gens=read_json(point/'evaluation'/(split+'-generations.json'))
            assert [g['question'] for g in gens]==[r['question'] for r in rows]
            assert [g['id'] for g in gens]==list(range(len(rows)))
            saved=N.compare_exec(refs[split],read_json(point/'evaluation'/(split+'-execution.json')))
            assert saved==baseline['splits'][split]['exec']
        assert baseline['splits']['heldout4k']['exec']['matches']==expected
        baseline['train_seconds_total']=read_json(point/'training.json')['train_seconds_total']
        baselines[tag]=baseline
    write_json(working/f'gpu{args.gpu}-runtime.json',{'GPU':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
        'CUDA':torch.version.cuda,'SDPA':C.sdpa_flags(),'cache_sha256':cache.manifest_sha,'order_sha256':ORDER_SHA,
        'CPU_preparation_workers':2,'CPU_execution_workers':2,'reference_counts':{s:sum(r['status']=='numeric' for r in v['rows']) for s,v in refs.items()}})
    variant=30 if args.gpu==0 else 31
    root=working/f'gpu{args.gpu}-exp{variant}';scratch=Path(cfg['temp_root'])/f'gpu{args.gpu}'
    scratch.mkdir(parents=True,exist_ok=True)
    try:
        S.run(inputs,cache,order,order_meta,sets,refs,baselines,variant,device,root,scratch,
              cfg['deadline_monotonic']-1500 if args.gpu==0 else cfg['deadline_monotonic']-120,cfg['port_sha'])
    except C.BudgetExpired:
        write_json(root/'budget-stop.json',{'status':'partial','reason':'reserved budget exhausted; completed receipts retained'})
    if args.gpu==0:
        previous=resolve_dataset(cfg['input_root'],cfg['previous_output'])
        source=resolve_dataset(cfg['input_root'],cfg['b_output'])/'gpu1-exp28'
        receipt=read_json(source/'points/t04/checkpoint.json');weights=source/receipt['file']
        assert sha_file(weights)==receipt['sha256']=='cbbe292cf65e37ec4876b8206b19b06b6fbe7541fd7be1115799c3ce085f720a'
        payload=torch.load(weights,map_location='cpu',weights_only=True);spec=payload['model_spec'];del payload
        try:B.run(weights,spec,previous/'gpu0-exp27',sets,inputs/'tokenizer.json',device,working/'gpu0-exp32',cfg['deadline_monotonic']-60)
        except C.BudgetExpired:
            write_json(working/'gpu0-exp32/budget-stop.json',{'status':'partial','reason':'session deadline; saved score batches retained'})

if __name__=='__main__':
    try:main()
    except BaseException as error:
        try:
            config=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text());gpu=sys.argv[sys.argv.index('--gpu')+1]
            sys.path.insert(0,str(Path(config['working_root'])/'src'))
            from session_watchdog import atomic_json
            atomic_json(Path(config['working_root'])/f'gpu{gpu}-failure.json',{'status':'partial','error':type(error).__name__,'message':str(error)[:1000]})
        finally:traceback.print_exc()
        sys.exit(1)
