"""Spawn entrypoint imports Torch only inside main; sandbox workers stay Torch-free."""
import argparse,json,os,sys,time,traceback
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--config',type=Path,required=True)
    a=parser.parse_args();config=json.loads(a.config.read_text());working=Path(config['working_root'])
    sys.path.insert(0,str(working/'src'));os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch,numpy as np
    from exp26_common import write_json,read_json,sha_file
    import exp26_core as C
    import exp26_protocol as P
    import exp26_train as T
    import exp26_eval as E
    import exp26_workflow as W
    import kaggle_runtime as R
    from kaggle_io import disk_guard
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device=torch.device('cuda:0')
    assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0),'Each worker must see one T4'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    R.install_runtime_adapters()
    write_json(working/f'gpu{a.gpu}-runtime.json',{'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
        'CUDA':torch.version.cuda,'SDPA':C.sdpa_flags(),'exec_workers':2,'torch_threads':1,'visible_devices':os.environ['CUDA_VISIBLE_DEVICES']})
    # Materialized once by supervisor before both processes; mmap input shared read-only.
    prepared,sets,test,cache,run,order_meta=R.inputs(config,working)
    order=np.load(run/'order.npy',mmap_mode='r',allow_pickle=False)
    assert sha_file(run/'order.npy')==R.ORDER_SHA
    source_sha=config['source_sha'];deadline=config['deadline_monotonic']
    if a.gpu==0:
        root=R.prepare_a(config,working,run)
        if config.get('previous_output'):
            R.copy_output(R.resolve_dataset(config['input_root'],config['previous_output'])/'gpu0-exp27',working/'gpu0-exp27')
        refs=W.reference_execution(sets,prepared,root)
        # Always a fresh Kaggle D0 reproduction; Colab's receipts remain preserved.
        controls=root/('D0-kaggle-'+str(torch.__version__).replace('/','_'))
        D0=E.evaluate_point(prepared/'control-D0.pt',sets,refs,prepared/'tokenizer.json',device,controls,'D0')
        E.verify_D0(D0)
        callback=lambda p,t,prev:R.evaluate_a(p,t,prev,sets,refs,prepared,root,device,D0)
        result=T.run_training(cache,order,order_meta,prepared,root,root,source_sha,device,callback,
                    checkpoint_seconds=600,session_minutes=max(0,(deadline-time.monotonic()-1800)/60))
        report=W.collect_report(root,root);report['runtime_budget_scope']='whole S1 session incl setup; controller receives remaining time'
        if result['complete']:
            timing=W.generate_final_test(test,prepared,root,root,device)
            report['test1919']=timing
        write_json(root/'26b-final-summary.json',report)
        if result['complete'] and time.monotonic()<deadline-300:
            import exp27_decode as D
            final=read_json(root/'final-checkpoint.json');cp=root/final['file']
            metrics=read_json(root/'points/p050000/evaluation/metrics.json')
            D.run_experiment(cp,sets,refs,prepared,device,working/'gpu0-exp27',metrics,root/'points/p050000/evaluation',
                             deadline=deadline-180,batch_rows=config['sample_batch_rows'])
    else:
        import exp28_capacity as B
        root=working/'gpu1-exp28'
        if config.get('previous_output'):
            previous=R.resolve_dataset(config['input_root'],config['previous_output'])
            R.copy_output(previous/'gpu1-exp28',root)
            if not all((root/'references'/(s+'-receipt.json')).exists() for s in sets):
                R.reuse_verified_references(previous/'gpu0-a26/references',root/'references',sets,prepared)
        refs=W.reference_execution(sets,prepared,root)
        controls=root/('D0-kaggle-'+str(torch.__version__).replace('/','_'))
        D0=E.evaluate_point(prepared/'control-D0.pt',sets,refs,prepared/'tokenizer.json',device,controls,'D0');E.verify_D0(D0)
        def evaluate(cp,tag,previous,target):
            C.check_deadline(deadline-900)
            prev=read_json(root/'points'/previous/'evaluation/metrics.json') if previous else None
            m=E.evaluate_point(cp,sets,refs,prepared/'tokenizer.json',device,root/'points'/tag/'evaluation',tag,D0,prev,spec=B.SPEC)
            ap=run/'points'/target['A_point']/'evaluation/metrics.json'
            if ap.exists():
                am=read_json(ap);m['McNemar_against_A']={s:E.paired_mcnemar(m['splits'][s]['exec'],am['splits'][s]['exec']) for s in sets}
            write_json(root/'points'/tag/'evaluation/metrics.json',m);return m
        report=B.run(cache,order,order_meta,prepared,root,config['port_sha'],device,evaluate,deadline-1200)
        a_metrics={}
        for target in B.TARGETS:
            p=run/'points'/target['A_point']/'evaluation/metrics.json'
            if p.exists():a_metrics[target['A_point']]=read_json(p)
        common=[(tag,v) for tag,v in report['points'].items() if v['time_aligned'] and v['A_point'] in a_metrics]
        if common and time.monotonic()<deadline-600:
            import exp27_decode as D
            tag,point=max(common,key=lambda kv:kv[1]['target_train_seconds']);r=read_json(root/'points'/tag/'checkpoint.json');cp=root/r['file']
            def timed(cfg,dest):
                p=dest/'timing.json'
                if p.exists():
                    old=read_json(p)
                    assert old['model_sha256']==sha_file(cp) and old['config']==cfg
                    assert old['answers_sha256']==sha_file(dest/'test-answers.jsonl')
                    return old
                return D.time_test(cp,prepared/'test.json',prepared/'tokenizer.json',device,dest,cfg,B.SPEC,
                                   deadline=deadline-180,batch_rows=config['sample_batch_rows'])
            greedy=timed(D.configurations()[0],root/'test'/tag/'greedy')
            report['greedy_test1919']=greedy
            chosen_path=working/'gpu0-exp27/27-summary.json'
            # Give GPU0 the remainder of its budget while GPU1 is available for chosen27 timing.
            def decoding_ready():
                return chosen_path.exists() and read_json(chosen_path).get('status')=='complete'
            while not decoding_ready() and time.monotonic()<deadline-600:time.sleep(15)
            if decoding_ready():
                chosen=read_json(chosen_path)['decision']['config']
                chosen_time=greedy if chosen['mode']=='greedy' else timed(chosen,root/'test'/tag/('selected27-'+chosen['name']))
                report['selected27_test1919']=chosen_time
                report['decision']=B.decision(report['points'],a_metrics,chosen_time)
            else:report['decision']=B.decision(report['points'],a_metrics,None)
        else:report['decision']=B.decision(report['points'],a_metrics,None)
        write_json(root/'28-summary.json',report)
    disk_guard(working)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        # Preserve last complete state; never serialize live tensors interrupted inside fused update.
        try:
            args=sys.argv;config=json.loads(Path(args[args.index('--config')+1]).read_text());gpu=args[args.index('--gpu')+1]
            file=Path(config['working_root'])/f'gpu{gpu}-failure.json'
            file.write_text(json.dumps({'status':'partial','error':type(e).__name__,'message':str(e)[:1000]},indent=2))
        finally:traceback.print_exc()
        sys.exit(1)
