"""Two isolated S2 processes: GPU0 27b then M; GPU1 B. No Torch at import time."""
import argparse,json,os,sys,time,traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--config',type=Path,required=True)
    args=p.parse_args();cfg=json.loads(args.config.read_text());working=Path(cfg['working_root']);sys.path.insert(0,str(working/'src'))
    import torch,numpy as np
    import exp26_core as C,exp26_eval as E,exp28_capacity as B,kaggle_runtime as R,s2_runtime as S,exp27b_fast as D
    from exp26_common import read_json,write_json,sha_file
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device=torch.device('cuda:0');assert torch.cuda.device_count()==1 and 'T4' in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    S.configure_storage(cfg['temp_root']);S.install()
    deadline=cfg['deadline_monotonic'];train_deadline=deadline-1200
    write_json(working/f'gpu{args.gpu}-runtime.json',{'GPU':torch.cuda.get_device_name(0),'torch':str(torch.__version__),'CUDA':torch.version.cuda,
        'visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'torch_threads':1,'exec_workers':2,'wall_limit':3.,'cpu_limit':2.,'SDPA':C.sdpa_flags()})
    inputs,sets,test,cache,run,order_meta=R.inputs(cfg,working)
    order=np.load(run/'order.npy',mmap_mode='r',allow_pickle=False);assert sha_file(run/'order.npy')==R.ORDER_SHA
    previous=R.resolve_dataset(cfg['input_root'],cfg['previous_output']);a_root=previous/'gpu0-a26'
    targets=cfg['targets'];experiment=29 if args.gpu==0 else 28;spec=B.M_SPEC if experiment==29 else B.SPEC
    root=working/f'gpu{args.gpu}-exp{experiment}';root.mkdir(parents=True,exist_ok=True)
    if cfg.get('s2_resume_output'):
        attached=R.resolve_dataset(cfg['input_root'],cfg['s2_resume_output']);R.copy_output(attached/root.name,root)
        if args.gpu==0:R.copy_output(attached/'gpu0-exp27b',working/'gpu0-exp27b')
    refs=S.references(previous,sets,inputs,root)
    if args.gpu==0:
        a_metrics={t['A_point']:read_json(a_root/'points'/t['A_point']/'evaluation/metrics.json') for t in targets}
        write_json(working/'a-baselines.json',a_metrics)
        a_final=read_json(a_root/'final-checkpoint.json');a_cp=a_root/a_final['file'];assert sha_file(a_cp)==a_final['sha256']
        with S.evaluation_lock(working,deadline-1800) as waited:
            fast=D.run(a_cp,sets,refs,inputs,device,working/'gpu0-exp27b',a_root/'points/p050000/evaluation',deadline-1500)
        write_json(working/'27b-lock-wait.json',{'seconds':waited})
        chosen=fast['decision']['config']
        # Final A test uses this exact already measured checkpoint.
        write_json(working/'a-test/t04/selected/timing.json',next(t for t in fast['test'] if t['config']==chosen))
    # Reproduction gate is serialized, so slow reference/program imports cannot race another eval.
    with S.evaluation_lock(working,train_deadline) as waited:
        D0=E.evaluate_point(inputs/'control-D0.pt',sets,refs,inputs/'tokenizer.json',device,root/'D0-kaggle','D0')
        try:gate=E.verify_D0(D0)
        except AssertionError:
            write_json(root/'D0-gate.json',{'status':'failed; stop before training','metrics':D0});raise
        write_json(root/'D0-gate.json',gate)
    def evaluate(cp,tag,prev,target):
        with S.evaluation_lock(working,deadline-900) as waited:
            C.check_deadline(deadline-900)
            previous_metrics=read_json(root/'points'/prev/'evaluation/metrics.json') if prev else None
            metrics=E.evaluate_point(cp,sets,refs,inputs/'tokenizer.json',device,root/'points'/tag/'evaluation',tag,D0,previous_metrics,spec=spec)
            am=read_json(a_root/'points'/target['A_point']/'evaluation/metrics.json')
            metrics['McNemar_against_A']={s:E.paired_mcnemar(metrics['splits'][s]['exec'],am['splits'][s]['exec']) for s in sets}
            metrics['mean_generated_length']={s:sum(len(g['generated_token_ids']) for g in read_json(root/'points'/tag/'evaluation'/(s+'-generations.json')))/len(sets[s]) for s in sets}
            metrics['evaluation_lock_wait_seconds']=waited
            write_json(root/'points'/tag/'evaluation/metrics.json',metrics);return metrics
    report=B.run(cache,order,order_meta,inputs,root,cfg['port_sha'],device,evaluate,train_deadline,
        cfg=B.config(experiment=experiment),spec=spec,targets=targets,stop_after_targets=True,allow_batch_fallback=experiment==28)
    write_json(root/'training-finished.json',{'points':list(report['points']),'missing_points':report['missing_points']})
    chosen_file=working/'gpu0-exp27b/27b-summary.json'
    if chosen_file.exists() and read_json(chosen_file).get('status')=='complete':
        chosen=read_json(chosen_file)['decision']['config']
        def measure(tag):
            receipt=read_json(root/'points'/tag/'checkpoint.json');cp=root/receipt['file']
            with S.evaluation_lock(working,deadline-180) as waited:
                probe=D.probe_batch(cp,test,inputs/'tokenizer.json',device,root/'test'/tag/'memory',k=max(chosen['k'],16),spec=spec,deadline=deadline-180)
                greedy=D.time_test(cp,inputs/'test.json',inputs/'tokenizer.json',device,root/'test'/tag/'greedy',D.config(),spec,probe['batch_rows'],deadline=deadline-180)
                selected=greedy if chosen['mode']=='greedy' else D.time_test(cp,inputs/'test.json',inputs/'tokenizer.json',device,root/'test'/tag/'selected',chosen,spec,probe['batch_rows'],deadline=deadline-180)
            report.setdefault('test_by_point',{})[tag]={'greedy':greedy,'selected':selected,'batch_probe':probe,'lock_wait_seconds':waited}
            write_json(root/f'{experiment}-summary.json',report)
        eligible=[tag for tag,v in report['points'].items() if v['time_aligned']]
        if eligible and time.monotonic()<deadline-600:measure(max(eligible))
        other=working/('gpu1-exp28' if args.gpu==0 else 'gpu0-exp29')
        # Wait only within reserved test budget; no additional training or hidden quota extension.
        other_gpu=1 if args.gpu==0 else 0
        while not (other/'training-finished.json').exists() and not (working/f'gpu{other_gpu}-failure.json').exists() and time.monotonic()<deadline-700:time.sleep(5)
        if (other/'training-finished.json').exists():
            other_report=read_json(other/('28-summary.json' if args.gpu==0 else '29-summary.json'))
            common=[tag for tag in eligible if other_report.get('points',{}).get(tag,{}).get('time_aligned')]
            if common and max(common) not in report.get('test_by_point',{}) and time.monotonic()<deadline-600:measure(max(common))
            if args.gpu==0 and common:
                tag=max(common);target=targets[int(tag[1:])-1];dest=working/'a-test'/tag/'selected'
                if not (dest/'timing.json').exists() and time.monotonic()<deadline-400:
                    receipt_path=a_root/'points'/target['A_point']/'checkpoint.json'
                    if receipt_path.exists():
                        receipt=read_json(receipt_path);cp=a_root/receipt['file']
                        if cp.exists():
                            with S.evaluation_lock(working,deadline-180):D.time_test(cp,inputs/'test.json',inputs/'tokenizer.json',device,dest,chosen,batch_rows=read_json(chosen_file)['batch_probe']['batch_rows'],deadline=deadline-180)
                        else:write_json(working/'a-test'/tag/'missing-checkpoint.json',{'A_point':target['A_point'],'reason':'S1 did not retain this tensor checkpoint; final selection pending when A timing needed'})
    write_json(root/f'{experiment}-summary.json',report)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        try:
            config=json.loads(Path(sys.argv[sys.argv.index('--config')+1]).read_text());gpu=sys.argv[sys.argv.index('--gpu')+1]
            path=Path(config['working_root'])/f'gpu{gpu}-failure.json'
            from session_watchdog import atomic_json
            atomic_json(path,{'status':'partial','error':type(error).__name__,'message':str(error)[:1000]})
        finally:traceback.print_exc()
        sys.exit(1)
