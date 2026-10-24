"""Bounded two-process supervisor and registered common-time C/D decision."""
import json, math, os, subprocess, sys, time
from pathlib import Path
from session_watchdog import Watchdog as ShortWatchdog, atomic_json
from session_runner import stop_workers, tail
from kaggle_io import write, sha, disk_guard

class Watchdog(ShortWatchdog):
    def __init__(self,config):
        super().__init__(config)
        self.hard_deadline=min(config['hard_deadline_monotonic'],config['session_start_monotonic']+43200)

def decision(working):
    working=Path(working);models={}
    for name in ('C','D'):
        points={}
        # Full resume carries completed historical metrics even when old generation/weight
        # files stay in the attached previous kernel output or an older live version.
        histories=[]
        for filename in ('summary.json','progress.json'):
            path=working/f'model-{name}'/filename
            if path.exists():
                value=json.loads(path.read_text());state=value.get('state',value)
                histories.append(state)
        if histories:
            state=max(histories,key=lambda s:s.get('updates',0))
            for item in state.get('points',[]):
                target=item['target_main_seconds']
                if abs(item['main_seconds']-target)<=max(5.,target*.005):
                    points[item['tag']]={'training':item,'metrics':item['metrics']}
        for p in (working/f'model-{name}'/'points').glob('*/complete.json'):
            point=json.loads(p.read_text());train=point['training'];target=train['target_main_seconds']
            if abs(train['main_seconds']-target)<=max(5.,target*.005):points[p.parent.name]=point
        models[name]=points
    common=sorted(set(models['C'])&set(models['D']),key=lambda tag:models['C'][tag]['training']['target_main_seconds'])
    if not common:return {'status':'pending','reason':'No completed common pure-main-time point'}
    def accuracy(name,tag):return models[name][tag]['metrics']['splits']['heldout4k']['exec']['accuracy']
    last=common[-1];c=accuracy('C',last);d=accuracy('D',last);slopes={};previous=None
    if len(common)>=2:
        previous=common[-2]
        for name in ('C','D'):
            a=models[name][previous]['training']['target_main_seconds'];b=models[name][last]['training']['target_main_seconds']
            slopes[name]=100*(accuracy(name,last)-accuracy(name,previous))/math.log2(b/a)
    both=(d<c and (c-d)*100<2 and slopes and slopes['D']>slopes['C'])
    return {'status':'measured','common_points':common,'last_common_point':last,'previous_common_point':previous,
            'heldout4k_accuracy':{'C':c,'D':d},'D_minus_C_pp':100*(d-c),'slope_pp_per_doubling':slopes,
            'continue_L2':['C','D'] if both else ['D' if d>c else 'C'],
            'rule':'best last common exec4k; both if D lags <2pp and has steeper latest log2-time slope; exact ties C',
            'time_basis':'pure main training time; cooldown compute recorded separately'}

def launch(config,worker,watchdog,live=None,smoke=False,heartbeat=30.):
    working=Path(config['working_root']);working.mkdir(parents=True,exist_ok=True)
    path=working/'session-config.json';write(path,config);handles=[];processes=[];reason='workers completed';stop_until=None
    try:
        for gpu in (0,1):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if smoke else str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            for secret in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'):env.pop(secret,None)
            h=(working/f'gpu{gpu}-process.log').open('w');handles.append(h)
            p=subprocess.Popen([sys.executable,str(worker),'--gpu',str(gpu),'--config',str(path)],env=env,
                               stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(p);watchdog.register(p)
        while any(p.poll() is None for p in processes):
            if live is not None and (live.uninsured or time.monotonic()-live.last_success>=7200.):
                reason='live insurance stale >=2h';stop_until=stop_until or min(time.monotonic()+120.,config['deadline_monotonic'])
            if time.monotonic()>=config['deadline_monotonic']:reason='session deadline';break
            # If one model failed, stop the other rather than leave an idle GPU billed for hours.
            if any(p.poll() not in (None,0) for p in processes):
                reason='worker failed';stop_until=stop_until or min(time.monotonic()+120.,config['deadline_monotonic'])
            if stop_until is not None:
                if not (working/'stop-request.json').exists():write(working/'stop-request.json',{'reason':reason})
                if time.monotonic()>=stop_until:break
            for gpu,p in enumerate(processes):
                print('L1 GPU',gpu,'status',p.poll(),' | '.join(tail(working/f'gpu{gpu}-process.log'))[:700],flush=True)
            write(working/'session-live.json',{'elapsed_seconds':time.monotonic()-config['session_start_monotonic'],
                  'exit_codes':[p.poll() for p in processes],'live_insurance_stale':bool(live and live.uninsured)})
            disk_guard(working,limit=10*1024**3)
            time.sleep(min(heartbeat,max(.01,config['deadline_monotonic']-time.monotonic())))
    finally:
        survivors=stop_workers(processes,watchdog)
        for h in handles:h.close()
        comparison=decision(working)
        report={'status':'complete' if len(processes)==2 and all(p.returncode==0 for p in processes) else 'partial',
                'reason':reason,'exit_codes':[p.poll() for p in processes],'unreaped_worker_pids':survivors,
                'wall_seconds':time.monotonic()-config['session_start_monotonic'],'comparison':comparison,
                'output_policy':'individual files; no ZIP; full stable states + all evaluated FP16 weights',
                'hard_shutdown_seconds':config['hard_shutdown_seconds'],'server_ceiling_seconds':43200}
        write(working/'session-summary.json',report)
        if live is not None:report['final_live_uploaded']=live.final_sync('normal-exit',timeout=max(.1,min(600.,watchdog.hard_deadline-time.monotonic()-60.)))
        write(working/'session-summary.json',report)
        files=[p for p in working.rglob('*') if p.is_file() and p.relative_to(working).parts[0]!='src'
               and p.name!='l1-output-manifest.json' and '.tmp' not in p.name]
        write(working/'l1-output-manifest.json',{'files':[{'file':p.relative_to(working).as_posix(),
              'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]})
        if not survivors:watchdog.finish()
    return report
