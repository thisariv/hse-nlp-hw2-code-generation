"""Independent workers: a failed sibling never requests a healthy worker stop."""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
from l1_session import Watchdog
from session_runner import stop_workers, tail
from kaggle_io import write, sha

def disk_guard(root,limit):
    """Rotation and atomic JSON publication may remove a path during this scan."""
    used=0
    for p in Path(root).rglob('*'):
        try:
            if p.is_file():used+=p.stat().st_size
        except FileNotFoundError:continue
    if used>limit:raise RuntimeError('Working output exceeds reserve')
    if shutil.disk_usage(root).free<512*1024**2:raise RuntimeError('Less than 512 MiB free')
    return used

def launch(config,worker,watchdog,live=None,smoke=False,heartbeat=30.):
    working=Path(config['working_root']);working.mkdir(parents=True,exist_ok=True)
    path=working/'session-config.json';write(path,config);handles=[];processes=[];reason='workers completed';stop_until=None;failed=set()
    try:
        for gpu in (0,1):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if smoke else str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            for secret in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN',
                           'KAGGLE_USER_SECRETS_TOKEN','KAGGLE_IAP_TOKEN'):env.pop(secret,None)
            h=(working/f'gpu{gpu}-process.log').open('w');handles.append(h)
            p=subprocess.Popen([sys.executable,str(worker),'--gpu',str(gpu),'--config',str(path)],env=env,
                               stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(p);watchdog.register(p)
        while any(p.poll() is None for p in processes):
            if live is not None and (live.uninsured or time.monotonic()-live.last_success>=7200.):
                reason='live insurance stale >=2h';stop_until=stop_until or min(time.monotonic()+120.,config['deadline_monotonic'])
            if time.monotonic()>=config['deadline_monotonic']:reason='session deadline';break
            for gpu,p in enumerate(processes):
                if p.poll() not in (None,0) and gpu not in failed:
                    failed.add(gpu)
                    write(working/f'gpu{gpu}-failure.json',{'exit_code':p.poll(),'healthy_worker_stopped':False,
                          'available_state':(working/('model-C-main' if gpu==0 else 'model-C-long2x')/'latest.json').exists()})
                    print('L2 worker failed',gpu,'; other worker continues independently',flush=True)
            if stop_until is not None:
                if not (working/'stop-request.json').exists():write(working/'stop-request.json',{'reason':reason})
                if time.monotonic()>=stop_until:break
            for gpu,p in enumerate(processes):
                print('L2 GPU',gpu,'status',p.poll(),' | '.join(tail(working/f'gpu{gpu}-process.log'))[:700],flush=True)
            write(working/'session-live.json',{'elapsed_seconds':time.monotonic()-config['session_start_monotonic'],
                  'exit_codes':[p.poll() for p in processes],'live_insurance_stale':bool(live and live.uninsured)})
            disk_guard(working,limit=10*1024**3)
            time.sleep(min(heartbeat,max(.01,config['deadline_monotonic']-time.monotonic())))
    except Exception as error:
        reason='supervisor error: '+type(error).__name__
        write(working/'supervisor-failure.json',{'error_type':type(error).__name__})
    finally:
        survivors=stop_workers(processes,watchdog)
        for h in handles:h.close()
        if not survivors:
            for p in working.glob('model-*/checkpoints/*.pt.tmp'):p.unlink(missing_ok=True)
        from l2_compare import compare
        try:comparison=compare(working)
        except Exception as error:comparison={'status':'incomplete','error_type':type(error).__name__}
        for gpu,p in enumerate(processes):
            if p.returncode not in (None,0):
                failed.add(gpu)
                if not (working/f'gpu{gpu}-failure.json').exists():
                    write(working/f'gpu{gpu}-failure.json',{'exit_code':p.returncode,'healthy_worker_stopped':False})
        worker_status={}
        for name in ('C-main','C-long2x'):
            p=working/f'model-{name}'/'summary.json'
            try:worker_status[name]=json.loads(p.read_text()).get('status') if p.exists() else 'missing'
            except (OSError,ValueError):worker_status[name]='unreadable'
        report={'status':'complete' if len(processes)==2 and all(p.returncode==0 for p in processes)
                 and all(s=='complete' for s in worker_status.values()) else 'partial','worker_status':worker_status,
                'reason':reason,'failed_workers':sorted(failed),'exit_codes':[p.poll() for p in processes],'unreaped_worker_pids':survivors,
                'wall_seconds':time.monotonic()-config['session_start_monotonic'],'comparison':comparison,
                'output_policy':'individual files; no ZIP; full stable states + all evaluated FP16 weights',
                'hard_shutdown_seconds':config['hard_shutdown_seconds'],'server_ceiling_seconds':43200}
        write(working/'session-summary.json',report)
        if live is not None:
            try:report['final_live_uploaded']=live.final_sync('normal-exit',timeout=max(.1,min(1200.,watchdog.hard_deadline-time.monotonic()-120.)))
            except Exception as error:
                report.update(final_live_uploaded=False,final_live_error_type=type(error).__name__)
        write(working/'session-summary.json',report)
        files=[p for p in working.rglob('*') if p.is_file() and p.relative_to(working).parts[0]!='src'
               and p.name!='l2-output-manifest.json' and '.tmp' not in p.name]
        write(working/'l2-output-manifest.json',{'files':[{'file':p.relative_to(working).as_posix(),
              'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]})
        if not survivors:watchdog.finish()
    return report
