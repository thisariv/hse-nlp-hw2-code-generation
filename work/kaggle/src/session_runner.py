"""Two independent processes with one visible GPU each; wall deadline starts at first cell."""
import json,os,shutil,signal,subprocess,sys,time,zipfile
from session_watchdog import Watchdog,atomic_json,clean_checkpoints
from pathlib import Path
from kaggle_io import write,sha,disk_guard

def archive(working,kind='s1'):
    working=Path(working);target=working/f'{kind}-results.zip';tmp=working/f'{kind}-results.zip.tmp'
    if kind in ('s2','s3'):
        clean_checkpoints(working)
        for path in working.rglob('*'):
            if path.is_file() and (path.name.endswith('.tmp') or '.tmp-' in path.name):path.unlink(missing_ok=True)
    # Always leave a per-experiment summary, including interruptions before the first metric.
    experiments=([('gpu0-exp30','30-summary.json',30),('gpu1-exp31','31-summary.json',31),('gpu0-exp32','32-summary.json',32)] if kind=='s3' else
        [('gpu0-exp27b','27b-summary.json','27b'),('gpu0-exp29','29-summary.json',29),('gpu1-exp28','28-summary.json',28)]
        if kind=='s2' else [('gpu0-a26','26b-final-summary.json','26b'),('gpu0-exp27','27-summary.json',27),('gpu1-exp28','28-summary.json',28)])
    for directory,filename,experiment in experiments:
        root=working/directory;p=root/filename
        if not p.exists():
            progress=root/'progress.json';partial=json.loads(progress.read_text()) if progress.exists() else {}
            partial.update(experiment=experiment,status='partial; see process/failure log and latest complete checkpoint')
            write(p,partial)
    files=[p for p in sorted(working.rglob('*')) if p.is_file() and
           p.relative_to(working).parts[0] not in ('src','inputs','unpacked-cache','unpacked-resume') and
           p.name not in (f'{kind}-results.zip',f'{kind}-results.zip.tmp',f'{kind}-archive-receipt.json')]
    omitted=[]
    if kind in ('s2','s3'):
        # Individual Kaggle outputs only: weights/AdamW are never duplicated in a ZIP.
        manifest={'output_policy':'individual files; no ZIP','files':[
            {'file':p.relative_to(working).as_posix(),'bytes':p.stat().st_size} for p in files
            if p.name not in ('session-summary.json',kind+'-summary.json',kind+'-output-manifest.json')],
            'mutable_summaries_outside_manifest':['session-summary.json',kind+'-summary.json']}
        manifest['output_bytes']=disk_guard(working,limit=4*1024**3)
        atomic_json(working/(kind+'-output-manifest.json'),manifest)
        return working
    # A ZIP temporarily duplicates retained artifacts; reserve before writing, including archive overhead.
    total=sum(p.stat().st_size for p in working.rglob('*') if p.is_file())
    if total+sum(p.stat().st_size for p in files)+64*1024**2>19*1024**3:
        raise RuntimeError('ZIP would exceed19GiB working safety cap')
    manifest=[{'file':p.relative_to(working).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files]
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in files:z.write(p,p.relative_to(working))
        z.writestr('zip-manifest.json',json.dumps({'files':manifest,'retained_on_kaggle_outside_zip':[p.relative_to(working).as_posix() for p in omitted]},indent=2))
    tmp.replace(target);write(working/f'{kind}-archive-receipt.json',{'file':target.name,'bytes':target.stat().st_size,'sha256':sha(target)})
    disk_guard(working);return target

def tail(path):
    with Path(path).open('rb') as handle:
        handle.seek(0,2);handle.seek(max(0,handle.tell()-4096))
        return handle.read().decode(errors='replace').splitlines()[-3:]

def group_alive(group):
    try:os.killpg(group,0);return True
    except ProcessLookupError:return False

def stop_workers(processes,watchdog):
    # One shared grace period, never a per-worker extension; signal groups even if leaders exited.
    watchdog.terminate()
    until=min(watchdog.term_at+watchdog.grace,watchdog.hard_deadline)
    while time.monotonic()<until and any(group_alive(p.pid) for p in processes):
        for p in processes:p.poll()
        time.sleep(min(.1,max(0.,until-time.monotonic())))
    watchdog.kill()
    until=min(time.monotonic()+2.,watchdog.hard_deadline)
    while time.monotonic()<until and any(p.poll() is None for p in processes):time.sleep(.05)
    for p in processes:
        try:p.wait(timeout=.01)
        except subprocess.TimeoutExpired:pass  # D-state must not hold up report/output.
    return [p.pid for p in processes if p.poll() is None]

def launch(config,worker,smoke=False,heartbeat_seconds=30,watchdog=None,live_sync=None):
    working=Path(config['working_root']);working.mkdir(parents=True,exist_ok=True)
    watchdog=watchdog or Watchdog(config).start()
    config_path=working/'session-config.json';atomic_json(config_path,config)
    handles=[];processes=[];kind=config.get('session_kind','s1');report={};survivors=[]
    try:
        for gpu in (0,1):
            if time.monotonic()>=config['deadline_monotonic']:break
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu) if not smoke else '',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            for name in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'):env.pop(name,None)
            h=(working/f'gpu{gpu}-process.log').open('w');handles.append(h)
            cmd=[sys.executable,str(worker),'--gpu',str(gpu),'--config',str(config_path)]
            p=subprocess.Popen(cmd,env=env,stdout=h,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append(p);watchdog.register(p)
        while any(p.poll() is None for p in processes):
            remaining=config['deadline_monotonic']-time.monotonic()
            for gpu,p in enumerate(processes):
                print('GPU',gpu,'status',p.poll(),'deadline in',round(remaining,1),'s; last:',
                      ' | '.join(tail(working/f'gpu{gpu}-process.log'))[:1800],flush=True)
            atomic_json(working/'session-live.json',{'elapsed_seconds':time.monotonic()-config['session_start_monotonic'],
                'remaining_seconds':remaining,'processes':[p.poll() for p in processes]})
            disk_guard(working,limit=(4 if kind in ('s2','s3') else 18)*1024**3)
            if remaining<=0:break
            time.sleep(min(heartbeat_seconds,max(.01,remaining)))
    except BaseException as error:
        report['supervisor_error']={'type':type(error).__name__,'message':str(error)[:1000]}
    finally:
        survivors=stop_workers(processes,watchdog)
        for h in handles:h.close()
        report.update(status='complete' if len(processes)==2 and all(p.returncode==0 for p in processes) and not report else 'partial',
            exit_codes=[p.returncode for p in processes],unreaped_worker_pids=survivors,
            process_groups=[p.pid for p in processes],wall_seconds=time.monotonic()-config['session_start_monotonic'],
            budget_seconds=config['budget_seconds'],hard_deadline_monotonic=watchdog.hard_deadline,
            termination_grace_seconds=watchdog.grace,term_to_kill_seconds=watchdog.kill_at-watchdog.term_at,
            GPU_launch=not smoke,processes=len(processes))
        # Persist a useful session result before any potentially slow report/archive operation.
        atomic_json(working/'session-summary.json',report);atomic_json(working/f'{kind}-summary.json',report)
        try:
            if kind=='s2':
                from s2_report import collect
                report['comparison']=collect(working)
                if any(r['status']!='complete' for r in report['comparison']['models'].values()) or report['comparison']['decision']['status']!='measured':report['status']='partial'
            elif kind=='s3':
                report['experiments']={}
                for directory,experiment in [('gpu0-exp30',30),('gpu1-exp31',31),('gpu0-exp32',32)]:
                    path=working/directory/(str(experiment)+'-summary.json')
                    partial=json.loads(path.read_text()) if path.exists() else {}
                    report['experiments'][str(experiment)]={'status':partial.get('status','missing'),
                        'points':list(partial.get('points',{})), 'missing_points':partial.get('missing_points'),
                        'scored_candidates':sum(s.get('candidates',0) for s in partial.get('splits',{}).values())}
                if any(e['status']!='complete' for e in report['experiments'].values()):report['status']='partial'
            # Unpack/anchor scratch is outside working and never becomes a Kaggle output.
            if config.get('temp_root') and not survivors and live_sync is None:shutil.rmtree(config['temp_root'],ignore_errors=True)
            archive(working,kind)
            report['output_bytes']=disk_guard(working,limit=(4 if kind in ('s2','s3') else 18)*1024**3)
            report['output_policy']='individual files; no ZIP' if kind in ('s2','s3') else 'ZIP'
        except BaseException as error:
            report.update(status='partial',report_error={'type':type(error).__name__,'message':str(error)[:1000]})
        report['wall_seconds']=time.monotonic()-config['session_start_monotonic']
        atomic_json(working/'session-summary.json',report);atomic_json(working/f'{kind}-summary.json',report)
        if live_sync is not None:
            try:report['live_final_uploaded']=live_sync.final_sync('normal-exit',timeout=30.)
            except Exception:report['live_final_uploaded']=False
            atomic_json(working/'session-summary.json',report);atomic_json(working/f'{kind}-summary.json',report)
            # Live staging is outside output; leave it alone if an upload thread is unreaped.
        # Leave the hard fuse armed if SIGKILL did not reap a worker.
        if not survivors:watchdog.finish()
    return report
