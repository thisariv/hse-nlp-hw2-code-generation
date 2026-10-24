#!/usr/bin/env python3
"""CPU-only live snapshots, mocked private CLI, real bounded hung-upload children."""
import ctypes,gzip,json,os,signal,subprocess,sys,tempfile,time,types
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import s2_live_sync as L
from session_watchdog import Watchdog,atomic_json

def main():
    import torch
    results={};start=time.monotonic();ref='thisarivson/nlp-hw02-s2-live'
    with tempfile.TemporaryDirectory(prefix='s2-live-cpu-') as temporary:
        base=Path(temporary);working=base/'working';working.mkdir()
        fake={'KAGGLE_USERNAME':'mock-secret-user-936f','KAGGLE_KEY':'mock-secret-key-78bdb199-not-real'}
        class Secrets:
            def get_secret(self,name):return fake[name]
        with patch.dict(os.environ,fake),patch.dict(sys.modules,{'kaggle_secrets':types.SimpleNamespace(UserSecretsClient=Secrets)}):
            assert L.load_secrets()
            atomic_json(working/'session-summary.json',{'status':'partial'})
            for directory,experiment in [('gpu0-exp29',29),('gpu1-exp28',28)]:
                root=working/directory;point=root/'points/t01';weights=root/'weights/t01.pt';weights.parent.mkdir(parents=True)
                torch.save({'model':{'test':torch.tensor([1.123456,2.,3.])},'model_spec':{'fixture':True},
                    'experiment':experiment,'optimizer':{'deliberately_excluded':True}},weights)
                atomic_json(point/'checkpoint.json',{'file':'weights/t01.pt','sha256':'CPU fixture'})
                atomic_json(point/'evaluation/metrics.json',{'splits':{'heldout4k':{},'old1024':{}},'fixture':True})
                atomic_json(point/'training.json',{'point':'t01'})
                atomic_json(point/'evaluation/heldout4k-generations.json',[{'code':'print(2)'}]*25000)
                atomic_json(root/'test/t01/selected/timing.json',{'seconds':1.})
                (root/'test/t01/selected/test-answers.jsonl').write_text('{"id":0,"code":"print(2)"}\n')
            (working/'gpu0-process.log').write_text('worker log\n'*10000)
            live=L.LiveSync(working,base/'scratch',ref)
            calls=[];exists=False;actual=live.bounded_call
            def mocked(args,timeout,final=False):
                nonlocal exists
                assert timeout<=600 and all(v not in str(args) for v in fake.values())
                if args[0]=='kaggle':
                    calls.append(args);assert '--public' not in args
                    assert args[args.index('--dir-mode')+1]=='zip';exists=True
                elif 'privacy' in args:
                    atomic_json(Path(args[args.index('--receipt')+1]),{'exists':exists,**({'private':True} if exists else {})})
                else:actual(args,timeout,final=final)
            with patch.object(live,'bounded_call',side_effect=mocked):
                assert live.sync('first') and live.sync('second')
                assert [command[2] for command in calls]==['create','version']
                assert live.final_sync(timeout=10.)
            for path in live.stage.glob('snapshot/gpu*-exp*/weights/*.pt'):
                payload=torch.load(path,map_location='cpu',weights_only=True)
                assert payload['dtype']=='FP16' and not payload['optimizer_included'] and 'optimizer' not in payload
                assert payload['model']['test'].dtype==torch.float16
                assert torch.equal(payload['model']['test'],torch.tensor([1.123456,2.,3.]).half())
            compressed=next(live.stage.rglob('*generations.json.gz'))
            assert len(json.loads(gzip.decompress(compressed.read_bytes())))==25000
            for root in (working,live.scratch):
                for path in root.rglob('*'):
                    if path.is_file():assert not L.contains_secret(path,[v.encode() for v in fake.values()])
            size=sum(p.stat().st_size for p in live.stage.rglob('*') if p.is_file());assert size<L.STAGE_LIMIT
            results.update(mock_create_then_version_private_zip=True,FP16_weights_no_optimizer=True,
                compressed_generations_and_test_answers=True,normal_final_sync=True,credential_file_scan=True,staging_bytes=size)
            # Fail closed for credentials even in binary files and across scan-block boundaries.
            polluted=working/'bad.bin';polluted.write_bytes(b'x'*(1024*1024-3)+fake['KAGGLE_KEY'].encode())
            try:L.stage_snapshot(working,base/'bad-stage',ref,'injected-secret')
            except RuntimeError:pass
            else:raise AssertionError('Secret file was accepted')
            polluted.unlink();results['injected_secret_upload_refused']=True
            public=L.LiveSync(working,base/'public',ref);public_calls=[]
            def public_mock(args,timeout,final=False):
                if 'privacy' in args:atomic_json(Path(args[args.index('--receipt')+1]),{'exists':True,'private':False})
                elif args[0]=='kaggle':public_calls.append(args)
            with patch.object(public,'bounded_call',side_effect=public_mock):assert not public.sync('public')
            assert not public_calls;results['public_target_refused']=True
            # Real upload-like process forks a grandchild, ignores TERM, emits credentials;
            # both output streams are DEVNULL, and timeout kills the whole process group.
            hanging=L.LiveSync(working,base/'hung',ref);pidfile=base/'hung-pids.json'
            script="import json,os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); pid=os.fork(); "
            script+="open(os.environ['PIDFILE'],'w').write(json.dumps({'pid':os.getpid(),'group':os.getpgrp()})) if pid else None; "
            script+="print(os.environ['KAGGLE_KEY'],flush=True); time.sleep(3600)"
            hanging.env['PIDFILE']=str(pidfile);began=time.monotonic()
            try:hanging.bounded_call([sys.executable,'-c',script],.4)
            except subprocess.TimeoutExpired:pass
            else:raise AssertionError('Upload did not time out')
            assert time.monotonic()-began<1.2
            group=json.loads(pidfile.read_text())['group']
            try:
                process_list=subprocess.run(['ps','-eo','pgid=,stat='],capture_output=True,text=True,timeout=2.)
                assert not any(int(line.split()[0])==group and not line.split()[1].startswith('Z') for line in process_list.stdout.splitlines())
            except PermissionError:
                # macOS sandbox denies ps. Let launchd reap the orphan and check its group.
                alive=True
                for _ in range(20):
                    try:os.killpg(group,0)
                    except ProcessLookupError:alive=False;break
                    time.sleep(.05)
                assert not alive
            results['hung_upload_group_killed']=True
            # final_sync itself stays bounded if even sync() is wedged in native IO.
            wedged=L.LiveSync(working,base/'wedged',ref)
            def blocked(*a,**kw):ctypes.CDLL(None).sleep(3600)
            with patch.object(wedged,'sync',side_effect=blocked):
                began=time.monotonic();assert not wedged.final_sync(timeout=.25)
                assert time.monotonic()-began<.8
            results['wedged_final_sync_bounded']=True
            # Actual watchdog process exits despite a hanging final upload.
            fixture=base/'watchdog';fixture.mkdir();began=time.monotonic()
            child="from pathlib import Path; import time,ctypes; from s2_live_sync import LiveSync; from session_watchdog import Watchdog; "
            child+="start=time.monotonic(); cfg={'working_root':%r,'session_start_monotonic':start,'deadline_monotonic':start+.2,'hard_deadline_monotonic':start+1.5,'budget_seconds':1.,'terminate_grace_seconds':.1}; " % str(fixture)
            child+="live=LiveSync(cfg['working_root'],%r,%r); live.sync=lambda *a,**kw:ctypes.CDLL(None).sleep(3600); " % (str(base/'fuse'),ref)
            child+="fuse=Watchdog(cfg); fuse.live_sync=live; fuse.start(); time.sleep(3600)"
            environment=dict(os.environ,PYTHONPATH=str(ROOT/'src'))
            run=subprocess.run([sys.executable,'-c',child],env=environment,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=4.)
            assert run.returncode==0 and time.monotonic()-began<3.
            assert json.loads((fixture/'session-summary.json').read_text())['watchdog_forced_exit']
            results['watchdog_exit_with_hung_final_upload']=True
            # New metrics during an upload remain visible, plus a heartbeat without new points.
            periodic=L.LiveSync(working,base/'periodic',ref,interval=.25,poll_seconds=.02)
            events=[]
            def record(reason,**kwargs):
                events.append((reason,time.monotonic()));time.sleep(.03);return True
            with patch.object(periodic,'sync',side_effect=record):
                periodic.start();time.sleep(.08)
                atomic_json(working/'gpu0-exp29/points/t02/evaluation/metrics.json',{'new':True})
                time.sleep(.35);periodic.stop_event.set();periodic.thread.join(timeout=1.)
            assert len(events)>=3 and events[1][1]-events[0][1]<.25
            results['new_metric_and_periodic_upload_triggers']=True
    results.update(status='passed',wall_seconds=time.monotonic()-start,network_calls=0,GPU_runs=0,
        unverified=['Kaggle attached Secrets access in committed run','real Dataset create/version, processing latency and final upload speed',
            'full-size staging/output and CUDA/driver D-state'])
    atomic_json(ROOT.parent/'checks/kaggle-s2-live-cpu-checks.json',results)
    print('PASSED live CPU: private create/version ZIP, FP16, no credentials, timed-out upload tree and watchdog exit')

if __name__=='__main__':main()
