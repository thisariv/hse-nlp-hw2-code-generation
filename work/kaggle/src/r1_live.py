"""Private live insurance: latest full states + latest evaluated points, flat <=2.5GB."""
import argparse, gzip, json, os, shutil, sys, time
from pathlib import Path
from kaggle_io import sha, write, safe_name
from s2_live_sync import LiveSync, audit_tree, credential_patterns, privacy_probe

STAGE_LIMIT=2500000000
SECRET_NAMES=('KAGGLE_USERNAME','KAGGLE_KEY')

def gate(torch_module, secret_getter=None,require_secrets=True):
    """Secrets are fetched only into env; a missing secret always aborts before install/data."""
    if not require_secrets:
        if torch_module.cuda.device_count()!=2 or not all('T4' in torch_module.cuda.get_device_name(i) for i in (0,1)):
            return {'passed':False,'reason':'GPU smoke requires exactly two T4 GPUs'}
        return {'passed':True,'reason':'GPU smoke T4x2 gate passed; Secrets/live not required'}
    missing=[]
    try:
        if secret_getter is None:
            from kaggle_secrets import UserSecretsClient
            secret_getter=UserSecretsClient().get_secret
        for name in SECRET_NAMES:
            try:
                value=secret_getter(name)
                if not value:missing.append(name)
                else:os.environ[name]=value
            except Exception:missing.append(name)
    except Exception:missing=list(SECRET_NAMES)
    if missing:return {'passed':False,'reason':'Kaggle Secrets unavailable: '+', '.join(missing),
                       'action':'Attach both Secrets and use Edit -> Save Version -> Save & Run All'}
    if torch_module.cuda.device_count()!=2:return {'passed':False,'reason':'Requires exactly two T4 GPUs'}
    if not all('T4' in torch_module.cuda.get_device_name(i) for i in (0,1)):
        return {'passed':False,'reason':'Both GPU devices must be T4'}
    return {'passed':True,'reason':'Kaggle Secrets and T4x2 gate passed'}

def optional_gate(torch_module, owner, smoke=False, secret_getter=None):
    """Compute is independent of live insurance; only a complete owner key enables it."""
    gpu = gate(torch_module, require_secrets=False)
    if not gpu['passed']: return gpu
    if smoke: return dict(gpu, live_enabled=False)
    for name in SECRET_NAMES: os.environ.pop(name, None)
    values = {}
    try:
        if secret_getter is None:
            from kaggle_secrets import UserSecretsClient
            secret_getter = UserSecretsClient().get_secret
        for name in SECRET_NAMES:
            try: values[name] = secret_getter(name)
            except Exception: pass
    except Exception: pass
    enabled = all(values.get(n) for n in SECRET_NAMES) and values['KAGGLE_USERNAME'] == owner
    if enabled: os.environ.update({n:values[n] for n in SECRET_NAMES})
    return dict(passed=True, live_enabled=bool(enabled),
                reason='Owner Secrets enable private live-sync' if enabled else
                       'Owner Secrets unavailable; local stable checkpoints and watchdog enabled')

def stage_snapshot(working,stage,ref,reason,limit=STAGE_LIMIT):
    working,stage=Path(working),Path(stage);stage.mkdir(parents=True,exist_ok=True)
    # Includes sources, binaries and nonselected files, as in S2v2. Secrets never go to logs/config.
    cache_path=stage.parent/'audit-cache.json'
    cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache=audit_tree(working,ref,cache);write(cache_path,cache)
    for p in stage.iterdir():
        if p.is_dir():shutil.rmtree(p)
        else:p.unlink()
    patterns=credential_patterns(ref);used=0;files=[];selected=[]
    for root in sorted(working.glob('model-*')):
        if not root.is_dir():continue
        selected.extend(root.glob('*.json'))
        latest=root/'latest.json'
        if latest.exists():
            rec=json.loads(latest.read_text());weight=root/safe_name(rec['file'])
            if not weight.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe state path')
            selected.append(weight)
        # Older point weights and generations remain in earlier immutable Dataset versions.
        complete=sorted((root/'points').glob('*/checkpoint.json'))
        if complete:
            point=complete[-1].parent;selected.extend(p for p in point.rglob('*') if p.is_file())
            rec=json.loads((point/'checkpoint.json').read_text());selected.append(root/safe_name(rec['file']))
    selected.extend(p for p in working.glob('*.json') if p.name!='session-config.json')
    # L3 reuses the exact weighted permutation, rather than rebuilding it under a new NumPy.
    selected.extend(p for p in (working/'weighted-order').glob('*') if p.is_file())
    selected.extend(p for folder in working.glob('order-gpu*') for p in folder.glob('*') if p.is_file())
    for p in sorted(set(selected)):
        if p.is_symlink() or '.tmp' in p.name:raise ValueError('Unsafe staging source')
        relative=p.relative_to(working).as_posix();flat=relative.replace('/','__');target=stage/flat
        if p.suffix in ('.json','.jsonl'):
            data=p.read_bytes()
            if any(v in data for v in patterns):raise ValueError('Secret detected')
            if len(data)>128*1024:data=gzip.compress(data,compresslevel=1,mtime=0);flat+='.gz';target=stage/flat
            size=len(data)
            if used+size>limit-1024**2:raise ValueError('Live staging size limit')
            target.write_bytes(data)
        else:
            size=p.stat().st_size
            if used+size>limit-1024**2:raise ValueError('Live staging size limit')
            shutil.copyfile(p,target)
        used+=size;files.append({'file':flat,'original_path':relative,'bytes':size,'sha256':sha(target),
                                 'gzip':flat.endswith('.gz')})
    # A concurrent atomic checkpoint replacement must not produce a false resume receipt.
    by_original={r['original_path']:r for r in files}
    for root in working.glob('model-*'):
        entry=by_original.get(root.name+'/latest.json')
        if not entry:continue
        data=(stage/entry['file']).read_bytes();rec=json.loads(gzip.decompress(data) if entry['gzip'] else data)
        item=by_original.get(root.name+'/'+rec['file'])
        assert item and item['sha256']==rec['sha256'] and item['bytes']==rec['bytes']
    write(stage/'recover-manifest.json',{'format':'nlp-hw02-l2-live-v1','reason':reason,'unix_time':time.time(),
          'files':files,'payload_bytes':used,'staging_limit_bytes':limit,'credentials_in_files':False,
          'point_policy':'latest cooled point (including partial evaluation) per model; each preceding point retained in its uploaded Dataset version'})
    write(stage/'dataset-metadata.json',{'id':ref,'title':'NLP HW02 L2 live','licenses':[{'name':'other'}],
          'description':'Private training insurance: full main model/AdamW/scaler/RNG/cursor and evaluated FP16 points.'})
    audit_tree(stage,ref)
    assert sum(p.stat().st_size for p in stage.iterdir() if p.is_file())<=limit

def restore_snapshot(source,destination):
    source,destination=Path(source),Path(destination)
    manifest=json.loads((source/'recover-manifest.json').read_text())
    assert manifest['format']=='nlp-hw02-l2-live-v1'
    for rec in manifest['files']:
        p=source/safe_name(rec['file']);assert sha(p)==rec['sha256'] and p.stat().st_size==rec['bytes']
        target=destination/safe_name(rec['original_path']);target.parent.mkdir(parents=True,exist_ok=True)
        if rec['gzip']:target.write_bytes(gzip.decompress(p.read_bytes()))
        else:shutil.copyfile(p,target)
    return destination

class R1LiveSync(LiveSync):
    def __init__(self,working,scratch,ref,interval=5400.,poll_seconds=10.,upload_timeout=600.):
        super().__init__(working,scratch,ref,interval,poll_seconds,upload_timeout)
        # Helpers authenticate with the explicitly attached API key only, never platform JWTs.
        for name in ('KAGGLE_USER_SECRETS_TOKEN','KAGGLE_IAP_TOKEN'):self.env.pop(name,None)
        self.upload_timeout=min(upload_timeout,600.);self.last_success=time.monotonic();self.failures=0
        self.uninsured=False;self.last_attempt=0.
    def log(self,event,**fields):
        record={'unix_time':time.time(),'event':event,**fields}
        with (self.working/'l2-live-sync.log').open('a') as f:f.write(json.dumps(record)+'\n')
        write(self.working/'l2-live-sync-status.json',record)
    def signature(self):
        return sorted((str(p.relative_to(self.working)),p.stat().st_mtime_ns)
                      for p in self.working.glob('model-*/points/*/complete.json'))
    def sync(self,reason,timeout=900.,final=False,wait_ready=False):
        until=time.monotonic()+min(timeout,900.);helper=str(Path(__file__).resolve())
        def call(args,cap):
            remaining=until-time.monotonic()
            if remaining<=0:raise TimeoutError('Live deadline')
            self.bounded_call(args,min(remaining,cap),final=final)
        if not self.lock.acquire(timeout=min(1.,timeout)):return False
        try:
            for p in (self.scratch/'cli-temp').iterdir():
                if p.is_dir():shutil.rmtree(p)
                else:p.unlink()
            call([sys.executable,helper,'stage','--working',str(self.working),'--stage',str(self.stage),
                  '--ref',self.ref,'--reason',reason],240.)
            privacy=self.scratch/'privacy.json';privacy.unlink(missing_ok=True)
            call([sys.executable,helper,'privacy','--ref',self.ref,'--receipt',str(privacy)],60.)
            info=json.loads(privacy.read_text())
            if info.get('exists') and not info.get('private'):raise ValueError('Public Dataset refused')
            command=['kaggle','datasets','version' if info['exists'] else 'create','-p',str(self.stage),
                     '--dir-mode','skip','--keep-tabular','--quiet']
            if info['exists']:command+=['-m','L2 recovery '+reason]
            call(command,self.upload_timeout)
            if wait_ready:
                # Re-probe owner-private server identity, not merely a successful local CLI exit.
                ready=False
                while time.monotonic()<until:
                    privacy.unlink(missing_ok=True)
                    call([sys.executable,helper,'privacy','--ref',self.ref,'--receipt',str(privacy)],30.)
                    current=json.loads(privacy.read_text())
                    if current.get('exists'):
                        if not current.get('private'):raise ValueError('Server privacy check failed')
                        ready=True;break
                    time.sleep(min(5.,max(0.,until-time.monotonic())))
                if not ready:raise TimeoutError('Dataset not visible after startup upload')
            # Successful upload is insured only after Dataset processing reports ready.
            status=self.scratch/'ready.json';ready=False
            while time.monotonic()<until:
                status.unlink(missing_ok=True)
                call([sys.executable,helper,'ready','--ref',self.ref,'--receipt',str(status)],30.)
                state=json.loads(status.read_text())['status']
                if state=='ready':ready=True;break
                if state=='error':raise RuntimeError('Dataset processing failed')
                time.sleep(min(10.,max(0.,until-time.monotonic())))
            if not ready:raise TimeoutError('Dataset processing deadline')
            self.last_success=time.monotonic();self.failures=0
            self.log('uploaded',reason=reason,operation='version' if info['exists'] else 'create',
                     staging_bytes=sum(p.stat().st_size for p in self.stage.iterdir() if p.is_file()))
            return True
        except Exception as e:
            self.failures+=1
            self.log('sync_error',reason=reason,error_type=type(e).__name__,returncode=getattr(e,'returncode',None))
            return False
        finally:self.lock.release()
    def run(self):
        while not self.stop_event.is_set():
            try:
                now=time.monotonic();signature=self.signature()
                new=signature!=self.last_signature
                due=now-self.last_success>=self.interval
                eligible=now-self.last_attempt>=(300. if self.failures else 30.)
                if (new or due or self.failures) and eligible:
                    self.last_attempt=now
                    if self.sync('new-point' if new else 'full-state-heartbeat'):self.last_signature=signature
                    self.last_attempt=time.monotonic()
                if time.monotonic()-self.last_success>=7200.:
                    self.uninsured=True;self.log('insurance_expired',age_seconds=time.monotonic()-self.last_success)
                    return
            except Exception as e:self.log('thread_error',error_type=type(e).__name__)
            self.stop_event.wait(self.poll_seconds)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['stage','privacy','ready','restore'])
    for name in ('working','stage','ref','receipt','reason','destination'):p.add_argument('--'+name)
    a=p.parse_args()
    try:
        if a.mode=='stage':stage_snapshot(a.working,a.stage,a.ref,a.reason)
        elif a.mode=='privacy':privacy_probe(a.ref,a.receipt)
        elif a.mode=='ready':
            from kaggle.api.kaggle_api_extended import KaggleApi
            api=KaggleApi();api.authenticate();write(a.receipt,{'status':api.dataset_status(a.ref)})
        else:restore_snapshot(a.stage,a.destination)
    except Exception:sys.exit(1)
