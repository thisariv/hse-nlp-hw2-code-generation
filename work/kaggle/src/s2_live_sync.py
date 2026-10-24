"""Best-effort private Dataset snapshots. Supervisor stays Torch-free and bounded."""
import argparse,gzip,json,os,shutil,signal,subprocess,sys,threading,time
from pathlib import Path
from session_watchdog import atomic_json

SECRET_NAMES=('KAGGLE_USERNAME','KAGGLE_KEY')
STAGE_LIMIT=450_000_000  # FP16 weights + compressed journals; reserve another copy for CLI ZIP.

def load_secrets():
    # Values live only in process memory/environment. Never return, log or serialize them.
    try:
        from kaggle_secrets import UserSecretsClient
        client=UserSecretsClient()
        for name in SECRET_NAMES:os.environ[name]=client.get_secret(name)
        return all(os.environ.get(name) for name in SECRET_NAMES)
    except Exception:
        print('LIVE: Kaggle Secrets unavailable; uploads disabled',flush=True)
        return False

def credential_patterns(ref):
    values=[os.environ.get(name,'').encode() for name in SECRET_NAMES]
    # The fixed public owner in Dataset IDs is not authentication data. All other
    # username values and every key byte are forbidden (including in binary files).
    owner=ref.split('/')[0].encode()
    return [v for i,v in enumerate(values) if v and not (i==0 and v==owner)]

def contains_secret(path,patterns):
    overlap=max([len(v) for v in patterns]+[1])-1;previous=b''
    with Path(path).open('rb') as handle:
        while True:
            part=handle.read(1024*1024)
            if not part:return False
            block=previous+part
            if any(value in block for value in patterns):return True
            previous=block[-overlap:] if overlap else b''

def audit_tree(root,ref,cache=None):
    patterns=credential_patterns(ref);cache={} if cache is None else cache
    for path in Path(root).rglob('*'):
        if not path.is_file():continue
        if path.is_symlink():raise RuntimeError('Symlink in snapshot source')
        # Active atomic temporaries are never uploaded; their published versions are audited.
        if '.tmp' in path.name:continue
        stat=path.stat();stamp=[stat.st_size,stat.st_mtime_ns]
        key=str(path.relative_to(root))
        if cache.get(key)!=stamp:
            if contains_secret(path,patterns):raise RuntimeError('Credential bytes detected; upload refused')
            cache[key]=stamp
    return cache

def stage_snapshot(working,stage,ref,reason):
    """Runs in a killable CPU child; no Torch or heavy IO on watchdog thread."""
    working,stage=Path(working),Path(stage);stage.mkdir(parents=True,exist_ok=True)
    audit_path=stage.parent/'live-audit-cache.json'
    cache=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    cache=audit_tree(working,ref,cache);atomic_json(audit_path,cache)
    payload=stage/'snapshot';payload.mkdir(exist_ok=True)
    old_path=stage.parent/'live-weight-cache.json'
    weights=json.loads(old_path.read_text()) if old_path.exists() else {}
    keep=set();manifest=[];used=0
    patterns=credential_patterns(ref)
    def reserve(size):
        nonlocal used
        if used+size>STAGE_LIMIT:raise RuntimeError('Live staging size limit')
        used+=size
    def copy_file(path):
        relative=path.relative_to(working);target=payload/relative
        if path.is_symlink():raise RuntimeError('Symlink in snapshot source')
        data=path.read_bytes()
        if any(v in data for v in patterns):raise RuntimeError('Credential bytes detected; upload refused')
        if path.suffix in ('.json','.jsonl') and len(data)>256*1024:
            data=gzip.compress(data,compresslevel=1,mtime=0);target=target.with_name(target.name+'.gz')
        reserve(len(data));target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        keep.add(target);manifest.append({'file':target.relative_to(stage).as_posix(),'bytes':len(data)})
    selected=set()
    for path in working.glob('*.json'):
        if path.name!='session-config.json':selected.add(path)
    for root in working.glob('gpu*-exp*'):
        selected.update(root.glob('*summary.json'));selected.update(root.glob('*gate.json'))
        selected.update(root.glob('progress.json'));selected.update(root.glob('training-finished.json'))
        for point in (root/'points').glob('*'):
            metric=point/'evaluation/metrics.json'
            if not metric.exists():continue
            selected.update(point.glob('training.json'))
            selected.update((point/'evaluation').rglob('*.json'))
            receipt_path=point/'checkpoint.json'
            if not receipt_path.exists():continue
            receipt=json.loads(receipt_path.read_text());source=root/receipt['file']
            if not source.resolve().is_relative_to(root.resolve()) or source.is_symlink():raise RuntimeError('Invalid point weight path')
            destination=payload/root.name/'weights'/f'{point.name}-fp16.pt'
            stat=source.stat();stamp=[stat.st_size,stat.st_mtime_ns];key=root.name+'/'+point.name
            if weights.get(key)!=stamp or not destination.exists():
                import torch
                torch.set_num_threads(1)
                original=torch.load(source,map_location='cpu',weights_only=True)
                model={k:(v.detach().cpu().half() if v.is_floating_point() else v.detach().cpu()) for k,v in original['model'].items()}
                estimate=sum(v.numel()*v.element_size() for v in model.values())+1024*1024
                if used+estimate>STAGE_LIMIT:raise RuntimeError('Live staging weight size limit')
                destination.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'format':'nlp-hw02-model-only-v1','model':model,'model_spec':original['model_spec'],
                    'experiment':original['experiment'],'point':point.name,'dtype':'FP16','optimizer_included':False,
                    'source_fp32_sha256':receipt['sha256'],'purpose':'live recovery; quantized evaluated weights'},destination)
                weights[key]=stamp
                del original,model
            reserve(destination.stat().st_size);keep.add(destination)
            manifest.append({'file':destination.relative_to(stage).as_posix(),'bytes':destination.stat().st_size,'dtype':'FP16','optimizer_included':False})
            target=payload/receipt_path.relative_to(working);target.parent.mkdir(parents=True,exist_ok=True)
            import hashlib
            digest=hashlib.sha256(destination.read_bytes()).hexdigest()
            atomic_json(target,{'file':destination.relative_to(payload/root.name).as_posix(),'sha256':digest,
                'bytes':destination.stat().st_size,'dtype':'FP16','optimizer_included':False,'source_fp32':receipt})
            reserve(target.stat().st_size);keep.add(target)
        # Heldout 27b candidate journals and complete test generations/answers.
        for folder in ('heldout4k','test'):
            for path in (root/folder).rglob('*'):
                if path.is_file() and path.suffix in ('.json','.jsonl'):selected.add(path)
    for path in (working/'a-test').rglob('*'):
        if path.is_file() and path.suffix in ('.json','.jsonl'):selected.add(path)
    for path in sorted(selected):copy_file(path)
    for path in working.glob('gpu*-process.log'):
        with path.open('rb') as handle:
            handle.seek(0,2);handle.seek(max(0,handle.tell()-32*1024));data=handle.read()
        for value in patterns:data=data.replace(value,b'[REDACTED]')
        target=payload/path.name;target.write_bytes(data);reserve(len(data));keep.add(target)
    for path in payload.rglob('*'):
        if path.is_file() and path not in keep:path.unlink()
    atomic_json(stage/'dataset-metadata.json',{'id':ref,'title':'NLP HW02 S2 live','licenses':[{'name':'other'}],
        'description':'Private incremental S2 recovery: evaluated FP16 model weights, metrics and generation journals.'})
    atomic_json(payload/'live-manifest.json',{'reason':reason,'unix_time':time.time(),'files':manifest,
        'payload_bytes':used,'staging_limit_bytes':STAGE_LIMIT,'credentials_in_files':False,
        'weights':'FP16 model only; optimizer excluded; FP32 evaluation remains in notebook output'})
    atomic_json(old_path,weights)
    audit_tree(stage,ref)
    assert sum(p.stat().st_size for p in stage.rglob('*') if p.is_file())<STAGE_LIMIT+64*1024

def privacy_probe(ref,path):
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();page=1
    while True:
        rows=api.dataset_list(mine=True,page=page) or []
        found=next((d for d in rows if d.ref==ref),None)
        if found:
            atomic_json(path,{'exists':True,'private':bool(found.is_private)});return
        if not rows:atomic_json(path,{'exists':False});return
        page+=1

class LiveSync:
    def __init__(self,working,scratch,ref,interval=900.,poll_seconds=5.,upload_timeout=540.):
        self.working=Path(working);self.scratch=Path(scratch)/'live-sync';self.scratch.mkdir(parents=True,exist_ok=True)
        self.stage=self.scratch/'stage';self.ref=ref;self.interval=interval;self.poll_seconds=poll_seconds
        self.upload_timeout=min(upload_timeout,540.);self.lock=threading.Lock();self.active_lock=threading.Lock()
        self.active=None;self.stop_event=threading.Event();self.thread=None;self.last_attempt=0.;self.last_signature=None
        self.env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        # No credential file is needed or written by CLI; only inherited env auth.
        config_dir=self.scratch/'empty-config';config_dir.mkdir(exist_ok=True)
        cli_tmp=self.scratch/'cli-temp';cli_tmp.mkdir(exist_ok=True)
        self.env.update(KAGGLE_CONFIG_DIR=str(config_dir),TMPDIR=str(cli_tmp))
        self.env.pop('KAGGLE_API_TOKEN',None)

    def log(self,event,**fields):
        # No subprocess output, exception text, argv, URLs or env in this log.
        record={'unix_time':time.time(),'event':event,**fields}
        with (self.working/'s2-live-sync.log').open('a') as handle:handle.write(json.dumps(record)+'\n')
        atomic_json(self.working/'s2-live-sync-status.json',record)

    def kill_active(self):
        with self.active_lock:process=self.active
        if process is not None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except OSError:pass

    def bounded_call(self,args,timeout,final=False):
        until=time.monotonic()+min(timeout,600.)
        process=subprocess.Popen(args,env=self.env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        with self.active_lock:self.active=process
        try:
            while process.poll() is None:
                if time.monotonic()>=until or (self.stop_event.is_set() and not final):
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except OSError:pass
                    # No communicate(), and no implicit unbounded wait in subprocess.run().
                    try:process.wait(timeout=.1)
                    except subprocess.TimeoutExpired:pass
                    raise subprocess.TimeoutExpired('live child',timeout)
                time.sleep(min(.05,max(.001,until-time.monotonic())))
            if process.returncode:raise subprocess.CalledProcessError(process.returncode,'live child')
        finally:
            with self.active_lock:
                if self.active is process:self.active=None

    def signature(self):
        paths=list(self.working.glob('gpu*-exp*/points/*/evaluation/metrics.json'))
        paths+=list(self.working.glob('gpu*-exp*/test/**/timing.json'))
        paths+=list(self.working.glob('gpu*-exp*/*summary.json'))
        return sorted((str(p.relative_to(self.working)),p.stat().st_size,p.stat().st_mtime_ns) for p in paths)

    def sync(self,reason,timeout=590.,final=False):
        until=time.monotonic()+min(timeout,600.);helper=str(Path(__file__).resolve())
        def call(args,cap):
            remaining=until-time.monotonic()
            if remaining<=0:raise subprocess.TimeoutExpired('live snapshot',timeout)
            return self.bounded_call(args,min(cap,remaining),final=final)
        if not self.lock.acquire(timeout=min(1.,timeout)):return False
        try:
            # Remove abandoned CLI archives after a timed-out upload, before another copy.
            for path in (self.scratch/'cli-temp').iterdir():
                if path.is_dir():shutil.rmtree(path)
                else:path.unlink()
            call([sys.executable,helper,'stage','--working',str(self.working),'--stage',str(self.stage),'--ref',self.ref,'--reason',reason],120.)
            privacy=self.scratch/'privacy.json';privacy.unlink(missing_ok=True)
            call([sys.executable,helper,'privacy','--ref',self.ref,'--receipt',str(privacy)],30.)
            info=json.loads(privacy.read_text())
            if info.get('exists') and not info.get('private'):raise RuntimeError('Public target refused')
            command=['kaggle','datasets','version' if info['exists'] else 'create','-p',str(self.stage),'--dir-mode','zip','--keep-tabular','--quiet']
            if info['exists']:command+=['-m','S2 incremental recovery '+reason]
            call(command,self.upload_timeout)
            self.log('uploaded',operation='version' if info['exists'] else 'create',reason=reason,
                staging_bytes=sum(p.stat().st_size for p in self.stage.rglob('*') if p.is_file()))
            return True
        except Exception as error:
            self.log('sync_error',error_type=type(error).__name__,returncode=getattr(error,'returncode',None),reason=reason)
            return False
        finally:self.lock.release()

    def run(self):
        while not self.stop_event.is_set():
            try:
                signature=self.signature();now=time.monotonic()
                if signature!=self.last_signature or now-self.last_attempt>=self.interval:
                    before=signature;self.last_attempt=now
                    if self.sync('new-point' if signature else 'heartbeat'):
                        # Changes published during upload trigger the next snapshot too.
                        self.last_signature=before
                    else:self.last_signature=before  # Failed upload retries at heartbeat/new metric.
            except Exception as error:self.log('thread_error',error_type=type(error).__name__)
            self.stop_event.wait(self.poll_seconds)

    def start(self):
        self.thread=threading.Thread(target=self.run,name='s2-live-sync',daemon=True);self.thread.start();return self

    def final_sync(self,reason='final',timeout=30.):
        self.stop_event.set();self.kill_active();until=time.monotonic()+timeout
        if self.thread is not None:self.thread.join(timeout=min(.5,timeout))
        # IO/upload on a daemon; even a wedged filesystem or child cannot hold watchdog exit.
        result=[]
        thread=threading.Thread(target=lambda:result.append(self.sync(reason,max(.1,until-time.monotonic()-.2),final=True)),daemon=True)
        thread.start();thread.join(timeout=max(0.,until-time.monotonic()))
        self.kill_active()
        return bool(result and result[0])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['stage','privacy'])
    for name in ('working','stage','ref','receipt','reason'):parser.add_argument('--'+name)
    args=parser.parse_args()
    try:
        if args.mode=='stage':stage_snapshot(args.working,args.stage,args.ref,args.reason)
        else:privacy_probe(args.ref,args.receipt)
    except Exception:sys.exit(1)  # No credential-bearing exception text/traceback to disk.
