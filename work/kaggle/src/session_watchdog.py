"""Torch-free notebook watchdog, armed before installs, input preparation and workers."""
import json,os,signal,subprocess,threading,time,uuid
from pathlib import Path

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex[:8])
    with tmp.open('w') as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())
    tmp.replace(path)

def clean_checkpoints(working):
    """Small, Torch-free cleanup also works if a kill interrupted receipt publication."""
    for root in Path(working).glob('gpu*-exp*'):
        keep=set();blobs=set();valid=True
        for selector in ('latest.json','final-checkpoint.json'):
            path=root/selector
            if not path.exists():continue
            try:
                receipt=json.loads(path.read_text());keep.add(receipt['file'])
                if receipt.get('tensor_blob'):blobs.add(receipt['tensor_blob']['relative_path'])
            except (OSError,ValueError,KeyError):valid=False
        for directory in ('checkpoints','weights'):
            for path in (root/directory).glob('*.pt.tmp'):path.unlink(missing_ok=True)
        if valid:
            for path in (root/'checkpoints').glob('*.pt'):
                if path.relative_to(root).as_posix() not in keep:
                    path.unlink(missing_ok=True);path.with_suffix('.pt.json').unlink(missing_ok=True)
            for path in (root/'checkpoints/blobs').glob('*.pt'):
                if 'blobs/'+path.name not in blobs:path.unlink(missing_ok=True)

class Watchdog:
    def __init__(self,config):
        self.config=config;self.working=Path(config['working_root'])
        self.deadline=config['deadline_monotonic']
        self.hard_deadline=min(config.get('hard_deadline_monotonic',self.deadline+600),
                               config['session_start_monotonic']+3*3600+40*60)
        self.grace=config.get('terminate_grace_seconds',20.)
        self.groups=set();self.lock=threading.Lock();self.done=threading.Event()
        self.term_at=None;self.kill_at=None;self.thread=None
        self.own_group=os.getpgrp()
        self.signal_errors=[]
        self.live_sync=None

    def register(self,process):
        # Popen(start_new_session=True): PID is the process group ID, including exec pools.
        with self.lock:self.groups.add(process.pid)
        if self.term_at is not None:self.signal_groups(signal.SIGTERM)
        if self.kill_at is not None:self.signal_groups(signal.SIGKILL)

    def signal_groups(self,sig,extra=()):
        with self.lock:groups=self.groups.copy()
        groups.update(extra)
        for group in groups:
            if group==self.own_group:continue
            try:os.killpg(group,sig)
            except ProcessLookupError:pass
            except OSError as error:self.signal_errors.append({'group':group,'signal':sig,'error':str(error)})

    def terminate(self):
        with self.lock:
            if self.term_at is None:self.term_at=time.monotonic()
        self.signal_groups(signal.SIGTERM)

    def kill(self):
        with self.lock:self.kill_at=time.monotonic()
        self.signal_groups(signal.SIGKILL)

    def descendants(self):
        # Also catch setup subprocesses or grandchildren that created a new session.
        rows=[]
        if Path('/proc').is_dir():
            for path in Path('/proc').glob('[0-9]*/stat'):
                try:
                    fields=path.read_text().rsplit(')',1)[1].split()
                    rows.append((int(path.parent.name),int(fields[1]),int(fields[2])))
                except (OSError,ValueError,IndexError):pass
        try:
            if not rows:
                result=subprocess.run(['ps','-eo','pid=,ppid=,pgid='],capture_output=True,text=True,
                                      timeout=1.,start_new_session=True,check=True)
                rows=[tuple(map(int,line.split())) for line in result.stdout.splitlines()]
        except (OSError,ValueError,subprocess.SubprocessError):pass
        parents={os.getpid()};children=set()
        while True:
            found={pid for pid,ppid,_ in rows if ppid in parents}-children
            if not found:break
            children.update(found);parents.update(found)
        return sorted(children),{group for pid,_,group in rows if pid in children}

    def force_exit(self):
        # Heavy snapshot/FP16 work is isolated in a child; final_sync has a hard wall timeout.
        report={'status':'partial; notebook watchdog forced exit','watchdog_forced_exit':True,
                'wall_seconds':time.monotonic()-self.config['session_start_monotonic'],
                'budget_seconds':self.config['budget_seconds'],'hard_deadline_monotonic':self.hard_deadline,
                'process_groups':sorted(self.groups),'output_policy':'individual files; no ZIP',
                'signal_errors':self.signal_errors,
                'partial_results':'per-experiment summaries and points/*/evaluation/metrics.json'}
        try:
            self.signal_groups(signal.SIGKILL)
            children,groups=self.descendants()
            self.signal_groups(signal.SIGKILL,groups)
            for pid in children:
                try:os.kill(pid,signal.SIGKILL)
                except OSError:pass
            try:clean_checkpoints(self.working)
            except OSError as error:report['cleanup_error']=str(error)
            atomic_json(self.working/'session-summary.json',report)
            atomic_json(self.working/(self.config.get('session_kind','s2')+'-summary.json'),report)
            if self.live_sync is not None:
                try:self.live_sync.final_sync('watchdog',timeout=min(20.,max(.1,self.hard_deadline-time.monotonic()-1.)))
                except Exception:pass  # A failed upload must never prevent emergency exit.
            print('WATCHDOG: saved session-summary; os._exit(0)',flush=True)
        finally:os._exit(0)

    def run(self):
        while not self.done.is_set():
            now=time.monotonic()
            # Reserve 25s inside the existing hard ceiling for the final best-effort upload.
            if now>=self.hard_deadline-(25. if self.live_sync is not None else 0.):self.force_exit()
            if now>=self.deadline and self.term_at is None:self.terminate()
            if self.term_at is not None and now>=self.term_at+self.grace and self.kill_at is None:self.kill()
            self.done.wait(.1)

    def start(self):
        if self.thread is not None:return self
        atomic_json(self.working/'session-summary.json',{'status':'partial; watchdog armed',
            'budget_seconds':self.config['budget_seconds'],'deadline_monotonic':self.deadline,
            'hard_deadline_monotonic':self.hard_deadline})
        self.thread=threading.Thread(target=self.run,name='notebook-watchdog',daemon=True)
        self.thread.start();return self

    def finish(self):
        self.done.set()
        if self.thread is not None:self.thread.join(timeout=1.)
