#!/usr/bin/env python3
"""CPU-only subprocess fixtures: TERM-resistant process trees, blocked report and setup."""
import argparse,ctypes,json,multiprocessing as mp,os,signal,subprocess,sys,tempfile,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from session_watchdog import Watchdog,atomic_json

def hang(kind):
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    if kind=='c-sleep':
        ctypes.CDLL(None).sleep(3600)
    else:
        while True:pass

def pool_child(working,gpu):
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    atomic_json(Path(working)/f'grandchild-{gpu}-{os.getpid()}.json',{'pid':os.getpid(),'pgid':os.getpgrp()})

def worker(args):
    cfg=json.loads(args.config.read_text());working=Path(cfg['working_root'])
    signal.signal(signal.SIGTERM,signal.SIG_IGN)
    pool=mp.get_context('spawn').Pool(2,initializer=pool_child,initargs=(str(working),args.gpu))
    for kind in ('c-sleep','loop'):pool.apply_async(hang,(kind,))
    atomic_json(working/f'worker-{args.gpu}.json',{'pid':os.getpid(),'pgid':os.getpgrp(),
        'children':[p.pid for p in pool._pool],'ignores_SIGTERM':True})
    point=working/('gpu0-exp29' if args.gpu==0 else 'gpu1-exp28')/'points/t01'
    atomic_json(point/'evaluation/metrics.json',{'CPU_fixture_only':True,'status':'complete'})
    print('CPU fixture: ignoring TERM, grandchildren active, point metric saved',flush=True)
    hang('c-sleep' if args.gpu==0 else 'loop')

def notebook(args):
    import session_runner as runner
    working=Path(args.output);start=time.monotonic()
    cfg={'working_root':str(working),'session_kind':'s2','session_start_monotonic':start,
         'deadline_monotonic':start+4.,'hard_deadline_monotonic':start+29.,
         'budget_seconds':4.,'terminate_grace_seconds':20.}
    # Avoid importing Torch in supervisor fixtures; real reporting is tested in s2_smoke.py.
    def collect(root):
        if args.case=='blocked-report':ctypes.CDLL(None).sleep(3600)
        return {'models':{'M':{'status':'partial'},'B':{'status':'partial'}},
                'decision':{'status':'pending; CPU fixture'}}
    sys.modules['s2_report']=types.SimpleNamespace(collect=collect)
    if args.case=='unreapable':
        actual=subprocess.Popen
        class Unreapable:
            def __init__(self,*a,**kw):self.real=actual(*a,**kw);self.pid=self.real.pid;self.returncode=None
            def poll(self):self.real.poll();return None
            def wait(self,timeout=None):
                assert timeout is not None,'An unbounded wait would hang the notebook'
                raise subprocess.TimeoutExpired('CPU unreapable proxy',timeout)
        # Real children are killed; only the Python leader interface simulates D-state.
        runner.subprocess=types.SimpleNamespace(Popen=Unreapable,TimeoutExpired=subprocess.TimeoutExpired,STDOUT=subprocess.STDOUT)
    if args.case=='blocked-setup':
        import build
        boot_cfg=dict(cfg,budget_seconds=11700)
        nb=build.notebook('s2',boot_cfg)
        first=next(cell['source'] for cell in nb['cells'] if cell['cell_type']=='code')
        namespace={};exec(first,namespace)
        fuse=namespace['WATCHDOG'];fuse.deadline=start+4.;fuse.hard_deadline=start+8.;fuse.grace=1.
        config_path=working/'setup-config.json';atomic_json(config_path,cfg)
        setup=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--gpu','0','--config',str(config_path)],
                               start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        fuse.register(setup)  # same registration as the notebook dependency installer
        atomic_json(working/'setup-marker.json',{'watchdog_armed_before_install':True,'CPU_fixture_only':True})
        ctypes.CDLL(None).sleep(3600)
        raise AssertionError('Watchdog did not exit blocked first-cell setup')
    fuse=Watchdog(cfg).start()
    report=runner.launch(cfg,Path(__file__).resolve(),smoke=True,heartbeat_seconds=.5,watchdog=fuse)
    atomic_json(working/'launch-returned.json',report)
    if args.case=='unreapable':ctypes.CDLL(None).sleep(3600)

def live_pids(pids):
    if not pids:return []
    try:
        result=subprocess.run(['ps','-p',','.join(map(str,pids)),'-o','pid=,stat='],capture_output=True,text=True,timeout=2.)
        return [int(line.split()[0]) for line in result.stdout.splitlines() if not line.split()[1].startswith('Z')]
    except OSError:
        alive=[]
        for pid in pids:
            try:os.kill(pid,0);alive.append(pid)
            except ProcessLookupError:pass
        return alive

def main():
    tmp=Path(tempfile.mkdtemp(prefix='nlp-s2-watchdog-smoke-'));checks=[]
    for case in ('group-kill','blocked-report','unreapable','blocked-setup'):
        working=tmp/case;working.mkdir();log=working/'notebook.log';start=time.monotonic()
        with log.open('w') as handle:
            notebook_process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--case',case,'--output',str(working)],
                                              stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=notebook_process.wait(timeout=40.)
            except subprocess.TimeoutExpired:
                os.killpg(notebook_process.pid,signal.SIGKILL)
                notebook_process.wait(timeout=2.)
                raise AssertionError('Notebook exceeded accelerated hard ceiling: '+case)
        elapsed=time.monotonic()-start
        assert code==0,(case,code,log.read_text()[-4000:])
        assert elapsed<35.,(case,elapsed)
        summary=json.loads((working/'session-summary.json').read_text())
        trees=[json.loads(path.read_text()) for path in working.glob('worker-*.json')]
        assert len(trees)==(1 if case=='blocked-setup' else 2)
        pids=[pid for tree in trees for pid in [tree['pid'],*tree['children']]]
        for tree in trees:
            assert tree['pgid']==tree['pid']  # isolated group leader
            for path in working.glob('grandchild-*.json'):
                child=json.loads(path.read_text())
                if child['pid'] in tree['children']:assert child['pgid']==tree['pgid']
        for _ in range(20):
            alive=live_pids(pids)
            if not alive:break
            time.sleep(.1)
        assert not alive,(case,alive)
        assert len(list(working.glob('grandchild-*.json')))>=2*len(trees)
        assert (working/'gpu0-exp29/points/t01/evaluation/metrics.json').exists()
        assert not list(working.glob('*.zip'))
        if case=='group-kill':
            assert summary['exit_codes']==[-signal.SIGKILL,-signal.SIGKILL]
            assert 19.8<=summary['term_to_kill_seconds']<=21.
            assert (working/'s2-output-manifest.json').exists()
            assert 'deadline in' in log.read_text()
        else:
            assert summary['watchdog_forced_exit'] is True
            if case=='unreapable':
                returned=json.loads((working/'launch-returned.json').read_text())
                assert returned['unreaped_worker_pids'] and returned['output_policy']=='individual files; no ZIP'
                assert (working/'s2-output-manifest.json').exists()
            if case=='blocked-setup':assert (working/'setup-marker.json').exists()
        checks.append({'case':case,'passed':True,'wall_seconds':elapsed,'notebook_exit_code':code,
                       'workers_and_grandchildren':len(pids),'running_survivors':alive,
                       'partial_metric_preserved':True,'output_bytes':sum(p.stat().st_size for p in working.rglob('*') if p.is_file())})
        print('PASSED',case,round(elapsed,2),'seconds',flush=True)
    atomic_json(ROOT.parent/'checks/kaggle-s2-watchdog-cpu-checks.json',{'status':'passed','checks':checks,
        'production_budget_seconds':11700,'production_worker_deadline_seconds':11400,
        'production_watchdog_exit_seconds':12000,'session_ceiling_seconds':13200,
        'GPU_runs':0,'unverified':['Real CUDA/driver D-state and Kaggle output collection after os._exit(0)'],
        'fixtures':'TERM ignored; libc sleep / busy loop; multiprocessing spawn pool grandchildren; unreapable leader proxy; blocked report; actual first-cell watchdog'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int);parser.add_argument('--config',type=Path)
    parser.add_argument('--case');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.gpu is not None:worker(args)
    elif args.case:notebook(args)
    else:main()
