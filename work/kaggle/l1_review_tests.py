#!/usr/bin/env python3
"""Independent regressions for L1 failure paths; no GPU or real credentials required."""
import copy, json, os, shutil, subprocess, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import numpy as np, torch
import l1_runtime as L, l1_live as Live, l1_eval as E, exp26_train as T
from kaggle_io import sha,write
from l1_resume import select_resume
from l1_session import decision
SPEC=dict(d_model=8,heads=2,encoder_layers=1,decoder_layers=1,swiglu_dim=16,vocab=32,max_positions=2048)
CPU=torch.device('cpu')
class Cache:
    manifest_sha='tiny-regression'
    def __init__(self):
        self.rows=[([1,3+i%8,2],[4+i%8]*(1+i%5)) for i in range(137)]
        self.source_lengths=np.array([len(s) for s,t in self.rows]);self.target_lengths=np.array([len(t) for s,t in self.rows])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[int(i)]

class Regressions(unittest.TestCase):
    def setUp(self):
        self.tmp=Path(tempfile.mkdtemp(prefix='l1-review-'));self.cache=Cache();self.order=np.arange(len(self.cache),dtype=np.uint32)
    def tearDown(self):shutil.rmtree(self.tmp)
    def run_model(self,root,updates=4,**kw):
        args=dict(smoke=True,spec=SPEC,batch=8,warmup=1000,point_seconds=10000.,checkpoint_seconds=10000.,max_updates=updates)
        args.update(kw)
        return L.run('C',self.cache,self.order,'tiny-order',self.tmp/root,self.tmp/(root+'-scratch'),CPU,time.monotonic()+30,
                     args.pop('evaluate',lambda *a:None),**args)
    def payload(self,root):
        root=self.tmp/root;r=json.loads((root/'latest.json').read_text())
        return L.read_full(root/r['file'],json.loads((root/'identity.json').read_text()))
    def assert_tensors(self,a,b):
        for key,val in a['model'].items():self.assertTrue(torch.equal(val,b['model'][key]),key)
        for key,state in a['optimizer']['state'].items():
            for k,v in state.items():self.assertTrue(torch.equal(v,b['optimizer']['state'][key][k]),str((key,k)))
        self.assertEqual(a['scaler'],b['scaler']);self.assertEqual(a['rng']['python'],b['rng']['python'])
        self.assertEqual(a['rng']['numpy'],b['rng']['numpy']);self.assertTrue(torch.equal(a['rng']['torch_cpu'],b['rng']['torch_cpu']))
    def test_token_weighted_96_plus_32(self):
        models=[L.new_model(SPEC,CPU) for _ in range(2)];losses=[]
        for m,micro in zip(models,[128,96]):
            opt=torch.optim.SGD(m.parameters(),lr=0.);scaler=torch.amp.GradScaler('cpu',enabled=False)
            losses.append(L.update(m,opt,scaler,self.cache,np.arange(128),CPU,micro,0.)['CE'])
        self.assertAlmostEqual(*losses,places=5)
        self.assertLess(max(float((a.grad-b.grad).abs().max()) for a,b in zip(*[list(m.parameters()) for m in models])),1e-6)
    def test_late_oom_discards_partial_grad_and_rng(self):
        a=L.new_model(SPEC,CPU);b=L.new_model(SPEC,CPU)
        oa=L.optimizer(a,.001,True);ob=L.optimizer(b,.001,True)
        sa=torch.amp.GradScaler('cpu',enabled=False);sb=torch.amp.GradScaler('cpu',enabled=False)
        rng=T.rng_state();original=L.collate;calls=[0]
        def fail(*args):
            calls[0]+=1
            if calls[0]==2:
                torch.rand(9);raise torch.cuda.OutOfMemoryError('injected backward allocation')
            return original(*args)
        with patch.object(L,'collate',fail):result,micro=L.resilient_update(a,oa,sa,self.cache,np.arange(8),CPU,4,.001)
        self.assertEqual(micro,2);self.assertEqual(len(result['OOM_retries']),1)
        T.restore_rng(rng,CPU);L.update(b,ob,sb,self.cache,np.arange(8),CPU,2,.001)
        for x,y in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(x,y))
        self.assertTrue(all(int(v['step'])==1 for v in oa.state.values()))
    def test_adam_partial_oom_rolls_back_committed_state(self):
        self.run_model('normal');original=torch.optim.AdamW.step;calls=[0]
        def fail(opt,*args,**kw):
            calls[0]+=1
            if calls[0]==2:
                with torch.no_grad():opt.param_groups[0]['params'][0].add_(100.)
                raise torch.cuda.OutOfMemoryError('injected partial Adam')
            return original(opt,*args,**kw)
        with patch.object(torch.optim.AdamW,'step',fail):self.run_model('recovered')
        a,b=self.payload('normal'),self.payload('recovered')
        self.assertEqual(b['state']['updates'],4);self.assertEqual(b['state']['cursor'],32)
        # Different micro-batch can round differently, but no partial +100 Adam update survives.
        self.assertLess(max(float((v-b['model'][k]).abs().max()) for k,v in a['model'].items()),1e-5)
        self.assertTrue((self.tmp/'recovered/adam-oom-recovery.json').exists())
    def test_branch_adam_partial_oom_restores_anchor(self):
        self.run_model('continuous');original=torch.optim.AdamW.step;calls=[0]
        def fail(opt,*args,**kw):
            calls[0]+=1
            if calls[0]==3:
                with torch.no_grad():opt.param_groups[0]['params'][0].add_(100.)
                raise torch.cuda.OutOfMemoryError('injected partial branch Adam')
            return original(opt,*args,**kw)
        with patch.object(torch.optim.AdamW,'step',fail):
            self.run_model('branch-oom',point_updates=1,max_points=1)
        self.assert_tensors(self.payload('continuous'),self.payload('branch-oom'))
        self.assertTrue(list((self.tmp/'branch-oom/points').glob('*/incomplete.json')))
    def test_failed_eval_rolls_back_warmup_anchor_then_backward(self):
        self.run_model('continuous')
        logs=[];original=L.update
        def record(*args):
            logs.append(args[-1]);return original(*args)
        def timeout(*args):raise E.C.BudgetExpired('injected evaluation timeout')
        with patch.object(L,'update',record):self.run_model('branch',point_updates=1,max_points=1,evaluate=timeout)
        a,b=self.payload('continuous'),self.payload('branch');self.assert_tensors(a,b)
        self.assertTrue(list((self.tmp/'branch/points').glob('*/incomplete.json')))
        self.assertEqual(b['state']['points'],[]);self.assertGreater(b['state']['branch_attempted_updates'],0)
        self.assertLessEqual(max(logs),L.PEAK_LR['C']*4/1000)
    def test_output_and_live_resume_preserve_all_state(self):
        self.run_model('continuous',updates=6);self.run_model('first',updates=3)
        # Build the actual model-C/model-D directory shape without consuming credentials.
        work=self.tmp/'working';work.mkdir()
        for name in ('C','D'):
            shutil.copytree(self.tmp/'first',work/f'model-{name}')
            root=work/f'model-{name}';r=json.loads((root/'latest.json').read_text())
            r['identity']['model']=name;write(root/'latest.json',r)
        Live.stage_snapshot(work,self.tmp/'stage','thisarivson/nlp-hw02-l1-live','regression')
        restored=Live.restore_snapshot(self.tmp/'stage',self.tmp/'restored')
        broken=self.tmp/'broken';shutil.copytree(work,broken)
        for name in ('C','D'):
            r=json.loads((broken/f'model-{name}/latest.json').read_text());r['updates']=999;write(broken/f'model-{name}/latest.json',r)
            (broken/f'model-{name}'/r['file']).write_bytes(b'corrupt')
        selected=select_resume([broken,work,restored],self.tmp/'selected')
        self.assertEqual(len(json.loads((selected/'selection.json').read_text())['rejected']),2)
        self.run_model('continued',updates=3,resume=selected/'model-C')
        self.assert_tensors(self.payload('continuous'),self.payload('continued'))
        self.assertEqual(self.payload('continued')['state']['cursor'],48)
        with self.assertRaises(ValueError):select_resume([broken],self.tmp/'no-fresh')
    def test_checkpoint_rotation_and_cyclic_tail(self):
        self.run_model('rotated',checkpoint_seconds=0.)
        self.assertEqual(len(list((self.tmp/'rotated/checkpoints').glob('*.pt'))),2)
        seq=np.concatenate([L.take(self.order,i,8) for i in range(0,144,8)])
        self.assertTrue(np.array_equal(seq[:137],self.order));self.assertTrue(np.array_equal(seq[137:],self.order[:7]))
    def test_cache_memmap_boundaries_and_overflow(self):
        from l1_cache import CompactCache,FILES
        root=self.tmp/'cache';folder=root/'shard';folder.mkdir(parents=True)
        arrays={'source.npy':np.array([1,3,2],dtype=np.int16),'source-offsets.npy':np.array([0,3],dtype=np.int64),
                'target.npy':np.array([4],dtype=np.int16),'target-offsets.npy':np.array([0,1],dtype=np.int64),
                'provenance-rows.npy':np.array([0],dtype=np.uint32),'question-sha256.npy':np.zeros((1,32),dtype=np.uint8)}
        for n,a in arrays.items():np.save(folder/n,a)
        write(root/'compact-manifest.json',{'sealed':True,'representation':30,'rows':1,
              'shards':[{'folder':'shard','rows':1,'files':[]}]})
        cache=CompactCache(root);self.assertIsInstance(cache.parts[0]['source.npy'],np.memmap)
        for index in (-1,1):
            with self.assertRaises(IndexError):cache[index]
        np.save(folder/'target.npy',np.array([32767],dtype=np.int16))
        with self.assertRaisesRegex(AssertionError,'Token ID'):CompactCache(root)
        np.save(folder/'target.npy',np.ones(65536,dtype=np.int16)*4)
        np.save(folder/'target-offsets.npy',np.array([0,65536],dtype=np.int64))
        with self.assertRaisesRegex(AssertionError,'overflow'):CompactCache(root,verify=False)
    def test_live_failure_retries_after_300_seconds(self):
        class Stop:
            done=False
            def is_set(self):return self.done
            def wait(self,*a):self.done=True
        live=Live.L1LiveSync(self.tmp,self.tmp/'scratch','thisarivson/nlp-hw02-l1-live')
        live.stop_event=Stop();live.failures=1;live.last_attempt=100.;live.last_success=100.
        with patch.object(Live.time,'monotonic',lambda:200.),patch.object(live,'sync') as sync:
            live.run();sync.assert_not_called()
        live.stop_event=Stop()
        with patch.object(Live.time,'monotonic',lambda:401.),patch.object(live,'sync',return_value=False) as sync:
            live.run();sync.assert_called_once();self.assertIsNone(live.last_signature)
    def test_graceful_stop_publishes_state_and_identity_rejects_changes(self):
        write(self.tmp/'stop-request.json',{'reason':'insurance expired'})
        value=self.run_model('stopped');self.assertEqual(value['state']['updates'],0)
        payload=self.payload('stopped');bad=copy.deepcopy(payload['identity']);bad['recipe']['weight_decay']=.02
        r=json.loads((self.tmp/'stopped/latest.json').read_text())
        with self.assertRaisesRegex(AssertionError,'identity'):L.read_full(self.tmp/'stopped'/r['file'],bad)
    def test_executor_wall_timeout_and_actual_restore(self):
        start=time.monotonic();original=subprocess.Popen;children=[]
        def hung(*args,**kw):
            child=original([sys.executable,'-c','import time;time.sleep(30)'],**kw);children.append(child);return child
        with patch.object(E.subprocess,'Popen',hung),self.assertRaises(E.C.BudgetExpired):
            E.execute([{'generated_code':'while True: pass'}],deadline=time.monotonic()+.05)
        self.assertLess(time.monotonic()-start,1.);self.assertIsNotNone(children[0].poll())
        import s3_representation as R
        code='def solve() -> int:\n    result = 2 + 3\n    return result\n'
        result=E.execute([{'generated_code':R.restore_compact(R.compact(code))}],deadline=time.monotonic()+10)
        self.assertEqual(result['rows'][0]['value'],5)
    def test_executor_group_dies_with_parent(self):
        pidfile=self.tmp/'child.pid';ready=self.tmp/'ready'
        childcode='import sys,time;from pathlib import Path;sys.path.insert(0,'+repr(str(ROOT/'src'))+');from l1_execute import guard_parent;guard_parent();Path('+repr(str(ready))+').touch();time.sleep(30)'
        parentcode='import subprocess,sys;from pathlib import Path;p=subprocess.Popen([sys.executable,"-c",'+repr(childcode)+'],start_new_session=True);Path('+repr(str(pidfile))+').write_text(str(p.pid));p.wait()'
        parent=subprocess.Popen([sys.executable,'-c',parentcode],start_new_session=True)
        try:
            until=time.monotonic()+3
            while not ready.exists() and time.monotonic()<until:time.sleep(.01)
            self.assertTrue(ready.exists());pid=int(pidfile.read_text())
            parent.terminate();parent.wait(timeout=2);time.sleep(.3)
            try:os.kill(pid,0);alive=True
            except ProcessLookupError:alive=False
            stat=Path('/proc')/str(pid)/'stat'
            zombie=stat.exists() and stat.read_text().rsplit(')',1)[1].split()[0]=='Z'
            self.assertTrue(not alive or zombie,'Executor remained alive after parent exit')
        finally:
            if parent.poll() is None:parent.kill();parent.wait(timeout=1)
            if pidfile.exists():
                import signal
                try:os.killpg(int(pidfile.read_text()),signal.SIGKILL)
                except ProcessLookupError:pass
    def test_resume_at_anchor_uses_historical_throughput_for_length(self):
        self.run_model('anchor',updates=200)
        root=self.tmp/'anchor';r=json.loads((root/'latest.json').read_text());p=self.payload('anchor')
        p['state']['main_seconds']=100.;p['state']['next_point_seconds']=100.
        torch.save(p,root/r['file']);r.update(sha256=sha(root/r['file']),bytes=(root/r['file']).stat().st_size,main_seconds=100.)
        write(root/'latest.json',r)
        value=self.run_model('resume-anchor',updates=1,resume=root,point_seconds=100.,cooldown_cap=3.,evaluate=lambda *a:{})
        point=value['state']['points'][0]
        self.assertEqual(point['planned_cooldown_updates'],5);self.assertEqual(point['cooldown_updates'],5)
        self.assertEqual(value['state']['updates'],201)
    def test_ce_late_oom_reduces_batch(self):
        class Tok:
            def encode(self,*a,**kw):return type('Encoded',(),{'ids':[0,1]})()
        rows=[{'question':'q','code':'def solve(): return 5'}]*6;m=L.new_model(SPEC,CPU);original=m.forward;calls=[0]
        def fail(*args):
            calls[0]+=1
            if calls[0]==2:raise torch.cuda.OutOfMemoryError('CE')
            return original(*args)
        with patch.object(m,'forward',fail):value=E.ce(m,rows,Tok(),CPU,time.monotonic()+10,smoke=True)
        self.assertEqual(value['batch'],1);self.assertEqual(value['tokens'],18);self.assertEqual(value['OOM_retries'],[2])
    def test_secrets_gate_and_same_notebook_recipe(self):
        fake=type('Torch',(),{'cuda':type('Cuda',(),{'device_count':staticmethod(lambda:2),'get_device_name':staticmethod(lambda i:'Tesla T4')})()})()
        def missing(name):raise RuntimeError('absent')
        self.assertFalse(Live.gate(fake,missing)['passed'])
        self.assertTrue(Live.gate(fake,missing,require_secrets=False)['passed'])
        import build_l1
        paths=[ROOT/'notebooks'/n for n in ('nlp-hw02-kaggle-l1.ipynb','nlp-hw02-l1-gpu-smoke.ipynb')]
        saved={p:p.read_bytes() if p.exists() else None for p in paths}
        try:
            for kind in ('l1','l1_gpu_smoke'):
                nb,cfg=build_l1.build(kind,cache_sha='46ef881bba60dba2563a25a7490becf05104a0537b70c3510629bef8c93902be')
                for cell in json.loads(nb.read_text())['cells']:
                    if cell['cell_type']=='code':compile(cell['source'],str(nb),'exec')
                if kind=='l1':self.assertEqual((cfg['worker_deadline_seconds'],cfg['hard_shutdown_seconds']),(40680,42300))
        finally:
            for p,data in saved.items():
                if data is None:p.unlink(missing_ok=True)
                else:p.write_bytes(data)
    def test_decision_uses_numeric_time_order(self):
        for name in ('C','D'):
            for tag,t,acc in [('t999000',999000,.5),('t1008000',1008000,.6)]:
                write(self.tmp/f'model-{name}/points/{tag}/complete.json',{'training':{'main_seconds':t,'target_main_seconds':t},
                      'metrics':{'splits':{'heldout4k':{'exec':{'accuracy':acc}}}}})
        self.assertEqual(decision(self.tmp)['last_common_point'],'t1008000')

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Regressions)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(ROOT.parent/'checks/kaggle-l1-review-regressions.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,
          'failures':len(result.failures),'errors':len(result.errors),'names':unittest.defaultTestLoader.getTestCaseNames(Regressions)})
    sys.exit(not result.wasSuccessful())
