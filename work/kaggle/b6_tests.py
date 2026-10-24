"""Both cooldown math paths, serialization/RNG, exact N, cursor, partial/unsafe recovery."""
import copy,hashlib,json,shutil,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
import b5_tests as base
import b5_state as S
import b6_state as D
import b6_runtime as G
import b6_gpu_smoke as Smoke
import l1_runtime as L
import l2_runtime as R
import exp26_train as T
import torch,numpy as np
from p1_math import sqrt_lr
from l2_gpu_smoke import digest
from build_l3 import ROOT,write
CPU=torch.device('cpu')


class Tests(base.Tests):
    def config(self,**kw):
        return dict(dict(gpu='0',N_by_gpu={'0':6,'1':6},deadline_monotonic=time.monotonic()+1000,
            io_reserve_seconds=0.,final_eval_seconds=1.,checkpoint_seconds=0.,gpu_smoke=False,save_midpoint=True),**kw)
    def run_b6(self,kind='W',evaluate=None,**cfg):
        src,pin=self.prepare(kind);out=self.root/('b6-'+kind)
        with patch.object(S,'read_full',side_effect=self.fixture_read):
            result=G.run(kind,self.cache,self.order,self.meta,out,CPU,self.config(**cfg),
                         evaluate or self.metrics,src,pin,cpu=True)
        return result,out,pin

    def test_b6_full_schedules_endpoints_and_budget_from_actual_logs(self):
        pins=json.loads((ROOT/'b6-anchors.json').read_text())
        for g,b in pins['budget'].items():
            n=b['N'];peak=b['peak_lr'];rates=[sqrt_lr(j,n,peak) for j in range(n+1)]
            self.assertEqual(rates[0],peak);self.assertEqual(rates[-1],0.)
            self.assertGreater(rates[1],0.);self.assertLess(rates[1],peak)
            self.assertTrue(all(x>y for x,y in zip(rates,rates[1:])))
            self.assertEqual(n,int(13200/b['reference_seconds_per_update']//100)*100)
            self.assertGreaterEqual(b['safety_factor'],1.1);self.assertLess(b['expected_wall_seconds'],18900)
        for j,n in ((-1,4),(5,4),(0,0)):
            with self.assertRaises(ValueError):sqrt_lr(j,n)

    def test_b6_pinned_mount_both_orders_no_full_state_copies_and_corruption(self):
        import b6_resume as Mount
        from kaggle_io import sha
        source=self.root/'B5-output';source.mkdir();pins=dict(anchors={},finals={})
        for g,kind in (('0','W'),('1','C')):
            src,pin=self.prepare(kind);shutil.copytree(src,source/f'model-GPU{g}')
            shutil.copytree(self.root/'order',source/f'weighted-order-GPU{g}')
            complete=source/f'model-GPU{g}/points/final/complete.json';write(complete,dict(training=dict(anchor=pin)))
            pins['anchors'][g]=pin;pins['finals'][g]=dict(tag='final',anchor=pin,complete_sha256=sha(complete))
        files=[dict(file=str(p.relative_to(source)),bytes=p.stat().st_size,sha256=sha(p)) for p in source.rglob('*') if p.is_file()]
        write(source/'b1-output-manifest.json',dict(files=files));pins['source_manifest_sha256']=sha(source/'b1-output-manifest.json')
        out=self.root/'mounted';roots=Mount.mount(source,out,self.cache,pins)
        self.assertEqual(roots,{g:str(source/f'model-GPU{g}') for g in ('0','1')})
        self.assertFalse(list(out.rglob('*.pt')))
        for g in ('0','1'):self.assertEqual(sha(out/f'weighted-order-GPU{g}/order.npy'),self.meta['file']['sha256'])
        (source/'weighted-order-GPU1/order.npy').write_bytes(b'corrupt')
        with self.assertRaises(AssertionError):Mount.mount(source,self.root/'corrupt-mount',self.cache,pins)

    def test_b6_reviewer_gates_source_config_and_retains_launch_history(self):
        from b6_review_api import launch_gate
        from b6_static_check import merge_review
        base=dict(ready=True,CPU_ready=True,modules={'W':'w'},C_overrides={'C':'c'},
            production_config_sha256='prod',smoke_config_sha256='smoke',smoke_passed=False,GPU_smoke_pending=True)
        source=dict(modules=base['modules'],C_overrides=base['C_overrides'],config_sha256='smoke')
        launch_gate(base,source,True)
        with self.assertRaises(AssertionError):launch_gate(base,dict(source,config_sha256='stale'),True)
        previous=dict(base,GPU_smoke_receipt='checks/smoke.json',production_url='private/B6',status='RUNNING')
        good=dict(source,passed=True,smoke=True)
        for receipt in (None,dict(good,passed=False),dict(good,config_sha256='old'),dict(good,C_overrides={'C':'wrong'})):
            r=merge_review(dict(base),previous,receipt);self.assertFalse(r['smoke_passed']);self.assertEqual(r['status'],'RUNNING')
        r=merge_review(dict(base),previous,good);self.assertTrue(r['smoke_passed']);self.assertFalse(r['GPU_smoke_pending'])
        launch_gate(r,dict(source,config_sha256='prod'),False,good)
        with self.assertRaises(AssertionError):launch_gate(r,dict(source,config_sha256='prod'),False,dict(good,config_sha256='old'))
        r=merge_review(dict(base,production_config_sha256='changed'),previous,good)
        self.assertFalse(r['smoke_passed']);self.assertNotIn('production_url',r)

    def test_b6_exactly_one_push_per_target_no_retry_or_unknown_version(self):
        from b6_review_api import require_single_push, reserve_push
        for smoke in (True, False):
            require_single_push(0, [], smoke)
            for remote, known in ((1, []), (1, [1]), (2, [1]), (0, [1]), (2, [1, 2])):
                with self.assertRaises(AssertionError):require_single_push(remote, known, smoke)
        path=self.root/'push.intent.json';reserve_push(path,dict(remote_version=0))
        with self.assertRaises(FileExistsError):reserve_push(path,dict(remote_version=1))
        self.assertEqual(json.loads(path.read_text()),dict(remote_version=0))

    def test_b6_output_receipt_accepts_relative_and_absolute_paths(self):
        import os
        from b6_review_output import relative_receipt
        absolute=self.root/'checks/smoke.json';relative=Path(os.path.relpath(absolute,Path.cwd()))
        self.assertEqual(relative_receipt(relative,self.root),'checks/smoke.json')
        self.assertEqual(relative_receipt(absolute,self.root),'checks/smoke.json')
        with self.assertRaises(ValueError):relative_receipt(self.root.parent/'outside.json',self.root)

    def test_b6_both_paths_exact_N_and_uninterrupted_control(self):
        for kind in ('W','C'):
            src,pin=self.prepare(kind);initial=copy.deepcopy(self.state);out=self.root/('exact-'+kind)
            p=self.fixture_read(src,self.ident);m,o,s,st=S.restore(p,CPU);cd=D.new_cooldown(st,6,pin['sha256'])
            for _ in range(6):D.step(m,o,s,st,cd,self.cache,self.order,CPU,self.ident)
            expected=digest((m.state_dict(),o.state_dict(),s.state_dict(),st,T.rng_state()))
            seen=[];original=D.step
            def record(*a,**kw):
                log,unsafe=original(*a,**kw);seen.append(log['lr']);return log,unsafe
            def evaluate(*a):torch.rand(17);return self.metrics()
            with patch.object(S,'read_full',side_effect=self.fixture_read),patch.object(D,'step',side_effect=record):
                result=G.run(kind,self.cache,self.order,self.meta,out,CPU,self.config(),evaluate,src,pin,cpu=True)
            self.assertEqual(result['status'],'complete');self.assertEqual(seen,[sqrt_lr(j,6,initial['last_lr']) for j in range(1,7)])
            loaded=D.read_full(out,self.ident,environment=False);m,o,s,st,cd=D.restore(loaded,CPU)
            self.assertEqual(expected,digest((m.state_dict(),o.state_dict(),s.state_dict(),st,T.rng_state())))
            self.assertEqual(st,initial);self.assertEqual(cd['updates'],6);self.assertEqual(cd['last_lr'],0.)
            self.assertEqual(cd['cursor'],pin['cursor']+768)
            ids=np.concatenate([L.take(self.order,pin['cursor']+128*j,128) for j in range(6)]).astype('<u4')
            self.assertEqual(cd['stream_sha256'],hashlib.sha256(ids.tobytes()).hexdigest())
            self.assertEqual(len(list((out/'checkpoints').glob('*.pt'))),2)
            self.assertTrue((out/'weights/b6-mid-fp16.pt').is_file());self.assertFalse((out/'points/b6-mid/evaluation').exists())
            fp=torch.load(out/'weights/b6-final-fp16.pt',weights_only=True)
            self.assertTrue(all(v.dtype==torch.float16 for v in fp['model'].values() if v.is_floating_point()))

    def test_b6_checkpoint_between_steps_preserves_all_RNGs(self):
        for kind in ('W','C'):
            src,pin=self.prepare(kind);p=self.fixture_read(src,self.ident)
            m,o,s,st=S.restore(p,CPU);cd=D.new_cooldown(st,4,pin['sha256'])
            D.step(m,o,s,st,cd,self.cache,self.order,CPU,self.ident)
            out=self.root/('disk-'+kind);D.save_full(out,m,o,s,st,pin['model_spec'],self.ident,cd)
            D.step(m,o,s,st,cd,self.cache,self.order,CPU,self.ident)
            expected=digest((m.state_dict(),o.state_dict(),s.state_dict(),st,T.rng_state()))
            m,o,s,st,cd=D.restore(D.read_full(out,self.ident,False),CPU)
            D.step(m,o,s,st,cd,self.cache,self.order,CPU,self.ident)
            self.assertEqual(expected,digest((m.state_dict(),o.state_dict(),s.state_dict(),st,T.rng_state())))

    def test_b6_incomplete_evaluation_preserves_FP16_and_partial(self):
        for kind in ('W','C'):
            def fail(*a):raise ValueError('injected evaluation failure')
            result,out,pin=self.run_b6(kind,fail)
            self.assertEqual(result['status'],'partial');self.assertFalse(result['final_complete'])
            self.assertTrue((out/'weights/b6-final-fp16.pt').exists())
            self.assertFalse((out/'points/b6-final/complete.json').exists())
            self.assertEqual(D.read_full(out,pin['identity'],False)['cooldown']['updates'],6)

    def test_b6_deadline_and_stop_do_not_change_N_or_claim_final(self):
        for mode in ('deadline','stop'):
            if mode=='stop':write(self.root/'stop-request.json',dict(reason='test'))
            result,out,pin=self.run_b6(N_by_gpu={'0':600,'1':600},deadline_monotonic=time.monotonic()+.01)
            self.assertEqual(result['status'],'partial');self.assertEqual(result['cooldown']['N'],600)
            self.assertEqual(result['cooldown']['updates'],0);self.assertIsNone(result['FP16_final'])

    def test_b6_unsafe_Adam_rolls_back_and_never_serializes_suspect_state(self):
        for kind in ('W','C'):
            src,pin=self.prepare(kind);original=D.step;calls=[0]
            def bad(*a,**kw):
                calls[0]+=1
                if calls[0]==3:
                    with torch.no_grad():next(a[0].parameters()).fill_(float('nan'))
                    raise L.UnsafeAdamOOM('injected')
                return original(*a,**kw)
            out=self.root/('unsafe-'+kind)
            with patch.object(S,'read_full',side_effect=self.fixture_read),patch.object(D,'step',side_effect=bad):
                r=G.run(kind,self.cache,self.order,self.meta,out,CPU,self.config(),self.metrics,src,pin,cpu=True)
            self.assertEqual(r['status'],'partial');self.assertEqual(r['cooldown']['updates'],2)
            self.assertEqual(D.read_full(out,pin['identity'],False)['cooldown']['updates'],2)
            self.assertIsNone(r['FP16_final'])

    def test_b6_AMP_skip_consumes_one_schedule_slot(self):
        src,pin=self.prepare('W');self.scaler=torch.amp.GradScaler('cpu',enabled=True,init_scale=1024)
        cd=D.new_cooldown(self.state,3,pin['sha256']);before=digest((self.m.state_dict(),self.o.state_dict()['state']))
        hook=next(self.m.parameters()).register_hook(lambda g:torch.full_like(g,float('inf')))
        try:log,_=D.step(self.m,self.o,self.scaler,self.state,cd,self.cache,self.order,CPU,self.ident)
        finally:hook.remove()
        self.assertEqual(log['AMP_skip'],1);self.assertEqual(cd['updates'],1);self.assertEqual(cd['cursor'],pin['cursor']+128)
        self.assertEqual(cd['last_lr'],sqrt_lr(1,3,.0006));self.assertEqual(before,digest((self.m.state_dict(),self.o.state_dict()['state'])))

    def test_b6_same_smoke_controller_200_updates_both_paths(self):
        for kind in ('W','C'):
            src,pin=self.prepare(kind);out=self.root/('smoke-'+kind)
            cfg=self.config(N_by_gpu={'0':200,'1':200},gpu_smoke=True,checkpoint_seconds=1000.)
            def metrics(*a):return dict(splits={k:dict(exec=dict(denominator=n),CE=dict(CE=.3)) for k,n in [('heldout4k',64),('old991',16)]},
                K8_tune2048=dict(complete=True,rows=36,target_rows=36),test=dict(rows=64))
            with patch.object(S,'read_full',side_effect=self.fixture_read):
                r=Smoke.exercise(kind,self.cache,self.order,self.meta,out,CPU,cfg,metrics,src,pin,cpu=True)
            self.assertEqual(r['status'],'complete');self.assertTrue(json.loads((out/'gpu-smoke.json').read_text())['passed'])


def main():
    names=sorted(n for n in Tests.__dict__ if n.startswith('test_b6_'))
    r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Tests(n) for n in names))
    write(ROOT.parent/'checks/b6-cpu.json',dict(passed=r.wasSuccessful(),tests=r.testsRun,
        failures=[(str(t),s) for t,s in r.failures+r.errors],GPU=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        modules={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}))
    raise SystemExit(not r.wasSuccessful())


if __name__=='__main__':main()
