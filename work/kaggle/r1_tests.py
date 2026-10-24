"""CPU contracts for R1 growth, Adam state, loaders, data and terminal branches."""
import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import exp26_core as C
import exp26_train as T
import l1_runtime as L
import l2_runtime as R
import r1_growth as G
import r1_runtime as Q
from r1_data import filtered_order
from kaggle_io import sha,write
from l2_gpu_smoke import digest
from p1_tests import Cache
from p1_runtime import new_reviewer_state
from build_r1 import default_branches,validate

CPU=torch.device('cpu')
SPEC=dict(d_model=8,heads=2,encoder_layers=6,decoder_layers=2,swiglu_dim=16,vocab=24)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='r1-test-');self.root=Path(self.tmp.name)
        self.cache=Cache();self.order=np.arange(128,dtype=np.uint32)
        self.m=L.new_model(SPEC,CPU);self.o=L.optimizer(self.m,.0015,True)
        self.o.defaults['fused']=False
        for group in self.o.param_groups:group['fused']=False
        self.s=torch.amp.GradScaler('cpu',enabled=False)
        self.state=new_reviewer_state();self.state['micro_batch']=64
        self.ident=L.identity_for('C',SPEC,self.cache,'tiny',True)
    def tearDown(self):self.tmp.cleanup()
    def anchor(self):return copy.deepcopy(R.payload(self.m,self.o,self.s,self.state,SPEC,self.ident))

    def test_growth_function_identity_and_rng(self):
        src=torch.tensor([[1,4,5,2,0],[1,8,7,6,2]]);tgt=torch.tensor([[1,8,9],[1,5,6]])
        self.m.train();before=self.m(src,tgt).detach();rng=digest(T.rng_state())
        old=dict(self.m.named_parameters());spec=G.deepen(self.m,self.o,SPEC)
        after=self.m(src,tgt).detach()
        self.assertEqual(rng,digest(T.rng_state()));self.assertEqual(spec['encoder_layers'],12)
        torch.testing.assert_close(before,after,rtol=0,atol=1e-6)
        self.assertTrue(torch.equal(before,after))
        for i in range(6):
            layer=self.m.transformer.encoder.layers[2*i+1]
            self.assertTrue(torch.count_nonzero(layer.self_attn.out_proj.weight)==0)
            self.assertTrue(torch.count_nonzero(layer.linear2.weight)==0)
            self.assertTrue(torch.equal(layer.norm1.weight,torch.ones(8)))
        for name,p in old.items():
            if name.startswith('transformer.encoder.layers.'):
                parts=name.split('.');parts[3]=str(int(parts[3])*2);name='.'.join(parts)
            self.assertIs(dict(self.m.named_parameters())[name],p)

    def test_nine_layers_insert_after_1_3_5_and_half_dtype(self):
        self.m.half().eval();src=torch.tensor([[1,4,2]]);tgt=torch.tensor([[1,8]])
        before=self.m(src,tgt).detach();old=list(self.m.transformer.encoder.layers)
        spec=G.deepen(self.m,self.o,SPEC,encoder_layers=9)
        self.assertEqual(spec['encoder_layers'],9)
        for i,j in enumerate((0,1,3,4,6,7)):self.assertIs(self.m.transformer.encoder.layers[j],old[i])
        self.assertTrue(all(p.dtype==torch.float16 for p in self.m.parameters()))
        torch.testing.assert_close(before,self.m(src,tgt),rtol=0,atol=0)

    def test_optional_secrets_missing_partial_wrong_owner_and_enabled(self):
        import os
        from unittest.mock import patch
        from types import SimpleNamespace
        from r1_live import optional_gate
        fake=SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:2,get_device_name=lambda i:'Tesla T4'))
        def missing(name):raise KeyError(name)
        with patch.dict(os.environ,{},clear=True):
            for getter in (missing,lambda name:'friend2' if name=='KAGGLE_USERNAME' else None,
                           lambda name:'thisarivson' if name=='KAGGLE_USERNAME' else 'fake-test-key'):
                gate=optional_gate(fake,'friend2',secret_getter=getter)
                self.assertTrue(gate['passed']);self.assertFalse(gate['live_enabled'])
                self.assertNotIn('KAGGLE_KEY',os.environ)
            gate=optional_gate(fake,'friend2',secret_getter=lambda name:'friend2' if name=='KAGGLE_USERNAME' else 'fake-test-key')
            self.assertTrue(gate['live_enabled']);self.assertEqual(os.environ['KAGGLE_USERNAME'],'friend2')
        fake.cuda.device_count=lambda:1
        self.assertFalse(optional_gate(fake,'friend2',secret_getter=missing)['passed'])

    def test_dad_build_is_private_optional_live_cross_owner_and_smoke_budget(self):
        from build_r1 import build
        from build_l3 import literals
        for owner,smoke in (('friend2',False),('thisarivson',True)):
            path,cfg=build(owner,smoke=smoke)
            code=[c['source'] for c in json.loads(path.read_text())['cells'] if c['cell_type']=='code']
            self.assertFalse(cfg['require_secrets'])
            self.assertIn('optional_gate',code[0]);self.assertIn("if GATE.get('live_enabled'):",code[2])
            self.assertEqual(cfg['anchor_source'],'thisarivson/nlp-hw02-l3-states')
            self.assertIn("resolve_cache_mount(inp,CONFIG['cache_source'],cache_kind)",code[3])
            self.assertIn('r1_gpu_smoke.py',literals(code[0])['SOURCES'])
            self.assertIn('final_inference_v7.py',literals(code[0])['SOURCES'])
            if smoke:
                self.assertLessEqual(cfg['session_ceiling_seconds'],900)
                self.assertGreaterEqual(cfg['branches'][0]['max_main_updates'],200)
            else:
                self.assertEqual(cfg['branches'][0]['cooldown_seconds'],1200)
                self.assertEqual(cfg['branches'][0]['q'],.1)

    def test_r1_worker_failure_isolation_and_credential_env(self):
        import os
        import r1_session as S
        from unittest.mock import patch,MagicMock
        from types import SimpleNamespace
        start=time.monotonic();root=self.root/'supervisor';envs=[];procs=[]
        cfg=dict(working_root=str(root),session_start_monotonic=start,deadline_monotonic=start+10,
                 hard_shutdown_seconds=20,session_ceiling_seconds=30)
        def spawn(*args,**kwargs):
            envs.append(kwargs['env']);p=SimpleNamespace(returncode=1 if not procs else None)
            p.poll=lambda:p.returncode;procs.append(p);return p
        def finish(_):
            self.assertFalse((root/'stop-request.json').exists())
            procs[1].returncode=0;write(root/'model-GPU1/summary.json',dict(status='complete'))
        secrets={k:'fake-test-value' for k in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN','KAGGLE_USER_SECRETS_TOKEN','KAGGLE_IAP_TOKEN')}
        watchdog=MagicMock()
        with patch.dict(os.environ,secrets),patch.object(S.subprocess,'Popen',spawn),patch.object(S.time,'sleep',finish),patch.object(S,'stop_workers',return_value=[]):
            report=S.launch(cfg,self.root/'worker.py',watchdog,heartbeat=.01)
        self.assertEqual(report['exit_codes'],[1,0]);self.assertEqual(report['status'],'partial')
        for env in envs:
            self.assertTrue(set(secrets).isdisjoint(env))
            self.assertIn('empty-kaggle-gpu',env['KAGGLE_CONFIG_DIR'])
        self.assertTrue((root/'r1-output-manifest.json').exists());watchdog.finish.assert_called_once()

    def test_old_adam_moments_and_new_warmup(self):
        R.safe_update(self.m,self.o,self.s,self.cache,self.order,CPU,64,.0015)
        before={p:copy.deepcopy(v) for p,v in self.o.state.items()}
        G.deepen(self.m,self.o,SPEC)
        for p,saved in before.items():self.assertEqual(digest(saved),digest(self.o.state[p]))
        new=self.o.param_groups[1]['params'];self.assertTrue(all(p not in self.o.state for p in new))
        log,_=G.update(self.m,self.o,self.s,self.cache,self.order,CPU,64,.0015,1,1000,.2)
        self.assertEqual(log['truncation']['requested_q'],.2)
        self.assertEqual(self.o.param_groups[0]['lr'],.0015)
        self.assertEqual(self.o.param_groups[1]['lr'],.0015/1000)
        self.assertTrue(all(int(self.o.state[p]['step'])==2 for p in before))
        self.assertTrue(all(int(self.o.state[p]['step'])==1 for p in new))

    def test_none_exact_next_steps_and_full_r1_roundtrip(self):
        a=self.anchor();reference=[]
        for j in range(2):
            R.safe_update(self.m,self.o,self.s,self.cache,self.order,CPU,64,.0015)
            reference.append(digest((self.m.state_dict(),self.o.state_dict(),T.rng_state())))
        m,o,s,state,spec=Q.restore_anchor(a,default_branches()[1],CPU,True)
        for j in range(2):
            G.update(m,o,s,self.cache,self.order,CPU,64,.0015,j+1,1000,0)
            self.assertEqual(reference[j],digest((m.state_dict(),o.state_dict(),T.rng_state())))
        Q.save_full(self.root,m,o,s,state,spec,self.ident)
        before=digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state(),state))
        m,o,s,state,_,_=Q.reload_latest(self.root,CPU,True)
        self.assertEqual(before,digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state(),state)))

    def test_deep_optimizer_serialization_and_next_update(self):
        spec=G.deepen(self.m,self.o,SPEC)
        G.update(self.m,self.o,self.s,self.cache,self.order,CPU,64,.0015,1,1000,0)
        Q.save_full(self.root,self.m,self.o,self.s,self.state,spec,self.ident)
        G.update(self.m,self.o,self.s,self.cache,self.order,CPU,64,.0015,2,1000,0)
        expected=digest((self.m.state_dict(),self.o.state_dict(),T.rng_state()))
        m,o,s,_,_,_=Q.reload_latest(self.root,CPU,True)
        G.update(m,o,s,self.cache,self.order,CPU,64,.0015,2,1000,0)
        self.assertEqual(expected,digest((m.state_dict(),o.state_dict(),T.rng_state())))

    def test_filtered_order_membership_seed_and_tamper(self):
        src=self.root/'source';src.mkdir();np.save(src/'order.npy',np.arange(4,128,dtype=np.uint32))
        h=sha(src/'order.npy')
        write(src/'order.json',dict(rows=128,slots=124,cache_manifest_sha256='tiny',
              file=dict(sha256=h,bytes=(src/'order.npy').stat().st_size)))
        a,ma=filtered_order(src/'order.npy',self.cache,h,17,self.root/'a')
        b,mb=filtered_order(src,self.cache,h,17,self.root/'b')
        self.assertTrue(np.array_equal(a,b));self.assertEqual(set(a),set(range(4,128)))
        self.assertEqual(ma,mb)
        with self.assertRaises(AssertionError):filtered_order(src,self.cache,'0'*64,17,self.root/'c')
        np.save(src/'order.npy',np.zeros(124,dtype=np.uint32))
        with self.assertRaises(AssertionError):filtered_order(src,self.cache,h,17,self.root/'c')

    def test_filtered_long2x_checks_every_retained_multiplicity(self):
        import hashlib
        self.cache.target_lengths[64:]=100
        kept=np.setdiff1d(np.arange(4,128,dtype=np.uint32),[5,65]).astype(np.uint32)
        slots=np.r_[kept,kept[kept>=64]]
        src=self.root/'weighted';src.mkdir()
        meta=dict(rows=128,slots=len(slots),cache_manifest_sha256='tiny',
                  selected_cache_indices_sha256=hashlib.sha256(kept.astype('<u4').tobytes()).hexdigest(),
                  filtered_multiplicity=dict(format='nlp-hw02-filtered-long2x-v1',threshold=79,short=1,long=2))
        def save(values):
            np.save(src/'order.npy',values)
            h=sha(src/'order.npy');meta.update(slots=len(values),file=dict(sha256=h,bytes=(src/'order.npy').stat().st_size))
            write(src/'order.json',meta)
            return h
        h=save(slots)
        actual,_=filtered_order(src,self.cache,h,17,self.root/'out')
        np.testing.assert_array_equal(np.bincount(actual,minlength=128),np.bincount(slots,minlength=128))
        # A valid new SHA must not disguise removal of only one long occurrence.
        bad=np.delete(slots,np.flatnonzero(slots==64)[0]);h=save(bad)
        with self.assertRaisesRegex(AssertionError,'multiplicity'):filtered_order(src,self.cache,h,17,self.root/'bad')
        h=save(np.r_[slots,np.uint32(4)])
        with self.assertRaisesRegex(AssertionError,'multiplicity'):filtered_order(src,self.cache,h,17,self.root/'bad')
        meta.pop('filtered_multiplicity');h=save(slots)
        with self.assertRaisesRegex(AssertionError,'unique'):filtered_order(src,self.cache,h,17,self.root/'bad')

    def test_v6_v7_load_12_encoder_layers(self):
        # Import exact final dependencies from their frozen notebook source map.
        from build_l3 import literals
        nb=json.loads((ROOT/'notebooks/nlp-hw02-ensemble-inference-v7.ipynb').read_text())
        sources=literals(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))['SOURCES']
        modules=self.root/'modules';modules.mkdir()
        for name,source in sources.items():(modules/name).write_text(source)
        sys.path.insert(0,str(modules))
        try:
            import final_inference_v6 as V6
            import final_inference_v7 as V7
            spec=G.deepen(self.m,self.o,SPEC)
            point=L.save_point(self.root,'test',self.m,spec,self.ident,{})
            artifact=torch.load(point,weights_only=True)
            reference=C.SimpleTransformer(**spec,dropout=0.);reference.load_state_dict(artifact['model'])
            src=torch.tensor([[1,4,2]]);tgt=torch.tensor([[1,8]])
            for loader in (V6.load,V7.load):
                m=loader(point,CPU);self.assertEqual(len(m.transformer.encoder.layers),12)
                torch.testing.assert_close(m(src,tgt),reference(src,tgt),rtol=0,atol=1e-6)
        finally:sys.path.remove(str(modules))

    def test_two_short_cpu_branches_and_stable_restore(self):
        self.state.update(cursor=385,cursor_origin=385)
        a=self.anchor();branches=default_branches();branches[0].update(q=.2,data='filtered')
        for i,b in enumerate(branches):
            b.update(main_seconds=10.,max_main_updates=2,cooldown='1-sqrt',cooldown_fraction=.5,
                     cooldown_seconds=10.,eval_seconds=1.,new_warmup=2)
            expected=dict(splits=dict(heldout4k=2,old991=1),test=2,k8=1)
            def evaluate(point,out,deadline):
                return dict(splits={s:dict(exec=dict(denominator=n),CE=dict(CE=1.)) for s,n in expected['splits'].items()},
                            test=dict(rows=2),K8_tune2048=dict(complete=True,rows=1,target_rows=1))
            report=Q.run(copy.deepcopy(a),b,self.cache,self.order,dict(file=dict(sha256='tiny')),
                         self.root/f'model-GPU{i}',CPU,dict(deadline_monotonic=time.monotonic()+60,
                         expected_rows=expected,io_reserve_seconds=1),evaluate,True)
            self.assertEqual(report['status'],'complete');self.assertEqual(report['stable_updates'],2)
            self.assertEqual(report['stable_cursor'],385+2*128)
            stable=torch.load(self.root/f'model-GPU{i}'/json.loads((self.root/f'model-GPU{i}/latest.json').read_text())['file'],weights_only=True)
            self.assertEqual(stable['state']['r1']['order_start_slot'],1)
            self.assertEqual(report['training']['cooldown_updates'],2)
            self.assertEqual(report['training']['last_update_lr'],0.)

    def test_budget_and_fraction_schedule(self):
        self.assertEqual(validate(default_branches()),default_branches())
        b=default_branches()[0];b.update(cooldown='1-sqrt',cooldown_fraction=.1)
        self.assertEqual(Q.schedule(b,9000),1000)
        b['main_seconds']=40000
        with self.assertRaises(AssertionError):validate([b,default_branches()[1]])

    def test_failed_evaluation_never_replaces_stable(self):
        b=default_branches()[0];b.update(main_seconds=10.,max_main_updates=1,cooldown='1-sqrt',
              cooldown_fraction=.5,cooldown_seconds=10.,eval_seconds=1.)
        def incomplete(*args):return {'K8_tune2048':dict(complete=False,rows=0,target_rows=1)}
        report=Q.run(self.anchor(),b,self.cache,self.order,dict(file=dict(sha256='tiny')),self.root,CPU,
              dict(deadline_monotonic=time.monotonic()+60,expected_rows=dict(splits={},test=2,k8=1),
                   io_reserve_seconds=1),incomplete,True)
        self.assertEqual(report['status'],'partial');self.assertEqual(report['stable_updates'],1)
        self.assertFalse((self.root/'points/final/complete.json').exists())
        self.assertTrue((self.root/'weights/final-fp16.pt').exists())

    def test_live_snapshot_carries_both_group_state_and_new_order(self):
        import r1_live
        model_root=self.root/'working/model-GPU0'
        spec=G.deepen(self.m,self.o,SPEC)
        G.update(self.m,self.o,self.s,self.cache,self.order,CPU,64,.0015,1,1000,0)
        Q.save_full(model_root,self.m,self.o,self.s,self.state,spec,self.ident)
        order_root=self.root/'working/order-gpu0';order_root.mkdir()
        np.save(order_root/'order.npy',self.order)
        write(order_root/'order.json',dict(seed=17))
        stage=self.root/'stage'
        r1_live.stage_snapshot(self.root/'working',stage,'thisarivson/nlp-hw02-r1-live','test')
        restored=self.root/'restored';r1_live.restore_snapshot(stage,restored)
        self.assertEqual(sha(order_root/'order.npy'),sha(restored/'order-gpu0/order.npy'))
        before=digest((self.m.state_dict(),self.o.state_dict()))
        m,o,_,_,_,_=Q.reload_latest(restored/'model-GPU0',CPU,True)
        self.assertEqual(before,digest((m.state_dict(),o.state_dict())))


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(ROOT.parent/'checks/kaggle-r1-regressions.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,GPU=False))
    sys.exit(not result.wasSuccessful())
