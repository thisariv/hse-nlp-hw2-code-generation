"""CPU B1 contracts: init/count/dropout, exact resume/rollback, strict transfer and v7."""
import copy
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT = Path(__file__).resolve().parent; sys.path.insert(0, str(ROOT/'src'))
import exp26_core as C
import exp26_train as T
import l1_runtime as L
import l2_runtime as R
import b1_runtime as B
from b1_data import build_order, verify_order
from kaggle_io import sha, write
from l2_gpu_smoke import digest
from p1_tests import Cache
from b1_transfer import transfer, Client, CONFIG_DIRS
from b1_resume import materialize

CPU = torch.device('cpu')
SPEC = dict(d_model=8, heads=2, encoder_layers=1, decoder_layers=1, swiglu_dim=16, vocab=24)


class MockClient:
    def __init__(self, source, remote): self.source=source; self.remote=remote; self.created=False; self.uploads=0
    def kernel_file(self, source, name, folder):
        path=folder/name; path.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(self.source/name, path); return path
    def private_files(self, ref):
        return dict(exists=True, private=True, version=1, status='ready',
                    files={p.name:p.stat().st_size for p in self.remote.iterdir()}) if self.created else dict(exists=False)
    def create(self, stage, owner):
        self.uploads+=1; self.created=True; shutil.copytree(stage, self.remote)
    def dataset_file(self, ref, name, folder):
        path=folder/name; shutil.copyfile(self.remote/name, path); return path


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='b1-cpu-'); self.root=Path(self.tmp.name)
        self.cache=Cache(); self.order,self.meta=build_order(self.cache, self.root/'order')
        self.m=B.new_model(SPEC, CPU, .2); self.o=L.optimizer(self.m, .001, True)
        self.s=torch.amp.GradScaler('cpu', enabled=False); self.state=B.new_state(64)
        self.ident=B.identity_for('E', SPEC, self.cache, self.meta, .2, True)
        self.cfg=dict(owner='friend2', dropout=.2, deadline_monotonic=time.monotonic()+120,
                      io_reserve_seconds=0, main_seconds=9999., max_updates=2, checkpoint_seconds=999.,
                      probe_cooldown_seconds=.001, probe_eval_seconds=1., final_cooldown_seconds=.001,
                      final_eval_seconds=1., final_q=.1)
    def tearDown(self): self.tmp.cleanup()
    def save(self, root=None):
        return B.save_full(root or self.root/'state', self.m, self.o, self.s, self.state, SPEC, self.ident)
    def step(self): return B.main_step(self.m, self.o, self.s, self.state, self.cache, self.order, CPU)
    def numerical(self): return digest((self.m.state_dict(), self.o.state_dict(), self.s.state_dict(), T.rng_state()))
    def fake_metrics(self, *args):
        return dict(splits={k:dict(exec=dict(denominator=n), CE=dict(CE=.3))
                            for k,n in (('heldout4k',4096),('old991',991))})

    def test_real_parameter_counts_and_normalized_init(self):
        counts={}
        for name, spec in B.SPECS.items():
            m=B.new_model(spec, CPU); counts[name]=sum(p.numel() for p in m.parameters())
            self.assertEqual(counts[name], B.parameter_count(spec)); del m
        self.assertEqual(counts, dict(E=108694019, W=113421827))
        L.seed_all(42); original=C.SimpleTransformer(**SPEC, dropout=.2)
        normalized=B.new_model(SPEC, CPU, .2)
        torch.testing.assert_close(original.tokens.weight, normalized.tokens.weight*SPEC['d_model']**.5)
        self.assertLess(abs(float(normalized.tokens.weight.detach().std())-SPEC['d_model']**-.5), .05)

    def test_dropout_attention_ffn_residual_and_eval(self):
        for layer in [*self.m.transformer.encoder.layers, *self.m.transformer.decoder.layers]:
            self.assertEqual(layer.self_attn.dropout, .2)
            if hasattr(layer, 'multihead_attn'): self.assertEqual(layer.multihead_attn.dropout, .2)
            for module in layer.modules():
                if isinstance(module, torch.nn.Dropout): self.assertEqual(module.p, .2)
        src=torch.tensor([[1,4,2]]); tgt=torch.tensor([[1,8,9]])
        self.assertFalse(torch.equal(self.m(src,tgt), self.m(src,tgt)))
        self.m.eval(); torch.testing.assert_close(self.m(src,tgt), self.m(src,tgt), rtol=0, atol=0)
        m=B.new_model(SPEC,CPU,0.)
        torch.testing.assert_close(m(src,tgt), m(src,tgt), rtol=0, atol=0)

    def test_bit_exact_full_resume_dropout_rng_adam_and_warmup(self):
        for _ in range(3): self.step()
        self.save(); before=self.numerical(); loaded=B.read_full(self.root/'state',self.ident)
        m,o,s,state=B.restore(loaded,CPU)
        self.assertEqual(before,digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state())))
        self.assertEqual(digest(self.state),digest(state))
        expected=[]
        for _ in range(3): self.step(); expected.append(self.numerical())
        m,o,s,state=B.restore(loaded,CPU)
        for fingerprint in expected:
            B.main_step(m,o,s,state,self.cache,self.order,CPU)
            self.assertEqual(fingerprint,digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state())))
        self.assertEqual(state['last_lr'],.001*6/1000)
        state['updates']=1000; self.assertEqual(B.learning_rate(state), .001)

    def test_stable_tamper_recipe_and_counters_rejected(self):
        self.step(); rec=self.save()
        changed=copy.deepcopy(self.ident);changed['recipe']['dropout']=.1
        with self.assertRaises(AssertionError):B.read_full(self.root/'state', changed)
        p=self.root/'state'/rec['file']; payload=torch.load(p,weights_only=True)
        payload['state']['cursor']+=1;torch.save(payload,p)
        rec.update(bytes=p.stat().st_size,sha256=sha(p));write(self.root/'state/latest.json',rec)
        with self.assertRaises(AssertionError):B.read_full(self.root/'state',self.ident)

    def test_weighted_order_reproducible_new_seed_and_tamper(self):
        self.cache.target_lengths[64:]=79
        a,ma=build_order(self.cache,self.root/'a',17);b,mb=build_order(self.cache,self.root/'b',17)
        self.assertTrue(np.array_equal(a,b)); self.assertEqual(ma,mb)
        self.assertTrue(np.array_equal(np.bincount(a),np.r_[np.ones(64),np.full(64,2)]))
        c,_=build_order(self.cache,self.root/'c',18);self.assertFalse(np.array_equal(a,c))
        np.save(self.root/'a/order.npy',np.zeros(len(a),dtype=np.uint32))
        with self.assertRaises(AssertionError):verify_order(self.root/'a',self.cache)

    def test_disposable_probe_restores_all_numerics_and_eval_failure(self):
        self.state.update(next_point_main_seconds=0.,main_seconds=1.);self.save()
        p=B.read_full(self.root/'state',self.ident)
        self.m,self.o,self.s,self.state=B.restore(p,CPU)
        for _ in range(2):self.step()
        expected=self.numerical()
        def evaluate(*args):
            torch.rand(100);np.random.random();return self.fake_metrics()
        for evaluator in (evaluate, lambda *args:(_ for _ in ()).throw(RuntimeError('test eval failure'))):
            out=self.root/('success' if evaluator is evaluate else 'failed')
            result=B.run('E',self.cache,self.order,self.meta,out,CPU,dict(self.cfg,probe_cooldown_seconds=.2),evaluator,resume=self.root/'state',spec=SPEC,cpu=True)
            saved=B.read_full(out,self.ident);m,o,s,state=B.restore(saved,CPU)
            self.assertEqual(expected,digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state())))
            self.assertEqual(len(state['points']),1)
            self.assertEqual(state['points'][0]['complete'],evaluator is evaluate)
            self.assertEqual(state['cursor'],256)

    def test_recovery_restores_checkpoint_and_persists_halving(self):
        self.step();self.step();self.save();numerics=self.numerical()
        h=copy.deepcopy(self.state['stability']);h.update(total=3,consecutive=3,window=[1,1,1])
        m,o,s,state,stop=B.recover(self.root/'state',self.ident,CPU,h,99.,1.,64,'nonfinite threshold')
        self.assertFalse(stop);self.assertEqual(state['lr_factor'],.5);self.assertEqual(state['main_seconds'],99.)
        self.assertEqual(state['cursor'],256);self.assertEqual(state['stability']['escalations'],1)
        self.assertEqual(B.learning_rate(state),.001*.5*3/1000)
        for a,b in zip(self.m.parameters(),m.parameters()):self.assertTrue(torch.equal(a,b))
        B.save_full(self.root/'recovered',m,o,s,state,SPEC,self.ident)
        B.read_full(self.root/'recovered',self.ident)

    def test_nan_update_never_steps_adam_and_threshold_rollback(self):
        self.step();self.save();before=digest((self.m.state_dict(),self.o.state_dict()['state']))
        ce=L.F.cross_entropy
        with patch.object(L.F,'cross_entropy',lambda *a,**k:ce(*a,**k)*float('nan')):
            log,_=self.step()
        self.assertEqual(log['nonfinite_skip'],1)
        self.assertEqual(before,digest((self.m.state_dict(),self.o.state_dict()['state'])))
        actual=R.safe_update;calls=[0]
        def bad(*a,**k):
            calls[0]+=1
            if calls[0]<=3:
                return dict(seconds=.01,nonfinite_skip=1,AMP_skip=0,lr=a[-1],CE=None),a[-2]
            return actual(*a,**k)
        with patch.object(R,'safe_update',bad):
            result=B.run('E',self.cache,self.order,self.meta,self.root/'nan',CPU,dict(self.cfg,max_updates=6),
                         self.fake_metrics,resume=self.root/'state',spec=SPEC,cpu=True)
        self.assertEqual(result['state']['stability']['escalations'],1)
        self.assertEqual(result['state']['lr_factor'],.5)
        self.assertGreater(result['state']['main_seconds'],.03)

    def test_final_cooldown_sqrt_truncation_stable_never_overwritten(self):
        self.step();self.save();stable=self.numerical(); seen=[]
        def evaluate(path,out,deadline,final):
            p=torch.load(path,weights_only=True);self.assertEqual(p['format'],'nlp-hw02-model-only-v1')
            seen.append(final)
            m=self.fake_metrics();m['splits']['old991']=dict(exec=dict(denominator=991),CE=dict(CE=.3))
            m.update(K8_tune2048=dict(complete=True,rows=2048,target_rows=2048),test=dict(rows=1919));return m
        cfg=dict(self.cfg,main_seconds=.000001,max_updates=1,final_session=True)
        result=B.run('E',self.cache,self.order,self.meta,self.root/'final',CPU,cfg,evaluate,
                     resume=self.root/'state',spec=SPEC,cpu=True)
        self.assertTrue(result['final_complete']);self.assertEqual(seen,[True])
        p=B.read_full(self.root/'final',self.ident)
        self.assertEqual(p['state']['updates'],2)
        self.assertEqual(p['state']['points'][-1]['training']['endpoint_lr'],0.)
        self.assertEqual(p['state']['points'][-1]['training']['q'],.1)
        self.assertEqual(p['state']['last_lr'],.001*2/1000)

    def test_v7_model_spec_load_for_both_architecture_shapes(self):
        import final_inference_v7 as V
        for name, spec in B.SPECS.items():
            m=B.new_model(spec,CPU,.2);point=L.save_point(self.root,name,m,spec,self.ident,{})
            loaded=V.load(point,CPU)
            self.assertEqual(len(loaded.transformer.decoder.layers),spec['decoder_layers'])
            src=torch.tensor([[1,4,2]]);tgt=torch.tensor([[1,8]])
            self.assertTrue(torch.isfinite(loaded(src,tgt)).all())
            ctx=V.Fast.Context(loaded,src,4)
            tokens,_,_=V.Speed.decode(ctx,k=1,deadline=time.monotonic()+10,options=V.DECODE_OPTIONS)
            self.assertEqual(tokens.shape[0],1)
            del m,loaded,ctx,tokens

    def test_requested_shapes_match_measured_specs_and_micro_caps(self):
        from shape_specs import spec
        for name, shape, micro in (('E',(8,4,768),64), ('W',(6,6,768),32)):
            expected=spec(shape);expected.pop('dropout');expected.pop('max_positions')
            self.assertEqual(B.SPECS[name],expected);self.assertEqual(B.MICRO_CAPS[name],micro)

    def test_probe_allocates_adam_and_restores_model_rng(self):
        before=digest((self.m.state_dict(),T.rng_state()))
        micro, attempts=B.probe(self.m,self.cache,CPU,.001,'E',cpu=True)
        self.assertEqual(micro,64);self.assertTrue(attempts[-1]['Adam_step_verified'])
        self.assertEqual(before,digest((self.m.state_dict(),T.rng_state())))
        actual=L.update
        def skipped(*args,**kwargs):
            log=actual(*args,**kwargs);log['AMP_skip']=1;return log
        with patch.object(L,'update',skipped),self.assertRaisesRegex(AssertionError,'skipped Adam'):
            B.probe(self.m,self.cache,CPU,.001,'W',cpu=True)
        self.assertEqual(before,digest((self.m.state_dict(),T.rng_state())))

    def test_C_marks_end_main_timed_q0_and_restore(self):
        import r1_growth
        actual=R.safe_update;durations=iter((9000.,19370.,5830.));cooldowns=[]
        def main_update(*args,**kwargs):
            log,micro=actual(*args,**kwargs);log['seconds']=next(durations);return log,micro
        def cool(*args):
            cooldowns.append(args[-1]);return actual(*args[:8])
        cfg=dict(self.cfg,main_seconds=34200.,max_updates=3)
        with patch.object(R,'safe_update',main_update),patch.object(r1_growth,'update',cool):
            result=B.run('E',self.cache,self.order,self.meta,self.root/'marks',CPU,cfg,self.fake_metrics,spec=SPEC,cpu=True)
        self.assertEqual(result['status'],'complete');self.assertTrue(result['end_main_complete'])
        points=result['state']['points']
        self.assertEqual([p['training']['target_main_seconds'] for p in points],[9000.,28370.,34200.])
        self.assertTrue(all(p['complete'] and p['training']['cooldown_seconds']>=.001 for p in points))
        self.assertTrue(cooldowns and all(q==0. for q in cooldowns))
        self.assertEqual(result['state']['updates'],3);self.assertEqual(result['state']['cursor'],384)
        self.assertEqual(B.next_mark(9000.),28370.);self.assertGreater(B.next_mark(28370.),34200.)

    def test_incomplete_old991_or_deadline_cannot_report_complete(self):
        metrics=self.fake_metrics();metrics['splits'].pop('old991')
        self.assertFalse(B.metric_complete(metrics))
        cfg=dict(self.cfg,main_seconds=.000001,max_updates=1)
        result=B.run('E',self.cache,self.order,self.meta,self.root/'missing-old',CPU,cfg,lambda *a:metrics,spec=SPEC,cpu=True)
        self.assertIn('partial',result['status']);self.assertFalse(result['end_main_complete'])
        cfg=dict(self.cfg,deadline_monotonic=time.monotonic()-1)
        result=B.run('E',self.cache,self.order,self.meta,self.root/'expired',CPU,cfg,self.fake_metrics,spec=SPEC,cpu=True)
        self.assertIn('partial',result['status']);self.assertFalse(result['main_target_reached'])
        B.read_full(self.root/'expired',self.ident)

    def test_intermediate_evaluator_runs_both_splits(self):
        import b1_eval as E
        rows={'heldout4k':[{}], 'old991':[{},{}]};refs={k:{} for k in rows};seen=[]
        def gen(model,rows,*a,**kw):seen.append(len(rows));return [],dict(seconds=.1)
        point=self.root/'point.pt';point.write_bytes(b'fixture')
        with patch.object(E.S,'tokenizer',return_value=(None,None)),patch.object(E.E,'load',return_value=None), \
             patch.object(E.E,'generate',gen),patch.object(E.E,'execute',return_value={}), \
             patch.object(E.E,'ce',return_value={'CE':.3}),patch.object(E.N,'compare_exec',return_value={'denominator':1}), \
             patch.object(E.Q,'greedy_journals'):
            result=E.evaluate(point,self.root/'eval',self.root,rows,refs,CPU,time.monotonic()+10)
        self.assertEqual(seen,[1,2]);self.assertEqual(set(result['splits']),set(rows))

    def test_transfer_cli_account_env_and_private_roundtrip_stub(self):
        source=self.root/'source';source.mkdir()
        for gpu,name in enumerate(('E','W')):
            ident=copy.deepcopy(self.ident);ident['model']=name
            B.save_full(source/f'model-GPU{gpu}',self.m,self.o,self.s,self.state,SPEC,ident)
        shutil.copytree(self.root/'order',source/'weighted-order')
        files=[dict(file=p.relative_to(source).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in source.rglob('*') if p.is_file()]
        write(source/'b1-output-manifest.json',dict(files=files))
        client=MockClient(source,self.root/'remote')
        r=transfer('friend2/nlp-hw02-kaggle-b1/1','thisarivson','b1-states-test',self.root/'receipt.json',client,self.root)
        self.assertTrue(r['remote_SHA_verified']);self.assertTrue(r['local_payload_deleted'])
        self.assertFalse(list(self.root.glob('b1-transfer-*')))
        recovered=materialize(self.root/'remote',self.root/'recovered')
        for gpu in (0,1):
            ident=self.ident if gpu==0 else dict(self.ident,model='W')
            self.assertEqual(digest(B.read_full(source/f'model-GPU{gpu}',ident)),
                             digest(B.read_full(recovered/f'model-GPU{gpu}',ident)))
        for owner,path in CONFIG_DIRS.items():
            with patch.dict('os.environ',dict(KAGGLE_KEY='placeholder',KAGGLE_USERNAME='placeholder',KAGGLE_API_TOKEN='placeholder')):
                env=Client().env(owner)
            self.assertEqual(env['KAGGLE_CONFIG_DIR'],str(Path(path).expanduser()))
            self.assertNotIn('KAGGLE_KEY',env);self.assertNotIn('KAGGLE_API_TOKEN',env)
        # A wrong SHA aborts before upload and cleanup removes every temporary payload.
        client=MockClient(source,self.root/'not-uploaded')
        (source/'weighted-order/order.npy').write_bytes(b'corrupted')
        with self.assertRaises(AssertionError):transfer('friend2/x/1','friend1','states',self.root/'bad.json',client,self.root)
        self.assertEqual(client.uploads,0)

    def test_warmup_AMP_skip_changes_group_lr_without_changing_weights_moments(self):
        # This runs a real CPU FP16 autocast/GradScaler overflow, mirroring the CUDA smoke.
        self.s=torch.amp.GradScaler('cpu',init_scale=1024.)
        for _ in range(3): self.step()
        before=digest((self.m.state_dict(),self.o.state_dict()['state']))
        old_lr=self.o.param_groups[0]['lr'];scale=self.s.get_scale()
        hook=next(self.m.parameters()).register_hook(lambda g:torch.full_like(g,float('inf')))
        try:log,_=self.step()
        finally:hook.remove()
        self.assertEqual(log['AMP_skip'],1);self.assertEqual(self.s.get_scale(),scale/2)
        self.assertEqual(before,digest((self.m.state_dict(),self.o.state_dict()['state'])))
        self.assertNotEqual(old_lr,self.o.param_groups[0]['lr'])
        self.assertEqual(self.o.param_groups[0]['lr'],.001*4/1000)

    def test_watchdog_output_transfers_without_aggregate_manifest(self):
        source=self.root/'watchdog';source.mkdir()
        for gpu,name in enumerate(('E','W')):
            ident=dict(self.ident,model=name)
            B.save_full(source/f'model-GPU{gpu}',self.m,self.o,self.s,self.state,SPEC,ident)
        shutil.copytree(self.root/'order',source/'weighted-order')
        client=MockClient(source,self.root/'watchdog-remote')
        result=transfer('friend2/forced-exit/1','thisarivson','watchdog-states',
                        self.root/'watchdog-receipt.json',client,self.root,emergency_receipts=True)
        self.assertTrue(result['passed']);self.assertTrue(result['remote_SHA_verified'])
        recovered=materialize(self.root/'watchdog-remote',self.root/'watchdog-recovered')
        B.read_full(recovered/'model-GPU0',self.ident)
        # An inconsistent order hash aborts before creating a Dataset.
        order=source/'weighted-order/order.json';meta=json.loads(order.read_text());meta['file']['sha256']='0'*64;write(order,meta)
        client=MockClient(source,self.root/'bad-emergency')
        with self.assertRaises(AssertionError):
            transfer('friend2/forced-exit/1','thisarivson','bad',self.root/'bad-emergency.json',
                     client,self.root,emergency_receipts=True)
        self.assertEqual(client.uploads,0)

    def test_transfer_dataset_single_file_zip_and_private_create_cli(self):
        import zipfile
        client=Client();calls=[];folder=self.root/'zip';folder.mkdir()
        def call(args,owner,timeout=1800.):
            calls.append((args,owner))
            if args[:2]==['datasets','download']:
                with zipfile.ZipFile(folder/'model.pt.zip','w') as z:z.writestr('model.pt',b'weights')
        with patch.object(client,'call',call):
            path=client.dataset_file('friend1/states/1','model.pt',folder)
            self.assertEqual(path.read_bytes(),b'weights');self.assertFalse((folder/'model.pt.zip').exists())
            client.create(folder,'friend1')
        self.assertNotIn('--public',calls[-1][0]);self.assertEqual(calls[-1][1],'friend1')

    def test_output_download_explicit_version_and_streamed_exact_file(self):
        import contextlib
        from types import SimpleNamespace as NS
        from b1_output import download
        requests=[]
        def listing(request):
            requests.append(request)
            return NS(files=[NS(file_name='unselected.pt',url='unused')],next_page_token='page2') if len(requests)==1 else NS(
                files=[NS(file_name='model/latest.json',url='test-url')],next_page_token=None)
        @contextlib.contextmanager
        def build():yield NS(kernels=NS(kernels_api_client=NS(list_kernel_session_output=listing)))
        class Stream:
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def raise_for_status(self):pass
            def iter_content(self,n):return iter((b'one',b'two'))
        fake=NS(ApiListKernelSessionOutputRequest=NS)
        with patch.dict(sys.modules,{'kagglesdk.kernels.types.kernels_api_service':fake}):
            path=download(NS(build_kaggle_client=build),'friend2/kernel/7','model/latest.json',self.root,get=lambda *a,**k:Stream())
        self.assertEqual(path.read_bytes(),b'onetwo')
        self.assertEqual(len(requests),2)
        self.assertTrue(all(r.version_label=='v7' and r.user_name=='friend2' for r in requests))
        self.assertEqual(requests[1].page_token,'page2')


if __name__ == '__main__':
    torch.set_num_threads(1)
    names=unittest.defaultTestLoader.getTestCaseNames(Tests);started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(ROOT.parent/'checks/b1-cpu.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
          failures=len(result.failures),errors=len(result.errors),checks=names,seconds=time.monotonic()-started,
          torch=str(torch.__version__),device='CPU',real_parameter_counts={n:B.parameter_count(s) for n,s in B.SPECS.items()},
          GPU_run=False,push=False,runtime_sha256=sha(ROOT/'src/b1_runtime.py')))
    sys.exit(0 if result.wasSuccessful() else 1)
