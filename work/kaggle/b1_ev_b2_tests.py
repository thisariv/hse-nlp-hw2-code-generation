"""CPU checks for real QK attention/caches, masked z-loss, strict resume and frozen EV."""
import copy,json,math,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'src'),str(ROOT.parent.parent/'submission/code')]
import exp26_core as C
import exp26_train as T
import exp27b_fast as Fast
import final_core as FC
import final_fast as FF
import final_inference_v7 as V
import b2_runtime as B
import b2_optim as O
import l1_runtime as L
from b1_data import build_order
from l2_gpu_smoke import digest
from p1_tests import Cache
from kaggle_io import sha,write
from hw2_codegen import model as Clean,decode as Decode
from build_l3 import literals

CPU=torch.device('cpu');torch.set_num_threads(1)
SPEC=dict(d_model=8,heads=2,encoder_layers=1,decoder_layers=1,swiglu_dim=16,vocab=24,qk_norm='layernorm')

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.cache=Cache()
        self.order,self.meta=build_order(self.cache,self.root/'order')
        self.name='W-lr1000';self.ident=B.identity_for(self.name,SPEC,self.cache,self.meta,.2,True)
        self.m=B.new_model(SPEC,CPU,.2);self.o=L.optimizer(self.m,.001,True)
        self.scaler=torch.amp.GradScaler('cpu',enabled=False);self.state=B.new_state(32)
    def tearDown(self):self.tmp.cleanup()
    def step(self):return B.main_step(self.m,self.o,self.scaler,self.state,self.cache,self.order,CPU)
    def save(self):return B.save_full(self.root/'state',self.m,self.o,self.scaler,self.state,SPEC,self.ident)
    def metrics(self,*args):return dict(splits={k:dict(exec=dict(denominator=n),CE=dict(CE=.3)) for k,n in [('heldout4k',4096),('old991',991)]})

    def test_headwise_norm_forward_backward_and_masks(self):
        m=B.new_model(SPEC,CPU).eval();a=m.transformer.decoder.layers[0].self_attn
        x=torch.randn(2,2,5,4)*100+30;y=a.q_norm(x)
        torch.testing.assert_close(y.mean(-1),torch.zeros(2,2,5),atol=2e-6,rtol=0)
        torch.testing.assert_close(y.square().mean(-1),torch.ones(2,2,5),atol=2e-6,rtol=0)
        for module in m.modules():
            if hasattr(module,'q_norm'):module.record_logits=True
        src=torch.tensor([[1,4,2,0],[1,5,6,2]]);tgt=torch.tensor([[1,6,7],[1,8,9]])
        out=m(src,tgt);out.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
        self.assertIsNotNone(a.q_norm.weight.grad)
        changed=tgt.clone();changed[:,-1]=20
        torch.testing.assert_close(out[:,:-1],m(src,changed)[:,:-1],rtol=0,atol=0)
        extended=torch.cat([src,torch.zeros(2,2,dtype=torch.long)],1)
        torch.testing.assert_close(out,m(extended,tgt),rtol=2e-5,atol=2e-5)
        self.assertTrue(math.isfinite(float(a.sampled_logit_max)))
        with self.assertRaises(ValueError):C.SimpleTransformer(**dict(SPEC,qk_norm='bad'))

    def test_cached_logits_match_full_prefix_training_eval_and_clean_v7(self):
        m=B.new_model(SPEC,CPU).eval();src=torch.tensor([[1,4,2,0],[1,5,6,2]])
        tokens=torch.tensor([[1,7,8,9],[1,9,10,11]])
        final=FC.SimpleTransformer(**SPEC,dropout=0.).eval();final.load_state_dict(m.state_dict())
        clean=Clean.from_spec(SPEC).eval();clean.load_state_dict(m.state_dict())
        decoders=[Fast.Decoder(Fast.Context(m,src,8),1),FF.Decoder(FF.Context(final,src,8),1),
                  Decode.BaseDecoder(Decode.Context(clean,src,8),1),C.CachedDecoder(m,src,8)]
        for t in range(tokens.shape[1]):
            full=m(src,tokens[:,:t+1])[:,-1]
            for d in decoders:torch.testing.assert_close(d.step(tokens[:,t]),full,rtol=3e-5,atol=3e-5)

    def test_z_loss_mask_fp32_shift_gradient_and_microbatch_weight(self):
        logits=torch.randn(2,3,7,requires_grad=True);labels=torch.tensor([[3,2,0],[4,5,2]])
        ce,z=O.loss_terms(logits,labels,5)
        expected=1e-4*torch.logsumexp(logits.float(),-1).square()[labels.ne(0)].mean()
        torch.testing.assert_close(z,expected)
        ce2,z2=O.loss_terms(logits+5,labels,5);torch.testing.assert_close(ce,ce2)
        self.assertGreater(float(z2.detach()),float(z.detach()));(ce+z).backward()
        self.assertTrue(torch.equal(logits.grad[0,2],torch.zeros(7)))
        big=torch.full((1,1,7),1000.,dtype=torch.float16,requires_grad=True)
        a,b=O.loss_terms(big,torch.tensor([[2]]),1);self.assertEqual(b.dtype,torch.float32)
        (a+b).backward();self.assertTrue(torch.isfinite(big.grad).all())
        initial=copy.deepcopy(self.m.state_dict());logs=[];states=[]
        for micro in (128,32):
            m=B.new_model(SPEC,CPU);m.load_state_dict(initial);o=L.optimizer(m,.001,True)
            log,_=O.safe_update(m,o,self.scaler,self.cache,self.order[:128],CPU,micro,0.)
            logs.append(log);states.append({n:p.grad.clone() for n,p in m.named_parameters()})
        self.assertAlmostEqual(logs[0]['CE'],logs[1]['CE'],places=5)
        self.assertAlmostEqual(logs[0]['z_loss'],logs[1]['z_loss'],places=6)
        for k in states[0]:torch.testing.assert_close(states[0][k],states[1][k],rtol=2e-4,atol=2e-6)

    def test_resume_qk_adam_rng_warmup_and_recovery(self):
        for _ in range(3):self.step()
        self.save();p=B.read_full(self.root/'state',self.ident)
        self.step();expected=digest((self.m.state_dict(),self.o.state_dict(),self.scaler.state_dict(),T.rng_state()))
        m,o,s,state=B.restore(p,CPU);B.main_step(m,o,s,state,self.cache,self.order,CPU)
        self.assertEqual(expected,digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state())))
        self.assertEqual(state['last_lr'],.001*4/2000)
        history=copy.deepcopy(state['stability']);history['total']=5
        m,o,s,state,stop=B.recover(self.root/'state',self.ident,CPU,history,12.,1.,32,'nonfinite threshold')
        self.assertFalse(stop);self.assertEqual(state['lr_factor'],.5);self.assertEqual(state['main_seconds'],12.)
        self.assertEqual(state['last_lr'],.001*.5*3/2000)
        other=copy.deepcopy(self.ident);other['recipe']['z_loss']=0.
        with self.assertRaises(AssertionError):B.read_full(self.root/'state',other)

    def test_strict_model_spec_load_v7_clean_and_legacy(self):
        for flag in (False,'layernorm'):
            spec=dict(SPEC,qk_norm=flag);m=B.new_model(spec,CPU).eval()
            point=L.save_point(self.root, 'load-'+str(flag),m,spec,self.ident,{})
            loaded=V.load(point,CPU);clean=Clean.from_spec(spec).eval();clean.load_state_dict(torch.load(point,weights_only=True)['model'],strict=True)
            src=torch.tensor([[1,4,2]]);tgt=torch.tensor([[1,5]])
            torch.testing.assert_close(loaded(src,tgt),clean(src,tgt),rtol=0,atol=0)
            if flag:
                p=torch.load(point,weights_only=True);p['model_spec']['qk_norm']=False;torch.save(p,point)
                with self.assertRaises(RuntimeError):V.load(point,CPU)

    def test_pair_probe_and_end_budget_reserve_restore(self):
        self.state.update(main_seconds=1.,next_point_pairs=128);self.save()
        p=B.read_full(self.root/'state',self.ident);m,o,s,state=B.restore(p,CPU)
        for _ in range(2):B.main_step(m,o,s,state,self.cache,self.order,CPU)
        control=digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state()))
        cfg=dict(owner='test',dropout=.2,deadline_monotonic=time.monotonic()+120,io_reserve_seconds=0,
                 main_seconds=9999.,max_updates=2,checkpoint_seconds=999.,probe_cooldown_seconds=.001,probe_eval_seconds=1.,final_session=False)
        calls=[]
        def evaluate(*args):calls.append(1);torch.rand(10);return self.metrics()
        result=B.run(self.name,self.cache,self.order,self.meta,self.root/'run',CPU,cfg,evaluate,resume=self.root/'state',spec=SPEC,cpu=True)
        self.assertEqual(result['status'],'complete');self.assertFalse(result['main_target_reached'])
        self.assertEqual(len(calls),2);self.assertEqual(result['state']['points'][0]['training']['main_pairs'],128)
        m,o,s,state=B.restore(B.read_full(self.root/'run',self.ident),CPU)
        self.assertEqual(control,digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state())))
        self.assertEqual(B.PAIR_MARK,math.ceil(3100000/128)*128)

    def test_qk_zloss_cpu_amp_overflow_skips_adam(self):
        m=B.new_model(SPEC,CPU);o=L.optimizer(m,.001,True)
        scaler=torch.amp.GradScaler('cpu',init_scale=1.)
        ids=L.take(self.order,0,128)
        log,_=O.safe_update(m,o,scaler,self.cache,ids,CPU,32,.001)
        self.assertFalse(log['AMP_skip']);self.assertGreater(log['z_loss'],0.)
        before=digest((m.state_dict(),o.state_dict()['state']));scale=scaler.get_scale()
        hook=next(m.parameters()).register_hook(lambda g:torch.full_like(g,float('inf')))
        try:log,_=O.safe_update(m,o,scaler,self.cache,ids,CPU,32,.001)
        finally:hook.remove()
        self.assertTrue(log['AMP_skip']);self.assertEqual(scaler.get_scale(),scale/2)
        self.assertEqual(before,digest((m.state_dict(),o.state_dict()['state'])))
        json.dumps(log,allow_nan=False)

    def test_b2_transfer_materialize_and_strict_resume(self):
        from b1_tests import MockClient
        from b1_transfer import transfer
        from b1_resume import materialize
        import shutil
        source=self.root/'source';shutil.copytree(self.root/'order',source/'weighted-order')
        for gpu,name in enumerate(('W-lr1000','W-lr600')):
            identity=B.identity_for(name,SPEC,self.cache,self.meta,cpu=True)
            m=B.new_model(SPEC,CPU);o=L.optimizer(m,B.PEAKS[name],True)
            state=B.new_state(32,B.PEAKS[name]);scaler=torch.amp.GradScaler('cpu',enabled=False)
            B.main_step(m,o,scaler,state,self.cache,self.order,CPU)
            B.save_full(source/f'model-GPU{gpu}',m,o,scaler,state,SPEC,identity)
        write(source/'b1-output-manifest.json',dict(files=[dict(file=p.relative_to(source).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in source.rglob('*') if p.is_file()]))
        receipt=transfer('friend2/b2/1','thisarivson','b2-resume-test',self.root/'receipt.json',
                         client=MockClient(source,self.root/'remote'))
        self.assertTrue(receipt['remote_SHA_verified'])
        recovered=materialize(self.root/'remote',self.root/'recovered')
        for gpu,name in enumerate(('W-lr1000','W-lr600')):
            identity=B.identity_for(name,SPEC,self.cache,self.meta,cpu=True)
            self.assertEqual(digest(B.read_full(source/f'model-GPU{gpu}',identity)),
                             digest(B.read_full(recovered/f'model-GPU{gpu}',identity)))

    def test_ev_frozen_runtime_and_cooldown_only_cpu(self):
        nb=json.loads((ROOT/'notebooks/nlp-hw02-kaggle-b1.ipynb').read_text())
        sources=literals(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))['SOURCES']
        module=self.root/'frozen';module.mkdir()
        for n,s in sources.items():(module/n).write_text(s)
        (module/'b1_ev.py').write_bytes((ROOT/'src/b1_ev.py').read_bytes())
        script='''import sys,time,json,torch
from pathlib import Path
sys.path.insert(0,sys.argv[1]);sys.path.append(sys.argv[2])
import b1_runtime as B
from p1_tests import Cache
from b1_data import build_order
import l1_runtime as L
from b1_ev import cooldown
from kaggle_io import sha
root=Path(sys.argv[3]);torch.set_num_threads(1);cache=Cache();order,meta=build_order(cache,root/'order')
spec=dict(d_model=8,heads=2,encoder_layers=1,decoder_layers=1,swiglu_dim=16,vocab=24)
m=B.new_model(spec,torch.device('cpu'));o=L.optimizer(m,.001,True);s=torch.amp.GradScaler('cpu',enabled=False)
state=B.new_state(32);state.update(main_seconds=34200.,last_lr=.00025)
identity=B.identity_for('E',spec,cache,meta,cpu=True);B.save_full(root/'anchor',m,o,s,state,spec,identity)
hash=sha(root/'anchor/latest.json')
point,training=cooldown(root/'anchor',cache,order,torch.device('cpu'),root/'ev',time.monotonic()+20,seconds=.02)
assert point.is_file() and training['new_main_updates']==0 and training['cooldown_seconds']>=.02
assert training['main_seconds']==34200 and hash==sha(root/'anchor/latest.json')
'''
        r=subprocess.run([sys.executable,'-c',script,str(module),str(ROOT),str(self.root)],capture_output=True,text=True,timeout=60)
        self.assertEqual(r.returncode,0,r.stderr[-3000:])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(ROOT.parent/'checks/b1-ev-b2-cpu.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
          failures=[(str(t),s) for t,s in result.failures+result.errors],torch=str(torch.__version__),GPU_runs=0))
    raise SystemExit(not result.wasSuccessful())
