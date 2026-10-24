"""B4 objective, state/stream equality, disposable restore and failure contracts."""
import copy,json,time,unittest,random
from pathlib import Path
from unittest.mock import patch
import b3_tests as base
import b4_runtime as G
import b4_math as M
import b2_runtime as B
import exp26_train as T
import l2_runtime as R
import l1_runtime as L
import numpy as np
import torch
from exp26_order import collate
from l2_gpu_smoke import digest
from b3_gpu_smoke import BranchAudit
from kaggle_io import write
CPU=torch.device('cpu')
from kaggle_io import sha as base_module_sha


class Tests(base.Tests):
    def init_mature(self):
        self.name='W-lr600';self.ident=B.identity_for(self.name,base.base.SPEC,self.cache,self.meta,0.,True)
        self.m=B.new_model(base.base.SPEC,CPU,0.);self.o=L.optimizer(self.m,.0006,True)
        self.state=B.new_state(32,.0006);self.state.update(updates=2000,cursor=256000,main_seconds=59400.,last_lr=.0006)
        self.step()
        for v in self.o.state.values():v['step'].fill_(self.state['updates'])
    def variable(self):
        for i in range(128):self.cache.rows[i]=(self.cache.rows[i][0],np.asarray([8,9,10]*(1+i%4)))
        self.cache.target_lengths=np.asarray([len(t) for _,t in self.cache.rows])
    def test_b4_mask_stratification_top_ties_and_singletons(self):
        lengths=np.repeat([20,60,90,150],32);scores=np.arange(128,dtype=float)
        keep,stats=M.truncation_mask(scores,lengths)
        self.assertEqual(stats['excluded'],[4,3,3,3]);self.assertEqual(int((~keep).sum()),13)
        for b in range(4):
            pos=np.arange(b*32,(b+1)*32)
            self.assertGreaterEqual(scores[pos[~keep[pos]]].min(),scores[pos[keep[pos]]].max())
        a,_=M.truncation_mask(np.ones(128),lengths);b,_=M.truncation_mask(np.ones(128),lengths)
        self.assertTrue(np.array_equal(a,b))
        a,_=M.truncation_mask([1,100,2],[20,150,20],q=.5);self.assertTrue(a[1])
    def test_b4_two_pass_matches_full_batch_masked_oracle_gradients_and_normalization(self):
        self.init_mature();self.variable();anchor=copy.deepcopy(B.payload(self.m,self.o,self.scaler,self.state,base.base.SPEC,self.ident))
        src,tgt,labels,_=collate(self.cache,self.order,CPU)
        with torch.no_grad():
            logits=self.m(src,tgt)
            ce=torch.nn.functional.cross_entropy(logits.transpose(1,2),labels,ignore_index=0,reduction='none')
            counts=labels.ne(0).sum(1);scores=(ce.sum(1)/counts).numpy()
        lengths=np.asarray([len(self.cache[int(i)][1]) for i in self.order])
        self.assertTrue(np.array_equal(counts.numpy(),lengths+1))  # EOS in, PAD out.
        keep,_=M.truncation_mask(scores,lengths);tokens=int(counts[keep].sum())
        logits=self.m(src,tgt)
        ce=torch.nn.functional.cross_entropy(logits.transpose(1,2),labels,ignore_index=0,reduction='none')
        z=1e-4*torch.logsumexp(logits,dim=-1).square().masked_fill(labels.eq(0),0.)
        expected_ce=ce[keep].sum()/tokens;expected_z=z[keep].sum()/tokens
        self.o.zero_grad(set_to_none=True);(expected_ce+expected_z).backward()
        raw={k:p.grad.clone() for k,p in self.m.named_parameters()}
        norm=torch.nn.utils.clip_grad_norm_(self.m.parameters(),1.)
        expected={k:p.grad.clone() for k,p in self.m.named_parameters()}
        self.m,self.o,self.scaler,self.state=G.restore(anchor,CPU)
        seen=[];actual_raw={};clip=torch.nn.utils.clip_grad_norm_
        def capture(params,*a,**kw):
            actual_raw.update({k:p.grad.clone() for k,p in self.m.named_parameters()})
            return clip(params,*a,**kw)
        hook=self.m.register_forward_pre_hook(lambda m,a:seen.append((torch.is_grad_enabled(),len(a[0]))))
        try:
            with patch.object(torch.nn.utils,'clip_grad_norm_',side_effect=capture) as clipping, \
                 patch.object(self.scaler,'unscale_',wraps=self.scaler.unscale_) as unscale, \
                 patch.object(self.scaler,'step',wraps=self.scaler.step) as step, \
                 patch.object(self.scaler,'update',wraps=self.scaler.update) as update:
                actual,_=M.safe_update(self.m,self.o,self.scaler,self.cache,self.order,CPU,32,.0006)
                for call in (clipping,unscale,step,update):self.assertEqual(call.call_count,1)
        finally:hook.remove()
        self.assertEqual(sum(n for grad,n in seen if not grad),128)
        self.assertEqual(sum(n for grad,n in seen if grad),115)
        self.assertTrue(all(n<=32 for grad,n in seen));self.assertEqual(len(seen),8)
        self.assertEqual(actual['truncation']['excluded_positions'],np.flatnonzero(~keep).tolist())
        self.assertEqual(actual['tokens'],tokens)
        self.assertAlmostEqual(actual['CE'],float(expected_ce.detach()),places=6)
        self.assertAlmostEqual(actual['z_loss'],float(expected_z.detach()),places=7)
        self.assertAlmostEqual(actual['gradient_norm'],float(norm),places=5)
        for k,p in self.m.named_parameters():
            torch.testing.assert_close(actual_raw[k],raw[k],rtol=3e-4,atol=2e-7)
            torch.testing.assert_close(p.grad,expected[k],rtol=3e-4,atol=2e-7)
    def test_b4_selection_RNG_restored_before_training_and_on_failure(self):
        self.init_mature();before=digest(T.rng_state());select=M.per_example_ce
        def random_selection(*a,**kw):
            random.random();np.random.random();torch.rand(8)
            return select(*a,**kw)
        with patch.object(M,'per_example_ce',side_effect=random_selection):
            log,_=M.safe_update(self.m,self.o,self.scaler,self.cache,self.order,CPU,32,.0006)
        self.assertEqual(before,digest(T.rng_state()));self.assertTrue(log['truncation']['selection_rng_restored'])
        def failed(*a,**kw):
            random.random();np.random.random();torch.rand(8);raise FloatingPointError('selection NaN')
        with patch.object(M,'per_example_ce',side_effect=failed):
            log,_=M.safe_update(self.m,self.o,self.scaler,self.cache,self.order,CPU,32,.0006)
        self.assertEqual(log['nonfinite_skip'],1);self.assertEqual(before,digest(T.rng_state()))
    def test_b4_real_NaN_selection_and_retained_loss_never_update_Adam(self):
        self.init_mature();before=self.numerical()
        for pass_grad in (False,True):
            def corrupt(module,args,out):
                return out*float('nan') if torch.is_grad_enabled()==pass_grad else out
            hook=self.m.register_forward_hook(corrupt)
            try:log,_=M.safe_update(self.m,self.o,self.scaler,self.cache,self.order,CPU,32,.0006)
            finally:hook.remove()
            self.assertEqual(log['nonfinite_skip'],1);self.assertEqual(before,self.numerical())
            self.assertTrue(all(p.grad is None for p in self.m.parameters()))
    def test_b4_q0_main_is_bit_exact_and_both_streams_consume_all128(self):
        self.init_mature();pin=self.save();p=B.read_full(self.root/'state',self.ident)
        log,_=B.main_step(self.m,self.o,self.scaler,self.state,self.cache,self.order,CPU)
        expected=self.numerical();self.state.pop('main_seconds');counters=copy.deepcopy(self.state)
        self.m,self.o,self.scaler,self.state=G.restore(p,CPU)
        G.main_step(self.m,self.o,self.scaler,self.state,self.cache,self.order,CPU,0.)
        self.assertEqual(expected,self.numerical());self.state.pop('main_seconds');self.assertEqual(counters,self.state)
        streams=[]
        for q in (0.,.1):
            self.m,self.o,self.scaler,self.state=G.restore(p,CPU);rows=[]
            for step in range(3):
                ids=L.take(self.order,self.state['cursor'],128).copy();rng=digest(T.rng_state())
                log,_=G.main_step(self.m,self.o,self.scaler,self.state,self.cache,self.order,CPU,q)
                self.assertEqual(self.state['cursor'],pin['cursor']+(step+1)*128);self.assertEqual(log['pairs'],128)
                self.assertEqual(rng,digest(T.rng_state()));rows.append(ids)
            streams.append(np.concatenate(rows))
        self.assertTrue(np.array_equal(*streams))
    def test_b4_nonfinite_loss_or_backward_never_updates_adam(self):
        self.init_mature();before=digest((self.m.state_dict(),self.o.state_dict(),self.scaler.state_dict()))
        with patch.object(M,'truncation_mask',side_effect=FloatingPointError('injected')):
            log,_=M.safe_update(self.m,self.o,self.scaler,self.cache,self.order,CPU,32,.0006)
        self.assertEqual(log['nonfinite_skip'],1);self.assertEqual(before,digest((self.m.state_dict(),self.o.state_dict(),self.scaler.state_dict())))
        self.assertTrue(all(p.grad is None for p in self.m.parameters()))
        with patch.object(torch.Tensor,'backward',side_effect=FloatingPointError('injected after selection')):
            log,_=M.safe_update(self.m,self.o,self.scaler,self.cache,self.order,CPU,32,.0006)
        self.assertEqual(log['nonfinite_skip'],1);self.assertEqual(before,digest((self.m.state_dict(),self.o.state_dict(),self.scaler.state_dict())))
    def test_b4_two_pass_oom_rng_scaler_grad_reset(self):
        self.init_mature();anchor=copy.deepcopy(B.payload(self.m,self.o,self.scaler,self.state,base.base.SPEC,self.ident))
        expected,_=M.safe_update(self.m,self.o,self.scaler,self.cache,self.order,CPU,16,.0006);finger=self.numerical()
        self.m,self.o,self.scaler,self.state=G.restore(anchor,CPU)
        original=M.two_pass_update;count=0
        def fail(*a,**kw):
            nonlocal count
            count+=1
            if count==1:
                for p in self.m.parameters():p.grad=torch.ones_like(p)
                torch.rand(8);raise torch.cuda.OutOfMemoryError('two-pass fixture')
            return original(*a,**kw)
        with patch.object(M,'two_pass_update',side_effect=fail):actual,micro=M.safe_update(self.m,self.o,self.scaler,self.cache,self.order,CPU,32,.0006)
        self.assertEqual(micro,16);self.assertEqual(finger,self.numerical())
        self.assertEqual(actual['truncation']['forward_mode'],'two-pass');self.assertEqual(len(actual['OOM_retries']),1)
    def test_b4_adam_oom_is_unsafe(self):
        self.init_mature()
        with patch.object(self.scaler,'step',side_effect=torch.cuda.OutOfMemoryError('Adam fixture')):
            with self.assertRaises(L.UnsafeAdamOOM):M.safe_update(self.m,self.o,self.scaler,self.cache,self.order,CPU,32,.0006)
    def test_b4_enabled_AMP_overflow_skips_Adam_and_halves_scale(self):
        self.init_mature();self.scaler=torch.amp.GradScaler('cpu',enabled=True,init_scale=1024)
        before=digest((self.m.state_dict(),self.o.state_dict()['state']))
        hook=next(self.m.parameters()).register_hook(lambda g:torch.full_like(g,float('inf')))
        try:log,_=M.safe_update(self.m,self.o,self.scaler,self.cache,self.order,CPU,32,.0006)
        finally:hook.remove()
        self.assertEqual(log['AMP_skip'],1);self.assertEqual(log['nonfinite_skip'],0)
        self.assertEqual(self.scaler.get_scale(),512);self.assertEqual(before,digest((self.m.state_dict(),self.o.state_dict()['state'])))
    def test_b4_entire_smoke_controller_on_CPU_with_real_updates_and_restore(self):
        import b4_gpu_smoke as S
        self.init_mature();self.scaler=torch.amp.GradScaler('cpu',enabled=True,init_scale=1024)
        pin=self.save();cfg=self.cfg(gpu_smoke=True,truncation_q=.1,smoke_updates=200)
        sets=dict(heldout4k=list(range(4096)),old991=list(range(991)))
        refs={k:dict(rows=v) for k,v in sets.items()}
        def metrics(*args):
            return dict(splits={k:dict(exec=dict(denominator=n),CE=dict(CE=.3)) for k,n in [('heldout4k',64),('old991',16)]},
                        K8_tune2048=dict(complete=True,rows=36,target_rows=36),test=dict(rows=64))
        with patch.object(G,'read_anchor',side_effect=lambda root,pin,*_:B.read_full(root,pin['identity'])), \
             patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'max_memory_reserved',return_value=0), \
             patch.object(torch.cuda,'max_memory_allocated',return_value=0),patch.object(S.E,'evaluate',side_effect=metrics):
            S.exercise('W600-trunc',self.cache,self.order,self.meta,self.root/'smoke',CPU,cfg,None,sets,refs,self.root/'state',pin)
        report=json.loads((self.root/'smoke/gpu-smoke.json').read_text())
        self.assertTrue(report['passed']);self.assertEqual(report['main_updates'],200)
        self.assertEqual(report['nonfinite_skips'],0)
        self.assertEqual(len(report['restore_audits']),2);self.assertEqual(len(report['sampled_throughput']['samples']),9)
        self.assertTrue(all(r['forward_mode']=='two-pass' for r in report['sampled_throughput']['samples']))
    def test_b4_smoke_rejects_short_runs(self):
        import b4_gpu_smoke as S
        with self.assertRaisesRegex(AssertionError,'hundreds'):
            S.exercise('W600',None,None,None,self.root/'short',CPU,
                       dict(truncation_q=0.,smoke_updates=64),None,None,None,None,None)
    def test_b4_smoke_rejects_nonfinite_main_even_below_recovery_threshold(self):
        original=G.main_step;first=True
        def skipped(*a,**kw):
            nonlocal first
            if first:
                first=False
                with patch.object(M,'truncation_mask',side_effect=FloatingPointError('injected NaN selection')):
                    return original(*a,**kw)
            return original(*a,**kw)
        with patch.object(G,'main_step',side_effect=skipped):
            with self.assertRaisesRegex(AssertionError,'Main smoke rejected a nonfinite'):
                self.test_b4_entire_smoke_controller_on_CPU_with_real_updates_and_restore()
    def test_b4_static_handoff_requires_same_source_smoke_and_preserves_launch(self):
        from b4_static_check import reviewed_handoff
        fresh=lambda:dict(modules={'math':'A'},GPU_runs=0,pushes=0)
        old=dict(modules={'math':'A'},GPU_runs=2,pushes=2,production_url='private',ETA_utc='tomorrow',status='RUNNING')
        self.assertFalse(reviewed_handoff(fresh(),{},[])['ready'])
        self.assertFalse(reviewed_handoff(fresh(),old,[dict(passed=False,modules={'math':'A'})])['ready'])
        self.assertFalse(reviewed_handoff(fresh(),old,[dict(passed=True,modules={'math':'B'})])['ready'])
        actual=reviewed_handoff(fresh(),old,[dict(passed=True,modules={'math':'A'},receipt='verified')])
        self.assertTrue(actual['ready']);self.assertFalse(actual['GPU_smoke_pending'])
        self.assertEqual(actual['status'],'RUNNING');self.assertEqual(actual['pushes'],2)
        self.assertEqual(actual['production_url'],'private');self.assertEqual(actual['ETA_utc'],'tomorrow')
        self.assertNotIn('production_url',reviewed_handoff(fresh(),dict(old,modules={'math':'B'}),[]))
    def test_b4_supervisor_publication_remains_watched_and_soft(self):
        import b4_session as S
        from unittest.mock import Mock
        watchdog=Mock();events=[]
        def base(cfg,worker,guard,**kw):
            guard.finish();self.assertFalse(watchdog.finish.called)
            write(self.root/'b1-output-manifest.json',dict(files=[]))
            return dict(status='partial',unreaped_worker_pids=[],exit_codes=[0,0])
        def audited(root):
            self.assertFalse(watchdog.finish.called);return dict(passed=False)
        with patch.object(S,'base_launch',side_effect=base),patch.object(S,'audit',side_effect=audited):
            result=S.launch(dict(working_root=str(self.root)),'worker',watchdog)
        self.assertEqual(result['status'],'partial');watchdog.finish.assert_called_once()
        manifest=json.loads((self.root/'b1-output-manifest.json').read_text())
        for row in manifest['files']:self.assertEqual(base_module_sha(self.root/row['file']),row['sha256'])
    def run_b4(self,q,evaluate=None,**kw):
        pin=self.save();origin=self.state['main_seconds'];normal=B.O.safe_update;truncated=M.safe_update
        def fixed(fn):
            def call(*a,**k):
                log,micro=fn(*a,**k);log['seconds']=1.;return log,micro
            return call
        cfg=self.cfg(main_seconds=9.,probe_main_marks=[origin+3,origin+6],truncation_q=q,**kw)
        audit=BranchAudit(self.root)
        def ev(*a):random.random();np.random.random();torch.rand(3);return self.metrics()
        with patch.object(B.O,'safe_update',side_effect=fixed(normal)),patch.object(M,'safe_update',side_effect=fixed(truncated)):
            result=G.run(self.name,self.cache,self.order,self.meta,self.root/'run',CPU,cfg,evaluate or ev,self.root/'state',pin,cpu=True,branch_audit=audit)
        return result,audit
    def test_b4_both_main_and_cooldowns_truncate_restore_all_fields_and_history(self):
        for q in (0.,.1):
            with self.subTest(q=q):
                self.init_mature();pin=self.save();anchor=B.read_full(self.root/'state',self.ident)
                for _ in range(9):G.main_step(self.m,self.o,self.scaler,self.state,self.cache,self.order,CPU,q)
                control=self.numerical();self.m,self.o,self.scaler,self.state=G.restore(anchor,CPU)
                result,audit=self.run_b4(q)
                self.assertEqual(result['status'],'complete');self.assertEqual(len(audit.rows),3)
                self.assertTrue(all(r['before']==r['restored'] for r in audit.rows))
                identity=G.branch_identity(self.ident,q,pin['sha256'])
                # save() gives a fresh container SHA but same source payload; use returned identity.
                identity=result['identity'];self.m,self.o,self.scaler,state=G.restore(B.read_full(self.root/'run',identity),CPU)
                self.assertEqual(control,self.numerical());self.assertEqual(state['cursor'],anchor['state']['cursor']+9*128)
                self.assertEqual(state['last_lr'],.0006);self.assertEqual(state['lr_factor'],1.)
                self.assertTrue(all(p['training']['q']==q for p in state['points'][-3:]))
                for point in state['points'][-3:]:
                    rows=[json.loads(line) for line in (self.root/'run/points'/point['tag']/'cooldown-updates.jsonl').read_text().splitlines()]
                    self.assertEqual('truncation' in rows[0],bool(q));self.assertEqual(rows[0]['lr'],.0006)
                if q:self.assertEqual(result['truncation_totals']['excluded'],9*13)
                import shutil;shutil.rmtree(self.root/'run')
    def test_b4_partial_eval_keeps_stable_and_final_FP16(self):
        self.init_mature();result,audit=self.run_b4(.1,evaluate=lambda *a:{})
        self.assertTrue(result['status'].startswith('partial'));self.assertFalse(result['final_complete'])
        self.assertTrue((self.root/'run/latest.json').exists());self.assertEqual(len(list((self.root/'run/weights').glob('*.pt'))),3)
        self.assertEqual(len(audit.rows),3)
    def test_b4_stability_stops_without_halving_lr(self):
        self.init_mature();pin=self.save()
        cfg=self.cfg(main_seconds=999.,truncation_q=.1)
        original=G.main_step
        def escalate(*a,**kw):log,_=original(*a,**kw);return log,True
        with patch.object(G,'main_step',side_effect=escalate):
            result=G.run(self.name,self.cache,self.order,self.meta,self.root/'run',CPU,cfg,self.metrics,self.root/'state',pin,cpu=True)
        self.assertTrue(result['status'].startswith('partial'));self.assertEqual(result['state']['lr_factor'],1.)
        self.assertEqual(result['state']['last_lr'],.0006);self.assertEqual(result['state']['cursor'],pin['cursor'])
    def test_b4_AB_audit_detects_start_or_stream_difference(self):
        from b4_session import audit
        for gpu in (0,1):
            folder=self.root/f'model-GPU{gpu}';folder.mkdir()
            write(folder/'start-fingerprints.json',dict(source_sha256='same',fingerprints={'rng':'same'}))
            (folder/'updates.jsonl').write_text(json.dumps(dict(updates=5,stream_sha256='same'))+'\n')
        self.assertTrue(audit(self.root)['passed'])
        (self.root/'model-GPU1/updates.jsonl').write_text(json.dumps(dict(updates=5,stream_sha256='wrong'))+'\n')
        self.assertFalse(audit(self.root)['passed'])


if __name__=='__main__':
    suite=unittest.TestSuite(Tests(n) for n in sorted(Tests.__dict__) if n.startswith('test_b4_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(base.base.ROOT.parent/'checks/b4-cpu.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
          failures=[(str(t),s) for t,s in result.failures+result.errors],GPU=False,
          modules={p.name:base_module_sha(p) for p in (base.base.ROOT/'src').glob('b4_*.py')}))
    raise SystemExit(not result.wasSuccessful())
