"""B3 regressions: actual tiny QK training, cumulative probes, restore, wall reserve."""
import copy, json, time, unittest
from unittest.mock import patch
import b1_ev_b2_tests as base
import b3_runtime as G
import b2_runtime as B
import exp26_train as T
import l2_runtime as R
from l2_gpu_smoke import digest


class Tests(base.Tests):
    # Reuse fixtures without inheriting and rerunning all B2 tests here.
    def metrics(self,*args):
        return dict(splits={k:dict(exec=dict(denominator=n),CE=dict(CE=.3)) for k,n in [('heldout4k',4096),('old991',991)]},
                    K8_tune2048=dict(complete=True,rows=2048,target_rows=2048),test=dict(rows=1919))
    def cfg(self,**kw):
        return dict(dict(owner='test',dropout=.2,deadline_monotonic=time.monotonic()+1000,
                         io_reserve_seconds=0.,main_seconds=0.,probe_main_marks=[],final_cooldown_seconds=.01,
                         final_eval_seconds=1.,probe_cooldown_seconds=.01,probe_eval_seconds=1.,gpu_smoke=False),**kw)
    def run_b3(self,cfg,evaluate=None):
        pin=self.save()
        return G.run(self.name,self.cache,self.order,self.meta,self.root/'run',base.CPU,cfg,
                     evaluate or self.metrics,self.root/'state',pin,cpu=True)
    def numerical(self): return digest((self.m.state_dict(),self.o.state_dict(),self.scaler.state_dict(),T.rng_state()))

    def test_b3_final_linear_restore_and_history(self):
        self.step(); self.step(); before=self.numerical(); history=copy.deepcopy(self.state)
        rates=[]; original=B.O.safe_update
        def record(*args,**kw):
            out=original(*args,**kw); rates.append(args[-1]); return out
        with patch.object(B.O,'safe_update',side_effect=record): result=self.run_b3(self.cfg())
        self.assertEqual(result['status'],'complete'); self.assertTrue(result['final_complete'])
        m,o,s,state=G.restore(B.read_full(self.root/'run',self.ident),base.CPU)
        self.assertEqual(before,digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state())))
        for k in ('updates','cursor','warmup_updates','stability','last_lr'): self.assertEqual(state[k],history[k])
        self.assertEqual(rates[0],history['last_lr'])
        self.assertEqual(state['points'][-1]['training']['cooldown'],'C timed linear')

    def test_b3_two_cumulative_probes_and_final_bit_exact(self):
        import random, numpy as np, torch
        from b3_gpu_smoke import BranchAudit
        self.state.update(updates=2000,cursor=256000,main_seconds=27000.,last_lr=.001)
        # Initialize real Adam first, then use consistent step counters for this fixture.
        self.step()
        for v in self.o.state.values(): v['step'].fill_(self.state['updates'])
        origin=self.state['main_seconds']; pin=self.save(); original=B.O.safe_update
        def fixed(*args,**kw):
            log,micro=original(*args,**kw); log['seconds']=1.; return log,micro
        with patch.object(B.O,'safe_update',side_effect=fixed):
            for _ in range(9): self.step()
            control=self.numerical()
            cfg=self.cfg(main_seconds=9.,probe_main_marks=[origin+3.,origin+6.])
            audit=BranchAudit(self.root)
            def evaluate(*args):
                # Disposable evaluation consumes all CPU RNGs; restore must undo it.
                random.random(); np.random.random(3); torch.rand(3)
                return self.metrics()
            result=G.run(self.name,self.cache,self.order,self.meta,self.root/'run',base.CPU,cfg,evaluate,
                         self.root/'state',pin,cpu=True,branch_audit=audit)
        self.assertEqual(result['status'],'complete'); self.assertEqual(len(result['B3_points']),3)
        points=result['state']['points'][-3:]
        self.assertEqual([p['training']['main_seconds'] for p in points],[origin+3.,origin+6.,origin+9.])
        self.assertEqual([p['training']['main_pairs'] for p in points],[pin['cursor']+v*128 for v in (3,6,9)])
        m,o,s,_=G.restore(B.read_full(self.root/'run',self.ident),base.CPU)
        self.assertEqual(control,digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state())))
        self.assertEqual(len(audit.rows),3)
        self.assertTrue(all(r['passed'] and r['before']==r['restored'] for r in audit.rows))

    def test_b3_direct_restore_audit_rejects_each_omitted_field(self):
        import torch
        from b3_gpu_smoke import BranchAudit
        self.step(); pin=self.save()
        def scaler_change(m,o,s,state):
            scale=s.state_dict(); scale['scale']*=2; s.load_state_dict(scale)
        changes=dict(model=lambda m,o,s,state: next(m.parameters()).data.add_(1),
                     optimizer=lambda m,o,s,state: next(iter(o.state.values()))['exp_avg'].add_(1),
                     scaler=scaler_change,rng_torch_cpu=lambda *args: torch.rand(1),
                     state=lambda m,o,s,state: state.update(cursor=state['cursor']+128),
                     training=lambda m,o,s,state: m.eval(),
                     buffers=lambda m,o,s,state: m.smoke_buffer.add_(1))
        for field,change in changes.items():
            with self.subTest(field=field):
                m,o,_,state=G.restore(B.read_full(self.root/'state',pin['identity']),base.CPU)
                s=torch.amp.GradScaler('cpu',enabled=True,init_scale=1024)
                m.register_buffer('smoke_buffer',torch.zeros(1))
                audit=BranchAudit(self.root)
                audit('before','point',m,o,s,state)
                change(m,o,s,state)
                with self.assertRaisesRegex(AssertionError,'B3 restore changed'):
                    audit('restored','point',m,o,s,state)

    def test_b3_main_stops_for_full_final_reserve(self):
        self.step()
        cfg=self.cfg(main_seconds=999.,deadline_monotonic=1060.)
        with patch.object(G.time,'monotonic',return_value=1000.): result=self.run_b3(cfg)
        self.assertTrue(result['budget_stopped_main']); self.assertTrue(result['final_complete'])
        self.assertEqual(result['state']['updates'],self.state['updates'])
        self.assertEqual(result['status'],'complete')

    def test_b3_incomplete_eval_retains_uncooled_latest(self):
        self.step(); before=self.numerical()
        result=self.run_b3(self.cfg(),lambda *args: {})
        self.assertTrue(result['status'].startswith('partial')); self.assertFalse(result['final_complete'])
        m,o,s,_=G.restore(B.read_full(self.root/'run',self.ident),base.CPU)
        self.assertEqual(before,digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state())))

    def test_b3_recovery_keeps_history_and_halves_lr(self):
        self.step(); original=B.main_step; calls=[]
        def injected(*args):
            if not calls: calls.append(1); raise R.UnsafeState('injected')
            return original(*args)
        with patch.object(B,'main_step',side_effect=injected): result=self.run_b3(self.cfg(main_seconds=999.,max_updates=1))
        self.assertEqual(result['state']['lr_factor'],.5)
        self.assertEqual(result['state']['stability']['escalations'],1)
        self.assertEqual(result['state']['updates'],self.state['updates']+1)
        self.assertTrue(result['final_complete'])

    def test_b3_rejects_wrong_latest_without_loading(self):
        pin=self.save(); pin=copy.deepcopy(pin); pin['updates']+=1
        with self.assertRaises(AssertionError): G.read_anchor(self.root/'state',pin,self.cache,self.meta)

    def test_b3_cpu_adapter_handles_loaded_fused_amp_protocol(self):
        import contextlib, torch
        from b3_cpu_resume_check import cpu_adapt_optimizer
        opt=base.L.optimizer(self.m,.001,False)
        scaler=torch.amp.GradScaler('cpu',enabled=True,init_scale=1024)
        cpu_adapt_optimizer(opt)
        with patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
            log,_=B.O.safe_update(self.m,opt,scaler,self.cache,self.order[:128],base.CPU,32,.001)
        self.assertEqual(log['AMP_skip'],0); self.assertEqual(len(opt.state),len(list(self.m.parameters())))

    def test_b3_mini_final_checks_actual_frozen_tune_ids(self):
        metrics=dict(splits={k:dict(exec=dict(denominator=n),CE=dict(CE=.3)) for k,n in [('heldout4k',64),('old991',16)]},
                     K8_tune2048=dict(complete=True,rows=36,target_rows=36),test=dict(rows=64))
        self.assertTrue(G.metric_complete(metrics,True,True))
        metrics['K8_tune2048'].update(rows=32,target_rows=32)
        self.assertFalse(G.metric_complete(metrics,True,True))

    def test_b3_partial_worker_and_notebook_publish_without_exception(self):
        import ast
        from build_b3 import build
        from build_l3 import ROOT
        self.step(); before=self.numerical()
        result=self.run_b3(self.cfg(),lambda *args: {})
        self.assertTrue(result['status'].startswith('partial'))
        # Execute the real worker's post-run statements with the actual partial result.
        tree=ast.parse((ROOT/'b3_worker.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        pos=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='result' for t in n.targets))
        exec(compile(ast.Module(body=fn.body[pos+1:],type_ignores=[]),'worker publication','exec'),
             dict(result=result,name=self.name))
        nb,_=build(); nb=json.loads(nb.read_text()); code=nb['cells'][-1]['source']
        report=dict(status='partial',exit_codes=[0,0])
        exec(code[code.index('print(json.dumps(report'):],dict(report=report,json=json,WORKING=self.root,write=base.write))
        self.assertEqual(json.loads((self.root/'b3-publication.json').read_text())['status'],'partial')
        m,o,s,_=G.restore(B.read_full(self.root/'run',self.ident),base.CPU)
        self.assertEqual(before,digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state())))

    def test_b3_preflight_reports_each_actual_mount(self):
        from build_b3 import build
        nb,_=build(); code=json.loads(nb.read_text())['cells'][-1]['source']
        start=code.index("source=resolve_cache_mount(inp,CONFIG['cache_source'],kind)")
        end=code.index('\ncache=materialize_dataset',start)
        env=dict(resolve_cache_mount=lambda *a:'cache-mount',inp=None,CONFIG=dict(cache_source='cache'),kind='notebooks')
        exec(code[start:end],env); env['source']='resume-mount'
        start=code.index("write(WORKING/'input-preflight.json'"); end=code.index('\ndel validated',start)
        env.update(write=base.write,WORKING=self.root,starter='starter-mount',CONFIG=dict(cache_manifest_sha256='hash'),ROWS=3)
        exec(code[start:end],env)
        row=json.loads((self.root/'input-preflight.json').read_text())
        self.assertEqual((row['cache_mount'],row['resume_mount']),('cache-mount','resume-mount'))

    def test_b3_ready_gate_requires_passed_smoke_of_identical_modules(self):
        from b3_static_check import smoke_gate
        p=self.root/'smoke.json'; modules={'runtime.py':'tested-sha'}
        self.assertFalse(smoke_gate(modules,[]))
        for passed,sha,expected in ((False,'tested-sha',False),(True,'other-sha',False),(True,'tested-sha',True)):
            base.write(p,dict(passed=passed,modules={'runtime.py':sha}))
            self.assertEqual(smoke_gate(modules,[p]),expected)

    def test_b3_output_download_reads_every_page_and_excludes_PT(self):
        import re
        from types import SimpleNamespace
        from unittest.mock import Mock
        from b3_review_api import small_outputs
        fetch=Mock(side_effect=[(['complete.json'],'page2'),(['gpu-smoke.json'],None)])
        self.assertEqual(small_outputs(SimpleNamespace(kernels_output=fetch),'owner/smoke/1',self.root),2)
        self.assertEqual([c.kwargs['page_token'] for c in fetch.call_args_list],[None,'page2'])
        regex=fetch.call_args.kwargs['file_pattern']
        self.assertIsNone(re.search(regex,'checkpoints/main.pt'))
        self.assertIsNotNone(re.search(regex,'points/cooldown-updates.jsonl'))

    def test_b3_probe_cannot_spend_final_reserve(self):
        self.step(); origin=self.state['main_seconds']
        cfg=self.cfg(main_seconds=999.,max_updates=2,probe_main_marks=[origin+.001],deadline_monotonic=1180.,
                     probe_cooldown_seconds=1.,probe_eval_seconds=1.)
        with patch.object(G.time,'monotonic',return_value=1000.):result=self.run_b3(cfg)
        self.assertEqual(result['missed_points'],[origin+.001]); self.assertTrue(result['final_complete'])
        self.assertEqual(len(result['B3_points']),1);self.assertTrue(result['status'].startswith('partial'))

    def test_b3_embedded_watchdog_uses_actual_first_cell_deadlines(self):
        from build_b3 import build
        from b1_session import Watchdog
        for smoke,soft,hard in ((False,40500,40740),(True,660,690)):
            _,cfg=build(smoke)
            cfg.update(working_root=str(self.root),session_start_monotonic=1000.,deadline_monotonic=1000.+soft,
                       hard_deadline_monotonic=1000.+hard)
            watchdog=Watchdog(cfg)
            self.assertEqual(watchdog.deadline,1000.+soft);self.assertEqual(watchdog.hard_deadline,1000.+hard)
            self.assertLess(hard,cfg['session_ceiling_seconds'])

    def test_b3_smoke_runs_hundreds_and_restores_full_protocol_on_cpu(self):
        import contextlib, torch, b3_gpu_smoke as S
        # Exercise the smoke controller with real tiny QK/Adam/GradScaler; hardware APIs
        # and evaluator alone are replaced. The separate Kaggle run checks CUDA itself.
        self.step(); self.state.update(updates=2001,cursor=2001*128,last_lr=.001)
        for v in self.o.state.values():v['step'].fill_(2001)
        for g in self.o.param_groups:g['lr']=.001
        self.scaler=torch.amp.GradScaler('cpu',enabled=True,init_scale=1024)
        pin=self.save(); actual_run=G.run
        def short_run(*args,**kwargs):
            args=list(args);args[6]=dict(args[6],probe_cooldown_seconds=.01,final_cooldown_seconds=.01)
            return actual_run(*args,**kwargs)
        def metrics(*args):
            row=self.metrics();row['splits']['heldout4k']['exec']['denominator']=64
            row['splits']['old991']['exec']['denominator']=16
            row['K8_tune2048'].update(rows=36,target_rows=36);row['test']['rows']=64
            return row
        sets=dict(heldout4k=[{}]*64,old991=[{}]*16);refs={k:dict(rows=v) for k,v in sets.items()}
        with patch.object(G,'read_anchor',side_effect=lambda root,pin,*a:B.read_full(root,pin['identity'])), \
             patch.object(G,'run',side_effect=short_run),patch.object(S.E,'evaluate',side_effect=metrics), \
             patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'max_memory_reserved',return_value=0), \
             patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
            S.exercise(self.name,self.cache,self.order,self.meta,self.root/'smoke',base.CPU,
                       self.cfg(gpu_smoke=True),None,sets,refs,self.root/'state',pin)
            original_step=S.control_step; attempts=[]
            def varied_control(m,*args):
                out=original_step(m,*args); attempts.append(1)
                if len(attempts)==200:
                    # Simulate a numerical difference in the independent trajectory.
                    with torch.no_grad(): next(m.parameters()).view(-1)[0].add_(1e-4)
                return out
            with patch.object(S,'control_step',side_effect=varied_control):
                S.exercise(self.name,self.cache,self.order,self.meta,self.root/'varied',base.CPU,
                           self.cfg(gpu_smoke=True),None,sets,refs,self.root/'state',pin)
        row=json.loads((self.root/'smoke/gpu-smoke.json').read_text())
        self.assertTrue(row['passed']);self.assertTrue(row['trajectory_bit_exact'])
        self.assertTrue(row['noeval_replay']['bit_exact']);self.assertEqual(len(row['restore_audits']),2)
        self.assertEqual(row['main_updates'],200);self.assertTrue(row['AMP_overflow_skip'])
        varied=json.loads((self.root/'varied/gpu-smoke.json').read_text())
        self.assertTrue(varied['passed']); self.assertFalse(varied['trajectory_bit_exact'])
        self.assertIn('model',varied['trajectory_different'])
        self.assertTrue(all(r['passed'] for r in varied['restore_audits']))


if __name__=='__main__':
    suite=unittest.TestSuite(Tests(n) for n in sorted(Tests.__dict__) if n.startswith('test_b3_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    base.write(base.ROOT.parent/'checks/b3-cpu.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
               failures=[(str(t),s) for t,s in result.failures+result.errors],GPU=False))
    raise SystemExit(not result.wasSuccessful())
