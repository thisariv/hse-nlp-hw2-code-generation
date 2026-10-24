#!/usr/bin/env python3
"""CPU regressions: independent failure, exact finite math, skips, rollback, sampler, live."""
import copy, json, os, shutil, sys, tempfile, time, unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import numpy as np, torch
import l1_runtime as L, l2_runtime as R, l2_sampler as S, l2_live as Live, exp26_train as T
from l2_resume import initial_state, branch_identity, verify_anchor
from kaggle_io import sha,write
from l2_session import launch,Watchdog
SPEC=dict(d_model=8,heads=2,encoder_layers=1,decoder_layers=1,swiglu_dim=16,vocab=32,max_positions=2048)
CPU=torch.device('cpu')
class Cache:
    manifest_sha='tiny'
    def __init__(self):
        self.rows=[([1,3+i%8,2],[4+i%8]*(1+i%5)) for i in range(137)]
        self.source_lengths=np.array([len(s) for s,t in self.rows]);self.target_lengths=np.array([len(t) for s,t in self.rows])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[int(i)]

def exact(a,b):
    if torch.is_tensor(a): return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    return a==b

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=Path(tempfile.mkdtemp(prefix='l2-tests-'));self.cache=Cache();self.order=np.arange(len(self.cache),dtype=np.uint32)
        self.model=L.new_model(SPEC,CPU);self.opt=L.optimizer(self.model,.0015,True);self.scaler=torch.amp.GradScaler('cpu',enabled=False)
        self.ident=L.identity_for('C',SPEC,self.cache,'tiny-order',True,warmup=1)
        state={'updates':0,'cursor':0,'main_seconds':0.,'branch_seconds':0.,'branch_attempted_updates':0,
               'AMP_skips':0,'last_lr':.0015,'micro_batch':64,'points':[]}
        for _ in range(2):
            log=L.update(self.model,self.opt,self.scaler,self.cache,L.take(self.order,state['cursor'],128),CPU,64,.0015)
            state['updates']+=1;state['cursor']+=128;state['main_seconds']+=log['seconds']
        self.anchor=L.state_payload(self.model,self.opt,self.scaler,state,SPEC,self.ident)
    def tearDown(self):shutil.rmtree(self.tmp)
    def run_model(self,name='C-main',**kw):
        return R.run(name,copy.deepcopy(self.anchor),self.ident,self.cache,self.order,self.tmp/name,CPU,time.monotonic()+30,
                     kw.pop('evaluate',lambda *a:{}),smoke=True,**dict({'main_targets':(1000.,),'cooldown_seconds':(.01,),
                                                                   'eval_seconds':(1.,),'max_updates':2},**kw))
    def saved(self,name='C-main'):
        root=self.tmp/name;r=json.loads((root/'latest.json').read_text());return R.read_full(root/r['file'],self.ident)
    def test_finite_update_bit_exact_with_L1(self):
        a=copy.deepcopy(self.anchor);m=L.new_model(SPEC,CPU);o=L.optimizer(m,.0015,True);s=torch.amp.GradScaler('cpu',enabled=False)
        L.restore_full(copy.deepcopy(a),m,o,s,CPU);R.safe_update(m,o,s,self.cache,L.take(self.order,256,128),CPU,64,.0015)
        expected=(copy.deepcopy(m.state_dict()),copy.deepcopy(o.state_dict()),T.cpu(T.rng_state()))
        L.restore_full(copy.deepcopy(a),m,o,s,CPU);L.update(m,o,s,self.cache,L.take(self.order,256,128),CPU,64,.0015)
        self.assertTrue(exact(expected,(m.state_dict(),o.state_dict(),T.cpu(T.rng_state()))))
    def test_nonfinite_loss_skips_Adam_and_clears_partial_grads(self):
        before=copy.deepcopy(self.opt.state_dict());weights=copy.deepcopy(self.model.state_dict());calls=[0];original=L.F.cross_entropy
        def loss(*a,**kw):
            calls[0]+=1;return original(*a,**kw)*(float('nan') if calls[0]==2 else 1.)
        with patch.object(L.F,'cross_entropy',loss):log,_=R.safe_update(self.model,self.opt,self.scaler,self.cache,self.order[:128],CPU,64,.0015)
        self.assertEqual(log['nonfinite_skip'],1);self.assertTrue(exact(before,self.opt.state_dict()));self.assertTrue(exact(weights,self.model.state_dict()))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
    def test_nonfinite_gradient_skips_Adam(self):
        before=copy.deepcopy(self.opt.state_dict())
        with patch.object(torch.nn.utils,'clip_grad_norm_',return_value=torch.tensor(float('nan'))):
            log,_=R.safe_update(self.model,self.opt,self.scaler,self.cache,self.order[:128],CPU,64,.0015)
        self.assertEqual(log['nonfinite_skip'],1);self.assertTrue(exact(before,self.opt.state_dict()))
    def test_normal_AMP_skip_is_not_nonfinite_escalation(self):
        with patch.object(L,'resilient_update',return_value=({'AMP_skip':1,'seconds':1.},64)):
            log,_=R.safe_update(self.model,self.opt,self.scaler,self.cache,self.order[:128],CPU,64,.0015)
        st={'total':0,'consecutive':0,'window':[]};self.assertFalse(R.observe(st,log['nonfinite_skip']));self.assertEqual(st['total'],0)
    def test_window_threshold(self):
        st={'total':0,'consecutive':0,'window':[]};trigger=False
        for i in range(91):trigger=R.observe(st,i%10==0)
        self.assertTrue(trigger);self.assertEqual(st['total'],10)
    def test_rollback_halving_then_graceful_K_stop_preserves_good_state(self):
        with patch.object(L.F,'cross_entropy',return_value=torch.tensor(float('nan'))):
            report=self.run_model(max_updates=None)
        saved=self.saved();st=report['state']
        self.assertEqual(report['status'],'stability_stop');self.assertEqual(st['stability']['total'],9)
        self.assertEqual(st['stability']['escalations'],3);self.assertEqual(st['last_lr'],.0015/4)
        self.assertEqual(st['updates'],2);self.assertEqual(st['cursor'],256)
        self.assertTrue(exact(saved['model'],self.anchor['model']));self.assertEqual([int(s['step']) for s in saved['optimizer']['state'].values()],[2]*len(saved['optimizer']['state']))
        self.assertEqual([x['new_lr'] for x in st['stability']['history'][:-1]],[.0015/2,.0015/4])
    def test_fork_retains_state_and_changes_only_sampler_identity_cursor(self):
        meta={'file':{'sha256':'weighted'},'seed':S.SEED};identity=branch_identity(self.ident,meta)
        state=initial_state(self.anchor,'C-long2x');self.assertEqual(state['cursor'],0);self.assertEqual(state['updates'],2)
        recipe=copy.deepcopy(identity['recipe']);recipe.pop('sampler');recipe['data_policy']=self.ident['recipe']['data_policy']
        self.assertEqual(recipe,self.ident['recipe']);self.assertNotEqual(identity['recipe_sha256'],self.ident['recipe_sha256'])
        self.run_model('C-long2x');self.assertEqual(self.saved('C-long2x')['state']['cursor'],256)
    def test_cooldown_eval_failure_restores_exact_anchor_then_train(self):
        self.run_model('control',max_updates=4)
        def fail(*a):raise ValueError('injected eval')
        self.run_model('C-main',main_targets=(.001,1000.),cooldown_seconds=(.01,.01),eval_seconds=(1.,1.),max_updates=4,evaluate=fail)
        self.assertTrue(exact(self.saved()['model'],self.saved('control')['model']))
        self.assertTrue(exact(self.saved()['optimizer'],self.saved('control')['optimizer']))
        self.assertTrue(exact(self.saved()['rng'],self.saved('control')['rng']))
    def test_sampler_deterministic_multiplicity_and_nearest_fraction(self):
        x,a=S.build_order(self.cache,[10,90,100,10],self.tmp/'a');y,b=S.build_order(self.cache,[10,90,100,10],self.tmp/'b')
        self.assertTrue(np.array_equal(x,y));self.assertEqual(a['file']['sha256'],b['file']['sha256'])
        self.assertTrue(np.array_equal(np.bincount(x),1+(self.cache.target_lengths>=a['threshold'])))
        self.assertEqual(S.fixed_split(),S.fixed_split());self.assertEqual(len(S.fixed_split()['tune2048']),2048)
    def test_worker_failure_does_not_stop_healthy_worker(self):
        worker=self.tmp/'worker.py';worker.write_text("import argparse,json,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--gpu');p.add_argument('--config');a=p.parse_args();cfg=json.loads(Path(a.config).read_text())\nif a.gpu=='0':raise SystemExit(1)\ntime.sleep(.5)\nassert not (Path(cfg['working_root'])/'stop-request.json').exists()\n(Path(cfg['working_root'])/'healthy-finished.json').write_text('{}')\n")
        start=time.monotonic();cfg={'working_root':str(self.tmp/'supervisor'),'session_start_monotonic':start,'deadline_monotonic':start+5,
                                  'hard_deadline_monotonic':start+10,'hard_shutdown_seconds':10,'budget_seconds':5,'terminate_grace_seconds':.1}
        w=Watchdog(cfg).start();report=launch(cfg,worker,w,smoke=True,heartbeat=.1)
        self.assertEqual(report['exit_codes'],[1,0]);self.assertTrue((Path(cfg['working_root'])/'healthy-finished.json').exists())
        self.assertFalse((Path(cfg['working_root'])/'stop-request.json').exists())
    def test_bad_SHA_is_rejected_before_loading(self):
        f=self.tmp/'c.pt';f.write_bytes(b'bad')
        with patch.object(torch,'load',side_effect=AssertionError('must not load')):
            with self.assertRaisesRegex(AssertionError,'SHA'):verify_anchor(self.tmp,{'file':'c.pt','bytes':3,'sha256':'wrong'})
    def test_live_roundtrip_and_retention(self):
        self.run_model();working=self.tmp;stage=self.tmp.parent/(self.tmp.name+'-stage')
        try:
            # stage must stay outside working (same as real LiveSync).
            Live.stage_snapshot(working,stage,'thisarivson/nlp-hw02-l2-live','test')
            manifest=json.loads((stage/'recover-manifest.json').read_text())
            self.assertEqual(manifest['format'],'nlp-hw02-l2-live-v1')
            self.assertLess(manifest['payload_bytes'],Live.STAGE_LIMIT)
            restored=self.tmp.parent/(self.tmp.name+'-restored')
            try:
                Live.restore_snapshot(stage,restored)
                rec=json.loads((restored/'C-main/latest.json').read_text()) if (restored/'C-main/latest.json').exists() else None
                # Runtime root test name differs from production model-*; validate actual production selector below.
                shutil.copytree(self.tmp/'C-main',self.tmp/'model-C-main')
                Live.stage_snapshot(working,stage,'thisarivson/nlp-hw02-l2-live','test2')
                Live.restore_snapshot(stage,restored)
                rec=json.loads((restored/'model-C-main/latest.json').read_text())
                self.assertEqual(sha(restored/'model-C-main'/rec['file']),rec['sha256'])
            finally:shutil.rmtree(restored,ignore_errors=True)
        finally:shutil.rmtree(stage,ignore_errors=True)
    def test_secret_gate_smoke_optional_production_required(self):
        class GPU:
            cuda=type('CUDA',(),{'device_count':staticmethod(lambda:2),'get_device_name':staticmethod(lambda i:'T4')})
        self.assertTrue(Live.gate(GPU,require_secrets=False)['passed'])
        self.assertFalse(Live.gate(GPU,secret_getter=lambda n:None)['passed'])
    def test_post_Adam_nonfinite_is_unsafe_and_good_checkpoint_not_overwritten(self):
        original=self.opt.step
        def corrupt(*a,**kw):
            original(*a,**kw)
            with torch.no_grad():next(self.model.parameters()).fill_(float('nan'))
        with patch.object(self.opt,'step',corrupt):
            with self.assertRaises(R.UnsafeState):R.safe_update(self.model,self.opt,self.scaler,self.cache,self.order[:128],CPU,64,.0015)
    def test_stable_rotation_and_corruption_refusal(self):
        self.run_model();root=self.tmp/'C-main';good=json.loads((root/'latest.json').read_text())
        saved=self.saved();state=saved['state']
        L.restore_full(saved,self.model,self.opt,self.scaler,CPU)
        for _ in range(4):R.save_full(root,self.model,self.opt,self.scaler,state,SPEC,self.ident)
        self.assertEqual(len(list((root/'checkpoints').glob('main-*.pt'))),2)
        good=json.loads((root/'latest.json').read_text())
        with torch.no_grad():next(self.model.parameters()).fill_(float('inf'))
        with self.assertRaises(FloatingPointError):R.save_full(root,self.model,self.opt,self.scaler,state,SPEC,self.ident)
        self.assertEqual(json.loads((root/'latest.json').read_text()),good)
    def test_checkpoint_L3_continuation_preserves_sampler_cursor_and_history(self):
        self.run_model('C-long2x');saved=self.saved('C-long2x');old=copy.deepcopy(saved['state'])
        R.run('C-long2x',saved,self.ident,self.cache,self.order,self.tmp/'C-long2x',CPU,time.monotonic()+30,lambda *a:{},
              smoke=True,main_targets=(1000.,),cooldown_seconds=(.01,),eval_seconds=(1.,),max_updates=1)
        new=self.saved('C-long2x')['state']
        self.assertEqual(new['cursor'],old['cursor']+128);self.assertEqual(new['updates'],old['updates']+1)
        self.assertEqual(new['l2_origin'],old['l2_origin'])
        expected=copy.deepcopy(old['stability']);expected['window'].append(0)
        self.assertEqual(new['stability'],expected)
    def test_live_rejects_secret_and_retains_order(self):
        self.run_model();shutil.copytree(self.tmp/'C-main',self.tmp/'model-C-main')
        (self.tmp/'weighted-order').mkdir();(self.tmp/'weighted-order/order.npy').write_bytes(b'order bytes')
        stage=self.tmp.parent/(self.tmp.name+'-live')
        try:
            Live.stage_snapshot(self.tmp,stage,'thisarivson/nlp-hw02-l2-live','order')
            self.assertEqual((stage/'weighted-order__order.npy').read_bytes(),b'order bytes')
            (self.tmp/'injected.txt').write_text('synthetic-regression-credential')
            with patch.dict(os.environ,{'KAGGLE_KEY':'synthetic-regression-credential'}):
                with self.assertRaises(RuntimeError):Live.stage_snapshot(self.tmp,stage,'thisarivson/nlp-hw02-l2-live','bad')
        finally:shutil.rmtree(stage,ignore_errors=True)
    def test_fixed_split_matches_registered_candidate_selection(self):
        from exp26_common import json_hash
        registered=json.loads((ROOT.parent/'results/candidate-selection-cpu.json').read_text())['split']
        split=S.fixed_split()
        self.assertEqual(split['tune2048'],registered['tune_ids']);self.assertEqual(split['eval2048'],registered['eval_ids'])
        self.assertEqual(json_hash(split['tune2048']),registered['tune_ids_sha256'])
    def test_K8_journals_allow_gold_free_mixed_CPU_vote(self):
        from l2_compare import mixed
        rows=[]
        for i in range(2):
            row={'id':i,'question_sha256':str(i),'reference':{'status':'numeric','value':2},
                 'greedy':{'code':'print(2)','execution':{'status':'numeric','value':2},'mean_logprob':0},
                 'samples':{'0.7':[{'code':'print(2)','execution':{'status':'numeric','value':2},'mean_logprob':-.2,'length':3} for j in range(8)]},
                 'correct':True}
            rows.append(row)
        for name in ('a','b'):write(self.tmp/name/'batch-00000.json',rows)
        result=mixed(self.tmp/'a',self.tmp/'b')
        self.assertEqual(result['mixed_matches'],2);self.assertEqual(result['rows'],2);self.assertFalse(result['complete_tune2048'])
    def test_preflight_counter_mismatches_fail_early(self):
        f=self.tmp/'c.pt';f.write_bytes(b'x');p=copy.deepcopy(self.anchor)
        p['model_spec']=L.SPECS['C'];p['scaler_enabled']=True;p['rng']['torch_cuda']=[torch.tensor([1])]
        expected={'file':'c.pt','bytes':1,'sha256':sha(f),'identity':self.ident,
                  'updates':3,'cursor':256,'main_seconds':p['state']['main_seconds']}
        with patch.object(L,'read_full',return_value=p):
            with self.assertRaisesRegex(AssertionError,'updates'):verify_anchor(self.tmp,expected,False)
    def test_actual_CPU_eval_journals_K8_and_offline_vote(self):
        import l2_eval as E
        inputs=self.tmp/'inputs';inputs.mkdir()
        tokenizer=ROOT.parent/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json'
        shutil.copyfile(tokenizer,inputs/'tokenizer.json')
        rows=[{'question':f'What is {i}+1?','code':f'print({i+1})'} for i in range(8)]
        sets={'heldout4k':rows,'old991':rows[:2]}
        refs={s:{'rows':[{'status':'numeric','value':i+1} for i in range(len(v))]} for s,v in sets.items()}
        write(inputs/'test.json',rows[:2]);spec=dict(SPEC,vocab=8195);model=L.new_model(spec,CPU)
        point=L.save_point(self.tmp/'eval','tiny',model,spec,self.ident,{})
        metrics=E.evaluate(point,self.tmp/'eval-out',inputs,sets,refs,CPU,time.monotonic()+45,True,True,15.,test_rows=2)
        saved=json.loads((self.tmp/'eval-out/heldout4k-per-example.json').read_text())
        self.assertEqual(len(saved),8);self.assertTrue(all(isinstance(x['correct'],bool) for x in saved))
        journal=[r for p in (self.tmp/'eval-out/k8-tune2048').glob('batch-*.json') for r in json.loads(p.read_text())]
        self.assertGreater(len(journal),0)
        self.assertTrue(all(len(r['samples']['0.7'])==8 for r in journal))
        self.assertTrue(all('execution' in c and 'mean_logprob' in c and 'length' in c for r in journal for c in r['samples']['0.7']))
        self.assertEqual(metrics['test']['rows'],2)
        from l2_compare import mixed
        result=mixed(self.tmp/'eval-out/k8-tune2048',self.tmp/'eval-out/k8-tune2048')
        self.assertEqual(result['rows'],len(journal))
    def test_real_enabled_scaler_overflow_stays_a_normal_AMP_skip(self):
        import contextlib
        scaler=torch.amp.GradScaler('cpu',init_scale=1024,enabled=True)
        before=copy.deepcopy(self.opt.state_dict());weights=copy.deepcopy(self.model.state_dict())
        hook=next(self.model.parameters()).register_hook(lambda grad:torch.full_like(grad,float('inf')))
        try:
            with patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
                log,_=R.safe_update(self.model,self.opt,scaler,self.cache,self.order[:128],CPU,64,.0015)
        finally:hook.remove()
        self.assertEqual(log['AMP_skip'],1);self.assertEqual(log['nonfinite_skip'],0)
        self.assertEqual(scaler.get_scale(),512);self.assertTrue(exact(before,self.opt.state_dict()));self.assertTrue(exact(weights,self.model.state_dict()))
    def test_rejected_loss_resets_enabled_scaler_before_next_good_step(self):
        import contextlib
        scaler=torch.amp.GradScaler('cpu',init_scale=1024,enabled=True);snapshot=copy.deepcopy(scaler.state_dict())
        with patch.object(torch,'autocast',side_effect=lambda **kw:contextlib.nullcontext()):
            with patch.object(L.F,'cross_entropy',return_value=torch.tensor(float('nan'))):
                log,_=R.safe_update(self.model,self.opt,scaler,self.cache,self.order[:128],CPU,64,.0015)
            self.assertEqual(log['nonfinite_skip'],1);self.assertEqual(scaler.state_dict(),snapshot)
            good,_=R.safe_update(self.model,self.opt,scaler,self.cache,self.order[:128],CPU,64,.0015)
        self.assertEqual(good['nonfinite_skip'],0);self.assertEqual(good['AMP_skip'],0)
        self.assertEqual(scaler.state_dict()['_growth_tracker'],1)
    def test_cooldown_nonfinite_threshold_rolls_back_and_halves_worker_LR(self):
        original=L.F.cross_entropy;calls=[0]
        def fail(*a,**kw):
            calls[0]+=1
            return torch.tensor(float('nan')) if 3<=calls[0]<=5 else original(*a,**kw)
        with patch.object(L.F,'cross_entropy',fail):
            report=self.run_model(main_targets=(.001,1000.),cooldown_seconds=(.1,.01),eval_seconds=(1.,1.))
        state=report['state'];self.assertEqual(state['last_lr'],.0015/2)
        self.assertEqual(state['stability']['escalations'],1);self.assertEqual(state['stability']['total'],3)
        self.assertFalse(state['l2_points'][0]['complete']);self.assertEqual(state['updates'],4)

    def test_corrupt_Adam_moment_is_rejected_immediately(self):
        original=self.opt.step
        def corrupt(*a,**kw):
            original(*a,**kw)
            next(iter(self.opt.state.values()))['exp_avg'].fill_(float('inf'))
        with patch.object(self.opt,'step',corrupt):
            with self.assertRaises(R.UnsafeState):R.safe_update(self.model,self.opt,self.scaler,self.cache,self.order[:128],CPU,64,.0015)

    def test_persistent_AMP_skips_are_bounded_but_isolated_skips_are_normal(self):
        st={'total':0,'consecutive':0,'window':[]}
        for _ in range(31):self.assertFalse(R.observe(st,0,amp_skip=1))
        self.assertTrue(R.observe(st,0,amp_skip=1));self.assertEqual(st['total'],0)
        self.assertFalse(R.observe(st,0,amp_skip=0));self.assertEqual(st['amp_consecutive'],0)
        fake={'AMP_skip':1,'nonfinite_skip':0,'seconds':.001,'CE':1.,'OOM_retries':[]}
        with patch.object(R,'safe_update',return_value=(fake,64)):
            result=self.run_model(max_updates=None)
        self.assertEqual(result['status'],'stability_stop');self.assertEqual(result['state']['stability']['escalations'],3)
        self.assertEqual(result['state']['stability']['amp_total'],96);self.assertEqual(self.saved()['state']['cursor'],256)

    def test_unsafe_attempt_time_survives_rollback(self):
        def unsafe(*a,**k):time.sleep(.02);raise L.UnsafeAdamOOM('injected')
        with patch.object(R,'safe_update',unsafe):result=self.run_model(max_updates=None)
        self.assertEqual(result['status'],'stability_stop');self.assertGreaterEqual(result['state']['l2_main_seconds'],.06)
        self.assertEqual(result['state']['main_seconds'],result['state']['l2_origin']['main_seconds']+result['state']['l2_main_seconds'])

    def test_resume_does_not_overwrite_evaluated_points(self):
        self.run_model(main_targets=(.001,1000.),cooldown_seconds=(.001,.001),eval_seconds=(1.,1.))
        old=self.saved();marker=self.tmp/'C-main/points/l2-00000/complete.json';before=marker.read_bytes()
        self.assertEqual(len(old['state']['l2_points']),1)
        def forbidden(*a):raise AssertionError('Old point evaluated twice')
        R.run('C-main',old,self.ident,self.cache,self.order,self.tmp/'C-main',CPU,time.monotonic()+30,forbidden,
              smoke=True,main_targets=(.001,1000.),cooldown_seconds=(.001,.001),eval_seconds=(1.,1.),max_updates=1)
        self.assertEqual(marker.read_bytes(),before);self.assertEqual(len(self.saved()['state']['l2_points']),1)

    def test_order_SHA_bounds_integer_dtype_tail_and_large_cursor(self):
        folder=self.tmp/'order';order,meta=S.build_order(self.cache,[10,90,100],folder)
        got,_=S.load_order(folder,self.cache,meta['file']['sha256']);self.assertTrue(np.array_equal(got,order))
        cursor=2**40*len(order)+len(order)-3
        self.assertTrue(np.array_equal(L.take(got,cursor,128),order[(np.arange(128)+len(order)-3)%len(order)]))
        with self.assertRaisesRegex(AssertionError,'SHA'):S.load_order(folder,self.cache,'wrong')
        for bad in (np.array([len(self.cache)],dtype=np.uint32),np.array([-1],dtype=np.int64)):
            np.save(folder/'order.npy',bad);meta['file'].update(sha256=sha(folder/'order.npy'),bytes=(folder/'order.npy').stat().st_size)
            meta['slots']=1;write(folder/'order.json',meta)
            with self.assertRaises(AssertionError):S.load_order(folder,self.cache,meta['file']['sha256'])

    def test_unreadable_summary_and_failed_final_sync_still_package(self):
        worker=self.tmp/'worker.py';worker.write_text("import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--gpu');p.add_argument('--config');a=p.parse_args();cfg=json.loads(Path(a.config).read_text())\nr=Path(cfg['working_root'])/('model-C-main' if a.gpu=='0' else 'model-C-long2x');r.mkdir();(r/'summary.json').write_text('{broken')\n")
        start=time.monotonic();cfg={'working_root':str(self.tmp/'packaging'),'session_start_monotonic':start,'deadline_monotonic':start+5,
                                  'hard_deadline_monotonic':start+10,'hard_shutdown_seconds':10,'budget_seconds':5,'terminate_grace_seconds':.1}
        class FailedLive:
            uninsured=False;last_success=start
            def final_sync(self,*a,**k):raise TimeoutError('injected')
        report=launch(cfg,worker,Watchdog(cfg).start(),live=FailedLive(),smoke=True,heartbeat=.1)
        self.assertFalse(report['final_live_uploaded']);self.assertEqual(report['worker_status']['C-main'],'unreadable')
        self.assertTrue((Path(cfg['working_root'])/'l2-output-manifest.json').exists())

    def test_killed_worker_does_not_stop_sibling_or_packaging(self):
        worker=self.tmp/'worker.py';worker.write_text("import argparse,json,time,os,signal\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--gpu');p.add_argument('--config');a=p.parse_args();cfg=json.loads(Path(a.config).read_text())\nif a.gpu=='0':\n r=Path(cfg['working_root'])/'model-C-main/checkpoints';r.mkdir(parents=True);(r/'main-test.pt.tmp').write_bytes(b'incomplete');os.kill(os.getpid(),signal.SIGKILL)\ntime.sleep(.3)\nassert not (Path(cfg['working_root'])/'stop-request.json').exists()\n(Path(cfg['working_root'])/'healthy.json').write_text('{}')\n")
        start=time.monotonic();cfg={'working_root':str(self.tmp/'kill'),'session_start_monotonic':start,'deadline_monotonic':start+5,
                                  'hard_deadline_monotonic':start+10,'hard_shutdown_seconds':10,'budget_seconds':5,'terminate_grace_seconds':.1}
        report=launch(cfg,worker,Watchdog(cfg).start(),smoke=True,heartbeat=.1)
        self.assertEqual(report['exit_codes'],[-9,0]);self.assertTrue((Path(cfg['working_root'])/'healthy.json').exists())
        self.assertTrue((Path(cfg['working_root'])/'l2-output-manifest.json').exists())
        self.assertFalse((Path(cfg['working_root'])/'model-C-main/checkpoints/main-test.pt.tmp').exists())

    def test_platform_and_API_credentials_are_removed_from_workers_and_helpers(self):
        names=('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN','KAGGLE_USER_SECRETS_TOKEN','KAGGLE_IAP_TOKEN')
        worker=self.tmp/'worker.py';worker.write_text("import argparse,os\np=argparse.ArgumentParser();p.add_argument('--gpu');p.add_argument('--config');p.parse_args()\nassert all(n not in os.environ for n in "+repr(names)+")\n")
        start=time.monotonic();cfg={'working_root':str(self.tmp/'credentials'),'session_start_monotonic':start,'deadline_monotonic':start+5,
                                  'hard_deadline_monotonic':start+10,'hard_shutdown_seconds':10,'budget_seconds':5,'terminate_grace_seconds':.1}
        with patch.dict(os.environ,{n:'synthetic-'+n for n in names}):
            report=launch(cfg,worker,Watchdog(cfg).start(),smoke=True,heartbeat=.1)
            live=Live.L2LiveSync(self.tmp,self.tmp/'helper-scratch','thisarivson/nlp-hw02-l2-live')
            self.assertNotIn('KAGGLE_API_TOKEN',live.env)
            self.assertNotIn('KAGGLE_USER_SECRETS_TOKEN',live.env);self.assertNotIn('KAGGLE_IAP_TOKEN',live.env)
        self.assertEqual(report['exit_codes'],[0,0])

    def test_supervisor_error_still_records_and_packages_outputs(self):
        worker=self.tmp/'worker.py';worker.write_text("import time;time.sleep(30)\n")
        start=time.monotonic();cfg={'working_root':str(self.tmp/'supervisor-error'),'session_start_monotonic':start,'deadline_monotonic':start+5,
                                  'hard_deadline_monotonic':start+10,'hard_shutdown_seconds':10,'budget_seconds':5,'terminate_grace_seconds':.1}
        with patch('l2_session.disk_guard',side_effect=RuntimeError('injected disk guard')):
            report=launch(cfg,worker,Watchdog(cfg).start(),smoke=True,heartbeat=.1)
        self.assertEqual(report['reason'],'supervisor error: RuntimeError')
        self.assertTrue((Path(cfg['working_root'])/'supervisor-failure.json').exists())
        self.assertTrue((Path(cfg['working_root'])/'l2-output-manifest.json').exists())
        self.assertFalse(report['unreaped_worker_pids'])

    def test_disk_guard_tolerates_rotation_between_is_file_and_stat(self):
        from l2_session import disk_guard
        root=self.tmp/'disk';root.mkdir();gone=root/'gone.pt';gone.write_bytes(b'old');(root/'stable.pt').write_bytes(b'stable')
        original=Path.stat
        def rotate(p,*a,**k):
            value=original(p,*a,**k)
            if p==gone:gone.unlink()
            return value
        with patch.object(Path,'stat',rotate):self.assertEqual(disk_guard(root,1024),6)
        with self.assertRaises(RuntimeError):disk_guard(root,1)

if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests);result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(ROOT.parent/'checks/kaggle-l2-regressions.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,
         'failures':len(result.failures),'errors':len(result.errors),'GPU':False})
    sys.exit(not result.wasSuccessful())
