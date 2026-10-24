"""CPU contracts for P1 objectives, restores, stable reviewer and offline selectors."""
import copy
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import exp26_core as C
import l1_runtime as L
import l2_runtime as R
import p1_math as M
import p1_runtime as P
from exp26_order import collate
from kaggle_io import write
from l2_gpu_smoke import digest

CPU=torch.device('cpu')
SPEC=dict(d_model=8,heads=2,encoder_layers=1,decoder_layers=1,swiglu_dim=16,vocab=24)


class Cache:
    manifest_sha='tiny'
    def __init__(self):
        self.rows=[(np.asarray([1,3+i%7,5,2]),np.asarray([8,9,10+i%4])) for i in range(128)]
        self.source_lengths=np.asarray([len(s) for s,t in self.rows]);self.target_lengths=np.asarray([len(t) for s,t in self.rows])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[int(i)]


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='nlp-p1-tests-');self.root=Path(self.temp.name)
        self.cache=Cache();self.order=np.arange(128,dtype=np.uint32)
        self.model=L.new_model(SPEC,CPU);self.opt=L.optimizer(self.model,.0015,True)
        self.scaler=torch.amp.GradScaler('cpu',enabled=False)
        self.identity=L.identity_for('C',SPEC,self.cache,'tiny-order',True)
        self.state=P.new_reviewer_state()
        self.anchor=R.payload(self.model,self.opt,self.scaler,self.state,SPEC,self.identity)
    def tearDown(self):self.temp.cleanup()

    def test_sqrt_schedule_bounds_monotonicity_and_zero_endpoint(self):
        values=[M.sqrt_lr(i,100) for i in range(101)]
        self.assertEqual(values[0],.0015);self.assertEqual(values[-1],0.)
        self.assertAlmostEqual(values[25],.00075)
        self.assertTrue(all(a>=b>=0 for a,b in zip(values,values[1:])))
        for step,total in [(-1,10),(11,10),(0,0)]:
            with self.assertRaises(ValueError):M.sqrt_lr(step,total)

    def test_mask_stratification_determinism_top_losses_and_ties(self):
        lengths=np.repeat([20,60,90,150],32);scores=np.arange(128,dtype=float)
        keep,stats=M.truncation_mask(scores,lengths)
        self.assertEqual(int((~keep).sum()),13);self.assertEqual(stats['excluded'],[4,3,3,3])
        self.assertEqual(stats['counts'],[32]*4)
        for b in range(4):
            pos=np.arange(b*32,(b+1)*32);excluded=pos[~keep[pos]]
            self.assertTrue(np.all(scores[excluded]>=scores[pos[keep[pos]]].max()))
        a,_=M.truncation_mask(np.ones(128),lengths);b,_=M.truncation_mask(np.ones(128),lengths)
        self.assertTrue(np.array_equal(a,b));self.assertFalse(a[0]);self.assertTrue(a[31])
        keep,stats=M.truncation_mask([1,100,2],[20,150,20],q=.5)
        self.assertTrue(keep[1]);self.assertEqual(sum(keep),2)

    def test_truncated_loss_normalization_matches_manual_retained_batch(self):
        ids=np.arange(128);scores=M.per_example_ce(self.model,self.cache,ids,CPU,64)
        keep,_=M.truncation_mask(scores,self.cache.target_lengths)
        other=L.new_model(SPEC,CPU);other.load_state_dict(self.model.state_dict())
        opt=L.optimizer(other,.0015,True);scaler=torch.amp.GradScaler('cpu',enabled=False)
        expected,_=R.safe_update(other,opt,scaler,self.cache,ids[keep],CPU,64,.0015)
        actual,_=M.safe_truncated_update(self.model,self.opt,self.scaler,self.cache,ids,CPU,64,.0015,strategy='two-pass')
        self.assertEqual(actual['tokens'],int(sum(self.cache.target_lengths[keep]+1)))
        self.assertAlmostEqual(actual['CE'],expected['CE'],places=7)
        self.assertEqual(digest(other.state_dict()),digest(self.model.state_dict()))
        self.assertEqual(digest(opt.state_dict()),digest(self.opt.state_dict()))
        self.assertEqual(actual['truncation']['retained'],115)

    def test_nonfinite_selection_is_rejected_before_adam(self):
        before=digest((self.model.state_dict(),self.opt.state_dict(),self.scaler.state_dict()))
        with patch.object(M,'per_example_ce',return_value=np.full(128,np.nan)):
            log,_=M.safe_truncated_update(self.model,self.opt,self.scaler,self.cache,self.order,CPU,64,.0015,strategy='two-pass')
        self.assertEqual(log['nonfinite_skip'],1)
        self.assertEqual(before,digest((self.model.state_dict(),self.opt.state_dict(),self.scaler.state_dict())))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_graph_mask_global_token_weighted_gradient_and_OOM_fallback(self):
        # Variable target lengths/PAD across microbatches; masks must be global.
        for i in range(128):
            self.cache.rows[i]=(self.cache.rows[i][0],np.asarray([8,9,10]*(1+i%4)))
        self.cache.target_lengths=np.asarray([len(t) for s,t in self.cache.rows])
        anchor=copy.deepcopy(self.anchor)
        expected,_=M.safe_truncated_update(self.model,self.opt,self.scaler,self.cache,self.order,CPU,32,.0015,strategy='two-pass')
        reference=copy.deepcopy(self.model.state_dict())
        gradients={k:p.grad.clone() for k,p in self.model.named_parameters()}
        L.restore_full(anchor,self.model,self.opt,self.scaler,CPU)
        actual,_=M.safe_truncated_update(self.model,self.opt,self.scaler,self.cache,self.order,CPU,32,.0015)
        self.assertEqual(actual['truncation']['excluded_positions'],expected['truncation']['excluded_positions'])
        self.assertEqual(actual['tokens'],expected['tokens'])
        for k,p in self.model.named_parameters():
            torch.testing.assert_close(p.grad,gradients[k],rtol=1e-5,atol=1e-7)
            self.assertTrue(bool(torch.isfinite(p).all()))
        self.assertAlmostEqual(actual['CE'],expected['CE'],places=6)
        L.restore_full(anchor,self.model,self.opt,self.scaler,CPU)
        def oom(*args):
            next(self.model.parameters()).grad=torch.ones_like(next(self.model.parameters()))
            torch.rand(1)
            raise torch.cuda.OutOfMemoryError('injected retained graph OOM')
        with patch.object(M,'graph_truncated_update',oom):
            fallback,_=M.safe_truncated_update(self.model,self.opt,self.scaler,self.cache,self.order,CPU,32,.0015)
        self.assertTrue(fallback['truncation']['graph_OOM_fallback'])
        self.assertEqual(digest(self.model.state_dict()),digest(reference))

    def test_graph_partial_backward_NaN_restores_scaler_and_clears_gradients(self):
        before=digest((self.model.state_dict(),self.opt.state_dict(),self.scaler.state_dict()))
        scale=self.scaler.scale; calls=[0]
        def broken(loss):
            calls[0]+=1
            if calls[0]==2:raise FloatingPointError('injected after first microbatch backward')
            return scale(loss)
        with patch.object(self.scaler,'scale',broken):
            log,_=M.safe_truncated_update(self.model,self.opt,self.scaler,self.cache,self.order,CPU,64,.0015)
        self.assertEqual(log['nonfinite_skip'],1)
        self.assertEqual(before,digest((self.model.state_dict(),self.opt.state_dict(),self.scaler.state_dict())))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_reverse_source_target_PAD_BOS_EOS(self):
        cache=M.ReverseCache(self.cache);src,tgt,labels,counts=collate(cache,[0,1],CPU)
        self.assertEqual(src[0].tolist(),[1,8,9,10,2])
        self.assertEqual(tgt[0].tolist(),[1,3,5]);self.assertEqual(labels[0].tolist(),[3,5,2])
        self.assertEqual(counts['target_tokens'],6)
        self.cache.rows[1]=(np.asarray([1,4,2]),np.asarray([8]))
        src,tgt,labels,_=collate(M.ReverseCache(self.cache),[0,1],CPU)
        self.assertEqual(src[1].tolist(),[1,8,2,0,0])
        self.assertEqual(tgt[1].tolist(),[1,4,0]);self.assertEqual(labels[1].tolist(),[4,2,0])

    def pilot(self,evaluate):
        pin=dict(file='anchor.pt',sha256='tiny')
        cfg=dict(pilot_updates=3,pilot_a_cap_seconds=30,pilot_b_cap_seconds=30,
                 pilot_eval_seconds=10,pilot_k8_rows=4,deadline_monotonic=time.monotonic()+100)
        restores=[];original=P.restore_anchor
        def restore(p,*args):
            value=original(p,*args);restores.append(digest((value[0].state_dict(),value[1].state_dict(),value[2].state_dict(),value[3])))
            return value
        with patch('l3_resume.verify_anchor',side_effect=lambda *a,**k:(self.root/'anchor.pt',copy.deepcopy(self.anchor))),\
             patch.object(P,'restore_anchor',restore):
            result=P.run_pilot(self.root/'anchor.pt',pin,self.cache,self.order,self.root/'model-pilot',CPU,cfg,evaluate,cpu=True)
        self.assertEqual(len(restores),4);self.assertEqual(len(set(restores)),1)
        rec=json.loads((self.root/'model-pilot/latest.json').read_text())
        saved=R.read_full(self.root/'model-pilot'/rec['file'],self.identity,False)
        self.assertEqual(saved['state'],self.anchor['state'])
        return result

    def test_both_pilot_branches_same_stream_and_full_restore(self):
        result=self.pilot(lambda *a:dict(K8_tune2048=dict(complete=True,rows=4,target_rows=4)))
        self.assertEqual(result['status'],'complete');self.assertTrue(result['same_stream'])
        self.assertEqual(result['main_added_updates'],0)
        self.assertEqual(result['branches'][0]['training']['final_lr'],0.)
        self.assertEqual(result['branches'][0]['training']['start_fingerprints'],
                         result['branches'][1]['training']['start_fingerprints'])

    def test_full_state_fingerprint_detects_tensor_and_rng_corruption(self):
        before=P.state_fingerprints(self.model,self.opt,self.scaler,self.state)
        next(self.model.parameters()).data.view(-1)[0]+=1
        self.assertNotEqual(before['model'],P.state_fingerprints(self.model,self.opt,self.scaler,self.state)['model'])
        torch.rand(1)
        self.assertNotEqual(before['rng'],P.state_fingerprints(self.model,self.opt,self.scaler,self.state)['rng'])

    def test_budget_expired_pilot_saves_partial_weights_without_complete(self):
        cfg=dict(pilot_updates=3,pilot_a_cap_seconds=0,pilot_b_cap_seconds=0,
                 pilot_eval_seconds=10,pilot_k8_rows=4,deadline_monotonic=time.monotonic()+100)
        with patch('l3_resume.verify_anchor',side_effect=lambda *a,**k:(self.root/'anchor.pt',copy.deepcopy(self.anchor))):
            result=P.run_pilot(self.root/'anchor.pt',dict(sha256='tiny'),self.cache,self.order,
                              self.root/'model-pilot',CPU,cfg,lambda *a:{},cpu=True)
        self.assertEqual(result['status'],'partial')
        for name,tag in [('a','long'),('b','long-truncation')]:
            root=self.root/f'model-pilot-{name}'
            self.assertTrue((root/'weights'/f'{tag}-fp16.pt').exists())
            self.assertFalse((root/'points'/tag/'complete.json').exists())

    def test_failed_A_evaluation_does_not_stop_B_or_overwrite_stable(self):
        calls=[]
        def evaluate(*args):
            calls.append(1)
            if len(calls)==1:raise RuntimeError('injected')
            return dict(K8_tune2048=dict(complete=True,rows=4,target_rows=4))
        result=self.pilot(evaluate)
        self.assertEqual(result['status'],'partial');self.assertTrue(result['branches'][1]['complete'])

    def test_missing_or_false_K8_cannot_mark_complete(self):
        for m in [None,{},dict(K8_tune2048={}),dict(K8_tune2048=dict(complete=True,rows=4,target_rows=5))]:
            self.assertFalse(P.metrics_complete(m,4))

    def test_reviewer_real_WSD_keeps_uncooled_resumable_state(self):
        cfg=dict(deadline_monotonic=time.monotonic()+3,reviewer_eval_seconds=.4,
                 reviewer_io_reserve_seconds=.1,checkpoint_seconds=.4,reviewer_validation_interval=100,reviewer_warmup_updates=20,
                 reviewer_expected_rows={'tiny':2})
        result=P.run_reviewer(M.ReverseCache(self.cache),self.order,'tiny-order',self.root/'model-reviewer',CPU,
                             cfg,lambda *args:dict(tiny=dict(complete=True,CE=2.,rows=2,tokens=6)),cpu=True,spec=SPEC)
        self.assertEqual(result['status'],'complete')
        root=self.root/'model-reviewer';rec=json.loads((root/'latest.json').read_text())
        p=R.read_full(root/rec['file'],rec['identity'],check_environment=False)
        self.assertLess(p['state']['updates'],result['state']['updates'])
        self.assertEqual(p['state']['last_lr'],.0015)
        self.assertEqual(p['state']['cursor'],p['state']['updates']*128)
        self.assertTrue((root/'weights/final-fp16.pt').exists())

    def test_reviewer_failed_or_partial_CE_cannot_mark_complete_and_keeps_weights(self):
        for metrics in ({},{'tiny':dict(complete=False,rows=2,tokens=6,CE=2.)},
                        {'tiny':dict(complete=True,rows=1,tokens=6,CE=2.)},
                        {'tiny':dict(complete=True,rows=2,tokens=6,CE=float('nan'))}):
            self.assertFalse(P.reviewer_metrics_complete(metrics,{'tiny':2}))
        cfg=dict(deadline_monotonic=time.monotonic()+1,reviewer_eval_seconds=.2,
                 reviewer_io_reserve_seconds=.1,reviewer_validation_interval=100,
                 reviewer_expected_rows={'tiny':2})
        result=P.run_reviewer(M.ReverseCache(self.cache),self.order,'tiny-order',self.root/'model-reviewer',CPU,
                             cfg,lambda *a:{},cpu=True,spec=SPEC)
        self.assertEqual(result['status'],'partial')
        self.assertTrue((self.root/'model-reviewer/weights/final-fp16.pt').exists())
        self.assertTrue((self.root/'model-reviewer/latest.json').exists())
        self.assertFalse((self.root/'model-reviewer/points/final/complete.json').exists())

    def test_scorer_exp_beta_zero_and_reference_independence(self):
        spec=importlib.util.spec_from_file_location('reviewer_score',ROOT.parent/'research/score-reviewer.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        def candidate(value,score):return dict(execution=dict(status='numeric',value=value),mean_logprob=-1.,
                            reviewer=dict(mean_logprob=score,sum_logprob=10*score))
        row=dict(greedy=candidate(2,-1),samples={'0.7':[candidate(1,-2),candidate(1,-3),candidate(2,-1)]})
        self.assertEqual(module.select(row,'cluster-exp',0)['execution']['value'],1)
        self.assertEqual(module.select(row,'cluster-max')['execution']['value'],2)
        before=module.select(row,'vote-reviewer',1);row['reference']={'status':'numeric','value':999}
        self.assertEqual(before,module.select(row,'vote-reviewer',1))
        score=module.score_batch(self.model.eval(),[[8,9],[8]],[[3,5],[4]])
        self.assertEqual([s['question_tokens'] for s in score],[3,2])
        self.assertAlmostEqual(score[0]['sum_logprob']/3,score[0]['mean_logprob'])

    def test_live_snapshot_retains_both_pilot_FP16_points_and_reviewer(self):
        from l2_live import stage_snapshot
        for name,tag in [('pilot-a','long'),('pilot-b','long-truncation'),('reviewer','final')]:
            L.save_point(self.root/f'model-{name}',tag,self.model,SPEC,self.identity,dict(branch=name))
        stage=self.root/'live-stage'
        stage_snapshot(self.root,stage,'friend1/nlp-hw02-p1-live','cpu-test')
        manifest=json.loads((stage/'recover-manifest.json').read_text())
        names={r['original_path'] for r in manifest['files']}
        self.assertIn('model-pilot-a/weights/long-fp16.pt',names)
        self.assertIn('model-pilot-b/weights/long-truncation-fp16.pt',names)
        self.assertIn('model-reviewer/weights/final-fp16.pt',names)

    def test_p1_supervisor_healthy_worker_completes_after_sibling_crash(self):
        from p1_session import launch,Watchdog
        child=self.root/'child.py'
        child.write_text('''import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int);p.add_argument('--config');a=p.parse_args()
r=Path(json.loads(Path(a.config).read_text())['working_root'])
name='pilot' if a.gpu==0 else 'reviewer';(r/('model-'+name)).mkdir(exist_ok=True)
if a.gpu==1:
    time.sleep(.7)
    assert not (r/'stop-request.json').exists()
(r/('model-'+name)/'summary.json').write_text(json.dumps({'status':'complete'}))
raise SystemExit(1 if a.gpu==0 else 0)
''')
        now=time.monotonic();cfg=dict(working_root=str(self.root/'session'),session_start_monotonic=now,
              deadline_monotonic=now+5,hard_deadline_monotonic=now+10,hard_shutdown_seconds=10,
              budget_seconds=5,session_ceiling_seconds=10,terminate_grace_seconds=.1)
        report=launch(cfg,child,Watchdog(cfg).start(),smoke=True,heartbeat=.1)
        self.assertEqual(report['exit_codes'],[1,0]);self.assertEqual(report['worker_status']['reviewer'],'complete')
        self.assertFalse(report['unreaped_worker_pids']);self.assertFalse((self.root/'session/stop-request.json').exists())


if __name__=='__main__':
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(ROOT.parent/'checks/kaggle-p1-regressions.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
               failures=len(result.failures),errors=len(result.errors),GPU=False))
    sys.exit(not result.wasSuccessful())
