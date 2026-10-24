"""Both historical math paths, order/cursor, disposable full restore and partial safety."""
import copy,hashlib,json,random,sys,time,unittest
from pathlib import Path
from unittest.mock import patch
import b3_tests as base
import b5_runtime as G
import b5_state as S
import b5_resume as Mount
import b2_runtime as B
import l1_runtime as L
import l2_runtime as R
import exp26_train as T
import exp26_core as C
from exp26_common import json_hash
from l2_gpu_smoke import digest
from b3_gpu_smoke import BranchAudit
from kaggle_io import write,sha
import numpy as np
import torch
CPU=torch.device('cpu')


class Tests(base.Tests):
    def prepare(self,kind):
        self.ident=B.identity_for('W-lr600',base.base.SPEC,self.cache,self.meta,.2,True)
        self.m=B.new_model(base.base.SPEC,CPU,.2);self.o=L.optimizer(self.m,.0006,True)
        self.state=B.new_state(32,.0006);self.state.update(updates=2000,cursor=256000,main_seconds=91801.,last_lr=.0006)
        B.main_step(self.m,self.o,self.scaler,self.state,self.cache,self.order,CPU)
        spec=base.base.SPEC
        if kind=='C':
            spec={k:v for k,v in base.base.SPEC.items() if k!='qk_norm'}
            self.m=C.SimpleTransformer(**spec,dropout=.2);self.o=L.optimizer(self.m,.0015,True)
            self.ident=L.identity_for('C',spec,self.cache,self.meta['file']['sha256'],True)
            self.ident.update(model='C-long2x',cpu_test=True)
            self.ident['recipe'].update(dropout=.2,sampler=self.meta)
            self.ident['recipe_sha256']=json_hash(self.ident['recipe'])
            self.state.update(last_lr=.0015,cursor_origin=-1280,cursor=self.state['updates']*128-1280,
                 l2_main_seconds=62280.,l2_points=[dict(tag='l2-old',complete=True)],l2_origin={'main_seconds':17570.})
            R.safe_update(self.m,self.o,self.scaler,self.cache,L.take(self.order,self.state['cursor'],128),CPU,32,.0015)
        for v in self.o.state.values():v['step'].fill_(self.state['updates'])
        root=self.root/('source-'+kind)
        pin=S.save_full(root,self.m,self.o,self.scaler,self.state,spec,self.ident)
        return root,pin
    def fixture_read(self,root,identity,environment=True):
        rec=json.loads((Path(root)/'latest.json').read_text());path=Path(root)/rec['file']
        self.assertEqual(sha(path),rec['sha256']);self.assertEqual(path.stat().st_size,rec['bytes'])
        p=torch.load(path,map_location='cpu',weights_only=True)
        self.assertEqual(p['identity'],identity);return p
    def run_fixture(self,kind,**cfg):
        src,pin=self.prepare(kind)
        return src,pin,dict(self.cfg(),**cfg)
    def numerical(self):return digest((self.m.state_dict(),self.o.state_dict(),self.scaler.state_dict(),T.rng_state()))

    def test_b5_both_paths_probe_final_restore_next_stream_and_uncooled(self):
        for kind in ('W','C'):
            with self.subTest(kind=kind):
                src,pin,cfg=self.run_fixture(kind,main_seconds=9.,probe_main_marks=[],checkpoint_seconds=0.)
                origin=self.state['main_seconds'];initial=copy.deepcopy(self.state)
                cfg['probe_main_marks']=[origin+3.]
                original=B.O.safe_update if kind=='W' else R.safe_update
                def fixed(*args,**kw):
                    log,micro=original(*args,**kw);log['seconds']=1.;return log,micro
                target=B.O if kind=='W' else R
                ids=[]
                with patch.object(target,'safe_update',side_effect=fixed),patch.object(S,'read_full',side_effect=self.fixture_read):
                    for _ in range(9):
                        ids.extend(L.take(self.order,self.state['cursor'],128).tolist())
                        S.main_step(self.m,self.o,self.scaler,self.state,self.cache,self.order,CPU,self.ident)
                    expected=self.numerical();end=copy.deepcopy(self.state)
                    def evaluate(*a):random.random();np.random.random(3);torch.rand(8);return self.metrics()
                    out=self.root/('run-'+kind);audit=BranchAudit(out)
                    result=G.run(kind,self.cache,self.order,self.meta,out,CPU,cfg,evaluate,src,pin,cpu=True,branch_audit=audit)
                    self.assertEqual(result['status'],'complete');self.assertEqual(len(result['B5_points']),2)
                    self.assertTrue(all(x['passed'] for x in audit.rows));self.assertEqual(len(audit.rows),2)
                    self.m,self.o,self.scaler,actual=S.restore(self.fixture_read(out,self.ident),CPU)
                    self.assertEqual(expected,self.numerical())
                    for key in initial:
                        self.assertEqual(actual[key],end[key],key)
                    self.assertEqual(actual['cursor'],pin['cursor']+9*128)
                    self.assertEqual(result['stream']['sha256'],hashlib.sha256(np.asarray(ids,dtype='<u4').tobytes()).hexdigest())
                    self.assertEqual(result['B6_uncooled_latest']['cursor'],actual['cursor'])
                    self.assertEqual(actual['b5_points'][-1]['training']['q'],0.)
                    self.assertEqual(len(list((out/'checkpoints').glob('*.pt'))),2)
                    fp=torch.load(out/'weights'/f'{result["B5_points"][-1]}-fp16.pt',weights_only=True)
                    self.assertTrue(all(v.dtype==torch.float16 for v in fp['model'].values() if v.is_floating_point()))

    def test_b5_each_math_path_and_no_new_warmup(self):
        for kind in ('W','C'):
            src,pin=self.prepare(kind);p=self.fixture_read(src,self.ident)
            m,o,s,st=S.restore(copy.deepcopy(p),CPU)
            log,_=S.main_step(m,o,s,st,self.cache,self.order,CPU,self.ident)
            expected=digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state()))
            m,o,s,st=S.restore(copy.deepcopy(p),CPU)
            if kind=='W':ref,_=B.main_step(m,o,s,st,self.cache,self.order,CPU)
            else:
                ref,_=R.safe_update(m,o,s,self.cache,L.take(self.order,st['cursor'],128),CPU,st['micro_batch'],st['last_lr'])
            self.assertEqual(expected,digest((m.state_dict(),o.state_dict(),s.state_dict(),T.rng_state())))
            self.assertEqual(log['lr'],.0006 if kind=='W' else .0015)
            self.assertEqual('z_loss' in log,kind=='W');self.assertNotIn('truncation',log)

    def test_b5_failed_evaluation_is_partial_and_restores_both_stables(self):
        for kind in ('W','C'):
            src,pin,cfg=self.run_fixture(kind,main_seconds=0.)
            before=self.numerical();out=self.root/('failed-'+kind)
            def fail(*a):torch.rand(8);raise ValueError('evaluation failed')
            with patch.object(S,'read_full',side_effect=self.fixture_read):
                result=G.run(kind,self.cache,self.order,self.meta,out,CPU,cfg,fail,src,pin,cpu=True)
            self.assertTrue(result['status'].startswith('partial'));self.assertFalse(result['final_complete'])
            self.m,self.o,self.scaler,st=S.restore(self.fixture_read(out,self.ident),CPU)
            self.assertEqual(before,self.numerical());self.assertEqual(st['cursor'],pin['cursor'])
            self.assertTrue(list((out/'weights').glob('*-fp16.pt')))

    def test_b5_stability_stops_on_last_stable_without_LR_change(self):
        for kind in ('W','C'):
            src,pin,cfg=self.run_fixture(kind,main_seconds=999.)
            original=S.main_step
            def escalate(*a,**kw):log,_=original(*a,**kw);return log,True
            with patch.object(S,'read_full',side_effect=self.fixture_read),patch.object(G,'main_step',side_effect=escalate):
                result=G.run(kind,self.cache,self.order,self.meta,self.root/('stop-'+kind),CPU,cfg,self.metrics,src,pin,cpu=True)
            self.assertTrue(result['status'].startswith('partial'));self.assertEqual(result['state']['cursor'],pin['cursor'])
            self.assertEqual(result['state']['last_lr'],.0006 if kind=='W' else .0015)

    def test_b5_deadline_reserves_final_and_never_claims_complete(self):
        src,pin,cfg=self.run_fixture('W',main_seconds=999.,deadline_monotonic=time.monotonic()+.005)
        with patch.object(S,'read_full',side_effect=self.fixture_read):
            r=G.run('W',self.cache,self.order,self.meta,self.root/'deadline',CPU,cfg,self.metrics,src,pin,cpu=True)
        self.assertTrue(r['status'].startswith('partial'));self.assertFalse(r['final_complete'])
        self.assertEqual(r['state']['cursor'],pin['cursor'])

    def test_b5_entire_smoke_controller_both_formats_on_CPU(self):
        import b5_gpu_smoke as Smoke
        for kind in ('W','C'):
            src,pin=self.prepare(kind)
            self.scaler=torch.amp.GradScaler('cpu',enabled=True,init_scale=1024)
            pin=S.save_full(src,self.m,self.o,self.scaler,self.state,pin['model_spec'],self.ident)
            cfg=self.cfg(gpu_smoke=True,smoke_updates=200)
            sets=dict(heldout4k=list(range(4096)),old991=list(range(991)))
            refs={k:dict(rows=v) for k,v in sets.items()}
            def metrics(*a):
                return dict(splits={k:dict(exec=dict(denominator=n),CE=dict(CE=.3)) for k,n in [('heldout4k',64),('old991',16)]},
                            K8_tune2048=dict(complete=True,rows=36,target_rows=36),test=dict(rows=64))
            out=self.root/('smoke-'+kind)
            with patch.object(G,'read_anchor',side_effect=lambda root,pin,*_:self.fixture_read(root,pin['identity'])), \
                 patch.object(S,'read_full',side_effect=self.fixture_read),patch.object(torch.cuda,'reset_peak_memory_stats'), \
                 patch.object(torch.cuda,'max_memory_reserved',return_value=0),patch.object(torch.cuda,'max_memory_allocated',return_value=0), \
                 patch.object(Smoke.E,'evaluate',side_effect=metrics):
                Smoke.exercise(kind,self.cache,self.order,self.meta,out,CPU,cfg,None,sets,refs,src,pin)
            r=json.loads((out/'gpu-smoke.json').read_text())
            self.assertTrue(r['passed']);self.assertEqual(r['main_updates'],200);self.assertEqual(len(r['restore_audits']),2)
            self.assertTrue(r['AMP_overflow_skip']);self.assertFalse(r['truncation'])
            self.assertEqual(len(r['sampled_throughput']['samples']),9)

    def test_b5_rejects_transfer_manifest_and_blob_corruption(self):
        source=self.root/'transfer';source.mkdir();pin={'file':'checkpoints/main.pt'}
        (source/'b5-transfer.json').write_text('{}')
        with self.assertRaises(AssertionError):Mount.transfer_mount(source,self.root/'mounted','bad',pin)
        with self.assertRaises((AssertionError,KeyError)):Mount.transfer_mount(source,self.root/'mounted',sha(source/'b5-transfer.json'),pin)

    def test_b5_W_transfer_and_C_live_mount_preserve_exact_pins(self):
        source=self.root/'W-transfer';source.mkdir();pin=dict(file='checkpoints/main.pt',sha256='pinned')
        payloads={'model-GPU0/latest.json':json.dumps(pin).encode(),'model-GPU0/checkpoints/main.pt':b'W full state',
                  'weighted-order/order.npy':b'W inherited order','weighted-order/order.json':b'{}'}
        files=[]
        for original,raw in payloads.items():
            flat=original.replace('/','__');p=source/flat;p.write_bytes(raw)
            files.append(dict(file=flat,original=original,bytes=len(raw),sha256=sha(p)))
        write(source/'b5-transfer.json',dict(format='nlp-hw02-b5-transfer-v1',destination='friend1/nlp-hw02-b5-inputs',files=files))
        mounted=Mount.transfer_mount(source,self.root/'mounted-W',sha(source/'b5-transfer.json'),pin)
        self.assertEqual(json.loads((mounted/'model-GPU0/latest.json').read_text()),pin)
        self.assertTrue((mounted/'model-GPU0/checkpoints/main.pt').is_symlink())
        (source/files[1]['file']).write_bytes(b'corruption')
        with self.assertRaises(AssertionError):Mount.transfer_mount(source,self.root/'corrupt-W',sha(source/'b5-transfer.json'),pin)
        live=self.root/'C-live';live.mkdir();blob=live/'flat.pt';blob.write_bytes(b'C full state')
        rec=dict(file='checkpoints/main.pt',sha256=sha(blob),bytes=blob.stat().st_size,updates=209144,cursor=20734592,
                 main_seconds=79850.42887455746,l2_main_seconds=62280.4929961593,identity=dict(model='C-long2x'))
        write(live/'latest-flat.json',rec);latest=live/'latest-flat.json'
        pin=dict(rec,file='flat.pt',original_file=rec['file'])
        write(live/'recover-manifest.json',dict(format='nlp-hw02-l2-live-v1',files=[
            dict(original_path='model-C-long2x/latest.json',file=latest.name,sha256=sha(latest),bytes=latest.stat().st_size,gzip=False),
            dict(original_path='model-C-long2x/'+rec['file'],file=blob.name,sha256=sha(blob),bytes=blob.stat().st_size,gzip=False)]))
        mounted=Mount.c_mount(live,self.root/'mounted-C',pin)
        self.assertEqual(json.loads((mounted/'latest.json').read_text()),pin);self.assertTrue((mounted/'flat.pt').is_symlink())
        bad=dict(pin,cursor=pin['cursor']+128)
        with self.assertRaises(AssertionError):Mount.c_mount(live,self.root/'wrong-C',bad)

    def test_b5_legacy_C_namespace_isolated_and_all_frozen_SHA_match(self):
        from b5_sources import c_namespace
        import subprocess
        root=base.base.ROOT;folder=c_namespace(self.root/'C-modules')
        pin=json.loads((root/'b5-anchors.json').read_text())['anchors']['1']
        for name,h in pin['identity']['recipe']['implementation_sha256'].items():self.assertEqual(sha(folder/name),h)
        self.assertNotEqual(sha(folder/'exp26_core.py'),sha(root/'src/exp26_core.py'))
        code="import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import exp26_core as C,l2_runtime as R,b5_state as S;from kaggle_io import sha;pin=json.loads(Path(sys.argv[2]).read_text())['anchors']['1'];assert all(sha(Path(S.__file__).with_name(n))==h for n,h in pin['identity']['recipe']['implementation_sha256'].items());assert Path(C.__file__).parent==Path(sys.argv[1]);assert Path(R.__file__).parent==Path(sys.argv[1]);assert Path(S.__file__).parent==Path(sys.argv[1])"
        code+="\nimport torch,time,exp27b_fast as F\ntorch.set_num_threads(1)\nm=C.SimpleTransformer(d_model=8,heads=2,encoder_layers=1,decoder_layers=1,swiglu_dim=16,vocab=24,dropout=0.).eval()\nsrc=torch.tensor([[1,4,2,0],[1,5,6,2]]);tokens=torch.tensor([[1,7,8],[1,9,10]])\nd=F.Decoder(F.Context(m,src,8),1)\nfor t in range(tokens.shape[1]):torch.testing.assert_close(d.step(tokens[:,t]),m(src,tokens[:,:t+1])[:,-1],rtol=3e-5,atol=3e-5)\nseq,lp,_=F.decode(F.Context(m,src,8),k=8,temperature=.7,deadline=time.monotonic()+10,check_every=1)\nassert seq.shape[:2]==(2,8) and bool(torch.isfinite(lp).all())"
        subprocess.run([sys.executable,'-c',code,str(folder),str(root/'b5-anchors.json')],check=True)

    def test_b5_final_metric_contract(self):
        self.assertTrue(G.metric_complete(self.metrics(),True,False))
        for mutate in (lambda m:m['K8_tune2048'].update(rows=2047),lambda m:m['test'].update(rows=0),
                       lambda m:m['splits']['heldout4k']['CE'].update(CE=float('nan'))):
            m=self.metrics();mutate(m);self.assertFalse(G.metric_complete(m,True,False))

    def test_b5_static_preserves_launch_and_requires_both_source_namespaces(self):
        from b5_static_check import reviewed_handoff
        base=dict(modules={'W':'w'},C_overrides={'core':'c'},ready=True,readiness='CPU ready',GPU_runs=0,pushes=0)
        previous=dict(base,GPU_runs=2,pushes=2,production_url='private/B5',status='RUNNING',
                      started_at_utc='2026-10-18T12:00:00Z',ETA_utc='2026-10-18T23:20:00Z')
        good=dict(passed=True,modules=base['modules'],C_overrides=base['C_overrides'],receipt='checks/gpu.json')
        for receipts in ([],[dict(good,passed=False)],[dict(good,modules={'W':'changed'})],
                         [dict(good,C_overrides={'core':'changed'})]):
            r=reviewed_handoff(dict(base),previous,receipts)
            self.assertFalse(r['smoke_passed']);self.assertTrue(r['GPU_smoke_pending'])
            self.assertEqual(r['GPU_runs'],2);self.assertEqual(r['status'],'RUNNING')
        r=reviewed_handoff(dict(base),previous,[good])
        self.assertTrue(r['smoke_passed']);self.assertFalse(r['GPU_smoke_pending'])
        for key in ('production_url','status','started_at_utc','ETA_utc'):self.assertEqual(r[key],previous[key])
        self.assertEqual(r['GPU_smoke_receipt'],'checks/gpu.json')
        for changed in (dict(base,modules={'W':'new'}),dict(base,C_overrides={'core':'new'})):
            r=reviewed_handoff(changed,previous,[good])
            self.assertFalse(r['smoke_passed']);self.assertNotIn('production_url',r)

    def test_b5_push_rejects_untracked_version_and_duplicate_production(self):
        from b5_review_api import require_known_version,require_absent_target,require_cancelled_retry
        require_known_version(0,[],False)
        require_known_version(1,[1],True)
        require_known_version(3,[3],True)
        for remote,known,smoke in ((2,[1],True),(1,[],False),(0,[1],True),(1,[1],False),(3,[1,3],True)):
            with self.subTest(remote=remote,known=known,smoke=smoke),self.assertRaises(AssertionError):
                require_known_version(remote,known,smoke)
        require_absent_target('owner/new',{'owner/smoke'},[],True)
        for owned,known,fully in (({'owner/new'},[],True),(set(),[1],True),(set(),[],False)):
            with self.assertRaises(AssertionError):require_absent_target('owner/new',owned,known,fully)
        require_known_version(2,[1],False,True)
        for remote,known,smoke in ((3,[1,3],False),(2,[1],True),(1,[],False)):
            with self.assertRaises(AssertionError):require_known_version(remote,known,smoke,True)
        q=dict(used_seconds=497.44,reserved_seconds=0)
        require_cancelled_retry('CANCEL_ACKNOWLEDGED',False,q,q,True)
        for status,output,now,exact in (('RUNNING',False,q,True),('CANCEL_ACKNOWLEDGED',True,q,True),
             ('CANCEL_ACKNOWLEDGED',False,dict(q,used_seconds=498),True),
             ('CANCEL_ACKNOWLEDGED',False,dict(q,reserved_seconds=1),True),('CANCEL_ACKNOWLEDGED',False,q,False)):
            with self.assertRaises(AssertionError):require_cancelled_retry(status,output,q,now,exact)


if __name__=='__main__':
    suite=unittest.TestSuite(Tests(n) for n in sorted(Tests.__dict__) if n.startswith('test_b5_'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(base.base.ROOT.parent/'checks/b5-cpu.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
       failures=[(str(t),s) for t,s in result.failures+result.errors],GPU=False,
       modules={p.name:sha(p) for p in (base.base.ROOT/'src').glob('b5_*.py')}))
    raise SystemExit(not result.wasSuccessful())
