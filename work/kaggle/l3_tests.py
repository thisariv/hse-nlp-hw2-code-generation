#!/usr/bin/env python3
"""L3 input/phase/legacy-smoke adapters; full L2 regressions remain separate."""
import copy,gzip,hashlib,json,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import l3_resume as S,l2_runtime as R,l2_gpu_smoke as G,l1_runtime as L
from kaggle_io import write,sha

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='l3-tests-');self.root=Path(self.tmp.name)
        self.pins=json.loads((ROOT/'l3-anchors.json').read_text())
    def tearDown(self):self.tmp.cleanup()
    def test_exact_cumulative_targets_without_mutating_anchor_history(self):
        cfg={'phase_recipe':{'main_targets':[16740.,33480.],'cooldown_seconds':[900.,3000.],'eval_seconds':[300.,1800.]}}
        for pin in self.pins['anchors'].values():
            anchor={'state':{'l2_main_seconds':pin['l2_main_seconds'],'cursor':pin['cursor'],'l2_points':['old']}}
            before=copy.deepcopy(anchor);r=S.phase_recipe(cfg,anchor)
            self.assertEqual(anchor,before);self.assertEqual(cfg['phase_recipe']['main_targets'],[16740.,33480.])
            self.assertEqual(r['main_targets'],[pin['l2_main_seconds']+16740,pin['l2_main_seconds']+33480])
            self.assertEqual([int(x) for x in r['main_targets']],[45540,62280])
    def test_expanded_and_original_gzip_verify_producer_SHA(self):
        raw=b'{"rows":[1,2,3]}';packed=gzip.compress(raw,compresslevel=1,mtime=0)
        entry={'file':'payload.json.gz','gzip':True,'bytes':len(packed),'sha256':hashlib.sha256(packed).hexdigest()}
        (self.root/entry['file']).write_bytes(packed);self.assertEqual(S.live_bytes(self.root,entry),raw)
        (self.root/entry['file']).unlink();(self.root/'payload.json').write_bytes(raw)
        self.assertEqual(S.live_bytes(self.root,entry),raw)
        (self.root/'payload.json').write_bytes(raw+b' ')
        with self.assertRaises(AssertionError):S.live_bytes(self.root,entry)
    def receipt(self,name):
        pin=self.pins['anchors'][name];rec=copy.deepcopy(pin);rec['file']=rec.pop('original_file')
        latest=self.root/f'model-{name}__latest.json';write(latest,rec)
        write(self.root/'recover-manifest.json',{'format':'nlp-hw02-l2-live-v1','files':[
          {'original_path':f'model-{name}/latest.json','file':latest.name,'sha256':sha(latest),'bytes':latest.stat().st_size,'gzip':False},
          {'original_path':f'model-{name}/{rec["file"]}','file':pin['file'],'sha256':pin['sha256'],'bytes':pin['bytes'],'gzip':False}]})
        return pin
    def test_latest_manifest_pin_require_own_identity_and_counters(self):
        for name in self.pins['anchors']:
            pin=self.receipt(name);self.assertEqual(S.mounted_expected(self.root,name,pin),pin)
            for key in ('sha256','updates','cursor','main_seconds','l2_main_seconds','identity'):
                changed=copy.deepcopy(pin);changed[key]='bad'
                with self.assertRaises(AssertionError):S.mounted_expected(self.root,name,changed)
    def test_anchor_mismatch_fails_before_loading_large_PT(self):
        pin={'file':'blob.pt','bytes':4,'sha256':'bad'};(self.root/'blob.pt').write_bytes(b'tiny')
        with patch.object(R,'read_full') as read:
            with self.assertRaises(AssertionError):S.verify_anchor(self.root,pin)
            read.assert_not_called()
    def test_full_state_counters_LR_history_checked(self):
        f=self.root/'blob.pt';f.write_bytes(b'tiny');pin=copy.deepcopy(self.pins['anchors']['C-main'])
        pin.update(file=f.name,bytes=4,sha256=sha(f))
        payload={'model_spec':L.SPECS['C'],'state':{k:pin[k] for k in ('updates','cursor','main_seconds','l2_main_seconds')},
                 'scaler_enabled':True,'rng':{'torch_cuda':[1]}}
        payload['state'].update(last_lr=.0015,stability={'escalations':0},l2_points=[{'complete':True}]*3)
        with patch.object(R,'read_full',return_value=payload):S.verify_anchor(self.root,pin,False)
        for key in ('updates','cursor','main_seconds','l2_main_seconds','last_lr'):
            bad=copy.deepcopy(payload);bad['state'][key]=0
            with patch.object(R,'read_full',return_value=bad):
                with self.assertRaises(AssertionError):S.verify_anchor(self.root,pin,False)
    def test_legacy_GPU_smoke_preserves_own_cursor_and_old_points(self):
        history=[{'tag':'old'}];anchor={'state':{'cursor':129,'l2_main_seconds':28800.3,'l2_points':history}}
        order=np.arange(200,dtype=np.uint32);seen=[]
        def fake_run(*args,**kwargs):
            seen.append(kwargs['main_targets']);return {'state':{'l2_points':history+[{'tag':'a'},{'tag':'b'}]}}
        def exercise(gpu,name,payload,*args):
            self.assertEqual(G.initial_state(payload,name),payload['state'])
            self.assertTrue(np.array_equal(L.take(order,0,4),order[129:133]))
            self.assertTrue(np.array_equal(L.take(order,130,4),order[130:134]))
            first=R.run(main_targets=[150.,300.]);self.assertEqual(len(first['state']['l2_points']),2)
            tail=R.run(main_targets=[1e6]);self.assertEqual(len(tail['state']['l2_points']),3)
        with patch.object(R,'run',fake_run),patch.object(G,'exercise',exercise):
            S.exercise_smoke(1,'C-long2x',anchor,{},None,order,None,None,None,None,{})
        self.assertEqual(seen,[[28950.3,29100.3],[1028800.3]])
        self.assertEqual(anchor['state']['l2_points'],history)
    def test_real_failed_eval_reports_partial_and_preserves_resumable_state(self):
        from l2_tests import Tests as Fixture, CPU, SPEC
        from l2_resume import initial_state
        fixture=Fixture();fixture.setUp()
        try:
            anchor=R.payload(fixture.model,fixture.opt,fixture.scaler,
                             initial_state(fixture.anchor,'C-main'),SPEC,fixture.ident)
            cfg={'phase_recipe':{'main_targets':[.001,1.03],'cooldown_seconds':[.001,.001],'eval_seconds':[1.,1.]}}
            original=R.run
            def fail(*args):raise ValueError('injected evaluation failure')
            with patch.object(R,'run',side_effect=lambda *a,**kw:original(*a,smoke=True,**kw)):
                result=S.run_continuation('C-main',anchor,fixture.ident,fixture.cache,fixture.order,
                      self.root,CPU,time.monotonic()+30,fail,cfg)
            self.assertTrue(result['status'].startswith('partial;'))
            self.assertEqual(json.loads((self.root/'summary.json').read_text())['status'],result['status'])
            receipt=json.loads((self.root/'latest.json').read_text())
            saved=R.read_full(self.root/receipt['file'],fixture.ident)
            self.assertEqual(saved['state'],result['state'])
            self.assertTrue(all(not p['complete'] for p in result['state']['l2_points']))
        finally:fixture.tearDown()
    def test_incomplete_K8_or_missing_point_cannot_report_complete(self):
        def report(points):return {'status':'complete','state':{'l2_points':points}}
        p={'tag':'l2-00001','complete':True,'metrics':{'K8_tune2048':{'complete':False}}}
        for points in ([],[p],[dict(p,complete=False)]):
            self.assertTrue(S.completion_status(report(points),[1.])['status'].startswith('partial;'))
        p['metrics']['K8_tune2048']['complete']=True
        self.assertEqual(S.completion_status(report([p]),[1.])['status'],'complete')
        stopped=report([]);stopped['status']='stability_stop'
        self.assertEqual(S.completion_status(stopped,[1.])['status'],'stability_stop')
    def test_worker_without_L3_kind_keeps_both_L2_routes(self):
        import l2_worker as W, l1_cache, exp26_input, l2_resume, l2_sampler
        from l2_tests import Tests as Fixture
        fixture=Fixture();fixture.setUp()
        try:
            full={'heldout4k':[],'old1024':[{}]*1024}
            write(self.root/'heldout4k-reference-execution.json',{'rows':[]})
            write(self.root/'old1024-reference-execution.json',{'rows':[{'status':'numeric'}]*991+[{'status':'error'}]*33})
            cfg={'working_root':str(self.root),'cpu_smoke':True,'anchor_root':str(self.root),
                 'anchor_expected':{},'compact_cache':str(self.root),'weighted_order_root':str(self.root),
                 'cache_manifest_sha256':'tiny','weighted_order_sha256':'weighted',
                 'starter_dataset':str(self.root),'deadline_monotonic':time.monotonic()+30,
                 'phase_recipe':{'main_targets':[3.,6.,8.]}}
            path=self.root/'config.json';write(path,cfg)
            for gpu in (0,1):
                meta={'file':{'sha256':'tiny-order' if gpu==0 else 'weighted'}}
                with patch.object(sys,'argv',['l2_worker.py','--gpu',str(gpu),'--config',str(path)]),\
                     patch.object(l2_resume,'verify_anchor',return_value=(None,fixture.anchor)),\
                     patch.object(l2_resume,'branch_identity',return_value={'fork':'L2'}) as fork,\
                     patch.object(l1_cache,'CompactCache',return_value=fixture.cache),\
                     patch.object(l2_sampler,'load_order',return_value=(fixture.order,meta)) as load,\
                     patch.object(L,'identity_for',return_value=fixture.ident),\
                     patch.object(exp26_input,'validate_training_inputs',return_value=(full,None,None)),\
                     patch.object(R,'run') as run,patch.object(S,'run_continuation') as l3,\
                     patch.object(__import__('torch'),'set_num_interop_threads'):
                    W.main()
                run.assert_called_once();l3.assert_not_called()
                self.assertEqual(run.call_args.args[0],'C-main' if gpu==0 else 'C-long2x')
                self.assertEqual(run.call_args.kwargs,cfg['phase_recipe'])
                self.assertEqual(load.call_args.args[2],meta['file']['sha256'])
                self.assertEqual(fork.call_count,gpu)
        finally:fixture.tearDown()
    def test_owner_qualified_mounts_disambiguate_original_and_fallback_cache(self):
        from kaggle_io import resolve_dataset
        slug='nlp-hw02-compact-cache-verified'
        original=self.root/'notebooks'/'thisarivson'/slug
        fallback=self.root/'datasets'/'friend1'/slug
        original.mkdir(parents=True);fallback.mkdir(parents=True)
        with self.assertRaises(ValueError):resolve_dataset(self.root,slug)
        self.assertEqual(S.resolve_cache_mount(self.root,'thisarivson/'+slug+'/2','notebooks'),original)
        self.assertEqual(S.resolve_cache_mount(self.root,'friend1/'+slug,'datasets'),fallback)
        # Actual v2 layout: foreign notebook nested, own dataset at /input/slug.
        import shutil
        shutil.rmtree(fallback)
        flat=self.root/slug;flat.mkdir()
        self.assertEqual(S.resolve_cache_mount(self.root,'thisarivson/'+slug+'/2','notebooks'),original)
        self.assertEqual(S.resolve_cache_mount(self.root,'friend1/'+slug,'datasets'),flat)
        shutil.rmtree(original)
        self.assertEqual(S.resolve_cache_mount(self.root,'thisarivson/'+slug+'/2','notebooks'),flat)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(ROOT.parent/'checks/kaggle-l3-regressions.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,
           'failures':len(result.failures),'errors':len(result.errors),'GPU':False})
    sys.exit(not result.wasSuccessful())
