"""L4's five-point gate and production worker route; reuse frozen L2 math."""
import copy
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import l3_resume as S
import l2_runtime as R
from kaggle_io import sha,write


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='l4-tests-');self.root=Path(self.tmp.name)
        self.pins=json.loads((ROOT/'l4-anchors.json').read_text())
    def tearDown(self): self.tmp.cleanup()
    def test_five_inherited_points_and_exact_tags_required(self):
        pin=copy.deepcopy(self.pins['anchors']['C-main'])
        file=self.root/'anchor.pt';file.write_bytes(b'tiny')
        pin.update(file=file.name,bytes=4,sha256=sha(file))
        payload={'model_spec':pin['model_spec'],'scaler_enabled':True,'rng':{'torch_cuda':[1]},
                 'state':{k:pin[k] for k in ('updates','cursor','main_seconds','l2_main_seconds')}}
        payload['state'].update(last_lr=.0015,stability={'escalations':0},l2_points=[
            {'tag':t,'complete':True} for t in ['l2-10800','l2-21600','l2-28800','l2-45540','l2-62280']])
        payload['state']['l2_points'][-1]['metrics']={'K8_tune2048':{'complete':True}}
        with patch.object(R,'read_full',return_value=payload): S.verify_anchor(self.root,pin,False,expected_points=5)
        for mutate in [lambda h:h.pop(),lambda h:h[4].update(complete=False),lambda h:h[4].update(tag='l2-62279'),
                       lambda h:h[4]['metrics']['K8_tune2048'].update(complete=False)]:
            bad=copy.deepcopy(payload);mutate(bad['state']['l2_points'])
            with patch.object(R,'read_full',return_value=bad):
                with self.assertRaises(AssertionError):S.verify_anchor(self.root,pin,False,expected_points=5)
    def test_new_cumulative_targets_do_not_repeat_L3(self):
        cfg={'phase_recipe':{'main_targets':[15300.,30600.],'cooldown_seconds':[900.,3000.],'eval_seconds':[300.,1800.]}}
        for pin in self.pins['anchors'].values():
            anchor={'state':{'l2_main_seconds':pin['l2_main_seconds'],'l2_points':['inherited']}}
            before=copy.deepcopy(anchor);recipe=S.phase_recipe(cfg,anchor)
            self.assertEqual([int(t) for t in recipe['main_targets']],[77580,92880])
            self.assertEqual(anchor,before)
    def test_final_K8_requires_all_2048_rows_even_if_marked_complete(self):
        for vote in (None, {}, {'complete':True}, {'complete':True,'rows':0,'target_rows':2048},
                     {'complete':True,'rows':2048,'target_rows':32},
                     {'complete':False,'rows':2048,'target_rows':2048}):
            with self.subTest(vote=vote):
                metrics={} if vote is None else {'K8_tune2048':vote}
                result={'status':'complete','state':{'l2_points':[
                    {'tag':'l2-77580','complete':True,'metrics':{}},
                    {'tag':'l2-92880','complete':True,'metrics':metrics}]}}
                self.assertTrue(S.completion_status(result,[77580.,92880.],2048)['status'].startswith('partial;'))
        result['state']['l2_points'][-1]['metrics']={'K8_tune2048':{'complete':True,'rows':2048,'target_rows':2048}}
        result['status']='complete'
        self.assertEqual(S.completion_status(result,[77580.,92880.],2048)['status'],'complete')
    def test_real_missing_final_K8_reports_partial_and_preserves_full_state(self):
        from l2_tests import Tests as Fixture, CPU, SPEC
        from l2_resume import initial_state
        fixture=Fixture();fixture.setUp()
        try:
            anchor=R.payload(fixture.model,fixture.opt,fixture.scaler,
                             initial_state(fixture.anchor,'C-main'),SPEC,fixture.ident)
            cfg={'session_kind':'l4','phase_recipe':{'main_targets':[.001,1.03],
                 'cooldown_seconds':[.001,.001],'eval_seconds':[1.,1.]}}
            original=R.run
            with patch.object(R,'run',side_effect=lambda *a,**kw:original(*a,smoke=True,**kw)):
                result=S.run_continuation('C-main',anchor,fixture.ident,fixture.cache,fixture.order,
                     self.root,CPU,time.monotonic()+30,lambda *args:{},cfg)
            self.assertTrue(result['status'].startswith('partial; L4'))
            self.assertEqual(len(result['state']['l2_points']),2)
            self.assertTrue(all(p['complete'] for p in result['state']['l2_points']))
            self.assertEqual(json.loads((self.root/'summary.json').read_text())['status'],result['status'])
            receipt=json.loads((self.root/'latest.json').read_text())
            self.assertEqual(R.read_full(self.root/receipt['file'],fixture.ident)['state'],result['state'])
        finally:fixture.tearDown()
    def test_worker_uses_own_L3_anchor_and_five_point_gate_on_both_GPUs(self):
        import l2_worker as W,l1_cache,exp26_input,l2_sampler
        import numpy as np
        from types import SimpleNamespace
        full={'heldout4k':[],'old1024':[{}]*1024}
        inputs=self.root/'inputs';inputs.mkdir()
        for split in full:
            refs=[{'status':'numeric' if i<991 else 'error'} for i in range(len(full[split]))]
            write(inputs/(split+'-reference-execution.json'),{'rows':refs})
        for gpu,name in enumerate(['C-main','C-long2x']):
            pin=self.pins['anchors'][name];anchor={'identity':pin['identity']}
            cfg={'session_kind':'l4','inherited_points':5,'cpu_smoke':True,'anchor_root':'source',
                 'resume_anchors':self.pins['anchors'],'working_root':str(self.root),'compact_cache':'cache',
                 'cache_manifest_sha256':pin['identity']['cache_manifest_sha256'],'weighted_order_root':'weighted',
                 'starter_dataset':str(inputs),'deadline_monotonic':123.,'phase_recipe':{}}
            write(self.root/'config.json',cfg)
            cache=SimpleNamespace(manifest_sha=cfg['cache_manifest_sha256'])
            meta=pin['identity']['recipe'].get('sampler',{})
            with patch.object(sys,'argv',['worker','--gpu',str(gpu),'--config',str(self.root/'config.json')]),\
                 patch('torch.set_num_interop_threads'),\
                 patch.object(S,'verify_anchor',return_value=(None,anchor)) as verify,\
                 patch.object(S,'run_continuation') as run,\
                 patch.object(l1_cache,'CompactCache',return_value=cache),\
                 patch.object(l2_sampler,'load_order',return_value=(np.arange(4),meta)),\
                 patch.object(exp26_input,'validate_training_inputs',return_value=(full,None,None)):
                W.main()
                self.assertEqual(verify.call_args.kwargs['expected_points'],5)
                self.assertEqual(verify.call_args.args[1],pin)
                self.assertEqual(run.call_args.args[0],name)
                self.assertIs(run.call_args.args[1],anchor)
                self.assertEqual(run.call_args.args[2],pin['identity'])


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(ROOT.parent/'checks/kaggle-l4-regressions.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,
           'failures':len(result.failures),'errors':len(result.errors),'GPU':False})
    sys.exit(not result.wasSuccessful())
