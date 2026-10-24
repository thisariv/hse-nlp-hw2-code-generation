"""Independent B2 regressions; tiny real training plus the corrected wall-budget gate."""
import copy
import json
import math
import time
import unittest
from unittest.mock import patch
import b1_ev_b2_tests as base
import b2_runtime as B
import l2_runtime as R
import torch


class ReviewTests(unittest.TestCase):
    setUp = base.Tests.setUp
    tearDown = base.Tests.tearDown
    save = base.Tests.save
    metrics = base.Tests.metrics

    def test_builder_and_push_receipt_use_canonical_title_slug(self):
        import re
        import shutil
        import build_b2
        from b2_review_api import pushed_ref
        folder=self.root/'builder';(folder/'notebooks').mkdir(parents=True)
        (folder/'src').symlink_to(base.ROOT/'src',target_is_directory=True)
        original=base.ROOT/'notebooks/nlp-hw02-kaggle-b1.ipynb'
        shutil.copyfile(original,folder/'notebooks'/original.name)
        sources=base.literals(next(c['source'] for c in json.loads(original.read_text())['cells']
                                   if c['cell_type']=='code'))['SOURCES']
        for n in [*sources,'b2_worker.py']:
            if not (folder/'src'/n).exists():shutil.copyfile(base.ROOT/n,folder/n)
        with patch.object(build_b2,'ROOT',folder):build_b2.build('friend2')
        meta=json.loads((folder/'bundles/b2_friend2/kernel-metadata.json').read_text())
        canonical='friend2/'+re.sub('[^a-z0-9]+','-',meta['title'].lower()).strip('-')
        self.assertEqual(meta['id'],canonical)
        self.assertEqual(canonical,'friend2/nlp-hw02-b2')
        stdout='Kernel version 1 successfully pushed. https://www.kaggle.com/code/'+canonical
        self.assertEqual(pushed_ref(stdout,'friend2'),canonical)
        with self.assertRaises(AssertionError):pushed_ref('Kernel version 1 successfully pushed','friend2')
        with self.assertRaises(AssertionError):pushed_ref(stdout,'friend1')

    def test_pair_branch_cannot_spend_end_main_reserve(self):
        # Enough for one branch and the loop's 60s margin, insufficient for two.
        self.state.update(updates=0, cursor=0, next_point_pairs=0)
        self.save()
        cfg = dict(owner='test', dropout=.2, deadline_monotonic=time.monotonic()+63,
                   io_reserve_seconds=0, main_seconds=9999., max_updates=1,
                   probe_cooldown_seconds=1., probe_eval_seconds=1., final_session=False)
        calls=[]
        def evaluate(*args): calls.append(1); return self.metrics()
        with patch.object(B.time, 'monotonic', return_value=cfg['deadline_monotonic']-63):
            result=B.run(self.name,self.cache,self.order,self.meta,self.root/'run',base.CPU,cfg,
                         evaluate,resume=self.root/'state',spec=base.SPEC,cpu=True)
        self.assertEqual(len(calls),1)
        self.assertTrue(result['end_main_complete'])
        self.assertEqual(result['state']['missed_points'],[0])
        self.assertTrue(result['status'].startswith('partial'))

    def test_recovery_thresholds_ignore_isolated_amp_skips(self):
        h=copy.deepcopy(self.state['stability'])
        for _ in range(200):
            self.assertFalse(R.observe(h,0,amp_skip=1))
            self.assertFalse(R.observe(h,0))
        self.assertEqual(h['total'],0)
        for _ in range(31): self.assertFalse(R.observe(h,0,amp_skip=1))
        self.assertTrue(R.observe(h,0,amp_skip=1))
        for mode in ('consecutive','window'):
            h=copy.deepcopy(self.state['stability'])
            failures=3 if mode=='consecutive' else 10
            for j in range(failures):
                self.assertEqual(R.observe(h,1),j==failures-1)
                if mode=='window' and j<failures-1:self.assertFalse(R.observe(h,0))

    def test_half_norm_and_cached_prefix_v7_and_clean(self):
        m=B.new_model(base.SPEC,base.CPU).eval().half()
        q=torch.full((2,2,3,4),60000.,dtype=torch.float16)
        self.assertTrue(torch.isfinite(m.transformer.decoder.layers[0].self_attn.q_norm(q)).all())
        src=torch.tensor([[1,4,2,0],[1,5,6,2]]);tgt=torch.tensor([[1,7,8],[1,9,10]])
        final=base.FC.SimpleTransformer(**base.SPEC,dropout=0.).eval().half();final.load_state_dict(m.state_dict())
        clean=base.Clean.from_spec(base.SPEC).eval().half();clean.load_state_dict(m.state_dict())
        ds=[base.FF.Decoder(base.FF.Context(final,src,8),1),
            base.Decode.BaseDecoder(base.Decode.Context(clean,src,8),1),base.C.CachedDecoder(m,src,8)]
        for j in range(3):
            expected=m(src,tgt[:,:j+1])[:,-1]
            self.assertTrue(torch.isfinite(expected).all())
            for d in ds:torch.testing.assert_close(d.step(tgt[:,j]),expected,rtol=.01,atol=.003)

    def test_attention_matches_explicit_headwise_math_and_single_scale(self):
        from torch.nn import functional as F
        a=self.m.transformer.decoder.layers[0].self_attn.eval()
        x=torch.randn(2,3,8)*20+40
        q,k,v=F.linear(x,a.in_proj_weight,a.in_proj_bias).chunk(3,-1)
        def heads(t):return t.reshape(2,3,2,4).transpose(1,2)
        def norm(t,layer):
            t=heads(t);t=(t-t.mean(-1,keepdim=True))/torch.sqrt(t.var(-1,correction=0,keepdim=True)+1e-5)
            return t*layer.weight+layer.bias
        logits=norm(q,a.q_norm)@norm(k,a.k_norm).transpose(-1,-2)/2
        mask=torch.ones(3,3,dtype=torch.bool).triu(1)
        weights=logits.masked_fill(mask,float('-inf')).softmax(-1)
        expected=a.out_proj((weights@heads(v)).transpose(1,2).reshape(2,3,8))
        for need in (True,False):
            got,_=a(x,x,x,attn_mask=mask,need_weights=need)
            torch.testing.assert_close(got,expected,rtol=3e-5,atol=3e-5)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReviewTests))
    base.write(base.ROOT.parent/'checks/b2-review-regressions.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        failures=[(str(t),s) for t,s in result.failures+result.errors],torch=str(torch.__version__)))
    raise SystemExit(not result.wasSuccessful())
