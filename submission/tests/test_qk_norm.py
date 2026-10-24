"""Standalone clean-package QK-norm compatibility, masks, cache and strict checkpoint loading."""
import tempfile
import unittest
from pathlib import Path
import torch
from hw2_codegen.model import from_spec
from hw2_codegen.decode import Context, BaseDecoder
from hw2_codegen.pipeline import load_model as load

SPEC=dict(d_model=16,heads=2,encoder_layers=1,decoder_layers=1,swiglu_dim=24,vocab=31)

torch.set_num_threads(1)

class QKNorm(unittest.TestCase):
    def test_headwise_normalization_and_cached_prefix(self):
        m=from_spec(dict(SPEC,qk_norm='layernorm')).eval()
        src=torch.tensor([[1,4,2,0],[1,5,6,2]]);tgt=torch.tensor([[1,7,8],[1,9,10]])
        a=m.transformer.decoder.layers[0].self_attn
        q=torch.randn(2,2,3,8)*50+100
        torch.testing.assert_close(a.q_norm(q).mean(-1),torch.zeros(2,2,3),atol=1e-6,rtol=0)
        d=BaseDecoder(Context(m,src,8),1)
        for t in range(3):
            torch.testing.assert_close(d.step(tgt[:,t]),m(src,tgt[:,:t+1])[:,-1],rtol=2e-5,atol=2e-5)

    def test_flag_absent_and_false_legacy_parameters_exact(self):
        torch.manual_seed(42);old=from_spec(SPEC)
        torch.manual_seed(42);explicit=from_spec(dict(SPEC,qk_norm=False))
        self.assertEqual(old.state_dict().keys(),explicit.state_dict().keys())
        for k,v in old.state_dict().items():self.assertTrue(torch.equal(v,explicit.state_dict()[k]))
        self.assertFalse(any('q_norm' in k for k in old.state_dict()))

    def test_checkpoint_spec_strict_qk_and_legacy(self):
        for flag in (False,'layernorm'):
            spec=dict(SPEC,qk_norm=flag);m=from_spec(spec).eval()
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'model.pt'
                p=dict(format='nlp-hw02-model-only-v1',optimizer_included=False,model_spec=spec,model=m.state_dict())
                torch.save(p,path);loaded=load(path,torch.device('cpu'))
                src=torch.tensor([[1,4,2]]);tgt=torch.tensor([[1,7]])
                torch.testing.assert_close(loaded(src,tgt),m(src,tgt),rtol=0,atol=0)
                if flag:
                    p['model_spec']=dict(SPEC,qk_norm=False);torch.save(p,path)
                    with self.assertRaises(RuntimeError):load(path,torch.device('cpu'))

if __name__=='__main__':unittest.main()
