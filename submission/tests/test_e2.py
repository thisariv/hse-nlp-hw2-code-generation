"""E2 strict threshold, denominator and mixed temperatures under compaction."""
import unittest
from unittest.mock import patch

import torch
from hw2_codegen import decode as D
from hw2_codegen.vote import replay_policy, winner_fraction


def candidate(value, lp=-1.):
    return dict(code=str(value),mean_logprob=lp,execution=dict(status='numeric',value=value))


class E2Policy(unittest.TestCase):
    def test_failures_count_in_denominator_and_threshold_is_strict(self):
        specs=[dict(name='C',K1=1,K2=2,K3=2),dict(name='P',K1=1,K2=8,K3=16),dict(name='W',K1=1,K2=8,K3=8)]
        greedy=candidate(0,0.)
        # 19 votes: nine equal values, ten failures/other answers => stage3.
        samples={'C':[candidate(1),candidate(1)],'P':[candidate(1) for _ in range(7)]+[dict(code='fail',mean_logprob=-1.,execution=dict(status='error'))],
                 'W':[candidate(i+2) for i in range(8)]}
        samples['P'] += [candidate(77) for _ in range(8)]
        record=dict(greedy=greedy,samples=samples)
        self.assertEqual(winner_fraction(record,specs),9/19)
        self.assertTrue(replay_policy(greedy,samples,specs,e2_primary=True)[2])
        # Exactly one half must NOT trigger expansion. Generic two-vote case.
        small=[dict(name='C',K1=1,K2=1)]
        self.assertEqual(winner_fraction(dict(greedy=greedy,samples={'C':[candidate(2)]}),small),.5)

    def test_numeric_bridge_is_one_component(self):
        specs=[dict(name='C',K1=1,K2=2)]
        a,b,c=[candidate(v) for v in [1.,1.0000009,1.0000018]]
        self.assertEqual(winner_fraction(dict(greedy=a,samples={'C':[b,c]}),specs),1.)

    def test_mixed_temperatures_keep_slot_identity_after_compaction(self):
        # Controlled decoder: sequence0 finishes immediately and compacts away;
        # sequence1 retains T1.0 on its next step instead of taking slot0's T0.7.
        class Decoder:
            def __init__(self,context,k,share_cross):
                self.n=2;self.t=0;self.cross_bytes=0;self.active=torch.arange(2)
            def step(self,tokens):
                self.t+=1
                return torch.tensor([[0.,0.,0.,1.]]).repeat(self.n,1)
            def select(self,index):
                self.n=len(index);self.active=self.active[index]
        seen=[]
        def pick(prob,noise):
            seen.append(prob.clone())
            return torch.tensor([2,3]) if len(seen)==1 else torch.tensor([2])
        class Context:
            src=torch.tensor([[4]])
            capacity=4
        with patch.object(D,'Decoder',Decoder),patch.object(D,'race_pick',side_effect=pick):
            D.decode(Context(),k=2,temperature=[.7,1.],options=D.Options(eos_every=1))
        self.assertEqual([len(p) for p in seen],[2,1])
        self.assertTrue(torch.equal(seen[0][1],seen[1][0]))
        self.assertFalse(torch.equal(seen[0][0],seen[0][1]))


if __name__=='__main__':unittest.main()
