"""B7 config/full-length LR, resume and B6 regressions. CPU only, own receipts."""
import argparse, copy, hashlib, json, sys, unittest
from pathlib import Path
from unittest.mock import patch
import b6_tests as B6
import b5_state as S
import b6_state as D
import exp26_train as T
from l2_gpu_smoke import digest
from p1_math import sqrt_lr
from build_l3 import ROOT, literals, write
from build_b7 import build
from b7_budget import calculate, REQUESTED, HARD, WORKER


class Tests(B6.Tests):
    def test_b7_only_N_and_session_envelope_change(self):
        allowed = {'session_kind', 'comparison_protocol', 'N_by_gpu', 'budget_seconds',
            'worker_deadline_seconds', 'hard_shutdown_seconds', 'session_ceiling_seconds'}
        for smoke in (False, True):
            path, cfg = build(smoke)
            source = ROOT/'notebooks'/('nlp-hw02-b6-gpu-smoke.ipynb' if smoke else 'nlp-hw02-b6.ipynb')
            old = [c['source'] for c in json.loads(source.read_text())['cells'] if c['cell_type']=='code']
            new = [c['source'] for c in json.loads(path.read_text())['cells'] if c['cell_type']=='code']
            prior = literals(old[0])['CONFIG']
            self.assertEqual({k:v for k,v in prior.items() if k not in allowed}, {k:v for k,v in cfg.items() if k not in allowed})
            def strip_config(s):
                a=s.index('CONFIG='); b=s.index('WORKING=Path',a)
                return s[:a]+s[b:]
            self.assertEqual(strip_config(old[0]),strip_config(new[0])); self.assertEqual(old[1:],new[1:])
            self.assertEqual(cfg['N_by_gpu'], {'0':200,'1':200} if smoke else REQUESTED)
            self.assertEqual(cfg['session_ceiling_seconds'],720 if smoke else HARD)
            self.assertEqual(cfg['b6_pins']['anchors'],prior['anchors'])

    def test_b7_actual_budget_full_N_admitted(self):
        b=calculate(); self.assertEqual(b['N_by_gpu'],REQUESTED); self.assertEqual(b['proportional_scale'],1.)
        self.assertIsNone(b['reduction_reason']); self.assertLess(b['expected_wall_seconds'],WORKER)
        for r in b['per_gpu'].values():
            self.assertEqual(r['N'],2*r['B6_N'])
            self.assertAlmostEqual(r['expected_wall_seconds'],r['update_seconds']*1.1+480+1800+600)

    def test_b7_exact_full_N_steps_and_zero_endpoint(self):
        # Real schedule/cursor/skip controller for all 98800 slots; forwards mocked.
        pins=json.loads((ROOT/'b6-anchors.json').read_text())
        for g,n in REQUESTED.items():
            pin=pins['anchors'][g]; peak=pins['budget'][g]['peak_lr']
            state=dict(cursor=pin['cursor'],updates=pin['updates'],last_lr=peak,micro_batch=32,
                stability=dict(total=0,consecutive=0,window=[]))
            initial=copy.deepcopy(state); cd=D.new_cooldown(state,n,pin['sha256']); seen=[]
            def update(identity,m,o,s,cache,ids,device,micro,lr):
                seen.append(lr)
                return dict(seconds=0.,AMP_skip=int(len(seen)==3),nonfinite_skip=0,lr=lr,pairs=len(ids)),micro
            with patch.object(S,'update_for',side_effect=update):
                for _ in range(n):
                    _,unsafe=D.step(None,None,None,state,cd,self.cache,self.order,B6.CPU,{})
                    self.assertFalse(unsafe)
                with self.assertRaises(AssertionError):D.step(None,None,None,state,cd,self.cache,self.order,B6.CPU,{})
            self.assertEqual(len(seen),n); self.assertEqual(seen,[sqrt_lr(j,n,peak) for j in range(1,n+1)])
            self.assertTrue(all(a>b for a,b in zip(seen,seen[1:]))); self.assertEqual(seen[-1],0.)
            self.assertEqual(cd['cursor'],pin['cursor']+128*n); self.assertEqual(cd['AMP_skips'],1); self.assertEqual(state,initial)

    def test_b7_new_N_disk_resume_both_paths(self):
        for kind,g in (('W','0'),('C','1')):
            src,pin=self.prepare(kind); m,o,s,state=S.restore(self.fixture_read(src,self.ident),B6.CPU)
            cd=D.new_cooldown(state,REQUESTED[g],pin['sha256'])
            D.step(m,o,s,state,cd,self.cache,self.order,B6.CPU,self.ident)
            out=self.root/('b7-resume-'+kind); D.save_full(out,m,o,s,state,pin['model_spec'],self.ident,cd)
            D.step(m,o,s,state,cd,self.cache,self.order,B6.CPU,self.ident)
            normalized=copy.deepcopy(cd);normalized['seconds']=0.
            expected=digest((m.state_dict(),o.state_dict(),s.state_dict(),state,normalized,T.rng_state()))
            m,o,s,state,cd=D.restore(D.read_full(out,self.ident,False),B6.CPU)
            D.step(m,o,s,state,cd,self.cache,self.order,B6.CPU,self.ident)
            # Wall-clock seconds are observational, all numerical/scheduler fields must match.
            normalized=copy.deepcopy(cd);normalized['seconds']=0.
            self.assertEqual(expected,digest((m.state_dict(),o.state_dict(),s.state_dict(),state,normalized,T.rng_state())))
            self.assertEqual(cd['N'],REQUESTED[g]); self.assertEqual(cd['updates'],2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--regressions',action='store_true');a=p.parse_args()
    suites=[(Tests,'test_b7_')]
    if a.regressions:
        import b6_regressions as r
        suites += [(B6.Tests,'test_b6_')]
        for module,prefix in ((r.b5_tests,'test_b5_'),(r.b4_tests,'test_b4_'),(r.b3_tests,'test_b3_'),
            (r.b1_ev_b2_tests,'test_'),(r.b2_review_tests,'test_'),(r.b1_tests,'test_'),(r.p1_tests,'test_'),(r.l2_tests,'test_'),(r.l3_tests,'test_')):
            cls=getattr(module,'Tests',None) or module.ReviewTests
            if module is r.l2_tests and sys.platform=='darwin':
                class SandboxTests(cls):
                    @unittest.skip('macOS sandbox denies killpg; inherited Linux supervisor unchanged')
                    def test_supervisor_error_still_records_and_packages_outputs(self):pass
                cls=SandboxTests
                names=[n for n in module.Tests.__dict__ if n.startswith(prefix)]
            else:names=[n for n in cls.__dict__ if n.startswith(prefix)]
            suites.append((cls,names))
    rows=[]
    for cls,names in suites:
        if isinstance(names,str):names=[n for n in cls.__dict__ if n.startswith(names)]
        result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cls(n) for n in sorted(names)))
        rows.append(dict(suite=cls.__module__,passed=result.wasSuccessful(),tests=result.testsRun,
            failures=[(str(t),s) for t,s in result.failures+result.errors],skipped=[(str(t),s) for t,s in result.skipped]))
    rec=dict(passed=all(r['passed'] for r in rows),suites=rows,GPU=False,regressions=a.regressions,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scripts={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('b6_tests.py','b6_regressions.py','build_b7.py','b7_budget.py')},
        modules={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')})
    write(ROOT.parent/'checks'/('b7-regressions.json' if a.regressions else 'b7-cpu.json'),rec)
    raise SystemExit(not rec['passed'])


if __name__=='__main__':main()
