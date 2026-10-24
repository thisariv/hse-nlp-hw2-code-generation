"""Run B5/P1 and the inherited B4/B3/B2/B1/L2/L3 regressions, without changing old receipts."""
import hashlib,sys,unittest
import b5_tests,b4_tests,b3_tests,b1_ev_b2_tests,b2_review_tests,b1_tests,p1_tests,l2_tests,l3_tests
from pathlib import Path
from build_l3 import ROOT,write


def main():
    rows=[]
    for module,prefix in ((b5_tests,'test_b5_'),(b4_tests,'test_b4_'),(b3_tests,'test_b3_'),
        (b1_ev_b2_tests,'test_'),(b2_review_tests,'test_'),(b1_tests,'test_'),(p1_tests,'test_'),(l2_tests,'test_'),(l3_tests,'test_')):
        cls=getattr(module,'Tests',None) or module.ReviewTests
        names=sorted(n for n in cls.__dict__ if n.startswith(prefix))
        if module is l2_tests and sys.platform=='darwin':
            class SandboxTests(cls):
                @unittest.skip('macOS sandbox denies killpg; inherited Linux supervisor unchanged')
                def test_supervisor_error_still_records_and_packages_outputs(self):pass
            cls=SandboxTests
        r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cls(n) for n in names))
        rows.append(dict(suite=module.__name__,passed=r.wasSuccessful(),tests=r.testsRun,
            failures=[(str(t),s) for t,s in r.failures+r.errors],skipped=[(str(t),s) for t,s in r.skipped]))
    write(ROOT.parent/'checks/b6-regressions.json',dict(passed=all(r['passed'] for r in rows),suites=rows,GPU=False,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        modules={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}))
    raise SystemExit(not all(r['passed'] for r in rows))


if __name__=='__main__':main()
