"""Frozen B3/B2/B1/P1 regressions without rewriting their historical receipts."""
import unittest
import b3_tests,b1_ev_b2_tests,b2_review_tests,b1_tests,p1_tests
from build_l3 import ROOT,write


def main():
    rows=[]
    for module,prefix in ((b3_tests,'test_b3_'),(b1_ev_b2_tests,'test_'),(b2_review_tests,'test_'),(b1_tests,'test_'),(p1_tests,'test_')):
        cls=getattr(module,'Tests',None) or module.ReviewTests
        names=sorted(n for n in cls.__dict__ if n.startswith(prefix))
        result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(cls(n) for n in names))
        rows.append(dict(suite=module.__name__,passed=result.wasSuccessful(),tests=result.testsRun,
                         failures=[(str(t),s) for t,s in result.failures+result.errors]))
    write(ROOT.parent/'checks/b4-regressions.json',dict(passed=all(r['passed'] for r in rows),suites=rows,GPU=False))
    raise SystemExit(not all(r['passed'] for r in rows))


if __name__=='__main__':main()
