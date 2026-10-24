"""V3–V6 regressions without changing frozen receipts, notebooks or sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


def child(script, *args, timeout=180.):
    r = subprocess.run([sys.executable, str(ROOT/script), *args], capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, f'{script} {args}: {r.stderr}\n{r.stdout}'
    return json.loads(r.stdout.splitlines()[-1])


def legacy():
    import final_v6_regressions as old
    import final_session as session
    # Historical L2 weights were cleaned locally. The legacy session suite
    # mocks all launches; only supply names for its two unused weight paths.
    # Real L2 journals and real L3 weights are still checked without mocks.
    original = session.unique
    names = {f'model-{n}__weights__l2-28800-fp16.pt' for n in ['C-main', 'C-long2x']}
    with tempfile.TemporaryDirectory(prefix='v7-legacy-weight-names-') as tmp:
        tmp = Path(tmp)
        for name in names: (tmp/name).touch()
        def unique(root, name):
            if name in names and not list(Path(root).rglob(name)):
                return str(tmp/name)
            return original(root, name)
        with patch.object(session, 'unique', unique):
            report = old.legacy()
    report['legacy_mock_session_weight_path_fixture'] = True
    return report


def main():
    paths = list((WORK/'checks').glob('final-inference-v[3456]*.json'))
    paths += [ROOT/'final-v5-source-identity.json', ROOT/'final-v6-source-identity.json']
    paths += [WORK/'results/kaggle-runs/nlp-hw02-kaggle-l3/final'/name for name in
              ['next-inference.json', 'next-inference-v6.json']]
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    from final_v5_checks import unit
    report = {'v3': child('final_v7_regressions.py', '--legacy-child'), 'v5_unit': unit()}
    for mode in ['v4', 'real-decoder']:
        report[mode] = child('final_v6_regressions.py', '--original-child', mode)
    report['v5_fixtures'] = [child('final_v5_checks.py', '--fixture', case) for case in
                           ['complete', 'guard1', 'guard2', 'oom', 'priority', 'priority_guard']]
    report['v6_fixtures'] = [child('final_v6_checks.py', '--child', 'fixture', '--case', case) for case in
                           ['complete', 'guard1', 'guard2', 'oom', 'priority', 'priority_guard']]
    for mode in ['real_pipeline', 'decoder', 'executor']:
        report['v6_'+mode] = child('final_v6_checks.py', '--child', mode)
        print('v6', mode, 'passed', flush=True)
    from final_v6_checks import static, suite
    sources, report['v6_static'] = static()
    report['v6_session'] = suite(sources)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in digests.items())
    report.update(passed=True, GPU=False, push=False, frozen_receipts_unchanged=True)
    (WORK/'checks/final-inference-v7-regressions.json').write_text(json.dumps(report, indent=2)+'\n')
    print('V3–V6 regressions passed; frozen receipts unchanged', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--legacy-child', action='store_true')
    if parser.parse_args().legacy_child: print(json.dumps(legacy()))
    else: main()
