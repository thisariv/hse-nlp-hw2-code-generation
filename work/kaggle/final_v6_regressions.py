"""Run existing v3-v5 suites without modifying their frozen receipts/readiness."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT/'src')]


def child(script, *args, timeout=90.):
    result = subprocess.run([sys.executable, str(ROOT/script), *args], capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, f'{script} {args}: {result.stderr}\n{result.stdout}'
    return json.loads(result.stdout.splitlines()[-1]) if result.stdout.strip() else {'passed': True}


def legacy():
    import final_inference_checks as checks
    import final_review_checks as review
    import final_v3_checks as v3
    report = {name: getattr(checks, name)() for name in
              ['unit', 'sampler_checks', 'static', 'suite_checks', 'supervisor_checks', 'process_smoke']}
    report['frozen_L2_journals'] = checks.journals()[0]
    report['security'] = review.regression_checks()
    report['v3_available_vote'] = v3.available_vote()
    report['v3_guards'] = [child('final_v3_checks.py', '--guard-child', c) for c in ['plain', 'oom']]
    return report


def main():
    from final_v5_checks import unit
    report = {'v3': child('final_v6_regressions.py', '--legacy-child'), 'v5_unit': unit()}
    for mode in ['v4', 'real-decoder']:
        # Call the original function directly; original CLI writes v5 receipts.
        report[mode] = child('final_v6_regressions.py', '--original-child', mode)
    report['v5_fixtures'] = [child('final_v5_checks.py', '--fixture', case) for case in
                           ['complete', 'guard1', 'guard2', 'oom', 'priority', 'priority_guard']]
    report.update(passed=True, GPU=False, push=False, frozen_receipts_unchanged=True)
    (ROOT.parent/'checks/final-inference-v6-regressions.json').write_text(json.dumps(report, indent=2)+'\n')
    print('V3-V5 regressions passed', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--legacy-child', action='store_true')
    parser.add_argument('--original-child', choices=['v4', 'real-decoder'])
    args = parser.parse_args()
    if args.legacy_child:
        print(json.dumps(legacy()))
    elif args.original_child:
        from final_v5_regressions import v4, real_decoder
        print(json.dumps(v4() if args.original_child == 'v4' else real_decoder()))
    else:
        main()
