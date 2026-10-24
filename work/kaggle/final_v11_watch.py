"""Read-only observation for up to 45 minutes; never push/cancel/restart."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
API_PYTHON = Path('~/.local/pipx/venvs/kaggle/bin/python').expanduser()
LEDGER = WORK/'checks/final-inference-v11-launch.json'


def read(path):
    return json.loads(path.read_text())


def main():
    launched = datetime.fromisoformat(read(LEDGER)['requested_at_utc'])
    until = launched.timestamp() + 45*60
    while time.time() < until:
        interactive = WORK/'results/kaggle-runs/nlp-hw02-ensemble-inference-v11/interactive'
        if (interactive/'summary.json').exists() and any(p.exists() for p in [interactive/'test-1/answers.jsonl', interactive/'test-1-answers.jsonl']):
            print(json.dumps(dict(event='interactive_packet_ready', path=str(interactive.relative_to(WORK)))), flush=True)
            return
        try:
            result = subprocess.run([str(API_PYTHON), str(ROOT/'final_v11_api.py'), 'observe'],
                capture_output=True, text=True, timeout=50)
            if result.returncode == 0:
                observation = json.loads(result.stdout.strip().splitlines()[-1])
                print(json.dumps(observation), flush=True)
                if observation['status'] == 'COMPLETE' and observation['output_visible']:
                    return
            else:
                print(json.dumps(dict(event='read_only_observation_failed', returncode=result.returncode)), flush=True)
        except (subprocess.TimeoutExpired, ValueError):
            print(json.dumps(dict(event='read_only_observation_timeout_or_parse_error')), flush=True)
        time.sleep(min(30, max(0, until-time.time())))
    record = dict(event='45_minute_wait_finished', at_utc=datetime.now(timezone.utc).isoformat(),
        own_push_calls=read(LEDGER)['push_calls'], last_status=read(LEDGER).get('status'),
        monitor_version=read(LEDGER).get('monitor_version'), no_restart=True)
    (WORK/'checks/final-inference-v11-wait.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
