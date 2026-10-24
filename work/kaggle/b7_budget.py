"""Recompute B7 admission from SHA-verified complete B6 update journals; no API/GPU."""
import hashlib, json
from build_l3 import ROOT

HARD = 40800
WORKER = 40500
WATCHDOG = 40740
REQUESTED = {'0': 33000, '1': 65800}
SETUP, EVAL, IO, SAFETY = 480, 1800, 600, 1.10


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def calculate():
    pins = json.loads((ROOT/'b6-anchors.json').read_text())
    folder = ROOT.parent/'results/kaggle-runs/nlp-hw02-b6/1'
    inventory = {r['file']: r for r in json.loads((folder/'b1-output-manifest.json').read_text())['files']}
    def checked(name):
        path = folder/name
        assert path.stat().st_size == inventory[name]['bytes'] and sha(path) == inventory[name]['sha256'], name
        return path
    session = json.loads(checked('session-summary.json').read_text())
    assert session['status'] == 'complete' and session['exit_codes'] == [0, 0]
    branches = {}
    for g, n in REQUESTED.items():
        name = f'model-GPU{g}/cooldown-updates.jsonl'
        path = checked(name)
        logs = [json.loads(s) for s in path.read_text().splitlines()]
        summary = json.loads(checked(f'model-GPU{g}/summary.json').read_text())
        cd = summary['cooldown']; pin = pins['anchors'][g]
        assert summary['status'] == 'complete' and summary['final_complete']
        assert cd['N'] == cd['updates'] == len(logs) == pins['budget'][g]['N']
        assert cd['source_sha256'] == pin['sha256'] and cd['last_lr'] == 0.
        assert [r['branch_update'] for r in logs] == list(range(1, len(logs)+1))
        seconds = sum(r['seconds'] for r in logs)
        assert seconds == cd['seconds'] and seconds > 0
        per_update = seconds/len(logs)
        branches[g] = dict(N=n, B6_N=len(logs), peak_lr=pins['budget'][g]['peak_lr'],
            B6_update_seconds=seconds, reference_seconds_per_update=per_update,
            updates_per_second=1/per_update, B6_worker_wall_seconds=summary['wall_seconds'],
            log_file=name, log_sha256=sha(path), safety_factor=SAFETY,
            setup_seconds=SETUP, eval_seconds=EVAL, io_seconds=IO)
    # Both models receive exactly the same proportional reduction if admission ever requires it.
    available = WORKER-SETUP-EVAL-IO
    factor = min(1., *(available/(b['N']*b['reference_seconds_per_update']*SAFETY) for b in branches.values()))
    for b in branches.values():
        b['requested_N'] = b['N']; b['N'] = int(b['N']*factor)
        b['update_seconds'] = b['N']*b['reference_seconds_per_update']
        b['expected_wall_seconds'] = b['update_seconds']*SAFETY+SETUP+EVAL+IO
        b['hard_margin_seconds'] = HARD-b['expected_wall_seconds']
        assert b['expected_wall_seconds'] <= WORKER
    return dict(per_gpu=branches, N_by_gpu={g:b['N'] for g,b in branches.items()},
        proportional_scale=factor, reduction_reason=None if factor == 1. else 'Both N scaled proportionally to admit full session within worker deadline',
        expected_wall_seconds=max(b['expected_wall_seconds'] for b in branches.values()),
        hard_wall_seconds=HARD, worker_deadline_seconds=WORKER, watchdog_seconds=WATCHDOG,
        smoke_hard_seconds=720, B6_session_wall_seconds=session['wall_seconds'],
        B6_manifest_sha256=sha(folder/'b1-output-manifest.json'),
        basis='All B6 attempted-update logs, verified against producer manifest; 10% update reserve + setup480 + full final eval/K8 1800 + I/O600. T4x2 runs concurrently; session budget is max, not sum.')
