"""One E2 selection on E1 tune+eval, sealed before any phase-two run."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

WORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WORK/'research'), str(WORK/'kaggle/src')]
from e2_policies import grid, baseline, predict
spec = importlib.util.spec_from_file_location('e1_analysis', WORK/'research/e1-analysis.py')
E1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(E1)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def now(): return datetime.now(timezone.utc).isoformat()


def paired(a, b):
    d = np.asarray(b, dtype=np.int8)-np.asarray(a, dtype=np.int8)
    values, counts = np.unique(d, return_counts=True)
    boot = np.random.default_rng(2026102002).multinomial(len(d), counts/len(d), size=20000)@values/len(d)*100
    return dict(n=len(d), baseline_matches=int(sum(a)), matches=int(sum(b)),
                delta_pp=float(d.mean()*100), ci95_pp=np.quantile(boot, [.025, .975]).tolist(),
                gains=int((d>0).sum()), losses=int((d<0).sum()),
                bootstrap='paired tasks;20000;seed2026102002;percentile95')


def raw_extra(usage, rates):
    total = 0.
    for pool, n in usage.items():
        if pool.endswith('/0.7'):
            total += (n-1)*rates[pool+':K15']
        elif pool.endswith('/1.0'):
            total += n*rates[pool+':K8']
    return total


def select(raw):
    target = WORK/'results/e2-selection.json'
    assert not target.exists(), 'Selection already sealed; do not overwrite'
    prereg = read(WORK/'checks/e2-preregistration.json')
    text = (WORK/'notes/e2-policy.md').read_text().split('## Результаты')[0]
    assert text == prereg['text'] and hashlib.sha256(text.encode()).hexdigest() == prereg['sha256']
    rows = []; summaries = []; manifests = {}
    for split in ('tune2048', 'eval2048'):
        rr, ss, mm = E1.load(raw, split)
        # Cross LL and soups are irrelevant to E2; remove heavy unused values.
        for r in rr:
            for c in r['candidates']: c.pop('ll')
        rows += rr; summaries.append(ss); manifests.update(mm)
    assert len(rows) == len({r['id'] for r in rows}) == len({r['question_sha256'] for r in rows}) == 4096
    through = defaultdict(lambda: [0., 0])
    for summary in summaries:
        for t in summary['timings']:
            if t['kind'] == 'generation':
                key = t['label']+':K'+str(t['K'])
                through[key][0] += t['seconds']; through[key][1] += t['sequences']
    rates = {k: s/n for k, (s, n) in through.items()}
    timing_path = WORK/'results/kaggle-runs/nlp-hw02-ensemble-inference-v11/interactive/test-1-timing.json'
    timing = read(timing_path); phases = timing['phases_seconds']
    actual_stage2 = phases['stage2_sampling_restore_submit']
    fixed = timing['full_pipeline_seconds']-actual_stage2
    control = []; base_usage = Counter(); stops = []
    for row in rows:
        w, stop, third, used, f = predict(row, baseline())
        old, oldstop, oldused, _ = E1.predict(row, dict(anchor='C/g', limits=[1, 8, 8]))
        assert (w['code'], w['execution'], stop, used) == (old['code'], old['execution'], oldstop, E1.costs(oldused))
        control.append(E1.correct(row['reference'], w)); base_usage.update(used); stops.append(stop)
    mean_base = {k: v/len(rows) for k, v in base_usage.items()}
    raw_baseline = raw_extra(mean_base, rates)*1919
    factor = actual_stage2/raw_baseline
    results = []
    for i, policy in enumerate(grid()):
        bits = []; usage = Counter(); early = third_count = 0; winners = []
        for row in rows:
            w, stop, third, used, f = predict(row, policy)
            bits.append(E1.correct(row['reference'], w)); usage.update(used)
            early += stop; third_count += third
            winners.append((row['id'], w['code'], w['execution']))
        means = {k: v/len(rows) for k, v in usage.items()}
        extra = raw_extra(means, rates)*1919
        barrier = phases['stage1_decision_wait'] if 'threshold' in policy else 0.
        forecast = fixed+extra*factor+barrier
        results.append(dict(policy=policy, **paired(control, bits), early_stops=early,
                            third_rows=third_count, mean_usage=means, forecast_seconds=forecast,
                            raw_extra_seconds=extra, extra_barrier_seconds=barrier,
                            eligible=forecast<=150, winner_sha256=hashlib.sha256(
                                json.dumps(winners, ensure_ascii=False).encode()).hexdigest()))
        if i % 12 == 0: print('E2 evaluated', i+1, '/252', flush=True)
    eligible = [r for r in results if r['eligible']]
    primary = min(eligible, key=lambda r: (-r['matches'], r['forecast_seconds'], r['policy']['name']))
    selection = dict(at_utc=now(), selected_on='E1 tune2048+eval2048;4096 shared heldout4k tasks',
                     primary=primary['policy'], result=primary,
                     preregistration_sha256=prereg['sha256'],
                     analyzer_sha256=sha(__file__), policy_sha256=sha(WORK/'research/e2_policies.py'),
                     e1_analyzer_sha256=sha(WORK/'research/e1-analysis.py'),
                     raw_session_sha256=sha(raw/'e1/summary.json'), journal_sha256=manifests,
                     timing_sha256=sha(timing_path), time_model=dict(rates=rates,
                         fixed_seconds=fixed, stage2_seconds=actual_stage2,
                         calibration=factor, baseline_raw_extra_seconds=raw_baseline,
                         baseline_forecast_seconds=fixed+raw_baseline*factor),
                     rule='maximum combined matches;forecast<=150;ties:min forecast,lexical name',
                     phase2_started=False, independent_holdout_conflict=True)
    write(WORK/'results/e2-phase1.json', dict(at_utc=now(), rows=results,
          eligible=len(eligible), baseline_parity_rows=len(rows),
          family_best={f:min([r for r in eligible if r['policy']['family']==f],
            key=lambda r:(-r['matches'],r['forecast_seconds'],r['policy']['name']))
            for f in ('base','third')}, time_model=selection['time_model']))
    write(target, selection)
    write(WORK/'checks/e2-selection.json', dict(at_utc=now(), selection_sha256=sha(target),
        primary=primary['policy']['name'], phase2_started=False, independent_holdout_conflict=True))
    print(json.dumps(dict(primary=primary, selection_sha256=sha(target), eligible=len(eligible)),ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('raw', type=Path)
    select(parser.parse_args().raw)
