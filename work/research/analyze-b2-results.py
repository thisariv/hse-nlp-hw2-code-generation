"""CPU journal audit of B2, B1 EV and historical C; no model forwards or exec."""
import hashlib
import json
import math
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent
RUNS = WORK / 'results/kaggle-runs'
INPUTS = {}


def read(path):
    path = Path(path)
    INPUTS[str(path.relative_to(WORK))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def correct(reference, execution):
    return reference['status'] == execution['status'] == 'numeric' and math.isclose(
        reference['value'], execution['value'], rel_tol=1e-6, abs_tol=1e-6)


def paired(a, b):
    assert len(a) == len(b)
    n = len(a)
    gains = sum(not x and y for x, y in zip(a, b))
    losses = sum(x and not y for x, y in zip(a, b))
    delta = (gains-losses)/n
    half = 1.96*math.sqrt(((gains+losses)/n-delta**2)/n)
    return dict(n=n, gains=gains, losses=losses, delta_pp=100*delta,
                paired_normal_95CI_pp=[100*(delta-half), 100*(delta+half)])


def verify_rows(rows, reference):
    assert len(rows) == len(reference)
    for i, (row, ref) in enumerate(zip(rows, reference)):
        assert row['id'] == i
        for key in ['question_sha256', 'reference', 'original_bpe_length']:
            assert row[key] == ref[key], (i, key)
        assert row['correct'] == correct(row['reference'], row['execution'])


def bins(rows):
    bounds = [(0, 60), (60, 90), (90, 120), (120, math.inf)]
    out = []
    for lo, hi in bounds:
        sub = [r for r in rows if lo <= r['original_bpe_length'] < hi]
        out.append(dict(n=len(sub), matches=sum(r['correct'] for r in sub),
                        accuracy_pp=100*sum(r['correct'] for r in sub)/len(sub)))
    assert [b['n'] for b in out] == [2013, 1457, 488, 138]
    return out


def scores(rows):
    return dict(n=len(rows), matches=sum(r['correct'] for r in rows),
                accuracy_pp=100*sum(r['correct'] for r in rows)/len(rows))


def main():
    raw = RUNS/'nlp-hw02-b2/1'
    session = read(raw/'session-summary.json')
    assert session['status'] == 'complete' and session['exit_codes'] == [0, 0]
    manifest = read(raw/'b1-output-manifest.json')
    checked, missing = [], []
    for item in manifest['files']:
        path = raw/item['file']
        if not path.exists():
            missing.append(item['file'])
            continue
        assert path.stat().st_size == item['bytes'], item['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], item['file']
        checked.append(item['file'])

    base = RUNS/'nlp-hw02-kaggle-l2/1/model-C-main/points/l2-10800/evaluation'
    reference = {s: read(base/(s+'-per-example.json')) for s in ['heldout4k', 'old991']}
    records, points, stability = {}, [], {}
    for gpu in [0, 1]:
        root = raw/f'model-GPU{gpu}'
        summary = read(root/'summary.json')
        state = summary['state']; name = summary['model']
        logpath = root/'updates.jsonl'
        INPUTS[str(logpath.relative_to(WORK))] = hashlib.sha256(logpath.read_bytes()).hexdigest()
        logs = [json.loads(s) for s in logpath.read_text().splitlines()]
        assert len(logs) == state['updates']
        assert all(x['updates'] == i+1 and x['main_pairs'] == 128*(i+1)
                   for i, x in enumerate(logs))
        assert logs[-1]['main_seconds'] == state['main_seconds']
        assert math.isclose(sum(x['seconds'] for x in logs), state['main_seconds'])
        for logkey, statekey in [('AMP_skip', 'AMP_skips'), ('nonfinite_skip', 'nonfinite_skips')]:
            assert sum(x[logkey] for x in logs) == state[statekey]
        assert state['stability']['escalations'] == 0 and state['lr_factor'] == 1
        stability[name] = dict(main_attempts=len(logs), AMP_skips=state['AMP_skips'],
            nonfinite_skips=state['nonfinite_skips'], escalations=state['stability']['escalations'],
            lr_factor=state['lr_factor'], sampled_attention_max=max(
                x['sampled_attention_abs_logit_max'] for x in logs if 'sampled_attention_abs_logit_max' in x),
            OOM_retries=sum(len(x['OOM_retries']) for x in logs),
            token_weighted_z_loss=sum(x['z_loss']*x['tokens'] for x in logs)/sum(x['tokens'] for x in logs))
        for point in state['points']:
            p = root/'points'/point['tag']; training = read(p/'training.json')
            metrics = read(p/'evaluation/metrics.json'); complete = read(p/'complete.json')
            checkpoint = read(p/'checkpoint.json')
            assert complete['metrics'] == metrics == point['metrics']
            assert complete['training'] == training == point['training']
            assert complete['weights_sha256'] == metrics['weights_sha256']
            assert metrics['weights_sha256'] == checkpoint['sha256']
            n = training['anchor']['updates']; sub = logs[:n]
            row = dict(model=name, tag=point['tag'], main_s=training['main_seconds'],
                       updates=n, pairs=training['main_pairs'], cooldown_s=training['cooldown_seconds'],
                       cooldown_updates=training['completed_updates'],
                       AMP_skips=sum(x['AMP_skip'] for x in sub),
                       target_tokens=sum(x['tokens'] for x in sub),
                       successful_target_tokens=sum(x['tokens'] for x in sub if not x['AMP_skip']),
                       nonfinite_skips=sum(x['nonfinite_skip'] for x in sub),
                       CE={}, scores={}, generation_statuses={})
            for split in reference:
                rows = read(p/'evaluation'/(split+'-per-example.json'))
                verify_rows(rows, reference[split])
                assert scores(rows)['matches'] == metrics['splits'][split]['exec']['matches']
                row['scores'][split] = scores(rows)
                row['CE'][split] = metrics['splits'][split]['CE']['CE']
                row['generation_statuses'][split] = metrics['splits'][split]['exec']['generation_statuses']
                records[(name, point['tag'], split)] = rows
                if split == 'heldout4k': row['bins'] = bins(rows)
            points.append(row)

    # Original C9000 has generations/executions, rather than per-example rows.
    oldroot = RUNS/'nlp-hw02-kaggle-l2/1/l1-reference-live-v7'
    c9000 = {}
    for split in reference:
        gens = read(oldroot/f'model-C__points__t009000__evaluation__{split}-generations.json')
        executions = read(oldroot/f'model-C__points__t009000__evaluation__{split}-execution.json')['rows']
        rows = []
        for ref, gen, execution in zip(reference[split], gens, executions):
            assert hashlib.sha256(gen['question'].encode()).hexdigest() == ref['question_sha256']
            rows.append(dict(ref, execution=execution, correct=correct(ref['reference'], execution)))
        assert len(gens) == len(executions) == len(reference[split])
        c9000[split] = rows

    # B1 W9000 is original; W28370 is recovered from saved EV executions.
    b1points = []
    b1raw = RUNS/'nlp-hw02-kaggle-b1/1/model-GPU1'
    for tag, evtime in [('probe-009000', None), ('probe-028370', 28370), ('end-main-034200', 34200)]:
        training = read(b1raw/'points'/tag/'training.json')
        a = training['anchor']; evalroot = (b1raw/'points'/tag/'evaluation' if evtime is None
            else RUNS/f'nlp-hw02-b1-ev/1/model-GPU1/eval-{evtime}')
        metrics = read(evalroot/'metrics.json') if (evalroot/'metrics.json').exists() else None
        point = dict(model='W-B1', tag=tag, main_s=a['main_seconds'], updates=a['updates'], pairs=a['cursor'],
                     scores={}, CE={}, metrics_complete=metrics is not None)
        for split in reference:
            perp = evalroot/(split+'-per-example.json')
            if perp.exists(): rows = read(perp); verify_rows(rows, reference[split])
            elif (evalroot/(split+'-execution.json')).exists():
                gens = read(evalroot/(split+'-generations.json'))
                ex = read(evalroot/(split+'-execution.json'))['rows']
                assert len(gens) == len(ex) == len(reference[split])
                rows = []
                for ref, gen, execution in zip(reference[split], gens, ex):
                    assert hashlib.sha256(gen['question'].encode()).hexdigest() == ref['question_sha256']
                    rows.append(dict(ref, execution=execution, correct=correct(ref['reference'], execution)))
            else: continue
            point['scores'][split] = scores(rows)
            point['CE'][split] = metrics['splits'][split]['CE']['CE'] if metrics else None
            if split == 'heldout4k': point['bins'] = bins(rows)
            records[('W-B1', tag, split)] = rows
        b1points.append(point)

    cs = read(WORK/'results/l2-results.json')['points']
    cpoints = [p for p in cs if p['model'] in ['C (shared L1)', 'C-main']]
    finalc = [p for p in read(WORK/'results/l3-final-results.json')['points'] if p['tag'] == 'l2-62280']
    comparisons = {}
    for p in points:
        key = (p['model'], p['tag'], 'heldout4k')
        comparisons[p['model']+'/'+p['tag']+' minus C9000'] = paired(
            [r['correct'] for r in c9000['heldout4k']], [r['correct'] for r in records[key]])
    for tag0, tag1, label in [('probe-019117', 'probe-020091', 'equal_3.1M'),
                              ('end-main-027000', 'end-main-027000', 'equal_27000s')]:
        comparisons['lr1000 minus lr600 '+label] = paired(
            [r['correct'] for r in records[('W-lr600',tag1,'heldout4k')]],
            [r['correct'] for r in records[('W-lr1000',tag0,'heldout4k')]])

    b1state = read(b1raw/'summary.json')['state']
    b1stability = dict(AMP_skips=b1state['AMP_skips'],
        nonfinite_skips_surviving_state=b1state['nonfinite_skips'],
        nonfinite_total_history=b1state['stability']['total'],
        escalations=b1state['stability']['escalations'], lr_factor=b1state['lr_factor'])
    latest = {f'GPU{i}':read(raw/f'model-GPU{i}/latest.json') for i in [0,1]}
    for i in [0,1]:
        s = read(raw/f'model-GPU{i}/summary.json')['state']
        assert latest[f'GPU{i}']['updates'] == s['updates']
        assert latest[f'GPU{i}']['cursor'] == s['cursor']
        assert latest[f'GPU{i}']['main_seconds'] == s['main_seconds']
    efail = read(RUNS/'nlp-hw02-b1-ev/1/model-GPU0/ev-results.json')
    eexec = read(RUNS/'nlp-hw02-b1-ev/1/model-GPU0/eval-28370/heldout4k-execution.json')
    estatus = {}
    for row in eexec['rows']: estatus[row['status']] = estatus.get(row['status'],0)+1
    forecasts = {}
    for name in ['W-lr1000', 'W-lr600']:
        p, q = [x for x in points if x['model'] == name]
        a, b = [x['scores']['heldout4k']['accuracy_pp'] for x in [p, q]]
        slope = (b-a)/math.log(q['main_s']/p['main_s'])
        floor_alpha = math.log((90-a)/(90-b))/math.log(q['main_s']/p['main_s'])
        cross = []
        for target in [71., 71.3623046875]:
            for bonus in [0., 1., 2.]:
                hours_log = q['main_s']/3600*math.exp((target-b-bonus)/slope)
                hours_power = q['main_s']/3600*((90-b)/(90-target+bonus))**(1/floor_alpha)
                cross.append(dict(target_pp=target, cooldown_bonus_pp=bonus,
                    total_main_hours_log=hours_log, additional_main_hours_log=hours_log-q['main_s']/3600,
                    total_main_hours_floor90=hours_power))
        forecasts[name] = dict(slope_pp_per_log_time=slope, slope_pp_per_log_pairs=(b-a)/math.log(q['pairs']/p['pairs']),
            floor90_alpha=floor_alpha, crossings=cross,
            after_additional_sessions=[dict(sessions=n, total_main_h=q['main_s']/3600+9.5*n,
                short_log_pp=b+slope*math.log(1+9.5*n*3600/q['main_s']),
                half_slope_pp=b+.5*slope*math.log(1+9.5*n*3600/q['main_s']),
                short_floor90_pp=90-(90-b)*(1+9.5*n*3600/q['main_s'])**(-floor_alpha)) for n in [1,2,3]])

    architecture = read(WORK/'results/architecture-decision.json')
    hybrid = next(p for p in architecture['forecasts'] if p['shape']=='e6-d6-w768')
    result = dict(status='complete CPU audit', session=session, points=points, B1_W=b1points,
        B1_W_stability=b1stability, E_invalid=dict(final=efail, prepared28370_statuses=estatus), latest=latest,
        C=cpoints, C_final_50min=finalc, stability=stability, comparisons=comparisons, forecasts=forecasts,
        hybrid_timing_proxy=dict(sample_ratio=hybrid['decoder_cost_ratio'],
            greedy_ratio=hybrid['greedy_cost_ratio'], grid=hybrid['hybrid_grid'],
            warning='Historical C analog stop rates and non-QK shape benchmark; no W-QK cold timer'),
        provenance=dict(producer_manifest_verified_files=checked, absent_manifest_files=missing,
                        local_source_SHA256=INPUTS),
        limitations=['No GPU, forwards, reexecution or sampling',
                     'Only two B2 points; scenarios are not prediction intervals',
                     'Pairs are not exact token matching: W long2x, original C ordinary order',
                     'No B1 W28370 CE/old991 in downloaded EV; not inferable from generated programs',
                     'Ordinary AMP skips are not NaN escalations; target_tokens include skipped batches'])
    (WORK/'results/b2-results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['points','stability','comparisons','forecasts']}, ensure_ascii=False, indent=2))
    plot(points, b1points, cpoints, finalc, forecasts)


def plot(points, b1, cs, finalc, forecasts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    series = [('C-main', cs[:3], '#207044'), ('W B1', b1, '#888888')]
    for name, color in [('W-lr1000','#1f66b4'), ('W-lr600','#bd5a16')]:
        series.append((name, [p for p in points if p['model']==name], color))
    for name, rows, color in series:
        ys = [p['scores']['heldout4k']['accuracy_pp'] for p in rows]
        ps = [p.get('pairs',p.get('updates',0)*128)/1e6 for p in rows]
        ts = [p['main_s']/3600 for p in rows]
        axes[0].plot(ps,ys,'o-',label=name,color=color)
        axes[1].plot(ts,ys,'o-',label=name,color=color)
    q = next(p for p in points if p['model']=='W-lr1000' and p['tag'].startswith('end'))
    slope = forecasts['W-lr1000']['slope_pp_per_log_time']
    ts = [7.5+i*.2 for i in range(126)]
    ys = [q['scores']['heldout4k']['accuracy_pp']+slope*math.log(t/7.5) for t in ts]
    axes[1].plot(ts,ys,'--',color='#1f66b4',label='W .001: log, cooldown 15 мин')
    for p in finalc:
        axes[1].scatter([p['main_s']/3600],[p['scores']['heldout4k']['accuracy_pp']],
                        marker='*',s=120,color='#207044')
    axes[1].axhspan(70.751953125,71.3623046875,color='#207044',alpha=.12,label='C: финалы, cooldown 50 мин')
    for ax, xlabel in zip(axes,['Предъявленные main-пары, млн','Чистое main-время, часы']):
        ax.set(xlabel=xlabel,ylabel='Heldout4096 greedy, %',ylim=(48,75))
        ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower right')
    axes[0].set_title('Измерения после cooldown ≈15 мин')
    axes[1].set_title('Измерения и условный прогноз W')
    fig.tight_layout();fig.savefig(WORK/'results/b2-learning-curve.png',dpi=170);plt.close(fig)


if __name__ == '__main__': main()
