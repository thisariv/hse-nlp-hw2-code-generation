"""CPU audit and prefix-pool replay of B2+B3; no API, forwards or execution.

Run with NumPy/Matplotlib. Raw downloaded journals remain ignored. Config search
uses the same tune set as its reported pointwise CI, so is exploratory.
"""
import hashlib
import csv
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'nlp-b3-matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = Path(__file__).resolve().parents[1]
RUNS = WORK/'results/kaggle-runs'
RAW = RUNS/'nlp-hw02-b3/1'
sys.path[:0] = [str(WORK/'kaggle'), str(WORK/'kaggle/src')]
from final_v7_analysis import load_all, timing_estimate, paired, independent_select
from final_ensemble import replay_record
from final_adaptive import numeric, same
from final_choose import choose
from final_split import fixed_split

INPUTS = {}
IDS = fixed_split()['tune2048']
NAMES = ['C-main', 'C-long2x', 'pilot-b', 'W-lr1000', 'W-lr600']


def read(p):
    data = p.read_bytes()
    INPUTS[str(p.relative_to(WORK))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def save_result(result):
    """Keep the complete search as a small table rather than repetitive JSON."""
    e = result['ensemble']
    grid = e.pop('grid')
    e['grid_configurations'] = len(grid)
    e['grid_eligible_140'] = sum(r['timing']['conservative_s'] <= 140 for r in grid)
    eligible = [r for r in grid if r['timing']['conservative_s'] <= 140 and
                r['mean_C_sample_units'] <= e['baseline_v7']['mean_samples']]
    e['best_equal_mean_cost_140'] = max(eligible, key=lambda r:(r['matches'],-r['timing']['conservative_s']))
    e['family_best_140'] = []
    for names in sorted(set(tuple(r['models']) for r in grid)):
        rows = [r for r in grid if tuple(r['models']) == names and r['timing']['conservative_s'] <= 140]
        e['family_best_140'].append(max(rows,key=lambda r:(r['matches'],-r['timing']['conservative_s'])))
    p = WORK/'results/b3-ensemble-grid.csv'
    with p.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['models','anchor','K1','K2','matches','stopped','mean_samples','mean_C_sample_units',
            'delta_v7_pp','CI95_lo_pp','CI95_hi_pp','gained','lost','conservative_s','stress15_s'])
        for r in grid:
            c = r['paired_vs_v7']
            writer.writerow(['+'.join(r['models']),r['anchor'],'/'.join(map(str,r['K1'])),'/'.join(map(str,r['K2'])),
                r['matches'],r['stopped'],r['mean_samples'],r['mean_C_sample_units'],c['delta_pp'],*c['CI95_pp'],
                c['gained'],c['lost'],r['timing']['conservative_s'],r['timing']['stress15_s']])
    e['grid_csv_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    result['inputs_sha256'].update(INPUTS)
    write(WORK/'results/b3-results.json',result)


def correct(ref, ex):
    return numeric({'execution': ref}) and numeric({'execution': ex}) and same(ref['value'], ex['value'])


def scores(rows):
    return dict(n=len(rows), matches=sum(r['correct'] for r in rows),
                accuracy_pp=100*sum(r['correct'] for r in rows)/len(rows))


def verified_rows(p, refs):
    rows = read(p)
    assert len(rows) == len(refs)
    for i, (r, ref) in enumerate(zip(rows, refs)):
        assert r['id'] == i
        for k in ['question_sha256', 'reference', 'original_bpe_length']:
            assert r[k] == ref[k], (p, i, k)
        assert r['correct'] == correct(r['reference'], r['execution'])
    return rows


def audit():
    session = read(RAW/'session-summary.json')
    assert session['status'] == 'complete' and session['exit_codes'] == [0, 0]
    assert 'COMPLETE' in (RAW/'kernel-status.txt').read_text()
    manifest = read(RAW/'b1-output-manifest.json')
    checked, missing = [], []
    for item in manifest['files']:
        p = RAW/item['file']
        if not p.exists():
            missing.append(item['file']); continue
        data = p.read_bytes()
        assert len(data) == item['bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'], item['file']
        checked.append(item['file'])
        INPUTS[str(p.relative_to(WORK))] = item['sha256']
    b2 = read(WORK/'results/b2-results.json')
    c = read(WORK/'results/l3-final-results.json')['points']
    base = RUNS/'nlp-hw02-kaggle-l2/1/model-C-main/points/l2-10800/evaluation'
    refs = {s: read(base/f'{s}-per-example.json') for s in ['heldout4k', 'old991']}
    records, points, stability, finals = {}, [], {}, {}
    for g, name in enumerate(NAMES[-2:]):
        root = RAW/f'model-GPU{g}'
        summary = read(root/'summary.json'); state = summary['state']
        origin = read(root/'b3-origin.json')['anchor']
        latest = read(root/'latest.json')
        assert summary['final_complete'] and summary['main_target_reached'] and not summary['missed_points']
        assert latest == summary['B4_uncooled_latest']
        for k in ['updates', 'cursor', 'main_seconds']:
            assert latest[k] == state[k]
        assert state['lr_factor'] == 1 and not state['stability']['history']
        assert state['nonfinite_skips'] == state['stability']['escalations'] == 0
        oldstate = read(RUNS/f'nlp-hw02-b2/1/model-GPU{g}/summary.json')['state']
        assert origin['updates'] == oldstate['updates'] and origin['cursor'] == oldstate['cursor']
        stability[name] = dict(AMP_cumulative=state['AMP_skips'], AMP_B3=state['AMP_skips']-oldstate['AMP_skips'],
            attempts_cumulative=state['updates'], attempts_B3=state['updates']-origin['updates'],
            nonfinite_skips=state['nonfinite_skips'], escalations=state['stability']['escalations'],
            lr_factor=state['lr_factor'], last_lr=state['last_lr'], micro=state['micro_batch'], latest=latest,
            B3_pairs=state['cursor']-origin['cursor'], B3_main_s=state['main_seconds']-origin['main_seconds'])
        log = (RAW/f'gpu{g}-process.log').read_text()
        ticks = re.findall(r'B3 '+name+r' (\d+) ([\d.]+) pairs (\d+) LR ([\d.]+)', log)
        assert ticks and all(int(u)*128 == int(p) and float(lr) == state['peak_lr'] for u,t,p,lr in ticks)
        assert all(int(a[0]) < int(b[0]) and float(a[1]) < float(b[1]) for a,b in zip(ticks,ticks[1:]))
        stability[name]['periodic_log_ticks'] = len(ticks)
        for point in state['points']:
            tag = point['tag']
            p = root/'points'/tag if tag.startswith('b3-') else RUNS/f'nlp-hw02-b2/1/model-GPU{g}/points'/tag
            t = read(p/'training.json'); m = read(p/'evaluation/metrics.json')
            complete = read(p/'complete.json'); ck = read(p/'checkpoint.json')
            assert complete['training'] == t == point['training'] and complete['metrics'] == m == point['metrics']
            assert m['weights_sha256'] == ck['sha256'] == complete['weights_sha256']
            a = t['anchor']
            assert t['main_pairs'] == a['cursor'] == 128*a['updates']
            row = dict(model=name, tag=tag, main_s=t['main_seconds'], pairs=t['main_pairs'],
                updates=a['updates'], cooldown_s=t['cooldown_seconds'], cooldown_updates=t['completed_updates'],
                scores={}, CE={}, weights_sha256=ck['sha256'])
            for split in refs:
                rows = verified_rows(p/'evaluation'/f'{split}-per-example.json', refs[split])
                assert scores(rows)['matches'] == m['splits'][split]['exec']['matches']
                row['scores'][split] = scores(rows); row['CE'][split] = m['splits'][split]['CE']['CE']
                records[(name, tag, split)] = rows
                if split == 'heldout4k':
                    row['bins'] = [scores([r for r in rows if lo <= r['original_bpe_length'] < hi])
                        for lo,hi in [(0,60),(60,90),(90,120),(120,math.inf)]]
                    assert [x['n'] for x in row['bins']] == [2013,1457,488,138]
                    row['scores']['tune2048'] = scores([rows[i] for i in IDS])
                    row['scores']['eval2048'] = scores([r for r in rows if r['id'] not in set(IDS)])
            if not tag.startswith('b3-'):
                old = next(x for x in b2['points'] if x['model'] == name and x['tag'] == tag)
                row.update({k:old[k] for k in ['AMP_skips','target_tokens','nonfinite_skips']})
            else:
                # Per-update B3 files/cooldown journals are absent locally; do not infer intermediate AMP counts.
                row['AMP_skips'] = state['AMP_skips'] if t['final'] else None
                row['nonfinite_skips'] = 0
            if t.get('final'):
                row['test'] = m['test']; row['K8'] = m['K8_tune2048']; finals[name] = p
            points.append(row)
    # Exact saved controls on the same heldout set.
    controls = {}
    for name, folder, prefix in [
        ('C-main', RUNS/'nlp-hw02-kaggle-l3/final', 'model-C-main__points__l2-62280'),
        ('C-long2x', RUNS/'nlp-hw02-kaggle-l3/final', 'model-C-long2x__points__l2-62280'),
        ('pilot-b', RUNS/'nlp-hw02-kaggle-p1/final', 'model-pilot-b__points__long-truncation')]:
        controls[name] = {s:verified_rows(folder/f'{prefix}__evaluation__{s}-per-example.json', refs[s]) for s in refs}
    comparisons = {}
    for name in NAMES[-2:]:
        tag = finals[name].name
        for control in controls:
            for split in refs:
                comparisons[f'{name} minus {control}/{split}'] = paired(
                    [r['correct'] for r in records[(name,tag,split)]], [r['correct'] for r in controls[control][split]])
        prev = next(x for x in points if x['model'] == name and x['tag'].startswith('b3-probe-0486'))
        short = next(x for x in points if x['model'] == name and x['tag'].startswith('b3-probe-0378'))
        comparisons[name+' short +3h'] = paired([r['correct'] for r in records[(name,prev['tag'],'heldout4k')]],
            [r['correct'] for r in records[(name,short['tag'],'heldout4k')]])
    comparisons['W-lr600 minus W-lr1000 final'] = paired(
        [r['correct'] for r in records[('W-lr600',finals['W-lr600'].name,'heldout4k')]],
        [r['correct'] for r in records[('W-lr1000',finals['W-lr1000'].name,'heldout4k')]])
    return dict(session=session, points=points, C=c, stability=stability, comparisons=comparisons,
        producer_checked=checked, producer_missing=missing), finals


class Replay:
    """Cached pairwise isclose graph; connected components rebuilt for each subset.

    Never cluster the full K8 pool then slice: omitted bridge candidates can
    split a component. Candidate order, multiplicity and code tie rules persist.
    """
    def __init__(self, journals, anchor, names):
        self.names = names; self.rows = []
        for i in IDS:
            g = journals[anchor][i]['greedy']
            candidates = [g]+[c for name in names for c in journals[name][i]['samples']['0.7']]
            valid = [j for j,c in enumerate(candidates) if numeric(c) and
                     (j == 0 or c.get('mean_logprob') is not None and math.isfinite(c['mean_logprob']))]
            adjacency = [sum(1 << k for k in valid if same(candidates[j]['execution']['value'], candidates[k]['execution']['value']))
                         if j in valid else 0 for j in range(len(candidates))]
            self.rows.append((candidates, adjacency, sum(1 << j for j in valid), journals[anchor][i]['reference']))

    def mask(self, ks, greedy=True):
        return int(greedy)+sum(((1 << k)-1) << (1+8*j) for j,k in enumerate(ks))

    @staticmethod
    def vote(row, mask, preserve_greedy=True):
        candidates, adjacency, valid, _ = row
        remaining = mask & valid
        if not remaining: return candidates[0]
        best, bestkey = 0, None
        while remaining:
            seed = remaining & -remaining; component = seed; frontier = seed
            while frontier:
                bit = frontier & -frontier; frontier ^= bit
                new = adjacency[bit.bit_length()-1] & remaining & ~component
                component |= new; frontier |= new
            remaining &= ~component
            indices = [j for j in range(len(candidates)) if component >> j & 1]
            key = (len(indices), bool(component & adjacency[0]),
                   sum(candidates[j]['mean_logprob'] for j in indices)/len(indices))
            if bestkey is None or key > bestkey: best, bestkey = component, key
        if preserve_greedy and best & 1: return candidates[0]
        return max((candidates[j] for j in range(len(candidates)) if best >> j & 1), key=lambda c:c['mean_logprob'])

    def evaluate(self, ks, k1=None, greedy=True):
        mask = self.mask(ks, greedy); first = self.mask(k1) if k1 else None
        bits, stops, winners = [], [], []
        for row in self.rows:
            candidates, adjacency, valid, ref = row
            stop = first is not None and first & valid == first and first & adjacency[0] == first
            winner = self.vote(row, first if stop else mask, preserve_greedy=greedy)
            bits.append(correct(ref, winner['execution'])); stops.append(bool(stop)); winners.append(winner)
        return bits, sum(stops), winners

    def validate(self, ks, k1=None):
        bits, stopped, winners = self.evaluate(ks,k1)
        specs = [dict(name=n,K1=a,K2=b) for n,a,b in zip(self.names,k1 or ks,ks)]
        stop_sum = 0
        for row, win in zip(self.rows,winners):
            candidates = row[0]; g = candidates[0]
            samples = {n:candidates[1+8*j:1+8*(j+1)] for j,n in enumerate(self.names)}
            actual, stop, count = replay_record(g,samples,specs)
            assert (win['code'],win['execution']) == (actual['code'],actual['execution'])
            chosen = [c for n,k in zip(self.names,k1 if k1 and stop else ks) for c in samples[n][:k]]
            independent = independent_select(g, chosen)
            assert (win['code'],win['execution']) == (independent['code'],independent['execution'])
            stop_sum += stop
        assert k1 is None or stop_sum == stopped
        return len(bits)


def graph_fixtures():
    # A~B~C but A!~C. Removing bridge B must split the graph; two C votes win.
    candidates = [dict(code=str(j),execution=dict(status='numeric',value=v),mean_logprob=-float(j))
                  for j,v in enumerate([100.,100.00009,100.00018,100.00018])]
    adjacency = [sum(1 << k for k,d in enumerate(candidates) if same(c['execution']['value'],d['execution']['value'])) for c in candidates]
    row = (candidates,adjacency,15,dict(status='numeric',value=100.))
    assert Replay.vote(row,15) is candidates[0]
    assert Replay.vote(row,13) is candidates[2]
    # Duplicate votes remain distinct after the bridge is removed.
    assert len([candidates[j] for j in range(4) if 13 >> j & 1]) == 3
    return ['bridge removal splits connected components','duplicate samples preserve multiplicity','greedy code retained in winning component']


def timing(names, anchor, ks, stopped, ratios):
    specs = [dict(name=n,K1=1,K2=k) for n,k in zip(names,ks)]
    base = timing_estimate(specs,stopped)
    runs = {}
    for label, t in base['runs'].items():
        old = read(RUNS/f'nlp-hw02-kaggle-l3/final/inference-v5/final-inference-l3-v5/{label}/timing.json')
        p = old['phases_seconds']
        mean = sum(p[f'stage1_{n}_sampling_restore_submit'] for n in ['main','long'])/sum(b['stage']==1 for b in old['generation_batches'])/base['sample_speedup']
        extra = [k-1 for k in ks]; wave = 1024//max(extra) if max(extra) else 0
        first = math.ceil(t['rows']/1024)*sum(ratios[n]['sample'] for n in names)*mean
        second = math.ceil(t['escalated_rows_proxy']/wave)*sum(ratios[n]['sample'] for n,k in zip(names,extra) if k) if wave else 0
        greedy = p['greedy_generation_restore_submit']/1.6565*ratios[anchor]['greedy']
        fixed = t['fixed_seconds']+t['extra_model_load_warmup_reserve_seconds']+2+4*sum(n.startswith('W-') for n in names)
        nominal = fixed+greedy+first+second*mean
        variable = greedy+first+second*base['worst_old_batch_seconds']*1.10/base['sample_speedup']
        runs[label] = dict(nominal_s=nominal,conservative_s=fixed+variable,stress15_s=fixed+1.15*variable,
            escalated_rows=t['escalated_rows_proxy'],stage2_wave_questions=wave)
    return dict(runs=runs, conservative_s=max(t['conservative_s'] for t in runs.values()),
                stress15_s=max(t['stress15_s'] for t in runs.values()))


def ensembles(finals):
    journals, receipts = load_all()  # Includes producer-manifest verification of all 192 C journals.
    for name,p in finals.items():
        files = sorted((p/'evaluation/k8-tune2048').glob('batch-*.json')); assert len(files)==64
        rows = [r for f in files for r in read(f)]
        assert len(rows) == 2048 and sorted(r['id'] for r in rows) == IDS
        journals[name] = {r['id']:r for r in rows}
    for i in IDS:
        for name in NAMES:
            r = journals[name][i]; base = journals['C-main'][i]
            assert len(r['samples']['0.7']) == 8 and r['question_sha256'] == base['question_sha256'] and r['reference'] == base['reference']
            win = choose(r, dict(mode='vote',temperature=.7,k=8,greedy_vote=False))
            assert correct(r['reference'],win['execution']) == r['correct']
    bench = read(WORK/'results/architecture-shape-bench.json')['rows']
    cb = next(r for r in bench if r['shape']=='e6-d6-w512')
    wb = next(r for r in bench if r['shape']=='e6-d6-w768')
    ratios = {n:dict(sample=1.,greedy=1.) for n in NAMES}
    ratios.update({n:{mode:cb['generation'][mode]['seq_per_s']/wb['generation'][mode]['seq_per_s']
                      for mode in ['sample','greedy']} for n in NAMES[-2:]})
    base = Replay(journals,'C-main',NAMES[:3]); basebits, stopped, _ = base.evaluate([3,5,4],[1,1,1])
    assert sum(basebits)==1535 and stopped==1232
    checked = base.validate([3,5,4],[1,1,1])
    static, grid, curves = [], [], {}
    def summary(bits):
        return dict(matches=sum(bits),accuracy_pp=100*sum(bits)/2048,paired_vs_v7=paired(bits,basebits))
    for name in NAMES:
        sim = Replay(journals,name,[name]); curve=[]
        for k in range(9):
            bits,_,_=sim.evaluate([k]); curve.append(dict(k=k,**summary(bits)))
        curves[name]=curve
        for greedy in [False,True]:
            bits,_,_=sim.evaluate([8],greedy=greedy)
            static.append(dict(models=[name],anchor=name,K=[8],greedy_vote=greedy,
                C_sample_units=8*ratios[name]['sample'],**summary(bits)))
            if name.startswith('W-') and not greedy: assert sum(bits)==read(finals[name]/'evaluation/k8-tune2048/summary.json')['matches']
    # Explicit exploratory grid: K1=1/model, each K2<=8, total<=12.
    # C-main anchors triples; pair pools additionally test the W greedy anchor.
    for w in NAMES[-2:]:
        combinations = [(['C-main',w],'C-main'), ([w,'C-main'],w), ([w,'C-long2x'],w), ([w,'pilot-b'],w),
                        (['C-main','C-long2x',w],'C-main'), (['C-main','pilot-b',w],'C-main')]
        for names,anchor in combinations:
            sim = Replay(journals,anchor,names)
            for ks in itertools.product(*[range(1,9) for n in names]):
                if sum(ks)>12: continue
                bits,stop,_=sim.evaluate(ks,[1]*len(names))
                probability=(2048-stop)/2048
                means=[1+probability*(k-1) for k in ks]
                row=dict(models=names,anchor=anchor,K1=[1]*len(names),K2=list(ks),stopped=stop,
                    mean_samples=sum(means),mean_C_sample_units=sum(m*ratios[n]['sample'] for m,n in zip(means,names)),
                    **summary(bits),timing=timing(names,anchor,ks,stop,ratios))
                grid.append(row)
                if sum(ks) in [7,8,12] and max(ks)-min(ks)<=1:
                    bits0,_,_=sim.evaluate(ks)
                    static.append(dict(models=names,anchor=anchor,K=list(ks),greedy_vote=True,
                        C_sample_units=sum(k*ratios[n]['sample'] for k,n in zip(ks,names)),**summary(bits0)))
            print('replayed',names,'configs',len(grid),flush=True)
    eligible=[r for r in grid if r['timing']['conservative_s']<=140]
    best=max(eligible,key=lambda r:(r['matches'],-r['timing']['conservative_s'],-r['mean_C_sample_units']))
    stress=[r for r in grid if r['timing']['stress15_s']<=140]
    robust=max(stress,key=lambda r:(r['matches'],-r['timing']['stress15_s'],-r['mean_C_sample_units']))
    for r in [best,robust]: checked+=Replay(journals,r['anchor'],r['models']).validate(r['K2'],r['K1'])
    # C-only baselines and static budget references share the same selector.
    for names,ks in [(['C-main','C-long2x'],[4,4]),(['C-main','pilot-b'],[4,4]),
                      (NAMES[:3],[3,5,4]),(NAMES[:3],[4,4,4])]:
        sim=Replay(journals,'C-main',names); bits,_,_=sim.evaluate(ks)
        static.append(dict(models=names,anchor='C-main',K=ks,greedy_vote=True,C_sample_units=sum(ks),**summary(bits)))
    return dict(baseline_v7=dict(matches=1535,stopped=1232,mean_samples=6.5859375),
        ratios=ratios,curves=curves,static=static,grid=grid,best_140=best,best_stress140=robust,
        journal_sha256=receipts,frozen_and_independent_parity_rows=checked,
        selection='exploratory tune maximum; K1=1, each K2 1..8, total<=12; tie faster, then cheaper',
        timer_limits='v7 rounded waves, fixed CPU overhead retained; W sample1.35/greedy1.56 from no-QK shape bench; +4s W load reserve; QK, mixed memory, executor tails unmeasured')


def comparisons_and_plot(result):
    c=result['C']; points=result['points']
    # C-long resets ORDER cursor at L2, while updates retain the shared L1.
    # Total attempted pairs include that inherited training: 128*updates.
    cpoints=[]
    for p in c:
        q=dict(p)
        if p['tag']=='t009000': q['pairs']=p['updates']*128
        else:
            if p['tag']=='l2-45540': folder=RUNS/'nlp-hw02-kaggle-l3/mid'
            elif p['tag']=='l2-62280': folder=RUNS/'nlp-hw02-kaggle-l3/final'
            else: folder=None
            if folder:
                t=read(folder/f"model-{p['model']}__points__{p['tag']}__training.json")
            else: t=read(RUNS/f"nlp-hw02-kaggle-l2/1/model-{p['model']}/points/{p['tag']}/training.json")
            q['order_cursor']=t['anchor_cursor']
            q['pairs']=t['anchor_updates']*128
            q['cursor_origin']=q['order_cursor']-q['pairs']
            assert q['cursor_origin'] == (0 if p['model']=='C-main' else -6035840)
        cpoints.append(q)
    interpolations=[]
    for p in points:
        for cname in ['C-main','C-long2x']:
            control=[x for x in cpoints if x['model'] in ['C (shared L1)',cname] and (x['cooldown_s']<1000)==(p['cooldown_s']<1000)]
            for axis in ['pairs','main_s']:
                control=sorted(control,key=lambda q:q[axis]); target=p[axis]
                for a,b in zip(control,control[1:]):
                    if a[axis]<=target<=b[axis]:
                        av,bv=[q['scores']['heldout4k']['accuracy_pp'] for q in [a,b]]
                        estimate=av+(bv-av)*math.log(target/a[axis])/math.log(b[axis]/a[axis])
                        interpolations.append(dict(model=p['model'],tag=p['tag'],axis=axis,control=cname,
                            C_interpolated_pp=estimate,W_minus_C_pp=p['scores']['heldout4k']['accuracy_pp']-estimate,
                            bracket=[a['tag'],b['tag']]))
                        break
    result['C']=cpoints;result['matched_interpolations']=interpolations
    fits={}
    for name in NAMES[-2:]:
        short=[p for p in points if p['model']==name and p['cooldown_s']<1000]
        a,b=short[-2:]; slope=(b['scores']['heldout4k']['accuracy_pp']-a['scores']['heldout4k']['accuracy_pp'])/math.log(b['main_s']/a['main_s'])
        final=next(p for p in points if p['model']==name and p['cooldown_s']>1000)
        # Two latest SHORT points set a scenario increment, anchored at actual LONG final.
        fits[name]=dict(last_short_log_slope=slope,additional_9h_scenario_pp=final['scores']['heldout4k']['accuracy_pp']+slope*math.log((final['main_s']+32400)/final['main_s']),
            warning='not a prediction interval; W-lr1000 last slope negative; plateau possible')
    result['continuation_scenarios']=fits
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    colors={'W-lr1000':'#d05b35','W-lr600':'#7b48ab','C-main':'#2970aa','C-long2x':'#238763'}
    for ax,axis,div,label in zip(axes,['pairs','main_s'],[1e6,3600],['Main pairs, million','Main time, hours']):
        for name,color in colors.items():
            series=[p for p in (points if name.startswith('W-') else cpoints) if p['model'] in [name]+(['C (shared L1)'] if name.startswith('C-') else [])]
            short=sorted([p for p in series if p['cooldown_s']<1000],key=lambda p:p[axis])
            long=[p for p in series if p['cooldown_s']>1000]
            ax.plot([p[axis]/div for p in short],[p['scores']['heldout4k']['accuracy_pp'] for p in short],'-o',color=color,label=name+' / 15m')
            ax.scatter([p[axis]/div for p in long],[p['scores']['heldout4k']['accuracy_pp'] for p in long],marker='D',s=60,color=color,label=name+' / 50m')
        ax.axhline(72.0703125,color='#666',linestyle=':',label='P1-B 72.07 / long cooldown')
        ax.set(xlabel=label,ylabel='Greedy heldout4096, %');ax.grid(alpha=.2)
    axes[1].legend(fontsize=7,loc='lower right');fig.savefig(WORK/'results/b3-learning-curve.png',dpi=170);plt.close(fig)


def main():
    result,finals=audit();print('audited',len(result['producer_checked']),'producer files',flush=True)
    comparisons_and_plot(result)
    result['ensemble']=ensembles(finals)
    result['graph_fixtures']=graph_fixtures()
    result.update(status='complete',GPU=False,new_execution=False,new_sampling=False,
        limits=['B3 per-update main and cooldown journals/PT are not downloaded; intermediate AMP counts and tensor finiteness not audited',
                'Equal pairs do not equal tokens/FLOPs; W uses long2x; interpolated controls are calculations, not measurements',
                '15m and50m cooldowns kept separate; no W final15m measurement, so cooldown effect unidentified',
                'All CIs pointwise descriptive on reused heldout/tune, no multiple-search correction or fresh confirmation'])
    result['inputs_sha256']=INPUTS
    save_result(result)
    print(json.dumps({k:result['ensemble'][k] for k in ['best_140','best_stress140']},indent=2))


if __name__=='__main__': main()
