"""CPU-only B4 audit, matched-budget curves and saved K8 prefix replay.

Uses the verified B3 selector and original production/independent oracles.
No model forwards, new execution, sampling, training, or API calls.
"""
import csv
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('b3_analysis', HERE/'analyze-b3-results.py')
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
import numpy as np
import matplotlib.pyplot as plt

WORK, RUNS = B.WORK, B.RUNS
RAW = RUNS/'nlp-hw02-b4/1'
IDS = B.IDS
INPUTS = {}
NAMES = ['C-main', 'C-long2x', 'pilot-b', 'W600', 'W600-trunc']


def read(path):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(WORK))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def lines(path):
    raw = path.read_bytes()
    INPUTS[str(path.relative_to(WORK))] = hashlib.sha256(raw).hexdigest()
    return [json.loads(line) for line in raw.splitlines()]


def paired(a, b):
    out = B.paired(a, b)
    rng = np.random.default_rng(2026101801)
    n, g, l = out['n'], out['gained'], out['lost']
    draws = rng.multinomial(n, [g/n, l/n, 1-(g+l)/n], size=20000)
    out['bootstrap_CI95_pp'] = np.quantile(100*(draws[:,0]-draws[:,1])/n, [.025,.975]).tolist()
    out['method'] = 'paired normal plus 20000 paired multinomial bootstrap draws; seed2026101801; pointwise descriptive'
    return out


def truncation(logs):
    stats = [x['truncation'] for x in logs]
    counts = np.sum([x['counts'] for x in stats], axis=0)
    excluded = np.sum([x['excluded'] for x in stats], axis=0)
    positions, lengths, sources, indices = [], [], [], []
    for x, log in zip(stats, logs):
        assert x['presented'] == 128 and x['retained'] == 115
        assert sum(x['counts']) == 128 and sum(x['excluded']) == 13
        assert x['selection_rng_restored'] and x['extra_selection_forward']
        assert x['forward_mode'] == 'two-pass' and x['CE_z_backward'] == 'standard B2 autograd'
        pos = x['excluded_positions']; length = x['excluded_target_lengths']
        assert len(pos) == len(set(pos)) == len(length) == len(x['excluded_indices']) == 13
        assert min(pos) >= 0 and max(pos) < 128
        assert np.bincount(np.searchsorted([53,79,105],length,side='right'),minlength=4).tolist() == x['excluded']
        assert x['retained_tokens'] == log['tokens']
        assert x['presented_tokens']-x['retained_tokens'] == sum(v+1 for v in length)
        # Independently reconstruct proportional largest-remainder quotas.
        capacity = np.maximum(0,np.array(x['counts'])-1)
        quota = np.minimum(np.floor(.1*np.array(x['counts'])).astype(int),capacity)
        while sum(quota)<13:
            candidates = [b for b in range(4) if quota[b]<capacity[b]]
            b = min(candidates,key=lambda b:(-(.1*x['counts'][b]-quota[b]),b));quota[b]+=1
        assert quota.tolist() == x['excluded']
        positions.extend(pos);lengths.extend(length);sources.extend(x['excluded_source_lengths']);indices.extend(x['excluded_indices'])
    presented = sum(x['presented'] for x in stats)
    pt = sum(x['presented_tokens'] for x in stats); rt = sum(x['retained_tokens'] for x in stats)
    def quantiles(values): return np.quantile(values,[0,.25,.5,.75,.9,.99,1]).tolist()
    return dict(updates=len(logs),presented=presented,retained=presented-len(indices),excluded=len(indices),
        excluded_fraction=len(indices)/presented,presented_tokens=pt,retained_tokens=rt,
        excluded_token_fraction=(pt-rt)/pt,counts_by_bin=counts.tolist(),excluded_by_bin=excluded.tolist(),
        excluded_fraction_by_bin=(excluded/counts).tolist(),unique_excluded_cache_ids=len(set(indices)),
        excluded_target_length_quantiles=quantiles(lengths),excluded_source_length_quantiles=quantiles(sources),
        excluded_mean_CE_step_average=float(np.mean([x['excluded_mean_CE'] for x in stats])),
        retained_mean_CE_step_average=float(np.mean([x['retained_mean_CE'] for x in stats])),
        note='presentations including weighted repeats; source denominator and retained per-example scores not logged; not a noise label')


def audit():
    session=read(RAW/'session-summary.json'); ab=read(RAW/'b4-ab.json')
    assert 'COMPLETE' in (RAW/'kernel-status.txt').read_text()
    assert session['status']=='complete' and session['exit_codes']==[0,0] and not session['unreaped_worker_pids']
    checked=[];missing=[];manifest=read(RAW/'b1-output-manifest.json')['files']
    for rec in manifest:
        path=RAW/rec['file']
        if not path.exists():missing.append(rec);continue
        raw=path.read_bytes()
        assert len(raw)==rec['bytes'] and hashlib.sha256(raw).hexdigest()==rec['sha256'],rec['file']
        checked.append(rec['file']);INPUTS[str(path.relative_to(WORK))]=rec['sha256']
    prior=read(WORK/'results/b3-results.json')
    base=RUNS/'nlp-hw02-kaggle-l2/1/model-C-main/points/l2-10800/evaluation'
    refs={s:read(base/f'{s}-per-example.json') for s in ['heldout4k','old991']}
    rows={};points=[];finals={};stability={};logs={};trunc={};fingerprints=[]
    def load_point(name,path,row=None):
        t=read(path/'training.json');m=read(path/'evaluation/metrics.json');c=read(path/'complete.json');ck=read(path/'checkpoint.json')
        assert c['training']==t and c['metrics']==m and c['weights_sha256']==m['weights_sha256']==ck['sha256']
        if row is None:
            a=t['anchor'];assert a['cursor']==128*a['updates']==t['main_pairs']
            row=dict(model=name,tag=path.name,main_s=t['main_seconds'],pairs=t['main_pairs'],updates=a['updates'],
                     cooldown_s=t['cooldown_seconds'],cooldown_updates=t['completed_updates'],q=t['q'],scores={},CE={},weights_sha256=ck['sha256'])
        for split in refs:
            rr=read(path/'evaluation'/f'{split}-per-example.json')
            assert len(rr)==len(refs[split])
            for i,(x,r) in enumerate(zip(rr,refs[split])):
                assert x['id']==i
                for key in ['question_sha256','reference','original_bpe_length']:assert x[key]==r[key]
                assert x['correct']==B.correct(x['reference'],x['execution'])
            assert sum(x['correct'] for x in rr)==m['splits'][split]['exec']['matches']
            rows[(name,path.name,split)]=rr
            row['scores'][split]=B.scores(rr);row['CE'][split]=m['splits'][split]['CE']['CE']
            if split=='heldout4k':
                row['bins']=[B.scores([x for x in rr if lo<=x['original_bpe_length']<hi]) for lo,hi in [(0,60),(60,90),(90,120),(120,math.inf)]]
                row['scores']['tune2048']=B.scores([rr[i] for i in IDS])
                row['scores']['eval2048']=B.scores([x for x in rr if x['id'] not in set(IDS)])
        points.append(row)
        return t,m,row
    for old in prior['points']:
        if old['model']!='W-lr600':continue
        folder='nlp-hw02-b3/1' if old['tag'].startswith('b3-') else 'nlp-hw02-b2/1'
        load_point('W600',RUNS/folder/'model-GPU1/points'/old['tag'],dict(old,model='W600'))
    for gpu,name in enumerate(NAMES[-2:]):
        root=RAW/f'model-GPU{gpu}';summary=read(root/'summary.json');state=summary['state'];origin=read(root/'b4-origin.json')['anchor']
        fingerprints.append(read(root/'start-fingerprints.json'))
        assert summary['status']=='complete' and summary['final_complete'] and summary['main_target_reached']
        assert not summary['budget_stopped_main'] and not summary['missed_points']
        assert origin['updates']==73416 and origin['cursor']==9397248
        assert origin['sha256']==prior['stability']['W-lr600']['latest']['sha256']
        assert origin['main_seconds']==prior['stability']['W-lr600']['latest']['main_seconds']
        latest=read(root/'latest.json');assert latest==summary['B5_uncooled_latest']
        for k in ['updates','cursor','main_seconds']:assert latest[k]==state[k]
        assert state['lr_factor']==1 and state['last_lr']==.0006 and not state['stability']['history']
        assert state['nonfinite_skips']==state['stability']['escalations']==0
        ll=lines(root/'updates.jsonl');logs[name]=ll
        assert len(ll)==state['updates']-origin['updates']
        elapsed=origin['main_seconds']
        for j,x in enumerate(ll,1):
            assert x['updates']==origin['updates']+j and x['main_pairs']==origin['cursor']+128*j
            assert x['lr']==.0006 and x['nonfinite_skip']==0 and not x['OOM_retries']
            assert x['AMP_skip'] or all(math.isfinite(x[k]) for k in ['CE','z_loss','gradient_norm'])
            elapsed+=x['seconds'];assert math.isclose(elapsed,x['main_seconds'],abs_tol=1e-7)
        amp=sum(x['AMP_skip'] for x in ll);assert amp==state['AMP_skips']-21
        stability[name]=dict(attempts_B4=len(ll),AMP_B4=amp,AMP_cumulative=state['AMP_skips'],nonfinite=0,OOM=0,
            main_s_B4=state['main_seconds']-origin['main_seconds'],seconds_per_update=sum(x['seconds'] for x in ll)/len(ll),
            pairs_B4=128*len(ll),latest=latest,cooldowns=[])
        for tag in summary['B4_points']:
            path=root/'points'/tag;t,m,row=load_point(name,path)
            cl=lines(path/'cooldown-updates.jsonl');elapsed=0.
            assert len(cl)==t['completed_updates']
            for j,x in enumerate(cl,1):
                assert x['branch_update']==j and math.isclose(x['lr'],.0006*(1-min(1,elapsed/t['cooldown_target_seconds'])),abs_tol=1e-15)
                assert x['nonfinite_skip']==0 and not x['OOM_retries']
                elapsed+=x['seconds'];assert math.isclose(elapsed,x['cooldown_seconds'],abs_tol=1e-7)
            assert elapsed==t['cooldown_seconds'] and cl[-1]['lr']==t['endpoint_lr']
            stability[name]['cooldowns'].append(dict(tag=tag,updates=len(cl),AMP=sum(x['AMP_skip'] for x in cl),
                seconds=elapsed,endpoint_lr=t['endpoint_lr']))
            if gpu:trunc[tag]=truncation(cl)
            if t['final']:
                k=m['K8_tune2048']
                assert k['complete'] and k['rows']==k['target_rows']==2048 and k['K']==8
                assert k['T']==.7 and k['seed']==270042 and not k['OOM_attempts']
                assert k['weights_sha256']==m['weights_sha256'] and m['test']['rows']==1919
                finals[name]=path;row['K8']=m['K8_tune2048'];row['test']=m['test']
        if gpu:
            trunc['main']=truncation(ll)
            for k in ['presented','excluded','presented_tokens','retained_tokens','excluded_by_bin']:
                assert trunc['main'][k]==summary['truncation_totals'][k]
    assert fingerprints[0]['fingerprints']==fingerprints[1]['fingerprints']
    assert fingerprints[0]['source_sha256']==fingerprints[1]['source_sha256']
    common=min(map(len,logs.values()));a,b=logs.values()
    assert common==ab['compared_prefix_updates'] and ab['passed'] and ab['same_start'] and ab['same_stream']
    assert all(x['stream_sha256']==y['stream_sha256'] for x,y in zip(a,b))
    ab['independent_common_prefix_time_ratio']=sum(x['seconds'] for x in b[:common])/sum(x['seconds'] for x in a[:common])
    ab['plain_common_prefix_main_s']=a[common-1]['main_seconds']
    controls={}
    for name,folder,prefix in [('C-main','nlp-hw02-kaggle-l3/final','model-C-main__points__l2-62280'),
        ('C-long2x','nlp-hw02-kaggle-l3/final','model-C-long2x__points__l2-62280'),
        ('pilot-b','nlp-hw02-kaggle-p1/final','model-pilot-b__points__long-truncation')]:
        controls[name]={}
        for split in refs:
            rr=read(RUNS/folder/f'{prefix}__evaluation__{split}-per-example.json')
            assert len(rr)==len(refs[split])
            for x,r in zip(rr,refs[split]):
                assert all(x[k]==r[k] for k in ['id','question_sha256','reference','original_bpe_length'])
                assert x['correct']==B.correct(x['reference'],x['execution'])
            controls[name][split]=rr
    comparisons={}
    for name in finals:
        for cname,cr in controls.items():
            for split in refs:
                comparisons[f'{name} minus {cname}/{split}']=paired([x['correct'] for x in rows[(name,finals[name].name,split)]],[x['correct'] for x in cr[split]])
        current=[p for p in points if p['model']==name and p['tag'].startswith('b4-')]
        for a,b in zip(current,current[1:]):
            comparisons[f'{name} {b["tag"]} minus {a["tag"]}']=paired([x['correct'] for x in rows[(name,b['tag'],'heldout4k')]],[x['correct'] for x in rows[(name,a['tag'],'heldout4k')]])
    for a,b in zip([p for p in points if p['model']=='W600' and p['tag'].startswith('b4-')], [p for p in points if p['model']=='W600-trunc']):
        for split in refs:
            comparisons[f'trunc minus plain {a["tag"]}/{split}']=paired([x['correct'] for x in rows[('W600-trunc',b['tag'],split)]],[x['correct'] for x in rows[('W600',a['tag'],split)]])
    a=next(p for p in points if p['model']=='W600' and p['tag'].startswith('b3-final'))
    comparisons['plain B4 minus B3 final/heldout4k']=paired([x['correct'] for x in rows[('W600',finals['W600'].name,'heldout4k')]],[x['correct'] for x in rows[('W600',a['tag'],'heldout4k')]])
    return dict(session=session,AB_audit=ab,producer_checked=checked,producer_missing=missing,
                points=points,C=prior['C'],stability=stability,truncation=trunc,comparisons=comparisons),finals


def interpolate(points,target,axis):
    points=sorted(points,key=lambda p:p[axis])
    for a,b in zip(points,points[1:]):
        if a[axis]<=target<=b[axis]:
            weight=math.log(target/a[axis])/math.log(b[axis]/a[axis])
            return dict(estimate_pp=(1-weight)*a['scores']['heldout4k']['accuracy_pp']+weight*b['scores']['heldout4k']['accuracy_pp'],
                        bracket=[a['tag'],b['tag']],weight_right=weight)
    return None


def curves(result):
    points=result['points'];interpolations=[];matched=[]
    for p in points:
        for name in NAMES[:2]:
            controls=[x for x in result['C'] if x['model'] in [name,'C (shared L1)'] and (x['cooldown_s']<1000)==(p['cooldown_s']<1000)]
            for axis in ['pairs','main_s']:
                estimate=interpolate(controls,p[axis],axis)
                interpolations.append(dict(model=p['model'],tag=p['tag'],control=name,axis=axis,estimate=estimate,
                    delta_pp=p['scores']['heldout4k']['accuracy_pp']-estimate['estimate_pp'] if estimate else None))
        if p['model']=='W600-trunc':
            estimate=interpolate([x for x in points if x['model']=='W600' and (x['cooldown_s']<1000)==(p['cooldown_s']<1000)],p['pairs'],'pairs')
            matched.append(dict(tag=p['tag'],pairs=p['pairs'],plain_interpolation=estimate,
                trunc_minus_plain_pp=p['scores']['heldout4k']['accuracy_pp']-estimate['estimate_pp'] if estimate else None))
    result['C_interpolations']=interpolations;result['AB_equal_pairs_interpolation']=matched
    # Late C ends earlier than B4. Interpolate W at measured C points as well;
    # never extrapolate C beyond its last observed anchor.
    inverse=[]
    for p in result['C']:
        for name in NAMES[-2:]:
            candidates=[x for x in points if x['model']==name and (x['cooldown_s']<1000)==(p['cooldown_s']<1000)]
            for axis in ['pairs','main_s']:
                estimate=interpolate(candidates,p[axis],axis)
                if estimate:
                    inverse.append(dict(model=name,control=p['model'],control_tag=p['tag'],axis=axis,
                        target=p[axis],W_interpolation=estimate,
                        W_minus_C_pp=estimate['estimate_pp']-p['scores']['heldout4k']['accuracy_pp']))
    result['W_at_C_points_interpolations']=inverse
    scenarios={}
    for name in NAMES[-2:]:
        ps=[p for p in points if p['model']==name]
        short=[p for p in ps if p['tag'].startswith('b4-probe')];a,b=short
        final=next(p for p in ps if p['tag'].startswith('b4-final'))
        slope=(b['scores']['heldout4k']['accuracy_pp']-a['scores']['heldout4k']['accuracy_pp'])/math.log(b['main_s']/a['main_s'])
        oldfinal=[p for p in ps if p['cooldown_s']>1000 and not p['tag'].startswith('b4-')]
        long_slope=((final['scores']['heldout4k']['accuracy_pp']-oldfinal[-1]['scores']['heldout4k']['accuracy_pp']) /
                    math.log(final['main_s']/oldfinal[-1]['main_s'])) if oldfinal else None
        scenarios[name]=dict(last_short_log_slope=slope,final50_plus9h=final['scores']['heldout4k']['accuracy_pp']+slope*math.log((final['main_s']+32400)/final['main_s']),
            final50_log_slope=long_slope,final50_fit_plus9h=final['scores']['heldout4k']['accuracy_pp']+long_slope*math.log((final['main_s']+32400)/final['main_s']) if long_slope is not None else None,
            warning='scenario anchored at final50; two short points do not establish a law or CI; plateau possible')
    result['scenarios']=scenarios
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    colors={'C-main':'#2470a0','C-long2x':'#268866','W600':'#7744aa','W600-trunc':'#c26435'}
    for ax,axis,div,label in zip(axes,['pairs','main_s'],[1e6,3600],['Main presented pairs, million','Main update time, hours']):
        for name,color in colors.items():
            seq=[p for p in (result['C'] if name.startswith('C') else points) if p['model'] in [name]+(['C (shared L1)'] if name.startswith('C') else [])]
            sh=sorted([p for p in seq if p['cooldown_s']<1000],key=lambda p:p[axis]);lo=sorted([p for p in seq if p['cooldown_s']>1000],key=lambda p:p[axis])
            ax.plot([p[axis]/div for p in sh],[p['scores']['heldout4k']['accuracy_pp'] for p in sh],'-o',color=color,label=name+' / 15m')
            ax.plot([p[axis]/div for p in lo],[p['scores']['heldout4k']['accuracy_pp'] for p in lo],':D',color=color,label=name+' / 50m')
        ax.axhline(72.0703125,color='#777',ls='--',lw=1,label='P1-B / 3h40m')
        ax.set(xlabel=label,ylabel='Greedy heldout4096, %');ax.grid(alpha=.2)
    axes[1].legend(fontsize=7,loc='lower right');fig.savefig(WORK/'results/b4-learning-curve.png',dpi=180);plt.close(fig)


def ensemble(finals):
    journals,receipts=B.load_all()
    for name,path in finals.items():
        files=sorted((path/'evaluation/k8-tune2048').glob('batch-*.json'));assert len(files)==64
        rr=[r for f in files for r in read(f)];assert len(rr)==2048 and sorted(r['id'] for r in rr)==IDS
        journals[name]={r['id']:r for r in rr}
    for i in IDS:
        for name in NAMES:
            r=journals[name][i];base=journals['C-main'][i]
            assert r['question_sha256']==base['question_sha256'] and r['reference']==base['reference'] and len(r['samples']['0.7'])==8
            win=B.choose(r,dict(mode='vote',temperature=.7,k=8,greedy_vote=False))
            assert B.correct(r['reference'],win['execution'])==r['correct']
    ratios={n:dict(sample=1.,greedy=1.) for n in NAMES}
    ratios.update({n:read(WORK/'results/b3-results.json')['ensemble']['ratios']['W-lr600'] for n in NAMES[-2:]})
    baseline=B.Replay(journals,'C-main',NAMES[:3]);basebits,stop,_=baseline.evaluate([3,5,4],[1,1,1])
    assert sum(basebits)==1535 and stop==1232
    for name,path in finals.items():
        assert sum(journals[name][i]['correct'] for i in IDS)==read(path/'evaluation/metrics.json')['K8_tune2048']['matches']
    mean_v7=6.5859375;static=[];grid=[];parity=baseline.validate([3,5,4],[1,1,1])
    def summarize(bits):return dict(matches=sum(bits),accuracy_pp=100*sum(bits)/2048,paired_vs_v7=paired(bits,basebits))
    for name in NAMES:
        sim=B.Replay(journals,name,[name])
        for greedy in [False,True]:
            bits,_,_=sim.evaluate([8],greedy=greedy)
            static.append(dict(models=[name],anchor=name,K=[8],greedy_vote=greedy,C_sample_units=8*ratios[name]['sample'],**summarize(bits)))
    for w in NAMES[-2:]:
        families=[]
        for n in [1,2,3]:
            for cs in itertools.combinations(NAMES[:3],n):
                names=list(cs)+[w];anchor='C-main' if 'C-main' in cs else cs[0]
                families.append((names,anchor))
                if n==1:families.append(([w]+list(cs),w))
        for names,anchor in families:
            sim=B.Replay(journals,anchor,names);k1=[1]*len(names)
            for ks in itertools.product(range(1,9),repeat=len(names)):
                if sum(ks)>14:continue
                bits,stopped,_=sim.evaluate(ks,k1);prob=(2048-stopped)/2048
                means=[1+prob*(k-1) for k in ks]
                # Timing uses the historical W prefix as its width indicator.
                mapped=[n.replace('W600','W-600') for n in names]
                rat={m:ratios[n] for n,m in zip(names,mapped)}
                time=B.timing(mapped,anchor.replace('W600','W-600'),ks,stopped,rat)
                row=dict(models=names,anchor=anchor,K1=k1,K2=list(ks),max_samples=sum(ks),stopped=stopped,
                    mean_samples=sum(means),mean_C_sample_units=sum(m*ratios[n]['sample'] for m,n in zip(means,names)),
                    timing=time,**summarize(bits))
                grid.append(row)
            print('replayed',names,'configs',len(grid),flush=True)
        # Static controls: same raw sample counts with C or a W replacement.
        for names,ks in [(['C-main',w],[4,4]),(['C-main','C-long2x',w],[3,5,4]),
                         (['C-main','pilot-b',w],[3,4,5]),(NAMES[:3]+[w],[3,5,4,1])]:
            sim=B.Replay(journals,'C-main',names);bits,_,_=sim.evaluate(ks)
            static.append(dict(models=names,anchor='C-main',K=list(ks),greedy_vote=True,C_sample_units=sum(k*ratios[n]['sample'] for k,n in zip(ks,names)),**summarize(bits)))
    categories={
        'same_sample_budget':lambda r:r['max_samples']<=12 and r['mean_samples']<=mean_v7,
        'same_weighted_mean_budget':lambda r:r['max_samples']<=12 and r['mean_C_sample_units']<=mean_v7,
        'plus15pct_weighted_mean_budget':lambda r:r['max_samples']<=14 and r['mean_C_sample_units']<=1.15*mean_v7,
        'all_max14':lambda r:True}
    best={};selected=[]
    for key,predicate in categories.items():
        eligible=[r for r in grid if predicate(r) and r['timing']['conservative_s']<=140]
        winner=max(eligible,key=lambda r:(r['matches'],-r['timing']['conservative_s'],-r['mean_C_sample_units']))
        best[key]=winner
        if winner not in selected:selected.append(winner)
    best['families_same_weighted_mean_budget']={}
    for w in NAMES[-2:]:
        eligible=[r for r in grid if w in r['models'] and categories['same_weighted_mean_budget'](r) and r['timing']['conservative_s']<=140]
        winner=max(eligible,key=lambda r:(r['matches'],-r['timing']['conservative_s'],-r['mean_C_sample_units']))
        best['families_same_weighted_mean_budget'][w]=winner
        if winner not in selected:selected.append(winner)
    for row in selected:
        sim=B.Replay(journals,row['anchor'],row['models']);parity+=sim.validate(row['K2'],row['K1'])
    path=WORK/'results/b4-ensemble-grid.csv'
    with path.open('w') as f:
        writer=csv.writer(f);writer.writerow(['models','anchor','K1','K2','matches','stopped','max_samples','mean_samples','mean_C_units','delta_v7_pp','normal_lo','normal_hi','bootstrap_lo','bootstrap_hi','gained','lost','conservative_s','stress15_s'])
        for r in grid:
            c=r['paired_vs_v7'];writer.writerow(['+'.join(r['models']),r['anchor'],'/'.join(map(str,r['K1'])),'/'.join(map(str,r['K2'])),r['matches'],r['stopped'],r['max_samples'],r['mean_samples'],r['mean_C_sample_units'],c['delta_pp'],*c['CI95_pp'],*c['bootstrap_CI95_pp'],c['gained'],c['lost'],r['timing']['conservative_s'],r['timing']['stress15_s']])
    return dict(baseline_v7=dict(matches=1535,stopped=1232,mean_samples=mean_v7),ratios=ratios,
        static=static,best=best,grid_count=len(grid),grid_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        journal_sha256=receipts,parity_rows=parity,graph_fixtures=B.graph_fixtures(),
        selection='exploratory 2-4 models, one B4 W + 1-3 C, K1=1/model, K2=1..8/model, total<=14; paired intervals unadjusted for search; max test/tune conservative<=140s',
        timing_limit='proxy from B3/v7 rounded waves, C sample1/W1.3467, +4s per W load; no actual QK/mixed cold timer or memory measurement')


def expanded_C_control():
    """Same expanded K grid without W, so extra sample budget is not credited to W."""
    journals,_=B.load_all();names=NAMES[:3];sim=B.Replay(journals,'C-main',names)
    base,_,_=sim.evaluate([3,5,4],[1,1,1]);rows=[]
    for ks in itertools.product(range(1,9),repeat=3):
        if sum(ks)>14:continue
        bits,stopped,_=sim.evaluate(ks,[1,1,1]);mean=3+(sum(ks)-3)*(2048-stopped)/2048
        rows.append(dict(models=names,anchor='C-main',K1=[1,1,1],K2=list(ks),max_samples=sum(ks),
            matches=sum(bits),stopped=stopped,mean_samples=mean,mean_C_sample_units=mean,
            paired_vs_v7=paired(bits,base),timing=B.timing(names,'C-main',ks,stopped,{n:dict(sample=1.,greedy=1.) for n in names})))
    out={}
    for key,limit,maximum in [('same_budget',6.5859375,12),('plus15pct_budget',1.15*6.5859375,14)]:
        eligible=[r for r in rows if r['mean_samples']<=limit and r['max_samples']<=maximum and r['timing']['conservative_s']<=140]
        best=max(eligible,key=lambda r:(r['matches'],-r['timing']['conservative_s'],-r['mean_samples']))
        sim.validate(best['K2'],best['K1']);out[key]=best
    out['configurations']=len(rows)
    ablations=[]
    for nn,ks in [(['C-long2x','pilot-b'],[1,8]),(['C-long2x','pilot-b','C-main'],[1,8,1])]:
        ss=B.Replay(journals,'C-long2x',nn);bits,stopped,_=ss.evaluate(ks,[1]*len(nn));ss.validate(ks,[1]*len(nn))
        ablations.append(dict(models=nn,K1=[1]*len(nn),K2=ks,anchor='C-long2x',matches=sum(bits),
            stopped=stopped,mean_samples=len(nn)+(sum(ks)-len(nn))*(2048-stopped)/2048,paired_vs_v7=paired(bits,base)))
    out['winning_policy_ablation']=ablations
    return out


def main():
    result,finals=audit();print('audited',len(result['producer_checked']),'producer files',flush=True)
    curves(result);result['ensemble']=ensemble(finals)
    result['ensemble']['expanded_C_control']=expanded_C_control()
    result['ensemble']['winning_policy_ablation']=result['ensemble']['expanded_C_control'].pop('winning_policy_ablation')
    result.update(status='complete',GPU=False,new_execution=False,new_sampling=False,API=False,
        limits=['Equal-pair A/B interpolates cooled checkpoints, not a measured matched anchor; no CI for interpolations',
                'Equal time means accumulated main update time + same timed cooldown, not identical total billed wall or equal cooldown pairs',
                'Full states and FP16 not downloaded: tensor finiteness/restore not independently rerun',
                'Repeated heldout/tune; pointwise intervals, no search correction or independent final evaluation'])
    result['inputs_sha256']={**INPUTS,**B.INPUTS}
    for path in [Path(__file__),HERE/'analyze-b3-results.py']:
        result['inputs_sha256'][str(path.relative_to(WORK))]=hashlib.sha256(path.read_bytes()).hexdigest()
    B.write(WORK/'results/b4-results.json',result)
    B.write(WORK/'checks/b4-results-cpu.json',dict(passed=True,producer_files=len(result['producer_checked']),
        missing_files=len(result['producer_missing']),main_updates=sum(x['attempts_B4'] for x in result['stability'].values()),
        cooldown_updates=sum(x['updates'] for s in result['stability'].values() for x in s['cooldowns']),
        common_stream_prefix=result['AB_audit']['compared_prefix_updates'],K8_journals=320,
        replay_configurations=result['ensemble']['grid_count'],production_and_independent_parity_rows=result['ensemble']['parity_rows'],
        expanded_C_control_configurations=result['ensemble']['expanded_C_control']['configurations'],
        C_control_and_ablation_parity_rows=8192,
        no_forward=True,no_execution=True,no_GPU=True,no_API=True))
    print(json.dumps(result['ensemble']['best'],indent=2),flush=True)


if __name__=='__main__':main()
