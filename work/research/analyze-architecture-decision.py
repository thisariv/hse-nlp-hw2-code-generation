"""CPU saved-output replay and explicit scenario model, no new executions.

Single-model K>8 is unobserved. Model-size effects are assumptions, not fitted
from one C trajectory. --bench verifies and imports a saved T4 benchmark.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(WORK/'kaggle'),str(WORK/'kaggle/src')]
spec=importlib.util.spec_from_file_location('r1_audit',WORK/'research/analyze-r1-results.py')
R=importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
from final_adaptive import select, numeric, same, replay_record
from final_ensemble import replay_record as ensemble_replay
from final_v6_analysis import timing_estimate
from final_v7_analysis import timing_estimate as v7_time
from shape_specs import SHAPES, name, parameter_formula


def read(p): return json.loads(p.read_text())
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def paired(a,b):
    d=np.array(b,dtype=float)-np.array(a,dtype=float)
    return dict(delta_pp=100*float(d.mean()),CI95_pp=(100*np.array([d.mean()-1.96*d.std(ddof=1)/math.sqrt(len(d)),d.mean()+1.96*d.std(ddof=1)/math.sqrt(len(d))])).tolist(),
                gains=int((d>0).sum()),losses=int((d<0).sum()))


def summarize(bits,base,counts=None):
    return dict(matches=sum(bits),n=2048,accuracy_pp=100*np.mean(bits),paired_vs_greedy=paired(base,bits),
                mean_samples=float(np.mean(counts)) if counts is not None else None)


def saturation(rows):
    # Least squares of gain a*k/(b+k), fixed g(0)=0, no monotonicity assumption on data.
    k=np.array([r['k'] for r in rows if r['observed'] and r['k']>0])
    y=np.array([r['accuracy_pp']-rows[0]['accuracy_pp'] for r in rows if r['observed'] and r['k']>0])
    options=[]
    for b in np.geomspace(.05,100,800):
        x=k/(k+b); a=max(0.,float(x@y/(x@x))); rmse=float(np.sqrt(np.mean((a*x-y)**2)))
        options.append((rmse,a,float(b)))
    rmse,a,b=min(options)
    return dict(function='gain_pp=a*k/(b+k)',a=a,b=b,RMSE_pp=rmse,
                assumption='descriptive saturation only; extrapolation is not independent sampling evidence')


def gain(f,k): return f['a']*k/(f['b']+k) if k else 0.


def voting(journals):
    curves={}; bits_by_curve={}
    for label,names in [('C-main',['L3-main']),('C-long',['L3-long']),('P1-B',['P1-B']),
                        ('R1-deep',['deep']),('R1-control',['control']),
                        ('C-main+long',['L3-main','L3-long']),('C-main+P1-B',['L3-main','P1-B']),
                        ('C-main+R1-deep',['L3-main','deep'])]:
        base_name=names[0]; base=[R.l2.matches(r['reference'],r['greedy']['execution']) for r in journals[base_name]]
        rows=[]; bb={}
        for k in range(17):
            if k>8*len(names):
                rows.append(dict(k=k,observed=False,accuracy_pp=None));continue
            ks=[k//len(names)+(i<k%len(names)) for i in range(len(names))]
            bits=[]
            for j in range(2048):
                record=journals[base_name][j]
                pool=[c for m,count in zip(names,ks) for c in journals[m][j]['samples']['0.7'][:count]]
                winner=select(record['greedy'],pool,[],k=k)
                bits.append(R.l2.matches(record['reference'],winner['execution']))
            rows.append(dict(k=k,allocation=ks,observed=True,**summarize(bits,base,[k]*2048)))
            bb[k]=bits
        fit=saturation(rows)
        for row in rows: row['fit_accuracy_pp']=rows[0]['accuracy_pp']+gain(fit,row['k'])
        curves[label]=dict(models=names,rows=rows,fit=fit)
        bits_by_curve[label]=bb
    adaptive=[]; base=bits_by_curve['C-main'][0]
    for k in range(2,9):
        bits=[]; counts=[]; stopped=0
        for a,b in zip(journals['L3-main'],journals['L3-long']):
            winner,stop,count=replay_record(a['greedy'],a['samples']['0.7'],b['samples']['0.7'],k1=2,k2=k)
            bits.append(R.l2.matches(a['reference'],winner['execution']));counts.append(count);stopped+=stop
        adaptive.append(dict(K1=2,K2=k,max_samples=2*k,stopped=stopped,**summarize(bits,base,counts)))
    v7bits=[];v7counts=[]
    for j in range(2048):
        a=journals['L3-main'][j]
        winner,stop,count=ensemble_replay(a['greedy'],{m:journals[m][j]['samples']['0.7'] for m in ['L3-main','L3-long','P1-B']},
            [dict(name=m,K1=1,K2=k) for m,k in zip(['L3-main','L3-long','P1-B'],[3,5,4])])
        v7bits.append(R.l2.matches(a['reference'],winner['execution']));v7counts.append(count)
    assert sum(v7bits)==1535
    for row in adaptive:
        assert row['stopped']==1133
        if row['K2']==6: assert row['matches']==1533
    return curves,adaptive,summarize(v7bits,base,v7counts)


def timing_penalties(adaptive):
    # Replay the v6 actual stage structure; only measured generation phases scale.
    rows=[]
    for r in [1.,1.5,2.,2.5]:
        grid=[]
        for k in range(2,9):
            timing=timing_estimate(max(3,k)); runs=[]
            for t in timing['measured_v5_runs'].values():
                fixed=t['fixed_cold_load_restore_tail_seconds']+2
                second=0 if k==2 else t['stage2_batches_both_models']
                variable=t['greedy_seconds']+t['stage1_seconds']+second*timing['stage2_worst_old_batch_proxy_seconds']*1.1/timing['sample_speedup_assumed']
                runs.append(fixed+r*variable)
            grid.append(dict(K2=k,seconds=max(runs),mean_samples=next(x for x in adaptive if x['K2']==k)['mean_samples']))
        result=dict(decoder_cost_ratio=r,grid=grid)
        for budget in [160,180]:
            feasible=[x for x in grid if x['seconds']<=budget]
            selected=max(feasible,key=lambda x:x['K2']) if feasible else None
            result[str(budget)]=selected
            if selected:
                observed=next(x for x in adaptive if x['K2']==selected['K2'])
                result[str(budget)]=dict(selected,matches=observed['matches'],accuracy_pp=observed['accuracy_pp'],
                    loss_vs_K6_pp=(1533-observed['matches'])*100/2048)
        rows.append(result)
    return rows


def hybrid_replays(journals):
    """Observed quality of unequal hybrid pools, using the larger model's greedy."""
    rows=[]
    for large in ['deep','P1-B','L3-long']:
        for kl,kc in [(1,4),(2,6),(2,8),(4,4),(4,8),(8,8)]:
            bits=[];base=[];mean_large=[];mean_cheap=[];stops=0
            for a,c in zip(journals[large],journals['L3-main']):
                winner,stop,count=ensemble_replay(a['greedy'],
                    {large:a['samples']['0.7'],'cheap':c['samples']['0.7']},
                    [dict(name=large,K1=1,K2=kl),dict(name='cheap',K1=1,K2=kc)])
                bits.append(R.l2.matches(a['reference'],winner['execution']))
                base.append(R.l2.matches(a['reference'],a['greedy']['execution']))
                mean_large.append(1 if stop else kl);mean_cheap.append(1 if stop else kc);stops+=stop
            rows.append(dict(large=large,K2_large=kl,K2_C=kc,stopped=stops,
                mean_large=float(np.mean(mean_large)),mean_C=float(np.mean(mean_cheap)),
                **summarize(bits,base,np.array(mean_large)+np.array(mean_cheap))))
    return rows


def v7_penalties(journals):
    grid=[]
    for ks in [[1,1,1],[2,2,2],[3,3,3],[3,5,4]]:
        models=[dict(name=m,K1=1,K2=k) for m,k in zip(['L3-main','L3-long','P1-B'],ks)]
        bits=[];counts=[];stopped=0
        for j in range(2048):
            a=journals['L3-main'][j]
            win,stop,count=ensemble_replay(a['greedy'],{m['name']:journals[m['name']][j]['samples']['0.7'] for m in models},models)
            bits.append(R.l2.matches(a['reference'],win['execution']));counts.append(count);stopped+=stop
        grid.append(dict(K2=ks,stopped=stopped,matches=sum(bits),accuracy_pp=100*np.mean(bits),mean_samples=float(np.mean(counts)),
                         timing=v7_time(models,stopped)))
    results=[]
    for r in [1.,1.5,2.,2.5]:
        costs=[]
        for row in grid:
            times=[]
            for t in row['timing']['runs'].values():
                fixed=t['fixed_seconds']+t['extra_model_load_warmup_reserve_seconds']+2
                times.append(fixed+r*(t['conservative_seconds']-fixed))
            costs.append({k:row[k] for k in ['K2','stopped','matches','accuracy_pp','mean_samples']}|dict(seconds=max(times)))
        result=dict(decoder_cost_ratio=r,grid=costs)
        for budget in [160,180]:
            feasible=[x for x in costs if x['seconds']<=budget]
            selected=max(feasible,key=lambda x:sum(x['K2'])) if feasible else None
            result[str(budget)]=dict(selected,loss_vs_v7_pp=(1535-selected['matches'])*100/2048) if selected else None
        results.append(result)
    return results


def single_replays(journals):
    """One-model research policy, not a change to the production ensemble."""
    records=journals['L3-main']
    base=[R.l2.matches(r['reference'],r['greedy']['execution']) for r in records]
    rows=[]
    for k in range(2,9):
        models=[dict(name='single',K1=2,K2=k)]
        bits=[];counts=[];stopped=0
        for r in records:
            winner,stop,count=ensemble_replay(r['greedy'],{'single':r['samples']['0.7']},models)
            bits.append(R.l2.matches(r['reference'],winner['execution']))
            counts.append(count);stopped+=stop
        rows.append(dict(K1=2,K2=k,stopped=stopped,**summarize(bits,base,counts)))
    return rows


def policy_timing(models,stopped,greedy_ratio,sample_ratios):
    """Price rounded v7 waves using v5 phases and loaded v6 speedups.

    Keep cold load/CPU execution/tail/write intact, even for a single model.
    The measured seq/s ratio already includes any 1024 -> 512 lane fallback.
    """
    timing=v7_time(models,stopped)
    old=timing_estimate(6)
    runs={}
    for label,t in timing['runs'].items():
        p=old['measured_v5_runs'][label]
        n=t['rows']; extra=[m['K2']-m['K1'] for m in models]
        wave=1024//max(extra) if max(extra) else 0
        first_units=sum(math.ceil(n/(1024//m['K1']))*r for m,r in zip(models,sample_ratios))
        second_units=(math.ceil(t['escalated_rows_proxy']/wave)*sum(r for k,r in zip(extra,sample_ratios) if k>0)) if wave else 0
        old_first=sum(math.ceil(n/(1024//2)) for _ in range(2))
        mean_batch=p['stage1_seconds']/old_first
        worst_batch=old['stage2_worst_old_batch_proxy_seconds']*1.1/old['sample_speedup_assumed']
        fixed=t['fixed_seconds']+t['extra_model_load_warmup_reserve_seconds']+2
        variable=p['greedy_seconds']*greedy_ratio+first_units*mean_batch+second_units*worst_batch
        runs[label]=dict(fixed_seconds=fixed,greedy_seconds=p['greedy_seconds']*greedy_ratio,
            sample_seconds=variable-p['greedy_seconds']*greedy_ratio,seconds=fixed+variable,
            generation15pct_slower_seconds=fixed+1.15*variable,
            escalated_rows_proxy=t['escalated_rows_proxy'])
    return dict(seconds=max(r['seconds'] for r in runs.values()),runs=runs,
        generation15pct_slower_seconds=max(r['generation15pct_slower_seconds'] for r in runs.values()),
        production_measured=False,execution='retained v5 fixed overhead and loaded decoder profile; no fresh execution')


def policy_budget(grid):
    # Select by declared sample cap, never by observed correctness.
    return {str(b):max((r for r in grid if r['timing']['seconds']<=b),
                      key=lambda r:sum(r['K2']) if isinstance(r['K2'],list) else r['K2'],default=None)
            for b in [160,180]}


def measured_bench(path,data):
    """Verify downloaded and current modules; retain a small tracked receipt."""
    b=read(path)
    assert b['format']=='nlp-hw02-shape-bench-v1' and b['device']=='cuda:0'
    assert b['status']=='complete' and b['torch']=='2.11.0+cu128' and b['cuda']=='12.8'
    assert b['lengths_payload_sha256']==hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert [r['shape'] for r in b['rows']]==[name(s) for s in SHAPES]
    downloaded=path.parent.parent/'shape-bench-modules'
    for n,h in b['module_sha256'].items():
        assert h==hashlib.sha256((WORK/'kaggle/src'/n).read_bytes()).hexdigest(),n
        assert h==hashlib.sha256((downloaded/n).read_bytes()).hexdigest(),n
    assert read(downloaded/'lengths.json')==data
    assert 'KernelWorkerStatus.COMPLETE' in (path.parent.parent/'kernel-status.txt').read_text()
    for shape,r in zip(SHAPES,b['rows']):
        assert r['status']=='complete' and r['parameters']==parameter_formula(shape)
        assert r['training']['micro']>=16
        for mode in ['greedy','sample']:
            g=r['generation'][mode]
            assert g['sequences']==1024 and math.isclose(g['seq_per_s'],1024/g['seconds'])
    receipt=dict(b,summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source=str(path.relative_to(WORK)),kernel_status=(path.parent.parent/'kernel-status.txt').read_text().strip(),
        downloaded_and_current_module_hashes_match=True)
    write(WORK/'results/architecture-shape-bench.json',receipt)
    return receipt


def learning(data):
    points=read(WORK/'results/l3-final-results.json')['points']
    ordinary=[p for p in points if p['model'] in ['C (shared L1)','C-main']]
    fitted=[]
    for p in ordinary:
        fitted.append(dict(model=p['model'],tag=p['tag'],main_s=p['main_s'],updates=p['updates'],
            pair_tokens_estimate=data['C_main_curve_tokens'][str(p['updates'])]['pair'],
            target_tokens_estimate=data['C_main_curve_tokens'][str(p['updates'])]['target'],
            greedy_pp=p['scores']['heldout4k']['accuracy_pp'],cooldown50=int(p['cooldown_s']>2000)))
    D=np.array([p['target_tokens_estimate'] for p in fitted]); D0=int(D[-1])
    y=np.array([p['greedy_pp'] for p in fitted]); cool=np.array([p['cooldown50'] for p in fitted])
    # Floor and capacity exponents are not identifiable from only C; sensitivity grid.
    fits=[]
    for floor in [85,90,95]:
        choices=[]
        for beta in np.linspace(.02,.8,1000):
            X=np.stack([(D/D0)**(-beta),-cool],1)
            c,delta=np.linalg.lstsq(X,floor-y,rcond=None)[0]
            pred=floor-X@np.array([c,delta])
            if c>0 and 0<=delta<=5:
                choices.append((float(np.sqrt(np.mean((pred-y)**2))),float(beta),float(c),float(delta)))
        rmse,beta,c,delta=min(choices)
        fits.append(dict(floor_pp=floor,beta=beta,c=c,cooldown50_offset_pp=delta,RMSE_pp=rmse,D0_target_tokens=D0))
    # P1/R1 are stress/validation points, never treated as the same recipe curve.
    stress={m:{k:v for k,v in r.items() if k in ['main_s','updates','scores','cooldown_wall_s']} for m,r in read(WORK/'results/r1-results.json')['models'].items()}
    for m,r in stress.items():
        if m in data['stress_curve_tokens']: r['presented_tokens']=data['stress_curve_tokens'][m]
    return dict(points=fitted,fits=fits,P1_R1_stress=stress,
        token_accounting='C-main exact offset sums of cyclic ordinary order up to attempted updates x128; includes EOS and repeated presentations; long weighting/truncation are separate stress points',
        limitation='only C can identify data/time slope; size elasticity and CE-to-execution relationship are unidentifiable')


def forecasts(data,learn,curves,hybrids,single,adaptive,v7_baselines,bench):
    cparams=parameter_formula((6,6,512))
    reference=next(p for p in learn['points'] if p['tag']=='l2-62280')
    actual_c_rate=128*(213386-126452)/(79850.5201694716-46370.143)
    measured={r['shape']:r for r in bench['rows']}
    c=measured['e6-d6-w512']; calibration=actual_c_rate/c['training']['pairs_per_s']
    tune_shift=curves['C-main']['rows'][0]['accuracy_pp']-reference['greedy_pp']
    result=[]
    for shape in SHAPES:
        row=measured[name(shape)]; params=row['parameters']
        train_ratio=c['training']['pairs_per_s']/row['training']['pairs_per_s']
        sample_ratio=row['generation']['sample']['seconds']/c['generation']['sample']['seconds']
        greedy_ratio=row['generation']['greedy']['seconds']/c['generation']['greedy']['seconds']
        pairs_s=row['training']['pairs_per_s']; operational_rate=pairs_s*calibration
        total_tokens=36*3600*operational_rate*data['mean_pair_tokens']
        target_tokens=36*3600*operational_rate*data['mean_target_tokens']
        # Exact C-curve slope when n=1; capacity exponent remains an assumption.
        def estimate(fit,rate,gamma,eta):
            n=(params['encoder']/cparams['encoder'])**eta*(params['decoder']/cparams['decoder'])**(1-eta)
            dd=36*3600*rate*data['mean_target_tokens']/reference['target_tokens_estimate']
            return fit['floor_pp']-(fit['floor_pp']-reference['greedy_pp'])*dd**(-fit['beta'])*n**(-gamma)
        estimates=[estimate(fit,rate,lam*alpha,eta) for fit in learn['fits']
                   for rate in [operational_rate,pairs_s] for lam in [0.,.25,.5,.75]
                   for alpha in [.2,.34,.5] for eta in [0.,.25,.5,1.]]
        central=next(f for f in learn['fits'] if f['floor_pp']==90)
        greedy_full=estimate(central,operational_rate,.17,.25)
        greedy_tune=greedy_full+tune_shift
        single_grid=[]
        for k in range(2,17):
            observed=next((s for s in single if s['K2']==k),None)
            stopped=single[0]['stopped']
            models=[dict(name='single',K1=2,K2=k)]
            mean=2+(1-stopped/2048)*(k-2)
            single_grid.append(dict(K2=k,mean_samples=mean,quality_observed=observed is not None,
                uplift_pp=observed['paired_vs_greedy']['delta_pp'] if observed else None,
                timing=policy_timing(models,stopped,greedy_ratio,[sample_ratio])))
        sampled=policy_budget([s for s in single_grid if s['quality_observed']])
        capacity=policy_budget(single_grid)
        stress_capacity={str(b):max((s for s in single_grid if s['timing']['generation15pct_slower_seconds']<=b),
                                  key=lambda s:s['K2'],default=None) for b in [160,180]}
        chosen=sampled['160']; assert chosen is not None
        uplift=chosen['uplift_pp']
        tail=[]
        for label,policies in [('nominal',capacity),('generation15pct_slower',stress_capacity)]:
            for budget,p in policies.items():
                assert p is not None
                # Anchor the observed adaptive K8. Only its unobserved increment
                # follows the descriptive static C-main saturation curve.
                extra=gain(curves['C-main']['fit'],p['K2'])-gain(curves['C-main']['fit'],8)
                future_uplift=(single[p['K2']-2]['paired_vs_greedy']['delta_pp'] if p['K2']<=8
                               else single[-1]['paired_vs_greedy']['delta_pp']+extra)
                tail.append(dict(timing_scenario=label,budget_seconds=int(budget),K2=p['K2'],
                    mean_samples=p['mean_samples'],uplift_pp=future_uplift,final_pp=greedy_tune+future_uplift,
                    quality_extrapolated=p['K2']>8,
                    seconds=p['timing']['seconds' if label=='nominal' else 'generation15pct_slower_seconds'],
                    scenario_range_pp=[min(min(estimates),reference['greedy_pp'])+tune_shift+max(0,future_uplift-3)-2,
                                       max(estimates)+tune_shift+future_uplift+5]))
        analogs={}
        for label,grid in [('v6',adaptive),('v7',v7_baselines)]:
            variants=[]
            for a in grid:
                ks=[a['K2']]*2 if label=='v6' else a['K2']
                models=[dict(name=str(i),K1=2 if label=='v6' else 1,K2=k) for i,k in enumerate(ks)]
                variants.append(dict(K2=ks,mean_samples=a['mean_samples'],accuracy_pp=a['accuracy_pp'],
                    timing=policy_timing(models,a['stopped'],greedy_ratio,[sample_ratio]*len(ks))))
            analogs[label]=policy_budget(variants)
        hybrid_grid=[]
        for analog in (h for h in hybrids if h['large']=='P1-B'):
            models=[dict(name='large',K1=1,K2=analog['K2_large']),dict(name='C',K1=1,K2=analog['K2_C'])]
            timing=policy_timing(models,analog['stopped'],greedy_ratio,[sample_ratio,1.])
            for t in timing['runs'].values():
                t['fixed_seconds']+=4; t['seconds']+=4; t['generation15pct_slower_seconds']+=4
            timing['seconds']+=4;timing['generation15pct_slower_seconds']+=4
            prediction=analog['accuracy_pp']+.5*(greedy_full-72.0703125)
            hybrid_grid.append(dict(K2=[analog['K2_large'],analog['K2_C']],mean_large=analog['mean_large'],mean_C=analog['mean_C'],
                observed_analog_pp=analog['accuracy_pp'],timing=timing,final_pp=prediction,
                scenario_range_pp=[min(estimates)*.5+analog['accuracy_pp']-.5*72.0703125-3,
                                   max(estimates)*.5+analog['accuracy_pp']-.5*72.0703125+3]))
        hybrid=next(h for h in hybrid_grid if h['K2']==[2,6])
        result.append(dict(shape=name(shape),parameters=params,training_cost_ratio=train_ratio,
            decoder_cost_ratio=sample_ratio,greedy_cost_ratio=greedy_ratio,
            throughput_source='measured forced T4 ratios; absolute training rate calibrated to historical C',
            pairs_per_s_bench=pairs_s,pairs_per_s_estimate=operational_rate,bench_to_production_factor=calibration,
            main_hours=36,total_GPU_hours=40,pair_tokens_B=total_tokens/1e9,target_tokens_B=target_tokens/1e9,
            pair_tokens_B_raw_bench=total_tokens/calibration/1e9,target_tokens_B_raw_bench=target_tokens/calibration/1e9,
            epochs=total_tokens/(data['source_tokens_per_epoch']+data['target_tokens_with_EOS_per_epoch']),
            greedy_full4k_pp=greedy_full,greedy_pp=greedy_tune,full4k_to_tune_shift_pp=tune_shift,
            greedy_scenario_range_pp=[min(estimates)+tune_shift-1.5,max(estimates)+tune_shift+1.5],
            single_adaptive_grid=single_grid,single_adaptive_budgets=sampled,timing_only_capacity_budgets=capacity,
            stress_capacity_budgets=stress_capacity,tail_extrapolation=tail,
            homogeneous_policy_analogs=analogs,vote_uplift_proxy_pp=uplift,final_pp=greedy_tune+uplift,
            final_scenario_range_pp=[min(min(estimates),reference['greedy_pp'])+tune_shift+max(0,uplift-1.5)-2,
                                     max(estimates)+tune_shift+uplift+3.5],
            no_training_gain_final_pp=reference['greedy_pp']+tune_shift+uplift,
            hybrid=hybrid,hybrid_grid=hybrid_grid,
            forecast_assumptions='C-data slope preserved; gamma=.17, encoder share=.25 unidentifiable; transfer single C adaptive uplift; hybrid half greedy gain; ranges are scenarios, not CI'))
    return result


def plot(curves,adaptive):
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for label in ['C-main','C-long','P1-B','R1-deep','C-main+long','C-main+R1-deep']:
        rows=[r for r in curves[label]['rows'] if r['observed']]
        axes[0].plot([r['k'] for r in rows],[r['accuracy_pp'] for r in rows],'.-',label=label)
    axes[0].set(xlabel='Stochastic samples + one greedy',ylabel='Execution accuracy, tune2048 (%)')
    axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    rows=curves['C-main+long']['rows']
    axes[1].plot([r['k'] for r in rows],[r['accuracy_pp'] for r in rows],'.-',label='Static C-main+long')
    axes[1].plot([r['mean_samples'] for r in adaptive],[r['accuracy_pp'] for r in adaptive],'o-',label='Adaptive 2+2 -> K+K')
    axes[1].scatter([6.5859375],[74.951171875],marker='*',s=100,label='Frozen v7')
    axes[1].set(xlabel='Mean stochastic sequences per task',ylabel='Execution accuracy, tune2048 (%)')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.savefig(WORK/'results/architecture-voting-curve.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bench',type=Path,default=WORK/'results/kaggle-runs/nlp-hw02-shape-bench/1/shape-bench/summary.json');args=parser.parse_args()
    journals={}; receipts={}
    for m in R.LOCATIONS:
        rows=R.l2.read(R.file(m,'evaluation__heldout4k-per-example'))
        journals[m], hashes=R.journal(m,rows); receipts.update(hashes)
    for m in journals:
        assert [(r['id'],r['question_sha256'],r['reference']) for r in journals[m]]==[(r['id'],r['question_sha256'],r['reference']) for r in journals['L3-main']]
    curves,adaptive,v7=voting(journals)
    data=read(WORK/'kaggle/shape-lengths.json');learn=learning(data)
    bench=measured_bench(args.bench.resolve(),data)
    single=single_replays(journals); v7_penalty=v7_penalties(journals)
    hybrid_replays_cache=hybrid_replays(journals)
    result=dict(GPU=False,new_sampling=False,new_execution=False,journal_files=len(receipts),journal_sha256=receipts,
        sampling='T0.7, seed270042, first k saved samples; single9..16 unobserved; balanced mixture order fixed',
        curves=curves,adaptive_v6=adaptive,adaptive_single=single,v7=v7,observed_hybrids=hybrid_replays_cache,
        benchmark_receipt='architecture-shape-bench.json',benchmark_summary_sha256=bench['summary_sha256'],
        timing_penalties=timing_penalties(adaptive),v7_timing_penalties=v7_penalty,learning=learn,
        forecasts=forecasts(data,learn,curves,hybrid_replays_cache,single,adaptive,v7_penalty[0]['grid'],bench),
        forecast_warning='scenario ranges, not statistical prediction CI; public77.92 is a different distribution, no conversion fitted',
        frozen_anchor='v7 tune1535/2048; user public77.92; neither is a fresh unbiased metric')
    write(WORK/'results/architecture-decision.json',result);plot(curves,adaptive)
    print(json.dumps(dict(journal_files=len(receipts),curves={k:[r.get('matches') for r in v['rows']] for k,v in curves.items()},
        adaptive=adaptive,timing=result['timing_penalties'],fits=learn['fits'],
        shapes=[{k:r[k] for k in ['shape','parameters','pair_tokens_B','greedy_pp','final_pp','single_adaptive_budgets']} for r in result['forecasts']]),indent=2))


if __name__=='__main__': main()
