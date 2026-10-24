"""Check measured inputs, calibration and rounded adaptive timing; no GPU/exec."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

WORK=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('architecture_analysis',WORK/'research/analyze-architecture-decision.py')
A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)


def main():
    d=A.read(WORK/'results/architecture-decision.json')
    data=A.read(WORK/'kaggle/shape-lengths.json')
    b=A.read(WORK/'results/architecture-shape-bench.json')
    raw=WORK/b['source']
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==b['summary_sha256']==d['benchmark_summary_sha256']
    assert b['status']=='complete' and len(b['rows'])==13
    assert b['downloaded_and_current_module_hashes_match']
    for n,h in b['module_sha256'].items():
        assert hashlib.sha256((WORK/'kaggle/src'/n).read_bytes()).hexdigest()==h
    assert d['journal_files']==384 and len(d['journal_sha256'])==384
    for c in d['curves'].values():
        assert [r['k'] for r in c['rows']]==list(range(17))
        for r in c['rows']:
            if r['observed']: assert r['accuracy_pp']==100*r['matches']/2048
            else: assert r['accuracy_pp'] is None
    assert all(not d['curves']['C-main']['rows'][k]['observed'] for k in range(9,17))
    assert d['curves']['C-main']['rows'][8]['matches']==1498
    assert d['curves']['C-main+long']['rows'][12]['matches']==1533
    assert d['v7']['matches']==1535 and d['v7']['mean_samples']==6.5859375
    assert next(r for r in d['adaptive_v6'] if r['K2']==6)['matches']==1533
    single=d['adaptive_single']
    assert single[-1]['matches']==1498 and single[-1]['stopped']==1336
    assert single[-1]['mean_samples']==4.0859375
    c=b['rows'][0]; f=d['forecasts'][0]
    rate=128*(213386-126452)/(79850.5201694716-46370.143)
    assert math.isclose(f['pairs_per_s_estimate'],rate)
    ref=d['learning']['points'][-1]; fit=next(x for x in d['learning']['fits'] if x['floor_pp']==90)
    expected=90-(90-ref['greedy_pp'])*(f['target_tokens_B']*1e9/ref['target_tokens_estimate'])**(-fit['beta'])
    assert math.isclose(f['greedy_full4k_pp'],expected)  # C slope is not halved.
    for r,row in zip(d['forecasts'],b['rows']):
        assert r['shape']==row['shape'] and r['parameters']==row['parameters']
        assert math.isclose(r['training_cost_ratio'],c['training']['pairs_per_s']/row['training']['pairs_per_s'])
        for mode,key in [('greedy','greedy_cost_ratio'),('sample','decoder_cost_ratio')]:
            assert math.isclose(r[key],row['generation'][mode]['seconds']/c['generation'][mode]['seconds'])
        assert math.isclose(r['pair_tokens_B'],129600*r['pairs_per_s_estimate']*data['mean_pair_tokens']/1e9)
        assert math.isclose(r['pair_tokens_B_raw_bench']*r['bench_to_production_factor'],r['pair_tokens_B'])
        assert math.isclose(r['greedy_pp']-r['greedy_full4k_pp'],-.390625)
        for budget,p in r['single_adaptive_budgets'].items():
            assert p is not None and p['K2']<=8 and p['quality_observed']
            assert p['timing']['seconds']<=int(budget)
            assert math.isclose(p['mean_samples'],2+(1-1336/2048)*(p['K2']-2))
        for t in r['tail_extrapolation']:
            assert t['seconds']<=t['budget_seconds'] and t['quality_extrapolated']==(t['K2']>8)
            assert math.isfinite(t['final_pp'])
        selected=r['single_adaptive_budgets']['160']
        assert math.isclose(r['final_pp'],r['greedy_pp']+selected['uplift_pp'])
        assert r['final_scenario_range_pp'][0]<r['final_pp']<r['final_scenario_range_pp'][1]
        assert r['hybrid']['K2']==[2,6] and len(r['hybrid_grid'])==6
        assert math.isclose(r['hybrid']['mean_large']+r['hybrid']['mean_C'],4.0419921875)
    models=[dict(name=str(i),K1=1,K2=k) for i,k in enumerate([3,5,4])]
    old=A.v7_time(models,1232)
    base=A.policy_timing(models,1232,1.,[1.,1.,1.])
    assert math.isclose(base['seconds'],old['conservative_seconds'])
    doubled=A.policy_timing(models,1232,2.,[2.,2.,2.])
    greedy_only=A.policy_timing(models,1232,2.,[1.,1.,1.])
    for label,x in base['runs'].items():
        y=doubled['runs'][label];z=greedy_only['runs'][label]
        assert y['fixed_seconds']==x['fixed_seconds']
        assert math.isclose(y['seconds'],x['fixed_seconds']+2*(x['seconds']-x['fixed_seconds']))
        assert math.isclose(z['seconds']-x['seconds'],x['greedy_seconds'])
    report=dict(passed=True,measured_T4_shapes=13,summary_sha256=b['summary_sha256'],
        saved_replay=True,journal_files=384,one_model_gt8_unobserved=True,
        calibrated_C_slope_and_tune_scale=True,separate_greedy_sample_ratios=True,
        unchanged_fixed_execution_overhead=True,rounded_v7_baseline_exact=True,
        raw_and_calibrated_tokens=True,GPU=False,push=False,new_execution=False)
    A.write(WORK/'checks/architecture-decision-cpu.json',report)
    print('architecture CPU checks passed; measured13 shapes, replay/calibration/timing; GPU/exec/push=0')


if __name__=='__main__': main()
