"""CPU-only L2 journal audit, paired comparisons, frozen vote and descriptive curves.

No model forward pass, GPU, fresh/eval sampling or gold-dependent candidate choice.
The exact choose/fixed_split functions are loaded from the recorded source AST to
avoid importing the CUDA/training stack into this small NumPy/Matplotlib analysis.
"""
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = Path(__file__).resolve().parent.parent
RAW = WORK/'results/kaggle-runs/nlp-hw02-kaggle-l2/1'
SRC = WORK/'kaggle/src'
RESULT = WORK/'results/l2-results.json'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(4*1024**2), b''): h.update(block)
    return h.hexdigest()


def function(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


choose = function(SRC/'exp27_decode.py','choose',{'math':math})
fixed_split = function(SRC/'l2_sampler.py','fixed_split',{'hashlib':hashlib})
SPLIT = fixed_split()


def matches(reference, execution):
    return reference['status'] == execution['status'] == 'numeric' and math.isclose(
        reference['value'], execution['value'], rel_tol=1e-6, abs_tol=1e-6)


def paired(a, b):
    a = np.asarray(a,dtype=bool); b = np.asarray(b,dtype=bool)
    assert len(a) == len(b) and len(a)
    gains = int((~a & b).sum()); losses = int((a & ~b).sum()); n = len(a)
    delta = (gains-losses)/n
    half = 1.96*math.sqrt(max(0.,(gains+losses)/n-delta**2)/n)
    discordant = gains+losses
    if discordant:
        k = min(gains,losses)
        p = min(1.,2*sum(math.exp(math.lgamma(discordant+1)-math.lgamma(j+1)-
                      math.lgamma(discordant-j+1)-discordant*math.log(2)) for j in range(k+1)))
    else: p = 1.
    return {'n':n,'gains':gains,'losses':losses,'delta_pp':100*delta,
            'paired_95CI_pp':[100*(delta-half),100*(delta+half)],'McNemar_exact_p':p}


def score(rows, ids):
    chosen = [rows[i] for i in ids]
    return {'n':len(chosen),'matches':sum(r['correct'] for r in chosen),
            'accuracy_pp':100*sum(r['correct'] for r in chosen)/len(chosen)}


def load_point(model, tag):
    p = RAW/('model-'+model)/'points'/tag
    training = read(p/'training.json'); metrics = read(p/'evaluation/metrics.json')
    complete = read(p/'complete.json')
    assert complete['weights_sha256'] == metrics['weights_sha256']
    assert complete['training']['target_l2_main_seconds'] == training['target_l2_main_seconds']
    records = {}
    for split,n in [('heldout4k',4096),('old991',991)]:
        rows = read(p/'evaluation'/(split+'-per-example.json'))
        assert len(rows) == n and [r['id'] for r in rows] == list(range(n))
        assert all(r['correct'] == matches(r['reference'],r['execution']) for r in rows)
        assert sum(r['correct'] for r in rows) == metrics['splits'][split]['exec']['matches']
        records[split] = rows
    for split,ids in SPLIT.items():
        assert score(records['heldout4k'],ids)['matches'] == metrics['paired_subsets'][split]['matches']
    return records,point_row(model,tag,training,metrics,records)


def point_row(model, tag, training, metrics, records):
    return {'model':model,'tag':tag,'main_s':training['main_seconds'],
            'l2_main_s':training.get('l2_main_seconds',0.),'updates':training['anchor_updates'],
            'cooldown_s':training['cooldown_seconds'],'cooldown_updates':training['cooldown_updates'],
            'scores':{'heldout4k':score(records['heldout4k'],range(4096)),
                      'old991':score(records['old991'],range(991)),
                      **{name:score(records['heldout4k'],ids) for name,ids in SPLIT.items()}},
            'length_bins':metrics['splits']['heldout4k']['length_accuracy'],
            'CE':{split:metrics['splits'][split]['CE']['CE'] for split in records},
            'test':metrics['test'],'K8_tune2048':metrics.get('K8_tune2048'),
            'execution':{split:metrics['splits'][split]['exec']['generation_statuses'] for split in records},
            'restore_errors':{split:metrics['splits'][split]['restore_errors'] for split in records}}


def l1_point(reference):
    root = RAW/'l1-reference-live-v7'
    metrics = read(root/'model-C__points__t009000__evaluation__metrics.json')
    training = read(root/'model-C__points__t009000__training.json'); records = {}
    for split,n in [('heldout4k',4096),('old991',991)]:
        generated = read(root/('model-C__points__t009000__evaluation__'+split+'-generations.json'))
        execution = read(root/('model-C__points__t009000__evaluation__'+split+'-execution.json'))['rows']
        assert len(generated) == len(execution) == n
        rows = []
        for i in range(n):
            h = hashlib.sha256(generated[i]['question'].encode()).hexdigest()
            assert h == reference[split][i]['question_sha256']
            row = dict(reference[split][i],execution=execution[i],generated_code=generated[i]['generated_code'])
            row['correct'] = matches(row['reference'],row['execution']); rows.append(row)
        assert sum(r['correct'] for r in rows) == metrics['splits'][split]['exec']['matches']
        records[split] = rows
    return records,point_row('C (shared L1)','t009000',training,metrics,records)


def vote_journal(model, greedy):
    root = RAW/('model-'+model)/'points/l2-28800/evaluation/k8-tune2048'
    rows = [r for p in sorted(root.glob('batch-*.json')) for r in read(p)]
    index = {r['id']:r for r in rows}; assert len(index) == len(rows) == 2048
    assert sorted(index) == SPLIT['tune2048']
    for i,r in index.items():
        assert r['question_sha256'] == greedy[i]['question_sha256']
        assert r['reference'] == greedy[i]['reference'] and r['greedy']['execution'] == greedy[i]['execution']
        samples = r['samples']['0.7']; assert len(samples) == 8
        assert [s['sample'] for s in samples] == list(range(8))
        assert all(s['sampling_status']=='complete' and s['mean_logprob'] is not None for s in samples)
        # Selection receives no correctness/reference fields.
        clean = {'greedy':r['greedy'],'samples':r['samples']}
        chosen = choose(clean,{'mode':'vote','temperature':.7,'k':8,'greedy_vote':False})
        assert chosen['execution'] == r['vote_execution']
        assert matches(r['reference'],chosen['execution']) == r['correct']
    summary = read(root/'summary.json')
    assert summary['complete'] and sum(r['correct'] for r in rows) == summary['matches']
    return index


def voting(x,y):
    ids = SPLIT['tune2048']; raw = {}; rows = []
    baseline = [matches(x[i]['reference'],x[i]['greedy']['execution']) for i in ids]
    for model,k in [('main',4),('main',8),('long',4),('long',8),('mixed',4),('mixed',8)]:
        for add_greedy in [False,True]:
            total_k = 2*k if model=='mixed' else k
            name = (f'mixed K{k}+K{k}' if model=='mixed' else f'C-{model} K{k}')+(' +greedy' if add_greedy else '')
            outcomes = []; oracle = []; oracle_greedy = []; chosen_ids = []
            for i in ids:
                row = y[i] if model=='long' else x[i]
                samples = (x[i]['samples']['0.7'][:k]+y[i]['samples']['0.7'][:k] if model=='mixed'
                           else row['samples']['0.7'][:k])
                clean = {'greedy':row['greedy'],'samples':{'0.7':samples}}
                chosen = choose(clean,{'mode':'vote','temperature':.7,'k':total_k,'greedy_vote':add_greedy})
                outcomes.append(matches(row['reference'],chosen['execution']))
                oracle.append(any(matches(row['reference'],s['execution']) for s in samples))
                oracle_greedy.append(oracle[-1] or matches(row['reference'],row['greedy']['execution']))
                chosen_ids.append(hashlib.sha256(json.dumps(chosen['execution'],sort_keys=True).encode()).hexdigest())
            raw[name] = outcomes
            rows.append({'name':name,'K_samples':total_k,'greedy_vote':add_greedy,
                         'greedy_tie_anchor':'C-long2x' if model=='long' else 'C-main',
                         'K_doubled':model=='mixed' and k==8,'matches':sum(outcomes),'n':2048,
                         'accuracy_pp':100*sum(outcomes)/2048,'oracle_samples_matches':sum(oracle),
                         'oracle_with_anchor_greedy_matches':sum(oracle_greedy),
                         'oracle_matches':sum(oracle_greedy if add_greedy else oracle),
                         'against_C_main_greedy':paired(baseline,outcomes),
                         'chosen_execution_sha256':hashlib.sha256(''.join(chosen_ids).encode()).hexdigest()})
    for r in rows:
        r['against_C_main_K8'] = paired(raw['C-main K8'],raw[r['name']])
        r['against_C_main_K8_plus_greedy'] = paired(raw['C-main K8 +greedy'],raw[r['name']])
        r['against_C_long_K8'] = paired(raw['C-long K8'],raw[r['name']])
    return {'baseline_C_main_greedy_matches':sum(baseline),
            'baseline_C_long_greedy_matches':sum(matches(y[i]['reference'],y[i]['greedy']['execution']) for i in ids),
            'rows':rows}


def fit_curves(points):
    curves = {}; fig,ax = plt.subplots(figsize=(8.2,4.8))
    for model,color in [('C-main','#2463ae'),('C-long2x','#d26c24')]:
        p = [points[0]]+[r for r in points if r['model']==model]
        t = np.array([r['main_s']/3600 for r in p]); a = np.array([r['scores']['heldout4k']['accuracy_pp'] for r in p])
        # Exclude the final 50-minute cooldown from the short-cooldown fit.
        logs = np.polyfit(np.log(t[:3]),a[:3],1)
        recent_slope = (a[2]-a[1])/math.log(t[2]/t[1])
        error_fit = np.polyfit(np.log(t[:3]),np.log(100-a[:3]),1)
        def power(tt): return 100-math.exp(float(error_fit[1]))*tt**float(error_fit[0])
        forecasts = []
        for n in [2,4,8,16]:
            deltas = {'log_all_short':float(logs[0]*math.log((t[3]+n)/t[3])),
                      'log_last_short':float(recent_slope*math.log((t[3]+n)/t[3])),
                      'power_error_no_floor':power(t[3]+n)-power(t[3])}
            forecasts.append({'extra_main_hours':n,'gain_pp_by_fit':deltas,
                              'accuracy_pp_scenarios':{k:float(a[3]+v) for k,v in deltas.items()}})
        residuals = {'log_all_short':float(a[3]-(logs[0]*np.log(t[3])+logs[1])),
                     'log_last_short':float(a[3]-(a[2]+recent_slope*np.log(t[3]/t[2]))),
                     'power_error_no_floor':float(a[3]-power(t[3]))}
        curves[model] = {'hours':t.tolist(),'accuracy_pp':a.tolist(),
                         'log_all_short_slope_pp_per_ln_hour':float(logs[0]),
                         'log_last_short_slope_pp_per_ln_hour':float(recent_slope),
                         'error_power_exponent':float(error_fit[0]),'forecasts':forecasts,
                         'final_gain_over_l2_21600_pp':float(a[3]-a[2]),
                         'final_residual_pp_not_causal_cooldown_estimate':residuals,
                         'fit_points':['t009000','l2-10800','l2-21600'],
                         'forecast_policy':'Anchor increments at actual final score; no statistical CI; same final cooldown assumed'}
        ax.plot(t,a,'o-',color=color,label=model)
        grid = np.linspace(t[0],t[-1]+8,150)
        ax.plot(grid,logs[0]*np.log(grid)+logs[1],'--',color=color,alpha=.45)
        future = np.linspace(t[-1],t[-1]+8,80)
        ax.plot(future,a[-1]+logs[0]*np.log(future/t[-1]),':',color=color)
        ax.scatter(t[-1],a[-1],s=120,facecolors='none',edgecolors=color)
    ax.set(xlabel='Накопленное main-время, часы (без cooldown)',
           ylabel='Greedy heldout4096, %',
           title='L2: измерения и описательные логарифмические сценарии')
    fig.text(.1,.025,'Штрих: fit по трём точкам с cooldown ≈15 мин. Кольцо: cooldown 50 мин.\n'
             'Пунктир: сценарий ещё 8 ч main, не CI. Историческая L1 общая для обеих веток.',fontsize=8)
    ax.grid(alpha=.2);ax.legend(loc='upper left');fig.tight_layout(rect=(0,.09,1,1))
    fig.savefig(WORK/'results/l2-learning-curve.png',dpi=180)
    plt.close(fig)
    return curves


def main():
    audit = {'GPU_used':False,'model_forwards':0,'fresh_read_or_evaluated':False,
             'source_SHA':{name:sha(SRC/name) for name in ['exp27_decode.py','l2_sampler.py']},
             'CI':'unadjusted paired normal 95%; variance of {-1,0,+1}; protocol l2_compare.py',
             'candidate_choice_receives_gold':False,'note':'Execution correctness recomputed from saved numeric journals; programs not rerun'}
    producer_source = {}
    for name in ['exp27_decode.py','l2_sampler.py','l2_eval.py','l1_eval.py']:
        assert (RAW/'src'/name).read_bytes() == (SRC/name).read_bytes(),name
        producer_source[name] = sha(RAW/'src'/name)
    audit['pinned_producer_source_matches_local'] = producer_source
    manifest = read(RAW/'l2-output-manifest.json')['files']; verified = 0; omitted = []
    for f in manifest:
        p = RAW/f['file']
        if p.is_file():
            assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],f['file']; verified += 1
        else: omitted.append(f['file'])
    audit['producer_manifest_verified_files'] = verified
    audit['producer_manifest_omitted_files'] = omitted
    assert all(not name.endswith(('.json','.jsonl','.log','.txt','.csv')) for name in omitted)
    live = RAW/'live-v11'
    recover = read(live/'recover-manifest.json')
    live_audit = {'raw_sha_verified':0,'gzip_sha_reconstructed':0,'kernel_content_equal':0,
                  'omitted':[],'kernel_content_differs':[]}
    for f in recover['files']:
        p = live/f['file']
        if not p.exists(): p = live/f['file'].removesuffix('.gz')
        if not p.exists(): live_audit['omitted'].append(f['file']);continue
        data = p.read_bytes()
        if f['gzip']:
            # API file inventory exposes .gz payloads expanded into .json/.jsonl.
            # Producer used level1/mtime0. Account for platform gzip OS byte.
            compressed = bytearray(gzip.compress(data,compresslevel=1,mtime=0));valid=False
            for os_byte in [3,255,19]:
                compressed[9] = os_byte
                if len(compressed)==f['bytes'] and hashlib.sha256(compressed).hexdigest()==f['sha256']:
                    valid=True;break
            assert valid,f['file'];live_audit['gzip_sha_reconstructed']+=1
        else:
            assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256']
            live_audit['raw_sha_verified']+=1
        original = RAW/f['original_path']
        if original.exists():
            if original.read_bytes()==data:live_audit['kernel_content_equal']+=1
            else:live_audit['kernel_content_differs'].append(f['original_path'])
    assert len(live_audit['omitted'])==3 # two full stable states and weighted order
    assert set(live_audit['kernel_content_differs'])=={'l2-live-sync-status.json','session-summary.json'}
    audit['live_recovery_manifest'] = live_audit
    all_records = {}; points = []; pair_results = {}
    for tag in ['l2-10800','l2-21600','l2-28800']:
        a,ar = load_point('C-main',tag); b,br = load_point('C-long2x',tag)
        assert abs(ar['l2_main_s']-br['l2_main_s'])<5 and abs(ar['cooldown_s']-br['cooldown_s'])<1
        for split in a:
            assert [(r['id'],r['question_sha256'],r['reference'],r['original_bpe_length']) for r in a[split]] == [
                (r['id'],r['question_sha256'],r['reference'],r['original_bpe_length']) for r in b[split]]
        all_records[('C-main',tag)] = a; all_records[('C-long2x',tag)] = b
        points.extend([ar,br]); h = a['heldout4k']
        subsets = {'heldout4k':list(range(4096)),**SPLIT,
                   'ge90':[i for i in range(4096) if h[i]['original_bpe_length']>=90],
                   'lt60':[i for i in range(4096) if h[i]['original_bpe_length']<60]}
        for split,ids in SPLIT.items():
            subsets[split+'-ge90'] = [i for i in ids if h[i]['original_bpe_length']>=90]
            subsets[split+'-lt60'] = [i for i in ids if h[i]['original_bpe_length']<60]
        for label,lo,hi in [('<60',0,60),('60-90',60,90),('90-120',90,120),('>=120',120,10**9)]:
            subsets['bin-'+label] = [i for i in range(4096) if lo<=h[i]['original_bpe_length']<hi]
        pair_results[tag] = {name:paired([a['heldout4k'][i]['correct'] for i in ids],
                                           [b['heldout4k'][i]['correct'] for i in ids]) for name,ids in subsets.items()}
        pair_results[tag]['old991'] = paired([r['correct'] for r in a['old991']],[r['correct'] for r in b['old991']])
    l1_records,l1 = l1_point(all_records[('C-main','l2-28800')]); points.insert(0,l1)
    for model in ['C-main','C-long2x']:
        for tag in ['l2-10800','l2-21600','l2-28800']:
            r = all_records[(model,tag)]
            for p in points:
                if p['model']==model and p['tag']==tag:
                    p['against_L1_heldout4k'] = paired([x['correct'] for x in l1_records['heldout4k']],
                                                       [x['correct'] for x in r['heldout4k']])
    x = vote_journal('C-main',all_records[('C-main','l2-28800')]['heldout4k'])
    y = vote_journal('C-long2x',all_records[('C-long2x','l2-28800')]['heldout4k'])
    vote = voting(x,y)
    audit['K8_rows'] = [len(x),len(y)];audit['samples_per_model'] = [16384,16384]
    audit['stored_K8_winner_and_correctness_recomputed'] = True
    # Frozen criteria: tune drives the decision; stored greedy eval is descriptive.
    gates = {}
    for tag,r in pair_results.items():
        d = r['tune2048']['delta_pp']; long = r['tune2048-ge90']['delta_pp']; short = r['tune2048-lt60']['delta_pp']
        gates[tag] = {'tune_overall_pp':d,'tune_ge90_pp':long,'tune_lt60_pp':short,
                      'overall_ge_0_5':d>=.5,'long_ge_3':long>=3,'short_ge_minus_0_5':short>=-.5,
                      'all_main_gates':d>=.5 and long>=3 and short>=-.5}
    mixed = next(r for r in vote['rows'] if r['name']=='mixed K4+K4')
    gates['specialist_same_K8'] = {'mixed_vs_main_K8':mixed['against_C_main_K8'],
                                  'tune_delta_ge_0_5':mixed['against_C_main_K8']['delta_pp']>=.5}
    curves = fit_curves(points)
    final = {p['model']:p for p in points if p['tag']=='l2-28800'}
    scale = 1919/2048
    sample_seconds = {model:final[model]['K8_tune2048']['seconds'] for model in final}
    # No separate sampler/exec timer was persisted inside vote_k8. These are
    # conservative additive scenarios from its measured aggregate and greedy test.
    main_greedy = final['C-main']['test']['full_pipeline_seconds']
    long_greedy = final['C-long2x']['test']['full_pipeline_seconds']
    single = max(sample_seconds['C-main']*scale+main_greedy,
                 sample_seconds['C-long2x']*scale+long_greedy)+3
    ideal_mixed = .5*sum(sample_seconds.values())*scale+main_greedy+5
    sequential_mixed = sum(sample_seconds.values())*scale+main_greedy+5
    timing = {'hardware':'one T4, sequential two-model generation; no two-GPU throughput credit',
              'measured_K8_tune_seconds':sample_seconds,'scale_1919_over_2048':scale,
              'separate_sampler_and_sample_execution_seconds_saved':False,
              'reserve_factor':1.15,'extra_cold_IO_single_s':3,'extra_cold_IO_mixed_s':5,
              'single_K8':{'additive_nominal_s':single,'with_15pct_reserve_s':single*1.15,'180s_verified':False},
              'mixed_K4_plus_K4':{'K4_relative_cost_scenarios':[.5,1.],
                                 'nominal_s':[ideal_mixed,sequential_mixed],
                                 'with_15pct_reserve_s':[ideal_mixed*1.15,sequential_mixed*1.15],
                                 '180s_verified':False},
              'mixed_K8_plus_K8':{'additive_nominal_s':sequential_mixed,
                                 'with_15pct_reserve_s':sequential_mixed*1.15,'180s_verified':False},
              'note':'K4 proportionality and test length/timeout distribution are assumptions; aggregate includes sample execution, journals and cleanup. Greedy used for ties even without +greedy; actual one-T4 full pipeline remains required.'}
    session = read(RAW/'session-summary.json')
    result = {'kernel_ref':'thisarivson/nlp-hw02-kaggle-l2/2',
              'session':{k:session[k] for k in ['status','worker_status','exit_codes','wall_seconds','failed_workers']},
              'audit':audit,'points':points,'paired_long_minus_main':pair_results,
              'voting_tune2048':vote,'long_weighting_gates':gates,'learning_curves':curves,
              'test_timing_scenarios':timing}
    RESULT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Wrote',RESULT.name)
    for p in points:
        print(p['model'],p['tag'],'heldout',p['scores']['heldout4k']['matches'],
              'tune/eval',p['scores']['tune2048']['matches'],p['scores']['eval2048']['matches'])
    for r in vote['rows']:
        print(r['name'],r['matches'],'vs greedy',r['against_C_main_greedy']['gains'],r['against_C_main_greedy']['losses'],
              'vs K8',r['against_C_main_K8']['gains'],r['against_C_main_K8']['losses'],r['against_C_main_K8']['paired_95CI_pp'])
    print('Gates',json.dumps(gates))


if __name__=='__main__':main()
