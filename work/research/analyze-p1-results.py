"""CPU audit of P1 A/B, paired frozen L3 comparisons and truncation receipts."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORK = Path(__file__).resolve().parents[1]
RAW = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-p1/final'
L3 = WORK / 'results/kaggle-runs/nlp-hw02-kaggle-l3/final'
spec = importlib.util.spec_from_file_location('l2_analysis', WORK / 'research/analyze-l2-results.py')
l2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(l2)
BINS = [('<60', 0, 60), ('60–90', 60, 90), ('90–120', 90, 120), ('≥120', 120, math.inf)]
MODELS = {'L3-long': (L3, 'model-C-long2x__points__l2-62280'),
          'A': (RAW, 'model-pilot-a__points__long'),
          'B': (RAW, 'model-pilot-b__points__long-truncation')}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def paired(a, b):
    a = np.asarray(a, dtype=bool); b = np.asarray(b, dtype=bool)
    r = l2.paired(a, b); n = len(a)
    # For binary outcomes this multinomial resampling is exactly the distribution
    # of paired question-row bootstrap, with much less memory than index arrays.
    draws = np.random.default_rng(2026101401).multinomial(
        n, [r['losses']/n, 1-(r['gains']+r['losses'])/n, r['gains']/n], size=10000)
    delta = 100*(draws[:, 2]-draws[:, 0])/n
    r.update(CI95_pp=np.quantile(delta, [.025, .975]).tolist(),
             CI97_5_pp=np.quantile(delta, [.0125, .9875]).tolist(),
             method='paired row bootstrap, multinomial equivalent; 10000 draws, seed2026101401')
    return r


def subsets(rows):
    result = {'heldout4k': list(range(4096)), **l2.SPLIT}
    result.update({'bin-'+label: [r['id'] for r in rows if lo <= r['original_bpe_length'] < hi]
                   for label, lo, hi in BINS})
    return result


def load(name):
    folder, prefix = MODELS[name]
    read = lambda suffix: l2.read(folder / (prefix+'__'+suffix+'.json'))
    tr = read('training'); m = read('evaluation__metrics'); complete = read('complete')
    ck = read('checkpoint')
    assert complete['metrics'] == m and ck['sha256'] == complete['weights_sha256'] == m['weights_sha256']
    rows = {}
    for split, n in [('heldout4k', 4096), ('old991', 991)]:
        rows[split] = read('evaluation__'+split+'-per-example')
        assert [r['id'] for r in rows[split]] == list(range(n))
        assert all(r['correct'] == l2.matches(r['reference'], r['execution']) for r in rows[split])
        assert sum(r['correct'] for r in rows[split]) == m['splits'][split]['exec']['matches']
        assert math.isfinite(m['splits'][split]['CE']['CE'])
    scores = {s: l2.score(rows['heldout4k'], ids) for s, ids in subsets(rows['heldout4k']).items()}
    scores['old991'] = l2.score(rows['old991'], range(991))
    for s, ids in l2.SPLIT.items():assert scores[s]['matches'] == m['paired_subsets'][s]['matches']
    for label, lo, hi in BINS:
        value = m['splits']['heldout4k']['length_accuracy'][label]
        assert value['matches'] == scores['bin-'+label]['matches']
        assert value['denominator'] == scores['bin-'+label]['n']
    weight = folder / ('model-'+{'A':'pilot-a','B':'pilot-b','L3-long':'C-long2x'}[name]+
                       '__'+ck['file'].replace('/', '__'))
    assert weight.stat().st_size == ck['bytes'] and l2.sha(weight) == ck['sha256']
    data = dict(scores=scores, CE={s:m['splits'][s]['CE'] for s in rows}, training=tr,
                checkpoint=ck, K8=m['K8_tune2048'],
                restore_errors={s:m['splits'][s]['restore_errors'] for s in rows})
    return rows, data


def aligned(a, b):
    for s in a:
        for field in ['id', 'question_sha256', 'reference', 'original_bpe_length']:
            assert [r[field] for r in a[s]] == [r[field] for r in b[s]]


def comparisons(a, b):
    aligned(a, b)
    out = {s:paired([a['heldout4k'][i]['correct'] for i in ids],
                    [b['heldout4k'][i]['correct'] for i in ids])
           for s, ids in subsets(a['heldout4k']).items()}
    out['old991'] = paired([r['correct'] for r in a['old991']], [r['correct'] for r in b['old991']])
    out['length_bins_by_subset'] = {}
    for subset, ids in l2.SPLIT.items():
        out['length_bins_by_subset'][subset] = {}
        for label, lo, hi in BINS:
            chosen = [i for i in ids if lo <= a['heldout4k'][i]['original_bpe_length'] < hi]
            r = paired([a['heldout4k'][i]['correct'] for i in chosen], [b['heldout4k'][i]['correct'] for i in chosen])
            r.update(baseline_matches=sum(a['heldout4k'][i]['correct'] for i in chosen),
                     variant_matches=sum(b['heldout4k'][i]['correct'] for i in chosen))
            out['length_bins_by_subset'][subset][label] = r
    for key, predicate in [('gain_ids', lambda x,y:not x and y), ('loss_ids', lambda x,y:x and not y)]:
        out[key] = [i for i,(x,y) in enumerate(zip(a['heldout4k'], b['heldout4k'])) if predicate(x['correct'],y['correct'])]
    return out


def journal(folder, prefix, greedy):
    paths = sorted(folder.glob(prefix+'__evaluation__k8-tune2048__batch-*.json'))
    assert len(paths) == 64
    rows = [r for p in paths for r in l2.read(p)]
    assert [r['id'] for r in rows] == l2.SPLIT['tune2048']
    for r in rows:
        g = greedy[r['id']]
        assert r['question_sha256'] == g['question_sha256'] and r['reference'] == g['reference']
        assert r['greedy']['execution'] == g['execution'] and len(r['samples']['0.7']) == 8
        assert all(c['sampling_status']=='complete' and math.isfinite(c['mean_logprob']) for c in r['samples']['0.7'])
        v = l2.choose(r, dict(mode='vote',temperature=.7,k=8,greedy_vote=False))
        assert v['execution'] == r['vote_execution'] and l2.matches(r['reference'],v['execution']) == r['correct']
    summary = l2.read(folder/(prefix+'__evaluation__k8-tune2048__summary.json'))
    assert summary['complete'] and summary['rows']==summary['target_rows']==2048 and not summary['OOM_attempts']
    assert sum(r['correct'] for r in rows) == summary['matches']
    return rows


def voting(journals):
    main = journals['L3-main']; out = {}; raw = {}
    for name in ['L3-long', 'A', 'B']:
        for k in [4,8]:
            for mixed in [False,True]:
                for g in [False,True]:
                    if not mixed and k==4:continue
                    label=name+(f' mixed{k}+{k}' if mixed else ' single8')+('+greedy' if g else '')
                    correct=[];oracle=[]
                    for x,y in zip(main,journals[name]):
                        samples=x['samples']['0.7'][:k]+y['samples']['0.7'][:k] if mixed else y['samples']['0.7'][:k]
                        row=x if mixed else y
                        win=l2.choose({'greedy':row['greedy'],'samples':{'0.7':samples}},
                                      dict(mode='vote',temperature=.7,k=len(samples),greedy_vote=g))
                        correct.append(l2.matches(row['reference'],win['execution']))
                        oracle.append(any(l2.matches(row['reference'],c['execution']) for c in
                                          ([row['greedy']] if g else [])+samples))
                    raw[label]=correct
                    out[label]=dict(n=len(correct),matches=sum(correct),accuracy_pp=100*np.mean(correct),
                                    oracle_matches=sum(oracle),missed_oracle=sum(o and not c for o,c in zip(oracle,correct)))
    for name in ['A','B']:
        baseline='L3-long' if name=='A' else 'A'
        for label in list(out):
            if label.startswith(name+' '):
                other=baseline+label[len(name):]
                out[label]['paired_vs_'+baseline]=paired(raw[other],raw[label])
    return out


def truncation(training):
    totals=dict(presented=0,excluded=0,presented_tokens=0,retained_tokens=0)
    counts=np.zeros(4,dtype=np.int64);excluded=np.zeros(4,dtype=np.int64)
    lengths=[];source_lengths=[];ce_ex=[];ce_ret=[];graph=0;fallback=0
    p=RAW/'model-pilot-b__points__long-truncation__truncation.jsonl'
    with p.open() as h:
        for step,line in enumerate(h,1):
            r=json.loads(line);assert r['step']==step and r['presented']==128 and r['retained']==115
            assert len(r['excluded_indices'])==len(r['excluded_positions'])==13
            assert len(set(r['excluded_positions']))==13 and all(0<=i<128 for i in r['excluded_positions'])
            assert sum(r['counts'])==128 and sum(r['excluded'])==13
            assert np.bincount(np.digitize(r['excluded_target_lengths'],[53,79,105]),minlength=4).tolist()==r['excluded']
            counts+=r['counts'];excluded+=r['excluded'];lengths.extend(r['excluded_target_lengths'])
            source_lengths.extend(r['excluded_source_lengths']);ce_ex.append(r['excluded_mean_CE']);ce_ret.append(r['retained_mean_CE'])
            for key in totals:totals[key]+=r['presented']-r['retained'] if key=='excluded' else r[key]
            graph+=r['forward_mode']=='graph';fallback+=r['graph_OOM_fallback']
    assert step==31372
    ts=training['truncation_summary']
    assert all(v==ts[k] for k,v in totals.items())
    assert counts.tolist()==ts['counts'] and excluded.tolist()==ts['excluded_by_bin']
    assert graph==ts['graph_updates'] and fallback==ts['graph_OOM_fallbacks']
    return dict(updates=step,**totals,actual_fraction=totals['excluded']/totals['presented'],
                excluded_token_fraction=1-totals['retained_tokens']/totals['presented_tokens'],
                proxy_bins=[dict(label=l,presented=int(c),excluded=int(e),fraction=float(e/c))
                            for l,c,e in zip(['<53','53–79','79–105','≥105'],counts,excluded)],
                excluded_target_length_quantiles=np.quantile(lengths,[0,.25,.5,.75,.9,.99,1]).tolist(),
                excluded_source_length_quantiles=np.quantile(source_lengths,[0,.25,.5,.75,.9,.99,1]).tolist(),
                mean_excluded_CE=float(np.mean(ce_ex)),mean_retained_CE=float(np.mean(ce_ret)),
                graph_updates=graph,OOM_fallbacks=fallback,
                limitation='Presented length counts are compact target proxy bins; no original BPE train denominators/source bins')


def plot(report):
    fig, axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    colors={'L3-long':'#777777','A':'#2876b9','B':'#db7c1c'}
    for name,data in report['models'].items():
        axes[0].plot(range(4),[data['scores']['bin-'+label]['accuracy_pp'] for label,_,_ in BINS],
                     'o-',label=name,color=colors[name])
    axes[0].set(xticks=range(4),xticklabels=[b[0] for b in BINS],ylabel='Execution accuracy, %',xlabel='Original target BPE length')
    axes[0].legend()
    for j,(name,values) in enumerate(report['comparisons'].items()):
        for i,split in enumerate(['heldout4k','tune2048','eval2048','old991']):
            r=values[split];lo,hi=r['CI95_pp'];y=i+(j-.5)*.22
            axes[1].errorbar(r['delta_pp'],y,xerr=[[r['delta_pp']-lo],[hi-r['delta_pp']]],fmt='o',
                             color=colors['A' if j==0 else 'B'],label=name if i==0 else None)
    axes[1].axvline(0,color='gray',lw=1);axes[1].set(yticks=range(4),yticklabels=['4k','tune','eval','old'],xlabel='Paired delta, pp (95% CI)');axes[1].legend()
    tr=report['truncation'];axes[2].bar(range(4),[100*r['fraction'] for r in tr['proxy_bins']],color=colors['B'])
    axes[2].axhline(100*13/128,ls='--',color='gray');axes[2].set(xticks=range(4),xticklabels=[r['label'] for r in tr['proxy_bins']],ylabel='Excluded examples, %',xlabel='Compact target proxy length',ylim=(0,12))
    fig.savefig(WORK/'results/p1-comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,3.5),layout='constrained')
    for split in ['heldout4k','old991']:
        ax.plot(range(1,5),[r[split]['CE'] for r in report['reviewer']['validation']], 'o-',label=split)
    ax.set(xticks=range(1,5),xticklabels=['Periodic 1 (~3h)','Periodic 2 (~6h)','Periodic 3 (~9h)','Final cooldown'],ylabel='Reverse token CE',xlabel='Validation order (exact periodic timestamps unavailable)')
    ax.tick_params(axis='x',labelsize=8);ax.legend();fig.savefig(WORK/'results/p1-reviewer-ce.png',dpi=160);plt.close(fig)


def adaptive(journals):
    sys.path.insert(0, str(WORK/'kaggle/src'))
    from final_adaptive import replay_record
    result={}
    for name in ['L3-long','A','B']:
        for k in [6,8]:
            correct=[];early=0;samples=0
            for main,long in zip(journals['L3-main'],journals[name]):
                win,stop,count=replay_record(main['greedy'],main['samples']['0.7'],long['samples']['0.7'],k1=2,k2=k)
                correct.append(l2.matches(main['reference'],win['execution']));early+=stop;samples+=count
            label=name+f' adaptive2→{k}'
            result[label]=dict(matches=sum(correct),early_stop=early,mean_stochastic_samples=samples/2048,
                               correct_ids=[r['id'] for r,c in zip(journals[name],correct) if c])
    return result


def attach_reviewer(report):
    path=WORK/'results/p1-reviewer/summary.json'
    if not path.exists():return
    scored=l2.read(path)
    assert scored['passed'] and not scored['smoke'] and scored['scored_candidates']==34816
    assert len(scored['identity']['task_ids'])==2048 and scored['identity']['unique_pairs']==26291
    assert scored['identity']['weights_sha256']==l2.read(RAW/'model-reviewer__points__final__checkpoint.json')['sha256']
    for pool in scored['pools'].values():
        assert sorted(sum(pool['folds'],[]))==scored['identity']['task_ids']
        assert len(pool['folds'][0])==len(pool['folds'][1])==1024
        for selector in pool['selectors'].values():
            ci=selector['paired_vs_vote']
            assert selector['oof_matches']-pool['vote_matches']==ci['gains']-ci['losses']
        # Neutral weighting must replay saved vote correctness, including ties.
        for selector in pool['selectors'].values():
            if '0.0' in selector['full_tune_grid']:assert selector['full_tune_grid']['0.0']==pool['vote_matches']
    full=scored['pools']['mixed8+8+greedy'];joint=full['joint_cv']
    report['reviewer_scoring']=dict(summary_file=str(path.relative_to(WORK)),sha256=l2.sha(path),
        scored_candidates=scored['scored_candidates'],unique_pairs=scored['identity']['unique_pairs'],
        task_count=2048,subsample=False,CPU_precision=scored['identity']['precision'],
        forward_seconds=l2.read(path.parent/'progress.json')['seconds'],
        end_to_end_seconds_observed=path.stat().st_mtime-(path.parent/'identity.json').stat().st_mtime,
        requested_missed_tasks=248,reproduced_missed_tasks=len(full['missed_oracle_ids']),
        missed_ids=full['missed_oracle_ids'],full_pool_joint_cv=joint,
        caveat='OOF bootstrap conditional on trained model and fold-fitted rules; rule selection is not refitted inside bootstrap')
    fig,ax=plt.subplots(figsize=(7.5,4.4),layout='constrained')
    rows=list(full['selectors'].items())+[('joint CV',joint)]
    for i,(name,s) in enumerate(rows):
        r=s['paired_vs_vote'];lo,hi=r['CI95_pp']
        ax.errorbar(r['delta_pp'],i,xerr=[[r['delta_pp']-lo],[hi-r['delta_pp']]],fmt='o',color='#2876b9')
    ax.set(yticks=range(len(rows)),yticklabels=[n.replace('_logprob',' LP') for n,s in rows],
           xlabel='Out-of-fold delta vs mixed8+8+greedy, pp (95% CI)')
    ax.axvline(0,color='gray',lw=1);ax.invert_yaxis()
    fig.savefig(WORK/'results/p1-reviewer-selectors.png',dpi=160);plt.close(fig)


def main():
    records={};models={}
    for name in MODELS:records[name],models[name]=load(name)
    a,b=[models[n]['training'] for n in ['A','B']]
    for key in ['anchor_sha256','anchor_updates','anchor_cursor','start_fingerprints','stream_sha256','planned_cooldown_updates','cooldown_updates','schedule','peak_lr','floor_lr']:
        assert a[key]==b[key]
    assert a['cooldown_updates']==31372 and a['anchor_updates']==209144 and a['anchor_cursor']==20734592
    for name in ['A','B']:
        prefix=MODELS[name][1];restore=l2.read(RAW/(prefix+'__restore.json'))
        assert restore['passed'] and restore['fingerprints']==a['start_fingerprints'] and not restore['main_continued']
    l3tr=models['L3-long']['training']
    assert l3tr['anchor_updates']==a['anchor_updates'] and l3tr['anchor_cursor']==a['anchor_cursor']
    journals={n:journal(*MODELS[n],records[n]['heldout4k']) for n in MODELS}
    main_rows=l2.read(L3/'model-C-main__points__l2-62280__evaluation__heldout4k-per-example.json')
    journals['L3-main']=journal(L3,'model-C-main__points__l2-62280',main_rows)
    reviewer=l2.read(RAW/'model-reviewer__summary.json')
    reviewer['training']=l2.read(RAW/'model-reviewer__points__final__training.json')
    ck=l2.read(RAW/'model-reviewer__points__final__checkpoint.json')
    assert l2.sha(RAW/'model-reviewer__weights__final-fp16.pt')==ck['sha256']
    assert reviewer['status']=='complete' and len(reviewer['validation'])==4
    for v in reviewer['validation']:
        assert all(v[s]['complete'] and v[s]['rows']==n for s,n in [('heldout4k',4096),('old991',991)])
    session=l2.read(RAW/'session-summary.json');assert session['status']=='complete' and session['exit_codes']==[0,0]
    report=dict(passed=True,GPU=False,new_training_updates=0,session=session,
                models=models,comparisons={'A-L3':comparisons(records['L3-long'],records['A']),
                                          'B-A':comparisons(records['A'],records['B'])},
                voting=voting(journals),adaptive_replay=adaptive(journals),truncation=truncation(b),reviewer=reviewer,
                protocol=dict(controlled_AB='same stable state, stream, schedule, attempted updates; truncation differs',
                              A_vs_L3='duration AND shape changed; effects not separable',
                              confirmatory='eval2048 two planned comparisons, positive lower CI97.5 required',
                              caveat='Repeated heldout/tune/eval; bootstrap conditional on trained models, no independent final test'),
                missing=['Full stable PT deliberately absent locally; resume/restore checked from producer fingerprints, not rerun',
                         'Exact reviewer periodic CE timestamps/steps; curve uses validation order and nominal interval',
                         'No new K8 on eval2048, no fresh1024 result, no mixed/reviewer single-T4 final timer',
                         'No supplied list of 248 missed tasks; exact journals reproduce 247 for full 8+8+greedy'],
                inputs=[dict(file=str(p.relative_to(WORK)),sha256=l2.sha(p)) for folder in [RAW,L3]
                        for p in sorted(folder.glob('*.json')) if 'per-example' in p.name or 'batch-' in p.name or 'training' in p.name or 'checkpoint' in p.name])
    for name,comparison in report['comparisons'].items():
        comparison['CE_delta']={s:models['A' if name=='A-L3' else 'B']['CE'][s]['CE']-
                               models['L3-long' if name=='A-L3' else 'A']['CE'][s]['CE'] for s in ['heldout4k','old991']}
        comparison['confirmatory_gate_passed']=comparison['eval2048']['CI97_5_pp'][0]>0
    attach_reviewer(report)
    report['recommendation']=dict(final_cooldown='retain L3 50min; long duration/shape gain unconfirmed',
                                  loss_truncation=False,reviewer=False,
                                  inference='retain adaptive vote, K2+K2 escalation; K6 current budget, K8 only after full T4 timer',
                                  after_2026_10_17='priority: inference gate; constant-LR continuation of both L3 stable models, final50min cooldown; frozen eval/fresh once')
    plot(report);write(WORK/'results/p1-results.json',report)
    print(json.dumps({n:{s:r[s] for s in ['heldout4k','eval2048','tune2048','old991']} for n,r in report['comparisons'].items()},ensure_ascii=False))


if __name__=='__main__':main()
