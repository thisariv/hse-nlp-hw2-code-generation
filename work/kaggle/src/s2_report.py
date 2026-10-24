"""A/M/B common-time comparisons and the pre-registered S2 final decision."""
from pathlib import Path
import exp26_eval as E
from exp26_common import read_json,write_json

def decision(common,timings,gates_passed=True):
    if not gates_passed:return {'status':'pending;27b gates not passed'}
    if not common:return {'status':'pending;no aligned A/M/B common point'}
    point=max(common,key=lambda p:p['target_seconds']);tag=point['tag']
    rank=sorted(point['models'],key=lambda m:point['models'][m]['splits']['heldout4k']['exec']['accuracy'],reverse=True)
    acc=lambda name:point['models'][name]['splits']['heldout4k']['exec']['accuracy']
    delta=100*(acc(rank[0])-acc(rank[1]));result={'point':tag,'A_point':point['A_point'],'leader':rank[0],'leader_gap_pp':delta}
    if delta>=2.:
        candidates=[rank[0]];rule='leader gap>=2pp and fulltest<=120s'
    else:candidates=[name for name in rank if 100*(acc(rank[0])-acc(name))<2.];rule='gap<2pp: minimum full inference time among near leaders'
    missing=[name for name in candidates if name not in timings.get(tag,{})]
    if missing:return dict(result,status='pending;missing corresponding checkpoint test timings',missing_models=missing)
    eligible=[name for name in candidates if timings[tag][name]['full_pipeline_seconds']<=120.]
    if not eligible:return dict(result,status='pending;no eligible candidate under120s',rule=rule)
    chosen=min(eligible,key=lambda name:timings[tag][name]['full_pipeline_seconds'])
    return dict(result,status='measured',final_model=chosen,rule=rule,full_test_seconds=timings[tag][chosen]['full_pipeline_seconds'])

def collect(working):
    working=Path(working);a_path=working/'a-baselines.json';a=read_json(a_path) if a_path.exists() else {}
    reports={name:read_json(working/folder/file) if (working/folder/file).exists() else {}
        for name,folder,file in [('M','gpu0-exp29','29-summary.json'),('B','gpu1-exp28','28-summary.json')]}
    for name,folder in [('M','gpu0-exp29'),('B','gpu1-exp28')]:
        # A kill between atomic metrics and the experiment summary still leaves a useful point.
        for path in (working/folder/'points').glob('*/training.json'):
            metrics=path.parent/'evaluation/metrics.json'
            if metrics.exists():
                point=read_json(path);point['metrics']=read_json(metrics)
                if all(s in point['metrics'].get('splits',{}) for s in ('heldout4k','old1024')):
                    point['evaluation_status']='complete'
                    reports[name].setdefault('points',{})[path.parent.name]=point
    common=[];comparisons=[];timings={}
    for tag in ('t01','t02','t03','t04'):
        models={};target=None;ap=None
        for name,report in reports.items():
            p=report.get('points',{}).get(tag)
            if p and p.get('time_aligned'):
                target=p['target_train_seconds'];ap=p['A_point'];models[name]=p['metrics']
        if ap in a:models['A']=a[ap]
        if len(models)<2:continue
        pairs={}
        names=sorted(models)
        for i,name in enumerate(names):
            for other in names[:i]:pairs[f'{name}_vs_{other}']={s:E.paired_mcnemar(models[name]['splits'][s]['exec'],models[other]['splits'][s]['exec']) for s in ('heldout4k','old1024')}
        row={'tag':tag,'A_point':ap,'target_seconds':target,'models':models,'McNemar':pairs};comparisons.append(row)
        if len(models)==3:common.append(row)
    for name,report in reports.items():
        for tag,tests in report.get('test_by_point',{}).items():
            if tests.get('selected'):timings.setdefault(tag,{})[name]=tests['selected']
    for p in (working/'a-test').glob('*/selected/timing.json'):
        timings.setdefault(p.parent.parent.name,{})['A']=read_json(p)
    fast_path=working/'gpu0-exp27b/27b-summary.json';fast=read_json(fast_path) if fast_path.exists() else {}
    passed=fast.get('status')=='complete' and fast.get('greedy_gate',{}).get('token_identical',False) and fast.get('heldout4k',{}).get('quality_gate',False)
    result={'comparisons':comparisons,'common_points':[r['tag'] for r in common],'test_timings':timings,
        'decision':decision(common,timings,passed),'27b_status':fast.get('status','missing'),
        'models':{name:{'status':r.get('status','missing'),'missing_points':r.get('missing_points',[])} for name,r in reports.items()}}
    write_json(working/'s2-comparison.json',result);return result
