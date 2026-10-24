#!/usr/bin/env python3
"""Offline audit of pinned S2 v2: hashes, scores, saved programs and B states.

Run in the homework CPU environment with torch, tokenizers, numpy and sympy.
No training, CUDA generation, network requests or test gold are used.
"""
import collections
import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / 'kaggle/src'))
from exp26_common import read_json, sha_file, write_json, json_hash
from exp26_fast_exec import run_chunks
import exp26_exec as N


def same(a, b):
    return a['status'] == b['status'] and (a['status'] != 'numeric' or
        math.isclose(a['value'], b['value'], rel_tol=1e-6, abs_tol=1e-6))


def paired(score, control):
    assert score['eligible_indices'] == control['eligible_indices']
    a, b = set(score['match_indices']), set(control['match_indices'])
    gains, losses = len(a-b), len(b-a)
    n = gains + losses
    p = min(1., 2 * sum(math.comb(n, i) for i in range(min(gains, losses)+1)) / (1 << n)) if n else 1.
    return dict(gains=gains, losses=losses, discordant=n, exact_two_sided_p=p,
                delta_percentage_points=100*(gains-losses)/score['denominator'])


def main():
    import torch
    from tokenizers import Tokenizer
    import exp26_core as C
    import exp26_eval as E
    import exp26_train as T
    import exp27_decode as D
    import s2_exec
    s2_exec.install()
    started = time.perf_counter()
    root = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s2/2'
    s1 = WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1'
    b_root = root/'gpu1-exp28'
    report = dict(date='2026-10-11', new_training_updates=0, new_GPU_runs=0,
                  root=str(root.relative_to(WORK)), checks={})
    receipt = read_json(root/'download-version.json')
    assert receipt['ref']=='thisarivson/nlp-hw02-kaggle-s2/2' and receipt['version_label']=='v2' and not receipt['errors']
    listing = read_json(root/'download-list.json')['files']
    assert len(listing)==receipt['files']
    inventory = []
    for row in listing:
        p=root/row['file'];assert p.is_file(), row['file']
        inventory.append(dict(file=row['file'],bytes=p.stat().st_size,sha256=sha_file(p)))
    report['download'] = dict(receipt, bytes=sum(r['bytes'] for r in inventory), inventory=inventory)
    manifest = read_json(root/'s2-output-manifest.json')
    byte_differences=[];missing_manifest=[]
    for row in manifest['files']:
        p=root/row['file']
        if not p.is_file():missing_manifest.append(row['file']);continue
        if p.stat().st_size!=row['bytes']:
            byte_differences.append(dict(file=row['file'],manifest_bytes=row['bytes'],downloaded_bytes=p.stat().st_size))
    assert missing_manifest==['__notebook__.ipynb'] and not byte_differences, (missing_manifest,byte_differences)
    report['manifest'] = dict(files=len(manifest['files']), all_experiment_bytes='passed',
        byte_differences=byte_differences,missing_manifest=missing_manifest,
        reason='Kaggle internal __notebook__.ipynb is absent from pinned SDK output listing; notebook export is visible in kernel log',
        SHA_scope='local inventory plus producer SHA receipts; output manifest itself has bytes only',
        ZIPs=[p.relative_to(root).as_posix() for p in root.rglob('*.zip')])
    assert not report['manifest']['ZIPs']
    sha_receipts=0
    for p in root.rglob('*.sha.json'):
        target=p.with_name(p.name.replace('.sha.json','.json'))
        assert sha_file(target)==read_json(p)['sha256'];sha_receipts+=1
    report['checks']['candidate_SHA_receipts'] = sha_receipts
    starter=WORK/'results/checkpoints/26b-training-inputs-d0-initial-and-evaluation.zip'
    with zipfile.ZipFile(starter) as z:
        assert z.testzip() is None
        sets={s:json.loads(z.read(s+'.json')) for s in ('heldout4k','old1024')}
        tok=Tokenizer.from_str(z.read('tokenizer.json').decode())
        input_identity=read_json(s1/'gpu0-a26/input-identity.json')
        for name,digest in input_identity['files'].items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    refs={s:read_json(b_root/'references'/f'{s}-reference-execution.json') for s in sets}
    for s in sets:assert refs[s]==read_json(s1/'gpu0-a26/references'/f'{s}-reference-execution.json')
    lengths={s:[len(x.ids) for x in tok.encode_batch([r['code'] for r in rows])] for s,rows in sets.items()}
    predicates=[lambda n:n<60,lambda n:60<=n<90,lambda n:90<=n<120,lambda n:n>=120]
    b_summary=read_json(b_root/'28-summary.json');a=read_json(root/'a-baselines.json')
    for tag,metrics in a.items():assert metrics==read_json(s1/f'gpu0-a26/points/{tag}/evaluation/metrics.json')
    metrics_by_tag={'D0':read_json(b_root/'D0-kaggle/metrics.json')}
    metrics_by_tag.update({tag:p['metrics'] for tag,p in b_summary['points'].items()})
    codes=[r['code'] for rows in sets.values() for r in rows]
    evaluations=[];static_pairs={}
    for tag,metrics in metrics_by_tag.items():
        directory=b_root/'D0-kaggle' if tag=='D0' else b_root/'points'/tag/'evaluation'
        assert read_json(directory/'metrics.json')==metrics
        for s,rows in sets.items():
            d=metrics['splits'][s];gens=read_json(directory/f'{s}-generations.json')
            execution=read_json(directory/f'{s}-execution.json')
            answers=[json.loads(x) for x in (directory/f'{s}-answers.jsonl').read_text().splitlines()]
            assert len(gens)==len(execution['rows'])==len(rows)==len(answers)
            assert sha_file(directory/f'{s}-answers.jsonl')==d['generation_time']['answers_sha256']
            for i,(g,r,ans) in enumerate(zip(gens,rows,answers)):
                assert g['id']==ans['id']==i and g['question']==ans['question']==r['question']
                assert g['generated_code']==ans['code']==tok.decode([t-3 for t in g['generated_token_ids'] if t>=3],skip_special_tokens=False)
                assert g['eos_emitted']==(2 in g['generated_token_ids']) and len(g['generated_token_ids'])<=512
            assert N.compare_exec(refs[s],execution)==d['exec']
            eligible,matched=set(d['exec']['eligible_indices']),set(d['exec']['match_indices'])
            for entry,predicate in zip(d['length_accuracy'].values(),predicates):
                indices={i for i in eligible if predicate(lengths[s][i])}
                assert entry['denominator']==len(indices) and entry['matches']==len(indices&matched)
            assert d['CE']['useful_target_tokens']==sum(lengths[s])+len(rows)
            if tag!='D0':
                assert metrics['mean_generated_length'][s]==sum(len(g['generated_token_ids']) for g in gens)/len(gens)
                controls={'A':a[b_summary['points'][tag]['A_point']]['splits'][s]['exec'],
                          'D0':metrics_by_tag['D0']['splits'][s]['exec']}
                if tag!='t01':controls['previous']=metrics_by_tag[f't{int(tag[1:])-1:02d}']['splits'][s]['exec']
                for name,control in controls.items():
                    p=paired(d['exec'],control)
                    original=metrics['McNemar_against_A'][s] if name=='A' else d['McNemar_against_'+name]
                    for k in p:assert math.isclose(p[k],original[k],rel_tol=1e-10,abs_tol=1e-300),(tag,s,name,k)
                    static_pairs[f'{tag}/{s}/vs_{name}']=p
            evaluations.append((tag,s,gens,execution,d))
            codes.extend(g['generated_code'] for g in gens)
    for tag,p in b_summary['points'].items():
        target=p['target_train_seconds'];assert p['time_aligned']==(abs(p['actual_train_seconds']-target)<=max(.02*target,5.))
        assert p['alignment_error_seconds']==p['actual_train_seconds']-target
    report['checks']['B_order_tokens_answers_scores_bins_lengths_CE_denominators_alignment']='passed'
    report['checks']['A_baselines_and_reference_receipts_exact']='passed'
    report['McNemar'] = static_pairs
    grids=[]
    candidate_directories={p.parent for p in (root/'gpu0-exp27b').rglob('batch-*.json') if p.stem.removeprefix('batch-').isdigit()}
    for directory in sorted(candidate_directories):
        rows=[]
        for p in sorted(directory.glob('batch-*.json')):
            if p.stem.removeprefix('batch-').isdigit():rows.extend(read_json(p))
        rows.sort(key=lambda r:r['id']);assert [r['id'] for r in rows]==list(range(len(rows)))
        flat=[c for r in rows for c in [r['greedy']]+r['samples'].get('0.7',[])]
        for r in rows:
            g=r['greedy'];assert tok.decode([t-3 for t in g['token_ids'] if t>=3],skip_special_tokens=False)==g['code']
        tag=(directory.parent if directory.name=='candidates' else directory).relative_to(root).as_posix()
        grids.append((tag,rows,flat))
        codes.extend(c['code'] for c in flat)
    heldout=next(rows for tag,rows,flat in grids if tag=='gpu0-exp27b/heldout4k')
    assert len(heldout)==4096
    control=read_json(s1/'gpu0-a26/points/p050000/evaluation/heldout4k-generations.json')
    for r,source in zip(heldout,sets['heldout4k']):assert r['question']==source['question']
    mismatches=[]
    for i,(r,c) in enumerate(zip(heldout,control)):
        x,y=r['greedy']['token_ids'],c['generated_token_ids']
        if x!=y:
            first=next((j for j,(v,w) in enumerate(zip(x,y)) if v!=w),min(len(x),len(y)))
            mismatches.append(dict(index=i,first_token_position=first,new_length=len(x),S1_length=len(y)))
    fast=read_json(root/'gpu0-exp27b/27b-summary.json')
    assert [r['index'] for r in mismatches]==fast['greedy_gate']['mismatch_indices']
    report['greedy_gate'] = dict(rows=4096,token_identical=not mismatches,mismatches=mismatches,
        CUDA_root_cause='not established; candidate question batch128 differs from saved greedy batch512; FP16/backend sensitivity is a hypothesis')
    unique=list(dict.fromkeys(codes));print('CPU replay',len(codes),'rows /',len(unique),'unique',flush=True)
    replay=run_chunks(unique,workers=2,chunk_rows=2000,wall_limit=3.,cpu_limit=2.,
        progress=lambda done,total,sec:print('CPU',done,'/',total,'seconds',round(sec,1),flush=True))
    lookup=dict(zip(unique,replay['rows']))
    report['CPU_execution']=dict(replay['summary'],total_saved_program_rows=len(codes),unique_programs=len(unique))
    report['reference_replay']={}
    newrefs={s:[lookup[r['code']] for r in rows] for s,rows in sets.items()}
    for s in sets:report['reference_replay'][s]=dict(numeric=sum(x['status']=='numeric' for x in newrefs[s]),
        different_indices=[i for i,(x,y) in enumerate(zip(newrefs[s],refs[s]['rows'])) if not same(x,y)])
    report['B_execution_replay']={}
    for tag,s,gens,execution,d in evaluations:
        fresh=[lookup[g['generated_code']] for g in gens];score=N.compare_exec(newrefs[s],fresh)
        report['B_execution_replay'][tag+'/'+s]=dict(matches=score['matches'],denominator=score['denominator'],
            different_match_indices=sorted(set(score['match_indices'])^set(d['exec']['match_indices'])),
            different_execution_indices=[i for i,(x,y) in enumerate(zip(fresh,execution['rows'])) if not same(x,y)],
            saved_execution_summary=execution['summary'])
    for tag,rows,flat in grids:
        for c in flat:c['execution']=lookup[c['code']]
    report['27b_saved_generation_replay']={tag:dict(rows=len(rows),programs=len(flat),
        statuses=dict(collections.Counter(c['execution']['status'] for c in flat)),saved_GPU_execution=False)
        for tag,rows,flat in grids}
    cfg=dict(name='vote-t0.7-k16-g0',mode='vote',temperature=.7,k=16,greedy_vote=False)
    winners=[D.choose(r,cfg) for r in heldout]
    score=N.compare_exec(newrefs['heldout4k'],[r['execution'] for r in winners])
    greedy=N.compare_exec(newrefs['heldout4k'],[r['greedy']['execution'] for r in heldout])
    report['27b_posthoc_CPU_diagnostic']=dict(
        eligible_for_registered_selection=False,reason='registered greedy token-identity gate failed before GPU execution/quality gate/test',
        k16_matches=score['matches'],k16_denominator=score['denominator'],
        k16_tolerance_would_pass=abs(score['matches']-2707)<=45,
        fast_greedy_matches=greedy['matches'],McNemar_vs_fast_greedy=paired(score,greedy),
        saved_GPU_execution=False,full_test_timings=[],selected_k=None,
        fast_greedy_vs_S1=paired(greedy,a['p050000']['splits']['heldout4k']['exec']))
    report['B_checkpoints']=[]
    for path in sorted(b_root.rglob('*.pt')):
        p=torch.load(path,map_location='cpu',weights_only=True)
        model=p['model'] if 'model' in p else p;tensor_sha=E.state_digest(model)
        if 'model' in p:assert tensor_sha==p['model_tensor_sha256']
        item=dict(file=path.relative_to(root).as_posix(),bytes=path.stat().st_size,sha256=sha_file(path),
                  model_tensor_sha256=tensor_sha,parameters=sum(t.numel() for t in model.values()))
        if path.name=='initial-B.pt':
            assert item['sha256']==b_summary['latest']['identity']['initial_sha256']
            item.update(role='seed42-initial',optimizer_included=False)
        elif path.parent.name=='weights':
            r=read_json(b_root/'points'/path.stem/'checkpoint.json')
            assert item['sha256']==r['sha256'] and item['bytes']==r['bytes'] and tensor_sha==r['model_tensor_sha256']
            assert p['dtype']=='FP32' and not p['optimizer_included']
            item.update(role='evaluated-weights',optimizer_included=False)
        elif path.parent.name=='checkpoints':
            r=read_json(path.with_suffix('.pt.json'))
            assert item['sha256']==r['sha256'] and item['bytes']==r['bytes'] and tensor_sha==r['model_tensor_sha256']
            s,h=p['training'],p['experiment_history']
            assert p['scheduler']==T.schedule_state(s,p['config'])
            assert {'python','numpy','torch_cpu','torch_cuda'}<=set(p['rng'])
            steps={int(v['step']) for v in p['optimizer']['state'].values()}
            assert steps=={s['main_updates']+s['branch_updates']-s['AMP_skips']}
            assert s['data_cursor']==128*(s['main_updates']+s['branch_updates'])
            state_fields=('phase','main_updates','branch_updates','branch_length','data_cursor','last_lr','AMP_skips','counts')
            history_fields=('compute_updates','AMP_skips','train_seconds','checkpoint_io_seconds','all_counts',
                            'completed_points','finite_loss','finite_successful_grad','finite_loss_and_grad')
            item.update(role=r['role'],training={k:s[k] for k in state_fields},
                        experiment_history={k:h[k] for k in history_fields},optimizer_steps=sorted(steps),scaler=p['scaler'])
        report['B_checkpoints'].append(item);del p,model
    final=next(r for r in report['B_checkpoints'] if r.get('role')=='final-point')
    model_only=next(r for r in report['B_checkpoints'] if r['file'].endswith('weights/t04.pt'))
    latest=next(r for r in report['B_checkpoints'] if r.get('role')=='session-stop')
    assert final['model_tensor_sha256']==model_only['model_tensor_sha256']!=latest['model_tensor_sha256']
    report['checks']['B_model_only_full_final_latest_optimizer_scheduler_scaler_rng_cursor']='passed'
    report['all_B_match_sets_and_execution_exact']=all(not x['different_match_indices'] and not x['different_execution_indices'] for x in report['B_execution_replay'].values())
    report['all_reference_values_and_finite_indices_exact']=all(not x['different_indices'] for x in report['reference_replay'].values())
    report['status']='passed' if report['all_B_match_sets_and_execution_exact'] and report['all_reference_values_and_finite_indices_exact'] else 'differences; see replay'
    report['wall_seconds']=time.perf_counter()-started
    write_json(WORK/'checks/kaggle-s2-v2-cpu-audit.json',report)
    print('AUDIT',report['status'],'seconds',round(report['wall_seconds'],1),'posthoc k16',score['matches'],flush=True)


if __name__=='__main__':main()
