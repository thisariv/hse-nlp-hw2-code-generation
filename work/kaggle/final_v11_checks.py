"""Frozen v11 CPU gates; never generates new tune samples or changes v8 receipts."""
import argparse, ast, csv, hashlib, importlib.util, json, os
from pathlib import Path
import subprocess, sys, tempfile, time
ROOT=Path(__file__).resolve().parent; WORK=ROOT.parent
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from build_final_inference_v11 import BUNDLE, NAME, default_config
from final_ensemble import replay_record
from final_split import fixed_split
from final_v8_checks import read, digest, INPUTS
B6=WORK/'results/kaggle-runs/nlp-hw02-b6/1'
DOWNLOAD=WORK/'results/kaggle-runs/nlp-hw02-b6/v10-pinned-1'

def embedded(path):
    nb=read(path)
    code=next(c['source']for c in nb['cells']if c['cell_type']=='code')
    return ast.literal_eval(next(n.value for n in ast.parse(code).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='SOURCES'))

def offline():
    from final_v7_analysis import load_all, independent_select
    journals,receipts=load_all();config=default_config();models=config['models'];names=[m['name']for m in models]
    manifest=read(B6/'b1-output-manifest.json');entries={r['file']:r for r in manifest['files']}
    assert digest((B6/'b1-output-manifest.json').read_bytes())=='dd00395d133cd5c005261bf39b41c9a967dbd36e9523b838ecfa865aae218dfe'
    for name,gpu in [('C-long2x-B6',1),('W-B6',0)]:
        records=[];hashes={}
        for p in sorted((B6/f'model-GPU{gpu}/points/b6-final/evaluation/k8-tune2048').glob('batch-*.json')):
            h=digest(p.read_bytes());rel=str(p.relative_to(B6));assert entries[rel]['sha256']==h and entries[rel]['bytes']==p.stat().st_size
            hashes[str(p.relative_to(WORK))]=h;records.extend(read(p))
        assert len(hashes)==64 and len(records)==2048
        journals[name]={r['id']:r for r in records};assert sorted(journals[name])==fixed_split()['tune2048'];receipts[name]=hashes
    spec=importlib.util.spec_from_file_location('b3_audit',WORK/'research/analyze-b3-results.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    oracle=mod.Replay(journals,names[0],names)
    bits,stops,expected=oracle.evaluate([m['K2']for m in models],[1]*3)
    winners=[];prod_stops=0
    for i,want,bit in zip(fixed_split()['tune2048'],expected,bits):
        anchor=journals[names[0]][i]
        for name in names:
            assert journals[name][i]['question_sha256']==anchor['question_sha256']
            assert journals[name][i]['reference']==anchor['reference']
            assert len(journals[name][i]['samples']['0.7'])==8
        win,stop,count=replay_record(anchor['greedy'],{n:journals[n][i]['samples']['0.7']for n in names},models)
        assert (win['code'],win['execution'])==(want['code'],want['execution']),i
        candidates=[c for m in models for c in journals[m['name']][i]['samples']['0.7'][:m['K1' if stop else 'K2']]]
        independent=independent_select(anchor['greedy'],candidates)
        assert (win['code'],win['execution'])==(independent['code'],independent['execution']),i
        assert mod.correct(anchor['reference'],win['execution'])==bit
        prod_stops+=stop;winners.append(dict(id=i,code=win['code'],execution=win['execution']))
    assert sum(bits)==1551 and prod_stops==stops
    return dict(passed=True,rows=2048,matches=sum(bits),early_stops=stops,winner_code_execution_mismatches=0,
        independent_winner_mismatches=0,winner_sha256=digest(json.dumps(winners,ensure_ascii=False,sort_keys=True).encode()),
        policy=config,orchestrator_fixed_policy=True,no_grid_search=True,journal_sha256=receipts)

def weights():
    import torch
    from final_inference_v7 import load,sha
    from final_session_v8 import weight_path
    from final_v7_analysis import P1
    torch.set_num_threads(1);out={}
    downloaded=read(WORK/'checks/final-inference-v10-download.json')['files']
    manifest=read(DOWNLOAD/'b1-output-manifest.json')
    assert (DOWNLOAD/'b1-output-manifest.json').read_bytes()==(B6/'b1-output-manifest.json').read_bytes()
    entries={r['file']:r for r in manifest['files']}
    for m in default_config()['models']:
        rec=next(r for r in downloaded if r['sha256']==m['sha256'])
        p=WORK/rec['local'];assert sha(p)==m['sha256']
        if m['name']!='pilot-b':
            assert p==Path(weight_path(DOWNLOAD,m));assert entries[m['relative_path']]['sha256']==m['sha256']
            assert entries[m['relative_path']]['bytes']==p.stat().st_size
        else:assert p.read_bytes()==(P1/'model-pilot-b__weights__long-truncation-fp16.pt').read_bytes()
        payload=torch.load(p,map_location='cpu',weights_only=True)
        assert payload['format']=='nlp-hw02-model-only-v1' and payload['optimizer_included']is False
        assert all(t.dtype==torch.float16 and torch.isfinite(t).all()for t in payload['model'].values())
        model=load(p,torch.device('cpu'));state=model.state_dict()
        assert state.keys()==payload['model'].keys()
        assert all(state[k].shape==v.shape and torch.equal(state[k],v.float())for k,v in payload['model'].items())
        out[m['name']]=dict(sha256=sha(p),spec=payload['model_spec'],keys=len(state),
            FP16_resident_bytes=sum(t.numel()*t.element_size()for t in payload['model'].values()),
            QK_norm_keys=sum('q_norm'in k or 'k_norm'in k for k in state),strict_load_exact=True,finite=True)
        if m['name']=='W-B6':assert out[m['name']]['QK_norm_keys']==72 and payload['model_spec']==dict(d_model=768,heads=12,encoder_layers=6,decoder_layers=6,swiglu_dim=2048,vocab=8195,qk_norm='layernorm')
        del model,payload,state
    return dict(passed=True,models=out,resident_bytes=sum(x['FP16_resident_bytes']for x in out.values()),B6_version1_manifest_exact=True,P1_v8_payload_exact=True)

def static():
    import nbformat
    nb=read(BUNDLE/NAME);nbformat.validate(nbformat.from_dict(nb));sources=embedded(BUNDLE/NAME)
    old=embedded(ROOT/'notebooks/nlp-hw02-ensemble-inference-v10.ipynb')
    assert sources==old,'All runtime sources frozen from v10'
    identity=read(BUNDLE/'source-identity.json')
    run=next(c['source']for c in nb['cells']if c['cell_type']=='code'and 'ENSEMBLE_CONFIG='in c['source'])
    config=ast.literal_eval(next(n.value for n in ast.parse(run).body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='ENSEMBLE_CONFIG'))
    assert config==identity['ensemble_config']==default_config(), 'Frozen orchestrator policy differs'
    previous=read(ROOT/'final-v10-config.json')
    changed=[]
    for i,(before,after)in enumerate(zip(previous['models'],config['models'])):
        changed.extend((i,k,before.get(k),after.get(k))for k in before.keys()|after.keys()if before.get(k)!=after.get(k))
    assert changed==[(1,'K2',5,8),(2,'K2',6,8)]
    assert {k:v for k,v in previous.items()if k!='models'}=={k:v for k,v in config.items()if k!='models'}
    for name in ['final_ensemble.py','final_adaptive.py','final_vote.py']:
        assert (ROOT/'src'/name).read_text()==sources[name]
    assert identity['notebook_sha256']==digest((BUNDLE/NAME).read_bytes())
    assert (BUNDLE/NAME).read_bytes()==(ROOT/'notebooks'/NAME).read_bytes()
    for c in nb['cells']:
        if c['cell_type']=='code':compile(c['source'],'v11-cell','exec')
    for name,src in sources.items():
        compile(src,name,'exec');assert digest(src.encode())==identity['embedded_source_sha256'][name]
        assert not any(x in src for x in ['/Users/','~/.kaggle','KAGGLE_KEY'])
    meta=read(BUNDLE/'kernel-metadata.json');assert meta['id']=='friend1/nlp-hw02-ensemble-inference-v11'
    assert meta['is_private'] and not meta['enable_internet'] and meta['enable_gpu'] and not meta['enable_tpu']
    assert meta['machine_shape']=='NvidiaTeslaT4'
    assert meta['kernel_sources']==['friend1/nlp-hw02-b6/1','friend1/nlp-hw02-kaggle-p1/3']
    assert meta['dataset_sources']==['thisarivson/nlp-hw02-26b-inputs']
    lim=read(BUNDLE/'session-limits.json');assert lim['session_timeout_seconds']==720 and lim['cold_test_runs']==lim['tune_runs']==1
    from final_mounts import resolve_cache_mount
    from final_session_v8 import weight_path
    with tempfile.TemporaryDirectory()as d:
        root=Path(d)
        for m in default_config()['models']:
            owner,slug=m['dataset'].split('/');p=root/'notebooks'/owner/slug/m['relative_path'];p.parent.mkdir(parents=True,exist_ok=True);p.touch()
            mount=resolve_cache_mount(root,m['dataset'],m['source_kind']);assert weight_path(mount,m)==str(p)
    return dict(passed=True,nbformat=True,all_20_v10_embedded_sources_exact=True,only_config_changes=changed,private=True,single_T4_GPU0=True,version_pinned_mounts=True,
        fixed_orchestrator_config_exact=True,guards=[160,166,170,172],session_timeout_seconds=720,maximum_T4_pair_quota_seconds=1440,notebook_sha256=identity['notebook_sha256'])

def pipeline():
    import torch
    from final_core import SimpleTransformer,EOS
    from final_vote import validate_answers
    static();sources=embedded(BUNDLE/NAME)
    with tempfile.TemporaryDirectory(prefix='v11-mixed-cold-')as d:
        tmp=Path(d);modules=tmp/'src';modules.mkdir();models=[]
        for n,s in sources.items():(modules/n).write_text(s)
        for i,m in enumerate(default_config()['models']):
            spec=dict(d_model=24 if i==2 else 16,heads=2,encoder_layers=1,decoder_layers=1,swiglu_dim=32,vocab=8195,max_positions=512,qk_norm='layernorm'if i==2 else False)
            model=SimpleTransformer(**spec,dropout=0.).eval()
            with torch.no_grad():model.head.weight.zero_();model.head.bias.fill_(-1000.);model.head.bias[EOS]=1000.
            weight=tmp/(m['name']+'.pt');torch.save(dict(format='nlp-hw02-model-only-v1',optimizer_included=False,model_spec=spec,model=model.half().state_dict()),weight)
            models.append(dict(m,path=str(weight),sha256=digest(weight.read_bytes())))
        rows=tmp/'rows.json';rows.write_text(json.dumps([dict(question='Compute 2 + 3.'),dict(question='What is 7 minus 2?')]))
        cfg=dict(default_config(),models=models,rows=str(rows),tokenizer=str(INPUTS/'tokenizer.json'),output=str(tmp/'out'),device='cpu',expected_rows=2,cap=16,sequence_budget=21,greedy_sequence_budget=2,budget_seconds=40.)
        p=tmp/'config.json';p.write_text(json.dumps(cfg))
        r=subprocess.run([sys.executable,str(modules/'final_runner.py'),str(p)],capture_output=True,text=True,timeout=45,env=dict(os.environ,FINAL_PROCESS_START=str(time.monotonic())))
        assert r.returncode==0,r.stderr+(tmp/'out/worker.log').read_text()
        report=read(tmp/'out/summary.json');validate_answers(tmp/'out/answers.jsonl',2)
        assert report['status']=='complete' and report['adaptive']['mean_sample_count']==17
        assert report['deadline_guard']['executed_sample_counts']==[17,17]
        assert {b['phase']for b in report['generation_batches']if b['stage']==2}=={'pilot-b','W-B6'}
        assert report['winner_reexecution']['complete'] and not report['winner_reexecution']['mismatch_rows']
        return dict(passed=True,embedded_cold_mixed_width_QK_SHA_load=True,stage2_P1_and_W=True,K2=[1,8,8],seconds=report['full_pipeline_seconds'])

def regressions():
    import final_v10_checks as v10
    results={}
    for name in ['offline','weights','static','pipeline','regressions']:
        results[name]=getattr(v10,name)();print('v10',name,'passed',flush=True)
    return dict(passed=True,v10_receipts_not_overwritten=True,checks=results)

def e1():
    """Additional read-only production parity on the frozen fresh E1 pool."""
    spec=importlib.util.spec_from_file_location('v11_e1_audit',WORK/'research/e1-analysis.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from e1_policies import materialize,predict,grid
    policy=next(p for p in grid()if p['name']=='budget-1-8-8')
    sketch=read(WORK/'results/e1-v11-policy.json')
    assert digest((WORK/'results/e1-selection.json').read_bytes())==sketch['source_selection_sha256']
    assert read(WORK/'results/e1-selection.json')['primary']==policy['name']
    raw=WORK/'results/kaggle-runs/nlp-hw02-e1-ensembles/4'
    models=default_config()['models'];out={}
    for split,expected in [('tune2048',1550),('eval2048',1574)]:
        rows,summary,receipts=module.load(raw,split);wins=[];matches=stops=0
        for row in rows:
            anchor=materialize(row,'C/g',1,True)[0]
            samples={m['name']:materialize(row,short+'/0.7',8)for m,short in zip(models,('C','P','W'))}
            winner,stop,count=replay_record(anchor,samples,models)
            oracle,oracle_stop,used,_=predict(row,policy)
            assert (winner['code'],winner['execution'],stop,count)==(oracle['code'],oracle['execution'],oracle_stop,len(used)-1)
            matches+=module.correct(row['reference'],winner);stops+=stop
            wins.append((row['id'],winner['code'],winner['execution']))
        assert matches==expected
        out[split]=dict(rows=2048,matches=matches,early_stops=stops,winner_code_execution_mismatches=0,
            winner_sha256=digest(json.dumps(wins,ensure_ascii=False).encode()),journal_sha256=receipts)
    return dict(passed=True,read_only_E1=True,policy=policy,splits=out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['offline','weights','static','pipeline','regressions','e1']);args=p.parse_args()
    result=globals()[args.mode]()
    (WORK/f'checks/final-inference-v11-{args.mode}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items()if k not in ['journal_sha256','checks','policy']},ensure_ascii=False,indent=2))
