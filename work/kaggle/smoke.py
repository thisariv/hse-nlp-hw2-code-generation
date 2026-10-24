#!/usr/bin/env python3
"""Offline CPU smoke: real tiny train/branch/resume, two processes, sharedKV and numeric votes."""
import argparse,json,os,pathlib,shutil,subprocess,sys,tempfile,time
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import sha,write,extract

WORKER='''import argparse,json,os,sys,time\nfrom pathlib import Path\ndef main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int);p.add_argument('--config');a=p.parse_args();cfg=json.loads(Path(a.config).read_text());sys.path.insert(0,cfg['src'])
 import torch,numpy as np
 import exp26_train as T
 from exp26_common import read_json,write_json
 from kaggle_runtime import install_runtime_adapters,copy_output
 torch.set_num_threads(1);torch.set_num_interop_threads(1);install_runtime_adapters()
 class Cache:
  manifest_sha='tiny-cache'
  def __len__(self):return 80
  def __getitem__(self,i):return np.array([1,4,5,2],dtype=np.int16),np.array([6,7],dtype=np.int16)
 root=Path(cfg['working_root'])/('gpu'+str(a.gpu));root.mkdir()
 order=np.arange(80,dtype=np.uint32);meta={'file':{'sha256':'tiny-order'}};cache=Cache();device=torch.device('cpu')
 protocol=cfg['tiny_protocol'];protocol['branch_lengths']={int(k):v for k,v in protocol['branch_lengths'].items()};spec=cfg['spec']
 callback=lambda p,t,prev:{'CPU_fixture_only':True,'point':t}
 first=T.run_training(cache,order,meta,Path(cfg['tiny_inputs']),root,root,'tiny-code',device,callback,smoke=True,cfg=protocol,spec=spec,stop_main_updates=6,checkpoint_seconds=0,max_updates_this_call=3)
 attached=Path(cfg['working_root'])/('attached-output'+str(a.gpu));copy_output(root,attached)
 nextroot=Path(cfg['working_root'])/('resumed'+str(a.gpu));copy_output(attached,nextroot)
 second=T.run_training(cache,order,meta,Path(cfg['tiny_inputs']),nextroot,nextroot,'tiny-code',device,callback,smoke=True,cfg=protocol,spec=spec,stop_main_updates=6,checkpoint_seconds=0)
 assert second['complete'] and second['state']['main_updates']==6 and second['history']['compute_updates']==8
 # Repeat monolithic path to establish complete model/optimizer/scaler/RNG/cursor parity.
 mono=Path(cfg['working_root'])/('monolithic'+str(a.gpu))
 third=T.run_training(cache,order,meta,Path(cfg['tiny_inputs']),mono,mono,'tiny-code',device,callback,smoke=True,cfg=protocol,spec=spec,stop_main_updates=6,checkpoint_seconds=0)
 s=torch.load(nextroot/second['latest']['file'],weights_only=True);m=torch.load(mono/third['latest']['file'],weights_only=True)
 for k in s['model']:assert torch.equal(s['model'][k],m['model'][k])
 for key in s['optimizer']['state']:
  for field in s['optimizer']['state'][key]:
   if isinstance(s['optimizer']['state'][key][field],torch.Tensor):assert torch.equal(s['optimizer']['state'][key][field],m['optimizer']['state'][key][field])
 assert s['scheduler']==m['scheduler'] and torch.equal(s['rng']['torch_cpu'],m['rng']['torch_cpu'])
 assert not list(nextroot.glob('checkpoints/periodic-*.pt'))
 write_json(Path(cfg['working_root'])/('worker'+str(a.gpu)+'-check.json'),{'complete':True,'pid':os.getpid(),'CPU':True,'visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'updates_per_trajectory':8,'resume_full_state_parity':True,'branches':2})
 print('CPU WORKER PASSED',a.gpu,flush=True)
if __name__=='__main__':main()
'''

def main():
    import torch,numpy as np,pyarrow as pa,pyarrow.parquet as pq
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import exp26_protocol as P
    import exp26_core as C
    import exp26_eval as E
    import exp26_train as T
    from exp26_input import import_starter
    from exp26_common import read_json,write_json,file_record,json_hash
    from cache_extension import run as extension
    import exp28_capacity as B
    import exp27_decode as D
    from kaggle_runtime import install_runtime_adapters
    from session_runner import launch
    tmp=pathlib.Path(tempfile.mkdtemp(prefix='nlp-kaggle-cpu-'));checks=[]
    def passed(name,**extra):checks.append(dict(check=name,passed=True,**extra));print('PASSED',name,flush=True)
    try:
        inputs=tmp/'inputs';import_starter(ROOT.parent/'results/checkpoints/26a-cache-inputs-tokenizer-and-exclusions.zip',inputs,'cache-inputs')
        # Exact starter wrapped as a Kaggle TAR, plus multiple Drive ZIP fragments and traversal rejection.
        from kaggle_runtime import prepare_starter
        import tarfile,zipfile
        wrapped=tmp/'dataset';wrapped.mkdir()
        with tarfile.open(wrapped/'payload.tar','w') as t:
            t.add(ROOT.parent/'results/checkpoints/26a-cache-inputs-tokenizer-and-exclusions.zip',arcname='26a-cache-inputs-tokenizer-and-exclusions.zip')
        prepare_starter(wrapped,tmp/'from-kaggle','cache-inputs')
        from kaggle_io import resolve_dataset
        nested=tmp/'kaggle-input/datasets/owner/nlp-hw02-26c-inputs';nested.mkdir(parents=True)
        assert resolve_dataset(tmp/'kaggle-input','nlp-hw02-26c-inputs')==nested
        assert sha(tmp/'from-kaggle/tokenizer.json')==P.TOKENIZER_SHA
        for i in (0,1):
            z=tmp/f'drive-part{i}.zip'
            with zipfile.ZipFile(z,'w') as f:f.writestr(f'26-token-cache/shards/part0000{i}/array.txt',str(i))
            extract(z,tmp/'merged-drive')
        assert len(list((tmp/'merged-drive').rglob('array.txt')))==2
        with zipfile.ZipFile(tmp/'bad.zip','w') as z:z.writestr('../escape','x')
        try:extract(tmp/'bad.zip',tmp/'rejected')
        except ValueError:pass
        else:raise AssertionError('ZIP traversal accepted')
        passed('Kaggle input wrapper + multiZIP Drive merge + safe path rejection')
        # Two real tiny parquet shards42/43 with numeric, blocked, nonnumeric and protected rows.
        protected=json.loads((inputs/'protected-questions.json').read_text())['groups']['heldout4k'][0]
        records=[{'question':f'tiny arithmetic example {i}','code':f'def solve():\n    return {i}+2'} for i in range(30)]
        records += [{'question':protected,'code':'def solve():\n return 1'},
                    {'question':'unsafe fixture','code':'def solve():\n return open("x")'},
                    {'question':'string fixture','code':'def solve():\n return "a"'}]
        specs=[];raw=tmp/'raw';raw.mkdir()
        for shard in (42,43):
            p=raw/f'part-{shard:05d}.parquet';pq.write_table(pa.Table.from_pylist(records),p)
            specs.append({'shard':shard,'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
        download=lambda spec,root:raw/spec['path']
        manifest=extension(inputs,tmp/'extension','tiny-code',specs,True,download,2)
        digest=sha(tmp/'extension/cache-manifest.json')
        again=extension(inputs,tmp/'extension','tiny-code',specs,True,download,2)
        assert sha(tmp/'extension/cache-manifest.json')==digest and manifest==again
        from exp26_cache import TokenCache
        cache=TokenCache(tmp/'extension',require_minimum=False)
        assert all(isinstance(a,np.memmap) for part in cache.parts for a in part.values())
        assert [s['shard'] for s in manifest['shards']]==[42,43] and manifest['train_protected_question_overlaps']==0
        passed('CPU extension excludes protected rows, preserves schema/SHA and shard resume',cache_rows=len(cache))
        tiny=tmp/'tiny';tiny.mkdir();shutil.copyfile(inputs/'tokenizer.json',tiny/'tokenizer.json')
        spec=dict(d_model=16,heads=2,encoder_layers=1,decoder_layers=1,swiglu_dim=32,vocab=8192,max_positions=64)
        torch.manual_seed(42);model=C.SimpleTransformer(**spec,dropout=0.);torch.save(model.state_dict(),tiny/'initial.pt')
        cfg=dict(P.TRAIN_PROTOCOL);cfg.update(batch=2,main_updates=6,warmup_updates=1,main_cooldown_start=4,main_cooldown_updates=2,branch_lengths={2:1,4:1})
        worker=tmp/'cpu_worker.py';worker.write_text(WORKER)
        session=tmp/'session';start=time.monotonic()
        config={'working_root':str(session),'src':str(ROOT/'src'),'tiny_inputs':str(tiny),'spec':spec,'tiny_protocol':cfg,
                'session_start_monotonic':start,'deadline_monotonic':start+90,'budget_seconds':90}
        report=launch(config,worker,smoke=True,heartbeat_seconds=2)
        assert report['exit_codes']==[0,0],[(p.name,p.read_text()[-3000:]) for p in session.glob('*.log')]
        workers=[read_json(session/f'worker{i}-check.json') for i in (0,1)]
        assert workers[0]['pid']!=workers[1]['pid'] and all(w['visible_devices']=='' for w in workers)
        with __import__('zipfile').ZipFile(session/'s1-results.zip') as z:
            assert z.testzip() is None
            for entry in json.loads(z.read('zip-manifest.json'))['files']:
                assert __import__('hashlib').sha256(z.read(entry['file'])).hexdigest()==entry['sha256']
        passed('two independent CPU subprocesses + attached output resume + short branches',full_model_updates=0,tiny_updates=32)
        # Test actual portable state boundary with source runtime fields, keeping hashes/content strict.
        connected=session/'attached-output0';receipt=read_json(connected/'latest.json');cp=connected/receipt['file']
        payload=torch.load(cp,weights_only=True);payload['identity']['torch']='source-runtime-fixture';receipt['identity']['torch']='source-runtime-fixture'
        torch.save(payload,cp);receipt['sha256']=sha(cp);receipt['bytes']=cp.stat().st_size;write_json(connected/'latest.json',receipt)
        install_runtime_adapters();new_identity=dict(receipt['identity'],torch=str(torch.__version__))
        manager=T.Checkpoints(connected,connected,new_identity);restored=manager.load(receipt,torch.device('cpu'))
        assert (connected/'runtime-migration.json').exists();del restored
        bad=dict(new_identity,order_sha256='wrong')
        try:T.Checkpoints(connected,connected,bad)
        except AssertionError:pass
        else:raise AssertionError('Wrong order accepted')
        passed('runtime migration records provenance but rejects changed order')
        # Real sample decoder: grouped/broadcast crossKV equals the original cached path.
        model.eval();src=torch.tensor([[1,4,5,2],[1,5,6,2]])
        ctx=D.EncoderContext(model,src,8);grouped=D.SampleDecoder(ctx,4);original=C.CachedDecoder(model,src,8)
        for token in (torch.tensor([1,1]),torch.tensor([4,5]),torch.tensor([6,7])):
            l1=grouped.step(token[:,None].expand(-1,4));l2=original.step(token)
            assert torch.allclose(l1,l2[:,None].expand(-1,4,-1),atol=2e-6,rtol=1e-5),(l1-l2[:,None]).abs().max()
        assert ctx.cross[0][0].shape[1]==1
        passed('shared encoder/crossKV broadcast agrees with original cached logits')
        toyrows=[{'question':'add two and three','code':'def solve():\n return 5'},{'question':'subtract one','code':'def solve():\n return 2'}]
        model_path=tiny/'model.pt';torch.save(model.state_dict(),model_path)
        hooks=[0]
        # Generate2 temps and compare resume files; controls from real tiny greedy.
        controls,timing=E.generate_one(model_path,toyrows,tiny/'tokenizer.json',torch.device('cpu'),tmp/'control','tiny',spec,batch=2,cap=8,half=False)
        candidates,gen=D.generate_grid(model_path,toyrows,tiny/'tokenizer.json',torch.device('cpu'),tmp/'decode',spec,{.7:4,.5:2},8,8,controls=controls)
        repeated,replay=D.generate_grid(model_path,toyrows,tiny/'tokenizer.json',torch.device('cpu'),tmp/'decode',spec,{.7:4,.5:2},8,8,controls=controls)
        assert candidates==repeated and replay['new_batches']==0
        execution=D.execute_candidates(candidates,tmp/'candidate-exec');assert execution['programs']==14
        # Numeric fixtures cover fallback, nontransitive clusters, greedy tie, logprob tie and greedy vote.
        def candidate(v,lp=-1):return {'code':str(v),'length':2,'mean_logprob':lp,'execution':{'status':'numeric','value':v} if v is not None else {'status':'non_number'}}
        row={'greedy':candidate(10,-3),'samples':{'0.7':[candidate(20,-2),candidate(20.0000001,-2),candidate(10,-5),candidate(10,-5)]}}
        cfgvote={'mode':'vote','temperature':.7,'k':4,'greedy_vote':False}
        assert D.choose(row,cfgvote)['execution']['value']==10 # tie -> greedy-cluster, even lower logprob
        row['greedy']=candidate(None);assert D.choose(row,cfgvote)['execution']['value'] in (20,20.0000001)
        fallback=dict(cfgvote,mode='fallback');assert D.choose(row,fallback)['execution']['value']==20
        tie={'greedy':candidate(30),'samples':{'0.7':[candidate(10,-3),candidate(20,-1),candidate(10,-3),candidate(20,-1)]}}
        assert D.choose(tie,cfgvote)['execution']['value']==20
        gv={'greedy':candidate(10),'samples':{'0.7':[candidate(10),candidate(20),candidate(20)]}}
        c3=dict(cfgvote,k=3);assert D.choose(gv,c3)['execution']['value']==20
        assert D.choose(gv,dict(c3,greedy_vote=True))['execution']['value']==10
        full=D.time_test(model_path,toyrows,tiny/'tokenizer.json',torch.device('cpu'),tmp/'full-timer',dict(cfgvote,k=2),spec,batch_rows=8,cap=8)
        assert full['full_pipeline_seconds']>=full['components']['load_generate_seconds'] and full['executed_candidates']==6
        baseline={'exec':{'matches':100},'timing':{'full_pipeline_seconds':20},'McNemar_against_greedy':{'delta_percentage_points':0}}
        alternative={'exec':{'matches':162},'timing':{'full_pipeline_seconds':120},'McNemar_against_greedy':{'delta_percentage_points':62/4096*100}}
        assert D.select_config([baseline,alternative])[0]
        alternative['McNemar_against_greedy']['delta_percentage_points']=61/4096*100
        assert not D.select_config([baseline,alternative])[0]
        passed('real sampling/resume/exec/full timer; nested prefixes + vote/fallback/ties and1.5pp/120s fixtures')
        # Actual B controller tiny configuration with time branches and calibration; no full B parameters/updates.
        bcfg=B.config(batch=2);bcfg.update(main_updates=10,warmup_updates=1,main_cooldown_start=8,main_cooldown_updates=2)
        order=np.arange(len(cache),dtype=np.uint32);order_meta={'file':{'sha256':'tiny-order'}}
        targets=[{'A_point':'fixture1','seconds':.4,'actual_pairs':4},{'A_point':'fixture2','seconds':1.,'actual_pairs':8}]
        # Deterministic clock fixture around real optimizer updates; CPU timings are not T4 estimates.
        actual_train_chunk=T.train_chunk
        def fixed_clock(*args,**kwargs):
            meta=args[4];before=meta['train_seconds'];log=actual_train_chunk(*args,**kwargs)
            meta['train_seconds']=before+.1*args[9]
            log['CPU_clock_fixture']=True
            return log
        T.train_chunk=fixed_clock
        def evaluate(cp,tag,prev,target):return {'CPU_fixture_only':True,'splits':{'heldout4k':{'exec':{'accuracy':.5}}}}
        r=B.run(cache,order,order_meta,tiny,tmp/'B','tiny-port',torch.device('cpu'),evaluate,time.monotonic()+20,
                cfg=bcfg,spec=spec,smoke=True,targets=targets,calibration=2,max_compute=3)
        connected_b=tmp/'B-attached';shutil.copytree(tmp/'B',connected_b)
        r2=B.run(cache,order,order_meta,tiny,connected_b,'tiny-port',torch.device('cpu'),evaluate,time.monotonic()+20,
                 cfg=bcfg,spec=spec,smoke=True,targets=targets,calibration=2,max_compute=20)
        T.train_chunk=actual_train_chunk
        assert len(r2['points'])==2 and (connected_b/'500-step-calibration.json').exists()
        p=B.projected_plan(.3,B.TARGETS);assert len(p)==3 and p[0]['projected_main_updates']>500
        am={p['A_point']:{'splits':{'heldout4k':{'exec':{'accuracy':.47}}}} for p in r2['points'].values()}
        assert B.decision(r2['points'],am,{'full_pipeline_seconds':119})['accepted_B']
        assert not B.decision(r2['points'],am,{'full_pipeline_seconds':121})['accepted_B']
        import copy
        borderline=copy.deepcopy(r2['points'])
        for point in borderline.values():point['metrics']['splits']['heldout4k']['exec']['accuracy']=.47+81/4096
        assert not B.decision(borderline,am,{'full_pipeline_seconds':119})['accepted_B']
        for point in borderline.values():point['metrics']['splits']['heldout4k']['exec']['accuracy']=.47+82/4096
        assert B.decision(borderline,am,{'full_pipeline_seconds':119})['accepted_B']
        for point in borderline.values():point['time_aligned']=False
        assert B.decision(borderline,am,{'full_pipeline_seconds':119})['status'].startswith('pending')
        passed('tiny B calibration/clock-fixture branches, attached resume, alignment and2pp/120s boundary')
        result={'status':'passed','checks':checks,'GPU_runs':0,'full_A_updates':0,'full_B_updates':0,'torch_CPU':str(torch.__version__),
                'unverified':['Kaggle T4x2/runtime migration/FP16/fusedAdamW/memory','full heldout/test decode and120s','real Drive downloads/cache/PT','full26c remote extension']}
        write(ROOT.parent/'checks/kaggle-phase-a-cpu-checks.json',result)
    except BaseException:
        print('CPU debug directory retained:',tmp);raise
    else:shutil.rmtree(tmp)
if __name__=='__main__':main()
