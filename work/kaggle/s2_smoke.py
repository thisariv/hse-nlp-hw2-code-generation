#!/usr/bin/env python3
"""CPU-only S2 verification with real sampler, saved references and two tiny controllers."""
import json,os,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))

def worker(args):
    import numpy as np,torch
    import exp28_capacity as B,exp26_train as T,s2_runtime as S
    from exp26_common import write_json,read_json
    cfg=read_json(args.config);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    S.configure_storage(Path(cfg['scratch'])/f'gpu{args.gpu}');S.install()
    class Cache:
        manifest_sha='tiny-S2-cache'
        def __len__(self):return 100
        def __getitem__(self,i):return np.array([1,4,5,2],dtype=np.int16),np.array([6,7],dtype=np.int16)
    actual=T.train_chunk
    def clock(*a,**kw):
        before=a[4]['train_seconds'];log=actual(*a,**kw);a[4]['train_seconds']=before+.1*a[9];return log
    T.train_chunk=clock;experiment=29 if args.gpu==0 else 28
    protocol=B.config(batch=2,experiment=experiment);protocol.update(main_updates=40,warmup_updates=1,main_cooldown_start=38,main_cooldown_updates=2)
    root=Path(cfg['working_root'])/f'gpu{args.gpu}-exp{experiment}'
    targets=[{'A_point':'fixture1','seconds':.8,'actual_pairs':4},{'A_point':'fixture2','seconds':2.,'actual_pairs':8}]
    order=np.arange(100,dtype=np.uint32);order_meta={'file':{'sha256':'tiny-order'}}
    def evaluate(cp,tag,prev,target):
        with S.evaluation_lock(cfg['working_root'],time.monotonic()+20):
            payload=S.load_payload(cp);assert payload['experiment']==experiment and 'optimizer' not in payload
            score={'accuracy':.5,'matches':1,'denominator':2,'eligible_indices':[0,1],'match_indices':[0]}
            result={'CPU_fixture_only':True,'point':tag,'splits':{s:{'exec':score} for s in ('heldout4k','old1024')}}
            write_json(cp.parent.parent/'points'/tag/'evaluation/metrics.json',result)
            return result
    def run(dest,limit):
        return B.run(Cache(),order,order_meta,cfg['inputs'],dest,'tiny-S2-port',torch.device('cpu'),evaluate,time.monotonic()+40,
            cfg=protocol,spec=cfg['spec'],smoke=True,targets=targets,calibration=2,max_compute=limit,checkpoint_seconds=0,stop_after_targets=True)
    first=run(root,18);assert first['status'].startswith('partial')
    saved=S.load_payload(root/first['latest']['file']);assert saved['training']['phase']=='branch' and saved['resume_anchor']
    assert (root/'29-summary.json' if experiment==29 else root/'28-summary.json').exists()
    import shutil
    attached=Path(cfg['working_root'])/f'original{args.gpu}';shutil.copytree(root,attached)
    shutil.rmtree(S._SCRATCH_ROOT);S.configure_storage(Path(cfg['scratch'])/f'gpu{args.gpu}')
    second=run(root,40);mono=run(Path(cfg['working_root'])/f'monolithic{args.gpu}',40)
    assert second['status']=='complete' and len(second['points'])==2
    s=S.load_payload(root/second['latest']['file']);m=S.load_payload(Path(cfg['working_root'])/f'monolithic{args.gpu}'/mono['latest']['file'])
    assert S.tensor_digest({'model':s['model'],'opt':s['optimizer']})==S.tensor_digest({'model':m['model'],'opt':m['optimizer']})
    assert s['scheduler']==m['scheduler'] and torch.equal(s['rng']['torch_cpu'],m['rng']['torch_cpu'])
    headers=list((root/'checkpoints').glob('*.pt'));weights=list((root/'weights').glob('*.pt'))
    assert len(headers)==2 and len(weights)==2 and not list((root/'checkpoints/blobs').glob('*.pt'))
    for point in weights:
        payload=S.load_payload(point);assert 'optimizer' not in payload and payload['optimizer_included'] is False
        assert all(t.dtype==torch.float32 for t in payload['model'].values() if t.is_floating_point())
    final=S.load_payload(root/read_json(root/'final-checkpoint.json')['file']);assert 'optimizer' in final and 'resume_anchor' not in final
    assert len(list((S._SCRATCH_ROOT/'anchors').rglob('*.pt')))==0
    write_json(Path(cfg['working_root'])/f'worker{args.gpu}-check.json',{'passed':True,'experiment':experiment,'pid':os.getpid(),
        'branches':len(second['points']),'resume_model_optimizer_scheduler_rng_parity':True,
        'resume_after_anchor_scratch_deletion':True,'full_checkpoints':len(headers),'model_only_points':len(weights)})

def main():
    import torch
    import exp26_core as C,exp26_eval as E,exp27_decode as Old,exp27b_fast as D,s2_runtime as S,s2_report as Report
    from exp26_common import read_json,write_json
    from session_runner import launch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);S.install()
    tmp=Path(tempfile.mkdtemp(prefix='nlp-S2-smoke-'));checks=[]
    def passed(name,**extra):checks.append({'check':name,'passed':True,**extra});print('PASSED',name,flush=True)
    spec={'d_model':16,'heads':2,'encoder_layers':1,'decoder_layers':1,'swiglu_dim':32,'vocab':8192,'max_positions':64}
    torch.manual_seed(42);model=C.SimpleTransformer(**spec,dropout=0.).eval();src=torch.tensor([[1,4,5,2,0],[1,5,6,7,2]])
    old=C.CachedDecoder(model,src,32);group=Old.SampleDecoder(Old.EncoderContext(model,src,32),4);new=D.Decoder(D.Context(model,src,32),4)
    for token in (torch.tensor([1,1]),torch.tensor([4,5]),torch.tensor([0,2])):
        original=old.step(token);grouped=group.step(token[:,None].expand(-1,4));fast=new.step(token.repeat_interleave(4))
        assert torch.allclose(fast,original.repeat_interleave(4,0),atol=2e-6,rtol=1e-5)
        assert torch.allclose(fast,grouped.reshape(-1,8192),atol=2e-6,rtol=1e-5)
    assert all(k.ndim==v.ndim==4 and k.is_contiguous() and v.is_contiguous() for k,v in new.cross)
    assert new.cross[0][0].data_ptr()!=new.c.cross[0][0].data_ptr()
    for half in (False,True):
        current=model.half() if half else model
        tokens,_=C.generate_preallocated(current,src,cap=32);fast,_,_=D.decode(D.Context(current,src,32))
        assert torch.equal(tokens,fast[:,0])
    model.float();passed('new4D logits agree with original and5D; greedy token-identical FP32/FP16 on tiny CPU')
    inputs=tmp/'inputs';inputs.mkdir();previous=ROOT.parent/'results/kaggle-runs/nlp-hw02-kaggle-s1/1'
    import shutil
    shutil.copyfile(previous/'inputs/tokenizer.json',inputs/'tokenizer.json')
    model_path=inputs/'model.pt';torch.save(model.state_dict(),model_path)
    toy=[{'question':'add two and three','code':'def solve():\n return 5'},{'question':'subtract one','code':'def solve():\n return 2'}]
    controls,_=E.generate_one(model_path,toy,inputs/'tokenizer.json',torch.device('cpu'),tmp/'control','tiny',spec,batch=2,cap=8,half=False)
    rows,timing=D.generate(model_path,toy,inputs/'tokenizer.json',torch.device('cpu'),tmp/'sampler',16,32,8,spec)
    resumed,replay=D.generate(model_path,toy,inputs/'tokenizer.json',torch.device('cpu'),tmp/'sampler',16,32,8,spec)
    assert rows==resumed and replay['resumed_batches']==1 and replay['new_batches']==0
    assert [r['greedy']['token_ids'] for r in rows]==[r['generated_token_ids'] for r in controls]
    ex=D.execute(rows,tmp/'execution');assert ex['programs']==34 and ex['wall_limit']==3. and ex['cpu_limit']==2. and ex['workers']==2
    from exp26_fast_exec import run_chunks
    bounded=run_chunks(['def solve():\n while True:\n  pass','def solve():\n return 7'],wall_limit=3.,cpu_limit=.1,chunk_rows=2)
    assert bounded['rows'][0]['status']=='timeout' and bounded['rows'][0]['timeout_limit']=='cpu'
    assert bounded['rows'][1]=={'status':'numeric','value':7.}
    for k in (0,8):
        timed=D.time_test(model_path,toy,inputs/'tokenizer.json',torch.device('cpu'),tmp/f'test{k}',D.config(k),spec,batch_rows=32,cap=8)
        assert timed['executed_candidates']==len(toy)*(1+k) and timed['complete'] and not timed['cache_reuse']
        assert timed['full_pipeline_seconds']>=timed['components']['generation']['load_generate_write_seconds']
    # Recover an incomplete test attempt with entirely fresh files, preserving the old journal.
    incomplete=tmp/'incomplete/attempt-000';incomplete.mkdir(parents=True);(incomplete/'partial.txt').write_text('fixture')
    recovered=D.time_test(model_path,toy,inputs/'tokenizer.json',torch.device('cpu'),tmp/'incomplete',D.config(),spec,batch_rows=32,cap=8)
    assert recovered['attempt']=='attempt-001' and (incomplete/'partial.txt').exists()
    probe=D.probe_batch(model_path,toy,inputs/'tokenizer.json',torch.device('cpu'),tmp/'probe',spec=spec,cap=8,choices=(32,16))
    assert probe['batch_rows']==32
    passed('real sampling/resume/reference tokens/full greedy and vote timers include execution; incomplete test retry preserves files')
    subprocess.run([sys.executable,str(ROOT.parent/'checks/kaggle-s2-reference-reuse-cpu-check.py')],check=True)
    passed('real S1 reference receipts reused without program execution; identity tamper rejected')
    working=tmp/'session';start=time.monotonic()
    cfg={'working_root':str(working),'session_kind':'s2','inputs':str(inputs),'src':str(ROOT/'src'),'spec':spec,'scratch':str(tmp/'anchor-scratch'),
        'session_start_monotonic':start,'deadline_monotonic':start+90,'budget_seconds':90}
    report=launch(cfg,Path(__file__),smoke=True,heartbeat_seconds=2)
    assert report['exit_codes']==[0,0],[(p.name,p.read_text()[-4000:]) for p in working.glob('*.log')]
    workers=[read_json(working/f'worker{i}-check.json') for i in (0,1)];assert workers[0]['pid']!=workers[1]['pid']
    assert not list(working.glob('*.zip'))
    manifest=read_json(working/'s2-output-manifest.json')
    assert manifest['output_bytes']<100*1024**2 and report['output_bytes']<100*1024**2
    assert all((working/item['file']).stat().st_size==item['bytes'] for item in manifest['files'])
    passed('two independent M/B processes; branch resume after scratch deletion; latest/final retention; model-only points; no ZIP',
        workers=workers,output_bytes=report['output_bytes'])
    # Recover a completed atomic metric when a worker dies before its experiment summary update.
    summary_path=working/'gpu0-exp29/29-summary.json';summary=read_json(summary_path)
    summary['points'].pop('t02');write_json(summary_path,summary)
    recovered=Report.collect(working)
    assert any(row['tag']=='t02' and 'M' in row['models'] for row in recovered['comparisons'])
    passed('session report recovers points from incremental training/metrics receipts')
    def metric(matches):return {'splits':{'heldout4k':{'exec':{'accuracy':matches/4096}}}}
    common=[{'tag':'t03','A_point':'p025000','target_seconds':4203.,'models':{'A':metric(2446),'M':metric(2527),'B':metric(2400)}}]
    times={'t03':{'A':{'full_pipeline_seconds':100},'M':{'full_pipeline_seconds':110},'B':{'full_pipeline_seconds':115}}}
    assert Report.decision(common,times)['final_model']=='A' #81 gain<2pp; faster A
    common[0]['models']['M']=metric(2528);assert Report.decision(common,times)['final_model']=='M' #82 gain>=2pp
    times['t03']['M']['full_pipeline_seconds']=121;assert Report.decision(common,times)['status'].startswith('pending')
    assert Report.decision(common,times,False)['status'].startswith('pending')
    assert D.select_k([{'config':D.config(8),'full_pipeline_seconds':120.,'complete':True},{'config':D.config(16),'full_pipeline_seconds':121.,'complete':True}])['k']==8
    assert D.select_k([{'config':D.config(8),'full_pipeline_seconds':121.,'complete':True}])['k']==0
    passed('registered largest-k/120s and2pp final decision boundaries; gates and missing timing yield pending')
    result={'status':'passed','checks':checks,'torch_CPU':str(torch.__version__),'GPU_runs':0,'full_M_updates':0,'full_B_updates':0,
        'unverified':['T4 FP16 backend, memory2048/1024, fusedAdamW and B128/OOM96','full heldout gates2707±45 and token identity','test1919 selected vote<=120s','both models reaching7313s within11700s']}
    write_json(ROOT.parent/'checks/kaggle-s2-cpu-checks.json',result)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int);parser.add_argument('--config',type=Path);a=parser.parse_args()
    if a.gpu is None:main()
    else:
        cfg=json.loads(a.config.read_text());sys.path.insert(0,cfg['src']);worker(a)
