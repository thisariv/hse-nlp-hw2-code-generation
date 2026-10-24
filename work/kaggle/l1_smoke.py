#!/usr/bin/env python3
"""Actual tiny CPU cache, two workers, eval/rollback/backward, resume and live CLI mocks."""
import copy, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))

def main():
    import numpy as np, torch
    import exp26_core as C, l1_cache as K, l1_runtime as L, l1_live as Live
    from kaggle_io import sha,write
    from l1_session import Watchdog,launch,decision
    from tokenizers import Tokenizer
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tmp=Path(tempfile.mkdtemp(prefix='nlp-l1-smoke-'));inputs=tmp/'inputs';inputs.mkdir()
    original=ROOT.parent/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json'
    shutil.copyfile(original,inputs/'tokenizer.json');tok=Tokenizer.from_file(str(original))
    rows=[{'question':f'Add {i} and 2','code':f'def solve() -> int:\n    # comment\n    result = {i} + 2\n    return result\n'} for i in range(32)]
    for shard,part in enumerate((rows[:16],rows[16:])):
        root=tmp/f'original-{shard}';folder=root/f'shards/part-{shard:05d}';folder.mkdir(parents=True)
        source=[];target=[];so=[0];to=[0]
        for row in part:
            source.extend([1]+C.encode(tok,row['question'])+[2]);target.extend(C.encode(tok,row['code']));so.append(len(source));to.append(len(target))
        arrays={'source.npy':np.array(source,dtype=np.int16),'source-offsets.npy':np.array(so,dtype=np.int64),
                'target.npy':np.array(target,dtype=np.int16),'target-offsets.npy':np.array(to,dtype=np.int64),
                'provenance-rows.npy':np.arange(16,dtype=np.uint32),'question-sha256.npy':np.zeros((16,32),dtype=np.uint8)}
        for name,a in arrays.items():np.save(folder/name,a,allow_pickle=False)
        receipt={'shard':shard,'folder':folder.relative_to(root).as_posix(),'files':[K.record(folder/n) for n in K.FILES]}
        write(folder/'manifest.json',receipt);write(root/'cache-manifest.json',{'shards':[receipt],'rows':16,'sealed':True,'fresh_evaluations':0})
    cache_root=tmp/'compact';cached=K.build(tmp/'original-0',tmp/'original-1',original,cache_root,workers=2,sample_rows=32,smoke=True)
    assert cached['answer_mismatches']==0 and cached['roundtrip_rows']==32 and cached['rows']==32
    cache=K.CompactCache(cache_root);order=np.load(cache_root/'order.npy',mmap_mode='r');order_sha=sha(cache_root/'order.npy')
    assert len(set(order.tolist()))==32 and np.array_equal(L.take(order,31,3),order[[31,0,1]])
    sets={'heldout4k':rows[:2],'old991':rows[2:4]};refs={s:{'rows':[{'status':'numeric','value':i+2} for i in (range(2) if s=='heldout4k' else range(2,4))]} for s in sets}
    write(inputs/'smoke-sets.json',sets);write(inputs/'smoke-refs.json',refs);write(inputs/'test.json',rows[4:6])
    spec={'d_model':16,'heads':2,'encoder_layers':1,'decoder_layers':1,'swiglu_dim':32,'vocab':8195,'max_positions':512}
    work=tmp/'working';work.mkdir();(work/'src').mkdir()
    for p in (ROOT/'src').glob('*.py'):shutil.copyfile(p,work/'src'/p.name)
    shutil.copyfile(ROOT/'l1_worker.py',work/'src/l1_worker.py')
    cfg={'working_root':str(work),'temp_root':str(tmp/'scratch'),'starter_dataset':str(inputs),'compact_cache':str(cache_root),
         'cache_manifest_sha256':cache.manifest_sha,'smoke':True,'smoke_spec':spec,
         'smoke_recipe':{'batch':2,'warmup':2,'checkpoint_seconds':.15,'point_seconds':.01,'cooldown_cap':20.,'max_updates':8},
         'session_start_monotonic':time.monotonic(),'deadline_monotonic':time.monotonic()+180,
         'hard_deadline_monotonic':time.monotonic()+210,'budget_seconds':180,'hard_shutdown_seconds':210,
         'terminate_grace_seconds':.2,'session_kind':'l1'}
    watch=Watchdog(cfg).start();report=launch(cfg,work/'src/l1_worker.py',watch,smoke=True,heartbeat=1.)
    assert report['exit_codes']==[0,0],report
    for name in ('C','D'):
        root=work/f'model-{name}';receipt=json.loads((root/'latest.json').read_text());payload=L.read_full(root/receipt['file'],json.loads((root/'identity.json').read_text()))
        assert payload['state']['updates']==8 and payload['state']['cursor']==16
        assert payload['state']['branch_attempted_updates']>=2 and len(payload['state']['points'])>=2
        assert all(int(s['step'])==8 for s in payload['optimizer']['state'].values())
        for p in (root/'weights').glob('*.pt'):
            value=torch.load(p,map_location='cpu',weights_only=True)
            assert value['optimizer_included'] is False and 'optimizer' not in value
            assert all(v.dtype==torch.float16 for v in value['model'].values() if v.is_floating_point())
    checks=['tiny full cache/source/provenance/order/finite roundtrip','two real independent CPU workers',
            'multiple cooldowns, real4Dgreedy/restoration/exec/CE/test timing and subsequent backward',
            'FP16 evaluated weights exclude optimizer; stable Adam steps/cursor exclude branches']
    # Exact resumed tensors/Adam/RNG versus uninterrupted training, independent of noisy wall times.
    def unused(*args):raise AssertionError('Unexpected evaluation in resume fixture')
    recipe=dict(smoke=True,spec=spec,batch=2,warmup=2,point_seconds=10000.,checkpoint_seconds=10000.)
    L.run('C',cache,order,order_sha,tmp/'continuous',tmp/'a',torch.device('cpu'),time.monotonic()+60,unused,max_updates=4,**recipe)
    L.run('C',cache,order,order_sha,tmp/'first',tmp/'b',torch.device('cpu'),time.monotonic()+60,unused,max_updates=2,**recipe)
    L.run('C',cache,order,order_sha,tmp/'resumed',tmp/'c',torch.device('cpu'),time.monotonic()+60,unused,max_updates=2,resume=tmp/'first',**recipe)
    def payload(folder):
        r=json.loads((folder/'latest.json').read_text());return torch.load(folder/r['file'],map_location='cpu',weights_only=True)
    a,b=payload(tmp/'continuous'),payload(tmp/'resumed')
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    for k,v in a['optimizer']['state'].items():
        assert all(torch.equal(x,b['optimizer']['state'][k][name]) for name,x in v.items())
    assert torch.equal(a['rng']['torch_cpu'],b['rng']['torch_cpu']) and a['state']['cursor']==b['state']['cursor']==8
    checks.append('exact resume model/Adam/RNG/cursor versus uninterrupted4updates')
    # Token-weighted micro accumulation agrees with fullbatch gradient/update within rounding tolerance.
    m1=L.new_model(spec,torch.device('cpu'));m2=L.new_model(spec,torch.device('cpu'))
    opt1=L.optimizer(m1,.0015,True);opt2=L.optimizer(m2,.0015,True)
    sc1=torch.amp.GradScaler('cpu',enabled=False);sc2=torch.amp.GradScaler('cpu',enabled=False)
    ids=np.array([0,1,30,31]);log1=L.update(m1,opt1,sc1,cache,ids,torch.device('cpu'),4,.0015)
    log2=L.update(m2,opt2,sc2,cache,ids,torch.device('cpu'),3,.0015)
    maxdiff=max(float((x-m2.state_dict()[k]).abs().max()) for k,x in m1.state_dict().items())
    grad_diff=max(float((x.grad-y.grad).abs().max()) for x,y in zip(m1.parameters(),m2.parameters()) if x.grad is not None)
    assert grad_diff<1e-6 and abs(log1['CE']-log2['CE'])<2e-6,(grad_diff,log1,log2)
    checks.append('effective batch accumulation independent full4 versus3+1 tokenweighted oracle')
    fakekey='MOCK_AUTH_'+os.urandom(24).hex()
    def secrets(name):return 'thisarivson' if name=='KAGGLE_USERNAME' else fakekey
    def fake_gpu(count=2,names=('Tesla T4','Tesla T4')):
        return SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:count,get_device_name=lambda i:names[i]))
    def missing(name):raise RuntimeError('mock unavailable')
    assert not Live.gate(fake_gpu(),missing)['passed']
    assert not Live.gate(fake_gpu(1),secrets)['passed']
    assert not Live.gate(fake_gpu(2,('T4','A100')),secrets)['passed']
    assert Live.gate(fake_gpu(),secrets)['passed']
    checks.append('secrets absent/oneGPU/wrongGPU gates; correct T4x2 passes')
    live=Live.L1LiveSync(work,tmp/'live','thisarivson/nlp-hw02-l1-live');calls=[];exists=[False]
    def mock_call(args,timeout,final=False):
        calls.append(args)
        if len(args)>2 and args[2]=='stage':Live.stage_snapshot(work,live.stage,live.ref,'mock')
        elif len(args)>2 and args[2]=='privacy':write(Path(args[args.index('--receipt')+1]),{'exists':exists[0],**({'private':True} if exists[0] else {})})
        elif len(args)>2 and args[2]=='ready':write(Path(args[args.index('--receipt')+1]),{'status':'ready'})
        elif args[:2]==['kaggle','datasets']:exists[0]=True
        else:raise AssertionError(args)
    live.bounded_call=mock_call
    assert live.sync('startup-probe',wait_ready=True) and live.sync('new-point')
    uploads=[args for args in calls if args[:2]==['kaggle','datasets']]
    assert [a[2] for a in uploads]==['create','version'] and all('--public' not in a and a[a.index('--dir-mode')+1]=='skip' for a in uploads)
    restored=Live.restore_snapshot(live.stage,tmp/'recovered')
    for name in ('C','D'):
        r=json.loads((restored/f'model-{name}/latest.json').read_text());assert sha(restored/f'model-{name}'/r['file'])==r['sha256']
    (work/'injected.bin').write_bytes(b'prefix'+fakekey.encode()+b'suffix')
    try:Live.stage_snapshot(work,live.stage,live.ref,'must-refuse');raise AssertionError('Credential was accepted')
    except RuntimeError:pass
    (work/'injected.bin').unlink()
    Live.stage_snapshot(work,live.stage,live.ref,'clean')
    try:Live.stage_snapshot(work,tmp/'too-small',live.ref,'size-limit',limit=1024);raise AssertionError('Limit ignored')
    except ValueError:pass
    checks.append('mockprivate create/version, exact full resume from live, latest evaluated artifacts, secret byte rejection and size limit')
    hung=Live.L1LiveSync(work,tmp/'hung',live.ref)
    start=time.monotonic()
    try:hung.bounded_call([sys.executable,'-c','import time;time.sleep(30)'],.2);raise AssertionError('Timeout ignored')
    except subprocess.TimeoutExpired:assert time.monotonic()-start<2
    checks.append('real hung subprocess group timeout')
    # Registered selection rule on synthetic points, independent of model's random tiny quality.
    synthetic=tmp/'decision'
    for name,accuracies in [('C',[.50,.60]),('D',[.47,.59])]:
        for j,acc in enumerate(accuracies,1):
            write(synthetic/f'model-{name}/points/t{j}/complete.json',{'training':{'main_seconds':9000*j,'target_main_seconds':9000*j},
                  'metrics':{'splits':{'heldout4k':{'exec':{'accuracy':acc}}}}})
    assert decision(synthetic)['continue_L2']==['C','D']
    checks.append('D lags1pp but slope12pp versusC10pp -> continue both')
    counts={}
    with torch.device('meta'):
        for name,specification in L.SPECS.items():counts[name]=sum(p.numel() for p in C.SimpleTransformer(**specification,dropout=0.).parameters())
    result={'passed':True,'checks':checks,'parameter_counts':counts,'micro_accumulation_max_weight_diff':maxdiff,
            'micro_accumulation_max_gradient_diff':grad_diff,
            'two_process_exit_codes':report['exit_codes'],'tiny_roundtrip':cached['roundtrip_rows'],
            'not_verified':['real Kaggle Secrets/live private upload','T4 FP16/fused/memory/backend/throughput','full GPU quality/test1919 timing','long watchdog CUDA D-state']}
    write(ROOT.parent/'checks/kaggle-l1-smoke.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    # No mock auth values in checked-in reports or generated files.
    for name in Live.SECRET_NAMES:os.environ.pop(name,None)

if __name__=='__main__':main()
