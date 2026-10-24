"""Runtime adapters: immutable inputs, portable complete state, bounded local retention."""
import copy,json,os,shutil,time,uuid
from pathlib import Path
from types import SimpleNamespace
from kaggle_io import materialize_dataset,resolve_dataset,sha,write,disk_guard

CACHE_SHA='2c691fef64fea8b4186b8acae54561d5620080286196a4390a23108776e80add'
ORDER_SHA='737dac5a146ad15bb0ec70cb06fea59493277e4332dea440b2bdf2a6316c1637'
CP_SHA='8400c8e59da764c65fbcb574758acc72e8e034aedeb59b10bf979465f110ab5e'

def copy_output(source,target):
    """Only copy prior selected run output; preserve every source file, reject local conflicts."""
    from exp26_common import copy_verified
    source,target=Path(source),Path(target)
    if source.resolve()==target.resolve():return
    for p in source.rglob('*'):
        if p.is_file():copy_verified(p,target/p.relative_to(source))

def reuse_verified_references(source,target,sets,inputs):
    """Reuse the pinned previous session's successful reference gate across GPU workers."""
    from exp26_common import read_json,json_hash,copy_verified
    import exp26_exec as N
    source,target,inputs=Path(source),Path(target),Path(inputs)
    records={}
    for split,rows in sets.items():
        name=split+'-reference-execution.json';receipt_name=split+'-receipt.json'
        execution=read_json(source/name);receipt=read_json(source/receipt_name)
        identity={'rows_sha256':json_hash(rows),'saved_reference_sha256':sha(inputs/name),
                  'policy':'same25 sandbox finite solve bool->int rel=abs1e-6'}
        assert receipt['identity']==identity and receipt['sha256']==sha(source/name),'Reference reuse identity/SHA'
        expected=read_json(inputs/name);comparison=N.compare_exec(expected,execution)
        count=4096 if split=='heldout4k' else 991
        assert comparison['matches']==comparison['denominator']==count,'Reference reuse values'
        assert [i for i,r in enumerate(execution['rows']) if r['status']=='numeric']==comparison['eligible_indices'],'Reference reuse finite indices'
        assert receipt['gate']==comparison,'Reference reuse gate receipt'
        copy_verified(source/name,target/name,receipt['sha256'])
        copy_verified(source/receipt_name,target/receipt_name)
        records[split]={'sha256':receipt['sha256'],'numeric':count}
    write(target.parent/'reference-reuse.json',{'source':'pinned previous output gpu0-a26/references',
          'policy':'reuse once-verified reference receipts; no new numeric filtering or timeout change','splits':records})
    return records

def install_runtime_adapters():
    import torch
    import exp26_train as T
    import exp26_eval as E
    import exp26_workflow as W
    import exp26_exec as N
    from exp26_fast_exec import run_chunks
    from exp26_common import read_json,write_json
    original=T.Checkpoints
    class RetainedCheckpoints(original):
        def __init__(self,drive,local,identity):
            # Input/protocol/tensor/scheduler identity remains strict. Runtime fields are provenance.
            latest=Path(drive)/'latest.json'
            if latest.exists():
                old=read_json(latest)['identity']
                for k in set(old)|set(identity):
                    if k not in ('torch','CUDA'):assert old.get(k)==identity.get(k),f'Resume identity: {k}'
                if old!=identity:
                    write_json(Path(drive)/'runtime-migration.json',{'source_runtime':{k:old[k] for k in ('torch','CUDA')},
                        'execution_runtime':{k:identity[k] for k in ('torch','CUDA')},
                        'checkpoint_identity_retains_source_provenance':True,'bitwise_cross_runtime_equivalence_claimed':False})
                identity=old
            super().__init__(drive,local,identity)
        def save(self,*args,**kwargs):
            model,opt,scaler,state,meta,cfg,spec=args
            role=kwargs.get('role','periodic');latest=kwargs.get('latest',True)
            T.core.sync(next(model.parameters()).device);started=time.perf_counter()
            name=f'checkpoints/{role}-{meta["compute_updates"]:06d}-{uuid.uuid4().hex[:10]}.pt'
            weights=T.cpu(model.state_dict());digest=E.state_digest(weights)
            payload={'format':'nlp-hw02-wsd-v1','experiment':cfg.get('experiment',26),'identity':self.identity,
                'config':cfg,'model_spec':spec,'model':weights,'optimizer':T.cpu(opt.state_dict()),
                'scaler':scaler.state_dict(),'scaler_enabled':scaler.is_enabled(),'scheduler':T.schedule_state(state,cfg),
                'rng':T.cpu(T.rng_state()),'training':copy.deepcopy(state),'experiment_history':copy.deepcopy(meta),
                'model_tensor_sha256':digest,'purpose':'CPU_smoke' if self.identity['smoke'] else 'Kaggle_T4_training',
                'execution_runtime':{'torch':str(torch.__version__),'CUDA':torch.version.cuda}}
            target=self.local/name;target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.pt.tmp')
            torch.save(payload,tmp);tmp.replace(target);digest_file=sha(target)
            from exp26_common import copy_verified
            copy_verified(target,self.drive/name,digest_file)
            receipt={'file':name,'sha256':digest_file,'bytes':target.stat().st_size,'model_tensor_sha256':digest,
                'main_updates':state['main_updates'],'branch_updates':state['branch_updates'],'phase':state['phase'],
                'data_cursor':state['data_cursor'],'role':role,'identity':self.identity}
            write_json(self.drive/(name+'.json'),receipt)
            if latest:write_json(self.drive/'latest.json',receipt)
            meta['checkpoint_io_seconds']+=time.perf_counter()-started
            keep={receipt['file']}
            for selector in ('latest.json','final-checkpoint.json'):
                p=self.drive/selector
                if p.exists():keep.add(read_json(p)['file'])
            if state.get('active_anchor'):keep.add(state['active_anchor']['file'])
            for p in (self.drive/'points').glob('*/checkpoint.json'):keep.add(read_json(p)['file'])
            for root in {self.drive,self.local}:
                for p in (root/'checkpoints').glob('*.pt'):
                    if p.relative_to(root).as_posix() not in keep:
                        p.unlink();p.with_suffix('.pt.json').unlink(missing_ok=True)
            disk_guard(self.drive.parents[0] if self.drive.name.startswith('gpu') else self.drive)
            return receipt
    T.Checkpoints=RetainedCheckpoints
    def fast(codes,**kwargs):
        result=run_chunks(codes,workers=2,chunk_rows=2000)
        result['summary']['complete']=True;return result
    proxy=SimpleNamespace(evaluate_programs=fast,compare_exec=N.compare_exec)
    E.numeric=proxy;W.numeric=proxy # leave N untouched for the dead-chunk per-program fallback
    return RetainedCheckpoints

def prepare_starter(dataset,prepared,kind,read_only=False):
    from exp26_input import import_starter
    from exp26_common import copy_verified
    import exp26_protocol as P
    dataset,prepared=Path(dataset),Path(prepared)
    name='26a-cache-inputs-tokenizer-and-exclusions.zip' if kind=='cache-inputs' else '26b-training-inputs-d0-initial-and-evaluation.zip'
    archives=list(dataset.rglob(name))
    direct=[p.parent for p in dataset.rglob('protected-questions.json') if (p.parent/'tokenizer.json').exists()]
    if not archives and not direct:
        root=materialize_dataset(dataset,prepared.parent/('unpacked-'+kind),name)
        archives=list(root.rglob(name))
    if archives:
        assert len(archives)==1
        import_starter(archives[0],prepared,kind)
    else:
        assert len(direct)==1
        names={'protected-questions.json','tokenizer.json'} if kind=='cache-inputs' else set(P.INPUT_FILES)
        if read_only:
            for n in names:assert sha(direct[0]/n)==P.INPUT_FILES[n]
            return direct[0]
        for n in names:copy_verified(direct[0]/n,prepared/n,P.INPUT_FILES[n])
    return prepared

def inputs(config,working):
    from exp26_input import import_starter,validate_training_inputs
    from exp26_cache import TokenCache
    from exp26_common import read_json,sha_file
    import exp26_protocol as P
    working=Path(working);input_root=Path(config['input_root'])
    scratch=Path(config['temp_root']) if config.get('session_kind')=='s2' else working
    if config.get('session_kind')=='s2':assert not scratch.resolve().is_relative_to(working.resolve())
    prepared=prepare_starter(resolve_dataset(input_root,config['starter_dataset']),scratch/'inputs','training-inputs',
                             read_only=config.get('session_kind')=='s2')
    sets,test,protected=validate_training_inputs(prepared)
    cache_root=materialize_dataset(resolve_dataset(input_root,config['cache_dataset']),scratch/'unpacked-cache','cache-manifest.json')
    assert sha_file(cache_root/'cache-manifest.json')==CACHE_SHA,'Pinned cache manifest required'
    cache=TokenCache(cache_root,protected)
    # Initial26b input or explicitly attached previous S1 output. Never choose by mtime.
    resume_base=resolve_dataset(input_root,config['resume_dataset'])
    run=materialize_dataset(resume_base,scratch/'unpacked-resume','latest.json')
    assert sha_file(run/'order.npy')==ORDER_SHA
    order_meta=read_json(run/'order.json');assert order_meta['file']['sha256']==ORDER_SHA
    return prepared,sets,test,cache,run,order_meta

def prepare_a(config,working,run):
    from exp26_common import read_json,sha_file
    root=Path(working)/'gpu0-a26'
    resume=resolve_dataset(config['input_root'],config['previous_output']) if config.get('previous_output') else None
    if config.get('previous_output'):
        copy_output(resume/'gpu0-a26',root)
    else:
        latest=read_json(Path(run)/'latest.json')
        assert latest['sha256']==CP_SHA and sha_file(Path(run)/latest['file'])==CP_SHA
        assert latest['main_updates']==46875 and latest['phase']=='main'
        copy_output(run,root)
    return root

def evaluate_a(model_path,tag,previous,sets,refs,inputs,root,device,D0):
    import exp26_eval as E
    from exp26_common import read_json,write_json
    prevpath=root/'points'/str(previous)/'evaluation/metrics.json'
    available=prevpath.exists() and all((root/'points'/str(previous)/'evaluation'/(split+'-generations.json')).exists() for split in sets)
    prev=read_json(prevpath) if available else None
    out=root/'points'/tag/'evaluation'
    metrics=E.evaluate_point(model_path,sets,refs,inputs/'tokenizer.json',device,out,tag,D0,prev)
    metrics['paired_comparison_basis']=previous if available else 'D0; previous generations unavailable in input'
    write_json(out/'metrics.json',metrics);return metrics
