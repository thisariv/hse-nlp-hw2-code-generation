"""S2: serialized evaluations, bounded full checkpoints and model-only point weights."""
import copy,fcntl,gc,hashlib,json,tempfile,time,uuid
from contextlib import contextmanager
from pathlib import Path
import torch
import exp26_train as T
import exp26_eval as E
import exp26_core as C
import exp26_exec as N
from exp26_common import read_json,write_json,sha_file,json_hash
from exp26_fast_exec import run_chunks

_SCRATCH_ROOT=None

def configure_storage(root):
    global _SCRATCH_ROOT
    _SCRATCH_ROOT=Path(root)
    _SCRATCH_ROOT.mkdir(parents=True,exist_ok=True)

@contextmanager
def evaluation_lock(working,deadline=None):
    path=Path(working)/'evaluation.lock';path.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with path.open('a') as handle:
        while True:
            C.check_deadline(deadline)
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(.2)
        try:yield time.monotonic()-started
        finally:fcntl.flock(handle,fcntl.LOCK_UN)

def load_payload(path):
    path=Path(path);p=torch.load(path,map_location='cpu',weights_only=True)
    if 'tensor_blob' in p:
        blob=path.parent/p['tensor_blob']['relative_path']
        assert sha_file(blob)==p['tensor_blob']['sha256'],'Checkpoint tensor blob SHA'
        tensors=torch.load(blob,map_location='cpu',weights_only=True)
        p['model']=tensors['model'];p['optimizer']={'state':tensors['optimizer_state'],'param_groups':p.pop('optimizer_groups')}
    return p

def tensor_digest(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous();h.update(str(a.dtype).encode());h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for item in v:visit(item)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()

class Checkpoints:
    def __init__(self,drive,local,identity):
        self.drive,self.local,self.identity=Path(drive),Path(local),identity
        assert self.drive==self.local,'S2 writes only one working copy'
        global _SCRATCH_ROOT
        if _SCRATCH_ROOT is None:configure_storage(tempfile.mkdtemp(prefix='s2-checkpoint-scratch-'))
        self.scratch=_SCRATCH_ROOT/'anchors'/hashlib.sha256(str(self.drive.resolve()).encode()).hexdigest()[:16]
        self.scratch.mkdir(parents=True,exist_ok=True)
        assert not self.scratch.resolve().is_relative_to(self.drive.parent.resolve()),'Anchor scratch must be outside output'
        self.drive.mkdir(parents=True,exist_ok=True)
        latest=self.drive/'latest.json'
        if latest.exists():
            old=read_json(latest);fallback=self.drive/'batch128-oom.json'
            if old['identity']!=identity:
                assert fallback.exists() and read_json(fallback)==old,'Strict S2 resume identity'
                assert all(old['identity'][k]==v for k,v in identity.items() if k!='protocol_sha256'),'Batch fallback identity'
    def prune(self,state):
        keep=set()
        for selector in ('latest.json','final-checkpoint.json'):
            path=self.drive/selector
            if path.exists():keep.add(read_json(path)['file'])
        # Only full latest + final stay in working; evaluated points are model-only weights.
        for path in (self.drive/'checkpoints').glob('*.pt'):
            if path.relative_to(self.drive).as_posix() not in keep:
                path.unlink();path.with_suffix('.pt.json').unlink(missing_ok=True)
        blobs={read_json(self.drive/s).get('tensor_blob',{}).get('relative_path') for s in ('latest.json','final-checkpoint.json') if (self.drive/s).exists()}
        for path in (self.drive/'checkpoints/blobs').glob('*.pt'):
            if 'blobs/'+path.name not in blobs:path.unlink()
        anchor=state.get('active_anchor',{});anchor=anchor['file'] if anchor else None
        for path in (self.scratch/'checkpoints').glob('*.pt'):
            if path.relative_to(self.scratch).as_posix()!=anchor:path.unlink()

    def save(self,model,opt,scaler,state,meta,cfg,spec,role='periodic',latest=True):
        C.sync(next(model.parameters()).device);started=time.perf_counter()
        weights=T.cpu(model.state_dict());digest=E.state_digest(weights)
        name=f'checkpoints/{role}-{meta["compute_updates"]:06d}-{uuid.uuid4().hex[:10]}.pt'
        scratch=role=='anchor';target=(self.scratch if scratch else self.drive)/name
        payload={'format':'nlp-hw02-wsd-v1','experiment':cfg['experiment'],'identity':self.identity,'config':cfg,'model_spec':spec,
            'model':weights,'optimizer':T.cpu(opt.state_dict()),'scaler':scaler.state_dict(),'scaler_enabled':scaler.is_enabled(),
            'scheduler':T.schedule_state(state,cfg),'rng':T.cpu(T.rng_state()),'training':copy.deepcopy(state),
            'experiment_history':copy.deepcopy(meta),'model_tensor_sha256':digest,
            'purpose':'CPU_smoke' if self.identity['smoke'] else 'Kaggle_S2_T4'}
        if role=='final-point':
            payload['training']['active_anchor']=None
            payload['resume_scope']='evaluated cooldown state; stable trajectory uses latest.json'
        elif state.get('active_anchor'):
            anchor=state['active_anchor']
            payload['resume_anchor']={'receipt':anchor,'payload':load_payload(self.local_file(anchor))}
        target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.pt.tmp')
        torch.save(payload,tmp);tmp.replace(target)
        receipt={'file':name,'sha256':sha_file(target),'bytes':target.stat().st_size,'model_tensor_sha256':digest,
            'main_updates':state['main_updates'],'branch_updates':state['branch_updates'],'phase':state['phase'],
            'data_cursor':state['data_cursor'],'role':role,'identity':self.identity}
        if scratch:receipt['storage']='scratch'
        else:
            write_json(target.with_suffix('.pt.json'),receipt)
            if latest:write_json(self.drive/'latest.json',receipt)
            if role=='final-point':write_json(self.drive/'final-checkpoint.json',receipt)
            self.prune(state)
        meta['checkpoint_io_seconds']+=time.perf_counter()-started
        return receipt

    def save_weights(self,model,state,meta,cfg,spec,tag):
        C.sync(next(model.parameters()).device)
        weights=T.cpu(model.state_dict());name=f'weights/{tag}.pt';target=self.drive/name
        target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.pt.tmp')
        torch.save({'format':'nlp-hw02-model-only-v1','model':weights,'model_spec':spec,
            'model_tensor_sha256':E.state_digest(weights),'experiment':cfg['experiment'],
            'point':tag,'dtype':'FP32','optimizer_included':False},tmp);tmp.replace(target)
        receipt={'file':name,'sha256':sha_file(target),'bytes':target.stat().st_size,
            'model_tensor_sha256':E.state_digest(weights),'identity':self.identity,
            'role':'evaluated-weights','optimizer_included':False}
        return receipt

    def local_file(self,receipt):
        assert receipt['identity']==self.identity
        root=self.scratch if receipt.get('storage')=='scratch' else self.drive
        path=root/receipt['file'];assert path.stat().st_size==receipt['bytes'] and sha_file(path)==receipt['sha256']
        return path
    def load(self,receipt,device):
        p=load_payload(self.local_file(receipt));assert p['identity']==self.identity
        cfg=p['config'];model=C.SimpleTransformer(**p['model_spec'],dropout=cfg['dropout']).to(device);model.load_state_dict(p['model'],strict=True)
        assert E.state_digest(model.state_dict())==p['model_tensor_sha256']==receipt['model_tensor_sha256']
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['peak_lr'],weight_decay=cfg['weight_decay'],**({'fused':True} if not self.identity['smoke'] else {}))
        opt.load_state_dict(p['optimizer']);scaler=torch.amp.GradScaler(device.type,enabled=p['scaler_enabled'],init_scale=1024);scaler.load_state_dict(p['scaler'])
        state,meta=p['training'],p['experiment_history'];assert p['scheduler']==T.schedule_state(state,cfg)
        if p.get('resume_anchor'):
            saved=p['resume_anchor'];anchor=copy.deepcopy(saved['receipt']);target=self.scratch/anchor['file']
            target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.pt.tmp')
            torch.save(saved['payload'],tmp);tmp.replace(target)
            anchor.update(storage='scratch',sha256=sha_file(target),bytes=target.stat().st_size)
            state['active_anchor']=anchor
        assert state['data_cursor']==cfg['batch']*(state['main_updates']+state['branch_updates'])
        if state['main_updates']+state['branch_updates']:assert opt.param_groups[0]['lr']==state['last_lr']
        steps=state['main_updates']+state['branch_updates']-state['AMP_skips'];assert all(int(s['step'])==steps for s in opt.state.values())
        T.restore_rng(p['rng'],device);model.train();return model,opt,scaler,state,meta

def install():
    """Frozen exp26 sources stay byte-identical; only S2 runtime gets these adapters."""
    T.Checkpoints=Checkpoints
    import s2_exec
    s2_exec.install()
    def load_model(path,device,dropout=.1,scaled_tokens=False,spec=None):
        model=C.SimpleTransformer(**(spec or C.MODEL_SPEC),dropout=dropout)
        p=load_payload(path);state=p['model'] if p.get('format') in ('nlp-hw02-wsd-v1','nlp-hw02-training-v1','nlp-hw02-model-only-v1') else p
        model.load_state_dict(state,strict=True)
        if scaled_tokens:
            with torch.no_grad():model.tokens.weight.div_(model.width**.5)
        return model.to(device)
    E.load_model=load_model
    def execute(codes,**kwargs):
        result=run_chunks(codes,workers=2,wall_limit=3.,cpu_limit=2.,chunk_rows=2000)
        result['summary']['complete']=True
        result['summary']['timeouts']=sum('timeout' in r['status'] for r in result['rows'])
        return result
    from types import SimpleNamespace
    E.numeric=SimpleNamespace(evaluate_programs=execute,compare_exec=N.compare_exec)

def references(previous,sets,inputs,root):
    from kaggle_runtime import reuse_verified_references
    import exp26_workflow as W
    reuse_verified_references(Path(previous)/'gpu0-a26/references',Path(root)/'references',sets,inputs)
    return W.reference_execution(sets,inputs,root)
