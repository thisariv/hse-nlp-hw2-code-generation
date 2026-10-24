"""WSD main trajectory + reversible side cooldowns; every Drive checkpoint is complete."""
import copy,gc,hashlib,json,math,random,time,uuid
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import exp26_core as core
import exp26_eval as evaluation
import exp26_protocol as P
from exp26_common import sha_file,json_hash,read_json,write_json,copy_verified
from exp26_order import collate,prefix_identity


def main_lr(update,cfg):
 if not 1<=update<=cfg['main_updates']:raise ValueError('Main update out of range')
 if update<=cfg['warmup_updates']:return cfg['peak_lr']*update/cfg['warmup_updates']
 if update<=cfg['main_cooldown_start']:return cfg['peak_lr']
 fraction=(update-cfg['main_cooldown_start'])/cfg['main_cooldown_updates']
 return cfg['peak_lr']+(cfg['floor_lr']-cfg['peak_lr'])*fraction

def branch_lr(update,length,cfg):
 assert 1<=update<=length
 return cfg['peak_lr']+(cfg['floor_lr']-cfg['peak_lr'])*update/length

def cpu(value):
 if isinstance(value,torch.Tensor):return value.detach().cpu().contiguous()
 if isinstance(value,dict):return {k:cpu(v)for k,v in value.items()}
 if isinstance(value,list):return [cpu(v)for v in value]
 if isinstance(value,tuple):return tuple(cpu(v)for v in value)
 return value

def rng_state():
 n=np.random.get_state()
 return {'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all()if torch.cuda.is_available()else[],
         'python':random.getstate(),'numpy':(n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4]))}

def restore_rng(rng,device):
 torch.set_rng_state(rng['torch_cpu'])
 if device.type=='cuda':torch.cuda.set_rng_state_all(rng['torch_cuda'])
 random.setstate(rng['python']);n=rng['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))

def schedule_state(state,cfg):
 phase='branch'if state['phase'].startswith('branch')else'main'
 return {'format':'wsd-main-and-side-linear-v1','mode':phase,'main_updates':state['main_updates'],
         'branch_origin':state['branch_origin'],'branch_updates':state['branch_updates'],
         'branch_length':state['branch_length'],'last_lr':state['last_lr'],
         'warmup_updates':cfg['warmup_updates'],'main_cooldown_start':cfg['main_cooldown_start'],
         'main_cooldown_updates':cfg['main_cooldown_updates'],'peak_lr':cfg['peak_lr'],'floor_lr':cfg['floor_lr']}

class Checkpoints:
 def __init__(self,drive,local,identity):
  self.drive,self.local=Path(drive),Path(local);self.identity=identity
  self.drive.mkdir(parents=True,exist_ok=True);self.local.mkdir(parents=True,exist_ok=True)
 def save(self,model,opt,scaler,state,meta,cfg,spec,role='periodic',latest=True):
  core.sync(next(model.parameters()).device);started=time.perf_counter();name=f'checkpoints/{role}-{meta["compute_updates"]:06d}-{uuid.uuid4().hex[:10]}.pt'
  m=cpu(model.state_dict());digest=evaluation.state_digest(m)
  payload={'format':'nlp-hw02-wsd-v1','experiment':26,'identity':self.identity,'config':cfg,'model_spec':spec,
    'model':m,'optimizer':cpu(opt.state_dict()),'scaler':scaler.state_dict(),'scaler_enabled':scaler.is_enabled(),
    'scheduler':schedule_state(state,cfg),'rng':cpu(rng_state()),'training':copy.deepcopy(state),'experiment_history':copy.deepcopy(meta),
    'model_tensor_sha256':digest,'purpose':'CPU_smoke'if self.identity['smoke']else'Colab_T4_training'}
  target=self.local/name;target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.pt.tmp');torch.save(payload,tmp);tmp.replace(target)
  digest_file=sha_file(target);copy_verified(target,self.drive/name,digest_file)
  receipt={'file':name,'sha256':digest_file,'bytes':target.stat().st_size,'model_tensor_sha256':digest,
           'main_updates':state['main_updates'],'branch_updates':state['branch_updates'],'phase':state['phase'],
           'data_cursor':state['data_cursor'],'role':role,'identity':self.identity}
  write_json(self.drive/(name+'.json'),receipt)
  if latest:write_json(self.drive/'latest.json',receipt)
  meta['checkpoint_io_seconds']+=time.perf_counter()-started
  return receipt
 def local_file(self,receipt):
  assert receipt['identity']==self.identity
  source=self.drive/receipt['file'];assert source.stat().st_size==receipt['bytes']
  target=self.local/receipt['file'];copy_verified(source,target,receipt['sha256']);return target
 def load(self,receipt,device):
  file=self.local_file(receipt);p=torch.load(file,map_location='cpu',weights_only=True)
  assert p['format']=='nlp-hw02-wsd-v1'and p['identity']==self.identity
  cfg=p['config'];model=core.SimpleTransformer(**p['model_spec'],dropout=cfg['dropout']).to(device);model.load_state_dict(p['model'],strict=True)
  assert evaluation.state_digest(model.state_dict())==p['model_tensor_sha256']==receipt['model_tensor_sha256']
  fused=not self.identity['smoke'];opt=torch.optim.AdamW(model.parameters(),lr=cfg['peak_lr'],weight_decay=cfg['weight_decay'],**({'fused':True}if fused else{}))
  opt.load_state_dict(p['optimizer']);scaler=torch.amp.GradScaler(device.type,enabled=p['scaler_enabled'],init_scale=1024);scaler.load_state_dict(p['scaler'])
  state,meta=p['training'],p['experiment_history'];assert p['scheduler']==schedule_state(state,cfg)
  if state['main_updates']or state['branch_updates']:assert math.isclose(opt.param_groups[0]['lr'],state['last_lr'],rel_tol=1e-12)
  assert state['data_cursor']==cfg['batch']*(state['main_updates']+state['branch_updates'])
  steps=state['main_updates']+state['branch_updates']-state['AMP_skips']
  assert all(int(s['step'])==steps for s in opt.state.values())
  restore_rng(p['rng'],device);model.train();del p;return model,opt,scaler,state,meta


def sync_folder(local,drive):
 local,drive=Path(local),Path(drive)
 for p in sorted(local.rglob('*')):
  if p.is_file()and not p.name.endswith('.tmp'):copy_verified(p,drive/p.relative_to(local),overwrite=True)


def initialize(initial,device,cfg,spec,smoke=False):
 if not smoke:assert sha_file(initial)==P.INITIAL_SHA
 model=evaluation.load_model(initial,device,dropout=cfg['dropout'],scaled_tokens=True,spec=spec)
 if not smoke:assert evaluation.state_digest(model.state_dict())==P.NORMALIZED_INITIAL_TENSORS
 random.seed(P.SEED);np.random.seed(P.SEED);torch.manual_seed(P.SEED)
 if device.type=='cuda':torch.cuda.manual_seed_all(P.SEED)
 model.train();opt=torch.optim.AdamW(model.parameters(),lr=cfg['peak_lr'],weight_decay=cfg['weight_decay'],**({'fused':True}if not smoke else{}))
 amp=device.type=='cuda';scaler=torch.amp.GradScaler(device.type,enabled=amp,init_scale=1024)
 state={'phase':'main','main_updates':0,'branch_origin':None,'branch_updates':0,'branch_length':0,'data_cursor':0,
        'counts':{},'last_lr':0.,'active_anchor':None,'AMP_skips':0}
 meta={'compute_updates':0,'AMP_skips':0,'finite_loss_and_grad':True,'all_counts':{},'train_seconds':0.,'checkpoint_io_seconds':0.,
       'completed_points':[],'point_training':{},'logs':[],'sessions':[], 'finite_loss':True,'finite_successful_grad':True}
 return model,opt,scaler,state,meta


def train_chunk(model,opt,scaler,state,meta,cfg,cache,order,device,updates,interrupt_fixture=None):
 core.sync(device);started=time.perf_counter();records=[];counts=Counter()
 for _ in range(updates):
  cursor=state['data_cursor'];end=cursor+cfg['batch']
  if end>len(order):raise ValueError('No repeats before exhaustion; cache too small for the specified trajectory')
  src,tgt,labels,c=collate(cache,order[cursor:end],device);opt.zero_grad(set_to_none=True)
  branch=state['phase']=='branch';step=state['branch_updates']+1 if branch else state['main_updates']+1
  lr=branch_lr(step,state['branch_length'],cfg)if branch else main_lr(step,cfg)
  for g in opt.param_groups:g['lr']=lr
  with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=scaler.is_enabled()):
   logits=model(src,tgt);loss=F.cross_entropy(logits.reshape(-1,model.vocab),labels.reshape(-1),ignore_index=core.PAD)
  if interrupt_fixture==meta['compute_updates']+1:raise KeyboardInterrupt('Uncommitted-update fixture')
  if scaler.is_enabled():
   scaler.scale(loss).backward();scaler.unscale_(opt);found=scaler._found_inf_per_device(opt)
   assert len(found)==1;skipped=next(iter(found.values())).detach()
  else:loss.backward();skipped=loss.new_zeros(())
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip'])
  records.append(torch.stack([loss.detach().float(),torch.isfinite(loss.detach()).float(),torch.isfinite(norm.detach()).float(),skipped.float()]))
  if scaler.is_enabled():scaler.step(opt);scaler.update()
  else:opt.step()
  state['branch_updates'if branch else'main_updates']=step;state['data_cursor']=end;state['last_lr']=lr
  counts.update(c);meta['compute_updates']+=1
  del src,tgt,labels,logits,loss,norm
 core.sync(device);elapsed=time.perf_counter()-started;records=torch.stack(records).cpu().tolist()
 skips=sum(int(r[3]>0)for r in records);loss_finite=all(bool(r[1])for r in records)
 successful_grad_finite=all(bool(r[2])or r[3]>0 for r in records)
 meta['AMP_skips']+=skips;state['AMP_skips']+=skips
 meta['finite_loss_and_grad']&=all(bool(r[1])and bool(r[2])for r in records)
 meta['finite_loss']&=loss_finite;meta['finite_successful_grad']&=successful_grad_finite
 if not loss_finite or not successful_grad_finite:raise FloatingPointError('Nonfinite loss/unsafe successful gradient; resume last verified checkpoint')
 for destination in (state['counts'],meta['all_counts']):
  for k,v in counts.items():destination[k]=destination.get(k,0)+v
 meta['train_seconds']+=elapsed
 return {'phase':state['phase'],'main_update':state['main_updates'],'branch_update':state['branch_updates'],'lr':lr,
         'CE_last_chunk':sum(r[0]for r in records)/len(records),'AMP_skips_chunk':skips,'loss_scale':scaler.get_scale(),'seconds_chunk':elapsed,'pairs_per_second':counts['pairs']/elapsed,
         'source_tokens_per_second':counts['source_tokens']/elapsed,'target_tokens_per_second':counts['target_tokens']/elapsed,'compute_updates':meta['compute_updates']}


def run_training(cache,order,order_meta,inputs,drive,local,code_sha,device,evaluate_callback,
                 stop_main_updates=50000,checkpoint_seconds=600,session_minutes=120,smoke=False,cfg=None,spec=None,
                 max_updates_this_call=None,interrupt_fixture=None):
 cfg=copy.deepcopy(cfg or P.TRAIN_PROTOCOL);spec=spec or core.MODEL_SPEC
 if not smoke:
  assert cfg==P.TRAIN_PROTOCOL and spec==core.MODEL_SPEC and device.type=='cuda'
  assert 'T4'in torch.cuda.get_device_name(device)and core.fused_adamw_available(device)
  assert len(cache)>=7000000 and cfg['batch']*cfg['main_updates']<len(cache)
  assert stop_main_updates in (6250,12500,25000,50000)
 identity={'protocol_sha256':json_hash(cfg),'code_sha256':code_sha,'cache_manifest_sha256':cache.manifest_sha,
           'order_sha256':order_meta['file']['sha256'],'tokenizer_sha256':sha_file(Path(inputs)/'tokenizer.json'),
           'initial_sha256':sha_file(Path(inputs)/'initial.pt'),'smoke':smoke,'fused_adamw':not smoke,
           'torch':str(torch.__version__),'CUDA':torch.version.cuda}
 drive,local=Path(drive),Path(local);mgr=Checkpoints(drive,local,identity)
 latest=drive/'latest.json';started=time.perf_counter();last_checkpoint=started;updates_this_call=0
 if latest.exists():
  model,opt,scaler,state,meta=mgr.load(read_json(latest),device);print('RESUME',state['phase'],state['main_updates'],state['branch_updates'],'cursor',state['data_cursor'],flush=True)
 else:
  model,opt,scaler,state,meta=initialize(Path(inputs)/'initial.pt',device,cfg,spec,smoke)
  mgr.save(model,opt,scaler,state,meta,cfg,spec,role='initial');last_checkpoint=time.perf_counter()
 initial_train_seconds=meta['train_seconds'];meta['sessions'].append({'session_index':len(meta['sessions'])+1,'start_compute_updates':meta['compute_updates'],'requested_main_stop':stop_main_updates})
 try:
  while True:
   # Evaluate a completed branch/final checkpoint before advancing the main cursor.
   if state['phase']in ('branch_eval','final_eval'):
    origin=state['branch_origin']if state['phase']=='branch_eval'else cfg['main_updates'];tag=f'p{origin:06d}'
    receipt_path=drive/'points'/tag/'checkpoint.json'
    if receipt_path.exists():point_receipt=read_json(receipt_path)
    else:
     point_receipt=mgr.save(model,opt,scaler,state,meta,cfg,spec,role=tag);write_json(receipt_path,point_receipt)
    assert point_receipt['model_tensor_sha256']==evaluation.state_digest(model.state_dict())
    prev=meta['completed_points'][-1]if meta['completed_points']else None
    metrics=evaluate_callback(mgr.local_file(point_receipt),tag,prev)
    point_info={'nominal_main_updates':origin,'nominal_pairs':origin*cfg['batch'],
       'actual_updates':state['main_updates']+state['branch_updates'],'successful_optimizer_steps':state['main_updates']+state['branch_updates']-state['AMP_skips'],'AMP_skips_trajectory':state['AMP_skips'],'actual_pairs':state['data_cursor'],
       'cooldown_updates':state['branch_length']if state['phase']=='branch_eval'else cfg['main_cooldown_updates'],
       'counts':copy.deepcopy(state['counts']),'data':prefix_identity(order,state['data_cursor'],len(cache)),
       'compute_updates_total':meta['compute_updates'],'train_seconds_total':meta['train_seconds'],'metrics':metrics}
    meta['point_training'][tag]=point_info
    if tag not in meta['completed_points']:meta['completed_points'].append(tag)
    write_json(drive/'points'/tag/'training.json',point_info)
    if state['phase']=='final_eval':
     state['phase']='complete';complete_receipt=mgr.save(model,opt,scaler,state,meta,cfg,spec,role='complete');write_json(drive/'final-checkpoint.json',complete_receipt);break
    # Preserve experiment costs, but restore every component of the main training trajectory.
    anchor=state['active_anchor'];saved_meta=meta
    del model,opt,scaler;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    model,opt,scaler,state,_=mgr.load(anchor,device);meta=saved_meta
    assert state['phase']=='main'and state['data_cursor']==origin*cfg['batch']
    mgr.save(model,opt,scaler,state,meta,cfg,spec,role='main-restored');last_checkpoint=time.perf_counter();continue
   if state['phase']=='complete':break
   if state['phase']=='main':
    tag=f'p{state["main_updates"]:06d}'
    if state['main_updates']in cfg['branch_lengths']and tag not in meta['completed_points']:
     anchor_path=drive/'anchors'/f'{state["main_updates"]:06d}.json'
     if anchor_path.exists():
      anchor=read_json(anchor_path);assert anchor['model_tensor_sha256']==evaluation.state_digest(model.state_dict())
     else:
      anchor=mgr.save(model,opt,scaler,state,meta,cfg,spec,role='anchor',latest=False);write_json(anchor_path,anchor)
     state.update(phase='branch',branch_origin=state['main_updates'],branch_updates=0,branch_length=cfg['branch_lengths'][state['main_updates']],active_anchor=anchor)
     mgr.save(model,opt,scaler,state,meta,cfg,spec,role='branch-start');last_checkpoint=time.perf_counter();continue
    if state['main_updates']==cfg['main_updates']:
     state['phase']='final_eval';r=mgr.save(model,opt,scaler,state,meta,cfg,spec,role='final-evaluation-pending');write_json(drive/'points'/f'p{cfg["main_updates"]:06d}'/'checkpoint.json',r);continue
    if state['main_updates']>=stop_main_updates:
     mgr.save(model,opt,scaler,state,meta,cfg,spec,role='requested-stop');break
   if time.perf_counter()-started>=session_minutes*60 or(max_updates_this_call is not None and updates_this_call>=max_updates_this_call):
    mgr.save(model,opt,scaler,state,meta,cfg,spec,role='session-stop');break
   if state['phase']=='branch':remaining=state['branch_length']-state['branch_updates']
   else:
    future=[v for v in cfg['branch_lengths']if v>state['main_updates']]+[cfg['main_updates'],stop_main_updates]
    remaining=min(future)-state['main_updates']
   n=min(25 if not smoke else 1,remaining)
   if max_updates_this_call is not None:n=min(n,max_updates_this_call-updates_this_call)
   log=train_chunk(model,opt,scaler,state,meta,cfg,cache,order,device,n,interrupt_fixture)
   updates_this_call+=n
   if state['phase']=='branch'and state['branch_updates']==state['branch_length']:
    state['phase']='branch_eval';r=mgr.save(model,opt,scaler,state,meta,cfg,spec,role='branch-evaluation-pending');write_json(drive/'points'/f'p{state["branch_origin"]:06d}'/'checkpoint.json',r);continue
   if meta['compute_updates']%(1000 if not smoke else 1)==0:
    log['unique_seen']=state['data_cursor'];log['unique_fraction_of_cache']=state['data_cursor']/len(cache)
    meta['logs'].append(log);print('TRAIN',json.dumps(log),flush=True)
    write_json(drive/'live-progress.json',dict(log,session_elapsed_seconds=time.perf_counter()-started,train_seconds_total=meta['train_seconds'],resume_from=read_json(latest)['file']))
   if time.perf_counter()-last_checkpoint>=checkpoint_seconds:
    mgr.save(model,opt,scaler,state,meta,cfg,spec);last_checkpoint=time.perf_counter()
 except (KeyboardInterrupt,FloatingPointError,torch.cuda.OutOfMemoryError)as error:
  # A Python interrupt can land inside a fused update. Never serialize such live state.
  write_json(drive/'interruption.json',{'error':type(error).__name__,'message':str(error)[:300],
      'resume_from':read_json(latest)['file'],'uncommitted_work_may_be_repeated':True})
  print('INTERRUPTED: restart from the last verified Drive checkpoint; uncommitted chunk discarded',flush=True)
  if not isinstance(error,KeyboardInterrupt):raise
 finally:
  session={'wall_seconds':time.perf_counter()-started,'new_train_seconds':meta['train_seconds']-initial_train_seconds,
           'updates_this_call':updates_this_call,'checkpoint_file':read_json(latest)['file']}
  write_json(drive/f'session-{uuid.uuid4().hex[:10]}.json',session)
  del model,opt,scaler;gc.collect()
  if device.type=='cuda':torch.cuda.empty_cache()
 return {'state':state,'history':meta,'latest':read_json(latest),'session':session,'complete':state['phase']=='complete'}


def decisions(points,D0,cfg=None):
 cfg=cfg or P.TRAIN_PROTOCOL;keys=[f'p{x:06d}'for x in sorted(cfg['branch_lengths'])]+[f'p{cfg["main_updates"]:06d}']
 if any(k not in points for k in keys):return {'status':'incomplete; no slope/quality decision until all4 points evaluated'}
 get=lambda k:points[k]['splits']['heldout4k']['exec']['accuracy']
 acc=[get(k)for k in keys];base=D0['splits']['heldout4k']['exec']['accuracy']
 delta0=100*(acc[0]-base);slope=100*(acc[-1]-acc[0])/3.;last=100*(acc[-1]-acc[-2])
 return {'status':'complete','compatibility_0_8M_delta_pp':delta0,'compatible_with_D0_primary4k':abs(delta0)<=1.,
         'mean_nominal_slope_pp_per_doubling':slope,'data_compute_remain_main_lever_by_rule':slope>=2.,
         'last_nominal_segment_delta_pp':last,'capacity_hint_next_model_B_by_rule':last<1.,
         'actual_exposure_mean_slope_pp_per_doubling':100*(acc[-1]-acc[0])/math.log2(6.4/.96),
         'actual_exposure_last_slope_pp_per_doubling':last/math.log2(6.4/3.84),
         'interpretation':'Thresholds are registered heuristics, not proof of capacity. One seed; filtered data/global permutation/WSD differ from25; fresh untouched.'}
