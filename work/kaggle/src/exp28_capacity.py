"""Capacity B: equal cumulative pure training time incl reversible cooldown costs."""
import copy,gc,math,random,time
from pathlib import Path
import numpy as np
import torch
import exp26_core as C
import exp26_train as T
import exp26_eval as E
import exp26_protocol as P
from exp26_common import write_json,read_json,json_hash,sha_file
from exp26_order import prefix_identity

SPEC={'d_model':384,'heads':6,'encoder_layers':6,'decoder_layers':6,'swiglu_dim':1024}
M_SPEC={'d_model':256,'heads':4,'encoder_layers':6,'decoder_layers':6,'swiglu_dim':680}
TARGETS=[{'A_point':'p006250','seconds':928.1166084509971,'actual_pairs':960000},
         {'A_point':'p012500','seconds':2018.225873703993,'actual_pairs':1920000},
         {'A_point':'p025000','seconds':4203.38492590799,'actual_pairs':3840000}]

def config(batch=128,experiment=28,spec=None,peak_lr=None):
    cfg=copy.deepcopy(P.TRAIN_PROTOCOL)
    spec=spec or (M_SPEC if experiment==29 else SPEC);peak_lr=peak_lr or (.003 if experiment==29 else .002)
    cfg.update(experiment=experiment,model=spec,batch=batch,peak_lr=peak_lr,floor_lr=peak_lr/10,branch_lengths={})
    return cfg

def initial_state(path,device,cfg,spec,smoke=False):
    random.seed(P.SEED);np.random.seed(P.SEED);torch.manual_seed(P.SEED)
    if device.type=='cuda':torch.cuda.manual_seed_all(P.SEED)
    model=C.SimpleTransformer(**spec,dropout=0.)
    with torch.no_grad():model.tokens.weight.div_(math.sqrt(model.width))
    if not smoke and cfg['experiment']==28:assert sum(p.numel() for p in model.parameters())==31940096
    if not Path(path).exists():torch.save(model.state_dict(),path)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.to(device).train()
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['peak_lr'],weight_decay=cfg['weight_decay'],**({'fused':True} if not smoke else {}))
    scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda',init_scale=1024)
    state={'phase':'main','main_updates':0,'branch_origin':None,'branch_updates':0,'branch_length':0,'data_cursor':0,
           'counts':{},'last_lr':0.,'active_anchor':None,'AMP_skips':0}
    meta={'compute_updates':0,'AMP_skips':0,'finite_loss_and_grad':True,'finite_loss':True,'finite_successful_grad':True,
          'all_counts':{},'train_seconds':0.,'checkpoint_io_seconds':0.,'completed_points':[],'point_training':{},'logs':[],
          'sessions':[],'capacity':{'calibration_steps':500,'targets':TARGETS,'batch':cfg['batch'],'LR_attempts':[cfg['peak_lr']],
                                  'parameters':sum(p.numel() for p in model.parameters()),
                                  'time_scope':'cumulative pure train_chunk time, all main and side compute; setup/eval/CP excluded'}}
    return model,opt,scaler,state,meta

def projected_plan(step_seconds,targets,calibration=500):
    side=0;plan=[]
    for t in targets:
        anchor=max(calibration,math.floor((t['seconds']/step_seconds-side)/1.2))
        cooldown=max(1,round(.2*anchor));side+=cooldown
        plan.append(dict(t,projected_main_updates=anchor,projected_cooldown_updates=cooldown))
    return plan

def run(cache,order,order_meta,inputs,root,port_sha,device,evaluate,deadline,cfg=None,spec=None,smoke=False,
        targets=None,calibration=500,max_compute=None,checkpoint_seconds=600,stop_after_targets=False,allow_batch_fallback=True):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);targets=targets or TARGETS;spec=spec or SPEC;cfg=cfg or config()
    if (root/'latest.json').exists():
        receipt=read_json(root/'latest.json');payload=torch.load(root/receipt['file'],map_location='cpu',weights_only=True)
        cfg=payload['config'];del payload
    initial=root/('initial-M.pt' if cfg['experiment']==29 else 'initial-B.pt')
    if not initial.exists():
        initial_result=initial_state(initial,device,cfg,spec,smoke);del initial_result;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    identity={'protocol_sha256':json_hash(cfg),'code_sha256':port_sha,'cache_manifest_sha256':cache.manifest_sha,
              'order_sha256':order_meta['file']['sha256'],'tokenizer_sha256':sha_file(Path(inputs)/'tokenizer.json'),
              'initial_sha256':sha_file(initial),'smoke':smoke,'fused_adamw':not smoke,'torch':str(torch.__version__),'CUDA':torch.version.cuda}
    mgr=T.Checkpoints(root,root,identity)
    if (root/'latest.json').exists():model,opt,scaler,state,meta=mgr.load(read_json(root/'latest.json'),device)
    else:
        model,opt,scaler,state,meta=initial_state(initial,device,cfg,spec,smoke)
        meta['capacity'].update(calibration_steps=calibration,targets=targets)
        mgr.save(model,opt,scaler,state,meta,cfg,spec,role='initial')
    last_checkpoint=time.monotonic();start=time.monotonic();initial_compute=meta['compute_updates']
    try:
        while True:
            if state['phase']=='branch_eval':
                idx=len(meta['completed_points']);target=targets[idx];tag=f't{idx+1:02d}';base=root/'points'/tag
                if (base/'checkpoint.json').exists():
                    receipt=read_json(base/'checkpoint.json')
                    assert receipt['model_tensor_sha256']==E.state_digest(model.state_dict())
                else:
                    receipt=(mgr.save_weights(model,state,meta,cfg,spec,tag) if stop_after_targets
                             else mgr.save(model,opt,scaler,state,meta,cfg,spec,role=tag))
                    write_json(base/'checkpoint.json',receipt)
                error=meta['train_seconds']-target['seconds']
                info={'point':tag,'A_point':target['A_point'],'target_train_seconds':target['seconds'],'actual_train_seconds':meta['train_seconds'],
                      'alignment_error_seconds':error,'time_aligned':abs(error)<=max(.02*target['seconds'],5.),
                      'main_updates':state['main_updates'],'cooldown_updates':state['branch_updates'],'actual_pairs':state['data_cursor'],
                      'compute_updates_total':meta['compute_updates'],'AMP_skips':meta['AMP_skips'],'batch':cfg['batch'],
                      'data':prefix_identity(order,state['data_cursor'],len(cache))}
                # Alignment survives an interruption after metrics but before the controller summary.
                if stop_after_targets:write_json(base/'training.json',dict(info,evaluation_status='pending'))
                metrics=evaluate(root/receipt['file'],tag,meta['completed_points'][-1] if meta['completed_points'] else None,target)
                info['metrics']=metrics;meta['point_training'][tag]=info;meta['completed_points'].append(tag)
                write_json(base/'training.json',info)
                if stop_after_targets:
                    write_json(root/f'{cfg["experiment"]}-summary.json',{'experiment':cfg['experiment'],'status':'partial; training ongoing',
                        'points':meta['point_training'],'capacity':meta['capacity'],'train_seconds_total':meta['train_seconds'],
                        'latest':read_json(root/'latest.json'),'fresh_encoded':0,'fresh_evaluated':0,
                        'missing_points':[t for i,t in enumerate(targets) if f't{i+1:02d}' not in meta['point_training']]})
                    # One complete evaluated final state replaces the preceding point's AdamW.
                    mgr.save(model,opt,scaler,state,meta,cfg,spec,role='final-point',latest=False)
                anchor=state['active_anchor'];saved_meta=meta;del model,opt,scaler;gc.collect()
                if device.type=='cuda':torch.cuda.empty_cache()
                model,opt,scaler,state,_=mgr.load(anchor,device);meta=saved_meta
                # Stable latest restores the main path; evaluated weights remain separate.
                state['active_anchor']=None
                mgr.save(model,opt,scaler,state,meta,cfg,spec,role='main-restored');last_checkpoint=time.monotonic();continue
            if stop_after_targets and len(meta['completed_points'])==len(targets):break
            if time.monotonic()>=deadline or (max_compute is not None and meta['compute_updates']-initial_compute>=max_compute):break
            if state['main_updates']>=cfg['main_updates'] or state['data_cursor']+cfg['batch']>len(order):break
            if meta['compute_updates']>=calibration and 'seconds_per_update' not in meta['capacity']:
                rate=meta['train_seconds']/meta['compute_updates'];meta['capacity']['seconds_per_update']=rate
                meta['capacity']['projected_points']=projected_plan(rate,targets,calibration)
                write_json(root/'500-step-calibration.json',meta['capacity']);print('B CALIBRATION',meta['capacity'],flush=True)
                mgr.save(model,opt,scaler,state,meta,cfg,spec,role='calibrated')
            n=1 if smoke else 25
            if state['phase']=='branch':n=min(n,state['branch_length']-state['branch_updates'])
            else:
                n=min(n,cfg['main_updates']-state['main_updates'],max(1,calibration-meta['compute_updates'])) if meta['compute_updates']<calibration else n
                idx=len(meta['completed_points'])
                if idx<len(targets) and 'seconds_per_update' in meta['capacity']:
                    rate=meta['train_seconds']/meta['compute_updates'];remaining=targets[idx]['seconds']-meta['train_seconds']
                    cooldown=max(1,round(.2*state['main_updates']))
                    until=max(0,math.floor(remaining/rate-cooldown))
                    if until<=0:
                        anchor=mgr.save(model,opt,scaler,state,meta,cfg,spec,role='anchor',latest=False)
                        length=max(1,round(remaining/rate))
                        state.update(phase='branch',branch_origin=state['main_updates'],branch_updates=0,branch_length=length,active_anchor=anchor)
                        meta['capacity'].setdefault('actual_forks',[]).append({'main_update':state['main_updates'],'cooldown_updates':length,
                            'target_seconds':targets[idx]['seconds'],'running_seconds_per_update':rate})
                        mgr.save(model,opt,scaler,state,meta,cfg,spec,role='branch-start');last_checkpoint=time.monotonic();continue
                    n=min(n,until)
            if max_compute is not None:n=min(n,max_compute-(meta['compute_updates']-initial_compute))
            try:log=T.train_chunk(model,opt,scaler,state,meta,cfg,cache,order,device,n)
            except torch.cuda.OutOfMemoryError:
                if not smoke and allow_batch_fallback and cfg['batch']==128 and meta['compute_updates']<calibration:
                    # Discard uncommitted live tensors; one batch fallback, no LR search.
                    del model,opt,scaler;gc.collect();torch.cuda.empty_cache()
                    cfg=config(96,cfg['experiment'],spec,cfg['peak_lr']);model,opt,scaler,state,meta=initial_state(initial,device,cfg,spec,smoke)
                    meta['capacity'].update(calibration_steps=calibration,targets=targets)
                    meta['capacity']['batch_fallback']={'from':128,'to':96,'reason':'OOM in first500','discarded_unsafe_prefix':True}
                    identity.update(protocol_sha256=json_hash(cfg))
                    # Existing128 receipt is no longer a valid96 resume. Preserve provenance separately.
                    write_json(root/'batch128-oom.json',read_json(root/'latest.json'))
                    # S2 retains the fallback receipt and prunes obsolete full states.
                    if not stop_after_targets:(root/'latest.json').unlink()
                    mgr=T.Checkpoints(root,root,identity);mgr.save(model,opt,scaler,state,meta,cfg,spec,role='batch96-restart');continue
                raise
            if state['phase']=='branch' and state['branch_updates']==state['branch_length']:
                state['phase']='branch_eval';mgr.save(model,opt,scaler,state,meta,cfg,spec,role='branch-evaluation-pending');continue
            if meta['compute_updates']%1000<n or smoke:
                print('B TRAIN',json_hash(log)[:8],state['main_updates'],meta['train_seconds'],flush=True)
                write_json(root/'live-progress.json',dict(log,train_seconds_total=meta['train_seconds'],elapsed_seconds=time.monotonic()-start))
            if time.monotonic()-last_checkpoint>=checkpoint_seconds:
                mgr.save(model,opt,scaler,state,meta,cfg,spec);last_checkpoint=time.monotonic()
        receipt=mgr.save(model,opt,scaler,state,meta,cfg,spec,role='session-stop')
        report={'experiment':cfg['experiment'],'status':'complete' if stop_after_targets and len(meta['completed_points'])==len(targets) else 'partial; main resumable',
                'latest':receipt,'capacity':meta['capacity'],'points':meta['point_training'],
                'train_seconds_total':meta['train_seconds'],'main_updates':state['main_updates'],'all_compute_updates':meta['compute_updates'],
                'phase':state['phase'],'fresh_encoded':0,'fresh_evaluated':0,
                'missing_points':[t for i,t in enumerate(targets) if f't{i+1:02d}' not in meta['point_training']]}
        write_json(root/f'{cfg["experiment"]}-summary.json',report);return report
    finally:
        del model,opt,scaler;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()

def decision(points,a_metrics,test_timing):
    common=[(k,v) for k,v in points.items() if v['time_aligned'] and v['A_point'] in a_metrics]
    if not common:return {'status':'pending; no aligned common point','final_model':'A'}
    tag,latest=max(common,key=lambda kv:kv[1]['target_train_seconds'])
    b=latest['metrics']['splits']['heldout4k']['exec']['accuracy'];a=a_metrics[latest['A_point']]['splits']['heldout4k']['exec']['accuracy']
    delta=100*(b-a)
    if test_timing is None:return {'status':'pending; selected27 decoding timing for B not measured','point':tag,'delta_pp':delta,'final_model':'A pending test'}
    accepted=delta>=2 and test_timing['full_pipeline_seconds']<=120
    return {'status':'measured','accepted_B':accepted,'point':tag,'A_point':latest['A_point'],'delta_pp':delta,
            'full_test_seconds':test_timing['full_pipeline_seconds'],'final_model':'B' if accepted else 'A with more presentations'}
