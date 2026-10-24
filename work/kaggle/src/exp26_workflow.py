"""Colab evaluation orchestration; frozen input sets, Drive receipts, no fresh eval."""
import gc,math,time
from pathlib import Path
import torch
import exp26_protocol as P
import exp26_core as core
import exp26_eval as E
import exp26_exec as numeric
from exp26_common import read_json,write_json,sha_file,json_hash,copy_verified
from exp26_train import sync_folder


def initialize_experiment(drive,code_sha,cache,order_meta,inputs):
 drive=Path(drive);drive.mkdir(parents=True,exist_ok=True)
 identity={'code_sha256':code_sha,'protocol_sha256':json_hash(P.TRAIN_PROTOCOL),
    'cache_manifest_sha256':cache.manifest_sha,'order_sha256':order_meta['file']['sha256'],
    'input_files':P.INPUT_FILES,'torch':str(torch.__version__),'CUDA':torch.version.cuda}
 path=drive/'experiment-identity.json'
 if path.exists():assert read_json(path)==identity,'Existing experiment differs; preserve it and use another Drive folder'
 else:write_json(path,identity)
 write_json(drive/'protocol.json',P.TRAIN_PROTOCOL)
 write_json(drive/'input-identity.json',{'starter_sha256':P.STARTER_B_SHA,'files':P.INPUT_FILES,
    'D0_source_full_checkpoint_sha256':P.D0_SOURCE_CHECKPOINT_SHA,'D0_tensor_sha256':P.D0_TENSORS,
    'normalized_initial_tensor_sha256':P.NORMALIZED_INITIAL_TENSORS,'fresh':'question exclusion only'})
 copy_verified(cache.root/'cache-manifest.json',drive/'cache-manifest.json',cache.manifest_sha)
 return identity


def reference_execution(sets,inputs,drive):
 """Recheck reference finite indices and values once; subsequent sessions reuse verified receipts."""
 root=Path(drive)/'references';root.mkdir(parents=True,exist_ok=True);refs={}
 for split,rows in sets.items():
  expected=read_json(Path(inputs)/(split+'-reference-execution.json'))
  identity={'rows_sha256':json_hash(rows),'saved_reference_sha256':sha_file(Path(inputs)/(split+'-reference-execution.json')),
    'policy':'same25 sandbox finite solve bool->int rel=abs1e-6'}
  path=root/(split+'-reference-execution.json');rp=root/(split+'-receipt.json')
  if path.exists()and rp.exists():
   receipt=read_json(rp);assert receipt['identity']==identity and sha_file(path)==receipt['sha256'];execution=read_json(path)
  else:
   print('REFERENCE CPU',split,len(rows),'programs',flush=True)
   execution=numeric.evaluate_programs([r['code']for r in rows]);assert execution['summary']['complete']
   expected_count=4096 if split=='heldout4k'else 991
   assert sum(r['status']=='numeric'for r in execution['rows'])==expected_count
   comparison=numeric.compare_exec(expected,execution)
   assert comparison['matches']==comparison['denominator']==expected_count,'Reference values/finite indices changed'
   assert [i for i,r in enumerate(execution['rows'])if r['status']=='numeric']==comparison['eligible_indices']
   write_json(path,execution);write_json(rp,{'identity':identity,'sha256':sha_file(path),'gate':comparison})
  refs[split]=execution
 return refs


def restore_evaluation(drive,local):
 drive,local=Path(drive),Path(local)
 if not drive.exists():return
 local.mkdir(parents=True,exist_ok=True)
 for p in drive.rglob('*'):
  if p.is_file()and p.suffix in ('.json','.jsonl'):copy_verified(p,local/p.relative_to(drive),overwrite=True)


def prepare_D0(sets,refs,inputs,drive,local,device):
 drive,local=Path(drive),Path(local);restore_evaluation(drive/'D0',local/'D0')
 # The compact file changes serialization only; its model tensors must still be exactly D0.
 state=torch.load(Path(inputs)/'control-D0.pt',map_location='cpu',weights_only=True)
 assert E.state_digest(state)==P.D0_TENSORS;del state;gc.collect()
 print('D0 CONTROL: expected heldout4k1677/4096, old422/991; match tolerance3, CE tolerance1e-4',flush=True)
 m=E.evaluate_point(Path(inputs)/'control-D0.pt',sets,refs,Path(inputs)/'tokenizer.json',device,local/'D0','D0')
 gate=E.verify_D0(m);write_json(local/'D0'/'gate.json',gate);sync_folder(local/'D0',drive/'D0')
 print('D0 gate',gate,flush=True);return m


def point_callback(sets,refs,inputs,drive,local,device,D0):
 drive,local=Path(drive),Path(local)
 def evaluate(model_path,tag,previous):
  point_drive=drive/'points'/tag/'evaluation';point_local=local/'points'/tag/'evaluation'
  restore_evaluation(point_drive,point_local)
  previous_metrics=read_json(drive/'points'/previous/'evaluation'/'metrics.json')if previous is not None else None
  started=time.perf_counter();m=E.evaluate_point(model_path,sets,refs,Path(inputs)/'tokenizer.json',device,point_local,tag,D0,previous_metrics)
  write_json(point_local/'evaluation-session.json',{'wall_seconds':time.perf_counter()-started,'previous':previous})
  sync_folder(point_local,point_drive)
  print('POINT',tag,{s:{'matches':r['exec']['matches'],'denominator':r['exec']['denominator'],'CE':r['CE']['CE']}for s,r in m['splits'].items()},flush=True)
  return m
 return evaluate


def generate_final_test(test,inputs,drive,local,device,smoke=False):
 drive,local=Path(drive),Path(local);final_path=drive/'final-checkpoint.json'
 if not final_path.exists():return {'status':'not_complete; test not generated'}
 receipt=read_json(final_path);target=local/receipt['file'];copy_verified(drive/receipt['file'],target,receipt['sha256'])
 assert receipt['phase']=='complete'
 if not smoke:assert receipt['main_updates']==50000 and receipt['data_cursor']==6400000 and len(test)==1919
 out=local/'test';restore_evaluation(drive/'test',out);timing_path=out/'test-generation-time.json'
 if timing_path.exists()and (out/'test-answers.jsonl').exists()and(out/'test-generations.json').exists():
  timing=read_json(timing_path)
  assert timing['model_sha256']==receipt['sha256']and timing['tokenizer_sha256']==P.TOKENIZER_SHA
  assert timing['answers_sha256']==sha_file(out/'test-answers.jsonl')
  examples=read_json(out/'test-generations.json')
 else:
  examples,timing=E.generate_one(target,test,Path(inputs)/'tokenizer.json',device,out,'test',batch=2 if smoke else 512,cap=8 if smoke else 512,half=True)
  write_json(timing_path,timing)
 assert len(examples)==len(test) and [r['question']for r in examples]==[r['question']for r in test]and [r['id']for r in examples]==list(range(len(test)))
 write_json(out/'submission-candidate.json',{'answers':'test-answers.jsonl','rows':len(test),'smoke':smoke,'answers_sha256':timing['answers_sha256'],
    'checkpoint_sha256':receipt['sha256'],'full_seconds':timing['full_seconds'],'official_bot_format_verified':False})
 sync_folder(out,drive/'test');return timing


def collect_report(drive,local,cfg=None):
 """Only committed training state/point receipts; live interrupted tensors are never reported as saved."""
 cfg=cfg or P.TRAIN_PROTOCOL
 drive,local=Path(drive),Path(local);metrics={};table=[]
 for origin in sorted(cfg['branch_lengths'])+[cfg['main_updates']]:
  tag=f'p{origin:06d}';base=drive/'points'/tag
  if not(base/'training.json').exists()or not(base/'evaluation'/'metrics.json').exists():continue
  info=read_json(base/'training.json');m=read_json(base/'evaluation'/'metrics.json');metrics[tag]=m
  table.append({'point':tag,'nominal_M_pairs':info['nominal_pairs']/1e6,'actual_M_pairs':info['actual_pairs']/1e6,
    'exec4k':m['splits']['heldout4k']['exec']['matches'],'exec991':m['splits']['old1024']['exec']['matches'],
    'accuracy4k':m['splits']['heldout4k']['exec']['accuracy'],'CE4k':m['splits']['heldout4k']['CE']['CE'],
    'unique_fraction_of_cache':info['data']['unique_fraction_of_cache'],'target_tokens':info['counts']['target_tokens'],
    'compute_updates_total':info['compute_updates_total'],'train_seconds_total':info['train_seconds_total']})
 D0=read_json(drive/'D0'/'metrics.json');from exp26_train import decisions
 decision=decisions(metrics,D0,cfg);write_json(drive/'decision.json',decision)
 latest=read_json(drive/'latest.json');path=local/latest['file'];copy_verified(drive/latest['file'],path,latest['sha256'])
 payload=torch.load(path,map_location='cpu',weights_only=True);history=payload['experiment_history'];committed=payload['training'];del payload
 counters=history['all_counts'];seconds=history['train_seconds']
 totals={'committed_main_updates':committed['main_updates'],'committed_branch_updates':committed['branch_updates'],
    'phase':committed['phase'],'all_compute_updates':history['compute_updates'],'train_seconds':seconds,
    'pairs_per_second':counters.get('pairs',0)/seconds if seconds else None,
    'useful_source_tokens_per_second':counters.get('source_tokens',0)/seconds if seconds else None,
    'useful_target_tokens_per_second':counters.get('target_tokens',0)/seconds if seconds else None,
    'all_compute_presentations':counters.get('pairs',0),'checkpoint_io_seconds_committed':history['checkpoint_io_seconds'],
    'AMP_skips':history['AMP_skips'],'successful_optimizer_steps_all_compute':history['compute_updates']-history['AMP_skips'],
    'AMP_skips_final_trajectory':committed['AMP_skips'],'successful_optimizer_steps_final_trajectory':committed['main_updates']+committed['branch_updates']-committed['AMP_skips'],
    'finite_loss':history['finite_loss'],'finite_successful_grad':history['finite_successful_grad'],'finite_loss_and_grad':history['finite_loss_and_grad'],
    'unique_main_rows_seen':committed['main_updates']*cfg['batch'],'unique_trajectory_rows_seen':committed['data_cursor'],
    'recorded_session_wall_seconds':sum(read_json(p)['wall_seconds']for p in drive.glob('session-*.json')),
    'wall_timer_scope':'training controller, evaluations and checkpoint I/O per completed/interrupted Python call; setup/cache copy/D0 outside; kernel killed without finally has only heartbeat/last checkpoint',
    'uncommitted_work_after_last_checkpoint_may_be_repeated':True}
 report={'experiment':26,'table':table,'totals':totals,'decision':decision,'latest_checkpoint':latest,
   'protocol':cfg,'fresh_encoded':0,'fresh_executed':0,'GPU_budget_minutes_target':120,
   'baseline_D0_saved':{'exec4k':1677,'exec991':422,'CE4k':.4764168630129111}}
 write_json(drive/'summary.json',report);return report
