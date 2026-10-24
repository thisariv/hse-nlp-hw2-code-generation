#!/usr/bin/env python3
"""Audit all downloaded L1 review v2 files and full resume state, without CUDA."""
import ast,gc,hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'kaggle/src'))
from kaggle_io import sha,write
import l1_runtime as L,torch
out=root/'results/kaggle-runs/nlp-hw02-l1-gpu-smoke/2'
manifest=json.loads((out/'l1-output-manifest.json').read_text())
omitted=[]
for rec in manifest['files']:
 p=out/rec['file']
 if not p.exists() and rec['file']=='__notebook__.ipynb':
  omitted.append({'file':rec['file'],'reason':'Kaggle service notebook is not exposed by output API; source verified via pinned pull'});continue
 assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],rec['file']
summary=json.loads((out/'session-summary.json').read_text());assert summary['status']=='complete' and summary['exit_codes']==[0,0]
models={}
for name in ('C','D'):
 model_root=out/f'model-{name}';identity=json.loads((model_root/'identity.json').read_text())
 receipt=json.loads((model_root/'latest.json').read_text());assert sha(model_root/receipt['file'])==receipt['sha256']
 files=sorted((model_root/'checkpoints').glob('*.pt'));assert len(files)==2
 checkpoints=[]
 for p in files:
  payload=L.read_full(p,identity);state=payload['state']
  assert state['updates'] in (256,288) and state['cursor']==128*state['updates']
  assert len(payload['rng']['torch_cuda'])==1 and payload['scaler_enabled'] is True
  assert payload['scaler'] and len(state['points'])==1 and state['branch_attempted_updates']==19
  assert all(v.dtype==torch.float32 for v in payload['model'].values() if v.is_floating_point())
  checkpoints.append({'file':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'updates':state['updates'],'cursor':state['cursor'],
                       'optimizer_states':len(payload['optimizer']['state']),'CUDA_RNG_states':1,'FP32_full_main':True})
  del payload;gc.collect()
 latest=L.read_full(model_root/receipt['file'],identity);assert latest['state']['updates']==288 and latest['state']['AMP_skips']==0
 assert latest['state']['last_lr']==L.PEAK_LR[name]*288/1000
 del latest;gc.collect()
 r=json.loads((model_root/'gpu-smoke.json').read_text());assert r['passed'] and r['CUDA_late_OOM_regression']['passed']
 assert r['CUDA_late_OOM_regression']['max_weight_difference_from_clean_update']==0.
 test=r['points'][0]['metrics']['test'];assert test['rows']==256 and test['accuracy'] is None and not test['cache_reuse']
 assert test['generation']['batch']==256 and test['generation']['SDPA_dimensions']==4 and test['generation']['precision']=='FP16'
 weights=list((model_root/'weights').glob('*.pt'));assert len(weights)==1
 p=torch.load(weights[0],map_location='cpu',weights_only=True)
 assert p['optimizer_included'] is False and all(v.dtype==torch.float16 for v in p['model'].values() if v.is_floating_point())
 del p;gc.collect()
 log=(out/('gpu0-process.log' if name=='C' else 'gpu1-process.log')).read_text()
 assert 'GPU SMOKE PASSED '+name in log and 'Traceback' not in log and 'WORKER FAILED' not in log
 models[name]={k:r[k] for k in ('parameter_count','micro_batch','peak_reserved_bytes','AMP_skips','phases','CUDA_late_OOM_regression')}
 models[name].update(checkpoints=checkpoints,test256_full_pipeline_seconds=test['full_pipeline_seconds'],test256_greedy_seconds=test['generation']['seconds'],
                      effective_batch=128,main_updates=288,cursor=36864,cooldown_updates=19)
receipt=json.loads((out/'download-version.json').read_text());assert not receipt['metadata_only'] and receipt['PT_skipped']==0
result={'passed':True,'version_ref':'thisarivson/nlp-hw02-l1-gpu-smoke/2','server_status':'COMPLETE','internal_status':'complete','exit_codes':[0,0],
 'wall_seconds':summary['wall_seconds'],'manifest_files_verified':len(manifest['files'])-len(omitted),'manifest_service_files_unavailable':omitted,'downloaded_files':receipt['files'],
 'output_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),'full_states_and_weights_downloaded':True,
 'models':models,'source_static_check':'kaggle-l1-review-static.json','production_L1_launched':False}
write(root/'checks/kaggle-l1-review-gpu-smoke.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
