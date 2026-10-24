"""Independent small-output checks: source fingerprints, exact q0 LR and every mini-K8 sample. Run from kaggle/."""
import sys,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from l2_gpu_smoke import digest
from l2_sampler import fixed_split
from kaggle_io import write
folder=Path(sys.argv[1]);rows=[]
for g,peak in [('0',.0006),('1',.0015)]:
 root=folder/f'model-GPU{g}';summary=json.loads((root/'summary.json').read_text());cd=summary['cooldown']
 cpu=json.loads(Path(f'../checks/b6-real-resume-gpu{g}.json').read_text())
 expected=cpu['initial_full_state_digests'];start=json.loads((root/'start-fingerprints.json').read_text())['fingerprints']
 assert all(start[k]==expected[k] for k in ('model','optimizer','scaler','state'))
 assert digest(summary['main_state'])==expected['state']
 assert start['rng_torch_cuda']==cpu['restores'][0]['CUDA_rng_bytes_sha256']
 logs=[json.loads(s) for s in (root/'cooldown-updates.jsonl').read_text().splitlines()]
 assert len(logs)==200 and all(x['pairs']==128 and 'truncation' not in x for x in logs)
 assert all(x['lr']==peak*(1-math.sqrt(j/200)) for j,x in enumerate(logs,1))
 assert sum(x['nonfinite_skip'] for x in logs)==cd['nonfinite_skips']==0
 assert sum(x['AMP_skip'] for x in logs)==cd['AMP_skips']
 evalroot=root/'points/b6-final/evaluation';metrics=json.loads((evalroot/'metrics.json').read_text())
 for split,n in [('heldout4k',64),('old991',16)]:
  r=metrics['splits'][split];assert r['exec']['denominator']==n and math.isfinite(r['CE']['CE'])
  assert r['generation']['precision']=='FP16'
 records=[]
 for p in sorted((evalroot/'k8-tune2048').glob('batch-*.json')):records+=json.loads(p.read_text())
 assert [r['id'] for r in records]==[i for i in fixed_split()['tune2048'] if i<64]
 for r in records:
  samples=r['samples']['0.7'];assert len(samples)==8
  assert len(r['question_sha256'])==64
  assert all('represented_code' in s and 'execution' in s and math.isfinite(s['mean_logprob']) for s in samples)
 smoke=json.loads((root/'gpu-smoke.json').read_text());assert smoke['passed'] and smoke['checks']['FP16_exact']
 rows.append(dict(gpu=g,updates=200,nonfinite=0,AMP_skips=cd['AMP_skips'],OOM_retries=sum(len(x['OOM_retries']) for x in logs),
   first_lr=logs[0]['lr'],last_lr=logs[-1]['lr'],initial_model_Adam_scaler_state_exact=True,initial_CUDA_RNG_exact=True,
   historical_main_state_unchanged=True,K8_rows=len(records),K8_samples=8*len(records),FP16=True,
   update_seconds=sum(x['seconds'] for x in logs),smoke_checks=smoke['checks']))
rec=dict(passed=True,workers=rows,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());write(ROOT.parent/'checks/b6-review-extra-gpu-smoke.json',rec);print(json.dumps(rec))
