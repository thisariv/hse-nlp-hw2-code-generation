#!/usr/bin/env python3
import sys,time,json,zipfile,pathlib
WORK=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/"kaggle/src"))
import torch,exp26_eval as E,exp26_core as C,exp26_train as T
from exp26_common import read_json,sha_file,write_json

def main():
 torch.set_num_threads(2);root=WORK/"results/kaggle-runs/nlp-hw02-kaggle-s1/1";receipt=read_json(root/"gpu0-a26/final-checkpoint.json");cp=root/"gpu0-a26"/receipt["file"]
 checkpoints=[]
 for path in sorted((root/"gpu0-a26/checkpoints").glob("*.pt")):
  payload=torch.load(path,map_location="cpu",weights_only=True);r=read_json(path.with_suffix(".pt.json"));s=payload["training"];h=payload["experiment_history"]
  assert sha_file(path)==r["sha256"] and path.stat().st_size==r["bytes"]
  assert E.state_digest(payload["model"])==r["model_tensor_sha256"]==payload["model_tensor_sha256"]
  assert s["main_updates"]==r["main_updates"]==50000 and s["data_cursor"]==6400000
  assert payload["scheduler"]==T.schedule_state(s,payload["config"])
  assert {"python","numpy","torch_cpu","torch_cuda"}<=set(payload["rng"])
  steps={int(v["step"]) for v in payload["optimizer"]["state"].values()};assert steps=={50000-s["AMP_skips"]}
  assert h["compute_updates"]==58750
  checkpoints.append({"file":path.relative_to(root).as_posix(),"sha256":r["sha256"],"tensor_sha256":r["model_tensor_sha256"],"optimizer_steps":list(steps),"trajectory_AMP_skips":s["AMP_skips"],"all_compute_AMP_skips":h["AMP_skips"],"scaler":payload["scaler"],"execution_runtime":payload.get("execution_runtime"),"history_train_seconds":h["train_seconds"]});del payload
 print("Full checkpoint state integrity passed",flush=True)
 with zipfile.ZipFile(WORK/"results/checkpoints/26b-training-inputs-d0-initial-and-evaluation.zip")as z:
  sets={s:json.loads(z.read(s+".json"))for s in ["heldout4k","old1024"]};tok=C.Tokenizer.from_str(z.read("tokenizer.json").decode())
 start=time.perf_counter();ce64=E.ce_one(cp,sets["heldout4k"][:64],tok,torch.device("cpu"));print("CE64 benchmark",ce64,"ETA seconds",ce64["seconds"]*80,flush=True)
 report={"new_training_updates":0,"new_GPU_runs":0,"runtime":{"torch":str(torch.__version__),"platform":sys.platform},"full_checkpoints":checkpoints,"CPU_CE64":ce64}
 if ce64["seconds"]*80<=360:
  report["full_CPU_CE"]={}
  metrics=read_json(root/"gpu0-a26/points/p050000/evaluation/metrics.json")
  for s,rows in sets.items():
   measured=E.ce_one(cp,rows,tok,torch.device("cpu"));saved=metrics["splits"][s]["CE"];report["full_CPU_CE"][s]=dict(measured,saved_GPU_CE=saved["CE"],absolute_difference=abs(measured["CE"]-saved["CE"]));print(s,report["full_CPU_CE"][s],flush=True)
 else:report["full_CPU_CE"]={"status":"not rerun; CPU64 predicts more than360 seconds"}
 report["status"]=("passed" if all(v["absolute_difference"]<=1e-4 for v in report["full_CPU_CE"].values()) else "CE differs") if "heldout4k" in report["full_CPU_CE"] else "partial; full CE not rerun";report["wall_seconds"]=time.perf_counter()-start;write_json(WORK/"checks/kaggle-s1-v1-model-cpu-checks.json",report)
if __name__=="__main__":main()
