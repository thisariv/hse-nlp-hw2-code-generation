#!/usr/bin/env python3
import pathlib,sys,tempfile,json
WORK=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/"kaggle/src"))
from exp26_common import read_json,write_json
import kaggle_runtime as R,exp26_workflow as W
root=WORK/"results/kaggle-runs/nlp-hw02-kaggle-s1/1";inputs=root/"inputs";sets={s:read_json(inputs/(s+".json"))for s in ["heldout4k","old1024"]}
with tempfile.TemporaryDirectory(prefix="s1-reference-reuse-")as tmp:
 target=pathlib.Path(tmp)/"gpu1/references";records=R.reuse_verified_references(root/"gpu0-a26/references",target,sets,inputs)
 original=W.numeric.evaluate_programs
 def forbidden(*a,**kw):raise AssertionError("Verified references must not be re-executed")
 W.numeric.evaluate_programs=forbidden
 refs=W.reference_execution(sets,inputs,target.parent);assert len(refs)==2
 W.numeric.evaluate_programs=original
 changed=dict(sets);changed["heldout4k"]=[dict(r)for r in changed["heldout4k"]];changed["heldout4k"][0]["question"]+=" tampered"
 try:R.reuse_verified_references(root/"gpu0-a26/references",pathlib.Path(tmp)/"bad/references",changed,inputs)
 except AssertionError:tamper=True
 else:raise AssertionError("Tampered questions accepted")
 report={"status":"passed","checks":["real S1 reference SHA/identity/finite-index/value/gate receipts reused","original reference_execution reads reused receipts without execution","changed questions rejected before copy"],"references":records,"new_training_updates":0,"new_GPU_runs":0}
 write_json(WORK/"checks/kaggle-s2-reference-reuse-cpu-checks.json",report);print(json.dumps(report,indent=2))
