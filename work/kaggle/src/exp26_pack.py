"""Small receipts/results ZIP, independent of Torch and any completed training cell."""
import hashlib,json,os,shutil,zipfile
from pathlib import Path


def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()


def pack_results(root,out,kind,include_checkpoint=False):
 root,out=Path(root),Path(out)
 assert root.is_dir(),'Drive directory is missing; mount Drive and check the folder'
 selected=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file()or p.suffix not in ('.json','.jsonl','.csv','.md'):continue
  rel=p.relative_to(root)
  if 'raw'in rel.parts or p.name=='result-zip.json' or p.name.endswith('.tmp'):continue
  selected.append((p,str(rel)))
 checkpoint=None
 if include_checkpoint:
  ref=root/'final-checkpoint.json'
  if not ref.exists():ref=root/'latest.json'
  assert ref.exists(),'No verified checkpoint yet'
  checkpoint=json.loads(ref.read_text());p=root/checkpoint['file']
  assert file_sha(p)==checkpoint['sha256'];selected.append((p,checkpoint['file']))
 phase=None;lr=root/'latest.json';cm=root/'cache-manifest.json'
 if lr.exists():phase=json.loads(lr.read_text()).get('phase')
 manifest={'experiment':26,'kind':kind,'train_phase':phase,'include_checkpoint':include_checkpoint,
   'cache_sealed':json.loads(cm.read_text()).get('sealed')if cm.exists()else None,
   'purpose':'metrics/generations/provenance receipts; raw corpus/token arrays/ordinary checkpoints stay on Drive',
   'files':[{'file':name,'bytes':p.stat().st_size,'sha256':file_sha(p)}for p,name in selected]}
 out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.zip.tmp')
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3)as z:
  for p,name in selected:z.write(p,name)
  z.writestr('zip-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False))
 with zipfile.ZipFile(tmp)as z:assert z.testzip()is None
 tmp.replace(out);digest=file_sha(out);backup=root/out.name
 if backup.resolve()!=out.resolve():
  staged=backup.with_suffix('.zip.tmp');shutil.copyfile(out,staged);assert file_sha(staged)==digest;staged.replace(backup)
 receipt={'zip':out.name,'bytes':out.stat().st_size,'sha256':digest,'included_files':len(selected),
   'phase':phase,'cache_sealed':manifest['cache_sealed'],'checkpoint_in_zip':include_checkpoint}
 staged=root/'result-zip.json.tmp';staged.write_text(json.dumps(receipt,indent=2));staged.replace(root/'result-zip.json')
 return receipt
