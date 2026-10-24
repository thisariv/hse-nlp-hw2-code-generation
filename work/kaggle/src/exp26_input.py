"""Only prepared public inputs; fresh references are absent from these archives."""
import hashlib,shutil,zipfile
from pathlib import Path,PurePosixPath
import exp26_protocol as P
from exp26_common import sha_file,read_json,copy_verified

A_NAMES={'tokenizer.json','protected-questions.json'}
B_NAMES=set(P.INPUT_FILES)

def import_starter(archive,out,kind):
 archive,out=Path(archive),Path(out);expected=P.STARTER_A_SHA if kind=='cache-inputs'else P.STARTER_B_SHA
 if sha_file(archive)!=expected:raise ValueError('Need exact prepared26 starter; not the entire25 results archive')
 with zipfile.ZipFile(archive)as z:
  if z.testzip()is not None:raise ValueError('Input ZIP CRC mismatch')
  import json
  m=json.loads(z.read('manifest.json'));names=A_NAMES if kind=='cache-inputs'else B_NAMES
  assert m['experiment']==26 and m['kind']==kind and {e['file']for e in m['files']}==names
  assert set(z.namelist())==names|{'manifest.json'}and len(z.namelist())==len(names)+1
  for row in m['files']:
   n=row['file'];assert n in names and row['sha256']==P.INPUT_FILES[n]
   content=z.read(n);assert len(content)==row['bytes']and hashlib.sha256(content).hexdigest()==row['sha256']
   target=out/n
   if target.exists()and sha_file(target)!=row['sha256']:raise ValueError('Keep conflicting input')
  out.mkdir(parents=True,exist_ok=True)
  for row in m['files']:
   n=row['file'];target=out/n
   if not target.exists():
    tmp=target.with_suffix(target.suffix+'.tmp')
    with z.open(n)as f,tmp.open('wb')as dst:shutil.copyfileobj(f,dst)
    assert sha_file(tmp)==row['sha256'];tmp.replace(target)
 return {'kind':kind,'archive_sha256':expected,'files':len(names)}

def load_protected(path,strict=True):
 obj=read_json(path)
 if strict:assert sha_file(path)==P.PROTECTED_SHA
 groups=obj['groups']
 if strict:assert {k:len(v)for k,v in groups.items()}=={'test':1919,'old1024':1024,'heldout4k':4096,'fresh1024':1024}
 assert all(isinstance(q,str)for values in groups.values()for q in values)
 return set(q for values in groups.values()for q in values),groups

def validate_training_inputs(root):
 for name,digest in P.INPUT_FILES.items():assert sha_file(Path(root)/name)==digest,name
 old=read_json(Path(root)/'old1024.json');held=read_json(Path(root)/'heldout4k.json');test=read_json(Path(root)/'test.json')
 protected,groups=load_protected(Path(root)/'protected-questions.json')
 assert [r['question']for r in old]==groups['old1024']and [r['question']for r in held]==groups['heldout4k']
 assert [r['question']for r in test]==groups['test']
 assert len(set(groups['heldout4k']))==4096 and not set(groups['heldout4k'])&(set(groups['old1024'])|set(groups['test'])|set(groups['fresh1024']))
 return {'heldout4k':held,'old1024':old},test,protected
