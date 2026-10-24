"""Atomic local/Drive files; immutable SHA receipts."""
import hashlib,json,os,shutil,uuid
from pathlib import Path

def sha_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()

def json_hash(obj):return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text())
def write_json(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex[:8])
 with tmp.open('w')as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)

def copy_verified(source,target,expected=None,overwrite=False):
 source,target=Path(source),Path(target);expected=expected or sha_file(source)
 if sha_file(source)!=expected:raise ValueError(f'Source SHA mismatch: {source.name}')
 if target.exists():
  if sha_file(target)==expected:return expected
  if not overwrite:raise ValueError(f'Preserve conflicting file: {target.name}')
 target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_name(target.name+'.tmp-'+uuid.uuid4().hex[:8])
 shutil.copyfile(source,tmp)
 if sha_file(tmp)!=expected:raise ValueError('Copy SHA mismatch')
 tmp.replace(target)
 if sha_file(target)!=expected:raise ValueError('Destination SHA mismatch')
 return expected

def file_record(path,relative=None):
 p=Path(path);return {'file':relative or p.name,'bytes':p.stat().st_size,'sha256':sha_file(p)}

def verify_record(root,row):
 from pathlib import PurePosixPath
 name=row['file'];p=PurePosixPath(name)
 if p.is_absolute()or '..'in p.parts or '\\'in name:raise ValueError('Unsafe manifest path')
 target=Path(root)/name
 if not target.is_file()or target.stat().st_size!=row['bytes']or sha_file(target)!=row['sha256']:raise ValueError(f'File SHA/size mismatch: {name}')
 return target
