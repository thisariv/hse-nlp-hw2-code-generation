"""One global seeded permutation; local bucketing remains a bijection."""
import hashlib
from pathlib import Path
import numpy as np
import exp26_protocol as P
from exp26_common import read_json,write_json,json_hash,sha_file,file_record,copy_verified

def make_order(cache,seed=P.ORDER_SEED,batch=128,chunk=4096):
 assert chunk%batch==0
 rng=np.random.default_rng(seed);order=rng.permutation(len(cache)).astype(np.uint32)
 for lo in range(0,len(order),chunk):
  part=order[lo:lo+chunk];ns=cache.source_lengths[part].astype(np.int32);nt=cache.target_lengths[part].astype(np.int32)
  part=part[np.lexsort((nt,ns,ns+nt+1))];full=len(part)//batch
  if full:part[:full*batch]=part[:full*batch].reshape(full,batch)[rng.permutation(full)].reshape(-1)
  order[lo:lo+len(part)]=part
 seen=np.zeros(len(cache),dtype=np.bool_);seen[order]=True
 assert len(order)==len(cache)and int(order.max())<len(cache)and seen.all()
 return order

def prepare_order(cache,drive_root,local_root,smoke=False):
 drive_root,local_root=Path(drive_root),Path(local_root);drive_root.mkdir(parents=True,exist_ok=True);local_root.mkdir(parents=True,exist_ok=True)
 identity={'cache_manifest_sha256':cache.manifest_sha,'rows':len(cache),'seed':P.ORDER_SEED,'batch':128,'chunk':4096,'smoke':smoke}
 rec=drive_root/'order.json'
 if rec.exists():
  meta=read_json(rec);assert meta['identity']==identity
  assert sha_file(drive_root/'order.npy')==meta['file']['sha256']
 else:
  order=make_order(cache);p=local_root/'order.npy';tmp=p.with_suffix('.npy.tmp')
  with tmp.open('wb')as f:np.save(f,order,allow_pickle=False)
  tmp.replace(p);copy_verified(p,drive_root/'order.npy')
  meta={'identity':identity,'file':file_record(drive_root/'order.npy'),'unique_rows':len(cache),'duplicates_before_exhaustion':0,
        'semantics':'global permutation of all row identities; sort in4096 chunks then shuffle128-row batches; partial tail preserved'}
  write_json(rec,meta)
 copy_verified(drive_root/'order.npy',local_root/'order.npy',meta['file']['sha256'])
 order=np.load(local_root/'order.npy',mmap_mode='r',allow_pickle=False)
 assert order.dtype==np.uint32 and len(order)==len(cache)
 return order,meta

def collate(cache,indices,device):
 import torch
 pairs=[cache[int(i)]for i in indices];ns=[len(s)for s,t in pairs];nt=[len(t)for s,t in pairs]
 src=np.zeros((len(pairs),max(ns)),dtype=np.int64);tgt=np.zeros((len(pairs),max(nt)+1),dtype=np.int64);labels=np.zeros_like(tgt)
 for j,(s,t)in enumerate(pairs):
  src[j,:len(s)]=s;tgt[j,0]=1;tgt[j,1:len(t)+1]=t;labels[j,:len(t)]=t;labels[j,len(t)]=2
 counts={'pairs':len(pairs),'source_tokens':sum(ns),'target_tokens':sum(n+1 for n in nt),'source_slots':src.size,'target_slots':tgt.size}
 return torch.from_numpy(src).to(device),torch.from_numpy(tgt).to(device),torch.from_numpy(labels).to(device),counts

def prefix_identity(order,cursor,rows):
 assert 0<=cursor<=len(order)
 return {'presentations':int(cursor),'unique_rows_seen':int(cursor),'unique_fraction_of_cache':cursor/rows,
         'seen_rows_repeated':0,'row_indices_prefix_sha256':hashlib.sha256(np.asarray(order[:cursor],dtype=np.uint32).tobytes()).hexdigest()}
