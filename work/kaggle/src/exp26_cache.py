"""CPU-only shard cache: exact exclusions, finite references, int16 BPE.
Only one shard of text is in RAM. Completed shard receipts are immutable.
"""
import hashlib,json,os,socket,time,urllib.request
from array import array
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
import exp26_protocol as P
from exp26_common import sha_file,json_hash,read_json,write_json,file_record,verify_record
from exp26_input import load_protected
from exp26_fast_exec import run_chunks

FILES=['source.npy','source-offsets.npy','target.npy','target-offsets.npy','provenance-rows.npy','question-sha256.npy']

def is_validation(q):return int.from_bytes(hashlib.sha256(f'{P.SEED}:{q}'.encode()).digest()[:8],'big')%10==0

def classify(row,protected):
 if not isinstance(row.get('question'),str)or not isinstance(row.get('code'),str):return 'invalid_text'
 if len(row['question'].encode('utf-8'))>512 or len(row['code'].encode('utf-8'))>=512:return 'length'
 if row['question']in protected:return 'protected_question'
 if is_validation(row['question']):return 'hash_heldout'
 return None

def download_raw(spec,root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);p=root/Path(spec['path']).name
 if p.exists():
  assert p.stat().st_size==spec['bytes']and sha_file(p)==spec['sha256'],p.name
  return p
 url=f'https://huggingface.co/datasets/SlavaYus/tinysms_qa_12m/resolve/{P.REVISION}/{spec["path"]}'
 for attempt in range(3):
  tmp=p.with_suffix('.download.tmp');h=hashlib.sha256()
  try:
   with urllib.request.urlopen(url,timeout=30)as response,tmp.open('wb')as out:
    for block in iter(lambda:response.read(4*1024**2),b''):out.write(block);h.update(block)
   assert tmp.stat().st_size==spec['bytes']and h.hexdigest()==spec['sha256']
   tmp.replace(p);assert sha_file(p)==spec['sha256'];return p
  except (OSError,AssertionError):
   if attempt==2:raise
   time.sleep(1+attempt)

def collect_candidates(path,protected,limit=None):
 rows=[];row_ids=[];counts=Counter();offset=0
 for batch in pq.ParquetFile(path).iter_batches(batch_size=8192,columns=['question','code']):
  for j,row in enumerate(batch.to_pylist()):
   counts['raw_rows']+=1;reason=classify(row,protected)
   if reason:counts[reason]+=1;continue
   rows.append(row);row_ids.append(offset+j)
   if limit is not None and len(rows)==limit:return rows,row_ids,dict(counts)
  offset+=len(batch)
 return rows,row_ids,dict(counts)

def save_npy(path,value):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 with tmp.open('wb')as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
 tmp.replace(p)

def token_block(rows,row_ids,tokenizer,token_reuse=None,shard_id=None):
 source=array('h');target=array('h');so=array('q',[0]);to=array('q',[0]);qh=bytearray();reused=0
 for lo in range(0,len(rows),2048):
  part=rows[lo:lo+2048];locs=row_ids[lo:lo+2048]
  missing=[i for i,loc in enumerate(locs)if token_reuse is None or (shard_id,loc)not in token_reuse['indices']]
  qs=tokenizer.encode_batch([part[i]['question']for i in missing],add_special_tokens=False)if missing else []
  cs=tokenizer.encode_batch([part[i]['code']for i in missing],add_special_tokens=False)if missing else []
  encoded={i:([1]+[j+3 for j in q.ids]+[2],[j+3 for j in c.ids])for i,q,c in zip(missing,qs,cs)}
  for i,r in enumerate(part):
   if i in encoded:q,c=encoded[i]
   else:
    k=token_reuse['indices'][(shard_id,locs[i])];a=token_reuse
    q=a['source'][a['so'][k]:a['so'][k+1]];c=a['target'][a['to'][k]:a['to'][k+1]];reused+=1
   source.extend(q);so.append(len(source));target.extend(c);to.append(len(target));qh.extend(hashlib.sha256(r['question'].encode()).digest())
 return [np.frombuffer(source,dtype=np.int16),np.frombuffer(so,dtype=np.int64),np.frombuffer(target,dtype=np.int16),
         np.frombuffer(to,dtype=np.int64),np.asarray(row_ids,dtype=np.uint32),np.frombuffer(qh,dtype=np.uint8).reshape(-1,32)],reused

def build_tokens(rows,row_ids,tokenizer,out,token_reuse=None,shard_id=None,identity=None,stop_after_blocks=None,block_rows=16384):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);work=out/'token-work';work.mkdir(exist_ok=True)
 identity=identity or json_hash({'rows':row_ids,'questions':[r['question']for r in rows]})
 rp=work/'progress.json';dtypes=[np.int16,np.int64,np.int16,np.int64,np.uint32,np.uint8]
 if rp.exists():
  progress=read_json(rp);assert progress['identity']==identity
  for record in progress['blocks']:
   p=work/record['file']
   with p.open('rb')as f:f.seek(record['offset']);assert hashlib.sha256(f.read(record['bytes'])).hexdigest()==record['sha256']
  for name,size in progress['file_bytes'].items():
   p=work/name;assert p.stat().st_size>=size
   with p.open('r+b')as f:f.truncate(size)  # Only surplus from an uncommitted block.
 else:
  progress={'identity':identity,'committed_rows':0,'blocks':[],'file_bytes':{},'pilot_tokens_reused':0}
  for name,dtype in zip(FILES,dtypes):
   p=work/(name+'.bin')
   with p.open('wb')as f:
    if 'offsets'in name:f.write(np.asarray([0],dtype=np.int64).tobytes())
   progress['file_bytes'][p.name]=p.stat().st_size
  write_json(rp,progress)
 block_number=0
 for lo in range(progress['committed_rows'],len(rows),block_rows):
  values,reused=token_block(rows[lo:lo+block_rows],row_ids[lo:lo+block_rows],tokenizer,token_reuse,shard_id)
  for i,(name,value)in enumerate(zip(FILES,values)):
   p=work/(name+'.bin')
   if i in (1,3):
    token_name=FILES[i-1]+'.bin';start=progress['file_bytes'][token_name]//2-len(values[i-1])
    value=value[1:]+start
   raw=value.tobytes();offset=p.stat().st_size
   with p.open('ab')as f:f.write(raw);f.flush();os.fsync(f.fileno())
   progress['blocks'].append({'file':p.name,'offset':offset,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
   progress['file_bytes'][p.name]=offset+len(raw)
  progress['committed_rows']=min(len(rows),lo+block_rows);progress['pilot_tokens_reused']+=reused;write_json(rp,progress)
  block_number+=1
  if stop_after_blocks and block_number>=stop_after_blocks:raise KeyboardInterrupt('Token-cache interruption fixture')
 n=len(rows)
 for name,dtype in zip(FILES,dtypes):
  p=work/(name+'.bin');elements=p.stat().st_size//np.dtype(dtype).itemsize
  shape=(n,32)if name=='question-sha256.npy'else(elements,)
  src=np.memmap(p,dtype=dtype,mode='r',shape=shape)if elements else np.empty(shape,dtype=dtype)
  tmp=out/(name+'.tmp');dst=np.lib.format.open_memmap(tmp,mode='w+',dtype=dtype,shape=shape)
  for lo in range(0,len(src),1024**2):dst[lo:lo+1024**2]=src[lo:lo+1024**2]
  dst.flush();del dst,src;tmp.replace(out/name)
 return {'rows':n,'source_tokens':progress['file_bytes']['source.npy.bin']//2,
         'target_content_tokens':progress['file_bytes']['target.npy.bin']//2,
         'target_tokens_with_EOS':progress['file_bytes']['target.npy.bin']//2+n,'pilot_tokenizations_reused':progress['pilot_tokens_reused']}

def validate_shard(folder,receipt,protected=None):
 for entry in receipt['files']:verify_record(folder,entry)
 src,so,tgt,to,ids,qh=[np.load(Path(folder)/n,mmap_mode='r',allow_pickle=False)for n in FILES]
 n=receipt['tokens']['rows'];assert len(ids)==len(qh)==n and so.shape==to.shape==(n+1,)
 assert src.dtype==tgt.dtype==np.int16 and so.dtype==to.dtype==np.int64 and ids.dtype==np.uint32 and qh.dtype==np.uint8
 assert qh.shape==(n,32)and so[0]==to[0]==0 and so[-1]==len(src)and to[-1]==len(tgt)
 assert (np.diff(so)>=2).all()and (np.diff(to)>=0).all()and (np.diff(ids.astype(np.int64))>0).all()
 assert not n or ids.max()<receipt['raw_num_rows']
 if n:
  assert src.min()>=1 and src.max()<8192 and (src[so[:-1]]==1).all()and (src[so[1:]-1]==2).all()
  assert not len(tgt)or (tgt.min()>=3 and tgt.max()<8192)
 if protected is not None:
  protected_hashes=np.asarray([hashlib.sha256(q.encode()).digest()for q in protected],dtype='S32')
  for lo in range(0,n,65536):
   hashes=np.asarray(qh[lo:lo+65536]).copy().view('S32').reshape(-1)
   assert not np.isin(hashes,protected_hashes).any(),'Protected question in token cache'
 return n

STATUS_NAMES=['numeric','non_number','non_finite','missing_solve','execution_error','syntax_error','blocked_size','blocked_name','blocked_attribute','blocked_construct','blocked_import','timeout','timeout_or_worker_exit','worker_error','budget_not_evaluated']

def execute_journal(codes,out,identity,workers,progress=None):
 import struct
 out=Path(out);out.mkdir(parents=True,exist_ok=True);p=out/'reference-journal.bin';header=out/'reference-journal.json'
 fingerprint=json_hash({'identity':identity,'codes':codes,'chunk_rows':2000})
 if header.exists():assert read_json(header)['fingerprint']==fingerprint
 else:write_json(header,{'fingerprint':fingerprint,'status_names':STATUS_NAMES})
 saved={};good=0
 if p.exists():
  with p.open('rb')as f:
   while True:
    h=f.read(40)
    if not h:break
    if len(h)!=40:break
    i,n,digest=struct.unpack('<II32s',h);assert n<=2000 and i*2000<len(codes)
    payload=f.read(n)
    if len(payload)!=n:break
    assert hashlib.sha256(payload).digest()==digest
    assert n==min(2000,len(codes)-i*2000)and all(v<len(STATUS_NAMES)for v in payload)
    saved[i]=[{'status':STATUS_NAMES[v]}for v in payload];good=f.tell()
  with p.open('r+b')as f:f.truncate(good)
 def commit(i,rows):
  payload=bytes(STATUS_NAMES.index(r['status'])for r in rows)
  with p.open('ab')as f:f.write(struct.pack('<II32s',i,len(payload),hashlib.sha256(payload).digest())+payload);f.flush();os.fsync(f.fileno())
 result=run_chunks(codes,workers=workers,chunk_rows=2000,saved_chunks=saved,on_chunk=commit,progress=progress)
 result['summary']['programs_reused_from_journal']=sum(len(v)for v in saved.values())
 return result

def pilot(root,paths,specs,protected,tokenizer,code_sha,workers,limit=50000):
 root=Path(root);receipt_path=root/'pilot.json'
 if receipt_path.exists():
  info=read_json(receipt_path);assert info['code_sha256']==code_sha
  for e in info['files']:verify_record(root,e)
  return info,np.load(root/'pilot.npz',allow_pickle=False)
 started=time.monotonic();rows=[];locations=[];raw_scanned=0
 for spec,path in zip(specs,paths):
  part,ids,counts=collect_candidates(path,protected,limit=limit-len(rows));rows.extend(part)
  locations.extend((spec['shard'],i)for i in ids);raw_scanned+=counts['raw_rows']
  if len(rows)>=limit:break
 if len(rows)<limit:raise ValueError('Insufficient pilot eligible rows')
 selected_seconds=time.monotonic()-started
 last_report=[0]
 def progress(done,total,seconds):
  if done-last_report[0]>=10000 or done==total:
   rate=done/max(seconds,1e-9);print('PILOT EXEC',done,'/',total,'rows;',round(rate,1),'rows/s; remaining',round((total-done)/max(rate,1e-9),1),'s',flush=True);last_report[0]=done
 execution=execute_journal([r['code']for r in rows],root/'pilot-reference',json_hash({'code':code_sha,'purpose':'pilot'}),workers,progress=progress)
 encode_start=time.monotonic();keep=[i for i,r in enumerate(execution['rows'])if r['status']=='numeric']
 tokens=build_tokens([rows[i]for i in keep],[locations[i][1]for i in keep],tokenizer,root/'pilot-tokens')
 save_npy(root/'pilot-shards.npy',np.asarray([locations[i][0]for i in keep],dtype=np.uint8))
 tokenization_seconds=time.monotonic()-encode_start
 names=sorted({r['status']for r in execution['rows']});lookup={n:i for i,n in enumerate(names)}
 p=root/'pilot.npz';tmp=p.with_suffix('.npz.tmp')
 with tmp.open('wb')as f:np.savez(f,shard=np.asarray([a for a,b in locations],dtype=np.uint8),row=np.asarray([b for a,b in locations],dtype=np.uint32),status=np.asarray([lookup[r['status']]for r in execution['rows']],dtype=np.uint8))
 tmp.replace(p)
 elapsed=time.monotonic()-started
 info={'code_sha256':code_sha,'eligible_rows':len(rows),'raw_scanned':raw_scanned,'status_names':names,
       'selection_seconds':selected_seconds,'execution':execution['summary'],'tokenization_seconds':tokenization_seconds,
       'total_seconds':elapsed,'eligible_rows_per_second':len(rows)/elapsed,'tokens':tokens,
       'files':[file_record(p),file_record(root/'pilot-shards.npy')]+[file_record(root/'pilot-tokens'/n,'pilot-tokens/'+n)for n in FILES]}
 write_json(receipt_path,info)
 return info,np.load(p,allow_pickle=False)

def prepare_cache(root,inputs,code_sha,workers=2,smoke=False,specs=None,download_fn=download_raw,
                  pilot_rows=None,min_rows=None,eta_limit_minutes=60,stop_after_shards=None):
 root,inputs=Path(root),Path(inputs);root.mkdir(parents=True,exist_ok=True)
 specs=specs or P.SHARDS
 if not smoke:assert specs==P.SHARDS and pilot_rows in (None,50000)and min_rows in (None,7000000)
 pilot_rows=pilot_rows or (4 if smoke else 50000);min_rows=min_rows or (4 if smoke else 7000000)
 protected,_=load_protected(inputs/'protected-questions.json',strict=not smoke)
 tokenizer=Tokenizer.from_file(str(inputs/'tokenizer.json'))
 if not smoke:assert sha_file(inputs/'tokenizer.json')==P.TOKENIZER_SHA and tokenizer.get_vocab_size()==8189
 identity={'protocol':P.CACHE_PROTOCOL,'code_sha256':code_sha,'tokenizer_sha256':sha_file(inputs/'tokenizer.json'),
           'protected_sha256':sha_file(inputs/'protected-questions.json'),'specs':specs,'smoke':smoke}
 identity_sha=json_hash(identity);master=root/'cache-manifest.json';old=read_json(master)if master.exists()else None
 if old:
  assert old['identity_sha256']==identity_sha,'Cache input/protocol changed; use another directory'
  if old.get('sealed'):
   for e in old['shards']:validate_shard(root/e['folder'],e,protected)
   print('CACHE ALREADY SEALED; no download/selection/execution/tokenization repeated',old['rows']);return old
 # All60 files are downloaded even when the time rule chooses a processing subset.
 download_start=time.monotonic();paths=[];raw_counts=[]
 for j,spec in enumerate(specs):
  p=download_fn(spec,root/'raw');assert p.stat().st_size==spec['bytes']and sha_file(p)==spec['sha256']
  paths.append(p);raw_counts.append(pq.ParquetFile(p).metadata.num_rows)
  if (j+1)%5==0 or j+1==len(specs):print('DOWNLOAD',j+1,'/',len(specs),flush=True)
 download_seconds=(old or {}).get('download_seconds',time.monotonic()-download_start)
 info,pdata=pilot(root,paths,specs,protected,tokenizer,code_sha,workers,pilot_rows)
 estimated=download_seconds+info['total_seconds']*sum(raw_counts)/info['raw_scanned']
 mode=(old or {}).get('processing_policy')or ('minimum7M_if_ETA_over60min'if estimated>eta_limit_minutes*60 else 'all60')
 print('PILOT',info['eligible_rows'],'eligible;',round(info['eligible_rows_per_second'],1),'rows/s; full ETA',round(estimated/60,2),'min; POLICY',mode,flush=True)
 manifest={'format':'nlp-hw02-token-cache-v1','experiment':26,'identity':identity,'identity_sha256':identity_sha,
  'downloaded_shards':len(specs),'downloaded_raw_rows':sum(raw_counts),'download_seconds':download_seconds,
  'pilot':info,'estimated_full_minutes':estimated/60,'processing_policy':mode,'minimum_rows':min_rows,
  'rows':0,'shards':[],'sealed':False,'fresh_evaluations':0,'train_protected_question_overlaps':0}
 shard_ids=pdata['shard'];row_ids=pdata['row'];status=pdata['status'];names=info['status_names']
 elapsed_processing=info['total_seconds']
 pa={name:np.load(root/'pilot-tokens'/name,mmap_mode='r',allow_pickle=False)for name in FILES}
 ps=np.load(root/'pilot-shards.npy',allow_pickle=False)
 reuse={'indices':{(int(s),int(r)):i for i,(s,r)in enumerate(zip(ps,pa['provenance-rows.npy']))},
        'source':pa['source.npy'],'target':pa['target.npy'],'so':pa['source-offsets.npy'],'to':pa['target-offsets.npy']}
 for spec,path,raw_n in zip(specs,paths,raw_counts):
  folder=f'shards/part-{spec["shard"]:05d}';out=root/folder;rp=out/'manifest.json'
  if rp.exists():
   receipt=read_json(rp);assert receipt['identity_sha256']==identity_sha and receipt['raw_sha256']==spec['sha256']
   validate_shard(out,receipt,protected);print('REUSE SHARD',spec['shard'],receipt['tokens']['rows'],flush=True)
  else:
   started=time.monotonic();rows,ids,counts=collect_candidates(path,protected)
   assert counts['raw_rows']==raw_n
   cached={int(r):names[int(s)]for r,s in zip(row_ids[shard_ids==spec['shard']],status[shard_ids==spec['shard']])}
   missing=[i for i,loc in enumerate(ids)if loc not in cached]
   execution=execute_journal([rows[i]['code']for i in missing],out,identity_sha,workers)
   statuses=[None]*len(rows)
   for i,loc in enumerate(ids):
    if loc in cached:statuses[i]=cached[loc]
   for i,e in zip(missing,execution['rows']):statuses[i]=e['status']
   keep=[i for i,s in enumerate(statuses)if s=='numeric']
   selected=[rows[i]for i in keep];selected_ids=[ids[i]for i in keep]
   assert all(classify(r,protected)is None for r in selected)
   encoding_start=time.monotonic();tok=build_tokens(selected,selected_ids,tokenizer,out,reuse,spec['shard'],identity=json_hash({'cache':identity_sha,'raw_sha':spec['sha256'],'rows':selected_ids}))
   receipt={'shard':spec['shard'],'folder':folder,'identity_sha256':identity_sha,'raw_sha256':spec['sha256'],
    'raw_num_rows':raw_n,'selection_counts':counts,'pre_execution_rows':len(rows),'reference_statuses':dict(Counter(statuses)),
    'reference_non_numeric_dropped':len(rows)-len(keep),'reference_drop_fraction':1-len(keep)/len(rows)if rows else 0.,
    'pilot_reference_results_reused':len(cached),'execution':execution['summary'],'tokens':tok,
    'encoding_seconds':time.monotonic()-encoding_start,'wall_seconds':time.monotonic()-started,
    'protected_question_overlaps':0,'hash_heldout_overlaps':0,'files':[file_record(out/n)for n in FILES]}
   validate_shard(out,receipt,protected);write_json(rp,receipt)
   import shutil
   if (out/'token-work').exists():shutil.rmtree(out/'token-work')  # Only committed temporary binaries.
  manifest['shards'].append(receipt);manifest['rows']+=receipt['tokens']['rows'];elapsed_processing+=receipt['wall_seconds']
  manifest['processed_shards']=len(manifest['shards']);manifest['processed_raw_rows']=sum(r['raw_num_rows']for r in manifest['shards'])
  manifest['reference_drop_fraction']=sum(r['reference_non_numeric_dropped']for r in manifest['shards'])/sum(r['pre_execution_rows']for r in manifest['shards'])
  revised=download_seconds+elapsed_processing*sum(raw_counts)/manifest['processed_raw_rows']
  if revised>eta_limit_minutes*60:manifest['processing_policy']='minimum7M_if_ETA_over60min'
  manifest['revised_full_ETA_minutes']=revised/60;write_json(master,manifest)
  print('CACHE SHARD',spec['shard'],'rows',receipt['tokens']['rows'],'TOTAL',manifest['rows'],
    'drop',round(100*manifest['reference_drop_fraction'],2),'%; revised ETA',round(revised/60,2),'min',flush=True)
  if stop_after_shards and len(manifest['shards'])>=stop_after_shards:return manifest
  if manifest['processing_policy']=='minimum7M_if_ETA_over60min'and manifest['rows']>=min_rows:break
 if manifest['rows']<min_rows:raise ValueError('Not enough filtered rows; GPU training must not start')
 manifest['sealed']=True;manifest['status']='ready_all60'if len(manifest['shards'])==len(specs)else 'ready_subset_minimum7M'
 manifest['source_tokens']=sum(r['tokens']['source_tokens']for r in manifest['shards'])
 manifest['target_tokens_with_EOS']=sum(r['tokens']['target_tokens_with_EOS']for r in manifest['shards'])
 manifest['unique_row_identity_policy']='Each original(shard,row)once; duplicate question texts not deduplicated'
 write_json(master,manifest);print('CACHE READY',manifest['status'],manifest['rows'],'rows;',len(manifest['shards']),'processed /',len(specs),'downloaded')
 return manifest

class TokenCache:
 def __init__(self,root,protected=None,require_minimum=True):
  self.root=Path(root);self.manifest=read_json(self.root/'cache-manifest.json')
  assert self.manifest['sealed']and self.manifest['fresh_evaluations']==0
  if require_minimum:assert self.manifest['rows']>=7000000 and self.manifest['downloaded_shards']==60 and not self.manifest['identity']['smoke']
  self.parts=[];starts=[0];sl=[];tl=[]
  for receipt in self.manifest['shards']:
   folder=self.root/receipt['folder'];n=validate_shard(folder,receipt,protected)
   arrays={name:np.load(folder/name,mmap_mode='r',allow_pickle=False)for name in FILES}
   self.parts.append(arrays);starts.append(starts[-1]+n)
   sl.append(np.diff(arrays['source-offsets.npy']).astype(np.uint16));tl.append(np.diff(arrays['target-offsets.npy']).astype(np.uint16))
  self.starts=np.asarray(starts,dtype=np.int64);self.source_lengths=np.concatenate(sl);self.target_lengths=np.concatenate(tl)
  assert len(self)==self.manifest['rows'];self.manifest_sha=sha_file(self.root/'cache-manifest.json')
 def __len__(self):return int(self.starts[-1])
 def __getitem__(self,index):
  index=int(index)
  if not 0<=index<len(self):raise IndexError(index)
  shard=int(np.searchsorted(self.starts,index,side='right')-1);j=index-int(self.starts[shard]);a=self.parts[shard]
  so,to=a['source-offsets.npy'],a['target-offsets.npy']
  return a['source.npy'][so[j]:so[j+1]],a['target.npy'][to[j]:to[j+1]]
 def provenance(self,index):
  shard=int(np.searchsorted(self.starts,int(index),side='right')-1);j=int(index)-int(self.starts[shard])
  return self.manifest['shards'][shard]['shard'],int(self.parts[shard]['provenance-rows.npy'][j])


def copy_cache(drive_root,local_root,code_sha,protected):
 from exp26_common import copy_verified
 drive_root,local_root=Path(drive_root),Path(local_root);m=read_json(drive_root/'cache-manifest.json')
 assert m['sealed']and m['rows']>=7000000 and m['identity']['code_sha256']==code_sha and m['identity']['tokenizer_sha256']==P.TOKENIZER_SHA and m['identity']['protected_sha256']==P.PROTECTED_SHA
 for j,r in enumerate(m['shards']):
  for e in r['files']:copy_verified(drive_root/r['folder']/e['file'],local_root/r['folder']/e['file'],e['sha256'])
  copy_verified(drive_root/r['folder']/'manifest.json',local_root/r['folder']/'manifest.json')
  print('CACHE COPY',j+1,'/',len(m['shards']),flush=True)
 copy_verified(drive_root/'cache-manifest.json',local_root/'cache-manifest.json',overwrite=True)
 return TokenCache(local_root,protected)
