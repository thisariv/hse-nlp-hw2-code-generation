"""CPU compact cache, exact row identities from sealed 26a + 26c. No Torch."""
import argparse, hashlib, json, math, multiprocessing as mp, os, shutil, subprocess, sys, time
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
import s3_representation as R
from kaggle_io import sha, write, materialize_dataset, resolve_dataset

BASE_SHA = '2c691fef64fea8b4186b8acae54561d5620080286196a4390a23108776e80add'
EXT_SHA = '7417b6ffc0a0cfeeafb501610fdee8b43bd1bcea2935ded64cc964cba79800a3'
ROWS = 9725737
ORDER_SEED = 2026101201
FILES = ['source.npy', 'source-offsets.npy', 'target.npy', 'target-offsets.npy',
         'provenance-rows.npy', 'question-sha256.npy']

def record(path):
    path = Path(path)
    return {'file': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path)}

def decode(table, ids):
    return b''.join(table[int(i)] for i in ids if i >= 3).decode('utf8')

def convert_shard(job):
    original, receipt, destination, tokpath, sample_n = job
    original, destination = Path(original), Path(destination)
    src = original / receipt['folder']; out = destination / receipt['folder']
    out.mkdir(parents=True, exist_ok=True)
    for rec in receipt['files']:
        p = src / rec['file']
        assert p.stat().st_size == rec['bytes'] and sha(p) == rec['sha256']
    tok, table = R.tokenizer(tokpath, 30)
    target = np.load(src/'target.npy', mmap_mode='r', allow_pickle=False)
    offsets = np.load(src/'target-offsets.npy', mmap_mode='r', allow_pickle=False)
    n = len(offsets)-1; new_offsets = np.empty(n+1, dtype=np.int64); new_offsets[0] = 0
    chosen = set(np.random.default_rng(ORDER_SEED+receipt['shard']).choice(n, min(sample_n,n), replace=False).tolist())
    samples = []; started = time.monotonic(); total = 0; modes = {}
    raw = out/'target.bin.tmp'
    with raw.open('wb') as stream:
        for lo in range(0, n, 2048):
            codes = [decode(table, target[offsets[i]:offsets[i+1]]) for i in range(lo, min(n,lo+2048))]
            fragments = [R.compact(code) for code in codes]
            encoded = tok.encode_batch(fragments, add_special_tokens=False)
            block = array('h')
            for j, (code, text, enc) in enumerate(zip(codes, fragments, encoded)):
                i = lo+j; ids = [v+3 for v in enc.ids]
                block.extend(ids); total += len(ids); new_offsets[i+1] = total
                mode = text.split('|>',1)[0]+'|>'; modes[mode] = modes.get(mode,0)+1
                if i in chosen:
                    roundtrip = decode(table, ids)
                    assert roundtrip == text, 'BPE round-trip mismatch'
                    restored = R.restore_compact(roundtrip)
                    samples.append({'shard': receipt['shard'], 'local_row': i, 'original': code, 'restored': restored})
            stream.write(block.tobytes())
    mm = np.lib.format.open_memmap(out/'target.npy', mode='w+', dtype=np.int16, shape=(total,))
    binary = np.memmap(raw, mode='r', dtype=np.int16)
    for lo in range(0,total,4*1024**2): mm[lo:lo+4*1024**2] = binary[lo:lo+4*1024**2]
    mm.flush(); del mm, binary; raw.unlink()
    np.save(out/'target-offsets.npy',new_offsets,allow_pickle=False)
    for name in FILES:
        if name not in ('target.npy','target-offsets.npy'): shutil.copyfile(src/name,out/name)
    result = {'shard':receipt['shard'], 'folder':receipt['folder'], 'rows':n, 'target_tokens':total,
              'original_shard_manifest_sha256':sha(src/'manifest.json'),
              'original_files':receipt['files'], 'files':[record(out/name) for name in FILES],
              'modes':modes, 'seconds':time.monotonic()-started}
    write(out/'manifest.json',result)
    write(out/'roundtrip-sample.json',samples)
    print('COMPACT SHARD',receipt['shard'],'rows',n,'seconds',round(result['seconds'],1),flush=True)
    return result

class CompactCache:
    def __init__(self, root, verify=True):
        self.root=Path(root); self.manifest=json.loads((self.root/'compact-manifest.json').read_text())
        assert self.manifest['sealed'] and self.manifest['representation']==30
        self.manifest_sha=sha(self.root/'compact-manifest.json'); self.parts=[]; starts=[0]; sl=[]; tl=[]
        for rec in self.manifest['shards']:
            folder=self.root/rec['folder']
            if verify:
                for f in rec['files']:
                    p=folder/f['file']; assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
            a={n:np.load(folder/n,mmap_mode='r',allow_pickle=False) for n in FILES}
            n=len(a['source-offsets.npy'])-1
            assert n==rec['rows']==len(a['target-offsets.npy'])-1
            for prefix in ('source','target'):
                assert a[prefix+'.npy'].dtype==np.int16
                o=a[prefix+'-offsets.npy']; assert o.dtype==np.int64 and o[0]==0 and o[-1]==len(a[prefix+'.npy'])
                lengths=np.diff(o);assert np.all(lengths>0)
                assert int(lengths.max())+(prefix=='target')<=2048,'Position/length overflow'
                if verify:
                    ids=a[prefix+'.npy'];assert int(ids.min())>=1 and int(ids.max())<8195,'Token ID overflow'
            self.parts.append(a); starts.append(starts[-1]+n)
            sl.append(np.diff(a['source-offsets.npy']).astype(np.uint16)); tl.append(np.diff(a['target-offsets.npy']).astype(np.uint16))
        self.starts=np.asarray(starts); self.source_lengths=np.concatenate(sl); self.target_lengths=np.concatenate(tl)
        assert len(self)==self.manifest['rows']
    def __len__(self): return int(self.starts[-1])
    def __getitem__(self,index):
        index=int(index)
        if not 0<=index<len(self): raise IndexError(index)
        k=int(np.searchsorted(self.starts,index,side='right')-1); j=index-int(self.starts[k]); a=self.parts[k]
        so,to=a['source-offsets.npy'],a['target-offsets.npy']
        return a['source.npy'][so[j]:so[j+1]],a['target.npy'][to[j]:to[j+1]]

def make_order(cache, seed=ORDER_SEED, batch=128, chunk=4096):
    rng=np.random.default_rng(seed); order=rng.permutation(len(cache)).astype(np.uint32)
    for lo in range(0,len(order),chunk):
        part=order[lo:lo+chunk]; ns=cache.source_lengths[part].astype(np.int32); nt=cache.target_lengths[part].astype(np.int32)
        part=part[np.lexsort((nt,ns,ns+nt+1))]; full=len(part)//batch
        if full: part[:full*batch]=part[:full*batch].reshape(full,batch)[rng.permutation(full)].reshape(-1)
        order[lo:lo+len(part)]=part
    assert np.array_equal(np.sort(order),np.arange(len(cache),dtype=np.uint32))
    return order

def build(base, extension, tokpath, output, workers=4, sample_rows=50000, smoke=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); started=time.monotonic()
    roots=[Path(base),Path(extension)]; originals=[json.loads((p/'cache-manifest.json').read_text()) for p in roots]
    hashes=[sha(p/'cache-manifest.json') for p in roots]
    if not smoke:
        assert hashes==[BASE_SHA,EXT_SHA]
        assert [r['shard'] for m in originals for r in m['shards']]==list(range(60))
        assert sum(m['rows'] for m in originals)==ROWS
    assert all(m['sealed'] and m['fresh_evaluations']==0 for m in originals)
    tok,_=R.tokenizer(tokpath,30); tok.save(str(output/'tokenizer.json'))
    jobs=[(str(root),r,str(output),str(tokpath),math.ceil(sample_rows/sum(len(m['shards']) for m in originals)))
          for root,m in zip(roots,originals) for r in m['shards']]
    converted=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for f in as_completed([pool.submit(convert_shard,j) for j in jobs]): converted.append(f.result())
    converted.sort(key=lambda r:r['shard'])
    manifest={'format':'nlp-hw02-compact-cache-v1','representation':30,'mode_tokens':R.MODES,
              'representation_code_sha256':sha(Path(R.__file__)), 'original_manifest_sha256':hashes,
              'original_tokenizer_sha256':sha(tokpath),'tokenizer':record(output/'tokenizer.json'),
              'rows':sum(r['rows'] for r in converted),'shards':converted,'sealed':True,
              'fresh_evaluations':0,'selection':'exact sealed finite rows26a0-41 +26c42-59; no new filtering or deduplication',
              'smoke':smoke}
    write(output/'compact-manifest.json',manifest)
    cache=CompactCache(output); order=make_order(cache); np.save(output/'order.npy',order,allow_pickle=False)
    order_info={'seed':ORDER_SEED,'rows':len(cache),'batch':128,'chunk':4096,'bijection':True,
                'file':record(output/'order.npy'),'cache_manifest_sha256':cache.manifest_sha}
    write(output/'order.json',order_info)
    samples=[s for r in converted for s in json.loads((output/r['folder']/'roundtrip-sample.json').read_text())]
    # Numeric execution has a low virtual-address-space limit. Its spawn entrypoint
    # must not import NumPy/OpenBLAS or Torch before that limit is installed.
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,str(Path(__file__).with_name('l1_roundtrip.py')),'--root',str(output),
                    '--workers',str(workers)],env=env,check=True,timeout=1200)
    roundtrip=json.loads((output/'roundtrip-report.json').read_text());score=roundtrip['score']
    assert score['denominator']==score['matches']==len(samples), 'Finite answer round-trip failed'
    if not smoke: assert len(samples)>=50000
    result={'status':'complete','rows':len(cache),'shards':[r['shard'] for r in converted],
            'manifest_sha256':cache.manifest_sha,'order':order_info,'roundtrip_rows':len(samples),
            'answer_mismatches':0,'roundtrip':score,'seconds':time.monotonic()-started,'GPU':False,
            'roundtrip_executor':roundtrip['executor_entrypoint'],
            'representation_code_sha256':manifest['representation_code_sha256'],'tokenizer_sha256':manifest['tokenizer']['sha256']}
    write(output/'cache-summary.json',result)
    for r in converted:(output/r['folder']/'roundtrip-sample.json').unlink()
    print('COMPACT CACHE COMPLETE',json.dumps({k:result[k] for k in ('rows','manifest_sha256','roundtrip_rows','answer_mismatches','seconds')}),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text())
    inp=Path(cfg['input_root']); scratch=Path(cfg['temp_root'])
    base=materialize_dataset(resolve_dataset(inp,'nlp-hw02-26-token-cache'),scratch/'base','cache-manifest.json')
    ext=materialize_dataset(resolve_dataset(inp,'nlp-hw02-26c-cpu-cache'),scratch/'ext','cache-manifest.json')
    from kaggle_runtime import prepare_starter
    starter=prepare_starter(resolve_dataset(inp,'nlp-hw02-26b-inputs'),scratch/'starter','training-inputs',read_only=True)
    build(base,ext,starter/'tokenizer.json',Path(cfg['working_root'])/'compact-cache')

if __name__=='__main__':main()
