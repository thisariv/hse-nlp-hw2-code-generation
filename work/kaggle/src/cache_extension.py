"""26c: same selection/executor/token schema, independent immutable42–59 cache."""
import json,time,shutil
from pathlib import Path
from collections import Counter

def run(inputs,root,source_sha,specs=None,smoke=False,download=None,workers=2):
    import exp26_cache as C
    import exp26_protocol as P
    from exp26_input import load_protected
    from exp26_common import write_json,read_json,json_hash,sha_file,file_record
    from tokenizers import Tokenizer
    import pyarrow.parquet as pq
    root=Path(root);root.mkdir(parents=True,exist_ok=True);inputs=Path(inputs)
    specs=specs or P.SHARDS[42:60]
    if not smoke:assert [s['shard'] for s in specs]==list(range(42,60))
    protected,_=load_protected(inputs/'protected-questions.json',strict=not smoke)
    tokenizer=Tokenizer.from_file(str(inputs/'tokenizer.json'))
    if not smoke:assert sha_file(inputs/'tokenizer.json')==P.TOKENIZER_SHA
    identity={'protocol':P.CACHE_PROTOCOL,'code_sha256':source_sha,'tokenizer_sha256':sha_file(inputs/'tokenizer.json'),
              'protected_sha256':sha_file(inputs/'protected-questions.json'),'specs':specs,'smoke':smoke,
              'extension_code_sha256':sha_file(Path(__file__))}
    identity_sha=json_hash(identity);master=root/'cache-manifest.json'
    if master.exists():assert read_json(master)['identity']==identity
    manifest={'format':'nlp-hw02-token-cache-v1','experiment':'26c','identity':identity,'identity_sha256':identity_sha,
              'shard_range':[42,59],'base_cache_sha256':'2c691fef64fea8b4186b8acae54561d5620080286196a4390a23108776e80add',
              'rows':0,'shards':[],'sealed':False,'fresh_evaluations':0,'train_protected_question_overlaps':0,
              'processing_policy':'all42-59; base0-41 not mounted or modified'}
    start=time.monotonic()
    for spec in specs:
        folder=f'shards/part-{spec["shard"]:05d}';out=root/folder;rp=out/'manifest.json'
        if rp.exists():
            receipt=read_json(rp);assert receipt['identity_sha256']==identity_sha and receipt['raw_sha256']==spec['sha256']
            C.validate_shard(out,receipt,protected)
        else:
            started=time.monotonic();path=(download or C.download_raw)(spec,root/'raw')
            assert sha_file(path)==spec['sha256'] and path.stat().st_size==spec['bytes']
            raw_n=pq.ParquetFile(path).metadata.num_rows;rows,ids,counts=C.collect_candidates(path,protected)
            assert counts['raw_rows']==raw_n
            execution=C.execute_journal([r['code'] for r in rows],out,identity_sha,workers)
            keep=[i for i,r in enumerate(execution['rows']) if r['status']=='numeric']
            selected=[rows[i] for i in keep];locations=[ids[i] for i in keep]
            tokens=C.build_tokens(selected,locations,tokenizer,out,shard_id=spec['shard'],identity=json_hash({'cache':identity_sha,'rows':locations}))
            receipt={'shard':spec['shard'],'folder':folder,'identity_sha256':identity_sha,'raw_sha256':spec['sha256'],'raw_num_rows':raw_n,
                     'selection_counts':counts,'pre_execution_rows':len(rows),'reference_statuses':dict(Counter(r['status'] for r in execution['rows'])),
                     'reference_non_numeric_dropped':len(rows)-len(keep),'reference_drop_fraction':1-len(keep)/len(rows) if rows else 0.,
                     'execution':execution['summary'],'tokens':tokens,'wall_seconds':time.monotonic()-started,
                     'protected_question_overlaps':0,'hash_heldout_overlaps':0,'files':[file_record(out/n) for n in C.FILES]}
            C.validate_shard(out,receipt,protected);write_json(rp,receipt)
            shutil.rmtree(out/'token-work') # committed local temporary token binaries only
        manifest['shards'].append(receipt);manifest['rows']+=receipt['tokens']['rows'];manifest['processed_shards']=len(manifest['shards'])
        manifest['downloaded_shards']=len(manifest['shards']);write_json(master,manifest)
        print('26c',spec['shard'],'rows',receipt['tokens']['rows'],'total',manifest['rows'],'seconds',round(time.monotonic()-start,1),flush=True)
    manifest.update(sealed=True,status='ready_extension42-59',source_tokens=sum(s['tokens']['source_tokens'] for s in manifest['shards']),
                    target_tokens_with_EOS=sum(s['tokens']['target_tokens_with_EOS'] for s in manifest['shards']),
                    unique_row_identity_policy='Original(shard,row)once, no text deduplication')
    write_json(master,manifest);write_json(root/'26c-summary.json',{'experiment':'26c','status':'complete','rows':manifest['rows'],
        'shards':[s['shard'] for s in manifest['shards']],'manifest_sha256':sha_file(master),'wall_seconds':time.monotonic()-start,
        'GPU':False,'fresh_evaluations':0,'reference_statuses':dict(sum((Counter(s['reference_statuses']) for s in manifest['shards']),Counter()))})
    return manifest
