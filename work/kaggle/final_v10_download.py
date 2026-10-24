"""Read-only version-pinned v10 model download; SDK owns authentication."""
import contextlib, hashlib, io, json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent

def main():
    for k in ('KAGGLE_USERNAME','KAGGLE_KEY','KAGGLE_API_TOKEN'):os.environ.pop(k,None)
    os.environ['KAGGLE_CONFIG_DIR']=str(Path('~/.kaggle-friend').expanduser())
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
    from b1_output import download
    from build_final_inference_v10 import default_config
    files=[]
    dest=WORK/'results/kaggle-runs/nlp-hw02-b6/v10-pinned-1'
    path=download(api,'friend1/nlp-hw02-b6/1','b1-output-manifest.json',dest)
    assert path.read_bytes()==(WORK/'results/kaggle-runs/nlp-hw02-b6/1/b1-output-manifest.json').read_bytes()
    for m in default_config()['models']:
        destination=dest if m['name']!='pilot-b'else WORK/'results/kaggle-runs/nlp-hw02-kaggle-p1/v9-pinned-3'
        path=destination/m['relative_path']
        if not path.exists():path=download(api,m['source_ref'],m['relative_path'],destination)
        digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==m['sha256']
        files.append(dict(source=m['source_ref'],file=m['relative_path'],sha256=digest,bytes=path.stat().st_size,local=str(path.relative_to(WORK))))
        print(m['name'],'SHA verified',flush=True)
    (WORK/'checks/final-inference-v10-download.json').write_text(json.dumps(dict(files=files,B6_version1_manifest_exact=True,credentials_contents_read=False),indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except Exception as e:print(json.dumps(dict(error_type=type(e).__name__,reason=str(e)if isinstance(e,AssertionError)else'API failed; raw response suppressed')));raise SystemExit(1)
