#!/usr/bin/env python3
"""Build approved S3. No network calls or GPU launch during build."""
import hashlib, json, pathlib, shutil, sys
ROOT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import write
from build import INSTALL

def main():
    sources={p.name:p.read_text() for p in sorted((ROOT/'src').glob('*.py'))}
    sources['s3_worker.py']=(ROOT/'s3_worker.py').read_text()
    original=json.loads((ROOT/'source-identity.json').read_text())
    for name,digest in original['modules'].items():assert hashlib.sha256(sources[name].encode()).hexdigest()==digest
    config={'input_root':'/kaggle/input','working_root':'/kaggle/working','session_kind':'s3',
            'budget_seconds':5400,'worker_deadline_seconds':5100,'hard_shutdown_seconds':6000,'session_ceiling_seconds':6300,
            'terminate_grace_seconds':20,'output_policy':'individual files; no ZIP; no live upload',
            'cache_dataset':'nlp-hw02-26-token-cache','resume_dataset':'nlp-hw02-26-resume','starter_dataset':'nlp-hw02-26b-inputs',
            'previous_output':'nlp-hw02-kaggle-s1','b_output':'nlp-hw02-kaggle-s2',
            'previous_source':'thisarivson/nlp-hw02-kaggle-s1/1','b_source':'thisarivson/nlp-hw02-kaggle-s2/2',
            'source_sha':original['code_sha256'],'port_sha':hashlib.sha256(json.dumps(sources,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    cells=[]
    def cell(kind,text):
        c={'id':f's3-{len(cells):02d}','cell_type':kind,'metadata':{},'source':text}
        if kind=='code':c.update(outputs=[],execution_count=None)
        cells.append(c)
    cell('markdown',(ROOT.parent/'notes/s3-data-representations-and-b-scoring.md').read_text().split('---',2)[-1].lstrip())
    cell('code',"import time,json,os,sys,tempfile\nfrom pathlib import Path\nSESSION_START=time.monotonic()\nCONFIG="+repr(config)+"\nWORKING=Path(CONFIG['working_root']);WORKING.mkdir(exist_ok=True)\nCONFIG['session_start_monotonic']=SESSION_START\nCONFIG['deadline_monotonic']=SESSION_START+5100\nCONFIG['hard_deadline_monotonic']=SESSION_START+6000\nscratch_base=Path('/kaggle/temp') if Path('/kaggle/temp').is_dir() else Path(tempfile.gettempdir())\nCONFIG['temp_root']=tempfile.mkdtemp(prefix='nlp-s3-',dir=scratch_base)\nMODULE_DIR=WORKING/'src';MODULE_DIR.mkdir(exist_ok=True)\n(MODULE_DIR/'session_watchdog.py').write_text("+repr(sources['session_watchdog.py'])+")\nsys.path.insert(0,str(MODULE_DIR))\nfrom session_watchdog import Watchdog\nWATCHDOG=Watchdog(CONFIG).start()\nprint('Watchdog armed: workers85min; normal budget90min; emergency exit100min; server ceiling105min',flush=True)\n")
    cell('code',INSTALL)
    cell('code',"SOURCES="+repr(sources)+"\nfor name,source in SOURCES.items():(MODULE_DIR/name).write_text(source)\nos.environ['TOKENIZERS_PARALLELISM']='false'\n")
    cell('code',"""import torch
assert torch.cuda.device_count()==2, 'S3 requires exactly two GPUs'
assert all('T4' in torch.cuda.get_device_name(i) for i in (0,1)), 'Wrong accelerator'
print('GPUs:',[torch.cuda.get_device_name(i) for i in (0,1)],flush=True)
from kaggle_io import materialize_dataset,resolve_dataset
from kaggle_runtime import prepare_starter
inp=Path(CONFIG['input_root']);scratch=Path(CONFIG['temp_root'])
CONFIG['starter_dataset']=str(prepare_starter(resolve_dataset(inp,CONFIG['starter_dataset']),scratch/'inputs','training-inputs',read_only=True))
CONFIG['cache_dataset']=str(materialize_dataset(resolve_dataset(inp,CONFIG['cache_dataset']),scratch/'unpacked-cache','cache-manifest.json'))
CONFIG['resume_dataset']=str(materialize_dataset(resolve_dataset(inp,CONFIG['resume_dataset']),scratch/'unpacked-resume','latest.json'))
from session_runner import launch
report=launch(CONFIG,MODULE_DIR/'s3_worker.py',watchdog=WATCHDOG)
print(json.dumps(report,indent=2))
print('OUTPUT: individual files; no ZIP; no live upload',WORKING)
""")
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python'}},'cells':cells}
    target=ROOT/'notebooks/nlp-hw02-kaggle-s3.ipynb';write(target,notebook)
    folder=ROOT/'bundles/s3';folder.mkdir(parents=True,exist_ok=True);shutil.copyfile(target,folder/target.name)
    write(folder/'kernel-metadata.json',{'id':'thisarivson/nlp-hw02-kaggle-s3','title':'nlp hw02 kaggle s3','code_file':target.name,
        'language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_tpu':False,'enable_internet':True,
        'machine_shape':'NvidiaTeslaT4','dataset_sources':['thisarivson/'+config[k] for k in ('cache_dataset','resume_dataset','starter_dataset')],
        'kernel_sources':[config['previous_source'],config['b_source']],'competition_sources':[]})
    write(folder/'session-limits.json',{'session_timeout_seconds':6300,'budget_seconds':5400,'watchdog_exit_seconds':6000})
    print(folder)

if __name__=='__main__':main()
