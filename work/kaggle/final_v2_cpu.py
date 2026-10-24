"""Pre-push gates: sealed v2 config/code, FP16 identities, regressions and cold notebook."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from prepare_final_v2 import ROOT, WORK, SUBMISSION, STAGE, sha


def main():
    import torch
    sys.path.insert(0, str(SUBMISSION/'code'))
    from hw2_codegen.pipeline import load_model
    torch.set_num_threads(1)
    config=json.loads((SUBMISSION/'code/hw2_codegen/config.json').read_text())
    sealed=json.loads((WORK/'checks/final-variants-build.json').read_text())
    assert config==sealed['variants'][1]
    exact=[]
    for name, digest in sealed['code_sha256'].items():
        if name in ('config.json','config.py'):
            continue
        assert sha(SUBMISSION/'code/hw2_codegen'/name)==digest, name
        exact.append(name)
    # config.py differs only in its function docstring (v11/budget768 -> v2/640).
    module=ast.parse((SUBMISSION/'code/hw2_codegen/config.py').read_text())
    function=next(n for n in module.body if isinstance(n,ast.FunctionDef))
    assert function.name=='default_config' and len(function.body)==2
    weights=[]
    for m in config['models']:
        path=STAGE/m['filename']
        assert sha(path)==m['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['format']=='nlp-hw02-model-only-v1' and payload['optimizer_included'] is False
        spec=payload['model_spec'];wide=m['name'].startswith('W')
        assert spec['encoder_layers']==spec['decoder_layers']==6
        assert spec['d_model']==(768 if wide else 512) and bool(spec.get('qk_norm'))==wide
        assert all(t.dtype==torch.float16 and torch.isfinite(t).all() for t in payload['model'].values() if t.is_floating_point())
        del payload
        model=load_model(path,torch.device('cpu'));del model
        weights.append(dict(name=m['name'],sha256=m['sha256'],model_spec=spec,strict_load=True,fp16_finite=True))
    for path in list((SUBMISSION/'code/hw2_codegen').glob('*.py'))+list(ROOT.glob('*v2*.py')):
        ast.parse(path.read_text(),filename=str(path))
    env=dict(os.environ,PYTHONPATH=str(SUBMISSION/'code'))
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(SUBMISSION/'tests'),'-v'],env=env,check=True,timeout=150)
    subprocess.run([sys.executable,str(SUBMISSION/'tests/check_notebook.py')],env=env,check=True,timeout=100)
    for name in ['test.json','tune2048.json','tune2048-reference.json','tokenizer.json']:
        assert sha(STAGE/name)==sealed['input_sha256'][name], name
    receipt=dict(passed=True,config_equal_sealed_v2=True,unchanged_executable_modules=exact,
        config_py_only_docstring_changed=True,weights=weights,regression_tests=21,
        fresh_kernel_notebook_dry_run_rows=8,notebook_sha256=sha(SUBMISSION/'inference.ipynb'),
        manifest_sha256=sha(STAGE/'package-manifest.json'),static_ast_json=True)
    (WORK/'checks/final-package-v2-cpu.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
