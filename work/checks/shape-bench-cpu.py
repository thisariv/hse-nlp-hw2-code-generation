"""Meaningful CPU checks: shapes/counts, backward/Adam, forced EOS+compaction."""
import ast
import hashlib
import json
import sys
from pathlib import Path
import nbformat
import torch

WORK=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(WORK/'kaggle/src'))
import shape_bench as B
from shape_specs import SHAPES, name, spec, parameter_formula


def main():
    ready=WORK/'results/next-shape-bench.json'
    ready.write_text(json.dumps(dict(bundle='kaggle/bundles/shape_bench_mine',ready=False))+'\n')
    torch.set_num_threads(1)
    counts={}
    for shape in SHAPES:
        with torch.device('meta'): model=B.C.SimpleTransformer(**spec(shape))
        counts[name(shape)]=B.parameter_counts(model)
        assert counts[name(shape)]==parameter_formula(shape)
    assert counts['e6-d6-w512']['total']==53646403
    assert counts['e8-d8-w640']['total']==103902979
    device=torch.device('cpu')
    model=B.C.SimpleTransformer(d_model=16,heads=2,encoder_layers=2,decoder_layers=2,swiglu_dim=40,vocab=8195,dropout=0.)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    before=model.head.weight.detach().clone()
    B.update(model,opt,scaler,[[6,5],[8,7],[7,6],[5,4]],2,device)
    assert not torch.equal(model.head.weight,before)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    lengths=[[7,3],[8,35],[6,69],[9,100]]
    stats=[]
    for k,temp in [(1,None),(4,.7)]:
        model.eval()
        s=B.forced(model,lengths,k,temp,device,None); stats.append(s)
        assert s['compactions'] and len(s['compactions'])>=2
    bundle=WORK/'kaggle/bundles/shape_bench_mine'
    identity=json.loads((bundle/'source-identity.json').read_text())
    notebook=nbformat.read(bundle/'nlp-hw02-shape-bench.ipynb',as_version=4); nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code': compile(cell.source,cell.id,'exec')
    tree=ast.parse(notebook.cells[1].source)
    literals={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
              and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('SOURCES','LENGTHS')}
    for n,digest in identity['source_sha256'].items():
        assert hashlib.sha256((WORK/'kaggle/src'/n).read_bytes()).hexdigest()==digest
        assert (WORK/'kaggle/src'/n).read_text()==literals['SOURCES'][n]
    assert hashlib.sha256((bundle/'nlp-hw02-shape-bench.ipynb').read_bytes()).hexdigest()==identity['notebook_sha256']
    assert literals['LENGTHS']==json.loads((WORK/'kaggle/shape-lengths.json').read_text())
    metadata=json.loads((bundle/'kernel-metadata.json').read_text())
    assert metadata['is_private'] and not metadata['enable_internet'] and metadata['machine_shape']=='NvidiaTeslaT4'
    assert metadata['id'].split('/')[1]=='-'.join(metadata['title'].lower().split())
    assert metadata['kernel_sources']==['thisarivson/nlp-hw02-compact-cache-verified/2']
    report=dict(passed=True,parameter_counts=counts,real_backward_Adam_finite=True,
                forced_EOS_greedy_sample_compaction=stats,nbformat_compile_embedded_identity=True,
                GPU_run=False,push=False)
    (WORK/'checks/shape-bench-cpu.json').write_text(json.dumps(report,indent=2)+'\n')
    ready.write_text(json.dumps(dict(bundle='kaggle/bundles/shape_bench_mine',ready=True,
        CPU_static_passed=True,GPU_measured=False,push=False,receipt='checks/shape-bench-cpu.json'),indent=2)+'\n')
    print('shape bench CPU/static passed; ready:true; GPU/push=0')


if __name__=='__main__': main()
