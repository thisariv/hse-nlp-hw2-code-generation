"""Actual tiny CPU worker/executor round trip for every selected TTA branch."""
import ast,hashlib,json,os,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
import torch
from final_core import SimpleTransformer,EOS

def main():
    nb=json.loads((ROOT/'notebooks/nlp-hw02-tta.ipynb').read_text())
    code=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    sources=ast.literal_eval(next(n.value for n in ast.parse(code).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)and t.id=='SOURCES'for t in n.targets)))
    with tempfile.TemporaryDirectory(prefix='nlp-tta-cpu-')as folder:
        root=Path(folder);modules=root/'src';modules.mkdir()
        for k,v in sources.items():(modules/k).write_text(v)
        rows=[dict(question=q)for q in ['Pay $5.00 each for 1,000 units?','There are 1,234.50 units?','Pay $1.50 each?','Nothing to change?']]
        inputs={'heldout4k.json':str(root/'rows.json'),'heldout4k-reference-execution.json':str(root/'refs.json'),
                'tokenizer.json':str(WORK/'results/kaggle-runs/nlp-hw02-kaggle-s1/1/inputs/tokenizer.json')}
        (root/'rows.json').write_text(json.dumps(rows));(root/'refs.json').write_text(json.dumps(dict(rows=[dict(status='numeric',value=1.)]*len(rows))))
        models=[]
        for i in range(2):
            spec=dict(d_model=16,heads=2,encoder_layers=1,decoder_layers=1,swiglu_dim=32,vocab=8195,max_positions=512,qk_norm='layernorm'if i else False)
            model=SimpleTransformer(**spec,dropout=0.).eval()
            with torch.no_grad():model.head.weight.zero_();model.head.bias.fill_(-1000.);model.head.bias[EOS]=1000.
            path=root/f'model-{i}.pt';torch.save(dict(format='nlp-hw02-model-only-v1',optimizer_included=False,model_spec=spec,model=model.half().state_dict()),path)
            models.append(dict(name=f'M{i}',path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        cfg=dict(cpu_fixture=True,inputs=inputs,models=models,rules=['dollar-words','ungroup','price-trim'],K=8,batch=2,cap=16,seed=270042,output=str(root/'out'))
        path=root/'config.json';path.write_text(json.dumps(cfg))
        r=subprocess.run([sys.executable,str(modules/'tta_worker.py'),str(path)],capture_output=True,text=True,timeout=80,env=dict(os.environ,TTA_DEADLINE=str(time.monotonic()+75)))
        assert r.returncode==0,r.stdout+r.stderr
        report=json.loads((root/'out/summary.json').read_text());assert report['status']=='complete'and len(report['models'])==2
        n=0
        for p in (root/'out').glob('*/*/*-batch-*.json'):
            for row in json.loads(p.read_text()):
                assert len(row['samples']['0.7'])==8
                assert all('execution'in c for c in [row['greedy']]+row['samples']['0.7'])
                assert row['question_sha256']==hashlib.sha256(rows[row['id']]['question'].encode()).hexdigest()
                n+=1
        assert n==2*sum(report['coverage'].values()),(n,report['coverage'])
        result=dict(passed=True,actual_embedded_worker=True,QK_and_standard_models=True,K8_executed=True,rows=n,coverage=report['coverage'],seconds=report['seconds'])
        (WORK/'checks/tta-pipeline.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
