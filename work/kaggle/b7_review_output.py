"""Read-only audit of compact reviewer downloads. Does not download PT/order or launch GPU."""
import argparse,hashlib,json,sys
from pathlib import Path
from build_l3 import ROOT,write,literals
sys.path.insert(0,str(ROOT/'src'))
from kaggle_io import sha
from b5_static_check import reviewed_handoff


def relative_receipt(path, root=ROOT.parent):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def audit(folder,smoke):
    folder=Path(folder);cfg=json.loads((folder/'session-config.json').read_text())
    bundle=ROOT/'bundles'/('b7_gpu_smoke_friend1' if smoke else 'b7_friend1')
    identity=json.loads((bundle/'source-identity.json').read_text())
    metadata=json.loads((bundle/'kernel-metadata.json').read_text())
    nb=json.loads((bundle/metadata['code_file']).read_text())
    expected_config=literals(next(c['source'] for c in nb['cells'] if c['cell_type']=='code'))['CONFIG']
    assert all(cfg.get(k)==v for k,v in expected_config.items()), 'Smoke/production configuration differs from reviewed bundle'
    expected=json.loads((ROOT/'b6-anchors.json').read_text())
    assert cfg['owner']=='friend1' and cfg['gpu_smoke'] is smoke and cfg['b6_pins']==expected
    assert cfg['port_sha256']==hashlib.sha256(json.dumps(dict(SOURCES={n:(ROOT/'src'/n if (ROOT/'src'/n).exists() else ROOT/n).read_text() for n in identity['modules']},
        C_OVERRIDES={n:(ROOT/'legacy-c'/n).read_text() for n in identity['C_overrides']}),sort_keys=True).encode()).hexdigest()
    inventory=json.loads((folder/'b1-output-manifest.json').read_text())['files'];verified=0
    for r in inventory:
        path=folder/r['file']
        if path.is_file():assert path.stat().st_size==r['bytes'] and sha(path)==r['sha256'];verified+=1
    session=json.loads((folder/'session-summary.json').read_text());workers=[]
    for g in ('0','1'):
        root=folder/f'model-GPU{g}';r=json.loads((root/'summary.json').read_text());cd=r['cooldown']
        assert r['status']=='complete' and r['final_complete'] and r['main_added_updates']==0
        n=expected_config['N_by_gpu'][g];assert cd['N']==cd['updates']==n and cd['last_lr']==0. and cd['q']==0.
        assert cd['source_sha256']==expected['anchors'][g]['sha256']
        assert cd['cursor']==expected['anchors'][g]['cursor']+128*n
        logs=[json.loads(s) for s in (root/'cooldown-updates.jsonl').read_text().splitlines()]
        assert len(logs)==n and [r['branch_update'] for r in logs]==list(range(1,n+1))
        import math
        assert all(r['lr']==cd['peak_lr']*(1-math.sqrt(j/n)) for j,r in enumerate(logs,1))
        metrics=json.loads((root/'points/b6-final/evaluation/metrics.json').read_text())
        rows=36 if smoke else 2048;k=metrics['K8_tune2048']
        assert k['complete'] and k['rows']==k['target_rows']==rows and k['K']==8 and k['T']==.7 and k['seed']==270042
        assert not k['OOM_attempts'], 'Sampling shape changed; independent review needed'
        assert metrics['test']['rows']==(64 if smoke else 1919)
        if smoke:
            check=json.loads((root/'gpu-smoke.json').read_text());assert check['passed'] and all(check['checks'].values())
        workers.append(dict(gpu=g,N=n,final_lr=0.,source_sha256=cd['source_sha256'],K8_rows=rows))
    assert session['status']=='complete' and session['exit_codes']==[0,0]
    assert session['wall_seconds']<expected_config['session_ceiling_seconds']
    return dict(passed=True,smoke=smoke,modules=identity['modules'],C_overrides=identity['C_overrides'],
        config_sha256=identity['config_sha256'],workers=workers,verified_producer_files=verified,wall_seconds=session['wall_seconds'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=audit(a.folder,a.smoke);write(a.out,r)
    if a.smoke:
        path=ROOT.parent/'results/next-b7.json';hand=json.loads(path.read_text());r['receipt']=relative_receipt(a.out)
        write(path,reviewed_handoff(dict(hand),hand,[r]))
    print('B7 compact output audit passed')
