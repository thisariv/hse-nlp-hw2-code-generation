"""B1 independent workers + explicit common-start/common-prefix A/B receipt."""
import json
from pathlib import Path
from b1_session import launch as base_launch
from kaggle_io import write


def audit(root):
    root=Path(root);starts=[];streams=[]
    for gpu in (0,1):
        folder=root/f'model-GPU{gpu}'
        starts.append(json.loads((folder/'start-fingerprints.json').read_text()))
        rows={}
        with (folder/'updates.jsonl').open() as f:
            for line in f:
                r=json.loads(line);rows[r['updates']]=r['stream_sha256']
        streams.append(rows)
    common=sorted(set(streams[0])&set(streams[1]))
    same_start=starts[0]['fingerprints']==starts[1]['fingerprints'] and starts[0]['source_sha256']==starts[1]['source_sha256']
    same_stream=all(streams[0][i]==streams[1][i] for i in common)
    return dict(passed=same_start and same_stream and bool(common),same_start=same_start,same_stream=same_stream,
                compared_prefix_updates=len(common),last_common_update=common[-1] if common else None,
                note='equal wall budget; different throughput/endpoints; compare common prefix by updates/pairs')


class DeferredFinish:
    def __init__(self,watchdog):self.watchdog=watchdog
    def __getattr__(self,name):return getattr(self.watchdog,name)
    def finish(self):pass  # B4 publication is still protected by the hard watchdog.


def launch(config,worker,watchdog,**kwargs):
    report=base_launch(config,worker,DeferredFinish(watchdog),**kwargs)
    try:
        try:ab=audit(config['working_root'])
        except Exception as e:ab=dict(passed=False,reason=type(e).__name__)
        write(Path(config['working_root'])/'b4-ab.json',ab)
        if not ab['passed']:report['status']='partial';report['AB_audit_failed']=True
        report['AB_audit']=ab;write(Path(config['working_root'])/'session-summary.json',report)
        # Base manifest predates this updated summary; keep its SHA valid.
        path=Path(config['working_root'])/'b1-output-manifest.json'
        manifest=json.loads(path.read_text())
        from kaggle_io import sha
        for name in ('session-summary.json','b4-ab.json'):
            file=path.parent/name;manifest['files']=[r for r in manifest['files'] if r['file']!=name]
            manifest['files'].append(dict(file=name,bytes=file.stat().st_size,sha256=sha(file)))
        write(path,manifest)
    except Exception as e:
        report.update(status='partial',publication_error=type(e).__name__)
        write(Path(config['working_root'])/'session-summary.json',report)
    finally:
        if not report.get('unreaped_worker_pids'):watchdog.finish()
    return report
