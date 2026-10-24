"""D1 uses L3/P1 bounded private live-sync; stage immutable data receipts only."""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from kaggle_io import sha, write, safe_name
from l2_live import L2LiveSync
from s2_live_sync import audit_tree, privacy_probe
from d1_data import verify_receipt

LIMIT = 1500*1024**2


def stage_snapshot(working, stage, ref, reason):
    working, stage = Path(working), Path(stage)
    assert ref == 'friend1/nlp-hw02-d1-live'
    stage.mkdir(parents=True, exist_ok=True)
    audit_tree(working, ref)
    for p in stage.iterdir():
        if p.is_dir(): shutil.rmtree(p)
        else: p.unlink()
    selected = set(working.glob('*.json'))
    selected.discard(working/'session-config.json')
    receipts = sorted((working/'d1').rglob('*.receipt.json'))
    complete_folders = {p.parent for p in receipts if p.name == 'complete.receipt.json'}
    selected.update(p for p in (working/'d1').rglob('*.json') if not p.name.endswith('.receipt.json'))
    for receipt in receipts:
        # A complete consolidation replaces loss blocks in CURRENT live version.
        if receipt.name.startswith('block-') and receipt.parent in complete_folders: continue
        rec = verify_receipt(receipt)
        selected.add(receipt)
        selected.update(receipt.parent/f['file'] for f in rec['files'])
    total = 0; files = []
    for path in sorted(selected):
        if not path.is_file() or '.tmp' in path.name: continue
        assert not path.is_symlink()
        relative = path.relative_to(working).as_posix()
        flat = relative.replace('/', '__'); target = stage/flat
        total += path.stat().st_size
        if total > LIMIT: raise RuntimeError('D1 staging size limit')
        shutil.copyfile(path, target)
        files.append(dict(file=flat, original_path=relative, bytes=target.stat().st_size,
                          sha256=sha(target), gzip=False))
    write(stage/'recover-manifest.json', dict(format='nlp-hw02-d1-live-v1', files=files,
          payload_bytes=total, reason=reason, unix_time=time.time(), credentials_in_files=False))
    write(stage/'dataset-metadata.json', dict(id=ref, title='NLP HW02 D1 live',
          licenses=[dict(name='other')], description='Private row-loss and label-noise diagnostics; original-pair filtering only.'))
    audit_tree(stage, ref)


def restore_snapshot(source, destination):
    from l3_resume import live_bytes
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source/'recover-manifest.json').read_text())
    assert manifest['format'] == 'nlp-hw02-d1-live-v1'
    for rec in manifest['files']:
        p = destination/safe_name(rec['original_path']); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(live_bytes(source, rec))
    return destination


class D1LiveSync(L2LiveSync):
    def bounded_call(self, args, timeout, final=False):
        # Retain reviewed auth/privacy/ready/retry/timeout logic, swap only stager.
        if len(args) > 2 and args[2] == 'stage' and Path(args[1]).name == 'l2_live.py':
            args = [*args]; args[1] = str(Path(__file__).resolve())
        return super().bounded_call(args, timeout, final=final)

    def signature(self):
        # Batch new artifacts into milestones; uploading every GPU batch would
        # waste bandwidth and may keep the Dataset in processing for hours.
        root = self.working/'d1'; milestones = []
        for model in ('C-main', 'C-long2x'):
            folder = root/model
            milestones.append((model, len(list((folder/'loss').glob('block-*.receipt.json')))//8,
                len(list(folder.glob('generation-*.receipt.json')))//16))
        milestones.extend((str(p.relative_to(root)), p.stat().st_mtime_ns)
            for p in root.rglob('*.receipt.json') if p.name in ('complete.receipt.json', 'selection.receipt.json'))
        return sorted(milestones)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['stage', 'restore', 'privacy'])
    for name in ('working', 'stage', 'ref', 'reason', 'destination', 'receipt'): parser.add_argument('--'+name)
    args = parser.parse_args()
    try:
        if args.mode == 'stage': stage_snapshot(args.working, args.stage, args.ref, args.reason)
        elif args.mode == 'privacy': privacy_probe(args.ref, args.receipt)
        else: restore_snapshot(args.stage, args.destination)
    except Exception: sys.exit(1)
