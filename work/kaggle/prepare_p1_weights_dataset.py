"""Stage only three FP16 model-only P1 weights and their original checkpoint JSON.

No credential reads or API calls. Kaggle CLI creates datasets privately by
default; use the printed command with the default ~/.kaggle configuration.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
SOURCE = WORK/'results/kaggle-runs/nlp-hw02-kaggle-p1/final'
STAGING = ROOT/'staging/p1_weights_mine'
FILES = [('pilot-a', 'long'), ('pilot-b', 'long-truncation'), ('reviewer', 'final')]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def prepare():
    STAGING.mkdir(parents=True, exist_ok=True)
    receipts = []
    allowed = {'dataset-metadata.json'}
    for name, point in FILES:
        weight = f'model-{name}__weights__{point}-fp16.pt'
        checkpoint = f'model-{name}__points__{point}__checkpoint.json'
        expected = json.loads((SOURCE/checkpoint).read_text())
        assert expected['dtype'] == 'FP16' and expected['optimizer_included'] is False
        assert (SOURCE/weight).stat().st_size == expected['bytes'] and sha(SOURCE/weight) == expected['sha256']
        for filename in [weight, checkpoint]:
            destination = STAGING/filename
            if not destination.exists():
                shutil.copy2(SOURCE/filename, destination)
            assert sha(destination) == sha(SOURCE/filename)
            allowed.add(filename)
            receipts.append({'file': filename, 'bytes': destination.stat().st_size, 'sha256': sha(destination)})
    assert not {p.name for p in STAGING.iterdir()}-allowed, 'Unexpected staging files; never upload them'
    metadata = {'id': 'thisarivson/nlp-hw02-p1-weights', 'title': 'NLP HW02 P1 weights',
                'licenses': [{'name': 'CC0-1.0'}],
                'description': 'Model-only FP16 pilot-a, pilot-b and reviewer from the final P1 run, with original checkpoint JSON and SHA256. No optimizer or training data.'}
    (STAGING/'dataset-metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    receipt = WORK/'checks/final-inference-v7-dataset.json'
    previous = json.loads(receipt.read_text()) if receipt.exists() else {}
    if previous.get('created'):
        assert previous['id'] == metadata['id'] and previous['files'] == receipts
    report = {**previous, 'prepared': True, 'id': metadata['id'], 'private': True, 'files': receipts,
              'staging_relative_to_kaggle': 'staging/p1_weights_mine', 'created': previous.get('created', False),
              'credential_contents_read': False,
              'command_from_kaggle_directory': 'env -u KAGGLE_USERNAME -u KAGGLE_KEY -u KAGGLE_API_TOKEN -u KAGGLE_CONFIG_DIR kaggle datasets create -p staging/p1_weights_mine -r skip'}
    receipt.write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    report = prepare()
    print(json.dumps({k: report[k] for k in ['prepared', 'id', 'private', 'command_from_kaggle_directory']}, indent=2))
