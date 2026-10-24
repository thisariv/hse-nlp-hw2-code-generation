"""Create/verify private D1 seed Dataset, using friend config only. No kernels API."""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REF = 'friend1/nlp-hw02-d1-live'
STEP = 'startup'


def main():
    global STEP
    assert os.environ.get('KAGGLE_CONFIG_DIR') == str(Path('~/.kaggle-friend').expanduser())
    for name in ('KAGGLE_USERNAME', 'KAGGLE_KEY', 'KAGGLE_API_TOKEN'): os.environ.pop(name, None)
    # Authentication is delegated to the SDK. Never open credential files ourselves.
    from kaggle.api.kaggle_api_extended import KaggleApi
    sys.path.insert(0, str(ROOT/'src'))
    from kaggle_io import sha, write
    from d1_data import WEIGHTS, verify_receipt
    folder = ROOT/'staging/d1-live'; verify_receipt(folder/'seed.receipt.json')
    pin = json.loads((ROOT/'d1-seed.json').read_text())
    assert sha(folder/'seed-evidence.npz') == pin['sha256']
    write(folder/'dataset-metadata.json', dict(id=REF, title='NLP HW02 D1 live', licenses=[dict(name='other')],
          description='Private D1 seed and incremental row-loss/execution diagnostics. Original-pair filtering only.'))
    STEP = 'authenticate'; api = KaggleApi(); api.authenticate()
    mine = {}
    for page in range(1, 100):
        STEP = 'list-private-datasets'; rows = api.dataset_list(mine=True, page=page) or []
        for row in rows:
            if row.ref in (REF, 'friend1/nlp-hw02-l3-live', 'friend1/nlp-hw02-compact-cache-verified'):
                assert row.is_private
                mine[row.ref] = dict(private=True, version=row.current_version_number)
        if not rows: break
    assert 'friend1/nlp-hw02-l3-live' in mine
    files = []; token = None
    while True:
        STEP = 'list-L3-files'
        result = api.dataset_list_files('friend1/nlp-hw02-l3-live', page_token=token, page_size=100)
        files.extend(f.name for f in result.files); token = result.next_page_token
        if not token: break
    assert all(f'model-{name}__weights__l2-62280-fp16.pt' in files for name in WEIGHTS)
    found = REF in mine
    if not found:
        STEP = 'create-private-D1'
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            response = api.dataset_create_new(str(folder), public=False, quiet=True, convert_to_csv=False)
        assert not response.error, 'Private D1 creation failed'
    # Do not overwrite an existing live run on rerun; verify seed by remote byte count.
    STEP = 'verify-D1-files'; response = api.dataset_list_files(REF, page_size=100)
    seed = [f for f in response.files if f.name in ('seed-evidence.npz', 'd1__seed-evidence.npz')]
    assert len(seed) == 1 and seed[0].total_bytes == pin['bytes']
    verified = False
    for page in range(1, 100):
        STEP = 'verify-private-D1'; rows = api.dataset_list(mine=True, page=page) or []
        for row in rows:
            if row.ref == REF:
                assert row.is_private; verified = True; version = row.current_version_number
        if verified or not rows: break
    assert verified, 'Owner-private D1 identity not visible'
    receipt = dict(passed=True, account='friend1', ref=REF, private=True, version=version,
        already_existed=found, seed_sha256=pin['sha256'], seed_bytes=pin['bytes'],
        remote_seed_file=seed[0].name, source_datasets=mine, L3_final_weight_files_verified=True,
        GPU=False, kernel_push=False)
    write(ROOT.parent/'checks/kaggle-d1-live-created.json', receipt)
    print('Private D1 live verified; seed bytes', pin['bytes'], '; GPU/push=0')


if __name__ == '__main__':
    try: main()
    except Exception as error:
        print('D1 API FAILED', STEP, type(error).__name__, getattr(getattr(error, 'response', None), 'status_code', None))
        sys.exit(1)
