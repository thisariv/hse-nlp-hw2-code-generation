"""Friend-only private dataset preparation; never push kernels or print API errors."""
import argparse
import contextlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIVE = 'friend1/nlp-hw02-l3-live'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['probe', 'download', 'create'])
    p.add_argument('--file')
    p.add_argument('--path', type=Path)
    a = p.parse_args()
    assert os.environ.get('KAGGLE_CONFIG_DIR') == str(Path('~/.kaggle-friend').expanduser()), 'Set friend config directory'
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    if a.command == 'probe':
        datasets = {}
        for page in range(1, 100):
            rows = api.dataset_list(mine=True, page=page) or []
            for row in rows:
                if row.ref in [LIVE, 'friend1/nlp-hw02-p1-live']:
                    assert row.is_private
                    datasets[row.ref] = {'private': True, 'version': row.current_version_number}
            if not rows: break
        assert LIVE in datasets, 'Friend private L3 live unavailable'
        pinned = LIVE+'/'+str(datasets[LIVE]['version'])
        files = []; token = None
        while True:
            response = api.dataset_list_files(pinned, page_token=token, page_size=100)
            files.extend({'name': f.name, 'bytes': f.total_bytes} for f in response.files)
            token = response.next_page_token
            if not token: break
        required = [json.loads((ROOT/'l4-anchors.json').read_text())['anchors']['C-long2x']['file']]
        names = {r['name'] for r in files}
        assert all(n in names for n in required), 'Pinned L3 full state missing in remote live'
        report = {'passed': True, 'account': 'friend1', 'source': pinned, 'datasets': datasets,
                  'files': files, 'required_full_state': required, 'push': False}
        write(ROOT.parent/'checks/kaggle-p1-source-access.json', report)
        print(json.dumps({k: report[k] for k in ['passed', 'account', 'source', 'required_full_state']}))
    elif a.command == 'download':
        assert a.file and a.path
        source = json.loads((ROOT.parent/'checks/kaggle-p1-source-access.json').read_text())['source']
        a.path.mkdir(parents=True, exist_ok=True)
        # SDK prints dataset URL/license only; suppress all normal SDK output anyway.
        with contextlib.redirect_stdout(io.StringIO()):
            api.dataset_download_file(source, a.file, path=str(a.path), force=True, quiet=True)
        for zipped in a.path.glob('*.zip'):
            with zipfile.ZipFile(zipped) as archive:
                members = [n for n in archive.namelist() if Path(n).name == a.file]
                assert len(members) == 1
                with archive.open(members[0]) as src, (a.path/a.file).open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src, dst, 4*1024**2)
            zipped.unlink()
        assert (a.path/a.file).is_file()
        print('Downloaded', a.file, (a.path/a.file).stat().st_size)
    else:
        folder = ROOT/'staging/p1-live'
        folder.mkdir(parents=True, exist_ok=True)
        write(folder/'dataset-metadata.json', {'id': 'friend1/nlp-hw02-p1-live',
              'title': 'NLP HW02 P1 live', 'licenses': [{'name': 'CC0-1.0'}]})
        write(folder/'ready.json', {'session': 'P1', 'status': 'ready; no training yet', 'owner': 'friend1'})
        found = False
        for page in range(1, 100):
            rows = api.dataset_list(mine=True, page=page) or []
            for row in rows:
                if row.ref == 'friend1/nlp-hw02-p1-live':
                    assert row.is_private; found = True
            if found or not rows: break
        if not found:
            with contextlib.redirect_stdout(io.StringIO()):
                response = api.dataset_create_new(str(folder), public=False, quiet=True, convert_to_csv=False)
            assert not response.error, 'Private dataset creation failed'
        status_endpoint_http = None
        try:
            status = api.dataset_status('friend1/nlp-hw02-p1-live')
        except Exception as error:
            status_endpoint_http = getattr(getattr(error, 'response', None), 'status_code', None)
            if status_endpoint_http != 403: raise
            # This OAuth scope can create/list/download private data but denies status.
            # File availability is checked separately; do not retry the upload.
            listed = api.dataset_list_files('friend1/nlp-hw02-p1-live/1', page_size=100)
            assert any(f.name == 'ready.json' and f.total_bytes == (folder/'ready.json').stat().st_size for f in listed.files)
            status = 'ready (file listing verified; status endpoint denied)'
        write(ROOT.parent/'checks/kaggle-p1-live-created.json', {'private': True, 'account': 'friend1',
              'ref': 'friend1/nlp-hw02-p1-live', 'status': status, 'status_endpoint_http': status_endpoint_http,
              'already_existed': found, 'push': False})
        print('Private P1 live', status)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('P1 API FAILED', type(error).__name__, getattr(getattr(error, 'response', None), 'status_code', None))
        sys.exit(1)
