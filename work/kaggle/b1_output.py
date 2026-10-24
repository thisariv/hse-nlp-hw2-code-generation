#!/usr/bin/env python3
"""Read-only pinned single-file output CLI; explicit version_label fixes Kaggle CLI2.2.4.

SDK authenticates using the caller-selected KAGGLE_CONFIG_DIR. No auth/URLs printed.
"""
import argparse
from pathlib import Path


def download(api, source, name, destination, get=None):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    if get is None:
        import requests
        get = requests.get
    owner, slug, version = source.split('/'); assert version.isdecimal()
    token = None; matches = []
    while True:
        req = ApiListKernelSessionOutputRequest()
        req.user_name = owner; req.kernel_slug = slug; req.version_label = 'v'+version; req.page_size = 200
        if token: req.page_token = token
        with api.build_kaggle_client() as client:
            response = client.kernels.kernels_api_client.list_kernel_session_output(req)
        matches.extend(r for r in (response.files or []) if r.file_name == name)
        token = response.next_page_token
        if not token: break
    assert len(matches) == 1, 'Pinned output file missing/ambiguous'
    target = Path(destination)/name
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    target.parent.mkdir(parents=True, exist_ok=True); tmp = target.with_name(target.name+'.download-tmp')
    try:
        with get(matches[0].url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with tmp.open('wb') as f:
                for chunk in response.iter_content(4*1024**2): f.write(chunk)
        tmp.replace(target)
    finally: tmp.unlink(missing_ok=True)
    return target


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--name',required=True)
    p.add_argument('--path',type=Path,required=True);a=p.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate();download(api,a.source,a.name,a.path)
