"""Read-only latest-version observation; never emit URLs from the output inventory or auth."""
import argparse, json
from datetime import datetime, timezone
from b6_prepare import ReadClient
from build_l3 import ROOT, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    slug = 'nlp-hw02-b7' + ('-gpu-smoke' if a.smoke else '')
    bundle = ROOT/'bundles'/('b7_gpu_smoke_friend1' if a.smoke else 'b7_friend1')
    meta = json.loads((bundle/'kernel-metadata.json').read_text())
    local = json.loads((bundle/meta['code_file']).read_text())
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiListKernelSessionOutputRequest
    api = ReadClient().api
    req = ApiGetKernelRequest(); req.user_name = 'friend1'; req.kernel_slug = slug
    with api.build_kaggle_client() as c:
        remote = c.kernels.kernels_api_client.get_kernel(req)
    m = remote.metadata
    assert m.is_private and m.enable_gpu and m.enable_internet and m.machine_shape == 'NvidiaTeslaT4'
    server = json.loads(remote.blob.source)
    assert [c['source'] for c in server['cells'] if c['cell_type']=='code'] == [c['source'] for c in local['cells'] if c['cell_type']=='code']
    status = api.kernels_status(meta['id']).status.name
    req = ApiListKernelSessionOutputRequest(); req.user_name = 'friend1'; req.kernel_slug = slug; req.page_size = 200
    with api.build_kaggle_client() as c:
        out = c.kernels.kernels_api_client.list_kernel_session_output(req)
    row = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), ref=meta['id'],
        version=m.current_version_number, status=status, private=True, machine_shape=m.machine_shape,
        exact_server_code=True, output_first_page_files=len(out.files or []), has_output=bool(out.files), has_log=bool(out.log))
    path = ROOT.parent/'checks'/('b7-review-smoke-observations.json' if a.smoke else 'b7-review-production-observations.json')
    history = json.loads(path.read_text()) if path.exists() else []
    history.append(row); write(path, history); print(json.dumps(row))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('B7 observation failed:', type(e).__name__, 'details suppressed')
        raise SystemExit(1)
