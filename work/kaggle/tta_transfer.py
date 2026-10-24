"""Account-only TTA transfer: immutable notebook, one push, read-only follow-up.

Authentication is delegated to Kaggle; credential files are never inspected here.
Run with the Kaggle CLI's Python environment. No changes to experimental sources.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
BUNDLE = ROOT / 'bundles/tta_friend2'
REF = 'friend2/nlp-hw02-tta'
DATASET = 'friend2/nlp-hw02-tta-inputs'
LEDGER = WORK / 'checks/tta-friend2-launch.json'
CLI = str(Path('~/.local/bin/kaggle').expanduser())


def now():
    return datetime.now(timezone.utc).isoformat()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment(friend2=True):
    env = dict(os.environ)
    for key in ('KAGGLE_USERNAME', 'KAGGLE_KEY', 'KAGGLE_API_TOKEN', 'KAGGLE_CONFIG_DIR'):
        env.pop(key, None)
    if friend2:
        env['KAGGLE_CONFIG_DIR'] = str(Path('~/.kaggle-friend2').expanduser())
    return env


def original_status():
    run = subprocess.run([CLI, 'kernels', 'status', 'thisarivson/nlp-hw02-tta'],
                         env=environment(False), capture_output=True, text=True, timeout=60)
    match = re.search(r'KernelWorkerStatus\.([A-Z_]+)', run.stdout)
    assert run.returncode == 0 and match, 'Original status uncertain; do not push'
    return dict(ref='thisarivson/nlp-hw02-tta', status=match[1], at_utc=now())


def api():
    os.environ.clear()
    os.environ.update(environment_saved)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        client = KaggleApi()
        client.authenticate()
    return client


def remote(client):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = REF.split('/')
    with client.build_kaggle_client() as c:
        result = c.kernels.kernels_api_client.get_kernel(request)
    m = result.metadata
    assert m.is_private and m.enable_gpu and not m.enable_internet and not m.enable_tpu
    assert m.machine_shape == 'NvidiaTeslaT4'
    def codes(nb):
        return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    assert codes(json.loads(result.blob.source)) == codes(read(BUNDLE / 'nlp-hw02-tta.ipynb'))
    expected = read(BUNDLE / 'kernel-metadata.json')
    assert set(m.dataset_data_sources) == set(expected['dataset_sources'])
    assert not m.kernel_data_sources
    return m.current_version_number


def quota(client):
    with client.build_kaggle_client() as c:
        q = c.kernels.kernels_api_client.get_accelerator_quota_statistics()
    g = q.gpu_quota
    return dict(total_seconds=g.total_time_allowed.total_seconds(),
                used_seconds=g.time_used.total_seconds(), reserved_seconds=g.time_reserved.total_seconds(),
                free_seconds=(g.total_time_allowed - g.time_used - g.time_reserved).total_seconds())


def build():
    source = ROOT / 'bundles/tta_mine'
    identity = read(ROOT / 'tta-source-identity.json')
    assert sha(source / 'nlp-hw02-tta.ipynb') == identity['notebook_sha256']
    assert read(source / 'source-identity.json') == identity
    nb = read(source / 'nlp-hw02-tta.ipynb')
    init = ''.join(nb['cells'][1]['source'])
    sources = ast.literal_eval(next(n.value for n in ast.parse(init).body if isinstance(n, ast.Assign)
                                   and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in n.targets)))
    assert {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()} == identity['source_sha256']
    for name, code in sources.items():
        compile(code, name, 'exec')
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'cell', 'exec')
    import nbformat
    nbformat.validate(nbformat.read(source / 'nlp-hw02-tta.ipynb', as_version=4))
    design = (WORK / 'notes/tta-experiment.md').read_text()
    frozen = read(WORK / 'checks/tta-preregistration.json')['text']
    def section(text):
        return text.split('## ПРЕДРЕГИСТРАЦИЯ', 1)[1].split('## Результат', 1)[0]
    assert section(design) == section(frozen), 'Frozen preregistration changed'
    BUNDLE.mkdir(parents=True, exist_ok=True)
    for name in ('nlp-hw02-tta.ipynb', 'source-identity.json', 'session-limits.json'):
        shutil.copyfile(source / name, BUNDLE / name)
    metadata = read(source / 'kernel-metadata.json')
    metadata.update(id=REF, dataset_sources=[DATASET, 'thisarivson/nlp-hw02-26b-inputs'], kernel_sources=[])
    write(BUNDLE / 'kernel-metadata.json', metadata)
    assert sha(BUNDLE / 'nlp-hw02-tta.ipynb') == identity['notebook_sha256']
    receipt = dict(passed=True, at_utc=now(), notebook_sha256=identity['notebook_sha256'],
                   source_identity_exact=True, notebook_bytes_exact=True, config_exact=True,
                   frozen_preregistration_exact=True, modules=len(sources), private=True,
                   single_visible_T4=True, internet=False, timeout_seconds=1200,
                   mount_strategy='nested original slugs; unchanged find_input/resolve_cache_mount fallback')
    write(WORK / 'checks/tta-friend2-static.json', receipt)
    return receipt


def push(client):
    assert not LEDGER.exists(), 'Push already attempted; never retry'
    assert read(WORK / 'checks/tta-friend2-static.json')['passed']
    assert read(WORK / 'checks/tta-friend2-inputs.json')['roundtrip_sha256_passed']
    inventory = []
    for page in range(1, 101):
        rows = client.kernels_list(mine=True, page=page, page_size=100) or []
        inventory.extend(r.ref for r in rows)
        if not rows:
            break
    else:
        raise AssertionError('Owned inventory incomplete')
    assert REF not in inventory, 'Target already exists; do not create another version'
    q = quota(client)
    assert q['free_seconds'] >= 2400
    origin = original_status()
    if origin['status'] in ('RUNNING', 'COMPLETE'):
        receipt = dict(skipped=True, push_calls=0, reason='Original started or completed', original=origin)
        write(WORK / 'checks/tta-friend2-skipped.json', receipt)
        return receipt
    assert origin['status'] in ('QUEUED', 'ERROR', 'CANCEL_ACKNOWLEDGED'), 'Uncertain original status'
    rec = dict(ref=REF, push_calls=1, requested_at_utc=now(), original_before_push=origin,
               owned_inventory_count=len(inventory), target_absent=True, quota_before=q,
               notebook_sha256=sha(BUNDLE / 'nlp-hw02-tta.ipynb'), timeout_seconds=1200)
    write(LEDGER, rec)
    run = subprocess.run([CLI, 'kernels', 'push', '-p', str(BUNDLE), '--timeout', '1200',
                          '--accelerator', 'NvidiaTeslaT4'], env=environment_saved,
                         capture_output=True, text=True, timeout=150)
    match = re.search(r'Kernel version (\d+) successfully pushed', run.stdout)
    rec.update(CLI_returncode=run.returncode, push_accepted=run.returncode == 0 and bool(match),
               push_finished_at_utc=now())
    if match:
        rec['version'] = int(match[1])
    write(LEDGER, rec)
    assert rec['push_accepted'], 'Push uncertain; never retry'
    assert remote(client) == rec['version'], 'Server version mismatch; never retry'
    rec['server_version_verified'] = True
    write(LEDGER, rec)
    return rec


def observe(client):
    launch = read(LEDGER)
    assert remote(client) == launch['version']
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    request = ApiListKernelSessionOutputRequest()
    request.user_name, request.kernel_slug = REF.split('/')
    request.page_size = 200
    with client.build_kaggle_client() as c:
        output = c.kernels.kernels_api_client.list_kernel_session_output(request)
    result = dict(at_utc=now(), ref=REF, version=launch['version'],
                  status=client.kernels_status(REF).status.name,
                  output_files=len(output.files or []), output_has_more=bool(output.next_page_token),
                  code_cells_exact=True, private=True, quota=quota(client))
    p = WORK / 'checks/tta-friend2-observations.json'
    history = read(p) if p.exists() else []
    history.append(result)
    write(p, history)
    write(WORK / 'results/next-tta.json', dict(result, own_push_calls=1,
          requested_at_utc=launch['requested_at_utc'], agent_must_not_repush=True,
          bundle='kaggle/bundles/tta_friend2', backend_timeout_seconds=1200,
          watchdog_seconds=1140, quota_budget_seconds=2400,
          notebook_sha256=launch['notebook_sha256'], original_ref='thisarivson/nlp-hw02-tta',
          read_only_monitor='kaggle/tta_transfer.py observe', download='kaggle/tta_transfer.py output',
          output='results/kaggle-runs/nlp-hw02-tta/friend2',
          analyze_in_order=[
              'research/tta-analysis.py tune --raw results/kaggle-runs/nlp-hw02-tta/friend2',
              'research/tta-analysis.py confirm --raw results/kaggle-runs/nlp-hw02-tta/friend2']))
    return result


def download(client):
    launch = read(LEDGER)
    assert remote(client) == launch['version']
    assert client.kernels_status(REF).status.name in ('COMPLETE', 'ERROR')
    destination = WORK / 'results/kaggle-runs/nlp-hw02-tta/friend2'
    from kaggle_versioned import output
    build_client = client.build_kaggle_client
    output(client, REF + '/' + str(launch['version']), destination)
    client.build_kaggle_client = build_client
    receipt = read(destination / 'download-version.json')
    binding = 'explicit version label and exact current source'
    if not receipt['files']:
        assert remote(client) == launch['version']
        token = None
        while True:
            files, token = client.kernels_output(REF, str(destination), quiet=True,
                                                 page_size=200, page_token=token)
            receipt['files'] += len(files)
            if not token:
                break
        binding = 'latest fallback; current version/source checked before and after'
    assert remote(client) == launch['version']
    result = dict(at_utc=now(), ref=REF, version=launch['version'], files=receipt['files'],
                  output=str(destination.relative_to(WORK)), version_binding=binding)
    write(WORK / 'checks/tta-friend2-download.json', result)
    return result


if __name__ == '__main__':
    environment_saved = environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'push', 'observe', 'output', 'quota'])
    args = parser.parse_args()
    try:
        if args.action == 'build':
            result = build()
        else:
            client = api()
            result = {'push': push, 'observe': observe, 'output': download, 'quota': quota}[args.action](client)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        print(json.dumps(dict(error_type=type(error).__name__,
              reason=str(error) if isinstance(error, AssertionError) else 'API/CLI failed; raw response suppressed')))
        raise SystemExit(1)
