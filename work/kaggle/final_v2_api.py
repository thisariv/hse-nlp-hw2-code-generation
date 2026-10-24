"""Private final package: default SDK auth, one push, read-only monitoring.

No credential inspection, resource deletion, cancellation or public operations.
Raw HTTP errors and signed URLs are deliberately not printed.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
SUBMISSION = WORK.parent / "submission"
STAGE = ROOT / "staging/final-package-v2"
BUNDLE = ROOT / "bundles/final_package_v2"
DATASET = "thisarivson/nlp-hw02-final-weights"
KERNEL = "thisarivson/nlp-hw02-final-package-v2-budget640"
LEDGER = WORK / "checks/final-package-v2-launch.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def connect():
    os.environ["KAGGLE_CONFIG_DIR"] = str(Path("~/.kaggle").expanduser())
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        os.environ.pop(key, None)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    return api


def dataset(api):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    req = ApiGetDatasetRequest()
    req.owner_slug, req.dataset_slug = DATASET.split("/")
    with api.build_kaggle_client() as client:
        result = client.datasets.dataset_api_client.get_dataset(req)
    assert result.ref == DATASET and result.is_private
    return result


def inspect(api):
    remote = dataset(api)
    print(json.dumps(dict(ref=remote.ref, private=remote.is_private,
        version=remote.current_version_number, status=api.dataset_status(DATASET))))


def verify_dataset(api):
    remote = dataset(api)
    assert api.dataset_status(DATASET).lower() == "ready"
    manifest = read(STAGE / "package-manifest.json")
    checks = {}
    with tempfile.TemporaryDirectory(prefix="hw2-final-roundtrip-") as tmp:
        target = Path(tmp)
        all_files = dict(manifest["files"], **manifest["code_files"])
        all_files["package-manifest.json"] = dict(bytes=(STAGE / "package-manifest.json").stat().st_size,
                                                sha256=sha(STAGE / "package-manifest.json"))
        for number, (name, spec) in enumerate(all_files.items()):
            folder = target / str(number)
            folder.mkdir()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                api.dataset_download_file(DATASET + "/" + str(remote.current_version_number), name,
                                          path=str(folder), quiet=True)
            paths = list(folder.iterdir())
            assert len(paths) == 1, "Expected one downloaded payload"
            path = paths[0]
            assert path.stat().st_size == spec["bytes"] and sha(path) == spec["sha256"], name
            checks[name] = spec
    receipt = dict(ref=DATASET, private=remote.is_private, version=remote.current_version_number,
                   verified_at_utc=now(), full_download_sha_roundtrip=True, files=checks,
                   manifest_sha256=sha(STAGE / "package-manifest.json"))
    write(WORK / "checks/final-package-v2-dataset.json", receipt)
    print(json.dumps(dict(ref=DATASET, private=True, version=receipt["version"], sha_files=len(checks))))


def version_dataset(api):
    """Publish the authorized budget640 package; retain every old version."""
    marker = WORK / "checks/final-package-v2-dataset-version.json"
    assert not marker.exists(), "Version already attempted; inspect server"
    owned = []
    for page in range(1, 101):
        rows = api.dataset_list(mine=True, page=page) or []
        owned.extend(x.ref for x in rows)
        if not rows:
            break
    else:
        raise AssertionError("Incomplete owned dataset inventory")
    assert DATASET in owned, "Default account must own thisarivson final weights"
    old = dataset(api)
    assert old.current_version_number == 3
    receipt = dict(ref=DATASET, requested_at_utc=now(), private=True, version_calls=1,
                   manifest_sha256=sha(STAGE / "package-manifest.json"), delete_old_versions=False)
    write(marker, receipt)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = api.dataset_create_version(str(STAGE),
            version_notes="Final v2: W B7 greedy anchor / C-long2x B7 / P1-B, budget640; omit unused B6 from this version; retain all old versions", quiet=True,
            convert_to_csv=False, delete_old_versions=False, dir_mode="zip")
    assert not result.error, "Dataset version response failed or uncertain; no blind retry"
    receipt.update(accepted=True, finished_at_utc=now())
    write(marker, receipt)
    print(json.dumps(receipt))


def metadata(api, version=None):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    req = ApiGetKernelRequest()
    req.user_name, req.kernel_slug = KERNEL.split("/")
    if version:
        req.version_label = "v" + str(version)
    with api.build_kaggle_client() as client:
        return client.kernels.kernels_api_client.get_kernel(req)


def codes(nb):
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def parity(remote):
    m = remote.metadata
    assert m.is_private and m.enable_gpu and not m.enable_internet and not m.enable_tpu
    assert m.machine_shape == "NvidiaTeslaT4" and set(m.dataset_data_sources) == {DATASET}
    assert not m.kernel_data_sources
    assert codes(json.loads(remote.blob.source)) == codes(read(BUNDLE / "inference.ipynb"))
    return dict(private=True, all_code_cells_exact=True, machine_shape=m.machine_shape,
                current_version=m.current_version_number, dataset_sources=m.dataset_data_sources)


def push(api):
    assert not LEDGER.exists(), "Exactly one push: ledger already exists"
    proof = read(WORK / "checks/final-package-v2-dataset.json")
    assert proof["full_download_sha_roundtrip"] and proof["private"]
    assert proof["manifest_sha256"] == sha(STAGE / "package-manifest.json")
    cpu = read(WORK / "checks/final-package-v2-cpu.json")
    assert cpu["passed"] and cpu["notebook_sha256"] == sha(SUBMISSION / "inference.ipynb")
    assert cpu["manifest_sha256"] == proof["manifest_sha256"]
    assert dataset(api).current_version_number == proof["version"]
    owned = []
    for page in range(1, 101):
        rows = api.kernels_list(mine=True, page=page, page_size=100) or []
        owned.extend(x.ref for x in rows)
        if not rows:
            break
    else:
        raise AssertionError("Incomplete owned kernel inventory")
    assert KERNEL not in owned, "Target exists; inspect without pushing"
    with api.build_kaggle_client() as client:
        q = client.kernels.kernels_api_client.get_accelerator_quota_statistics().gpu_quota
    free = (q.total_time_allowed - q.time_used - q.time_reserved).total_seconds()
    assert free >= 1800
    BUNDLE.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SUBMISSION / "inference.ipynb", BUNDLE / "inference.ipynb")
    write(BUNDLE / "kernel-metadata.json", dict(id=KERNEL, title="NLP HW02 final package v2 budget640",
          code_file="inference.ipynb", language="python", kernel_type="notebook", is_private=True,
          enable_gpu=True, enable_tpu=False, enable_internet=False, machine_shape="NvidiaTeslaT4",
          dataset_sources=[DATASET + "/" + str(proof["version"])], kernel_sources=[], competition_sources=[]))
    receipt = dict(ref=KERNEL, push_calls=1, requested_at_utc=now(), private=True,
                   notebook_sha256=sha(BUNDLE / "inference.ipynb"), free_quota_seconds=free,
                   timeout_seconds=720, dataset_version=proof["version"])
    write(LEDGER, receipt)  # Uncertain responses must never trigger another push.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        result = api.kernels_push(str(BUNDLE), timeout="720", acc="NvidiaTeslaT4")
    assert not result.error, "Push failed or uncertain; no retry"
    receipt.update(version=result.version_number, accepted=True, finished_at_utc=now())
    write(LEDGER, receipt)
    proof = parity(metadata(api, result.version_number))
    write(WORK / "checks/final-package-v2-remote.json", dict(ref=KERNEL, **proof, checked_at_utc=now()))
    print(json.dumps(receipt))


def observe(api):
    launch = read(LEDGER)
    remote = metadata(api, launch["version"])
    proof = parity(remote)
    assert metadata(api).metadata.current_version_number == launch["version"], "External version changed"
    status = api.kernels_status(KERNEL).status.name
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    req = ApiListKernelSessionOutputRequest()
    req.user_name, req.kernel_slug = KERNEL.split("/")
    req.page_size = 1
    with api.build_kaggle_client() as client:
        out = client.kernels.kernels_api_client.list_kernel_session_output(req)
    path = WORK / "checks/final-package-v2-observations.json"
    history = read(path) if path.exists() else []
    first = next((x["at_utc"] for x in history if x["status"] == status), now())
    seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(first)).total_seconds()
    record = dict(at_utc=now(), status=status, version=launch["version"], own_push_calls=1,
                  output_visible=bool(out.files), log_chars=len(out.log or ""),
                  status_observed_seconds=seconds, **proof)
    record["wait_threshold_crossed"] = ((status == "QUEUED" and seconds > 600) or
        (status == "RUNNING" and seconds > 720 and not out.files))
    history.append(record)
    write(path, history)
    print(json.dumps(record))


def output(api):
    launch = read(LEDGER)
    assert metadata(api).metadata.current_version_number == launch["version"]
    parity(metadata(api, launch["version"]))
    target = WORK / "results/kaggle-runs/nlp-hw02-final-package-v2" / str(launch["version"])
    target.mkdir(parents=True, exist_ok=True)
    token, count = None, 0
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        while True:
            files, token = api.kernels_output(KERNEL, str(target), quiet=True, page_size=200, page_token=token)
            count += len(files)
            if not token:
                break
        api.kernels_pull(KERNEL, path=str(target / "server"), metadata=True, quiet=True)
    # The executed notebook is a special output, absent from the ordinary file list.
    from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest
    request = ApiDownloadKernelOutputRequest()
    request.owner_slug, request.kernel_slug = KERNEL.split("/")
    request.file_path = "__notebook__.ipynb"
    request.version_number = launch["version"]
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.download_kernel_output(request)
    response.raise_for_status()
    notebook = json.loads(response.content)
    assert codes(notebook) == codes(read(BUNDLE / "inference.ipynb"))
    assert all(c.get("execution_count") is not None for c in notebook["cells"] if c["cell_type"] == "code")
    (target / "__notebook__.ipynb").write_bytes(response.content)
    assert metadata(api).metadata.current_version_number == launch["version"]
    parity(metadata(api, launch["version"]))
    receipt = dict(ref=KERNEL, version=launch["version"], files=count, downloaded_at_utc=now(),
                   binding="latest output; exact server version and code verified before and after",
                   executed_notebook_sha256=sha(target / "__notebook__.ipynb"),
                   output=str(target.relative_to(WORK)))
    write(WORK / "checks/final-package-v2-download.json", receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "version", "verify", "push", "observe", "output"])
    args = parser.parse_args()
    try:
        {"inspect": inspect, "version": version_dataset, "verify": verify_dataset, "push": push,
         "observe": observe, "output": output}[args.action](connect())
    except Exception as error:
        print(json.dumps(dict(error_type=type(error).__name__,
              http=getattr(getattr(error, "response", None), "status_code", None),
              reason=str(error) if isinstance(error, AssertionError) else "Operation failed; raw response suppressed")))
        raise SystemExit(1)
