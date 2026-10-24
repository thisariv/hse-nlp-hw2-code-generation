"""Build separate reproducible Run All notebook. No API calls or GPU work."""
import json
import shutil
import base64
import zlib
from variants_common import *


def build():
    BUNDLE.mkdir(parents=True,exist_ok=True);STAGE.mkdir(parents=True,exist_ok=True)
    source=ROOT/'staging/final-package-v11-budget640'
    inputs={name:(source/name).read_text() for name in ['test.json','tune2048.json','tune2048-reference.json','tokenizer.json']}
    assert len(json.loads(inputs['test.json']))==1919
    assert len(json.loads(inputs['tune2048.json']))==2048
    assert sha(source/'test.json')=='20cacb334e3fc81291433d2173d9085b17a15e1834bd84aea67b350403182503'
    cfgs=variants()
    code={f.name:f.read_text() for f in sorted((SUBMISSION/'code/hw2_codegen').iterdir()) if f.suffix in ('.py','.json')}
    bootstrap='''import os, sys, json, hashlib
from pathlib import Path
DRY = os.environ.get("HW2_VARIANTS_DRY_RUN") == "1"
ROOT = Path(os.environ.get("HW2_VARIANTS_ROOT", "/kaggle/working"))
ROOT.mkdir(parents=True, exist_ok=True)
CODE = ROOT / "variants-src" / "hw2_codegen"
CODE.mkdir(parents=True, exist_ok=True)
INPUTS = ROOT / "variants-inputs"
INPUTS.mkdir(parents=True, exist_ok=True)
'''
    packed = base64.b64encode(zlib.compress(json.dumps(inputs,ensure_ascii=False).encode(),9)).decode()
    materialize='import base64, zlib\nCODE_FILES = '+repr(code)+'\nINPUT_FILES = json.loads(zlib.decompress(base64.b64decode('+repr(packed)+')))\nCONFIGS = '+repr(cfgs)+'\n'
    materialize+='''for name, content in CODE_FILES.items():
    (CODE / name).write_text(content)
for name, content in INPUT_FILES.items():
    (INPUTS / name).write_text(content)
sys.path.insert(0, str(CODE.parent))
from hw2_codegen.runner import run_cold

def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()

# Mount resolution occurs before the individual timers. Weight SHA/load,
# tokenizer, imports, warmup, inference, execution/vote and write are timed
# separately from a fresh process for EACH variant/split.
if not DRY:
    weights = Path("/tmp/hw2-variants-weights")
    weights.mkdir(parents=True, exist_ok=True)
    for spec in {m["name"]:m for c in CONFIGS for m in c["models"]}.values():
        slug = spec["source_ref"].split("/")[1]
        mounts = [p for p in Path("/kaggle/input").rglob(slug) if p.is_dir()]
        assert len(mounts) == 1, (slug, len(mounts))
        target = mounts[0] / spec["source_path"]
        assert target.is_file(), str(target)
        link = weights / spec["filename"]
        if not link.exists():
            link.symlink_to(target)
        assert link.resolve() == target.resolve()
else:
    (INPUTS / "dry-reference.json").write_text(json.dumps([dict(status="numeric",value=1)]*8))
print(json.dumps(dict(status="ready", variants=len(CONFIGS), dry_run=DRY)), flush=True)
'''
    run='''SUMMARY = dict(training=False, cold_process_per_split=True, single_visible_gpu=True, variants={})
for template in CONFIGS:
    name = template["variant"]
    reports = {}
    for split in ("test", "tune"):
        cfg = json.loads(json.dumps(template))
        out = ROOT / "variants" / name
        if split == "tune":
            out = out / "tune"
        cfg.update(weights_root="/tmp/hw2-variants-weights",
                   tokenizer=str(INPUTS / "tokenizer.json"),
                   rows=str(INPUTS / ("test.json" if split == "test" else "tune2048.json")),
                   output=str(out))
        if split == "tune":
            cfg["reference"] = str(INPUTS / ("dry-reference.json" if DRY else "tune2048-reference.json"))
        if DRY:
            cfg.update(device="cpu", dry_run=True)
        report = run_cold(cfg)
        reports[split] = report
        if split == "tune":
            quality = out / "quality.json"
            reports["quality"] = json.loads(quality.read_text()) if quality.exists() else None
        print(json.dumps(dict(variant=name, split=split, seconds=report["elapsed_seconds"],
            guards=report["guards"], peak_reserved_bytes=report.get("peak_reserved_bytes"),
            quality=reports.get("quality"))), flush=True)
    SUMMARY["variants"][name] = reports
    (ROOT / "variants/summary.json").write_text(json.dumps(SUMMARY,indent=2)+"\\n")
print(json.dumps(dict(status="finished", variants=list(SUMMARY["variants"]))), flush=True)
'''
    cells=[dict(cell_type='markdown',metadata={},source='# Пять вариантов ансамбля ДЗ2 NLP\n\nОдин private Run All: каждый вариант — cold test1919, затем cold tune2048 на одной видимой T4. Budget640, generation guard160с; timer включает весь pipeline. V4 — primary E2. Никакого обучения или отправки в бота.'),
           *[dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s) for s in [bootstrap,materialize,run]]]
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.12')),cells=cells)
    for i,c in enumerate(cells): c['id']='variants-'+str(i)
    write(BUNDLE/'variants.ipynb',nb)
    assert (BUNDLE/'variants.ipynb').stat().st_size < 900_000, 'Kaggle source payload must stay under 1MB'
    notebook=ROOT/'notebooks/nlp-hw02-variants.ipynb';shutil.copyfile(BUNDLE/'variants.ipynb',notebook)
    write(BUNDLE/'kernel-metadata.json',dict(id=KERNEL,title='NLP HW02 variants',code_file='variants.ipynb',
        language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_tpu=False,
        enable_internet=False,machine_shape='NvidiaTeslaT4',dataset_sources=[],kernel_sources=SOURCES,competition_sources=[]))
    write(WORK/'checks/final-variants-build.json',dict(passed=True,notebook_sha256=sha(BUNDLE/'variants.ipynb'),
        models=models(),variants=cfgs,code_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in code.items()},
        input_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in inputs.items()},
        submission_notebook_sha256=sha(SUBMISSION/'inference.ipynb')))
    print(json.dumps(dict(notebook=str(notebook.relative_to(WORK)),bytes=notebook.stat().st_size)))

if __name__=='__main__':build()
