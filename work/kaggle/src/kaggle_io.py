"""Filesystem utilities shared by local preparation and commit notebooks; no credentials."""
import hashlib, json, os, shutil, tarfile, zipfile
from pathlib import Path, PurePosixPath

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def safe_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name: raise ValueError('Unsafe archive path')
    if any(x in ('kaggle.json','.aws','.ssh') or x.startswith('.env') for x in p.parts):
        raise ValueError('Credentials are not dataset inputs')
    return p

def place(source, target):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(source)!=sha(target): raise ValueError('Conflicting split download: '+target.name)
        return
    shutil.copyfile(source,target)

def extract(archive, target):
    """CRC/SHA checks happen again at application boundaries. No links/traversal/overwrite."""
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    # TAR may contain a ZIP whose end record makes is_zipfile(TAR) return True.
    # Select the actual outer container by its explicit filename.
    if Path(archive).suffix.lower()=='.zip':
        with zipfile.ZipFile(archive) as z:
            if z.testzip(): raise ValueError('ZIP CRC failure')
            for info in z.infolist():
                p=safe_name(info.filename)
                if '__MACOSX' in p.parts or p.name=='.DS_Store': continue
                if (info.external_attr>>16)&0o170000==0o120000: raise ValueError('ZIP symlink')
                if info.is_dir(): continue
                dst=target/p;dst.parent.mkdir(parents=True,exist_ok=True)
                tmp=dst.with_name(dst.name+'.extract-tmp')
                with z.open(info) as src,tmp.open('wb') as out: shutil.copyfileobj(src,out)
                if dst.exists():
                    if sha(tmp)!=sha(dst): raise ValueError('Conflicting ZIP member '+str(p))
                    tmp.unlink()
                else: tmp.replace(dst)
    else:
        with tarfile.open(archive) as t:
            for info in t:
                p=safe_name(info.name)
                if info.isdir(): continue
                if not info.isfile(): raise ValueError('TAR links/devices forbidden')
                dst=target/p;dst.parent.mkdir(parents=True,exist_ok=True)
                tmp=dst.with_name(dst.name+'.extract-tmp')
                with t.extractfile(info) as src,tmp.open('wb') as out: shutil.copyfileobj(src,out)
                if dst.exists():
                    if sha(tmp)!=sha(dst): raise ValueError('Conflicting TAR member')
                    tmp.unlink()
                else: tmp.replace(dst)

def find_one(root, marker):
    hits=list(Path(root).rglob(marker))
    if len(hits)!=1: raise ValueError(f'Expected one {marker}, found {len(hits)} under {root}')
    return hits[0]

def resolve_dataset(input_root, configured):
    input_root=Path(input_root);path=input_root/Path(configured)
    if path.is_dir():return path
    # Support both /input/slug and the current /input/datasets/owner/slug layout,
    # plus notebook-source outputs under /input/notebooks/owner/slug.
    slug=Path(configured).name
    hits=[p for p in input_root.rglob(slug) if p.is_dir()]
    if len(hits)!=1:
        visible=[str(p.relative_to(input_root)) for p in input_root.rglob('*') if p.is_dir() and len(p.relative_to(input_root).parts)<=4]
        raise ValueError(f'Cannot resolve configured dataset {configured}; folders={visible[:60]}')
    return hits[0]

def materialize_dataset(dataset, working, marker):
    """Kaggle normally expands uploads; also accept a still-packed payload archive."""
    hits=list(Path(dataset).rglob(marker))
    if hits: return find_one(dataset,marker).parent
    out=Path(working)
    for p in sorted(Path(dataset).rglob('*')):
        if p.is_file() and (p.suffix in ('.zip','.tar') or p.name.endswith('.tar.gz')):
            extract(p,out)
    return find_one(out,marker).parent

def disk_guard(root, limit=18*1024**3):
    used=sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
    if used>limit: raise RuntimeError(f'Working output exceeds reserve: {used} bytes')
    if shutil.disk_usage(root).free<512*1024**2: raise RuntimeError('Less than 512 MiB free')
    return used
