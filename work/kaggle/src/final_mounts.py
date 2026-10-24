"""Resolve dataset and kernel-output mounts without hardcoded owner layouts."""
from pathlib import Path


def find_input(slug, root=Path('/kaggle/input')):
    root = Path(root)
    if (root/slug).is_dir():
        return root/slug
    hits = [p for p in root.rglob(slug) if p.is_dir()] if root.is_dir() else []
    assert len(hits) == 1, f'Cannot resolve input {slug}: {[str(p) for p in hits]}'
    return hits[0]


def resolve_cache_mount(input_root, source, kind):
    # Same precedence as l3_resume.resolve_cache_mount; no training imports.
    root = Path(input_root)
    owner, slug = source.split('/')[:2]
    assert kind in ('datasets', 'notebooks')
    for candidate in (root/kind/owner/slug, root/slug):
        if candidate.is_dir():
            return candidate
    return find_input(slug, root)
