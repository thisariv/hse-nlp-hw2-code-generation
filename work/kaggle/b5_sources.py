"""Isolate the original L3 core and decoder from the later W/QK source changes."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent


def c_overrides():
    return {name:(ROOT/'legacy-c'/name).read_text() for name in ('exp26_core.py','exp27b_fast.py')}


def c_namespace(destination):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    overrides=c_overrides()
    for p in (ROOT/'src').glob('*.py'):
        target=destination/p.name
        if p.name in overrides:target.write_text(overrides[p.name])
        else:target.symlink_to(p.resolve())
    return destination
