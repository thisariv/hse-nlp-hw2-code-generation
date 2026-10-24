"""Единственный файл политики — config.json; пути задаются вызывающим notebook."""

import json
from pathlib import Path


def default_config():
    """Вернуть независимую копию замороженной политики v2 с budget 640."""
    return json.loads(Path(__file__).with_name("config.json").read_text())
