"""Filesystem discovery for source documents."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def iter_files(root: Path, extensions: Iterable[str] = (".md",)) -> Iterable[Path]:
    """Yield source files below root with one of the selected extensions."""
    allowed = {extension.lower() for extension in extensions}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path

