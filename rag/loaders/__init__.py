"""File loaders: walk `data/`, dispatch by extension, return raw handles (not parsed)."""

from __future__ import annotations

from pathlib import Path


class Loader:
    """Interface: list source files under a corpus directory."""

    def load_files(self, data_dir: Path) -> list[Path]:
        raise NotImplementedError


class MarkdownLoader(Loader):
    """Walk `data_dir` recursively and return every Markdown file."""

    def load_files(self, data_dir: Path) -> list[Path]:
        return sorted(data_dir.rglob("*.md"))
