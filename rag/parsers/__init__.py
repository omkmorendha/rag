"""Parsers: one per format. raw file -> Document with clean text + provenance.

"This is where quality is won or lost" (ARCHITECTURE.md §3). For Markdown the text is
mostly clean already; we preserve heading structure because the recursive chunker splits
on it.
"""

from __future__ import annotations

from pathlib import Path

from rag.types import Document


class Parser:
    """Interface: parse one raw file into a Document."""

    def parse(self, path: Path) -> Document:
        raise NotImplementedError


class MarkdownParser(Parser):
    """Read a Markdown file verbatim, capturing source path and title in metadata.

    Heading structure is preserved (not stripped) so the recursive chunker has real
    boundaries to split on.
    """

    def parse(self, path: Path) -> Document:
        text = path.read_text(encoding="utf-8")
        title = self._first_heading(text) or path.stem
        return Document(
            id=path.stem,
            text=text,
            metadata={"source": str(path), "doc_type": "md", "title": title},
        )

    @staticmethod
    def _first_heading(text: str) -> str | None:
        for line in text.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return None
