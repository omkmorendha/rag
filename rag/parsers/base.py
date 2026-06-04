"""Parser interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from rag.types import Document


class Parser(Protocol):
    """Turn one source file into parser-normalized documents."""

    def parse(self, path: Path) -> list[Document]:
        """Parse a source file into one or more documents."""

