"""Chunker interfaces."""

from __future__ import annotations

from typing import Protocol

from rag.types import Chunk, Document


class Chunker(Protocol):
    """Split parser-normalized documents into retrieval units."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Chunk one document into retriever-ready text units."""
