"""Indexer interfaces.

The index is the boundary between the two pipelines (see ARCHITECTURE.md §0): ingest
builds and persists it; query loads it and searches. To keep stages swappable, the index
owns the chunks it was built from, so ``search`` returns whole ``Chunk`` objects (text +
metadata, per §1) rather than bare vector IDs.

Indexers are embedding-agnostic: ``build`` receives chunks that already carry
``.embedding`` and ``search`` receives a query vector. Embedding stays the embedder's job
so it can remain the single source of truth shared by both pipelines (§2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from rag.types import Chunk


@runtime_checkable
class Index(Protocol):
    """A persisted, searchable collection of embedded chunks."""

    def search(self, query_vector: list[float], k: int) -> list[Chunk]:
        """Return the ``k`` nearest chunks to ``query_vector`` with ``.score`` set."""
        ...

    def save(self, directory: Path) -> None:
        """Persist the index and its chunks under ``directory``."""
        ...


@runtime_checkable
class Indexer(Protocol):
    """Build a searchable index from embedded chunks."""

    name: str

    def build(self, chunks: list[Chunk]) -> Index:
        """Build an index over chunks that already carry ``.embedding``."""
        ...

    def load(self, directory: Path) -> Index:
        """Load a previously persisted index from ``directory``."""
        ...
