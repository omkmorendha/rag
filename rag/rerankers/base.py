"""Reranker interfaces.

A reranker re-scores the ``(query, chunk.text)`` pairs already in hand and never touches
the index (ARCHITECTURE.md §3). It compresses a large, mediocre candidate set down to a
small, high-quality one: re-score, sort, truncate to ``top_n``. The retriever's ``.score``
is **overwritten** by the reranker (see the ``Chunk`` contract in §1).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.types import Chunk


@runtime_checkable
class Reranker(Protocol):
    """Re-score and compress retrieved candidates for a query."""

    name: str

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Return the best chunks for ``query``, re-scored and ordered best-first."""
        ...
