"""Retriever interfaces.

A retriever turns a query *string* into candidate chunks (ARCHITECTURE.md §3). It is the
query-side counterpart of the indexer: where the indexer persists vectors offline, the
retriever loads that index online and searches it per question.

Retrievers over-retrieve (k≈50) on purpose — the reranker compresses the candidate set
down to a precise few afterwards.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag.types import Chunk


@runtime_checkable
class Retriever(Protocol):
    """Retrieve candidate chunks for a query string."""

    name: str

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        """Return up to ``k`` candidate chunks, ordered best-first, ``.score`` set."""
        ...
