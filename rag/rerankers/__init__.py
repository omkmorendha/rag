"""Rerankers: rerank(query, chunks) -> chunks. Re-scores text in hand; never touches the index.

`noop` is the first-slice passthrough. The real `cross_encoder` (ARCHITECTURE.md §3) is
layered in later — it compresses ~50 mediocre chunks into ~8 high-quality ones, drops
duplicates, and reorders for lost-in-the-middle.
"""

from __future__ import annotations

from rag.types import Chunk


class Reranker:
    """Interface: re-order/score retrieved chunks against the query."""

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        raise NotImplementedError


class NoopReranker(Reranker):
    """Passthrough: return retrieved chunks unchanged (keeps the retriever's order/score)."""

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        return chunks
