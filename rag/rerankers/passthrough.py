"""No-op reranker for the vertical slice.

Returns the retriever's candidates unchanged except for an optional ``top_n`` truncation,
so the pipeline runs end-to-end before a real cross-encoder is wired in (ARCHITECTURE.md
§3, build order §6.1).
"""

from __future__ import annotations

from rag.types import Chunk


class NoopReranker:
    """Pass candidates through, preserving retriever order; optionally cap at ``top_n``."""

    name = "noop"

    def __init__(self, *, top_n: int | None = None) -> None:
        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be greater than 0")
        self.top_n = top_n

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Return chunks unchanged, truncated to ``top_n`` when configured."""
        if self.top_n is None:
            return list(chunks)
        return list(chunks[: self.top_n])
