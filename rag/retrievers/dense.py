"""Dense (semantic) retrieval over a vector index.

The dense retriever embeds the query with the **same embedder used at ingest** (the shared
vector-space invariant, ARCHITECTURE.md §2) and searches the vector index for nearest
neighbours. This is the baseline retriever; ``bm25`` and ``hybrid_rrf`` layer on later.
"""

from __future__ import annotations

from rag.embedders import Embedder
from rag.indexers import Index
from rag.types import Chunk


class DenseRetriever:
    """Embed the query, then search the dense vector index for nearest chunks."""

    name = "dense"

    def __init__(self, index: Index, embedder: Embedder) -> None:
        self.index = index
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        """Embed ``query`` and return the ``k`` nearest chunks, best-first."""
        if k <= 0:
            raise ValueError("k must be greater than 0")
        if not query.strip():
            raise ValueError("query must not be empty")

        vectors = self.embedder.embed([query])
        if not vectors:
            raise ValueError("embedder returned no vector for the query")
        return self.index.search(vectors[0], k)
