"""Retrievers: query -> list[Chunk]. Over-retrieve here (k~50); the reranker compresses.

`dense` is semantic retrieval via the vector index. The retriever embeds the query with
the SAME embedder used at ingest (ARCHITECTURE.md §3) — made explicit by passing the
embedder in, not re-instantiating one.
"""

from __future__ import annotations

from rag.embedders import Embedder
from rag.indexers import Index
from rag.types import Chunk


class Retriever:
    """Interface: retrieve the top-k Chunks for a query string."""

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        raise NotImplementedError


class DenseRetriever(Retriever):
    """Embed the query, search the dense vector index, return scored Chunks."""

    def __init__(self, index: Index, embedder: Embedder, k: int = 50):
        self.index = index
        self.embedder = embedder
        self.k = k

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        query_vec = self.embedder.embed([query])[0]
        return self.index.search(query_vec, k or self.k)
