"""Indexers: build a searchable index over chunks, persist it, reload it.

The persisted index IS the boundary between the ingest and query pipelines
(ARCHITECTURE.md §0), so it stores the chunks themselves (text + metadata) next to the
vectors — a bare vector id can't survive into the prompt.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from rag.types import Chunk


class Index:
    """A built, searchable index. `search` returns scored Chunks for a query vector."""

    def search(self, query_vec: np.ndarray, k: int) -> list[Chunk]:
        raise NotImplementedError

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, directory: Path) -> "Index":
        raise NotImplementedError


class Indexer:
    """Interface: build an Index from embedded chunks."""

    def build(self, chunks: list[Chunk]) -> Index:
        raise NotImplementedError

    def load(self, directory: Path) -> Index:
        raise NotImplementedError


class FlatIndex(Index):
    """Brute-force exact inner-product index (faiss IndexFlatIP). Baseline: no ANN.

    Vectors are L2-normalized upstream, so inner product == cosine similarity. The chunks
    are held in parallel with the faiss index; search maps row ids back to Chunks and
    stamps the similarity onto `chunk.score`.
    """

    _VECTORS = "index.faiss"
    _CHUNKS = "chunks.pkl"
    _META = "meta.json"

    def __init__(self, faiss_index: "faiss.Index", chunks: list[Chunk]):
        self._index = faiss_index
        self._chunks = chunks

    def search(self, query_vec: np.ndarray, k: int) -> list[Chunk]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        k = min(k, len(self._chunks))
        scores, ids = self._index.search(q, k)
        results: list[Chunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:  # faiss pads with -1 when fewer than k results exist
                continue
            chunk = self._chunks[idx]
            # Fresh copy so the persisted chunk's score isn't mutated across queries.
            results.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=float(score),
                )
            )
        return results

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / self._VECTORS))
        with open(directory / self._CHUNKS, "wb") as f:
            pickle.dump(self._chunks, f)
        with open(directory / self._META, "w", encoding="utf-8") as f:
            json.dump(
                {"n_chunks": len(self._chunks), "dim": self._index.d, "kind": "faiss_flat"},
                f,
                indent=2,
            )

    @classmethod
    def load(cls, directory: Path) -> "FlatIndex":
        faiss_index = faiss.read_index(str(directory / cls._VECTORS))
        with open(directory / cls._CHUNKS, "rb") as f:
            chunks = pickle.load(f)
        return cls(faiss_index, chunks)


class FaissFlatIndexer(Indexer):
    """Build a FlatIndex from chunks that already carry embeddings."""

    def build(self, chunks: list[Chunk]) -> FlatIndex:
        if not chunks:
            raise ValueError("cannot build an index from zero chunks")
        missing = [c.id for c in chunks if c.embedding is None]
        if missing:
            raise ValueError(f"{len(missing)} chunks have no embedding (e.g. {missing[0]})")
        matrix = np.asarray([c.embedding for c in chunks], dtype=np.float32)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        # Drop the per-chunk vectors before persisting; the faiss index holds them now.
        stored = [Chunk(id=c.id, text=c.text, metadata=c.metadata) for c in chunks]
        return FlatIndex(index, stored)

    def load(self, directory: Path) -> FlatIndex:
        return FlatIndex.load(directory)
