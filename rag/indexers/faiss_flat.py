"""Exact (brute-force) FAISS index over normalized embeddings.

``faiss_flat`` is the baseline indexer from ARCHITECTURE.md §3: exact nearest-neighbour
search, no approximation. The embedder normalizes vectors (``normalize_embeddings: true``),
so cosine similarity equals inner product — we use ``IndexFlatIP`` and defensively
re-normalize on build so a misconfigured embedder cannot silently corrupt scores.

Persistence layout under ``vectorstore/``::

    index.faiss    the raw FAISS index (vectors only)
    chunks.jsonl   one chunk per line, aligned by row to the index
    meta.json      dim, count, metric, indexer/embedder identity (the §2 invariant)

Chunks are stored as transparent JSONL rather than a pickle so the store is diffable and
loading never executes arbitrary code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from rag.types import Chunk

INDEX_FILENAME = "index.faiss"
CHUNKS_FILENAME = "chunks.jsonl"
META_FILENAME = "meta.json"


class IndexMismatchError(RuntimeError):
    """Raised when a loaded index is incompatible with the active config.

    Embedding the corpus with model A and the query with model B silently returns garbage
    because a bi-encoder only works inside one shared vector space (ARCHITECTURE.md §2).
    This error makes that invariant enforceable at load time.
    """


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return vector
    return [component / norm for component in vector]


def _chunk_to_record(chunk: Chunk) -> dict[str, Any]:
    return {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata}


def _chunk_from_record(record: dict[str, Any], score: float | None = None) -> Chunk:
    return Chunk(
        id=record["id"],
        text=record["text"],
        metadata=record.get("metadata", {}),
        score=score,
    )


class FaissFlatIndex:
    """A built, searchable flat FAISS index plus its chunks and provenance."""

    def __init__(self, index: Any, chunks: list[Chunk], meta: dict[str, Any]) -> None:
        self._index = index
        self._chunks = chunks
        self.meta = meta

    def search(self, query_vector: list[float], k: int) -> list[Chunk]:
        """Return the ``k`` nearest chunks to ``query_vector``, ``.score`` set to similarity."""
        if k <= 0:
            raise ValueError("k must be greater than 0")
        if not self._chunks:
            return []

        import numpy as np

        expected = self.meta["dim"]
        if len(query_vector) != expected:
            raise ValueError(
                f"query vector has dimension {len(query_vector)}, index expects {expected}"
            )

        query = np.asarray([_normalize(query_vector)], dtype="float32")
        top_k = min(k, len(self._chunks))
        scores, indices = self._index.search(query, top_k)

        results: list[Chunk] = []
        for score, row in zip(scores[0], indices[0], strict=True):
            if row < 0:  # FAISS pads with -1 when fewer than k neighbours exist
                continue
            results.append(
                _chunk_from_record(_chunk_to_record(self._chunks[row]), float(score))
            )
        return results

    def save(self, directory: Path) -> None:
        """Persist the FAISS index, chunk sidecar, and provenance metadata."""
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / INDEX_FILENAME))

        with (directory / CHUNKS_FILENAME).open("w", encoding="utf-8") as file:
            for chunk in self._chunks:
                file.write(json.dumps(_chunk_to_record(chunk), ensure_ascii=False))
                file.write("\n")

        with (directory / META_FILENAME).open("w", encoding="utf-8") as file:
            json.dump(self.meta, file, indent=2, sort_keys=True)
            file.write("\n")


class FaissFlatIndexer:
    """Build and load an exact inner-product FAISS index."""

    name = "faiss_flat"
    metric = "inner_product"

    def __init__(self, *, embedder_model: str | None = None) -> None:
        # Recorded into meta.json so load() can enforce the embedder-match invariant.
        self.embedder_model = embedder_model

    def build(self, chunks: list[Chunk]) -> FaissFlatIndex:
        """Build an index over chunks that already carry ``.embedding``."""
        import faiss
        import numpy as np

        if not chunks:
            raise ValueError("cannot build an index from zero chunks")

        embeddings: list[list[float]] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"chunk {chunk.id!r} has no embedding; embed chunks before indexing"
                )
            embeddings.append(_normalize(list(chunk.embedding)))

        dim = len(embeddings[0])
        if any(len(vector) != dim for vector in embeddings):
            raise ValueError("all chunk embeddings must share one dimension")

        matrix = np.asarray(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

        meta = {
            "kind": self.name,
            "metric": self.metric,
            "dim": dim,
            "n_chunks": len(chunks),
            "embedder_model": self.embedder_model,
        }
        # Drop embeddings from the retained chunks; vectors live in the FAISS index.
        stored = [Chunk(id=c.id, text=c.text, metadata=c.metadata) for c in chunks]
        return FaissFlatIndex(index, stored, meta)

    def load(self, directory: Path) -> FaissFlatIndex:
        """Load a persisted index, enforcing the embedder-match invariant (§2)."""
        import faiss

        meta_path = directory / META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(f"no index metadata at {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if meta.get("kind") != self.name:
            raise IndexMismatchError(
                f"index kind {meta.get('kind')!r} does not match indexer {self.name!r}"
            )
        if (
            self.embedder_model is not None
            and meta.get("embedder_model") is not None
            and meta["embedder_model"] != self.embedder_model
        ):
            raise IndexMismatchError(
                f"index was built with embedder {meta['embedder_model']!r} but config "
                f"uses {self.embedder_model!r}; re-run ingest or restore the matching config"
            )

        index = faiss.read_index(str(directory / INDEX_FILENAME))
        if index.d != meta["dim"]:
            raise IndexMismatchError(
                f"index dimension {index.d} disagrees with metadata dim {meta['dim']}"
            )

        chunks: list[Chunk] = []
        with (directory / CHUNKS_FILENAME).open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    chunks.append(_chunk_from_record(json.loads(line)))

        if len(chunks) != index.ntotal:
            raise IndexMismatchError(
                f"chunk count {len(chunks)} disagrees with index size {index.ntotal}"
            )
        return FaissFlatIndex(index, chunks, meta)
