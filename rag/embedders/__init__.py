"""Embedders: texts -> vectors. A bi-encoder, used in BOTH pipelines (ARCHITECTURE.md §3).

Encodes query and documents independently (what makes search scale, and what makes it
lossy). The SAME embedder must run at ingest and query, or the two land in different
vector spaces and retrieval returns garbage — config.yaml enforces this.
"""

from __future__ import annotations

import numpy as np


class Embedder:
    """Interface: embed a list of texts into a 2-D float32 array (n_texts, dim)."""

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


class LocalEmbedder(Embedder):
    """sentence-transformers bi-encoder, run locally on CPU.

    Vectors are L2-normalized so a FAISS inner-product index computes cosine similarity.
    The model is loaded lazily so importing the registry doesn't pull in torch.
    """

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vecs = model.encode(
            texts,
            normalize_embeddings=True,  # cosine via inner product
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        )
        return np.asarray(vecs, dtype=np.float32)

    @property
    def dim(self) -> int:
        return self.embed(["probe"]).shape[1]
