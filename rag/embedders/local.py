"""Local embedding strategies."""

from __future__ import annotations

import hashlib
import math
from typing import Any


class SentenceTransformerEmbedder:
    """Encode text with a local sentence-transformers model."""

    name = "local"

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.dimension: int | None = None
        self._model: Any | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using the configured local model."""
        if not texts:
            return []

        model = self._load_model()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        embedded = vectors.tolist()
        if embedded:
            self.dimension = len(embedded[0])
        return embedded

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model


class HashingEmbedder:
    """Deterministic lightweight encoder for tests and smoke runs."""

    name = "hashing"

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed text with a stable hash projection over whitespace tokens."""
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in text.split():
            digest = hashlib.blake2b(token.lower().encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

