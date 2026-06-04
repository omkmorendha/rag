"""Cross-encoder reranker.

Unlike the bi-encoder embedder (which encodes query and document independently, making
search scalable but lossy), a cross-encoder scores each ``(query, chunk.text)`` pair
*jointly*. It is far more accurate but cannot scale to the whole corpus — so it runs only
over the handful of candidates the retriever already surfaced, then keeps the top ``top_n``
(ARCHITECTURE.md §3, build order §4: usually the biggest quality jump per line of code).
"""

from __future__ import annotations

from typing import Any

from rag.types import Chunk


class CrossEncoderReranker:
    """Re-score candidates with a local cross-encoder and keep the top ``top_n``."""

    name = "cross_encoder"

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-base",
        *,
        top_n: int = 8,
        batch_size: int = 32,
    ) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be greater than 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        self.model_name = model
        self.top_n = top_n
        self.batch_size = batch_size
        self._model: Any | None = None

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Score each (query, chunk) pair, overwrite ``.score``, return the top ``top_n``."""
        if not chunks:
            return []

        model = self._load_model()
        pairs = [[query, chunk.text] for chunk in chunks]
        scores = model.predict(pairs, batch_size=self.batch_size)

        scored = sorted(
            zip(chunks, scores, strict=True),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        return [
            Chunk(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                embedding=chunk.embedding,
                score=float(score),
            )
            for chunk, score in scored[: self.top_n]
        ]

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model
