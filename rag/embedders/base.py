"""Embedder interfaces."""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Encode text into dense vectors in a shared embedding space."""

    dimension: int | None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

