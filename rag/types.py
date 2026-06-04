"""Core data types shared by pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    """Parser-normalized document text plus traceability metadata."""

    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    """Retriever-ready text unit with append-only traceability metadata."""

    id: str
    text: str
    metadata: dict[str, Any]
    embedding: list[float] | None = None
    score: float | None = None
