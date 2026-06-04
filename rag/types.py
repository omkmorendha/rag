"""The types that thread every stage.

The single most important design decision for swappability (ARCHITECTURE.md §1):
**every stage speaks `Chunk`.** A chunk carries its `text` and `metadata` all the way
to the prompt, not just a vector id — that is what lets stages be swapped independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Document:
    """A parsed source document: clean text plus provenance metadata."""

    id: str
    text: str  # clean, parser-normalized text
    metadata: dict = field(default_factory=dict)  # source path, doc type, title, ...


@dataclass
class Chunk:
    """A retrieval unit. `text` survives all the way into the prompt.

    `metadata` is append-only and generous ("you cannot filter on what you don't have"):
    source, page/section, timestamp, parent_id. `score` is set by the retriever and
    OVERWRITTEN by the reranker.
    """

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: list[float] | None = None
    score: float | None = None
