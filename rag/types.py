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

