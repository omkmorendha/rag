"""Strategy registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.chunkers import (
    Chunker,
    FixedTokenChunker,
    RecursiveTextChunker,
    SentenceWindowChunker,
)
from rag.config import load_config, stage_config

CHUNKERS = {
    FixedTokenChunker.name: FixedTokenChunker,
    RecursiveTextChunker.name: RecursiveTextChunker,
    SentenceWindowChunker.name: SentenceWindowChunker,
}


def build_chunker(
    config: dict[str, Any] | None = None, *, path: Path | None = None
) -> Chunker:
    """Build the configured chunker strategy."""
    loaded = load_config(path or Path("config.yaml")) if config is None else config
    chunker_config = stage_config(loaded, "chunker")
    name = chunker_config.get("name", RecursiveTextChunker.name)
    if not isinstance(name, str):
        raise ValueError("chunker.name must be a string")
    if name not in CHUNKERS:
        available = ", ".join(sorted(CHUNKERS))
        raise ValueError(f"unknown chunker {name!r}; available: {available}")

    params = {key: value for key, value in chunker_config.items() if key != "name"}
    return CHUNKERS[name](**params)
