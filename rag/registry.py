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
from rag.embedders import Embedder, HashingEmbedder, SentenceTransformerEmbedder

CHUNKERS = {
    FixedTokenChunker.name: FixedTokenChunker,
    RecursiveTextChunker.name: RecursiveTextChunker,
    SentenceWindowChunker.name: SentenceWindowChunker,
}

EMBEDDERS = {
    SentenceTransformerEmbedder.name: SentenceTransformerEmbedder,
    HashingEmbedder.name: HashingEmbedder,
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


def build_embedder(
    config: dict[str, Any] | None = None, *, path: Path | None = None
) -> Embedder:
    """Build the configured embedder strategy."""
    loaded = load_config(path or Path("config.yaml")) if config is None else config
    embedder_config = stage_config(loaded, "embedder")
    name = embedder_config.get("name", SentenceTransformerEmbedder.name)
    if not isinstance(name, str):
        raise ValueError("embedder.name must be a string")
    if name not in EMBEDDERS:
        available = ", ".join(sorted(EMBEDDERS))
        raise ValueError(f"unknown embedder {name!r}; available: {available}")

    params = {key: value for key, value in embedder_config.items() if key != "name"}
    return EMBEDDERS[name](**params)
