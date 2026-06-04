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
from rag.indexers import FaissFlatIndexer, Indexer

CHUNKERS = {
    FixedTokenChunker.name: FixedTokenChunker,
    RecursiveTextChunker.name: RecursiveTextChunker,
    SentenceWindowChunker.name: SentenceWindowChunker,
}

EMBEDDERS = {
    SentenceTransformerEmbedder.name: SentenceTransformerEmbedder,
    HashingEmbedder.name: HashingEmbedder,
}

INDEXERS = {
    FaissFlatIndexer.name: FaissFlatIndexer,
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


def build_indexer(
    config: dict[str, Any] | None = None, *, path: Path | None = None
) -> Indexer:
    """Build the configured indexer strategy.

    The active embedder's model is threaded into the indexer so it is recorded in the
    index metadata; ``load`` rejects an index built with a different embedder (the shared
    vector-space invariant from ARCHITECTURE.md §2).
    """
    loaded = load_config(path or Path("config.yaml")) if config is None else config
    indexer_config = stage_config(loaded, "indexer")
    name = indexer_config.get("name", FaissFlatIndexer.name)
    if not isinstance(name, str):
        raise ValueError("indexer.name must be a string")
    if name not in INDEXERS:
        available = ", ".join(sorted(INDEXERS))
        raise ValueError(f"unknown indexer {name!r}; available: {available}")

    embedder_config = stage_config(loaded, "embedder")
    embedder_model = embedder_config.get("model")

    params = {key: value for key, value in indexer_config.items() if key != "name"}
    params.setdefault("embedder_model", embedder_model)
    return INDEXERS[name](**params)
