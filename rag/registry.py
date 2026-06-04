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
from rag.generator import AnthropicGenerator, Generator
from rag.indexers import FaissFlatIndexer, Index, Indexer
from rag.query_transform import (
    PassthroughTransform,
    QueryTransform,
    RewriteTransform,
    StepBackTransform,
)
from rag.rerankers import CrossEncoderReranker, NoopReranker, Reranker
from rag.retrievers import DenseRetriever, Retriever

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

QUERY_TRANSFORMS = {
    PassthroughTransform.name: PassthroughTransform,
    RewriteTransform.name: RewriteTransform,
    StepBackTransform.name: StepBackTransform,
}

RETRIEVERS = {
    DenseRetriever.name: DenseRetriever,
}

RERANKERS = {
    NoopReranker.name: NoopReranker,
    CrossEncoderReranker.name: CrossEncoderReranker,
}

GENERATORS = {
    AnthropicGenerator.name: AnthropicGenerator,
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


def build_query_transform(
    config: dict[str, Any] | None = None, *, path: Path | None = None
) -> QueryTransform:
    """Build the configured query-transform strategy.

    Defaults to :class:`~rag.query_transform.PassthroughTransform` (the identity baseline)
    when the ``query_transform`` block is absent, so an unconfigured pipeline retrieves
    exactly as it did before this stage existed.
    """
    loaded = load_config(path or Path("config.yaml")) if config is None else config
    transform_config = stage_config(loaded, "query_transform")
    name = transform_config.get("name", PassthroughTransform.name)
    if not isinstance(name, str):
        raise ValueError("query_transform.name must be a string")
    if name not in QUERY_TRANSFORMS:
        available = ", ".join(sorted(QUERY_TRANSFORMS))
        raise ValueError(f"unknown query_transform {name!r}; available: {available}")

    params = {key: value for key, value in transform_config.items() if key != "name"}
    return QUERY_TRANSFORMS[name](**params)


def build_retriever(
    index: Index,
    embedder: Embedder,
    config: dict[str, Any] | None = None,
    *,
    path: Path | None = None,
) -> Retriever:
    """Build the configured retriever, wiring in the loaded index and the query embedder.

    The retriever embeds the query with the *same* embedder used at ingest (ARCHITECTURE.md
    §2), so the embedder is passed in rather than rebuilt from config here.

    Note: ``k`` is a *query-time* argument to ``retrieve(query, k)``, not a constructor
    param — read it with :func:`retriever_k`. It is ignored here.
    """
    loaded = load_config(path or Path("config.yaml")) if config is None else config
    retriever_config = stage_config(loaded, "retriever")
    name = retriever_config.get("name", DenseRetriever.name)
    if not isinstance(name, str):
        raise ValueError("retriever.name must be a string")
    if name not in RETRIEVERS:
        available = ", ".join(sorted(RETRIEVERS))
        raise ValueError(f"unknown retriever {name!r}; available: {available}")

    params = {
        key: value
        for key, value in retriever_config.items()
        if key not in ("name", "k")
    }
    return RETRIEVERS[name](index, embedder, **params)


def retriever_k(
    config: dict[str, Any] | None = None, *, path: Path | None = None, default: int = 50
) -> int:
    """Return the configured over-retrieve count ``k`` for the retriever stage."""
    loaded = load_config(path or Path("config.yaml")) if config is None else config
    value = stage_config(loaded, "retriever").get("k", default)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("retriever.k must be a positive integer")
    return value


def build_reranker(
    config: dict[str, Any] | None = None, *, path: Path | None = None
) -> Reranker:
    """Build the configured reranker strategy."""
    loaded = load_config(path or Path("config.yaml")) if config is None else config
    reranker_config = stage_config(loaded, "reranker")
    name = reranker_config.get("name", NoopReranker.name)
    if not isinstance(name, str):
        raise ValueError("reranker.name must be a string")
    if name not in RERANKERS:
        available = ", ".join(sorted(RERANKERS))
        raise ValueError(f"unknown reranker {name!r}; available: {available}")

    params = {key: value for key, value in reranker_config.items() if key != "name"}
    return RERANKERS[name](**params)


def build_generator(
    config: dict[str, Any] | None = None, *, path: Path | None = None
) -> Generator:
    """Build the configured generator strategy."""
    loaded = load_config(path or Path("config.yaml")) if config is None else config
    generator_config = stage_config(loaded, "generator")
    name = generator_config.get("name", AnthropicGenerator.name)
    if not isinstance(name, str):
        raise ValueError("generator.name must be a string")
    if name not in GENERATORS:
        available = ", ".join(sorted(GENERATORS))
        raise ValueError(f"unknown generator {name!r}; available: {available}")

    params = {key: value for key, value in generator_config.items() if key != "name"}
    return GENERATORS[name](**params)
