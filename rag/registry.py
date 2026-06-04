"""The strategy switch: map a config block's `name` to a stage implementation.

`config.yaml` names which implementation to use per stage; this module maps name ->
class and constructs it from the block's params. The entry points call these builders so
no entry point ever hardcodes a concrete strategy (ARCHITECTURE.md §2).

Each builder takes the stage's config dict (e.g. {"name": "recursive", "size": 512}) and
any runtime dependencies that can't live in YAML (a built index, the shared embedder).
"""

from __future__ import annotations

from rag.chunkers import Chunker, RecursiveChunker
from rag.embedders import Embedder, LocalEmbedder
from rag.generator import AnthropicGenerator, Generator
from rag.indexers import FaissFlatIndexer, Index, Indexer
from rag.loaders import Loader, MarkdownLoader
from rag.parsers import MarkdownParser, Parser
from rag.query_transform import Passthrough, QueryTransform
from rag.rerankers import NoopReranker, Reranker
from rag.retrievers import DenseRetriever, Retriever

# name -> implementation, per stage. Adding a technique = adding one row here + a class.
LOADERS = {"markdown": MarkdownLoader}
PARSERS = {"md": MarkdownParser}
CHUNKERS = {"recursive": RecursiveChunker}
EMBEDDERS = {"local": LocalEmbedder}
INDEXERS = {"faiss_flat": FaissFlatIndexer}
RERANKERS = {"noop": NoopReranker}
QUERY_TRANSFORMS = {"passthrough": Passthrough}
GENERATORS = {"anthropic": AnthropicGenerator}
# `dense` needs a built index + the shared embedder, so it is built via build_retriever().
RETRIEVERS = {"dense": DenseRetriever}


def _params(block: dict) -> dict:
    """The config block minus the `name` selector — the rest are constructor kwargs."""
    return {k: v for k, v in block.items() if k != "name"}


def _pick(table: dict, block: dict, stage: str):
    name = block.get("name")
    if name not in table:
        raise KeyError(
            f"unknown {stage} '{name}'. Available: {sorted(table)}. Check config.yaml."
        )
    return table[name]


def build_loader(block: dict | None = None) -> Loader:
    block = block or {"name": "markdown"}
    return _pick(LOADERS, block, "loader")(**_params(block))


def build_parser(block: dict | None = None) -> Parser:
    block = block or {"name": "md"}
    return _pick(PARSERS, block, "parser")(**_params(block))


def build_chunker(block: dict) -> Chunker:
    return _pick(CHUNKERS, block, "chunker")(**_params(block))


def build_embedder(block: dict) -> Embedder:
    return _pick(EMBEDDERS, block, "embedder")(**_params(block))


def build_indexer(block: dict) -> Indexer:
    return _pick(INDEXERS, block, "indexer")(**_params(block))


def build_reranker(block: dict) -> Reranker:
    return _pick(RERANKERS, block, "reranker")(**_params(block))


def build_query_transform(block: dict) -> QueryTransform:
    return _pick(QUERY_TRANSFORMS, block, "query_transform")(**_params(block))


def build_generator(block: dict) -> Generator:
    return _pick(GENERATORS, block, "generator")(**_params(block))


def build_retriever(block: dict, index: Index, embedder: Embedder) -> Retriever:
    """Retriever needs runtime deps (the loaded index + the shared embedder), so they are
    injected here rather than read from YAML."""
    impl = _pick(RETRIEVERS, block, "retriever")
    return impl(index=index, embedder=embedder, **_params(block))
