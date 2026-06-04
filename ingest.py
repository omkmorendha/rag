"""Offline ingestion: load -> parse -> chunk -> embed -> index -> PERSIST to vectorstore/.

Runs once per corpus change. The output — a persisted index — is the boundary between
this pipeline and query.py (ARCHITECTURE.md §0). Run with:

    uv run python ingest.py
"""

from __future__ import annotations

from rag.config import REPO_ROOT, load_config
from rag.registry import (
    build_chunker,
    build_embedder,
    build_indexer,
    build_loader,
    build_parser,
)
from rag.types import Chunk

DATA_DIR = REPO_ROOT / "data" / "corpus"


def main() -> int:
    config = load_config()
    vectorstore = REPO_ROOT / config.get("vectorstore", "vectorstore")

    loader = build_loader(config.get("loader"))
    parser = build_parser(config.get("parser"))
    chunker = build_chunker(config["chunker"])
    embedder = build_embedder(config["embedder"])
    indexer = build_indexer(config["indexer"])

    files = loader.load_files(DATA_DIR)
    if not files:
        print(f"No Markdown files under {DATA_DIR}. Run scripts/prepare_data.py first.")
        return 1
    print(f"load:  {len(files)} files under {DATA_DIR}")

    docs = [parser.parse(f) for f in files]
    print(f"parse: {len(docs)} documents")

    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunker.chunk(doc))
    print(f"chunk: {len(chunks)} chunks ({config['chunker']})")

    vectors = embedder.embed([c.text for c in chunks])
    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec.tolist()
    print(f"embed: {vectors.shape[0]} x {vectors.shape[1]}-d vectors ({config['embedder']['model']})")

    index = indexer.build(chunks)
    index.save(vectorstore)
    print(f"index: persisted to {vectorstore} ({config['indexer']['name']})")
    print("Done. Next: uv run python query.py \"your question\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
