"""Online query: load index -> transform query -> retrieve -> rerank -> prompt -> generate.

Runs per question against the index ingest.py persisted. Run with:

    uv run python query.py "your question here"
"""

from __future__ import annotations

import sys

from rag.config import REPO_ROOT, load_config
from rag.registry import (
    build_embedder,
    build_generator,
    build_indexer,
    build_query_transform,
    build_reranker,
    build_retriever,
)


def answer(question: str, *, show_sources: bool = True) -> str:
    config = load_config()
    vectorstore = REPO_ROOT / config.get("vectorstore", "vectorstore")
    if not vectorstore.exists():
        raise SystemExit(
            f"No index at {vectorstore}. Run: uv run python ingest.py"
        )

    # The embedder is the shared invariant: the SAME one that built the index must embed
    # the query, or query and corpus land in different vector spaces (ARCHITECTURE.md §2).
    embedder = build_embedder(config["embedder"])
    index = build_indexer(config["indexer"]).load(vectorstore)

    query_transform = build_query_transform(config["query_transform"])
    retriever = build_retriever(config["retriever"], index=index, embedder=embedder)
    reranker = build_reranker(config["reranker"])
    generator = build_generator(config["generator"])

    transformed = query_transform.transform(question)
    retrieved = retriever.retrieve(transformed, config["retriever"].get("k", 50))
    reranked = reranker.rerank(transformed, retrieved)

    if show_sources:
        top_n = config["generator"].get("top_n", 8)
        print(f"\nretrieved {len(retrieved)} -> reranked, top {min(top_n, len(reranked))} sources:")
        for i, c in enumerate(reranked[:top_n], start=1):
            passage = c.metadata.get("passage_id")
            score = f"{c.score:.3f}" if c.score is not None else "n/a"
            print(f"  [{i}] passage {passage}  score={score}  {c.text[:80]}...")
        print()

    return generator.generate(transformed, reranked)


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: uv run python query.py "your question here"')
        return 1
    question = " ".join(sys.argv[1:])
    print(answer(question))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
