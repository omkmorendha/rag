"""Encode chunk JSONL records selected by config.yaml."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.registry import build_embedder  # noqa: E402
from rag.types import Chunk  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "chunks.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "embedded_chunks.jsonl"


def read_jsonl(path: Path = DEFAULT_INPUT_PATH) -> list[Chunk]:
    """Read chunk records written by scripts/chunk_corpus.py."""
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            chunks.append(
                Chunk(
                    id=record["id"],
                    text=record["text"],
                    metadata=record.get("metadata", {}),
                    embedding=record.get("embedding"),
                )
            )
    return chunks


def embed_chunks(
    chunks: list[Chunk],
    *,
    config_path: Path | None = None,
    batch_size: int = 32,
) -> list[Chunk]:
    """Attach embeddings to chunks in batches."""
    embedder = build_embedder(path=config_path)
    embedded: list[Chunk] = []
    for batch in _batches(chunks, batch_size):
        vectors = embedder.embed([chunk.text for chunk in batch])
        if len(vectors) != len(batch):
            raise ValueError(
                f"embedder returned {len(vectors)} vectors for {len(batch)} chunks"
            )
        for chunk, vector in zip(batch, vectors, strict=True):
            metadata = {
                **chunk.metadata,
                "embedding_dimension": len(vector),
            }
            embedded.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=metadata,
                    embedding=vector,
                    score=chunk.score,
                )
            )
    return embedded


def write_jsonl(chunks: list[Chunk], path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Write embedded chunks as JSONL for indexing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                        "embedding": chunk.embedding,
                    },
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def _batches(items: list[Chunk], size: int) -> Iterable[list[Chunk]]:
    if size <= 0:
        raise ValueError("batch size must be greater than 0")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> int:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    arg_parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    arg_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arg_parser.add_argument("--batch-size", type=int, default=32)
    arg_parser.add_argument("--write", action="store_true")
    args = arg_parser.parse_args()

    chunks = read_jsonl(args.input)
    embedded = embed_chunks(
        chunks,
        config_path=args.config,
        batch_size=args.batch_size,
    )
    print(f"embedded {len(embedded)} chunks from {args.input}")
    if embedded:
        first = embedded[0]
        dimension = len(first.embedding or [])
        print(f"first embedding: {first.id} ({dimension} dimensions)")

    if args.write:
        write_jsonl(embedded, args.output)
        print(f"wrote embedded chunks to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
