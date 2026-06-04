"""Build a persisted vector index from embedded chunk records selected by config.yaml.

This is the ``index`` stage of the ingest pipeline (ARCHITECTURE.md §0): it reads the
embedded chunks written by ``scripts/embed_chunks.py`` and persists a searchable index to
``vectorstore/``. The index is the boundary the query pipeline loads from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.registry import build_indexer  # noqa: E402
from rag.types import Chunk  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "data" / "embedded_chunks.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "vectorstore"


def read_embedded_chunks(path: Path = DEFAULT_INPUT_PATH) -> list[Chunk]:
    """Read embedded chunk records written by scripts/embed_chunks.py."""
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


def main() -> int:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    arg_parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    arg_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    arg_parser.add_argument("--write", action="store_true")
    args = arg_parser.parse_args()

    chunks = read_embedded_chunks(args.input)
    indexer = build_indexer(path=args.config)
    index = indexer.build(chunks)
    print(
        f"indexed {index.meta['n_chunks']} chunks "
        f"({index.meta['dim']}-dim, {index.meta['metric']}, kind={index.meta['kind']})"
    )

    if args.write:
        index.save(args.output_dir)
        print(f"wrote index to {args.output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
