"""Parse and chunk the Markdown corpus selected by config.yaml."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.loaders import iter_files  # noqa: E402
from rag.parsers import MarkdownPassageParser  # noqa: E402
from rag.registry import build_chunker  # noqa: E402
from rag.types import Chunk  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = REPO_ROOT / "data" / "corpus"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data" / "chunks.jsonl"


def build_chunks(
    corpus_dir: Path = DEFAULT_CORPUS_DIR, config_path: Path | None = None
) -> list[Chunk]:
    """Load Markdown files, parse them into documents, then chunk each document."""
    parser = MarkdownPassageParser()
    chunker = build_chunker(path=config_path)

    chunks: list[Chunk] = []
    for path in iter_files(corpus_dir, extensions=(".md",)):
        for document in parser.parse(path):
            chunks.extend(chunker.chunk(document))
    return chunks


def write_jsonl(chunks: list[Chunk], path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """Write chunks in a simple JSONL shape for inspection and later ingestion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    },
                    ensure_ascii=False,
                )
            )
            file.write("\n")


def main() -> int:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    arg_parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    arg_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arg_parser.add_argument("--write", action="store_true")
    args = arg_parser.parse_args()

    chunks = build_chunks(corpus_dir=args.corpus_dir, config_path=args.config)
    print(f"built {len(chunks)} chunks from {args.corpus_dir}")
    if chunks:
        first = chunks[0]
        print(f"first chunk: {first.id} ({first.metadata['token_count']} tokens)")

    if args.write:
        write_jsonl(chunks, args.output)
        print(f"wrote chunks to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
