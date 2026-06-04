"""Turn the raw rag-mini-wikipedia parquet into the Markdown corpus under `data/`.

End-to-end corpus setup for a third party from a clean checkout:

    uv run python scripts/prepare_data.py

This will:
  1. Download the parquet files if missing (via scripts/download_corpus.py).
  2. Repair text encoding (the raw passages contain UTF-8 mojibake, e.g. "RÃ\xado"
     instead of "Río") using ftfy.
  3. Write grouped Markdown files into `data/corpus/`, each passage as its own
     `## passage {id}` section so the recursive chunker has real heading structure
     and every chunk stays traceable back to its source passage id.

The source passage id is the anchor that lets `derive_golden.py` later compute
`expected_chunk_ids` for the eval golden set — keep it in the heading.

`data/` is gitignored, so this script is the reproducible way to (re)build the corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import ftfy
import pyarrow.parquet as pq

# scripts/ is a sibling of data/; import the downloader as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_corpus import RAW_DIR, REPO_ROOT, download_corpus  # noqa: E402

CORPUS_DIR = REPO_ROOT / "data" / "corpus"

# How many passages go into each Markdown file. 100 -> ~32 files for 3,200 passages.
# Grouping (rather than one file per passage) gives the chunker realistic multi-section
# documents to split on, while one section per passage keeps ids traceable.
PASSAGES_PER_FILE = 100


def _clean(text: str) -> str:
    """Repair mojibake and normalize whitespace."""
    fixed = ftfy.fix_text(text)
    # Collapse the stray runs of spaces the source has (e.g. "in  ; pron.  ,").
    return " ".join(fixed.split()).strip()


def build_corpus() -> int:
    """Read passages.parquet, write Markdown into data/corpus/. Returns file count."""
    passages_path = RAW_DIR / "passages.parquet"
    rows = pq.read_table(passages_path).to_pylist()
    rows.sort(key=lambda r: r["id"])

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    # Clear any stale corpus so re-runs are deterministic.
    for old in CORPUS_DIR.glob("*.md"):
        old.unlink()

    file_count = 0
    written = 0
    skipped = 0
    for start in range(0, len(rows), PASSAGES_PER_FILE):
        batch = rows[start : start + PASSAGES_PER_FILE]
        first_id = batch[0]["id"]
        last_id = batch[-1]["id"]
        lines = [f"# rag-mini-wikipedia passages {first_id}–{last_id}", ""]
        for r in batch:
            body = _clean(r["passage"])
            if not body:
                skipped += 1
                continue
            # The id in the heading is the load-bearing anchor for golden derivation.
            lines.append(f"## passage {r['id']}")
            lines.append("")
            lines.append(body)
            lines.append("")
            written += 1
        out_path = CORPUS_DIR / f"passages_{first_id:05d}_{last_id:05d}.md"
        out_path.write_text("\n".join(lines), encoding="utf-8")
        file_count += 1

    print(
        f"  wrote {written} passages across {file_count} Markdown files "
        f"({skipped} empty passages skipped)"
    )
    print(f"  corpus at: {CORPUS_DIR}")
    return file_count


def main() -> int:
    force = "--force" in sys.argv
    print("Step 1/2: ensure raw parquet is present")
    download_corpus(force=force)
    print("Step 2/2: build Markdown corpus")
    build_corpus()
    print("Done. Next: uv run python ingest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
