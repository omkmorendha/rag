"""Derive eval/golden.jsonl from the rag-mini-wikipedia QA pairs + the current chunks.

`expected_chunk_ids` cannot exist until the chunker has run (chunk IDs are assigned at
chunk time), so the golden set is **derived after the first ingest** and is coupled to the
active chunker config (ARCHITECTURE.md §5, CLAUDE.md). Re-run this whenever the chunker
changes.

The benchmark ships QA pairs but no question→passage links, so we derive
`expected_chunk_ids` by locating the answer string in the passages and keeping only
**single-passage, unambiguous** matches — those are the cases where the derived ground
truth is trustworthy. A curated list of question IDs (verified by reading the passage) is
used so the golden set is stable across runs rather than whatever the matcher happens to
find. Each selected passage's chunk IDs (from the current chunker) become the expected set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RAW_DIR = REPO_ROOT / "data" / "raw"
CHUNKS_PATH = REPO_ROOT / "data" / "chunks.jsonl"
GOLDEN_PATH = REPO_ROOT / "eval" / "golden.jsonl"

# Curated benchmark question IDs whose answer maps to exactly one passage, verified by
# reading the passage. Stable across runs; edit deliberately. (qid -> source passage id)
CURATED: dict[int, int] = {
    12: 382,  # Who suggested Lincoln grow a beard? -> Grace Bedell
    24: 282,  # Which county was Lincoln born in? -> Hardin County
    32: 344,  # General in charge at the Battle of Antietam? -> McClellan
    10: 361,  # What did the Legal Tender Act of 1862 establish?
    49: 305,  # What trail did Lincoln use a Farmers' Almanac in?
    162: 2383,  # Three sections of a beetle? -> head, thorax, abdomen
    231: 402,  # Which state was Coolidge born in? -> Vermont
    247: 405,  # What fraternity was Coolidge a member of? -> Phi Gamma Delta
    249: 407,  # In 1905 Coolidge met and married whom? -> Grace Anna Goodhue
    290: 820,  # One significant non-official language? -> Chinese
}


def chunk_ids_by_passage(path: Path = CHUNKS_PATH) -> dict[int, list[str]]:
    """Map passage_id -> the chunk IDs the current chunker produced for it."""
    by_passage: dict[int, list[str]] = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            pid = record["metadata"]["passage_id"]
            by_passage.setdefault(pid, []).append(record["id"])
    return by_passage


def build_golden() -> list[dict]:
    """Build golden rows from the curated qid->passage map and the current chunks."""
    qa = {r["id"]: r for r in pq.read_table(RAW_DIR / "test.parquet").to_pylist()}
    by_passage = chunk_ids_by_passage()

    rows: list[dict] = []
    for qid, passage_id in CURATED.items():
        if qid not in qa:
            raise ValueError(f"curated qid {qid} not in benchmark QA set")
        chunk_ids = by_passage.get(passage_id, [])
        if not chunk_ids:
            raise ValueError(
                f"passage {passage_id} has no chunks; re-run scripts/chunk_corpus.py"
            )
        rows.append(
            {
                "query": qa[qid]["question"],
                "expected_answer": qa[qid]["answer"],
                "expected_chunk_ids": chunk_ids,
                "source_passage_id": passage_id,
                "benchmark_qid": qid,
            }
        )
    return rows


def main() -> int:
    rows = build_golden()
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GOLDEN_PATH.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} golden rows to {GOLDEN_PATH}")
    for row in rows:
        print(
            f"  qid {row['benchmark_qid']}: passage {row['source_passage_id']} "
            f"-> {len(row['expected_chunk_ids'])} chunk(s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
