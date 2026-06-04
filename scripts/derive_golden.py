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

import argparse
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RAW_DIR = REPO_ROOT / "data" / "raw"
CHUNKS_PATH = REPO_ROOT / "data" / "chunks.jsonl"
GOLDEN_PATH = REPO_ROOT / "eval" / "golden.jsonl"

# Curated benchmark question IDs whose answer maps to a known passage (or passages),
# verified by reading the passage. Stable across runs; edit deliberately. The value is
# either a single passage id or a list of passage ids (multi-passage golden rows).
# (qid -> source passage id | list of source passage ids)
CURATED: dict[int, int | list[int]] = {
    # --- original single-passage set ---
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
    # --- added single-passage set (answer verified present in the passage) ---
    136: 732,  # What happened in 1833? -> Avogadro recalled to Turin university
    140: 147,  # What is named after Celsius? -> the Celsius crater on the Moon
    154: 142,  # 1730-1744? -> professor of astronomy at Uppsala University
    155: 144,  # What happened in 1745? -> the scale was reversed
    210: 405,  # When did Coolidge drop John? -> upon graduating from college
    313: 2274,  # How many provinces/territories does Canada have? -> ten + three
    317: 2327,  # Most densely populated part of Canada? -> Quebec City-Windsor
    319: 2326,  # Largest country in the world? -> Canada is second, after Russia
    340: 2273,  # What happened in 1867? -> Canada formed as a federal polity
    347: 3171,  # Specialized duck that catches large fish? -> the smew
    387: 3170,  # What lets a duck filter water? -> lamellae
    403: 880,  # Population of Egypt? -> more than 78 million
    416: 876,  # Since when has Egypt been a republic? -> June 18 1953
    436: 1177,  # How long may elephants live? -> 70 years
    442: 1177,  # Elephant weight at birth? -> 120 kilograms
    459: 1234,  # What are elephant ears important for? -> temperature regulation
    511: 1655,  # What is Finland's economy like? -> industrialised free-market
    567: 2003,  # Positions Ford played? -> center and linebacker
    571: 2074,  # Who did Ford nominate for VP? -> Bob Dole
    # --- multi-passage set (answer verified present in EVERY listed passage) ---
    26: [278, 321, 322],  # When did Lincoln first serve as President? -> Mar 4 1861
    164: [2407, 2409],  # Defense mechanism using colour/shape? -> mimicry
    172: [2386, 2397],  # Similarities between beetles and grasshoppers? -> mouthparts
    184: [2440, 2441],  # Study of beetles called? -> coleopterology
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


def expected_ids_for(
    passage_ids: list[int], by_passage: dict[int, list[str]]
) -> list[str]:
    """Union the chunk ids of the listed passages, preserving order and deduping.

    Raises ValueError if any passage has zero chunks, so a chunker that drops a passage
    surfaces as an error rather than a silent skip.
    """
    expected: list[str] = []
    seen: set[str] = set()
    for passage_id in passage_ids:
        chunk_ids = by_passage.get(passage_id, [])
        if not chunk_ids:
            raise ValueError(
                f"passage {passage_id} has no chunks; re-run scripts/chunk_corpus.py"
            )
        for chunk_id in chunk_ids:
            if chunk_id not in seen:
                seen.add(chunk_id)
                expected.append(chunk_id)
    return expected


def build_golden(
    chunks_path: Path = CHUNKS_PATH, raw_dir: Path = RAW_DIR
) -> list[dict]:
    """Build golden rows from the curated qid->passage map and the current chunks."""
    qa = {r["id"]: r for r in pq.read_table(raw_dir / "test.parquet").to_pylist()}
    by_passage = chunk_ids_by_passage(chunks_path)

    rows: list[dict] = []
    for qid, value in CURATED.items():
        if qid not in qa:
            raise ValueError(f"curated qid {qid} not in benchmark QA set")
        passage_ids = value if isinstance(value, list) else [value]
        chunk_ids = expected_ids_for(passage_ids, by_passage)
        rows.append(
            {
                "query": qa[qid]["question"],
                "expected_answer": qa[qid]["answer"],
                "expected_chunk_ids": chunk_ids,
                "source_passage_id": value,
                "benchmark_qid": qid,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks",
        type=Path,
        default=CHUNKS_PATH,
        help="path to chunks.jsonl (default: data/chunks.jsonl)",
    )
    parser.add_argument(
        "--output",
        "--golden",
        dest="output",
        type=Path,
        default=GOLDEN_PATH,
        help="path to write golden.jsonl (default: eval/golden.jsonl)",
    )
    args = parser.parse_args(argv)

    rows = build_golden(chunks_path=args.chunks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} golden rows to {args.output}")
    for row in rows:
        print(
            f"  qid {row['benchmark_qid']}: passage {row['source_passage_id']} "
            f"-> {len(row['expected_chunk_ids'])} chunk(s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
