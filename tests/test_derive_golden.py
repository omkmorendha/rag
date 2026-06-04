"""Tests for scripts/derive_golden.py — chunk mapping and golden union logic.

These tests avoid the real corpus and network: they exercise the pure helpers
(chunk_ids_by_passage on a fake jsonl, expected_ids_for on a fake mapping) and never
read the real parquet QA set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from derive_golden import chunk_ids_by_passage, expected_ids_for  # noqa: E402


def _write_chunks(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def test_chunk_ids_by_passage(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    _write_chunks(
        chunks,
        [
            {"id": "c1", "metadata": {"passage_id": 100}},
            {"id": "c2", "metadata": {"passage_id": 100}},
            {"id": "c3", "metadata": {"passage_id": 200}},
        ],
    )

    by_passage = chunk_ids_by_passage(chunks)

    assert by_passage == {100: ["c1", "c2"], 200: ["c3"]}


def test_chunk_ids_by_passage_skips_blank_lines(tmp_path: Path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    with chunks.open("w", encoding="utf-8") as file:
        file.write(json.dumps({"id": "c1", "metadata": {"passage_id": 1}}) + "\n")
        file.write("\n")
        file.write("   \n")
        file.write(json.dumps({"id": "c2", "metadata": {"passage_id": 1}}) + "\n")

    assert chunk_ids_by_passage(chunks) == {1: ["c1", "c2"]}


# A fake CURATED with both an int entry and a list entry, mirroring the production shape.
FAKE_CURATED: dict[int, int | list[int]] = {
    1: 100,  # single passage
    2: [100, 200],  # multi-passage
}


def test_expected_ids_for_single_passage() -> None:
    by_passage = {100: ["c1", "c2"], 200: ["c3"]}

    value = FAKE_CURATED[1]
    passage_ids = value if isinstance(value, list) else [value]

    assert expected_ids_for(passage_ids, by_passage) == ["c1", "c2"]


def test_expected_ids_for_multi_passage_ordered_union() -> None:
    by_passage = {100: ["c1", "c2"], 200: ["c3", "c4"]}

    value = FAKE_CURATED[2]
    passage_ids = value if isinstance(value, list) else [value]

    assert expected_ids_for(passage_ids, by_passage) == ["c1", "c2", "c3", "c4"]


def test_expected_ids_for_dedupes_preserving_order() -> None:
    # Overlapping chunk ids across passages are unioned, order preserved, deduped.
    by_passage = {100: ["c1", "c2"], 200: ["c2", "c3"]}

    assert expected_ids_for([100, 200], by_passage) == ["c1", "c2", "c3"]


def test_expected_ids_for_missing_passage_raises() -> None:
    by_passage = {100: ["c1"]}

    with pytest.raises(ValueError, match="passage 999 has no chunks"):
        expected_ids_for([100, 999], by_passage)
