"""Tests for the evaluation harness and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.eval import GoldenRow, aggregate, evaluate_row, load_golden, result_to_dict
from rag.eval.harness import QueryResult
from rag.eval.judge import JudgeScore, _parse_score
from rag.eval.metrics import (
    answer_contains,
    is_idk,
    mrr,
    precision_at_n,
    recall_at_k,
)
from rag.types import Chunk

# --- retrieval / rerank metrics ----------------------------------------------


def test_recall_at_k_counts_expected_in_topk() -> None:
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["a", "c"], k=4) == 1.0
    assert recall_at_k(retrieved, ["a", "z"], k=4) == 0.5
    assert recall_at_k(retrieved, ["a", "c"], k=1) == 0.5  # only "a" in top-1
    assert recall_at_k(retrieved, ["z"], k=4) == 0.0


def test_recall_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="expected_ids"):
        recall_at_k(["a"], [], k=1)
    with pytest.raises(ValueError, match="k must be"):
        recall_at_k(["a"], ["a"], k=0)


def test_mrr_uses_first_expected_rank() -> None:
    assert mrr(["a", "b", "c"], ["a"]) == 1.0
    assert mrr(["x", "b", "c"], ["b"]) == 0.5
    assert mrr(["x", "y", "c"], ["c"]) == pytest.approx(1 / 3)
    assert mrr(["x", "y", "z"], ["a"]) == 0.0


def test_precision_at_n() -> None:
    reranked = ["a", "x", "b", "y"]
    assert precision_at_n(reranked, ["a", "b"], n=2) == 0.5  # a hit, x miss
    assert precision_at_n(reranked, ["a", "b"], n=4) == 0.5  # 2 of 4
    assert precision_at_n(reranked, ["a", "b", "x", "y"], n=4) == 1.0
    assert precision_at_n([], ["a"], n=4) == 0.0


def test_precision_rejects_bad_n() -> None:
    with pytest.raises(ValueError, match="n must be"):
        precision_at_n(["a"], ["a"], n=0)


# --- generation deterministic checks -----------------------------------------


def test_answer_contains_is_case_insensitive() -> None:
    assert answer_contains("The answer is Grace Bedell.", "grace bedell")
    assert not answer_contains("Someone else entirely.", "Grace Bedell")


def test_is_idk() -> None:
    assert is_idk("I don't know.")
    assert is_idk("Honestly, I dont know the answer")
    assert not is_idk("The answer is Paris.")


# --- judge parsing (no API) --------------------------------------------------


def test_parse_score_extracts_and_clamps() -> None:
    score = _parse_score('Here: {"faithfulness": 1.0, "answer_relevance": 0.8} done')
    assert score == JudgeScore(faithfulness=1.0, answer_relevance=0.8)
    clamped = _parse_score('{"faithfulness": 1.7, "answer_relevance": -0.2}')
    assert clamped.faithfulness == 1.0
    assert clamped.answer_relevance == 0.0


def test_parse_score_rejects_non_json() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        _parse_score("no json here")


# --- golden loading ----------------------------------------------------------


def test_load_golden(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps({"query": "q", "expected_answer": "a", "expected_chunk_ids": ["c0"]})
        + "\n",
        encoding="utf-8",
    )
    rows = load_golden(path)
    assert rows == [
        GoldenRow(query="q", expected_answer="a", expected_chunk_ids=["c0"])
    ]


def test_load_golden_rejects_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_golden(path)


# --- evaluate_row + aggregate (fakes, no models/API) -------------------------


class _FakeRetriever:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def retrieve(self, query: str, k: int) -> list[Chunk]:  # noqa: ARG002
        return [Chunk(id=i, text=f"text {i}", metadata={}) for i in self.ids[:k]]


class _FakeReranker:
    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:  # noqa: ARG002
        return chunks  # identity


class _FakeGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, query: str, chunks: list[Chunk]) -> str:  # noqa: ARG002
        return self.answer


def test_evaluate_row_scores_each_stage() -> None:
    row = GoldenRow(
        query="who?", expected_answer="grace bedell", expected_chunk_ids=["c1"]
    )
    result = evaluate_row(
        row,
        retriever=_FakeRetriever(["c0", "c1", "c2"]),
        reranker=_FakeReranker(),
        generator=_FakeGenerator("The answer is Grace Bedell."),
        k=3,
        n=2,
    )
    assert result.recall_at_k == 1.0
    assert result.mrr == 0.5  # c1 at rank 2
    assert result.precision_at_n == 0.5  # c1 in top-2 of [c0, c1]
    assert result.answer_correct is True
    assert result.faithfulness is None  # no judge


def test_aggregate_means_and_skips_unset_judge() -> None:
    results = [
        QueryResult("q1", 1.0, 1.0, 1.0, True),
        QueryResult("q2", 0.0, 0.0, 0.0, False, faithfulness=0.5, answer_relevance=0.5),
    ]
    summary = aggregate(results)
    assert summary["recall_at_k"] == 0.5
    assert summary["answer_correct"] == 0.5
    # Only one row has judge scores; mean over present values.
    assert summary["faithfulness"] == 0.5


def test_result_to_dict_drops_unset_judge_fields() -> None:
    d = result_to_dict(QueryResult("q", 1.0, 1.0, 1.0, True))
    assert "faithfulness" not in d
    assert d["query"] == "q"
