"""Evaluation harness — run the golden set through the query path, score each stage.

This is the offline loop ARCHITECTURE.md §5 is built for: change a stage in ``config.yaml``,
re-run, read the per-stage delta. It is deliberately *not* part of the query path.

Default scoring is deterministic and free (recall@k, MRR, precision@n, substring match).
The LLM judge (faithfulness + answer_relevance) is opt-in because it needs the API.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import rag.eval.metrics as metrics
from rag.eval.judge import AnthropicJudge
from rag.types import Chunk


@dataclass(frozen=True)
class GoldenRow:
    """One golden example: a query, its expected answer, and expected chunk IDs."""

    query: str
    expected_answer: str
    expected_chunk_ids: list[str]


def load_golden(path: Path) -> list[GoldenRow]:
    """Load ``eval/golden.jsonl`` (one JSON object per line)."""
    rows: list[GoldenRow] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                GoldenRow(
                    query=record["query"],
                    expected_answer=record["expected_answer"],
                    expected_chunk_ids=list(record["expected_chunk_ids"]),
                )
            )
    if not rows:
        raise ValueError(f"golden set is empty: {path}")
    return rows


@dataclass
class QueryResult:
    """Per-query scores across all stages."""

    query: str
    recall_at_k: float
    mrr: float
    precision_at_n: float
    answer_correct: bool
    faithfulness: float | None = None
    answer_relevance: float | None = None


def evaluate_row(
    row: GoldenRow,
    *,
    retriever: Any,
    reranker: Any,
    generator: Any,
    k: int,
    n: int,
    judge: AnthropicJudge | None = None,
) -> QueryResult:
    """Run one golden row through retrieve → rerank → generate and score each stage."""
    candidates = retriever.retrieve(row.query, k)
    retrieved_ids = [c.id for c in candidates]

    reranked = reranker.rerank(row.query, candidates)
    reranked_ids = [c.id for c in reranked]

    answer = generator.generate(row.query, reranked)

    result = QueryResult(
        query=row.query,
        recall_at_k=metrics.recall_at_k(retrieved_ids, row.expected_chunk_ids, k),
        mrr=metrics.mrr(retrieved_ids, row.expected_chunk_ids),
        precision_at_n=metrics.precision_at_n(reranked_ids, row.expected_chunk_ids, n),
        answer_correct=metrics.answer_contains(answer, row.expected_answer),
    )

    if judge is not None:
        score = judge.score(row.query, reranked, row.expected_answer, answer)
        result.faithfulness = score.faithfulness
        result.answer_relevance = score.answer_relevance

    return result


def aggregate(results: list[QueryResult]) -> dict[str, float]:
    """Mean each metric across all query results (skipping unset judge scores)."""
    n = len(results)
    if n == 0:
        return {}

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary = {
        "recall_at_k": mean([r.recall_at_k for r in results]),
        "mrr": mean([r.mrr for r in results]),
        "precision_at_n": mean([r.precision_at_n for r in results]),
        "answer_correct": mean([1.0 if r.answer_correct else 0.0 for r in results]),
    }
    faith = [r.faithfulness for r in results if r.faithfulness is not None]
    rel = [r.answer_relevance for r in results if r.answer_relevance is not None]
    if faith:
        summary["faithfulness"] = mean(faith)
    if rel:
        summary["answer_relevance"] = mean(rel)
    return summary


def result_to_dict(result: QueryResult) -> dict[str, Any]:
    """Serialize a QueryResult, dropping unset judge fields."""
    return {key: value for key, value in asdict(result).items() if value is not None}


# Chunks aren't needed by callers of this module beyond the stages, but re-exported for
# type-checking convenience in tests.
__all__ = [
    "Chunk",
    "GoldenRow",
    "QueryResult",
    "aggregate",
    "evaluate_row",
    "load_golden",
    "result_to_dict",
]
