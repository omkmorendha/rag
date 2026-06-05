"""Evaluation harness — run the golden set through the query path, score each stage.

This is the offline loop ARCHITECTURE.md §5 is built for: change a stage in ``config.yaml``,
re-run, read the per-stage delta. It is deliberately *not* part of the query path.

Default scoring is deterministic and free (recall@k, MRR, precision@n, substring match).
The LLM judge (faithfulness + answer_relevance) is opt-in because it needs the API.
"""

from __future__ import annotations

import json
import time
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
    answer_correct: bool | None  # None when generation was skipped (--no-generate)
    faithfulness: float | None = None
    answer_relevance: float | None = None
    # Component 2: extra recall cutoffs (computed off the same retrieved id list).
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    # Component 2: per-stage wall-clock latency in milliseconds.
    transform_ms: float | None = None
    retrieve_ms: float | None = None
    rerank_ms: float | None = None
    generate_ms: float | None = None
    query_latency_ms: float | None = None


def evaluate_row(
    row: GoldenRow,
    *,
    retriever: Any,
    reranker: Any,
    generator: Any,
    k: int,
    n: int,
    judge: AnthropicJudge | None = None,
    query_transform: Any | None = None,
) -> QueryResult:
    """Run one golden row through retrieve → rerank → generate and score each stage.

    ``query_transform`` (optional; default ``None`` = no transform, exact back-compat)
    rewrites the query *only for retrieval*. The transformed query is what gets embedded
    and searched, but rerank, generation, and the judge all keep using the ORIGINAL
    ``row.query`` — the system answers the question the user actually asked.
    """
    _t0 = time.perf_counter()
    retrieval_query = (
        query_transform.transform(row.query) if query_transform else row.query
    )
    transform_ms = (time.perf_counter() - _t0) * 1000.0

    _t0 = time.perf_counter()
    candidates = retriever.retrieve(retrieval_query, k)
    retrieve_ms = (time.perf_counter() - _t0) * 1000.0
    retrieved_ids = [c.id for c in candidates]

    _t0 = time.perf_counter()
    reranked = reranker.rerank(row.query, candidates)
    rerank_ms = (time.perf_counter() - _t0) * 1000.0
    reranked_ids = [c.id for c in reranked]

    _t0 = time.perf_counter()
    answer = generator.generate(row.query, reranked)
    generate_ms = (time.perf_counter() - _t0) * 1000.0

    query_latency_ms = transform_ms + retrieve_ms + rerank_ms + generate_ms

    # An empty answer means generation was skipped (--no-generate / _NullGenerator);
    # scoring a substring match against "" would report a misleading 0.0, so leave
    # answer_correct unset (None → "not measured") in that case.
    answer_correct = metrics.answer_contains(answer, row.expected_answer) if answer else None

    result = QueryResult(
        query=row.query,
        recall_at_k=metrics.recall_at_k(retrieved_ids, row.expected_chunk_ids, k),
        mrr=metrics.mrr(retrieved_ids, row.expected_chunk_ids),
        precision_at_n=metrics.precision_at_n(reranked_ids, row.expected_chunk_ids, n),
        answer_correct=answer_correct,
        recall_at_5=metrics.recall_at_k(retrieved_ids, row.expected_chunk_ids, 5),
        recall_at_10=metrics.recall_at_k(retrieved_ids, row.expected_chunk_ids, 10),
        transform_ms=transform_ms,
        retrieve_ms=retrieve_ms,
        rerank_ms=rerank_ms,
        generate_ms=generate_ms,
        query_latency_ms=query_latency_ms,
    )

    if judge is not None:
        score = judge.score(row.query, reranked, row.expected_answer, answer)
        result.faithfulness = score.faithfulness
        result.answer_relevance = score.answer_relevance

    return result


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (``pct`` in [0, 100]) over ``values``.

    Returns 0.0 for an empty list. Used for p50/p95 of query latency.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


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
    }

    # answer_correct is None when generation was skipped; only average measured rows so a
    # --no-generate run reports no answer_correct rather than a misleading 0.0.
    answered = [r.answer_correct for r in results if r.answer_correct is not None]
    if answered:
        summary["answer_correct"] = mean([1.0 if ok else 0.0 for ok in answered])

    # Component 2: extra recall cutoffs (skip rows where they were not computed).
    recall5 = [r.recall_at_5 for r in results if r.recall_at_5 is not None]
    recall10 = [r.recall_at_10 for r in results if r.recall_at_10 is not None]
    if recall5:
        summary["recall_at_5"] = mean(recall5)
    if recall10:
        summary["recall_at_10"] = mean(recall10)

    # Component 2: per-stage latency means (skip unset).
    for field in ("transform_ms", "retrieve_ms", "rerank_ms", "generate_ms"):
        vals = [getattr(r, field) for r in results if getattr(r, field) is not None]
        if vals:
            summary[field] = mean(vals)

    # Component 2: end-to-end latency mean + p50/p95.
    latencies = [
        r.query_latency_ms for r in results if r.query_latency_ms is not None
    ]
    if latencies:
        summary["query_latency_ms"] = mean(latencies)
        summary["query_latency_ms_p50"] = _percentile(latencies, 50.0)
        summary["query_latency_ms_p95"] = _percentile(latencies, 95.0)

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
