"""Evaluation harness: per-component scoring of the query path (ARCHITECTURE.md §5)."""

from rag.eval.harness import (
    GoldenRow,
    QueryResult,
    aggregate,
    evaluate_row,
    load_golden,
    result_to_dict,
)
from rag.eval.judge import AnthropicJudge, JudgeScore

__all__ = [
    "AnthropicJudge",
    "GoldenRow",
    "JudgeScore",
    "QueryResult",
    "aggregate",
    "evaluate_row",
    "load_golden",
    "result_to_dict",
]
