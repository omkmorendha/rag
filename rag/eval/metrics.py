"""Per-component scoring metrics (ARCHITECTURE.md §5).

The principle is to **score each stage separately** so a config change can be traced to the
stage it helped or hurt. Retrieval and rerank are pure set/rank math against
``expected_chunk_ids`` — no model, deterministic, free. Generation has a free deterministic
check here (substring / "I don't know"); the optional LLM-as-judge lives in ``judge.py``.

All functions take the *ordered* list of retrieved/reranked chunk IDs (best-first) and the
set of expected IDs.
"""

from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Fraction of expected chunks present in the top-``k`` retrieved.

    Answers "did retrieval surface the right chunks at all?" 1.0 means every expected
    chunk is somewhere in the top-k.
    """
    if not expected_ids:
        raise ValueError("expected_ids must not be empty")
    if k <= 0:
        raise ValueError("k must be greater than 0")
    top = set(retrieved_ids[:k])
    hits = sum(1 for cid in expected_ids if cid in top)
    return hits / len(expected_ids)


def mrr(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """Reciprocal rank of the *first* expected chunk (1-indexed); 0.0 if none present.

    Rewards getting a relevant chunk high in the list, not just present.
    """
    expected = set(expected_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in expected:
            return 1.0 / rank
    return 0.0


def precision_at_n(reranked_ids: list[str], expected_ids: list[str], n: int) -> float:
    """Fraction of the top-``n`` reranked chunks that are expected.

    Answers "after reranking, how clean is the head of the list?" — the reranker's job is
    to make the top few precise.
    """
    if n <= 0:
        raise ValueError("n must be greater than 0")
    expected = set(expected_ids)
    top = reranked_ids[:n]
    if not top:
        return 0.0
    hits = sum(1 for cid in top if cid in expected)
    return hits / len(top)


def answer_contains(answer: str, expected_answer: str) -> bool:
    """Crude, deterministic generation check: is ``expected_answer`` present in the output?

    Case-insensitive substring match. Brittle to paraphrase (use the LLM judge for the real
    signal), but free and good enough to catch gross regressions.
    """
    return expected_answer.strip().lower() in answer.lower()


def is_idk(answer: str) -> bool:
    """Whether the generator abstained ("I don't know")."""
    return "i don't know" in answer.lower() or "i dont know" in answer.lower()
