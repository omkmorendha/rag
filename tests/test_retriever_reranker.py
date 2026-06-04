"""Tests for the dense retriever and the reranker stages."""

from __future__ import annotations

import pytest

from rag.embedders import HashingEmbedder
from rag.indexers import FaissFlatIndexer
from rag.registry import build_reranker, build_retriever, retriever_k
from rag.rerankers import CrossEncoderReranker, NoopReranker
from rag.retrievers import DenseRetriever
from rag.types import Chunk


def _index() -> FaissFlatIndexer:
    embedder = HashingEmbedder(dimension=64)
    texts = {
        "c0": "the capital of france is paris",
        "c1": "photosynthesis converts light into chemical energy",
        "c2": "the eiffel tower stands in paris france",
    }
    vectors = embedder.embed(list(texts.values()))
    chunks = [
        Chunk(id=cid, text=text, metadata={"source": f"{cid}.md"}, embedding=vector)
        for (cid, text), vector in zip(texts.items(), vectors, strict=True)
    ]
    return FaissFlatIndexer().build(chunks)


# --- dense retriever ---------------------------------------------------------


def test_dense_retriever_returns_best_match_first() -> None:
    retriever = DenseRetriever(_index(), HashingEmbedder(dimension=64))
    results = retriever.retrieve("where is the eiffel tower in paris", k=3)
    assert results[0].id == "c2"
    assert results[0].metadata == {"source": "c2.md"}
    scores = [c.score for c in results]
    assert scores == sorted(scores, reverse=True)


def test_dense_retriever_respects_k() -> None:
    retriever = DenseRetriever(_index(), HashingEmbedder(dimension=64))
    assert len(retriever.retrieve("paris", k=2)) == 2


def test_dense_retriever_rejects_empty_query() -> None:
    retriever = DenseRetriever(_index(), HashingEmbedder(dimension=64))
    with pytest.raises(ValueError, match="query"):
        retriever.retrieve("   ", k=3)


def test_dense_retriever_rejects_bad_k() -> None:
    retriever = DenseRetriever(_index(), HashingEmbedder(dimension=64))
    with pytest.raises(ValueError, match="k must be"):
        retriever.retrieve("paris", k=0)


# --- noop reranker -----------------------------------------------------------


def _candidates() -> list[Chunk]:
    return [
        Chunk(id=f"c{i}", text=f"text {i}", metadata={}, score=1.0 - i * 0.1)
        for i in range(5)
    ]


def test_noop_reranker_preserves_order() -> None:
    chunks = _candidates()
    out = NoopReranker().rerank("q", chunks)
    assert [c.id for c in out] == [c.id for c in chunks]


def test_noop_reranker_truncates_to_top_n() -> None:
    out = NoopReranker(top_n=2).rerank("q", _candidates())
    assert [c.id for c in out] == ["c0", "c1"]


def test_noop_reranker_rejects_bad_top_n() -> None:
    with pytest.raises(ValueError, match="top_n"):
        NoopReranker(top_n=0)


# --- cross-encoder reranker (model stubbed) ----------------------------------


class _FakeCrossEncoder:
    """Scores a pair by whether the query token appears in the chunk text."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(self, pairs: list[list[str]], batch_size: int = 32) -> list[float]:
        return [10.0 if query in text else -10.0 for query, text in pairs]


def test_cross_encoder_reorders_and_truncates(monkeypatch: pytest.MonkeyPatch) -> None:
    reranker = CrossEncoderReranker(top_n=2)
    monkeypatch.setattr(reranker, "_load_model", lambda: _FakeCrossEncoder("fake"))

    chunks = [
        Chunk(id="a", text="nothing relevant", metadata={}, score=0.9),
        Chunk(id="b", text="contains paris here", metadata={}, score=0.1),
        Chunk(id="c", text="also paris mentioned", metadata={}, score=0.2),
        Chunk(id="d", text="unrelated content", metadata={}, score=0.8),
    ]
    out = reranker.rerank("paris", chunks)

    # The two paris-bearing chunks rise to the top despite low retriever scores...
    assert {c.id for c in out} == {"b", "c"}
    # ...and the retriever score is overwritten by the cross-encoder score.
    assert all(c.score == 10.0 for c in out)


def test_cross_encoder_empty_candidates_short_circuits() -> None:
    # No model load should be triggered for an empty candidate list.
    assert CrossEncoderReranker().rerank("paris", []) == []


# --- registry wiring ---------------------------------------------------------


def test_build_retriever_from_config() -> None:
    config = {"retriever": {"name": "dense", "k": 50}}
    retriever = build_retriever(_index(), HashingEmbedder(dimension=64), config)
    assert isinstance(retriever, DenseRetriever)


def test_build_reranker_from_config() -> None:
    config = {"reranker": {"name": "cross_encoder", "top_n": 5}}
    reranker = build_reranker(config)
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.top_n == 5


def test_build_reranker_noop() -> None:
    assert isinstance(build_reranker({"reranker": {"name": "noop"}}), NoopReranker)


def test_retriever_k_reads_config_and_defaults() -> None:
    assert retriever_k({"retriever": {"name": "dense", "k": 42}}) == 42
    assert retriever_k({"retriever": {"name": "dense"}}, default=7) == 7


def test_retriever_k_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="retriever.k"):
        retriever_k({"retriever": {"k": 0}})


def test_build_retriever_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown retriever"):
        build_retriever(
            _index(), HashingEmbedder(dimension=64), {"retriever": {"name": "x"}}
        )


def test_build_reranker_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown reranker"):
        build_reranker({"reranker": {"name": "x"}})
