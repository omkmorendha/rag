"""Tests for the faiss_flat indexer stage."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.embedders import HashingEmbedder
from rag.indexers import FaissFlatIndexer, IndexMismatchError
from rag.registry import build_indexer
from rag.types import Chunk


def _embedded_chunks() -> list[Chunk]:
    """Three distinct chunks embedded with the deterministic hashing embedder."""
    embedder = HashingEmbedder(dimension=64)
    texts = {
        "c0": "the capital of france is paris",
        "c1": "photosynthesis converts light into chemical energy",
        "c2": "the eiffel tower stands in paris france",
    }
    vectors = embedder.embed(list(texts.values()))
    return [
        Chunk(id=cid, text=text, metadata={"source": f"{cid}.md"}, embedding=vector)
        for (cid, text), vector in zip(texts.items(), vectors, strict=True)
    ]


def test_build_records_provenance_metadata() -> None:
    index = FaissFlatIndexer(embedder_model="hashing-64").build(_embedded_chunks())
    assert index.meta == {
        "kind": "faiss_flat",
        "metric": "inner_product",
        "dim": 64,
        "n_chunks": 3,
        "embedder_model": "hashing-64",
    }


def test_search_returns_chunks_with_scores_ranked() -> None:
    chunks = _embedded_chunks()
    index = FaissFlatIndexer().build(chunks)

    # Query identical to a stored chunk must return that chunk first with score ~1.0.
    query = HashingEmbedder(dimension=64).embed([chunks[0].text])[0]
    results = index.search(query, k=3)

    assert results[0].id == "c0"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[0].text == chunks[0].text
    assert results[0].metadata == {"source": "c0.md"}
    # Scores are sorted descending (similarity, not distance).
    scores = [c.score for c in results]
    assert scores == sorted(scores, reverse=True)


def test_search_caps_k_at_corpus_size() -> None:
    index = FaissFlatIndexer().build(_embedded_chunks())
    query = HashingEmbedder(dimension=64).embed(["paris"])[0]
    assert len(index.search(query, k=100)) == 3


def test_round_trip_save_load(tmp_path: Path) -> None:
    chunks = _embedded_chunks()
    indexer = FaissFlatIndexer(embedder_model="hashing-64")
    indexer.build(chunks).save(tmp_path)

    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "chunks.jsonl").exists()
    assert (tmp_path / "meta.json").exists()

    loaded = FaissFlatIndexer(embedder_model="hashing-64").load(tmp_path)
    query = HashingEmbedder(dimension=64).embed([chunks[2].text])[0]
    results = loaded.search(query, k=1)
    assert results[0].id == "c2"
    assert results[0].text == chunks[2].text


def test_load_rejects_embedder_mismatch(tmp_path: Path) -> None:
    FaissFlatIndexer(embedder_model="model-a").build(_embedded_chunks()).save(tmp_path)
    with pytest.raises(IndexMismatchError, match="model-a"):
        FaissFlatIndexer(embedder_model="model-b").load(tmp_path)


def test_load_allows_match_and_unknown_model(tmp_path: Path) -> None:
    FaissFlatIndexer(embedder_model="model-a").build(_embedded_chunks()).save(tmp_path)
    # A loader that does not assert a model (None) accepts any index.
    loaded = FaissFlatIndexer(embedder_model=None).load(tmp_path)
    assert loaded.meta["embedder_model"] == "model-a"


def test_build_rejects_missing_embeddings() -> None:
    chunks = [Chunk(id="c0", text="hi", metadata={})]  # no embedding
    with pytest.raises(ValueError, match="no embedding"):
        FaissFlatIndexer().build(chunks)


def test_build_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="zero chunks"):
        FaissFlatIndexer().build([])


def test_search_rejects_wrong_query_dimension() -> None:
    index = FaissFlatIndexer().build(_embedded_chunks())
    with pytest.raises(ValueError, match="dimension"):
        index.search([0.1, 0.2, 0.3], k=1)


def test_build_indexer_threads_embedder_model_from_config() -> None:
    config = {
        "indexer": {"name": "faiss_flat"},
        "embedder": {"name": "local", "model": "BAAI/bge-small-en-v1.5"},
    }
    indexer = build_indexer(config)
    assert isinstance(indexer, FaissFlatIndexer)
    assert indexer.embedder_model == "BAAI/bge-small-en-v1.5"


def test_build_indexer_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown indexer"):
        build_indexer({"indexer": {"name": "nope"}})
