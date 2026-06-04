"""Tests for the query_transform stage (no live API)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.query_transform import (
    PassthroughTransform,
    QueryTransform,
    RewriteTransform,
    StepBackTransform,
)
from rag.query_transform.rewrite import _parse_response
from rag.registry import build_query_transform


# --- passthrough baseline ----------------------------------------------------


def test_passthrough_returns_query_unchanged() -> None:
    transform = PassthroughTransform()
    assert transform.transform("Who freed the slaves?") == "Who freed the slaves?"
    assert transform.transform("") == ""


def test_passthrough_satisfies_protocol() -> None:
    assert isinstance(PassthroughTransform(), QueryTransform)
    assert isinstance(RewriteTransform(), QueryTransform)
    assert isinstance(StepBackTransform(), QueryTransform)


# --- registry ----------------------------------------------------------------


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_build_query_transform_defaults_to_passthrough(tmp_path: Path) -> None:
    # A config with no query_transform block must default to the identity baseline.
    path = _write_config(tmp_path, "chunker:\n  name: recursive\n")
    transform = build_query_transform(path=path)
    assert isinstance(transform, PassthroughTransform)
    assert transform.name == "passthrough"


def test_build_query_transform_named(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "query_transform:\n  name: rewrite\n")
    transform = build_query_transform(path=path)
    assert isinstance(transform, RewriteTransform)


def test_build_query_transform_unknown_name_raises(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "query_transform:\n  name: nope\n")
    with pytest.raises(ValueError, match="unknown query_transform"):
        build_query_transform(path=path)


# --- rewrite/step_back parse + fallback (no API) -----------------------------


class _Block:
    def __init__(self, text: str, type: str = "text") -> None:
        self.text = text
        self.type = type


class _Response:
    def __init__(self, blocks: list[_Block]) -> None:
        self.content = blocks


def test_parse_response_returns_stripped_text() -> None:
    response = _Response([_Block("  Abraham Lincoln assassination  ")])
    assert _parse_response(response, fallback="orig") == "Abraham Lincoln assassination"


def test_parse_response_falls_back_on_empty() -> None:
    assert _parse_response(_Response([]), fallback="orig") == "orig"
    assert _parse_response(_Response([_Block("   ")]), fallback="orig") == "orig"


def test_parse_response_ignores_non_text_blocks() -> None:
    response = _Response([_Block("ignored", type="thinking"), _Block("kept")])
    assert _parse_response(response, fallback="orig") == "kept"


class _FakeClient:
    """Records the create() call and returns a canned response."""

    def __init__(self, response: _Response) -> None:
        self._response = response
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs: object) -> _Response:
        self.calls.append(kwargs)
        return self._response


def test_rewrite_falls_back_to_original_on_empty_response() -> None:
    transform = RewriteTransform()
    transform._client = _FakeClient(_Response([_Block("   ")]))
    assert transform.transform("who?") == "who?"


def test_step_back_returns_model_output_when_present() -> None:
    transform = StepBackTransform()
    transform._client = _FakeClient(_Response([_Block("What is the biography of X?")]))
    assert transform.transform("When was X born?") == "What is the biography of X?"


def test_rewrite_sends_query_as_user_message_with_cached_system() -> None:
    client = _FakeClient(_Response([_Block("rewritten")]))
    transform = RewriteTransform()
    transform._client = client
    transform.transform("orig query")
    call = client.calls[0]
    assert call["messages"] == [{"role": "user", "content": "orig query"}]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_transforms_reject_bad_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        RewriteTransform(max_tokens=0)
    with pytest.raises(ValueError, match="max_tokens"):
        StepBackTransform(max_tokens=0)
