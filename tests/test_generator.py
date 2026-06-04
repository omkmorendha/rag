"""Tests for the Anthropic grounded generator."""

from __future__ import annotations

import pytest

from rag.generator import AnthropicGenerator, Prompt
from rag.generator.anthropic import SYSTEM_PROMPT, render_chunks
from rag.registry import build_generator
from rag.types import Chunk


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id="passage:344:recursive:0",
            text="McClellan led at Antietam.",
            metadata={"passage_id": 344},
            score=0.9,
        ),
        Chunk(
            id="passage:9:recursive:0",
            text="Unrelated text.",
            metadata={"passage_id": 9},
            score=0.1,
        ),
    ]


# --- prompt building (no API) ------------------------------------------------


def test_build_prompt_uses_grounding_system_prompt() -> None:
    prompt = AnthropicGenerator().build_prompt("Who led at Antietam?", _chunks())
    assert isinstance(prompt, Prompt)
    assert prompt.system == SYSTEM_PROMPT
    # The grounding rules the user specified are present verbatim.
    assert 'strictly state that "I don\'t know"' in prompt.system
    assert "always prefer the chunk shown first" in prompt.system


def test_user_message_carries_query_and_chunks() -> None:
    prompt = AnthropicGenerator().build_prompt("Who led at Antietam?", _chunks())
    assert "<user_query>\nWho led at Antietam?\n</user_query>" in prompt.user
    assert "<retrieved_chunks>" in prompt.user
    assert "McClellan led at Antietam." in prompt.user


def test_render_chunks_numbers_and_labels_sources() -> None:
    rendered = render_chunks(_chunks())
    assert "<chunk n=1>" in rendered
    assert "<chunk n=2>" in rendered
    # passage_id becomes the citable source token matching the prompt's example format.
    assert "<metadata>source: passage 344</metadata>" in rendered
    # First chunk renders first (chunks arrive best-first → "shown first" rule holds).
    assert rendered.index("passage 344") < rendered.index("passage 9")


def test_render_chunks_falls_back_when_no_passage_id() -> None:
    chunk = Chunk(id="doc:1", text="hi", metadata={"source_path": "/a/b.md"})
    assert "source: /a/b.md" in render_chunks([chunk])


# --- generate() with a stubbed client ----------------------------------------


class _FakeBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, recorder: dict) -> None:
        self._recorder = recorder

    def create(self, **kwargs):
        self._recorder.update(kwargs)
        return _FakeResponse("McClellan [source: passage 344]")


class _FakeClient:
    def __init__(self, recorder: dict) -> None:
        self.messages = _FakeMessages(recorder)


def test_generate_returns_text_and_sends_cached_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder: dict = {}
    gen = AnthropicGenerator(model="claude-haiku-4-5")
    monkeypatch.setattr(gen, "_load_client", lambda: _FakeClient(recorder))

    answer = gen.generate("Who led at Antietam?", _chunks())

    assert answer == "McClellan [source: passage 344]"
    assert recorder["model"] == "claude-haiku-4-5"
    # System prompt is sent as a cached block.
    assert recorder["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert recorder["system"][0]["text"] == SYSTEM_PROMPT
    # Effort is NOT sent — Haiku doesn't support it.
    assert "output_config" not in recorder
    assert recorder["messages"][0]["role"] == "user"


def test_generate_concatenates_only_text_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Mixed:
        content = [
            _FakeBlock("part one "),
            type("B", (), {"type": "thinking"})(),
            _FakeBlock("part two"),
        ]

    gen = AnthropicGenerator()
    monkeypatch.setattr(
        gen,
        "_load_client",
        lambda: type(
            "C",
            (),
            {"messages": type("M", (), {"create": lambda self, **k: _Mixed()})()},
        )(),
    )
    assert gen.generate("q", _chunks()) == "part one part two"


# --- registry wiring ----------------------------------------------------------


def test_build_generator_from_config() -> None:
    gen = build_generator(
        {"generator": {"name": "anthropic", "model": "claude-haiku-4-5"}}
    )
    assert isinstance(gen, AnthropicGenerator)
    assert gen.model == "claude-haiku-4-5"


def test_build_generator_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown generator"):
        build_generator({"generator": {"name": "nope"}})


def test_generator_rejects_bad_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        AnthropicGenerator(max_tokens=0)
