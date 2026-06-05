"""Tests for citation parsing in the agentic-RAG runner (scripts/run_agent_experiments.py).

The runner is a script, not a package module, so we load it by path. We exercise only the
pure citation parser, which decides what counts toward recall@k / precision@n.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_agent_experiments.py"


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("agent_runner", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_citation(runner):
    assert runner._cited_passage_ids("Killed in 1865 [source: passage 334].") == [334]


def test_multi_id_one_citation(runner):
    # "passage 12, 13" must yield BOTH ids, not just the first.
    assert runner._cited_passage_ids("Per [source: passage 12, 13].") == [12, 13]


def test_paren_and_no_brackets(runner):
    assert runner._cited_passage_ids("X (source: passage 7).") == [7]
    assert runner._cited_passage_ids("source: passage 88") == [88]


def test_multiple_citations_ordered_deduped(runner):
    answer = "A [source: passage 5]. B [source: passage 2]. Again [source: passage 5]."
    assert runner._cited_passage_ids(answer) == [5, 2]


def test_prose_mention_is_not_a_citation(runner):
    # A bare prose mention must NOT count — only explicit source citations do.
    assert runner._cited_passage_ids("As described in passage 5, the war ended.") == []


def test_no_citation(runner):
    assert runner._cited_passage_ids("I don't know.") == []
