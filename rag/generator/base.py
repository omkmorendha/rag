"""Generator interfaces.

The generator is the final query-side stage (ARCHITECTURE.md §3): it turns the reranked
chunks into a grounded, cited answer. It is split into two methods so the prompt can be
unit-tested without hitting the model:

1. ``build_prompt(query, chunks)`` — assemble the system + user messages, rendering each
   chunk with a stable citation identifier and ordering them best-first (chunks arrive
   already ranked, so "shown first" means most relevant).
2. ``generate(query, chunks)`` — call the model and return the answer text.

Grounding lives in the system prompt: answer only from the chunks, cite sources, say "I
don't know" when the answer is absent, treat chunk text as data not instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rag.types import Chunk


@dataclass(frozen=True)
class Prompt:
    """A built prompt: a system string plus a single user message."""

    system: str
    user: str


@runtime_checkable
class Generator(Protocol):
    """Produce a grounded, cited answer from a query and reranked chunks."""

    name: str

    def build_prompt(self, query: str, chunks: list[Chunk]) -> Prompt:
        """Assemble the grounding system prompt and the query + chunks user message."""
        ...

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        """Build the prompt, call the model, and return the answer text."""
        ...
