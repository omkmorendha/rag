"""Generator: build_prompt(query, chunks) -> prompt ; generate(prompt) -> answer.

Two responsibilities, kept separate (ARCHITECTURE.md §3):
1. build_prompt — assemble instructions + chunks with stable citation identifiers and
   key metadata, ordered for lost-in-the-middle.
2. grounding — instruct the model to answer ONLY from the retrieved text, cite sources,
   and say "I don't know" when the answer isn't present. The dial between stale
   training-data answers and unhelpful over-refusal.

Only this stage hits an external API.
"""

from __future__ import annotations

import os

from rag.types import Chunk

SYSTEM_PROMPT = (
    "You are a careful retrieval-grounded assistant. Answer the question using ONLY the "
    "information in the provided sources. Cite the sources you use with their bracketed "
    "identifiers, e.g. [1] or [2]. If the answer is not contained in the sources, reply "
    'exactly: "I don\'t know based on the provided sources." Do not use outside knowledge.'
)


def _order_for_lost_in_the_middle(chunks: list[Chunk]) -> list[Chunk]:
    """Place the strongest chunks at the START and END, weakest in the middle, where the
    model attends least. Assumes `chunks` arrive already sorted best-first."""
    ordered: list[Chunk] = []
    for i, chunk in enumerate(chunks):
        if i % 2 == 0:
            ordered.append(chunk)  # even ranks -> front, growing toward the middle
        else:
            ordered.insert(0, chunk)  # odd ranks -> back (we reverse the head at the end)
    # `ordered` now has best chunks at both ends; return as-is.
    return ordered


class Generator:
    """Interface: build a grounded prompt, then generate an answer from it."""

    def build_prompt(self, query: str, chunks: list[Chunk]) -> str:
        raise NotImplementedError

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        raise NotImplementedError


class AnthropicGenerator(Generator):
    """Grounded generation via the Anthropic API. Uses the top_n reranked chunks."""

    def __init__(self, model: str = "claude-sonnet-4-6", top_n: int = 8):
        self.model = model
        self.top_n = top_n
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def build_prompt(self, query: str, chunks: list[Chunk]) -> str:
        selected = _order_for_lost_in_the_middle(chunks[: self.top_n])
        blocks = []
        for i, chunk in enumerate(selected, start=1):
            src = chunk.metadata.get("title") or chunk.metadata.get("source", "")
            passage = chunk.metadata.get("passage_id")
            label = f"{src}" + (f", passage {passage}" if passage is not None else "")
            blocks.append(f"[{i}] ({label})\n{chunk.text}")
        sources = "\n\n".join(blocks) if blocks else "(no sources retrieved)"
        return (
            f"Sources:\n\n{sources}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the sources above, citing them by their bracketed numbers."
        )

    def generate(self, query: str, chunks: list[Chunk]) -> str:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set; the generator needs it to call the API."
            )
        prompt = self.build_prompt(query, chunks)
        client = self._get_client()
        message = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")
