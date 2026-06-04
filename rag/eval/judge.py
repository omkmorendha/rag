"""LLM-as-judge for generation scoring (opt-in, ARCHITECTURE.md §5).

Two scores per answer, judged by Claude against the chunks and the expected answer:

- **faithfulness** — is the answer grounded in the retrieved chunks (no fabrication)?
- **answer_relevance** — does it actually answer the query, and agree with the expected
  answer (allowing paraphrase)?

This is the only generation metric that survives paraphrase — the substring check in
``metrics.py`` cannot. It needs the API, so the harness keeps it behind ``--judge``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rag.types import Chunk

JUDGE_SYSTEM = """\
You are a strict evaluator of a retrieval-augmented answer. You are given a question, the \
retrieved context chunks, the expected answer, and the generated answer. Score two things \
on a 0.0-1.0 scale:

- faithfulness: is every claim in the generated answer supported by the retrieved chunks? \
1.0 = fully grounded, 0.0 = fabricated or contradicted by the chunks. An honest "I don't \
know" when the chunks lack the answer is faithful (1.0).
- answer_relevance: does the generated answer correctly answer the question, agreeing with \
the expected answer? Accept paraphrases and extra correct detail. 1.0 = correct, 0.0 = \
wrong or non-responsive. A correct "I don't know" when the expected answer is also absent \
is relevant (1.0); an "I don't know" when the answer was available is not (0.0).

Respond with ONLY a JSON object: {"faithfulness": <float>, "answer_relevance": <float>}"""


@dataclass(frozen=True)
class JudgeScore:
    faithfulness: float
    answer_relevance: float


class AnthropicJudge:
    """Score a generated answer with Claude."""

    def __init__(
        self, model: str = "claude-haiku-4-5", *, max_tokens: int = 256
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client: Any | None = None

    def score(
        self,
        query: str,
        chunks: list[Chunk],
        expected_answer: str,
        answer: str,
    ) -> JudgeScore:
        """Judge one answer; returns faithfulness + answer_relevance in [0, 1]."""
        context = "\n".join(f"- {c.text}" for c in chunks)
        user = (
            f"Question:\n{query}\n\n"
            f"Retrieved chunks:\n{context}\n\n"
            f"Expected answer:\n{expected_answer}\n\n"
            f"Generated answer:\n{answer}"
        )
        client = self._load_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": JUDGE_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        return _parse_score(text)

    def _load_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client


def _parse_score(text: str) -> JudgeScore:
    """Extract the JSON score object from the judge's reply, clamped to [0, 1]."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"judge did not return a JSON object: {text!r}")
    data = json.loads(text[start : end + 1])
    return JudgeScore(
        faithfulness=_clamp(float(data["faithfulness"])),
        answer_relevance=_clamp(float(data["answer_relevance"])),
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
