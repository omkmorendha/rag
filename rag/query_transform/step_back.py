"""Step-back query transform — generalize the question to retrieve background.

Asks Claude to turn a specific question into a broader "step-back" question whose answer is
the general principle or background the original needs (e.g. "What year was X born?" ->
"What is the biography of X?"). The broader query often retrieves the context the narrow
one misses. The system prompt is static and cached (prompt caching), mirroring
:class:`rag.generator.anthropic.AnthropicGenerator`.

As with :mod:`rag.query_transform.rewrite`, an empty/whitespace response falls back to the
original query so a failed transform never crashes a sweep, and the generator still answers
the *original* question.
"""

from __future__ import annotations

from typing import Any

from rag.query_transform.rewrite import _parse_response

SYSTEM_PROMPT = """\
<purpose>You take a specific user question and produce a broader "step-back" question whose \
answer supplies the general background, principle, or context needed to answer the \
original.</purpose>

<step_back_criteria>
1. Generalize the question: replace specific details with the broader topic or concept they
   belong to.
2. The step-back question should retrieve useful background, not the narrow fact directly.
3. Keep it as ONE self-contained question — do not split it into multiple questions.
4. Do not answer either question.
5. Output ONLY the step-back question, with no preamble, quotes, or explanation.
</step_back_criteria>"""


class StepBackTransform:
    """Generalize the query into a step-back question via the Anthropic Messages API."""

    name = "step_back"

    def __init__(
        self,
        model: str = "claude-haiku-4-5",
        *,
        max_tokens: int = 256,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        self.model = model
        self.max_tokens = max_tokens
        self._client: Any | None = None

    def transform(self, query: str) -> str:
        """Generalize ``query``, returning the original on an empty response."""
        client = self._load_client()
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": query}],
        )
        return _parse_response(response, fallback=query)

    def _load_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client
