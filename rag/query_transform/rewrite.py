"""LLM query rewriting for better dense retrieval.

Asks Claude to rewrite the user's question into a single, retrieval-optimized search query
— expanding abbreviations and adding key entities/synonyms — before it is embedded. The
system prompt is static, so it is sent as a cached block (prompt caching); only the
per-query user message varies (mirrors :class:`rag.generator.anthropic.AnthropicGenerator`).

A failed rewrite must never crash a sweep: if the model returns empty/whitespace, the
transform falls back to the original query. The retrieved chunks are still answered against
the *original* question by the generator (see the harness), so a poor rewrite only costs
retrieval quality, never correctness of the grounding contract.
"""

from __future__ import annotations

from typing import Any

SYSTEM_PROMPT = """\
<purpose>You rewrite a user's question into a single search query optimized for dense \
(embedding-based) retrieval over a knowledge corpus.</purpose>

<rewrite_criteria>
1. Expand abbreviations and acronyms to their full forms.
2. Add the key named entities, synonyms, and closely related terms that help retrieval.
3. Keep it as ONE self-contained search query — do not split it into multiple questions.
4. Preserve the original intent; do not answer the question.
5. Output ONLY the rewritten query, with no preamble, quotes, or explanation.
</rewrite_criteria>"""


def _parse_response(response: Any, *, fallback: str) -> str:
    """Extract the rewritten query text, falling back to ``fallback`` when empty.

    Pulled out so the parse/fallback contract is unit-testable without a network call.
    """
    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    return text or fallback


class RewriteTransform:
    """Rewrite the query for dense retrieval via the Anthropic Messages API."""

    name = "rewrite"

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
        """Rewrite ``query`` for retrieval, returning the original on any failure.

        A provider/network error (or an empty response) must never abort a sweep, so
        client creation and the API call are guarded and fall back to the input query.
        """
        try:
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
        except Exception:
            return query
        return _parse_response(response, fallback=query)

    def _load_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client
