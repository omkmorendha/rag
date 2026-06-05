"""Query-transform interfaces.

A query transform rewrites the query *string* before retrieval (ARCHITECTURE.md §3). It
sits in front of the retriever: the transformed query is what gets embedded and searched,
but the **generator answers the original** user question. The harness threads both strings
through so the rewrite only ever influences which chunks are fetched, never how they are
answered.

This is the honest place to experiment with query rewriting, HyDE-style expansion, and
step-back prompting: swap the strategy, re-run the eval, read the retrieval delta.

List[str] fan-out (decomposing one question into several sub-queries) is deliberately
deferred — every transform here maps one query string to exactly one query string.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class QueryTransform(Protocol):
    """Rewrite a query string before retrieval."""

    name: str

    def transform(self, query: str) -> str:
        """Return the query to *retrieve* with (the generator still sees the original)."""
        ...
