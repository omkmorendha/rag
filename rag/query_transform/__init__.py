"""Query transforms: run BEFORE retrieval to fight query-document asymmetry.

`passthrough` is the default/first-slice no-op. step_back / hyde / decomposition
(ARCHITECTURE.md §3) are added later behind eval.
"""

from __future__ import annotations


class QueryTransform:
    """Interface: transform a query into one (or, later, several) queries."""

    def transform(self, query: str) -> str:
        raise NotImplementedError


class Passthrough(QueryTransform):
    """Return the query unchanged."""

    def transform(self, query: str) -> str:
        return query
