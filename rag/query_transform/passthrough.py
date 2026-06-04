"""Identity query transform — the honest baseline.

Returns the query unchanged, so a pipeline with this stage retrieves *exactly* what it
would with no query-transform stage at all (ARCHITECTURE.md §3). This is the default and
the control any rewriting strategy is measured against.
"""

from __future__ import annotations


class PassthroughTransform:
    """Return the query unchanged."""

    name = "passthrough"

    def transform(self, query: str) -> str:
        """Return ``query`` verbatim."""
        return query
