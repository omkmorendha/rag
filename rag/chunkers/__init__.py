"""Chunkers: Document -> list[Chunk].

The dual mandate (ARCHITECTURE.md §3): small enough for a precise embedding signal, large
enough to carry usable context.
"""

from __future__ import annotations

import re

from rag.types import Chunk, Document


class Chunker:
    """Interface: split a Document into Chunks."""

    def chunk(self, doc: Document) -> list[Chunk]:
        raise NotImplementedError


# Split priority for the recursive chunker: try to break on the largest structural unit
# first (a `## passage N` section), then blank-line paragraphs, then sentences, then words.
_HEADING_RE = re.compile(r"^##\s+passage\s+(\d+)\s*$", re.MULTILINE)


class RecursiveChunker(Chunker):
    """Split on document structure, largest unit first (heading -> paragraph -> sentence).

    The corpus is one `## passage {id}` section per source passage. We split on those
    headings so each chunk maps back to a passage id (captured into metadata as
    `passage_id` — the anchor `derive_golden.py` needs). Sections longer than `size`
    characters are further split on paragraph/sentence boundaries with `overlap` carried
    between adjacent windows.
    """

    def __init__(self, size: int = 512, overlap: int = 64):
        self.size = size
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for passage_id, body in self._iter_sections(doc.text):
            for i, window in enumerate(self._split_to_size(body)):
                chunk_id = f"{doc.id}::p{passage_id}::{i}"
                meta = dict(doc.metadata)
                meta.update({"passage_id": passage_id, "chunk_index": i})
                chunks.append(Chunk(id=chunk_id, text=window, metadata=meta))
        return chunks

    def _iter_sections(self, text: str):
        """Yield (passage_id, section_body) for each `## passage N` section."""
        matches = list(_HEADING_RE.finditer(text))
        for j, m in enumerate(matches):
            passage_id = int(m.group(1))
            start = m.end()
            end = matches[j + 1].start() if j + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                yield passage_id, body

    def _split_to_size(self, text: str) -> list[str]:
        """Recursively split text that exceeds `size`, breaking on the largest unit that
        helps: paragraphs, then sentences, then words. Adjacent windows share `overlap`
        characters so context is not severed at a boundary."""
        if len(text) <= self.size:
            return [text]
        for separator in ("\n\n", ". ", " "):
            parts = text.split(separator)
            if len(parts) == 1:
                continue
            return self._pack(parts, separator)
        # No separator helped (one giant token): hard-slice with overlap.
        return self._slice_with_overlap(text)

    def _pack(self, parts: list[str], sep: str) -> list[str]:
        """Greedily pack parts into <= size windows, recursing into any oversized part."""
        windows: list[str] = []
        buf = ""
        for part in parts:
            candidate = part if not buf else buf + sep + part
            if len(candidate) <= self.size:
                buf = candidate
                continue
            if buf:
                windows.append(buf)
            # Start the next window with an overlap tail from the previous one.
            tail = buf[-self.overlap :] if buf and self.overlap else ""
            if len(part) > self.size:
                # This single part is itself too big; recurse to split it further.
                sub = self._split_to_size(part)
                if tail and sub:
                    sub[0] = tail + sep + sub[0]
                windows.extend(sub)
                buf = ""
            else:
                buf = (tail + sep + part) if tail else part
        if buf:
            windows.append(buf)
        return windows

    def _slice_with_overlap(self, text: str) -> list[str]:
        step = max(1, self.size - self.overlap)
        return [text[i : i + self.size] for i in range(0, len(text), step)]
