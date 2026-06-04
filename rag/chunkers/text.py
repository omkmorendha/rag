"""Text chunking strategies."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from rag.types import Chunk, Document

TOKEN = re.compile(r"\S+")
SENTENCE = re.compile(r"(?<=[.!?])\s+")


class FixedTokenChunker:
    """Split text into fixed-size whitespace-token windows."""

    name = "fixed"

    def __init__(self, size: int = 256, overlap: int = 32) -> None:
        _validate_window(size=size, overlap=overlap)
        self.size = size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        tokens = _token_spans(document.text)
        windows = _window_spans(tokens, size=self.size, overlap=self.overlap)
        return _chunks_from_spans(
            document=document,
            spans=windows,
            chunker_name=self.name,
            config={"size": self.size, "overlap": self.overlap},
        )


class RecursiveTextChunker:
    """Split on paragraphs and sentences, falling back to token windows."""

    name = "recursive"

    def __init__(self, size: int = 256, overlap: int = 32) -> None:
        _validate_window(size=size, overlap=overlap)
        self.size = size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        units = _paragraph_sentence_units(document.text)
        spans: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end: int | None = None
        current_tokens = 0

        for start, end, token_count in units:
            if token_count > self.size:
                if current_start is not None and current_end is not None:
                    spans.append((current_start, current_end))
                    current_start = None
                    current_end = None
                    current_tokens = 0

                text_offset = start
                oversized_tokens = _token_spans(document.text[start:end])
                for window_start, window_end in _window_spans(
                    oversized_tokens,
                    size=self.size,
                    overlap=self.overlap,
                ):
                    spans.append((text_offset + window_start, text_offset + window_end))
                continue

            if current_start is None:
                current_start = start
                current_end = end
                current_tokens = token_count
                continue

            if current_tokens + token_count <= self.size:
                current_end = end
                current_tokens += token_count
                continue

            spans.append((current_start, current_end or start))
            overlap_text = _tail_tokens(
                document.text[current_start : current_end or start], self.overlap
            )
            if overlap_text:
                overlap_start = (current_end or start) - len(overlap_text)
                current_start = overlap_start
                current_tokens = len(_token_spans(overlap_text)) + token_count
            else:
                current_start = start
                current_tokens = token_count
            current_end = end

        if current_start is not None and current_end is not None:
            spans.append((current_start, current_end))

        return _chunks_from_spans(
            document=document,
            spans=_dedupe_spans(spans),
            chunker_name=self.name,
            config={"size": self.size, "overlap": self.overlap},
        )


class SentenceWindowChunker:
    """Split text into overlapping sentence windows."""

    name = "sentence_window"

    def __init__(self, sentence_count: int = 4, overlap: int = 1) -> None:
        _validate_window(size=sentence_count, overlap=overlap)
        self.sentence_count = sentence_count
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = _sentence_spans(document.text)
        step = self.sentence_count - self.overlap
        spans: list[tuple[int, int]] = []
        for start_index in range(0, len(sentences), step):
            window = sentences[start_index : start_index + self.sentence_count]
            if not window:
                continue
            spans.append((window[0][0], window[-1][1]))
            if start_index + self.sentence_count >= len(sentences):
                break

        return _chunks_from_spans(
            document=document,
            spans=spans,
            chunker_name=self.name,
            config={
                "sentence_count": self.sentence_count,
                "overlap": self.overlap,
            },
        )


def _validate_window(*, size: int, overlap: int) -> None:
    if size <= 0:
        raise ValueError("size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be 0 or greater")
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in TOKEN.finditer(text)]


def _window_spans(
    tokens: Sequence[tuple[int, int]],
    *,
    size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    if not tokens:
        return []

    windows: list[tuple[int, int]] = []
    step = size - overlap
    for start_index in range(0, len(tokens), step):
        window = tokens[start_index : start_index + size]
        windows.append((window[0][0], window[-1][1]))
        if start_index + size >= len(tokens):
            break
    return windows


def _paragraph_sentence_units(text: str) -> list[tuple[int, int, int]]:
    units: list[tuple[int, int, int]] = []
    for paragraph in re.finditer(r"\S(?:.*?\S)?(?=\n{2,}|\Z)", text, re.DOTALL):
        paragraph_text = paragraph.group(0)
        sentence_spans = _sentence_spans(paragraph_text)
        if len(sentence_spans) <= 1:
            units.append(
                (
                    paragraph.start(),
                    paragraph.end(),
                    len(_token_spans(paragraph_text)),
                )
            )
            continue

        for start, end in sentence_spans:
            sentence = paragraph_text[start:end]
            units.append(
                (
                    paragraph.start() + start,
                    paragraph.start() + end,
                    len(_token_spans(sentence)),
                )
            )
    return units


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    stripped = text.strip()
    if not stripped:
        return []

    offset = len(text) - len(text.lstrip())
    spans: list[tuple[int, int]] = []
    cursor = 0
    for part in SENTENCE.split(stripped):
        start = stripped.find(part, cursor)
        end = start + len(part)
        spans.append((offset + start, offset + end))
        cursor = end
    return spans


def _tail_tokens(text: str, count: int) -> str:
    if count <= 0:
        return ""
    tokens = _token_spans(text)
    if not tokens:
        return ""
    start = tokens[max(0, len(tokens) - count)][0]
    return text[start:]


def _dedupe_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    deduped: list[tuple[int, int]] = []
    previous: tuple[int, int] | None = None
    for span in spans:
        if span == previous:
            continue
        deduped.append(span)
        previous = span
    return deduped


def _chunks_from_spans(
    *,
    document: Document,
    spans: Iterable[tuple[int, int]],
    chunker_name: str,
    config: dict[str, int],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for index, (start, end) in enumerate(spans):
        text = document.text[start:end].strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                id=f"{document.id}:{chunker_name}:{index}",
                text=text,
                metadata={
                    **document.metadata,
                    "document_id": document.id,
                    "parent_id": document.id,
                    "chunker": chunker_name,
                    "chunk_index": index,
                    "start_char": start,
                    "end_char": end,
                    "token_count": len(_token_spans(text)),
                    "chunker_config": config,
                },
            )
        )
    return chunks
