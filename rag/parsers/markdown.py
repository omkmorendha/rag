"""Markdown parsers."""

from __future__ import annotations

import re
from pathlib import Path

from rag.types import Document

PASSAGE_HEADING = re.compile(r"^## passage (?P<id>\d+)\s*$")


class MarkdownPassageParser:
    """Parse grouped rag-mini-wikipedia Markdown into one document per passage."""

    name = "markdown_passage"

    def parse(self, path: Path) -> list[Document]:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()

        group_title: str | None = None
        documents: list[Document] = []

        current_passage_id: int | None = None
        current_start_line: int | None = None
        current_lines: list[str] = []

        for line_number, line in enumerate(lines, start=1):
            if group_title is None and line.startswith("# "):
                group_title = line.removeprefix("# ").strip()
                continue

            match = PASSAGE_HEADING.match(line.strip())
            if match:
                if current_passage_id is not None and current_start_line is not None:
                    documents.append(
                        self._make_document(
                            path=path,
                            group_title=group_title,
                            passage_id=current_passage_id,
                            start_line=current_start_line,
                            lines=current_lines,
                        )
                    )

                current_passage_id = int(match.group("id"))
                current_start_line = line_number
                current_lines = []
                continue

            if current_passage_id is not None:
                current_lines.append(line)

        if current_passage_id is not None and current_start_line is not None:
            documents.append(
                self._make_document(
                    path=path,
                    group_title=group_title,
                    passage_id=current_passage_id,
                    start_line=current_start_line,
                    lines=current_lines,
                )
            )

        return documents

    def _make_document(
        self,
        *,
        path: Path,
        group_title: str | None,
        passage_id: int,
        start_line: int,
        lines: list[str],
    ) -> Document:
        body = "\n".join(lines).strip()
        return Document(
            id=f"passage:{passage_id}",
            text=body,
            metadata={
                "source_path": str(path),
                "doc_type": "markdown",
                "parser": self.name,
                "group_title": group_title,
                "passage_id": passage_id,
                "start_line": start_line,
            },
        )

