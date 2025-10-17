"""Utilities for extracting text from PDF documents."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(path: str | Path) -> str:
    """Extract textual content from a PDF file.

    Args:
        path: Path to the PDF document.

    Returns:
        The concatenated text from all pages in the document.
    """

    pdf_path = Path(path)
    reader = PdfReader(pdf_path)
    text_segments = [page.extract_text() or "" for page in reader.pages]
    cleaned_segments = [
        segment.strip() for segment in text_segments if segment is not None
    ]
    return "\n".join(cleaned_segments)


def iter_pdf_texts(directory: str | Path) -> Iterator[tuple[Path, str]]:
    """Yield `(path, text)` pairs for all PDF files under a directory."""

    root = Path(directory)
    for pdf_path in sorted(root.rglob("*.pdf")):
        yield pdf_path, extract_text_from_pdf(pdf_path)


__all__ = ["extract_text_from_pdf", "iter_pdf_texts"]
