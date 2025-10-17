"""Bibliographic ingestion utilities compatible with CAADRIA workflows."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pypdf import PdfReader

_REQUIRED_PAPER_COLUMNS = ("paper_id", "title")
_REQUIRED_AUTHORSHIP_COLUMNS = ("paper_id", "author_id")

_KNOWN_STEMS: dict[str, Sequence[str]] = {
    "papers": ("papers", "papers_metadata", "metadata"),
    "authors": ("authors", "researchers"),
    "authorships": ("authorships", "paper_authors", "links"),
}

_SUPPORTED_SUFFIXES = (".csv", ".json", ".parquet")


def load_records(
    directory: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load bibliographic tables from a directory.

    The loader looks for ``papers``, ``authors``, and ``authorships`` tables using
    a small set of conventional filenames (``papers.csv``, ``authors.json``,
    ``authorships.parquet``). Columns are normalised to include ``paper_id`` and
    ``author_id`` identifiers required by downstream metrics.

    Args:
        directory: Folder containing the bibliographic exports.

    Returns:
        A tuple ``(papers, authors, authorships)`` of pandas ``DataFrame``
        objects.

    Raises:
        FileNotFoundError: If any of the required tables cannot be located.
    """

    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {base}")

    papers = _read_table(base, "papers")
    authors = _read_table(base, "authors")
    authorships = _read_table(base, "authorships")

    papers = _normalise_papers(papers)
    authors = _normalise_authors(authors)
    authorships = _normalise_authorships(authorships)

    return papers, authors, authorships


def pdf_text(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Extract page-level text from a collection of PDF files.

    Args:
        paths: Iterable of file paths pointing to PDF documents.

    Returns:
        DataFrame with columns ``paper_id``, ``page``, and ``text``.
    """

    records: list[dict[str, Any]] = []
    for path_like in paths:
        pdf_path = Path(path_like)
        reader = PdfReader(pdf_path)
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            records.append(
                {
                    "paper_id": pdf_path.stem,
                    "page": index,
                    "text": text.strip(),
                    "path": str(pdf_path),
                }
            )

    return pd.DataFrame.from_records(records)


def keyword_filter(
    papers: pd.DataFrame,
    keywords: Sequence[str],
    *,
    columns: Sequence[str] | None = None,
    mode: str = "substring",
) -> pd.DataFrame:
    """Filter papers containing any of the specified keywords.

    Args:
        papers: The papers DataFrame returned by :func:`load_records`.
        keywords: Keywords or phrases to match.
        columns: Optional text columns to inspect. Defaults to ``("title",
            "abstract", "keywords")`` intersected with available columns.
        mode: Matching strategy. ``"substring"`` performs a case-insensitive
            substring match, while ``"lemma"`` applies a lightweight lemmatisation
            heuristic for English words.

    Returns:
        A filtered DataFrame preserving the original column order.
    """

    if not keywords:
        return papers.iloc[0:0]

    search_columns = _resolve_columns(papers, columns)
    matcher = (
        _LemmaMatcher(keywords) if mode == "lemma" else _SubstringMatcher(keywords)
    )

    mask = papers.apply(lambda row: matcher.matches(row, search_columns), axis=1)
    result = papers.loc[mask].copy()
    return result.reset_index(drop=True)


def _read_table(base: Path, table: str) -> pd.DataFrame:
    for stem in _KNOWN_STEMS[table]:
        for suffix in _SUPPORTED_SUFFIXES:
            candidate = base / f"{stem}{suffix}"
            if candidate.exists():
                if suffix == ".csv":
                    return pd.read_csv(candidate)
                if suffix == ".json":
                    return pd.read_json(candidate)
                return pd.read_parquet(candidate)
    raise FileNotFoundError(f"No table found for '{table}' in {base}")


def _normalise_papers(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    rename_map = {
        "id": "paper_id",
        "paperID": "paper_id",
        "paper": "paper_id",
        "year": "year",
    }
    frame.rename(columns=rename_map, inplace=True)
    if "paper_id" not in frame.columns:
        frame.insert(0, "paper_id", frame.index.astype(str))
    for column in _REQUIRED_PAPER_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"Missing required paper column: {column}")
    if "year" in frame.columns:
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    if "venue" not in frame.columns:
        frame["venue"] = "Unknown"
    return frame


def _normalise_authors(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.rename(columns={"id": "author_id", "author": "name"}, inplace=True)
    if "author_id" not in frame.columns:
        frame.insert(0, "author_id", frame.index.astype(str))
    if "name" not in frame.columns:
        raise ValueError("Missing required author column: 'name'")
    return frame


def _normalise_authorships(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.rename(
        columns={"paper": "paper_id", "author": "author_id", "person_id": "author_id"},
        inplace=True,
    )
    for column in _REQUIRED_AUTHORSHIP_COLUMNS:
        if column not in frame.columns:
            raise ValueError(f"Missing required authorship column: {column}")
    return frame


def _resolve_columns(papers: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    if columns is not None:
        return [col for col in columns if col in papers.columns]
    default_order = ["title", "abstract", "keywords", "summary"]
    resolved = [col for col in default_order if col in papers.columns]
    if not resolved:
        raise ValueError("No textual columns available for keyword filtering")
    return resolved


class _BaseMatcher:
    def __init__(self, keywords: Sequence[str]) -> None:
        self._keywords = [
            keyword.strip().lower() for keyword in keywords if keyword.strip()
        ]

    def matches(self, row: pd.Series, columns: Sequence[str]) -> bool:
        raise NotImplementedError


class _SubstringMatcher(_BaseMatcher):
    def matches(self, row: pd.Series, columns: Sequence[str]) -> bool:
        joined = " ".join(
            str(row[col]) for col in columns if pd.notna(row[col])
        ).lower()
        return any(keyword in joined for keyword in self._keywords)


class _LemmaMatcher(_BaseMatcher):
    def matches(self, row: pd.Series, columns: Sequence[str]) -> bool:
        text = " ".join(str(row[col]) for col in columns if pd.notna(row[col]))
        tokens = _tokenise(text)
        lemmas = {_lemmatise(token) for token in tokens}
        return any(_lemmatise(keyword) in lemmas for keyword in self._keywords)


def _tokenise(text: str) -> list[str]:
    import re

    return re.findall(r"[\w-]+", text.lower())


def _lemmatise(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


__all__ = ["load_records", "pdf_text", "keyword_filter"]
