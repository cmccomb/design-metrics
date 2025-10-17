"""Reference parsing and citation matching."""

from __future__ import annotations

import re

import pandas as pd


def extract(
    pages: pd.DataFrame,
    *,
    paper_column: str = "paper_id",
    text_column: str = "text",
) -> pd.DataFrame:
    """Extract reference strings from page-level text."""

    for column in (paper_column, text_column):
        if column not in pages.columns:
            raise ValueError(f"Column '{column}' missing from pages DataFrame")

    records: list[dict[str, str]] = []
    for paper_id, group in pages.groupby(paper_column):
        combined = "\n".join(group[text_column].dropna().astype(str))
        for reference in _split_references(combined):
            records.append({"paper_id": str(paper_id), "reference": reference})
    return pd.DataFrame.from_records(records)


def in_corpus_citations(
    references: pd.DataFrame,
    papers: pd.DataFrame,
    *,
    reference_column: str = "reference",
    title_column: str = "title",
    paper_id_column: str = "paper_id",
    corpus_id_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify citations that point to papers within the corpus."""

    if reference_column not in references.columns:
        raise ValueError(
            f"Column '{reference_column}' missing from references DataFrame"
        )
    if paper_id_column not in references.columns:
        raise ValueError(
            f"Column '{paper_id_column}' missing from references DataFrame"
        )
    if title_column not in papers.columns:
        raise ValueError(f"Column '{title_column}' missing from papers DataFrame")

    corpus_identifier = corpus_id_column or paper_id_column
    if corpus_identifier not in papers.columns:
        raise ValueError(f"Column '{corpus_identifier}' missing from papers DataFrame")

    title_lookup = {
        _normalise_title(title): str(row[corpus_identifier])
        for _, row in papers.iterrows()
        if isinstance(title := row[title_column], str)
    }

    citations: list[dict[str, str]] = []
    for _, reference_row in references.iterrows():
        normalised_reference = _normalise_title(reference_row[reference_column])
        for title_key, cited_id in title_lookup.items():
            if (
                title_key
                and title_key in normalised_reference
                and reference_row[paper_id_column] != cited_id
            ):
                citations.append(
                    {
                        "citing_paper_id": reference_row[paper_id_column],
                        "cited_paper_id": cited_id,
                        "reference": reference_row[reference_column],
                    }
                )
                break
    edges = pd.DataFrame.from_records(citations)
    indegree = (
        edges.groupby("cited_paper_id")
        .size()
        .reset_index(name="indegree")
        .rename(columns={"cited_paper_id": "paper_id"})
        if not edges.empty
        else pd.DataFrame(columns=["paper_id", "indegree"])
    )
    return edges, indegree


def _split_references(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    references: list[str] = []
    current: list[str] = []
    for line in lines:
        if re.match(r"^(\[?\d+\]?|\d+\.)\s", line) and current:
            references.append(" ".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        references.append(" ".join(current))
    return references


def _normalise_title(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.casefold())
    return cleaned.strip()


__all__ = ["extract", "in_corpus_citations"]
