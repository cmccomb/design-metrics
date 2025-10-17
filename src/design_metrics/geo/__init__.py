"""Geographic parsing for institutional affiliations."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def parse_affiliations(
    papers: pd.DataFrame,
    *,
    column: str = "affiliations",
    paper_id: str = "paper_id",
    separator: str = ";",
) -> pd.DataFrame:
    """Expand affiliation strings into structured records."""

    if column not in papers.columns:
        raise ValueError(f"Column '{column}' not found in papers DataFrame")
    if paper_id not in papers.columns:
        raise ValueError(f"Column '{paper_id}' not found in papers DataFrame")

    records: list[dict[str, str]] = []
    for _, row in papers.iterrows():
        raw_affiliations = row[column]
        if pd.isna(raw_affiliations):
            continue
        entries = _normalise_entries(raw_affiliations, separator)
        for entry in entries:
            country = _infer_country(entry)
            records.append(
                {
                    "paper_id": str(row[paper_id]),
                    "affiliation": entry,
                    "country": country,
                }
            )
    return pd.DataFrame.from_records(records)


def aggregate(frame: pd.DataFrame, level: str = "country") -> pd.DataFrame:
    """Aggregate affiliations by geography."""

    if level not in frame.columns:
        raise ValueError(f"Column '{level}' missing from affiliation data")
    aggregated = (
        frame.groupby(level, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return aggregated.reset_index(drop=True)


def _normalise_entries(value: object, separator: str) -> list[str]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(separator)]
        return [part for part in parts if part]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _infer_country(entry: str) -> str:
    if "(" in entry and ")" in entry:
        inside = entry.rsplit("(", 1)[-1].rstrip(") ")
        if inside:
            return inside
    if "," in entry:
        parts = [part.strip() for part in entry.split(",") if part.strip()]
        if parts:
            return parts[-1]
    return "Unknown"


__all__ = ["parse_affiliations", "aggregate"]
