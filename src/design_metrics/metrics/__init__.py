"""High-level analytics for bibliometric time series."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import pandas as pd


def trend(
    frame: pd.DataFrame,
    *,
    by: str,
    groupby: str | Sequence[str] | None = None,
    weight: str | None = None,
) -> pd.DataFrame:
    """Compute aggregated trends for a bibliographic subset."""

    if by not in frame.columns:
        raise ValueError(f"Column '{by}' missing from DataFrame")

    grouping = [by]
    if groupby is not None:
        if isinstance(groupby, str):
            grouping.append(groupby)
        else:
            grouping.extend(groupby)
    for column in grouping:
        if column not in frame.columns:
            raise ValueError(f"Column '{column}' missing from DataFrame")

    if weight is not None and weight not in frame.columns:
        raise ValueError(f"Weight column '{weight}' missing from DataFrame")

    if weight is None:
        aggregated = (
            frame.groupby(grouping, dropna=False).size().reset_index(name="count")
        )
    else:
        aggregated = (
            frame.groupby(grouping, dropna=False)[weight]
            .sum()
            .reset_index(name="value")
        )
    return aggregated.sort_values(grouping).reset_index(drop=True)


def topk(
    frame: pd.DataFrame,
    *,
    field: str,
    k: int = 10,
    separator: str | None = None,
) -> pd.DataFrame:
    """Return the most frequent values for ``field``."""

    if field not in frame.columns:
        raise ValueError(f"Column '{field}' missing from DataFrame")
    if k <= 0:
        raise ValueError("k must be positive")

    values = _iter_values(frame[field], separator)
    counts = Counter(values)
    most_common = counts.most_common(k)
    return pd.DataFrame(most_common, columns=[field, "count"])


def _iter_values(series: pd.Series, separator: str | None) -> Iterable[str]:
    for value in series.dropna():
        if isinstance(value, list | tuple | set):
            for item in value:
                text = str(item).strip()
                if text:
                    yield text
        elif isinstance(value, str) and separator is not None:
            for item in value.split(separator):
                text = item.strip()
                if text:
                    yield text
        else:
            text = str(value).strip()
            if text:
                yield text


__all__ = ["trend", "topk"]
