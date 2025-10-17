"""Subsetting and labelling helpers for bibliometric corpora."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass

import pandas as pd

from design_metrics.clean import _normalise_text


def by_keywords(
    papers: pd.DataFrame,
    terms: Sequence[str],
    *,
    mode: str = "lemma",
    columns: Sequence[str] | None = None,
    min_match: int | None = None,
) -> pd.DataFrame:
    """Return rows that satisfy keyword constraints."""

    if not terms:
        return papers.iloc[0:0]

    search_columns = _resolve_columns(papers, columns)
    normalised_terms = [_normalise_text(term) for term in terms if term.strip()]

    if not normalised_terms:
        return papers.iloc[0:0]

    matcher: _BaseMatcher
    if mode == "lemma":
        matcher = _LemmaMatcher(normalised_terms)
    elif mode == "minmatch":
        matcher = _MinimumMatchMatcher(normalised_terms, min_match)
    elif mode == "substring":
        matcher = _SubstringMatcher(normalised_terms)
    else:
        raise ValueError(
            "Unsupported mode. Choose from 'lemma', 'minmatch', 'substring'."
        )

    mask = papers.apply(lambda row: matcher.matches(row, search_columns), axis=1)
    return papers.loc[mask].copy()


def label_by_rules(
    papers: pd.DataFrame,
    rules_yaml: str | Mapping[str, object],
    *,
    label_column: str = "labels",
) -> pd.DataFrame:
    """Assign labels to papers using YAML rule definitions."""

    rules = _parse_rules(rules_yaml)
    frame = papers.copy()
    frame[label_column] = [[] for _ in range(len(frame))]

    for rule in rules:
        subset = by_keywords(
            frame,
            rule.terms,
            mode=rule.mode,
            columns=rule.columns,
            min_match=rule.min_match,
        )
        if subset.empty:
            continue
        for row_index in subset.index:
            existing_labels = set(frame.at[row_index, label_column])
            existing_labels.add(rule.label)
            frame.at[row_index, label_column] = sorted(existing_labels)
    return frame


@dataclass
class _Rule:
    label: str
    terms: Sequence[str]
    mode: str
    columns: Sequence[str] | None
    min_match: int | None


def _parse_rules(rules_yaml: str | Mapping[str, object]) -> list[_Rule]:
    if isinstance(rules_yaml, str):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Parsing YAML rules requires the optional PyYAML dependency"
            ) from exc
        parsed = yaml.safe_load(rules_yaml) or []
    else:
        parsed = rules_yaml

    if isinstance(parsed, Mapping):
        parsed_rules = parsed.get("rules", [])
    else:
        parsed_rules = parsed

    rules: list[_Rule] = []
    for entry in parsed_rules or []:
        if not isinstance(entry, Mapping):
            continue
        label = str(entry.get("label", "")).strip()
        if not label:
            continue
        terms: MutableSequence[str] = []
        for key in ("terms", "any", "all"):
            value = entry.get(key)
            if isinstance(value, str):
                terms.append(value)
            elif isinstance(value, Iterable):
                terms.extend(str(item) for item in value)
        terms = [term for term in terms if term.strip()]
        if not terms:
            continue
        mode = str(entry.get("mode", "lemma")).lower()
        columns_value = entry.get("columns")
        columns: Sequence[str] | None
        if isinstance(columns_value, Iterable) and not isinstance(
            columns_value, str | bytes
        ):
            columns = [str(column) for column in columns_value]
        else:
            columns = None
        min_match = entry.get("min_match")
        min_match_int: int | None
        if isinstance(min_match, int):
            min_match_int = min_match
        else:
            match_type = str(entry.get("match", "any")).lower()
            if match_type == "all" and mode == "lemma":
                mode = "minmatch"
            min_match_int = len(terms) if match_type == "all" else None
        rules.append(
            _Rule(
                label=label,
                terms=terms,
                mode=mode,
                columns=columns,
                min_match=min_match_int,
            )
        )
    return rules


def _resolve_columns(papers: pd.DataFrame, columns: Sequence[str] | None) -> list[str]:
    if columns is not None:
        resolved = [column for column in columns if column in papers.columns]
        if resolved:
            return resolved
    default_order = ["title", "abstract", "keywords", "summary"]
    resolved = [column for column in default_order if column in papers.columns]
    if not resolved:
        raise ValueError("No textual columns available for keyword filtering")
    return resolved


class _BaseMatcher:
    def __init__(self, terms: Sequence[str]) -> None:
        self._terms = [term for term in terms if term]

    def matches(self, row: pd.Series, columns: Sequence[str]) -> bool:
        raise NotImplementedError


class _SubstringMatcher(_BaseMatcher):
    def matches(self, row: pd.Series, columns: Sequence[str]) -> bool:
        haystack = " ".join(_normalise_text(row.get(column, "")) for column in columns)
        return any(term in haystack for term in self._terms)


class _LemmaMatcher(_BaseMatcher):
    def matches(self, row: pd.Series, columns: Sequence[str]) -> bool:
        haystack_terms: set[str] = set()
        for column in columns:
            value = row.get(column, "")
            haystack_terms.update(_normalise_text(value).split())
        return any(term in haystack_terms for term in self._terms)


class _MinimumMatchMatcher(_BaseMatcher):
    def __init__(self, terms: Sequence[str], min_match: int | None) -> None:
        super().__init__(terms)
        self._min_match = max(1, min_match or len(self._terms))

    def matches(self, row: pd.Series, columns: Sequence[str]) -> bool:
        haystack = " ".join(_normalise_text(row.get(column, "")) for column in columns)
        count = sum(1 for term in self._terms if term in haystack)
        return count >= self._min_match


__all__ = ["by_keywords", "label_by_rules"]
