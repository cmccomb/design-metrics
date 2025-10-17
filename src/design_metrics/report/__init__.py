"""Lightweight reporting utilities."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


class _ReportRegistry:
    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def capture(self, key: str, value: object) -> None:
        self._data[key] = value

    def snapshot(self) -> dict[str, object]:
        return dict(self._data)

    def reset(self) -> None:
        self._data.clear()


_registry = _ReportRegistry()


def capture(key: str, value: object) -> None:
    _registry.capture(key, value)


def reset() -> None:
    _registry.reset()


@dataclass
class NotebookTemplate:
    name: str
    _context: MutableMapping[str, object] = field(default_factory=dict)

    def with_context(self, **kwargs: object) -> NotebookTemplate:
        self._context.update(kwargs)
        return self

    def render(
        self, *, out: str | Path, context: Mapping[str, object] | None = None
    ) -> Path:
        output_path = Path(out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_context: dict[str, object] = {}
        combined_context.update(_registry.snapshot())
        combined_context.update(self._context)
        if context is not None:
            combined_context.update(context)
        html = _render_html(self.name, combined_context)
        output_path.write_text(html, encoding="utf-8")
        return output_path


def notebook(name: str) -> NotebookTemplate:
    return NotebookTemplate(name=name)


def _render_html(name: str, context: Mapping[str, object]) -> str:
    sections = []
    sections.append(f"<h1>{name.title()} Report</h1>")
    for key, title in (
        ("trend", "Trends"),
        ("topics", "Topics"),
        ("coauthor_stats", "Co-authorship"),
        ("geo", "Geography"),
    ):
        value = context.get(key)
        if _has_content(value):
            sections.append(_html_section(title, _to_table(value)))
    if not sections:
        sections.append("<p>No data available.</p>")
    return "\n".join(["<html><body>"] + sections + ["</body></html>"])


def _to_table(data: object) -> str:
    if isinstance(data, pd.DataFrame):
        return str(data.to_html(index=False, border=0, justify="left"))
    if isinstance(data, Mapping):
        df = pd.DataFrame([data])
        return str(df.to_html(index=False, border=0, justify="left"))
    series = pd.Series([data])
    return f"<pre>{series.to_string(index=False)}</pre>"


def _html_section(title: str, body: str) -> str:
    return f"<section><h2>{title}</h2>{body}</section>"


def _has_content(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, pd.DataFrame):
        return not value.empty
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, list | tuple | set):
        return bool(value)
    if isinstance(value, str):
        return bool(value.strip())
    return True


__all__ = ["capture", "reset", "notebook", "NotebookTemplate"]
