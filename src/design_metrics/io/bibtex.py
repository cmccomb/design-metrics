"""Lightweight BibTeX parsing utilities.

The goal of this module is to provide a dependency-free parser that captures
common BibTeX entry patterns. It intentionally focuses on the fields most
useful for scientometric analyses and does not attempt to be a complete TeX
parser.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass(slots=True)
class BibEntry:
    """Representation of a BibTeX entry."""

    entry_type: str
    citation_key: str
    fields: OrderedDict[str, str]

    def to_dict(self) -> dict[str, str]:
        """Return the entry as a regular dictionary."""

        data = {"ENTRYTYPE": self.entry_type, "ID": self.citation_key}
        data.update(self.fields)
        return data


def _clean_field_value(value: str) -> str:
    value = value.strip().strip(",")
    if value.startswith("{") and value.endswith("}"):
        return value[1:-1].strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1].strip()
    return value


def parse_bibtex_entries(text: str) -> list[BibEntry]:
    """Parse BibTeX entries into :class:`BibEntry` records.

    Args:
        text: Raw BibTeX text containing one or more entries.

    Returns:
        A list of parsed BibTeX entries.

    Raises:
        ValueError: If an entry is malformed or missing critical components.
    """

    entries: list[BibEntry] = []
    current_type: str | None = None
    current_key: str | None = None
    current_fields: OrderedDict[str, str] = OrderedDict()
    collecting_field: str | None = None
    buffer: list[str] = []
    brace_level = 0

    lines = (line.strip() for line in text.splitlines())
    for line in lines:
        if not line or line.startswith("%"):
            continue
        if line.startswith("@"):  # start of entry
            if current_type is not None:
                raise ValueError("Nested BibTeX entries detected.")
            try:
                header, remainder = line[1:].split("{", 1)
            except ValueError as exc:  # pragma: no cover - defensive branch
                raise ValueError("Malformed BibTeX entry header.") from exc
            current_type = header.strip()
            current_key = remainder.rstrip(",").strip()
            current_fields = OrderedDict()
            continue

        if current_type is None or current_key is None:
            raise ValueError("Encountered field without an active BibTeX entry.")

        if collecting_field:
            buffer.append(line)
            brace_level += line.count("{") - line.count("}")
            if brace_level <= 0 and line.endswith(","):
                value = " ".join(buffer)
                current_fields[collecting_field] = _clean_field_value(value)
                collecting_field = None
                buffer = []
            continue

        if "=" not in line:
            continue

        field, value = [part.strip() for part in line.split("=", 1)]
        if value.endswith(",") and value.count("{") == value.count("}"):
            current_fields[field.lower()] = _clean_field_value(value)
        else:
            collecting_field = field.lower()
            buffer = [value]
            brace_level = value.count("{") - value.count("}")

        if line.endswith("}") and brace_level <= 0:
            entries.append(
                BibEntry(
                    entry_type=current_type,
                    citation_key=current_key,
                    fields=current_fields,
                )
            )
            current_type = None
            current_key = None
            current_fields = OrderedDict()

    if current_type is not None:
        entries.append(
            BibEntry(
                entry_type=current_type,
                citation_key=current_key or "",
                fields=current_fields,
            )
        )

    return entries


__all__ = ["BibEntry", "parse_bibtex_entries"]
