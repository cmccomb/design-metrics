"""Namespace conveniences for the :mod:`design_metrics` package."""

from __future__ import annotations

from importlib import import_module
from pkgutil import extend_path
from types import ModuleType

__path__ = list(extend_path(__path__, __name__))

_SUBMODULES: dict[str, str] = {
    "bib": "design_metrics.bib",
    "clean": "design_metrics.clean",
    "filter": "design_metrics.filter",
    "metrics": "design_metrics.metrics",
    "topics": "design_metrics.topics",
    "graphs": "design_metrics.graphs",
    "geo": "design_metrics.geo",
    "refs": "design_metrics.refs",
    "report": "design_metrics.report",
}


def __getattr__(name: str) -> ModuleType:
    """Lazy-load selected submodules.

    This allows ``import design_metrics as dm`` followed by ``dm.bib`` without
    importing the ``design_metrics.bib`` module eagerly. Only known submodules
    are exposed to avoid polluting the public namespace.
    """

    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        if isinstance(module, ModuleType):
            globals()[name] = module
            return module
    raise AttributeError(f"module 'design_metrics' has no attribute {name!r}")


__all__ = tuple(_SUBMODULES)
