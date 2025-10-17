"""Network centrality helpers built on top of NetworkX."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import networkx as nx


def degree_centrality(graph: nx.Graph) -> Mapping[Any, float]:
    """Wrap :func:`networkx.degree_centrality` for consistency."""

    result: dict[Any, float] = dict(nx.degree_centrality(graph))
    return result


def betweenness_centrality(
    graph: nx.Graph, *, normalized: bool = True
) -> Mapping[Any, float]:
    """Wrap :func:`networkx.betweenness_centrality`."""

    result: dict[Any, float] = dict(
        nx.betweenness_centrality(graph, normalized=normalized)
    )
    return result


__all__ = ["degree_centrality", "betweenness_centrality"]
