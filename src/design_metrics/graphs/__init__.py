"""Graph analytics for collaboration networks."""

from __future__ import annotations

from itertools import combinations

import networkx as nx
import pandas as pd


def coauthors(
    authorships: pd.DataFrame,
    *,
    paper_column: str = "paper_id",
    author_column: str = "author_id",
) -> nx.Graph:
    """Construct an undirected co-authorship network."""

    for column in (paper_column, author_column):
        if column not in authorships.columns:
            raise ValueError(f"Column '{column}' missing from authorship data")

    graph = nx.Graph()
    for paper_id, group in authorships.groupby(paper_column):
        authors = group[author_column].dropna().astype(str).unique()
        if len(authors) == 1:
            graph.add_node(authors[0])
        for source, target in combinations(authors, 2):
            if graph.has_edge(source, target):
                graph[source][target]["weight"] += 1
                graph[source][target].setdefault("papers", set()).add(paper_id)
            else:
                graph.add_edge(source, target, weight=1, papers={paper_id})
    return graph


def stats(graph: nx.Graph) -> dict[str, float]:
    """Summarise a collaboration network with standard metrics."""

    if graph.number_of_nodes() == 0:
        return {
            "nodes": 0,
            "edges": 0,
            "density": 0.0,
            "average_degree": 0.0,
            "components": 0,
            "largest_component_fraction": 0.0,
        }

    components = list(nx.connected_components(graph))
    largest = max((len(component) for component in components), default=0)
    total_nodes = graph.number_of_nodes()
    stats_dict = {
        "nodes": float(total_nodes),
        "edges": float(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "average_degree": float(sum(dict(graph.degree()).values()) / total_nodes),
        "components": float(len(components)),
        "largest_component_fraction": float(largest / total_nodes),
    }
    return stats_dict


def communities(
    graph: nx.Graph,
    *,
    method: str = "louvain",
    weight: str | None = "weight",
) -> pd.DataFrame:
    """Detect communities within the collaboration network."""

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "community"])

    method_lower = method.lower()
    if method_lower == "louvain":
        communities_iter = nx.algorithms.community.louvain_communities(
            graph,
            weight=weight,
        )
    elif method_lower == "leiden":  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Leiden community detection requires the 'leidenalg' package"
        )
    else:
        raise ValueError("Unsupported method. Choose from 'louvain' or 'leiden'.")

    rows = []
    for community_id, members in enumerate(communities_iter):
        for node in members:
            rows.append({"node": node, "community": community_id})
    return pd.DataFrame(rows)


__all__ = ["coauthors", "stats", "communities"]
