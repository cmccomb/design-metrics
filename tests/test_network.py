import networkx as nx

from design_metrics.net.centrality import betweenness_centrality, degree_centrality


def test_degree_centrality_triangle() -> None:
    graph = nx.cycle_graph(3)

    centrality = degree_centrality(graph)

    assert all(value == 1.0 for value in centrality.values())


def test_betweenness_centrality_line_graph() -> None:
    graph = nx.path_graph(3)

    centrality = betweenness_centrality(graph, normalized=True)

    assert centrality[1] > centrality[0]
