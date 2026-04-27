from __future__ import annotations

from dataclasses import dataclass

import numpy as np


AdjList = list[list[tuple[int, float]]]
Edge = tuple[int, int, float]  # (u, v, w) with u < v


@dataclass(frozen=True)
class GeneratedGraph:
    """A connected weighted undirected graph in multiple representations."""

    n: int
    m_undirected: int
    adj: AdjList
    matrix: np.ndarray  # inf for missing edges, 0 on diagonal
    edges: list[Edge]


def _max_undirected_edges(n: int) -> int:
    return n * (n - 1) // 2


def _empty_adj(n: int) -> AdjList:
    return [[] for _ in range(n)]


def _edges_to_adj(n: int, edges: list[Edge]) -> AdjList:
    adj = _empty_adj(n)
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def _edges_to_matrix(n: int, edges: list[Edge]) -> np.ndarray:
    dist = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)

    for u, v, w in edges:
        ww = float(w)
        if ww < dist[u, v]:
            dist[u, v] = ww
            dist[v, u] = ww

    return dist


def generate_sparse_connected_graph(
    n: int,
    extra_edges_per_node: int,
    *,
    weight_min: int,
    weight_max: int,
    rng: np.random.Generator,
) -> GeneratedGraph:
    """Generate a sparse connected weighted undirected graph.

    Strategy: build a random spanning tree (guarantees connectivity), then add
    extra random edges to reach approximately (n-1) + extra_edges_per_node*n edges.
    """
    if n < 2:
        edges: list[Edge] = []
        return GeneratedGraph(
            n=n,
            m_undirected=0,
            adj=_edges_to_adj(n, edges),
            matrix=_edges_to_matrix(n, edges),
            edges=edges,
        )

    max_edges = _max_undirected_edges(n)
    target_total_edges = min(max_edges, (n - 1) + max(0, int(extra_edges_per_node * n)))

    edge_keys: set[tuple[int, int]] = set()
    edges: list[Edge] = []

    # Spanning tree
    for v in range(1, n):
        u = int(rng.integers(0, v))
        a, b = (u, v) if u < v else (v, u)
        w = float(rng.integers(weight_min, weight_max + 1))
        edge_keys.add((a, b))
        edges.append((a, b, w))

    # Extra edges
    while len(edges) < target_total_edges:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue

        a, b = (i, j) if i < j else (j, i)
        if (a, b) in edge_keys:
            continue

        w = float(rng.integers(weight_min, weight_max + 1))
        edge_keys.add((a, b))
        edges.append((a, b, w))

    adj = _edges_to_adj(n, edges)
    matrix = _edges_to_matrix(n, edges)
    return GeneratedGraph(n=n, m_undirected=len(edges), adj=adj, matrix=matrix, edges=edges)


def generate_dense_graph(
    n: int,
    p: float,
    *,
    weight_min: int,
    weight_max: int,
    rng: np.random.Generator,
) -> GeneratedGraph:
    """Generate a dense connected weighted undirected graph.

    Strategy: build a random spanning tree (guarantees connectivity), then for each
    missing edge (i, j) add it with probability p.
    """
    if n < 2:
        edges: list[Edge] = []
        return GeneratedGraph(
            n=n,
            m_undirected=0,
            adj=_edges_to_adj(n, edges),
            matrix=_edges_to_matrix(n, edges),
            edges=edges,
        )

    edge_keys: set[tuple[int, int]] = set()
    edges: list[Edge] = []

    # Spanning tree
    for v in range(1, n):
        u = int(rng.integers(0, v))
        a, b = (u, v) if u < v else (v, u)
        w = float(rng.integers(weight_min, weight_max + 1))
        edge_keys.add((a, b))
        edges.append((a, b, w))

    # Add remaining edges with probability p
    pp = float(p)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in edge_keys:
                continue
            if rng.random() < pp:
                w = float(rng.integers(weight_min, weight_max + 1))
                edge_keys.add((i, j))
                edges.append((i, j, w))

    adj = _edges_to_adj(n, edges)
    matrix = _edges_to_matrix(n, edges)
    return GeneratedGraph(n=n, m_undirected=len(edges), adj=adj, matrix=matrix, edges=edges)
