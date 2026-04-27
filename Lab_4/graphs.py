from __future__ import annotations

from dataclasses import dataclass

import numpy as np


AdjList = list[list[tuple[int, float]]]


@dataclass(frozen=True)
class GeneratedGraph:
    n: int
    m_undirected: int
    adj: AdjList
    matrix: np.ndarray  # inf for missing edges, 0 on diagonal


def _empty_adj(n: int) -> AdjList:
    return [[] for _ in range(n)]


def _add_undirected_edge(adj: AdjList, i: int, j: int, w: float) -> None:
    adj[i].append((j, w))
    adj[j].append((i, w))


def _adj_to_matrix(adj: AdjList) -> np.ndarray:
    n = len(adj)
    dist = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(dist, 0.0)

    for u in range(n):
        for v, w in adj[u]:
            if w < dist[u, v]:
                dist[u, v] = float(w)

    return dist


def _count_undirected_edges(adj: AdjList) -> int:
    return sum(len(neigh) for neigh in adj) // 2


def generate_sparse_connected_graph(
    n: int,
    extra_edges_per_node: int,
    *,
    weight_min: int,
    weight_max: int,
    rng: np.random.Generator,
) -> GeneratedGraph:
    """Generate a sparse connected weighted undirected graph.

    First builds a random spanning tree, then adds extra edges.
    """
    if n < 2:
        adj = _empty_adj(n)
        matrix = _adj_to_matrix(adj)
        return GeneratedGraph(n=n, m_undirected=0, adj=adj, matrix=matrix)

    adj = _empty_adj(n)

    # Spanning tree => guarantees connectivity
    for v in range(1, n):
        u = int(rng.integers(0, v))
        w = float(rng.integers(weight_min, weight_max + 1))
        _add_undirected_edge(adj, u, v, w)

    target_extra = max(0, int(extra_edges_per_node * n))
    added = 0

    while added < target_extra:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue

        w = float(rng.integers(weight_min, weight_max + 1))
        _add_undirected_edge(adj, i, j, w)
        added += 1

    matrix = _adj_to_matrix(adj)
    return GeneratedGraph(
        n=n,
        m_undirected=_count_undirected_edges(adj),
        adj=adj,
        matrix=matrix,
    )


def generate_dense_graph(
    n: int,
    p: float,
    *,
    weight_min: int,
    weight_max: int,
    rng: np.random.Generator,
) -> GeneratedGraph:
    """Generate a dense connected weighted undirected graph.

    Connectivity is guaranteed by first building a spanning tree,
    then adding random edges with probability p.
    """
    if n < 2:
        adj = _empty_adj(n)
        matrix = _adj_to_matrix(adj)
        return GeneratedGraph(n=n, m_undirected=0, adj=adj, matrix=matrix)

    adj = _empty_adj(n)

    # First: spanning tree for guaranteed connectivity
    existing_edges: set[tuple[int, int]] = set()
    for v in range(1, n):
        u = int(rng.integers(0, v))
        w = float(rng.integers(weight_min, weight_max + 1))
        _add_undirected_edge(adj, u, v, w)
        existing_edges.add((min(u, v), max(u, v)))

    # Then: add more edges probabilistically
    for i in range(n):
        for j in range(i + 1, n):
            key = (i, j)
            if key in existing_edges:
                continue
            if rng.random() < p:
                w = float(rng.integers(weight_min, weight_max + 1))
                _add_undirected_edge(adj, i, j, w)
                existing_edges.add(key)

    matrix = _adj_to_matrix(adj)
    return GeneratedGraph(
        n=n,
        m_undirected=_count_undirected_edges(adj),
        adj=adj,
        matrix=matrix,
    )