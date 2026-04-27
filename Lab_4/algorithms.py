from __future__ import annotations

import heapq
from typing import Final

import numpy as np

from graphs import AdjList


INF: Final[float] = float("inf")


def dijkstra_array(adj: AdjList, source: int) -> list[float]:
    """Dijkstra using array-based minimum selection.

    Complexity:
        O(n^2 + m)

    Suitable for:
        dense graphs or smaller graphs.

    Requirements:
        edge weights must be non-negative.
    """
    n = len(adj)
    dist = [INF] * n
    used = [False] * n
    dist[source] = 0.0

    for _ in range(n):
        u = -1
        best = INF

        for i in range(n):
            if not used[i] and dist[i] < best:
                best = dist[i]
                u = i

        if u == -1:
            break

        used[u] = True
        du = dist[u]

        for v, w in adj[u]:
            nd = du + w
            if nd < dist[v]:
                dist[v] = nd

    return dist


def dijkstra_heap(adj: AdjList, source: int) -> list[float]:
    """Dijkstra optimized with a binary heap.

    Complexity:
        O((n + m) log n)

    Suitable for:
        sparse graphs.

    Requirements:
        edge weights must be non-negative.
    """
    n = len(adj)
    dist = [INF] * n
    dist[source] = 0.0

    pq: list[tuple[float, int]] = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)

        if d != dist[u]:
            continue

        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist


def floyd_warshall_python(dist0: np.ndarray) -> np.ndarray:
    """Floyd–Warshall with classic triple loop.

    Supports:
        - positive weights
        - zero weights
        - negative edge weights

    Note:
        If a negative cycle exists, the returned matrix is not meaningful
        as a shortest-path solution. Use detect_negative_cycle_from_dist()
        or floyd_warshall_python_checked().
    """
    dist = dist0.astype(np.float64, copy=True)
    n = dist.shape[0]

    for k in range(n):
        for i in range(n):
            dik = dist[i, k]
            if np.isinf(dik):
                continue

            for j in range(n):
                nd = dik + dist[k, j]
                if nd < dist[i, j]:
                    dist[i, j] = nd

    return dist


def floyd_warshall_numpy(dist0: np.ndarray) -> np.ndarray:
    """Floyd–Warshall with NumPy vectorized update.

    Supports:
        - positive weights
        - zero weights
        - negative edge weights

    Note:
        If a negative cycle exists, the returned matrix is not meaningful
        as a shortest-path solution. Use detect_negative_cycle_from_dist()
        or floyd_warshall_numpy_checked().

    Still O(n^3), but usually faster than pure Python
    because the inner work is done inside NumPy.
    """
    dist = dist0.astype(np.float64, copy=True)
    n = dist.shape[0]

    for k in range(n):
        dist = np.minimum(dist, dist[:, [k]] + dist[[k], :])

    return dist


def detect_negative_cycle_from_dist(dist: np.ndarray) -> bool:
    """Detect a negative cycle from a completed Floyd distance matrix.

    A negative value on the diagonal means there is a negative cycle.
    """
    return bool(np.any(np.diag(dist) < 0))


def floyd_warshall_python_checked(dist0: np.ndarray) -> tuple[np.ndarray, bool]:
    """Run Floyd–Warshall (Python) and also report negative-cycle presence."""
    dist = floyd_warshall_python(dist0)
    has_negative_cycle = detect_negative_cycle_from_dist(dist)
    return dist, has_negative_cycle


def floyd_warshall_numpy_checked(dist0: np.ndarray) -> tuple[np.ndarray, bool]:
    """Run Floyd–Warshall (NumPy) and also report negative-cycle presence."""
    dist = floyd_warshall_numpy(dist0)
    has_negative_cycle = detect_negative_cycle_from_dist(dist)
    return dist, has_negative_cycle