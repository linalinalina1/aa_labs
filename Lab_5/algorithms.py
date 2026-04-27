from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Final

import numpy as np

from graphs import AdjList, Edge


INF: Final[float] = float("inf")


@dataclass(frozen=True)
class MSTResult:
    """Result of a minimum spanning tree/forest computation."""

    total_weight: float
    edges_used: int


def prim_matrix(matrix: np.ndarray, *, start: int = 0) -> MSTResult:
    """Prim's algorithm (baseline): adjacency matrix + O(n^2) min search.

    Complexity: O(n^2) time, O(n) extra memory.
    """
    n = int(matrix.shape[0])
    if not 0 <= start < n:
        raise ValueError("start vertex out of range")
    if n <= 1:
        return MSTResult(total_weight=0.0, edges_used=0)

    used = [False] * n
    key = [INF] * n
    key[start] = 0.0

    total = 0.0
    edges_used = 0

    for _ in range(n):
        u = -1
        best = INF

        for i in range(n):
            if not used[i] and key[i] < best:
                best = key[i]
                u = i

        if u == -1 or best == INF:
            break  # disconnected

        used[u] = True
        total += float(best)
        if u != start:
            edges_used += 1

        row = matrix[u]
        for v in range(n):
            w = float(row[v])
            if not used[v] and w < key[v]:
                key[v] = w

    return MSTResult(total_weight=total, edges_used=edges_used)


def prim_heap(adj: AdjList, *, start: int = 0) -> MSTResult:
    """Prim's algorithm (optimized): adjacency list + min-heap.

    Complexity: O(m log n) time, O(m) memory for the heap in the worst case.
    """
    n = len(adj)
    if not 0 <= start < n:
        raise ValueError("start vertex out of range")
    if n <= 1:
        return MSTResult(total_weight=0.0, edges_used=0)

    in_mst = [False] * n
    in_mst[start] = True

    pq: list[tuple[float, int]] = []  # (w, v) where v is the new vertex
    for v, w in adj[start]:
        heapq.heappush(pq, (float(w), int(v)))

    total = 0.0
    edges_used = 0

    while pq and edges_used < n - 1:
        w, v = heapq.heappop(pq)
        if in_mst[v]:
            continue

        in_mst[v] = True
        total += float(w)
        edges_used += 1

        for to, wt in adj[v]:
            if not in_mst[to]:
                heapq.heappush(pq, (float(wt), int(to)))

    return MSTResult(total_weight=total, edges_used=edges_used)


class _DSUNaive:
    """Disjoint Set Union without path compression or union-by-size."""

    __slots__ = ("parent",)

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


class _DSU:
    """Disjoint Set Union with path compression + union by size."""

    __slots__ = ("parent", "size")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False

        size = self.size
        if size[ra] < size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        size[ra] += size[rb]
        return True


def kruskal_naive(n: int, edges: list[Edge]) -> MSTResult:
    """Kruskal's algorithm (baseline): sort edges + naive union-find."""
    if n <= 1:
        return MSTResult(total_weight=0.0, edges_used=0)

    dsu = _DSUNaive(n)
    total = 0.0
    used = 0

    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if dsu.union(int(u), int(v)):
            total += float(w)
            used += 1
            if used == n - 1:
                break

    return MSTResult(total_weight=total, edges_used=used)


def kruskal_dsu(n: int, edges: list[Edge]) -> MSTResult:
    """Kruskal's algorithm (optimized): sort edges + DSU with heuristics."""
    if n <= 1:
        return MSTResult(total_weight=0.0, edges_used=0)

    dsu = _DSU(n)
    total = 0.0
    used = 0

    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if dsu.union(int(u), int(v)):
            total += float(w)
            used += 1
            if used == n - 1:
                break

    return MSTResult(total_weight=total, edges_used=used)
