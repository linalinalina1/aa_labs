from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    seed: int = 12345

    # Dijkstra
    dijkstra_sizes: tuple[int, ...] = (20, 30, 40, 50, 80, 120, 200, 300, 400, 600, 800)
    #dijkstra_sizes: tuple[int, ...] = (10, 20, 30, 40, 80)
    # Floyd–Warshall
    floyd_sizes: tuple[int, ...] = (20, 30, 40, 50, 80, 120, 200, 300, 400, 600, 800)
    #floyd_sizes: tuple[int, ...] = (10, 20, 30, 40, 80)
    # Sparse graph: about O(n) extra edges beyond the spanning tree
    sparse_extra_edges_per_node: int = 4

    # Dense graph: higher probability for a much denser graph
    dense_edge_probability: float = 0.60

    weight_min: int = 1
    weight_max: int = 10

    repeats: int = 3
    warmup_runs: int = 1

    # For Dijkstra, test multiple random sources
    benchmark_sources: int = 2  