from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark parameters for Lab 7 (Greedy Algorithms).

    The lab compares MST algorithms (Prim and Kruskal) in baseline and optimized
    variants, on sparse and dense connected graphs, for increasing n.
    """

    seed: int = 12345

    # Number of nodes to benchmark.
    sizes: tuple[int, ...] = (100, 200, 400, 800, 1200, 1600, 2000, 5000)

    # Sparse graph: a random spanning tree + O(n) extra edges.
    sparse_extra_edges_per_node: int = 4

    # Dense graph: start with a spanning tree, then add edges with probability p.
    dense_edge_probability: float = 0.60

    # Edge weights are positive integers in [weight_min, weight_max].
    weight_min: int = 1
    weight_max: int = 10

    # Benchmark repetitions.
    repeats: int = 3
    warmup_runs: int = 1
