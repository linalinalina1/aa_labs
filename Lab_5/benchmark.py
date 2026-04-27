from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Callable

import numpy as np

from algorithms import kruskal_dsu, kruskal_naive, prim_heap, prim_matrix
from config import BenchmarkConfig
from graphs import generate_dense_graph, generate_sparse_connected_graph
from io_utils import write_csv_dicts


@dataclass(frozen=True)
class BenchmarkRow:
    algorithm: str
    variant: str
    density: str
    n: int
    m_undirected: int
    repeat: int
    seconds: float


def log(msg: str) -> None:
    print(msg, flush=True)


def _time_once(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f}m"
    hours = minutes / 60
    return f"{hours:.2f}h"


def validate_algorithms(cfg: BenchmarkConfig) -> None:
    """Small correctness check before benchmarks."""
    log("Running correctness checks...")

    rng = np.random.default_rng(cfg.seed)

    g_sparse = generate_sparse_connected_graph(
        40,
        cfg.sparse_extra_edges_per_node,
        weight_min=cfg.weight_min,
        weight_max=cfg.weight_max,
        rng=rng,
    )

    g_dense = generate_dense_graph(
        40,
        cfg.dense_edge_probability,
        weight_min=cfg.weight_min,
        weight_max=cfg.weight_max,
        rng=rng,
    )

    def check_graph(g, name: str) -> None:
        r_pm = prim_matrix(g.matrix)
        r_ph = prim_heap(g.adj)
        r_kn = kruskal_naive(g.n, g.edges)
        r_kd = kruskal_dsu(g.n, g.edges)

        expected_edges = max(0, g.n - 1)
        for algo_name, r in (
            ("prim_matrix", r_pm),
            ("prim_heap", r_ph),
            ("kruskal_naive", r_kn),
            ("kruskal_dsu", r_kd),
        ):
            if r.edges_used != expected_edges:
                raise ValueError(
                    f"MST correctness check failed on {name} graph: {algo_name} "
                    f"produced edges_used={r.edges_used}, expected {expected_edges}."
                )

        weights = [r_pm.total_weight, r_ph.total_weight, r_kn.total_weight, r_kd.total_weight]
        if max(weights) - min(weights) > 1e-9:
            raise ValueError(
                f"MST correctness check failed on {name} graph: total weights differ: {weights}"
            )

    check_graph(g_sparse, "sparse")
    check_graph(g_dense, "dense")

    log("Correctness checks passed.\n")


def run_benchmarks(cfg: BenchmarkConfig, *, out_dir: Path) -> Path:
    """Run benchmarks and save raw + aggregated CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_algorithms(cfg)

    rng = np.random.default_rng(cfg.seed)
    rows: list[BenchmarkRow] = []

    total_runs = len(cfg.sizes) * 2 * cfg.repeats * 4
    completed_runs = 0
    benchmark_start = time.perf_counter()

    def print_progress(algorithm: str, variant: str, density: str, n: int, rep: int) -> None:
        nonlocal completed_runs
        completed_runs += 1
        percent = 100.0 * completed_runs / max(1, total_runs)
        elapsed = time.perf_counter() - benchmark_start

        log(
            f"[{completed_runs}/{total_runs} | {percent:6.2f}% | elapsed={_format_seconds(elapsed)}] "
            f"[{algorithm}:{variant}][{density:<6}] n={n} repeat={rep}/{cfg.repeats}"
        )

    def warmup_for_graph(g, *, density: str) -> None:
        log(f"  Warm-up on {density} graph: n={g.n}, m={g.m_undirected}")
        for _ in range(cfg.warmup_runs):
            prim_matrix(g.matrix)
            prim_heap(g.adj)
            kruskal_naive(g.n, g.edges)
            kruskal_dsu(g.n, g.edges)

    def bench_one_graph(g, *, density: str) -> None:
        warmup_for_graph(g, density=density)

        for rep in range(cfg.repeats):
            if rep % 2 == 0:
                order = [
                    ("prim", "matrix"),
                    ("prim", "heap"),
                    ("kruskal", "naive"),
                    ("kruskal", "dsu"),
                ]
            else:
                order = [
                    ("kruskal", "dsu"),
                    ("kruskal", "naive"),
                    ("prim", "heap"),
                    ("prim", "matrix"),
                ]

            for algorithm, variant in order:
                print_progress(algorithm, variant, density, g.n, rep + 1)

                if algorithm == "prim" and variant == "matrix":
                    elapsed = _time_once(lambda: prim_matrix(g.matrix))
                elif algorithm == "prim" and variant == "heap":
                    elapsed = _time_once(lambda: prim_heap(g.adj))
                elif algorithm == "kruskal" and variant == "naive":
                    elapsed = _time_once(lambda: kruskal_naive(g.n, g.edges))
                else:
                    elapsed = _time_once(lambda: kruskal_dsu(g.n, g.edges))

                rows.append(
                    BenchmarkRow(
                        algorithm=algorithm,
                        variant=variant,
                        density=density,
                        n=g.n,
                        m_undirected=g.m_undirected,
                        repeat=rep,
                        seconds=elapsed,
                    )
                )

    log("Starting benchmarks by size...")
    for n in cfg.sizes:
        block_start = time.perf_counter()
        log("\n==================== SIZE BLOCK ====================")
        log(f"n = {n}")
        log("===================================================")

        log(f"Creating sparse graph for n={n} ...")
        g_sparse = generate_sparse_connected_graph(
            n,
            cfg.sparse_extra_edges_per_node,
            weight_min=cfg.weight_min,
            weight_max=cfg.weight_max,
            rng=rng,
        )
        log(f"Finished sparse graph: n={g_sparse.n}, m={g_sparse.m_undirected}")
        bench_one_graph(g_sparse, density="sparse")

        log(f"Creating dense graph for n={n} ...")
        g_dense = generate_dense_graph(
            n,
            cfg.dense_edge_probability,
            weight_min=cfg.weight_min,
            weight_max=cfg.weight_max,
            rng=rng,
        )
        log(f"Finished dense graph: n={g_dense.n}, m={g_dense.m_undirected}")
        bench_one_graph(g_dense, density="dense")

        block_elapsed = time.perf_counter() - block_start
        log(f"Finished size block n={n} in {_format_seconds(block_elapsed)}")

    total_elapsed = time.perf_counter() - benchmark_start
    log(f"\nAll benchmarks finished in {_format_seconds(total_elapsed)}")

    raw_rows = [asdict(r) for r in rows]
    raw_csv = out_dir / "results_raw.csv"
    write_csv_dicts(
        raw_csv,
        raw_rows,
        fieldnames=[
            "algorithm",
            "variant",
            "density",
            "n",
            "m_undirected",
            "repeat",
            "seconds",
        ],
    )

    (out_dir / "run_meta.json").write_text(
        json.dumps({"config": asdict(cfg)}, indent=2),
        encoding="utf-8",
    )

    by_key: dict[tuple[str, str, str, int], list[BenchmarkRow]] = defaultdict(list)
    for r in rows:
        by_key[(r.algorithm, r.variant, r.density, r.n)].append(r)

    agg_rows: list[dict[str, Any]] = []
    for (algorithm, variant, density, n), rs in by_key.items():
        secs = [x.seconds for x in rs]
        ms = [x.m_undirected for x in rs]
        agg_rows.append(
            {
                "algorithm": algorithm,
                "variant": variant,
                "density": density,
                "n": n,
                "m_undirected": int(median(ms)) if ms else 0,
                "seconds_median": float(median(secs)) if secs else float("nan"),
                "seconds_mean": float(fmean(secs)) if secs else float("nan"),
                "samples": len(secs),
            }
        )

    agg_rows.sort(key=lambda x: (str(x["algorithm"]), str(x["density"]), str(x["variant"]), int(x["n"])))

    agg_csv = out_dir / "results_agg.csv"
    write_csv_dicts(
        agg_csv,
        agg_rows,
        fieldnames=[
            "algorithm",
            "variant",
            "density",
            "n",
            "m_undirected",
            "seconds_median",
            "seconds_mean",
            "samples",
        ],
    )

    return agg_csv
