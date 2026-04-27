from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any, Callable

import numpy as np

from algorithms import (
    dijkstra_array,
    dijkstra_heap,
    floyd_warshall_numpy,
    floyd_warshall_python,
)
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
    source: int | None
    repeat: int
    seconds: float


def log(msg: str) -> None:
    print(msg, flush=True)


def _time_once(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def _dist_lists_close(a: list[float], b: list[float], *, atol: float = 1e-9) -> bool:
    if len(a) != len(b):
        return False

    for x, y in zip(a, b):
        if np.isinf(x) and np.isinf(y):
            continue
        if abs(x - y) > atol:
            return False

    return True


def validate_algorithms(cfg: BenchmarkConfig) -> None:
    """Small correctness check before benchmarks."""
    log("Running correctness checks...")

    rng = np.random.default_rng(cfg.seed)

    g_sparse = generate_sparse_connected_graph(
        25,
        cfg.sparse_extra_edges_per_node,
        weight_min=cfg.weight_min,
        weight_max=cfg.weight_max,
        rng=rng,
    )

    g_dense = generate_dense_graph(
        25,
        cfg.dense_edge_probability,
        weight_min=cfg.weight_min,
        weight_max=cfg.weight_max,
        rng=rng,
    )

    for graph_name, g in (("sparse", g_sparse), ("dense", g_dense)):
        for source in (0, min(5, g.n - 1)):
            d_arr = dijkstra_array(g.adj, source)
            d_heap = dijkstra_heap(g.adj, source)
            if not _dist_lists_close(d_arr, d_heap):
                raise ValueError(
                    f"Dijkstra correctness check failed on {graph_name} graph: "
                    "array and heap results differ."
                )

        f_py = floyd_warshall_python(g.matrix)
        f_np = floyd_warshall_numpy(g.matrix)
        if not np.allclose(f_py, f_np, equal_nan=True):
            raise ValueError(
                f"Floyd–Warshall correctness check failed on {graph_name} graph: "
                "python and numpy results differ."
            )

    log("Correctness checks passed.\n")


def _format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.2f}m"
    hours = minutes / 60
    return f"{hours:.2f}h"


def run_benchmarks(cfg: BenchmarkConfig, *, out_dir: Path) -> Path:
    """Run benchmarks and save raw + aggregated CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_algorithms(cfg)

    if cfg.dijkstra_sizes != cfg.floyd_sizes:
        log("WARNING: dijkstra_sizes and floyd_sizes differ.")
        log("This version benchmarks per size by zipping the two lists.")
        log("Only matching pairs by position will be processed.\n")

    rng = np.random.default_rng(cfg.seed)
    rows: list[BenchmarkRow] = []

    paired_sizes = list(zip(cfg.dijkstra_sizes, cfg.floyd_sizes))

    total_dijkstra_runs = len(paired_sizes) * 2 * cfg.benchmark_sources * cfg.repeats * 2
    total_floyd_runs = len(paired_sizes) * 2 * cfg.repeats * 2
    total_runs = total_dijkstra_runs + total_floyd_runs
    completed_runs = 0

    benchmark_start = time.perf_counter()

    def print_progress_header(
        algorithm: str,
        density: str,
        n: int,
        *,
        source: int | None,
        rep: int,
        total_rep: int,
        variant: str,
    ) -> None:
        nonlocal completed_runs

        completed_runs += 1
        percent = 100.0 * completed_runs / total_runs
        elapsed = time.perf_counter() - benchmark_start

        if source is None:
            log(
                f"[{completed_runs}/{total_runs} | {percent:6.2f}% | elapsed={_format_seconds(elapsed)}] "
                f"[{algorithm}][{density:<6}] n={n} repeat={rep}/{total_rep} variant={variant}"
            )
        else:
            log(
                f"[{completed_runs}/{total_runs} | {percent:6.2f}% | elapsed={_format_seconds(elapsed)}] "
                f"[{algorithm}][{density:<6}] n={n} source={source} repeat={rep}/{total_rep} variant={variant}"
            )

    def bench_dijkstra_for_graph(adj, *, n: int, m: int, density: str) -> None:
        log(f"  Generating Dijkstra benchmark on {density} graph: n={n}, m≈{m}")

        sources = rng.integers(0, n, size=cfg.benchmark_sources).tolist()

        for source in sources:
            log(f"  Warm-up for Dijkstra [{density}] n={n}, source={source}")
            for _ in range(cfg.warmup_runs):
                dijkstra_heap(adj, int(source))
                dijkstra_array(adj, int(source))

            for rep in range(cfg.repeats):
                order = ["heap", "array"] if rep % 2 == 0 else ["array", "heap"]

                for variant in order:
                    print_progress_header(
                        "Dijkstra",
                        density,
                        n,
                        source=int(source),
                        rep=rep + 1,
                        total_rep=cfg.repeats,
                        variant=variant,
                    )

                    if variant == "heap":
                        elapsed = _time_once(lambda: dijkstra_heap(adj, int(source)))
                    else:
                        elapsed = _time_once(lambda: dijkstra_array(adj, int(source)))

                    rows.append(
                        BenchmarkRow(
                            algorithm="dijkstra",
                            variant=variant,
                            density=density,
                            n=n,
                            m_undirected=m,
                            source=int(source),
                            repeat=rep,
                            seconds=elapsed,
                        )
                    )

    def bench_floyd_for_graph(matrix, *, n: int, m: int, density: str) -> None:
        log(f"  Generating Floyd benchmark on {density} graph: n={n}, m≈{m}")

        log(f"  Warm-up for Floyd [{density}] n={n}")
        for _ in range(cfg.warmup_runs):
            floyd_warshall_numpy(matrix)
            floyd_warshall_python(matrix)

        for rep in range(cfg.repeats):
            order = ["numpy", "python"] if rep % 2 == 0 else ["python", "numpy"]

            for variant in order:
                print_progress_header(
                    "Floyd",
                    density,
                    n,
                    source=None,
                    rep=rep + 1,
                    total_rep=cfg.repeats,
                    variant=variant,
                )

                if variant == "numpy":
                    elapsed = _time_once(lambda: floyd_warshall_numpy(matrix))
                else:
                    elapsed = _time_once(lambda: floyd_warshall_python(matrix))

                rows.append(
                    BenchmarkRow(
                        algorithm="floyd",
                        variant=variant,
                        density=density,
                        n=n,
                        m_undirected=m,
                        source=None,
                        repeat=rep,
                        seconds=elapsed,
                    )
                )

    log("Starting interleaved benchmarks by size...")
    for d_n, f_n in paired_sizes:
        block_start = time.perf_counter()
        log(f"\n==================== SIZE BLOCK ====================")
        log(f"Dijkstra size = {d_n}")
        log(f"Floyd size    = {f_n}")
        log("===================================================")

        # Dijkstra for this size
        log(f"\n=== Dijkstra n={d_n} ===")

        log(f"Creating sparse graph for Dijkstra n={d_n} ...")
        g_sparse_d = generate_sparse_connected_graph(
            d_n,
            cfg.sparse_extra_edges_per_node,
            weight_min=cfg.weight_min,
            weight_max=cfg.weight_max,
            rng=rng,
        )
        log(f"Finished sparse graph for Dijkstra n={d_n}, m={g_sparse_d.m_undirected}")
        bench_dijkstra_for_graph(
            g_sparse_d.adj,
            n=g_sparse_d.n,
            m=g_sparse_d.m_undirected,
            density="sparse",
        )

        log(f"Creating dense graph for Dijkstra n={d_n} ...")
        g_dense_d = generate_dense_graph(
            d_n,
            cfg.dense_edge_probability,
            weight_min=cfg.weight_min,
            weight_max=cfg.weight_max,
            rng=rng,
        )
        log(f"Finished dense graph for Dijkstra n={d_n}, m={g_dense_d.m_undirected}")
        bench_dijkstra_for_graph(
            g_dense_d.adj,
            n=g_dense_d.n,
            m=g_dense_d.m_undirected,
            density="dense",
        )

        d_elapsed = time.perf_counter() - block_start
        log(f"Finished Dijkstra size block n={d_n} in {_format_seconds(d_elapsed)}")

        # Floyd for this size
        floyd_start = time.perf_counter()
        log(f"\n=== Floyd n={f_n} ===")

        log(f"Creating sparse graph for Floyd n={f_n} ...")
        g_sparse_f = generate_sparse_connected_graph(
            f_n,
            cfg.sparse_extra_edges_per_node,
            weight_min=cfg.weight_min,
            weight_max=cfg.weight_max,
            rng=rng,
        )
        log(f"Finished sparse graph for Floyd n={f_n}, m={g_sparse_f.m_undirected}")
        bench_floyd_for_graph(
            g_sparse_f.matrix,
            n=g_sparse_f.n,
            m=g_sparse_f.m_undirected,
            density="sparse",
        )

        log(f"Creating dense graph for Floyd n={f_n} ...")
        g_dense_f = generate_dense_graph(
            f_n,
            cfg.dense_edge_probability,
            weight_min=cfg.weight_min,
            weight_max=cfg.weight_max,
            rng=rng,
        )
        log(f"Finished dense graph for Floyd n={f_n}, m={g_dense_f.m_undirected}")
        bench_floyd_for_graph(
            g_dense_f.matrix,
            n=g_dense_f.n,
            m=g_dense_f.m_undirected,
            density="dense",
        )

        f_elapsed = time.perf_counter() - floyd_start
        total_block_elapsed = time.perf_counter() - block_start
        log(f"Finished Floyd size block n={f_n} in {_format_seconds(f_elapsed)}")
        log(f"Finished full size block in {_format_seconds(total_block_elapsed)}")

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
            "source",
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