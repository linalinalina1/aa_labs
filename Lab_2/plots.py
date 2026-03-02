from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt

from benchmark import Measurement


def generate_plots(measurements: List[Measurement], *, graphs_dir: Path) -> List[Path]:
    """Generate PNG graphs into artifacts/graphs.

    Requirements implemented:
    - Linear x and linear y axes.
    - X axis shows all measured sizes (ticks at every n).
    - One image per input type containing 4 subplots (2x2), one per algorithm
      family, comparing basic vs optimized.
    - Also generates one image per algorithm family comparing basic vs
      optimized across input types (subplots).
    """

    graphs_dir.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []

    for input_type, ms in _group_by_input(measurements).items():
        saved.append(_plot_all_variants_for_input(input_type, ms, graphs_dir))
        saved.append(_plot_by_algorithm_for_input(input_type, ms, graphs_dir))

    for algo, ms in _group_by_algorithm_family(measurements).items():
        path = _plot_by_input_for_algorithm(algo, ms, graphs_dir)
        if path is not None:
            saved.append(path)

    return saved


def _plot_all_variants_for_input(input_type: str, measurements: List[Measurement], graphs_dir: Path) -> Path:
    # For one input type, plot all available algorithm variants on one chart.
    by_key: Dict[str, List[Measurement]] = defaultdict(list)
    for m in measurements:
        by_key[m.algorithm_key].append(m)

    key_order = [
        "quicksort_basic",
        "quicksort_optimized",
        "mergesort_basic",
        "mergesort_optimized",
        "heapsort_basic",
        "heapsort_optimized",
        "patiencesort_basic",
        "patiencesort_optimized",
    ]

    fig, ax = plt.subplots(figsize=(10.8, 6.2))
    all_xs: set[int] = set()

    for key in key_order:
        series = by_key.get(key, [])
        if not series:
            continue
        points = sorted([(m.n, m.seconds_median) for m in series], key=lambda p: p[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        all_xs.update(xs)
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=key)

    if all_xs:
        ax.set_xticks(sorted(all_xs))

    ax.set_title(f"All algorithm variants comparison (input: {input_type})")
    ax.set_xlabel("n")
    ax.set_ylabel("median seconds")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.set_ylim(bottom=0)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=8, ncol=2)

    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_horizontalalignment("right")

    fig.tight_layout()

    out_path = graphs_dir / f"compare_all__{input_type}.png"
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)
    return out_path


def _group_by_input(measurements: Iterable[Measurement]) -> Dict[str, List[Measurement]]:
    groups: Dict[str, List[Measurement]] = defaultdict(list)
    for m in measurements:
        groups[m.input_type].append(m)
    return groups


def _group_by_algorithm_family(measurements: Iterable[Measurement]) -> Dict[str, List[Measurement]]:
    groups: Dict[str, List[Measurement]] = defaultdict(list)
    for m in measurements:
        groups[m.algorithm].append(m)
    return groups


def _plot_by_algorithm_for_input(input_type: str, measurements: List[Measurement], graphs_dir: Path) -> Path:
    # For one input type, draw 4 subplots (one per algorithm family): basic vs optimized.
    by_algo: Dict[str, List[Measurement]] = defaultdict(list)
    for m in measurements:
        by_algo[m.algorithm].append(m)

    algo_order = ["quicksort", "mergesort", "heapsort", "patiencesort"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), squeeze=False)

    for idx, algo in enumerate(algo_order):
        ax = axes[idx // 2][idx % 2]
        ms = by_algo.get(algo, [])

        basic_points = sorted([(m.n, m.seconds_median) for m in ms if m.variant == "basic"], key=lambda p: p[0])
        opt_points = sorted([(m.n, m.seconds_median) for m in ms if m.variant == "optimized"], key=lambda p: p[0])

        if basic_points:
            xs = [p[0] for p in basic_points]
            ys = [p[1] for p in basic_points]
            ax.plot(xs, ys, marker="o", linewidth=1.6, label="basic")
            ax.set_xticks(xs)

        if opt_points:
            xs = [p[0] for p in opt_points]
            ys = [p[1] for p in opt_points]
            ax.plot(xs, ys, marker="o", linewidth=1.6, label="optimized")
            ax.set_xticks(xs)

        ax.set_title(algo)
        ax.set_xlabel("n")
        ax.set_ylabel("median seconds")
        ax.grid(True, linestyle=":", linewidth=0.6)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9)

        # Ensure y-axis starts at 0 for "ascending" readability.
        ax.set_ylim(bottom=0)

        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_horizontalalignment("right")

    fig.suptitle(f"Basic vs optimized per algorithm (input: {input_type})", y=0.99)
    fig.tight_layout()

    out_path = graphs_dir / f"by_algo__{input_type}.png"
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)
    return out_path


def _plot_by_input_for_algorithm(algo: str, measurements: List[Measurement], graphs_dir: Path) -> Path | None:
    # For one algorithm family, draw one subplot per input type: basic vs optimized.
    by_variant: Dict[str, List[Measurement]] = defaultdict(list)
    by_input: Dict[str, List[Measurement]] = defaultdict(list)

    for m in measurements:
        by_variant[m.variant].append(m)
        by_input[m.input_type].append(m)

    if "basic" not in by_variant or "optimized" not in by_variant:
        return None

    input_types = sorted(by_input.keys())
    cols = 2
    rows = (len(input_types) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.9 * rows), squeeze=False)

    for idx, input_type in enumerate(input_types):
        ax = axes[idx // cols][idx % cols]

        ms = by_input[input_type]
        basic_points = sorted([(m.n, m.seconds_median) for m in ms if m.variant == "basic"], key=lambda p: p[0])
        opt_points = sorted([(m.n, m.seconds_median) for m in ms if m.variant == "optimized"], key=lambda p: p[0])

        if basic_points:
            xs = [p[0] for p in basic_points]
            ys = [p[1] for p in basic_points]
            ax.plot(xs, ys, marker="o", linewidth=1.6, label="basic")
            ax.set_xticks(xs)

        if opt_points:
            xs = [p[0] for p in opt_points]
            ys = [p[1] for p in opt_points]
            ax.plot(xs, ys, marker="o", linewidth=1.6, label="optimized")
            ax.set_xticks(xs)

        ax.set_title(input_type)
        ax.set_xlabel("n")
        ax.set_ylabel("median seconds")
        ax.grid(True, linestyle=":", linewidth=0.6)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9)
        ax.set_ylim(bottom=0)

        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_horizontalalignment("right")

    # Hide unused axes.
    for idx in range(len(input_types), rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    fig.suptitle(f"{algo}: basic vs optimized (all sizes shown)", y=0.99)
    fig.tight_layout()

    out_path = graphs_dir / f"compare_variants__{algo}.png"
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)
    return out_path
