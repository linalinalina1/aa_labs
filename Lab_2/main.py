from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from benchmark import DEFAULT_SIZES, benchmark_selected, print_console_tables
from inputs import INPUT_TYPES
from plots import generate_plots


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AA Lab 2: Sorting algorithms benchmark")

    parser.add_argument(
        "--algo",
        choices=["quicksort", "mergesort", "heapsort", "patiencesort", "all"],
        default="all",
        help="Select algorithm family (default: all)",
    )
    parser.add_argument(
        "--variant",
        choices=["basic", "optimized", "all"],
        default="all",
        help="Select variant (default: all)",
    )
    parser.add_argument(
        "--input",
        choices=INPUT_TYPES + ["all"],
        default="all",
        help="Select input type (default: all)",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_SIZES),
        help="List of input sizes (default: 1000 2000 5000 10000 20000 50000 100000 200000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller size set for a fast smoke run",
    )

    parser.add_argument(
        "--verify-each-repeat",
        action="store_true",
        help="Verify correctness on every repeat run (slower)",
    )

    parser.add_argument(
        "--no-console-tables",
        action="store_true",
        help="Skip printing console timing tables",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"

    results_dir = artifacts / "results"
    graphs_dir = artifacts / "graphs"

    sizes: List[int]
    if args.quick:
        sizes = [500, 1000, 2000, 5000, 10000]
    else:
        sizes = list(args.sizes)

    print("Running benchmarks...")
    print(f"Algorithms: {args.algo} | Variant: {args.variant} | Input: {args.input}")
    print(f"Sizes: {sizes}")

    measurements = benchmark_selected(
        algo=args.algo,
        variant=args.variant,
        input_type=args.input,
        sizes=sizes,
        base_seed=2026,
        artifacts_dir=results_dir,
        verify_correctness=True,
        verify_each_repeat=bool(args.verify_each_repeat),
        show_progress=True,
    )

    print("\nGenerating plots...")
    graphs_dir.mkdir(parents=True, exist_ok=True)
    for path in graphs_dir.glob("*.png"):
        if not path.name.startswith("demo__"):
            path.unlink()
    generate_plots(measurements, graphs_dir=graphs_dir)

    if args.no_console_tables:
        printed_tables = False
    else:
        print("\nConsole tables:\n")
        print_console_tables(measurements)
        printed_tables = True

    print("\nDone.")
    print(f"Results: {results_dir}")
    print(f"Graphs:   {graphs_dir}")
    print(f"Console tables printed: {printed_tables}")


if __name__ == "__main__":
    main()
