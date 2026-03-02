from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Iterable, List, Optional, Sequence

from inputs import INPUT_TYPES, derive_seed, make_input


Sorter = Callable[[Sequence[int]], list[int]]


DEFAULT_SIZES = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]

# Size caps to keep specific worst-case combinations from dominating runtime.
# The cap is applied per (algorithm family, variant, input type). Any run with
# n > cap will be skipped for that specific algorithm variant only; other
# variants still run.
#
# Requested caps (March 2026):
# - QuickSort (basic only): cap at 50_000 for sorted, reversed, nearly_sorted, almost_reversed, all_equal
# - Patience Sort (basic only): cap at 50_000 for sorted, nearly_sorted
SIZE_CAPS: dict[tuple[str, str, str], int] = {
    ("quicksort", "basic", "sorted"): 50_000,
    ("quicksort", "basic", "reversed"): 50_000,
    ("quicksort", "basic", "nearly_sorted"): 50_000,
    ("quicksort", "basic", "almost_reversed"): 50_000,
    ("quicksort", "basic", "all_equal"): 50_000,
    ("patiencesort", "basic", "sorted"): 50_000,
    ("patiencesort", "basic", "nearly_sorted"): 50_000,
}


def _size_cap(*, algorithm_family: str, variant: str, input_type: str) -> int | None:
    return SIZE_CAPS.get((algorithm_family, variant, input_type))


@dataclass(frozen=True)
class Algorithm:
    family: str  # quicksort / mergesort / heapsort / patiencesort
    variant: str  # basic / optimized
    key: str
    name: str
    sort: Sorter


@dataclass(frozen=True)
class RunConfig:
    sizes: List[int]
    input_types: List[str]
    base_seed: int


@dataclass(frozen=True)
class Measurement:
    input_type: str
    n: int
    algorithm: str
    variant: str
    algorithm_key: str
    algorithm_name: str
    repeats: int
    seconds_median: float


def repeats_for_size(n: int) -> int:
    """Adaptive repeats to keep end-to-end runtime reasonable."""

    if n <= 1_000:
        return 15
    if n <= 5_000:
        return 10
    if n <= 10_000:
        return 7
    if n <= 20_000:
        return 5
    if n <= 50_000:
        return 3
    if n <= 100_000:
        return 2
    return 1


def default_config() -> RunConfig:
    return RunConfig(sizes=list(DEFAULT_SIZES), input_types=list(INPUT_TYPES), base_seed=2026)


def get_all_algorithms() -> List[Algorithm]:
    """Return all algorithms (basic + optimized) available for the lab."""

    from heapsort import heapsort_basic, heapsort_optimized
    from mergesort import mergesort_basic, mergesort_optimized
    from patiencesort import patiencesort_basic, patiencesort_optimized
    from quicksort import quicksort_basic, quicksort_optimized

    return [
        Algorithm("quicksort", "basic", "quicksort_basic", "QuickSort (basic)", quicksort_basic),
        Algorithm(
            "quicksort",
            "optimized",
            "quicksort_optimized",
            "QuickSort (optimized)",
            quicksort_optimized,
        ),
        Algorithm("mergesort", "basic", "mergesort_basic", "MergeSort (basic)", mergesort_basic),
        Algorithm(
            "mergesort",
            "optimized",
            "mergesort_optimized",
            "MergeSort (optimized)",
            mergesort_optimized,
        ),
        Algorithm("heapsort", "basic", "heapsort_basic", "HeapSort (basic)", heapsort_basic),
        Algorithm(
            "heapsort",
            "optimized",
            "heapsort_optimized",
            "HeapSort (optimized)",
            heapsort_optimized,
        ),
        Algorithm(
            "patiencesort",
            "basic",
            "patiencesort_basic",
            "Patience Sort (basic)",
            patiencesort_basic,
        ),
        Algorithm(
            "patiencesort",
            "optimized",
            "patiencesort_optimized",
            "Patience Sort (optimized)",
            patiencesort_optimized,
        ),
    ]


def select_algorithms(*, algo: str, variant: str) -> List[Algorithm]:
    """Filter the algorithm registry.

    Parameters:
    - algo: quicksort/mergesort/heapsort/patiencesort/all
    - variant: basic/optimized/all
    """

    algos = get_all_algorithms()

    if algo != "all":
        algos = [a for a in algos if a.family == algo]

    if variant != "all":
        algos = [a for a in algos if a.variant == variant]

    return algos


def run_benchmark(
    *,
    algorithms: List[Algorithm],
    config: RunConfig,
    artifacts_dir: Path,
    verify_correctness: bool = True,
    verify_each_repeat: bool = False,
    show_progress: bool = True,
    write_partial_results: bool = True,
) -> List[Measurement]:
    """Run selected algorithms across selected inputs and sizes.

    Requirements satisfied:
    - Uses time.perf_counter() for timing.
    - Uses the median across repeats.
    - Verifies correctness against Python's sorted().
    - Writes artifacts/results/results.csv and results.json.
    """

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    measurements: List[Measurement] = []

    total_steps = len(config.input_types) * len(config.sizes) * len(algorithms)
    step = 0

    for kind in config.input_types:
        if show_progress:
            print(f"\nInput type: {kind}")

        for n in config.sizes:
            seed = derive_seed(config.base_seed, kind=kind, n=n)
            base_data = make_input(kind=kind, n=n, seed=seed)
            expected = sorted(base_data) if verify_correctness else None

            for algo in algorithms:
                step += 1
                pct = 100.0 * step / max(1, total_steps)

                cap = _size_cap(algorithm_family=algo.family, variant=algo.variant, input_type=kind)
                if cap is not None and n > cap:
                    if show_progress:
                        print(f"[{pct:6.2f}%] n={n:>6} | {algo.key} | SKIP (cap={cap})")
                    continue

                reps = repeats_for_size(n)
                times: List[float] = []

                if show_progress:
                    print(f"[{pct:6.2f}%] n={n:>6} | {algo.key} | repeats={reps}")

                if verify_correctness and not verify_each_repeat:
                    out0 = algo.sort(base_data)
                    if out0 != expected:
                        raise AssertionError(f"Incorrect result for {algo.key} on {kind}, n={n}")

                for r in range(reps):
                    data = list(base_data)
                    t0 = time.perf_counter()
                    out = algo.sort(data)
                    t1 = time.perf_counter()

                    if verify_correctness and verify_each_repeat:
                        if out != expected:
                            raise AssertionError(
                                f"Incorrect result for {algo.key} on {kind}, n={n} (repeat {r + 1}/{reps})"
                            )

                    times.append(t1 - t0)

                measurements.append(
                    Measurement(
                        input_type=kind,
                        n=n,
                        algorithm=algo.family,
                        variant=algo.variant,
                        algorithm_key=algo.key,
                        algorithm_name=algo.name,
                        repeats=reps,
                        seconds_median=float(median(times)),
                    )
                )

            if write_partial_results:
                write_results(measurements, artifacts_dir=artifacts_dir)

    write_results(measurements, artifacts_dir=artifacts_dir)
    return measurements


def write_results(measurements: List[Measurement], *, artifacts_dir: Path) -> None:
    json_path = artifacts_dir / "results.json"
    csv_path = artifacts_dir / "results.csv"

    payload = [asdict(m) for m in measurements]
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_type",
                "n",
                "algorithm",
                "variant",
                "algorithm_key",
                "algorithm_name",
                "repeats",
                "seconds_median",
            ],
        )
        writer.writeheader()
        writer.writerows(payload)


def load_results(*, artifacts_dir: Path) -> List[Measurement]:
    data = json.loads((artifacts_dir / "results.json").read_text(encoding="utf-8"))
    out: List[Measurement] = []
    for row in data:
        out.append(
            Measurement(
                input_type=str(row["input_type"]),
                n=int(row["n"]),
                algorithm=str(row["algorithm"]),
                variant=str(row["variant"]),
                algorithm_key=str(row["algorithm_key"]),
                algorithm_name=str(row["algorithm_name"]),
                repeats=int(row["repeats"]),
                seconds_median=float(row["seconds_median"]),
            )
        )
    return out


def benchmark_selected(
    *,
    algo: str = "all",
    variant: str = "all",
    input_type: str = "all",
    sizes: Optional[List[int]] = None,
    base_seed: int = 2026,
    artifacts_dir: Path,
    verify_correctness: bool = True,
    verify_each_repeat: bool = False,
    show_progress: bool = True,
) -> List[Measurement]:
    """Convenience entrypoint that implements the requested filtering."""

    algorithms = select_algorithms(algo=algo, variant=variant)

    if input_type == "all":
        input_types = list(INPUT_TYPES)
    else:
        if input_type not in INPUT_TYPES:
            raise ValueError(f"Unknown input type: {input_type}")
        input_types = [input_type]

    config = RunConfig(
        sizes=list(DEFAULT_SIZES if sizes is None else sizes),
        input_types=input_types,
        base_seed=base_seed,
    )

    return run_benchmark(
        algorithms=algorithms,
        config=config,
        artifacts_dir=artifacts_dir,
        verify_correctness=verify_correctness,
        verify_each_repeat=verify_each_repeat,
        show_progress=show_progress,
        write_partial_results=True,
    )


def print_console_tables(measurements: List[Measurement]) -> None:
    """Print clean timing tables to the console.

    Requirements:
    - One table per input type.
    - Columns are the algorithm variants (when present).
    - Times are formatted with fixed precision (6 decimals).
    - Includes a Fastest column.
    - Also prints a final summary table counting wins per variant.
    """

    by_input: dict[str, list[Measurement]] = {}
    for m in measurements:
        by_input.setdefault(m.input_type, []).append(m)

    # Determine a stable column order based on known variants.
    all_keys = sorted({m.algorithm_key for m in measurements})
    ordered_keys = _stable_algorithm_key_order(all_keys)

    # 1) Per input type: all 8 algorithm variants together.
    win_counts_by_input: dict[str, dict[str, int]] = {}
    for input_type in sorted(by_input.keys()):
        print(f"INPUT TYPE (all variants): {input_type}\n")
        win_counts_by_input[input_type] = _print_one_input_table(by_input[input_type], ordered_keys)
        print()

    # 2) Per input type, per algorithm family: basic vs optimized.
    _print_pairwise_variant_tables(by_input)

    # 3) Per algorithm family: compare input types (aggregated over sizes).
    _print_algorithm_input_comparison_tables(measurements)

    _print_final_summary_table(win_counts_by_input, ordered_keys)


def _stable_algorithm_key_order(keys: list[str]) -> list[str]:
    preferred = [
        "quicksort_basic",
        "quicksort_optimized",
        "mergesort_basic",
        "mergesort_optimized",
        "heapsort_basic",
        "heapsort_optimized",
        "patiencesort_basic",
        "patiencesort_optimized",
    ]
    ordered = [k for k in preferred if k in keys]
    ordered.extend([k for k in keys if k not in ordered])
    return ordered


def _short_label(algorithm_key: str) -> str:
    fam, var = algorithm_key.split("_", 1)
    fam_map = {
        "quicksort": "QS",
        "mergesort": "MS",
        "heapsort": "HS",
        "patiencesort": "PS",
    }
    var_map = {
        "basic": "b",
        "optimized": "o",
    }
    fam_label = fam_map.get(fam, fam[:2].upper())
    var_label = var_map.get(var, var[:1].lower())
    return f"{fam_label}_{var_label}"


def _format_seconds(x: float) -> str:
    return f"{x:.6f}"


def _print_one_input_table(ms: list[Measurement], ordered_keys: list[str]) -> dict[str, int]:
    sizes = sorted({m.n for m in ms})

    time_map: dict[tuple[int, str], float] = {}
    present_keys = sorted({m.algorithm_key for m in ms})
    cols = [k for k in ordered_keys if k in present_keys]

    for m in ms:
        time_map[(m.n, m.algorithm_key)] = m.seconds_median

    headers = ["n"] + [_short_label(k) for k in cols] + ["Fastest", "Slowest"]
    rows: list[list[str]] = []

    win_counts: dict[str, int] = {k: 0 for k in cols}

    for n in sizes:
        best_key = None
        best_time = None
        worst_key = None
        worst_time = None
        row: list[str] = [str(n)]

        for k in cols:
            t = time_map.get((n, k), float("nan"))
            row.append(_format_seconds(t) if t == t else "-")
            if t == t:
                if best_time is None or t < best_time:
                    best_time = t
                    best_key = k
                if worst_time is None or t > worst_time:
                    worst_time = t
                    worst_key = k

        if best_key is None:
            row.append("-")
        else:
            win_counts[best_key] += 1
            row.append(_short_label(best_key))

        row.append(_short_label(worst_key) if worst_key is not None else "-")

        rows.append(row)

    _print_ascii_table(headers, rows)
    return win_counts


def _print_final_summary_table(
    win_counts_by_input: dict[str, dict[str, int]],
    ordered_keys: list[str],
) -> None:
    print("FINAL SUMMARY (wins per variant across sizes)")

    # Determine which variants are present at all.
    all_present_keys = sorted({k for d in win_counts_by_input.values() for k in d.keys()})
    cols = [k for k in ordered_keys if k in all_present_keys]

    headers = ["input_type"] + [_short_label(k) for k in cols] + ["Winner", "Loser"]
    rows: list[list[str]] = []

    for input_type in sorted(win_counts_by_input.keys()):
        counts = win_counts_by_input[input_type]
        winner_key = None
        winner_count = None
        loser_key = None
        loser_count = None

        row = [input_type]
        for k in cols:
            c = int(counts.get(k, 0))
            row.append(str(c))
            if winner_count is None or c > winner_count:
                winner_count = c
                winner_key = k
            if loser_count is None or c < loser_count:
                loser_count = c
                loser_key = k

        row.append(_short_label(winner_key) if winner_key is not None else "-")
        row.append(_short_label(loser_key) if loser_key is not None else "-")
        rows.append(row)

    _print_ascii_table(headers, rows)


def _print_pairwise_variant_tables(by_input: dict[str, list[Measurement]]) -> None:
    for input_type in sorted(by_input.keys()):
        ms = by_input[input_type]
        by_family: dict[str, list[Measurement]] = {}
        for m in ms:
            by_family.setdefault(m.algorithm, []).append(m)

        for family in ["quicksort", "mergesort", "heapsort", "patiencesort"]:
            fam_ms = by_family.get(family, [])
            if not fam_ms:
                continue

            time_map: dict[tuple[int, str], float] = {}
            sizes = sorted({m.n for m in fam_ms})
            variants = sorted({m.variant for m in fam_ms})
            if "basic" not in variants or "optimized" not in variants:
                continue

            for m in fam_ms:
                time_map[(m.n, m.variant)] = m.seconds_median

            print(f"INPUT: {input_type} | ALGORITHM: {family} (basic vs optimized)\n")

            headers = ["n", "basic", "optimized", "Winner", "Loser"]
            rows: list[list[str]] = []
            for n in sizes:
                tb = time_map.get((n, "basic"), float("nan"))
                to = time_map.get((n, "optimized"), float("nan"))

                winner = "-"
                loser = "-"
                if tb == tb and to == to:
                    winner = "optimized" if to < tb else "basic"
                    loser = "basic" if winner == "optimized" else "optimized"
                elif tb == tb:
                    winner = "basic"
                    loser = "-"
                elif to == to:
                    winner = "optimized"
                    loser = "-"

                rows.append(
                    [
                        str(n),
                        _format_seconds(tb) if tb == tb else "-",
                        _format_seconds(to) if to == to else "-",
                        winner,
                        loser,
                    ]
                )

            _print_ascii_table(headers, rows)
            print()


def _print_algorithm_input_comparison_tables(measurements: list[Measurement]) -> None:
    by_family: dict[str, list[Measurement]] = {}
    for m in measurements:
        by_family.setdefault(m.algorithm, []).append(m)

    for family in ["quicksort", "mergesort", "heapsort", "patiencesort"]:
        fam_ms = by_family.get(family, [])
        if not fam_ms:
            continue

        agg: dict[tuple[str, str], list[float]] = {}
        for m in fam_ms:
            agg.setdefault((m.input_type, m.variant), []).append(m.seconds_median)

        rows: list[list[str]] = []
        for input_type in sorted({m.input_type for m in fam_ms}):
            basic_vals = agg.get((input_type, "basic"), [])
            opt_vals = agg.get((input_type, "optimized"), [])

            tb = float(median(basic_vals)) if basic_vals else float("nan")
            to = float(median(opt_vals)) if opt_vals else float("nan")

            winner = "-"
            loser = "-"
            if tb == tb and to == to:
                winner = "optimized" if to < tb else "basic"
                loser = "basic" if winner == "optimized" else "optimized"
            elif tb == tb:
                winner = "basic"
            elif to == to:
                winner = "optimized"

            rows.append(
                [
                    input_type,
                    _format_seconds(tb) if tb == tb else "-",
                    _format_seconds(to) if to == to else "-",
                    winner,
                    loser,
                ]
            )

        print(f"ALGORITHM: {family} | INPUT TYPES comparison (median over sizes)\n")
        headers = ["input_type", "basic", "optimized", "Winner", "Loser"]
        _print_ascii_table(headers, rows)
        print()


def _print_ascii_table(headers: list[str], rows: list[list[str]]) -> None:
    # Compute column widths.
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def sep() -> str:
        return "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def fmt_row(r: list[str]) -> str:
        parts: list[str] = []
        for i, cell in enumerate(r):
            # Right-align numeric columns (except first column which is n/input_type)
            is_first = i == 0
            is_numberish = (not is_first) and (cell.replace(".", "", 1).isdigit())
            if is_numberish:
                parts.append(" " + cell.rjust(widths[i]) + " ")
            else:
                parts.append(" " + cell.ljust(widths[i]) + " ")
        return "|" + "|".join(parts) + "|"

    print(sep())
    print(fmt_row(headers))
    print(sep())
    for r in rows:
        print(fmt_row(r))
    print(sep())
