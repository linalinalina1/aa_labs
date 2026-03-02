from __future__ import annotations

import random
import time
from pathlib import Path
from statistics import median
from typing import List, Sequence, Tuple


# QuickSort is a divide-and-conquer sorting algorithm that partitions an array
# around a pivot, then recursively sorts the left and right partitions.
# Its expected time complexity is O(n log n), but its worst case is O(n^2)
# when partitioning is highly unbalanced (e.g., deterministic bad pivots).


def quicksort_basic(arr: Sequence[int]) -> list[int]:
    """QuickSort (basic): classic 2-way in-place partition (Lomuto), on a copy.

    Properties:
    - Returns a sorted copy of the input (does not mutate the caller's sequence).
    - Uses an explicit stack instead of recursion to avoid recursion depth issues.

    Complexity (typical):
    - Expected time: O(n log n)
    - Worst-case time: O(n^2)
    - Extra space: O(log n) stack on average (O(n) worst-case)
    """

    a = list(arr)
    if len(a) < 2:
        return a

    stack: List[Tuple[int, int]] = [(0, len(a) - 1)]

    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue

        p = _partition_lomuto(a, lo, hi)

        # Push larger range first to keep stack shallow.
        left = (lo, p - 1)
        right = (p + 1, hi)
        if (left[1] - left[0]) > (right[1] - right[0]):
            if left[0] < left[1]:
                stack.append(left)
            if right[0] < right[1]:
                stack.append(right)
        else:
            if right[0] < right[1]:
                stack.append(right)
            if left[0] < left[1]:
                stack.append(left)

    return a


def _partition_lomuto(a: List[int], lo: int, hi: int) -> int:
    pivot = a[hi]
    i = lo
    for j in range(lo, hi):
        if a[j] <= pivot:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]
    return i


def quicksort_optimized(arr: Sequence[int]) -> list[int]:
    """QuickSort (optimized): random pivot + 3-way partition (Dutch flag).

    Differences vs basic:
    - Randomized pivot selection reduces the probability of worst-case splits.
    - 3-way partitioning groups equal keys together, improving performance when
      the input contains many duplicates.

    Reproducibility:
    - A deterministic RNG seed is used so benchmarks are repeatable.
    """

    a = list(arr)
    if len(a) < 2:
        return a

    rng = random.Random(123)
    stack: List[Tuple[int, int]] = [(0, len(a) - 1)]

    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue

        pivot_index = rng.randrange(lo, hi + 1)
        pivot = a[pivot_index]

        lt, gt = _partition_3way(a, lo, hi, pivot)

        left = (lo, lt - 1)
        right = (gt + 1, hi)
        if (left[1] - left[0]) > (right[1] - right[0]):
            if left[0] < left[1]:
                stack.append(left)
            if right[0] < right[1]:
                stack.append(right)
        else:
            if right[0] < right[1]:
                stack.append(right)
            if left[0] < left[1]:
                stack.append(left)

    return a


def _partition_3way(a: List[int], lo: int, hi: int, pivot: int) -> tuple[int, int]:
    lt = lo
    i = lo
    gt = hi

    while i <= gt:
        if a[i] < pivot:
            a[lt], a[i] = a[i], a[lt]
            lt += 1
            i += 1
        elif a[i] > pivot:
            a[gt], a[i] = a[i], a[gt]
            gt -= 1
        else:
            i += 1

    return lt, gt


def _demo() -> Path:
    """Run a small benchmark for QuickSort only and save a demo graph."""

    import matplotlib.pyplot as plt

    from inputs import derive_seed, make_input

    root = Path(__file__).resolve().parent
    graphs_dir = root / "artifacts" / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    kind = "random"
    base_seed = 2026
    sizes = [1000, 5000, 10000, 20000]

    series = {
        "QuickSort (basic)": [],
        "QuickSort (optimized)": [],
    }

    for n in sizes:
        data = make_input(kind=kind, n=n, seed=derive_seed(base_seed, kind=kind, n=n))
        expected = sorted(data)

        for label, fn in [
            ("QuickSort (basic)", quicksort_basic),
            ("QuickSort (optimized)", quicksort_optimized),
        ]:
            times: List[float] = []
            repeats = 7 if n <= 5000 else 5
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = fn(data)
                t1 = time.perf_counter()
                if out != expected:
                    raise AssertionError(f"Incorrect result for {label} at n={n}")
                times.append(t1 - t0)
            series[label].append(float(median(times)))

    plt.figure(figsize=(7.5, 4.8))
    for label, ys in series.items():
        plt.plot(sizes, ys, marker="o", linewidth=1.6, label=label)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Input size n")
    plt.ylabel("Median time (s)")
    plt.title("QuickSort demo benchmark (random input)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend(fontsize=8)

    out_path = graphs_dir / "demo__quicksort.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


if __name__ == "__main__":
    path = _demo()
    print(f"Saved: {path}")
