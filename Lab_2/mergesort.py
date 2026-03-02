from __future__ import annotations

import time
from pathlib import Path
from statistics import median
from typing import List, Sequence


# MergeSort is a stable divide-and-conquer algorithm. It recursively sorts the
# left and right halves, then merges them into a sorted result. Its time
# complexity is O(n log n) for all inputs; it typically uses O(n) auxiliary
# memory for merging.


def mergesort_basic(arr: Sequence[int]) -> list[int]:
    """MergeSort (basic): standard top-down mergesort using an auxiliary array."""

    a = list(arr)
    n = len(a)
    if n < 2:
        return a

    aux = a.copy()
    _mergesort(a, aux, 0, n, optimized=False)
    return a


def mergesort_optimized(arr: Sequence[int]) -> list[int]:
    """MergeSort (optimized): skip merge when already ordered + insertion cutoff.

    Optimizations:
    1) If after sorting both halves we have a[mid-1] <= a[mid], the merge step
       is unnecessary.
    2) For small subarrays, insertion sort can be faster due to lower overhead.
    """

    a = list(arr)
    n = len(a)
    if n < 2:
        return a

    aux = a.copy()
    _mergesort(a, aux, 0, n, optimized=True)
    return a


def _mergesort(a: List[int], aux: List[int], lo: int, hi: int, *, optimized: bool) -> None:
    if hi - lo <= 1:
        return

    if optimized and (hi - lo) <= 32:
        _insertion_sort(a, lo, hi)
        return

    mid = (lo + hi) // 2
    _mergesort(a, aux, lo, mid, optimized=optimized)
    _mergesort(a, aux, mid, hi, optimized=optimized)

    if optimized and a[mid - 1] <= a[mid]:
        return

    _merge(a, aux, lo, mid, hi)


def _merge(a: List[int], aux: List[int], lo: int, mid: int, hi: int) -> None:
    # Copy slice into auxiliary array.
    aux[lo:hi] = a[lo:hi]

    i = lo
    j = mid
    k = lo

    while i < mid and j < hi:
        if aux[i] <= aux[j]:
            a[k] = aux[i]
            i += 1
        else:
            a[k] = aux[j]
            j += 1
        k += 1

    # Copy remaining left part (right part is already in place).
    while i < mid:
        a[k] = aux[i]
        i += 1
        k += 1


def _insertion_sort(a: List[int], lo: int, hi: int) -> None:
    for i in range(lo + 1, hi):
        x = a[i]
        j = i - 1
        while j >= lo and a[j] > x:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = x


def _demo() -> Path:
    """Run a small benchmark for MergeSort only and save a demo graph."""

    import matplotlib.pyplot as plt

    from inputs import derive_seed, make_input

    root = Path(__file__).resolve().parent
    graphs_dir = root / "artifacts" / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    kind = "random"
    base_seed = 2026
    sizes = [1000, 5000, 10000, 20000]

    series = {
        "MergeSort (basic)": [],
        "MergeSort (optimized)": [],
    }

    for n in sizes:
        data = make_input(kind=kind, n=n, seed=derive_seed(base_seed, kind=kind, n=n))
        expected = sorted(data)

        for label, fn in [
            ("MergeSort (basic)", mergesort_basic),
            ("MergeSort (optimized)", mergesort_optimized),
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
    plt.title("MergeSort demo benchmark (random input)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend(fontsize=8)

    out_path = graphs_dir / "demo__mergesort.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


if __name__ == "__main__":
    path = _demo()
    print(f"Saved: {path}")
