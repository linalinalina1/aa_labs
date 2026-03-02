from __future__ import annotations

import time
from pathlib import Path
from statistics import median
from typing import List, Sequence


# HeapSort builds a binary heap (here, a max-heap) and repeatedly extracts the
# maximum element, placing it at the end of the array. HeapSort runs in
# O(n log n) time for all inputs and sorts in-place (O(1) extra space), but it
# is not stable.


def heapsort_basic(arr: Sequence[int]) -> list[int]:
    """HeapSort (basic): build max-heap + recursive heapify (sift-down)."""

    a = list(arr)
    n = len(a)

    for i in range(n // 2 - 1, -1, -1):
        _sift_down_recursive(a, i, n)

    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        _sift_down_recursive(a, 0, end)

    return a


def _sift_down_recursive(a: List[int], root: int, heap_size: int) -> None:
    left = 2 * root + 1
    right = left + 1
    largest = root

    if left < heap_size and a[left] > a[largest]:
        largest = left
    if right < heap_size and a[right] > a[largest]:
        largest = right

    if largest != root:
        a[root], a[largest] = a[largest], a[root]
        _sift_down_recursive(a, largest, heap_size)


def heapsort_optimized(arr: Sequence[int]) -> list[int]:
    """HeapSort (optimized): build max-heap + iterative sift-down.

    The iterative version avoids Python recursion overhead in the heapify step.
    """

    a = list(arr)
    n = len(a)

    for i in range(n // 2 - 1, -1, -1):
        _sift_down_iterative(a, i, n)

    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        _sift_down_iterative(a, 0, end)

    return a


def _sift_down_iterative(a: List[int], root: int, heap_size: int) -> None:
    while True:
        left = 2 * root + 1
        right = left + 1
        largest = root

        if left < heap_size and a[left] > a[largest]:
            largest = left
        if right < heap_size and a[right] > a[largest]:
            largest = right

        if largest == root:
            return

        a[root], a[largest] = a[largest], a[root]
        root = largest


def _demo() -> Path:
    """Run a small benchmark for HeapSort only and save a demo graph."""

    import matplotlib.pyplot as plt

    from inputs import derive_seed, make_input

    root = Path(__file__).resolve().parent
    graphs_dir = root / "artifacts" / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    kind = "random"
    base_seed = 2026
    sizes = [1000, 5000, 10000, 20000]

    series = {
        "HeapSort (basic)": [],
        "HeapSort (optimized)": [],
    }

    for n in sizes:
        data = make_input(kind=kind, n=n, seed=derive_seed(base_seed, kind=kind, n=n))
        expected = sorted(data)

        for label, fn in [
            ("HeapSort (basic)", heapsort_basic),
            ("HeapSort (optimized)", heapsort_optimized),
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
    plt.title("HeapSort demo benchmark (random input)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend(fontsize=8)

    out_path = graphs_dir / "demo__heapsort.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


if __name__ == "__main__":
    path = _demo()
    print(f"Saved: {path}")
