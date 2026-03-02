from __future__ import annotations

import bisect
import heapq
import time
from pathlib import Path
from statistics import median
from typing import List, Sequence


# Patience sorting is inspired by the card game "patience" (solitaire). It
# constructs piles by repeatedly placing each element onto the leftmost pile
# whose top is >= the element. The number of piles relates to the length of the
# longest increasing subsequence.
# To obtain the full sorted order, we must merge the piles by repeatedly taking
# the smallest available pile top.


def patiencesort_basic(arr: Sequence[int]) -> list[int]:
    """Patience Sort (basic): patience piles + naive merge (intentionally slow).

    Basic variant characteristics:
    - Pile placement: linear scan over piles (O(n * k) where k is #piles).
    - Merge: naive selection of the minimum pile top by scanning all piles each
      step (O(n * k)). This is intentionally unoptimized to contrast with the
      optimized k-way heap merge.
    """

    piles: List[List[int]] = []

    for x in arr:
        placed = False
        for pile in piles:
            if pile[-1] >= x:
                pile.append(x)
                placed = True
                break
        if not placed:
            piles.append([x])

    return _merge_piles_naive(piles)


def patiencesort_optimized(arr: Sequence[int]) -> list[int]:
    """Patience Sort (optimized): patience piles + heapq k-way merge.

    Optimizations:
    - Pile placement uses binary search over the current pile tops (O(n log k)).
    - Merge uses a min-heap to perform k-way merge in O(n log k).
    """

    piles: List[List[int]] = []
    tops: List[int] = []

    for x in arr:
        idx = bisect.bisect_left(tops, x)  # leftmost pile with top >= x
        if idx == len(piles):
            piles.append([x])
            tops.append(x)
        else:
            piles[idx].append(x)
            tops[idx] = x

    return _merge_piles_heap(piles)


def _merge_piles_naive(piles: List[List[int]]) -> list[int]:
    active = [p for p in piles if p]
    out: List[int] = []

    while active:
        # Find smallest available pile top by scanning all piles.
        min_i = 0
        min_val = active[0][-1]
        for i in range(1, len(active)):
            v = active[i][-1]
            if v < min_val:
                min_val = v
                min_i = i

        out.append(min_val)
        active[min_i].pop()
        if not active[min_i]:
            active.pop(min_i)

    return out


def _merge_piles_heap(piles: List[List[int]]) -> list[int]:
    heap: List[tuple[int, int]] = []
    for i, pile in enumerate(piles):
        if pile:
            heapq.heappush(heap, (pile[-1], i))

    out: List[int] = []
    while heap:
        value, i = heapq.heappop(heap)
        out.append(value)
        piles[i].pop()
        if piles[i]:
            heapq.heappush(heap, (piles[i][-1], i))

    return out


def _demo() -> Path:
    """Run a small benchmark for Patience Sort only and save a demo graph."""

    import matplotlib.pyplot as plt

    from inputs import derive_seed, make_input

    root = Path(__file__).resolve().parent
    graphs_dir = root / "artifacts" / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    kind = "few_unique"  # highlights the effect of pile/merge strategies
    base_seed = 2026
    sizes = [500, 1000, 2000, 5000]

    series = {
        "Patience Sort (basic)": [],
        "Patience Sort (optimized)": [],
    }

    for n in sizes:
        data = make_input(kind=kind, n=n, seed=derive_seed(base_seed, kind=kind, n=n))
        expected = sorted(data)

        for label, fn in [
            ("Patience Sort (basic)", patiencesort_basic),
            ("Patience Sort (optimized)", patiencesort_optimized),
        ]:
            times: List[float] = []
            repeats = 5
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
    plt.title("Patience Sort demo benchmark (few_unique input)")
    plt.grid(True, which="both", linestyle=":", linewidth=0.6)
    plt.legend(fontsize=8)

    out_path = graphs_dir / "demo__patiencesort.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path


if __name__ == "__main__":
    path = _demo()
    print(f"Saved: {path}")
