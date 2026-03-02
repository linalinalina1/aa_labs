# Visualize sorting algorithms (basic + optimized) with state generators.
#
# GREEN means: "this part is truly final" (Quick/Heap) or "final output prefix" (Patience).
# For MergeSort, we DO NOT keep segments green permanently (because later merges rewrite them).
# Instead, we "flash" the just-merged segment green briefly, and only make the whole array green at the end.
#
# Patience Sort upgrade:
#   - Uses a SUBPLOT layout:
#       TOP: piles structure (as vertical pile-height lines + highlighted pile/top)
#       BOTTOM: output array thin bars (prefix turns green as it becomes final)
#   - This makes the animation look much more “real” and algorithm-specific.

from __future__ import annotations

import bisect
import heapq
import random
from typing import Generator, List, Sequence, Tuple, Optional, Dict, Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from inputs import INPUT_TYPES, derive_seed, make_input



# Shared frame type for bar-based sorts

Frame = Tuple[List[int], List[bool]]  # (array_state, is_sorted_mask)



# QuickSort (basic) — Lomuto, pivot becomes final after partition

def quicksort_basic_states(arr: Sequence[int]) -> Generator[Frame, None, None]:
    a = list(arr)
    n = len(a)
    fixed = [False] * n
    yield a[:], fixed[:]

    def partition(lo: int, hi: int) -> Generator[Frame, None, int]:
        pivot = a[hi]
        i = lo
        for j in range(lo, hi):
            if a[j] <= pivot:
                a[i], a[j] = a[j], a[i]
                yield a[:], fixed[:]
                i += 1
        a[i], a[hi] = a[hi], a[i]
        yield a[:], fixed[:]
        return i

    if n < 2:
        for i in range(n):
            fixed[i] = True
        yield a[:], fixed[:]
        return

    stack: List[Tuple[int, int]] = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue

        gen = partition(lo, hi)
        try:
            while True:
                yield next(gen)
        except StopIteration as e:
            p = e.value

        # pivot at p is FINAL
        fixed[p] = True
        yield a[:], fixed[:]

        left = (lo, p - 1)
        right = (p + 1, hi)

        # Push larger first (like your quicksort.py)
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

    for i in range(n):
        fixed[i] = True
    yield a[:], fixed[:]



# QuickSort (optimized) — 3-way partition, [lt..gt] final

def quicksort_optimized_states(arr: Sequence[int]) -> Generator[Frame, None, None]:
    a = list(arr)
    n = len(a)
    fixed = [False] * n
    yield a[:], fixed[:]

    if n < 2:
        for i in range(n):
            fixed[i] = True
        yield a[:], fixed[:]
        return

    rng = random.Random(123)
    stack: List[Tuple[int, int]] = [(0, n - 1)]

    def partition_3way(lo: int, hi: int, pivot: int) -> Generator[Frame, None, Tuple[int, int]]:
        lt = lo
        i = lo
        gt = hi
        while i <= gt:
            if a[i] < pivot:
                a[lt], a[i] = a[i], a[lt]
                yield a[:], fixed[:]
                lt += 1
                i += 1
            elif a[i] > pivot:
                a[gt], a[i] = a[i], a[gt]
                yield a[:], fixed[:]
                gt -= 1
            else:
                i += 1
        return lt, gt

    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue

        pivot_index = rng.randrange(lo, hi + 1)
        pivot = a[pivot_index]

        gen = partition_3way(lo, hi, pivot)
        try:
            while True:
                yield next(gen)
        except StopIteration as e:
            lt, gt = e.value

        # [lt..gt] is FINAL
        for k in range(lt, gt + 1):
            fixed[k] = True
        yield a[:], fixed[:]

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

    for i in range(n):
        fixed[i] = True
    yield a[:], fixed[:]



# HeapSort (basic) — extracted suffix becomes final

def heapsort_basic_states(arr: Sequence[int]) -> Generator[Frame, None, None]:
    a = list(arr)
    n = len(a)
    fixed = [False] * n
    yield a[:], fixed[:]

    def sift_down_recursive(root: int, heap_size: int) -> Generator[Frame, None, None]:
        left = 2 * root + 1
        right = left + 1
        largest = root

        if left < heap_size and a[left] > a[largest]:
            largest = left
        if right < heap_size and a[right] > a[largest]:
            largest = right

        if largest != root:
            a[root], a[largest] = a[largest], a[root]
            yield a[:], fixed[:]
            yield from sift_down_recursive(largest, heap_size)

    for i in range(n // 2 - 1, -1, -1):
        yield from sift_down_recursive(i, n)

    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        yield a[:], fixed[:]
        fixed[end] = True
        yield a[:], fixed[:]
        yield from sift_down_recursive(0, end)

    if n:
        fixed[0] = True
    yield a[:], fixed[:]



# HeapSort (optimized) — iterative sift-down

def heapsort_optimized_states(arr: Sequence[int]) -> Generator[Frame, None, None]:
    a = list(arr)
    n = len(a)
    fixed = [False] * n
    yield a[:], fixed[:]

    def sift_down_iterative(root: int, heap_size: int) -> Generator[Frame, None, None]:
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
            yield a[:], fixed[:]
            root = largest

    for i in range(n // 2 - 1, -1, -1):
        yield from sift_down_iterative(i, n)

    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        yield a[:], fixed[:]
        fixed[end] = True
        yield a[:], fixed[:]
        yield from sift_down_iterative(0, end)

    if n:
        fixed[0] = True
    yield a[:], fixed[:]



# MergeSort (basic) — mask FIX: flash merged segment; only final all-green at end

def mergesort_basic_states(arr: Sequence[int]) -> Generator[Frame, None, None]:
    a = list(arr)
    n = len(a)
    fixed = [False] * n
    yield a[:], fixed[:]

    if n < 2:
        for i in range(n):
            fixed[i] = True
        yield a[:], fixed[:]
        return

    aux = a.copy()

    def flash_sorted(lo: int, hi: int) -> Generator[Frame, None, None]:
        for t in range(lo, hi):
            fixed[t] = True
        yield a[:], fixed[:]
        # Unmark unless it's the full array
        if not (lo == 0 and hi == n):
            for t in range(lo, hi):
                fixed[t] = False

    def merge(lo: int, mid: int, hi: int) -> Generator[Frame, None, None]:
        aux[lo:hi] = a[lo:hi]
        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if aux[i] <= aux[j]:
                a[k] = aux[i]
                i += 1
            else:
                a[k] = aux[j]
                j += 1
            k += 1
            yield a[:], fixed[:]
        while i < mid:
            a[k] = aux[i]
            i += 1
            k += 1
            yield a[:], fixed[:]
        yield from flash_sorted(lo, hi)

    def mergesort(lo: int, hi: int) -> Generator[Frame, None, None]:
        if hi - lo <= 1:
            if hi - lo == 1:
                yield from flash_sorted(lo, hi)
            return
        mid = (lo + hi) // 2
        yield from mergesort(lo, mid)
        yield from mergesort(mid, hi)
        yield from merge(lo, mid, hi)

    yield from mergesort(0, n)
    for i in range(n):
        fixed[i] = True
    yield a[:], fixed[:]



# MergeSort (optimized) — insertion cutoff + skip merge + same flash mask

def mergesort_optimized_states(arr: Sequence[int]) -> Generator[Frame, None, None]:
    a = list(arr)
    n = len(a)
    fixed = [False] * n
    yield a[:], fixed[:]

    if n < 2:
        for i in range(n):
            fixed[i] = True
        yield a[:], fixed[:]
        return

    aux = a.copy()

    def flash_sorted(lo: int, hi: int) -> Generator[Frame, None, None]:
        for t in range(lo, hi):
            fixed[t] = True
        yield a[:], fixed[:]
        if not (lo == 0 and hi == n):
            for t in range(lo, hi):
                fixed[t] = False

    def insertion_sort(lo: int, hi: int) -> Generator[Frame, None, None]:
        for i in range(lo + 1, hi):
            x = a[i]
            j = i - 1
            while j >= lo and a[j] > x:
                a[j + 1] = a[j]
                j -= 1
                yield a[:], fixed[:]
            a[j + 1] = x
            yield a[:], fixed[:]
        yield from flash_sorted(lo, hi)

    def merge(lo: int, mid: int, hi: int) -> Generator[Frame, None, None]:
        aux[lo:hi] = a[lo:hi]
        i, j, k = lo, mid, lo
        while i < mid and j < hi:
            if aux[i] <= aux[j]:
                a[k] = aux[i]
                i += 1
            else:
                a[k] = aux[j]
                j += 1
            k += 1
            yield a[:], fixed[:]
        while i < mid:
            a[k] = aux[i]
            i += 1
            k += 1
            yield a[:], fixed[:]
        yield from flash_sorted(lo, hi)

    def mergesort(lo: int, hi: int) -> Generator[Frame, None, None]:
        if hi - lo <= 1:
            if hi - lo == 1:
                yield from flash_sorted(lo, hi)
            return

        if (hi - lo) <= 32:
            yield from insertion_sort(lo, hi)
            return

        mid = (lo + hi) // 2
        yield from mergesort(lo, mid)
        yield from mergesort(mid, hi)

        if a[mid - 1] <= a[mid]:
            yield from flash_sorted(lo, hi)
            return

        yield from merge(lo, mid, hi)

    yield from mergesort(0, n)
    for i in range(n):
        fixed[i] = True
    yield a[:], fixed[:]



# Patience Sort — event frames for SUBPLOT animation

# We animate:
#   TOP: pile heights (structure) + highlighted pile (placement / extraction)
#   BOTTOM: output array with final prefix (green)
#
# We keep the algorithm logic matching your patience sorts:
#   - basic: linear scan to place, then linear scan to extract min top
#   - optimized: bisect on tops to place, heap to extract min top
#
# Notes:
#   - For very large n (like 10k+), this subplot view stays fast because
#     piles are drawn as pile-height lines (not thousands of points per pile).


PatFrame = Dict[str, Any]  # keys: phase, piles, highlight_pile, output, fixed, ymax


def patiencesort_basic_events(arr: Sequence[int]) -> Generator[PatFrame, None, None]:
    original = list(arr)
    n = len(original)
    fixed = [False] * n
    output = original[:]  # will become sorted output
    ymax = max(original) if original else 1

    piles: List[List[int]] = []

    # start
    yield {
        "phase": "build",
        "piles": [p[:] for p in piles],
        "highlight_pile": None,
        "output": output[:],
        "fixed": fixed[:],
        "ymax": ymax,
    }

    # Phase 1: build piles (basic scan)
    for x in original:
        placed = False
        target = None
        for i, pile in enumerate(piles):
            if pile[-1] >= x:
                pile.append(x)
                placed = True
                target = i
                break
        if not placed:
            piles.append([x])
            target = len(piles) - 1

        yield {
            "phase": "build",
            "piles": [p[:] for p in piles],
            "highlight_pile": target,
            "output": output[:],
            "fixed": fixed[:],
            "ymax": ymax,
        }

    # Phase 2: extract (basic scan for min top)
    active = [p for p in piles if p]
    out_pos = 0
    while active:
        min_i = 0
        min_val = active[0][-1]
        for i in range(1, len(active)):
            v = active[i][-1]
            if v < min_val:
                min_val = v
                min_i = i

        active[min_i].pop()
        if not active[min_i]:
            active.pop(min_i)

        output[out_pos] = min_val
        fixed[out_pos] = True
        out_pos += 1

        # We no longer have stable indices for original pile order after removing empties from active,
        # but for visualization, it's enough to highlight the "chosen pile" among current active piles.
        yield {
            "phase": "merge",
            "piles": [p[:] for p in active],
            "highlight_pile": min_i,
            "output": output[:],
            "fixed": fixed[:],
            "ymax": ymax,
        }

    yield {
        "phase": "done",
        "piles": [],
        "highlight_pile": None,
        "output": output[:],
        "fixed": fixed[:],
        "ymax": ymax,
    }


def patiencesort_optimized_events(arr: Sequence[int]) -> Generator[PatFrame, None, None]:
    original = list(arr)
    n = len(original)
    fixed = [False] * n
    output = original[:]
    ymax = max(original) if original else 1

    piles: List[List[int]] = []
    tops: List[int] = []

    yield {
        "phase": "build",
        "piles": [p[:] for p in piles],
        "highlight_pile": None,
        "output": output[:],
        "fixed": fixed[:],
        "ymax": ymax,
    }

    # Phase 1: build piles (bisect on tops)
    for x in original:
        idx = bisect.bisect_left(tops, x)
        if idx == len(piles):
            piles.append([x])
            tops.append(x)
        else:
            piles[idx].append(x)
            tops[idx] = x

        yield {
            "phase": "build",
            "piles": [p[:] for p in piles],
            "highlight_pile": idx,
            "output": output[:],
            "fixed": fixed[:],
            "ymax": ymax,
        }

    # Phase 2: extract (heap over pile tops)
    heap: List[Tuple[int, int]] = []
    for i, pile in enumerate(piles):
        if pile:
            heapq.heappush(heap, (pile[-1], i))

    out_pos = 0
    while heap:
        value, i = heapq.heappop(heap)
        piles[i].pop()
        if piles[i]:
            heapq.heappush(heap, (piles[i][-1], i))

        output[out_pos] = value
        fixed[out_pos] = True
        out_pos += 1

        yield {
            "phase": "merge",
            "piles": [p[:] for p in piles],
            "highlight_pile": i,
            "output": output[:],
            "fixed": fixed[:],
            "ymax": ymax,
        }

    yield {
        "phase": "done",
        "piles": [p[:] for p in piles],
        "highlight_pile": None,
        "output": output[:],
        "fixed": fixed[:],
        "ymax": ymax,
    }



# Thin-bar rendering helpers

def _make_segments_and_colors(frame: Frame):
    y, fixed = frame
    segments = [((i, 0), (i, yi)) for i, yi in enumerate(y)]
    colors = ["green" if fixed[i] else "steelblue" for i in range(len(y))]
    return segments, colors



# Generic thin-bar animations (for quick/heap/merge)

def animate_stream_thin_bars(gen, title, *, skip=100):
    fig, ax = plt.subplots()
    ax.set_title(title)

    try:
        first = next(gen)
    except StopIteration:
        print("Nothing to animate.")
        return None

    y0, _ = first
    n = len(y0)
    ymax = max(y0) if y0 else 1

    ax.set_xlim(-1, n)
    ax.set_ylim(0, ymax * 1.05)

    segs, cols = _make_segments_and_colors(first)
    lc = LineCollection(segs, linewidths=0.35)
    lc.set_color(cols)
    ax.add_collection(lc)

    def update(_):
        frame = None
        try:
            for _ in range(skip):
                frame = next(gen)

            if frame is not None:
                segs2, cols2 = _make_segments_and_colors(frame)
                lc.set_segments(segs2)
                lc.set_color(cols2)

        except StopIteration:
            if frame is not None:
                segs2, cols2 = _make_segments_and_colors(frame)
                lc.set_segments(segs2)
                lc.set_color(cols2)

            print("\nDone! Sorting completed.")
            plt.pause(0.3)

        return (lc,)

    ani = FuncAnimation(fig, update, interval=1, repeat=False)
    plt.show()
    return ani


def animate_compare_thin_bars(gen1, title1, gen2, title2, *, skip=100):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.set_title(title1)
    ax2.set_title(title2)

    try:
        first1 = next(gen1)
        first2 = next(gen2)
    except StopIteration:
        print("Nothing to animate.")
        return None

    y1, _ = first1
    y2, _ = first2
    n = len(y1)
    ymax = max(max(y1, default=1), max(y2, default=1))

    for ax in (ax1, ax2):
        ax.set_xlim(-1, n)
        ax.set_ylim(0, ymax * 1.05)

    segs1, cols1 = _make_segments_and_colors(first1)
    segs2, cols2 = _make_segments_and_colors(first2)

    lc1 = LineCollection(segs1, linewidths=0.35)
    lc2 = LineCollection(segs2, linewidths=0.35)
    lc1.set_color(cols1)
    lc2.set_color(cols2)

    ax1.add_collection(lc1)
    ax2.add_collection(lc2)

    done1 = False
    done2 = False

    def update(_):
        nonlocal done1, done2

        if not done1:
            f1 = None
            try:
                for _ in range(skip):
                    f1 = next(gen1)
                if f1 is not None:
                    segs, cols = _make_segments_and_colors(f1)
                    lc1.set_segments(segs)
                    lc1.set_color(cols)
            except StopIteration:
                if f1 is not None:
                    segs, cols = _make_segments_and_colors(f1)
                    lc1.set_segments(segs)
                    lc1.set_color(cols)
                done1 = True

        if not done2:
            f2 = None
            try:
                for _ in range(skip):
                    f2 = next(gen2)
                if f2 is not None:
                    segs, cols = _make_segments_and_colors(f2)
                    lc2.set_segments(segs)
                    lc2.set_color(cols)
            except StopIteration:
                if f2 is not None:
                    segs, cols = _make_segments_and_colors(f2)
                    lc2.set_segments(segs)
                    lc2.set_color(cols)
                done2 = True

        if done1 and done2:
            print("\nDone! Both sorts completed.")
            plt.pause(0.3)

        return (lc1, lc2)

    ani = FuncAnimation(fig, update, interval=1, repeat=False)
    plt.tight_layout()
    plt.show()
    return ani



# Patience subplot animation (single)

def _piles_segments(piles: List[List[int]]):
    # draw each pile as a vertical line from 0 to pile_height
    segs = []
    for i, pile in enumerate(piles):
        h = len(pile)
        segs.append(((i, 0), (i, h)))
    return segs


def animate_patience_subplot(
    event_gen: Generator[PatFrame, None, None],
    title: str,
    *,
    skip: int = 200,
):
    fig, (ax_piles, ax_out) = plt.subplots(
        2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 1.3]}
    )
    fig.suptitle(title)

    try:
        first = next(event_gen)
    except StopIteration:
        print("Nothing to animate.")
        return None

    ymax = first.get("ymax", 1)

    # --- PILES axis setup ---
    ax_piles.set_title("Patience Sort: Piles")
    ax_piles.set_xlabel("Pile index")
    ax_piles.set_ylabel("Pile height (#cards)")
    ax_piles.set_ylim(0, 1)  # will autoscale
    ax_piles.set_xlim(-1, 1)  # will autoscale

    piles0 = first["piles"]
    segs0 = _piles_segments(piles0)
    lc_piles = LineCollection(segs0, linewidths=2.0)
    ax_piles.add_collection(lc_piles)

    # --- OUTPUT axis setup ---
    ax_out.set_title("Output (final prefix turns GREEN)")
    out0 = first["output"]
    fixed0 = first["fixed"]
    n = len(out0)
    ax_out.set_xlim(-1, n)
    ax_out.set_ylim(0, ymax * 1.05)

    segs_out0, cols_out0 = _make_segments_and_colors((out0, fixed0))
    lc_out = LineCollection(segs_out0, linewidths=0.35)
    lc_out.set_color(cols_out0)
    ax_out.add_collection(lc_out)

    def _update_axes_limits(piles: List[List[int]]):
        m = len(piles)
        max_h = max((len(p) for p in piles), default=1)
        ax_piles.set_xlim(-1, max(1, m))
        ax_piles.set_ylim(0, max_h * 1.15)

    def _piles_colors(piles: List[List[int]], highlight: Optional[int]):
        m = len(piles)
        colors = ["steelblue"] * m
        if highlight is not None and 0 <= highlight < m:
            colors[highlight] = "orange"
        return colors

    def update(_):
        frame = None
        try:
            for _ in range(skip):
                frame = next(event_gen)
        except StopIteration:
            pass

        if frame is None:
            return (lc_piles, lc_out)

        piles = frame["piles"]
        highlight = frame["highlight_pile"]
        output = frame["output"]
        fixed = frame["fixed"]

        # update piles view
        segs = _piles_segments(piles)
        lc_piles.set_segments(segs)
        lc_piles.set_color(_piles_colors(piles, highlight))
        _update_axes_limits(piles)

        # update output bars
        segs2, cols2 = _make_segments_and_colors((output, fixed))
        lc_out.set_segments(segs2)
        lc_out.set_color(cols2)

        if frame["phase"] == "done":
            print("\nDone! Patience sort completed.")
            plt.pause(0.2)

        return (lc_piles, lc_out)

    ani = FuncAnimation(fig, update, interval=1, repeat=False)
    plt.tight_layout()
    plt.show()
    return ani



# Patience subplot animation (compare patience vs patience)

def animate_compare_patience_subplot(
    gen1: Generator[PatFrame, None, None],
    title1: str,
    gen2: Generator[PatFrame, None, None],
    title2: str,
    *,
    skip: int = 200,
):
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])
    ax1_p = fig.add_subplot(gs[0, 0])
    ax1_o = fig.add_subplot(gs[1, 0])
    ax2_p = fig.add_subplot(gs[0, 1])
    ax2_o = fig.add_subplot(gs[1, 1])

    try:
        f1 = next(gen1)
        f2 = next(gen2)
    except StopIteration:
        print("Nothing to animate.")
        return None

    ymax = max(f1.get("ymax", 1), f2.get("ymax", 1))

    # Setup left
    ax1_p.set_title(title1 + " — Piles")
    ax1_p.set_xlabel("Pile index")
    ax1_p.set_ylabel("Pile height")
    lc1p = LineCollection(_piles_segments(f1["piles"]), linewidths=2.0)
    ax1_p.add_collection(lc1p)

    out1 = f1["output"]
    fixed1 = f1["fixed"]
    n = len(out1)
    ax1_o.set_title("Output")
    ax1_o.set_xlim(-1, n)
    ax1_o.set_ylim(0, ymax * 1.05)
    lc1o = LineCollection(_make_segments_and_colors((out1, fixed1))[0], linewidths=0.35)
    lc1o.set_color(_make_segments_and_colors((out1, fixed1))[1])
    ax1_o.add_collection(lc1o)

    # Setup right
    ax2_p.set_title(title2 + " — Piles")
    ax2_p.set_xlabel("Pile index")
    ax2_p.set_ylabel("Pile height")
    lc2p = LineCollection(_piles_segments(f2["piles"]), linewidths=2.0)
    ax2_p.add_collection(lc2p)

    out2 = f2["output"]
    fixed2 = f2["fixed"]
    ax2_o.set_title("Output")
    ax2_o.set_xlim(-1, n)
    ax2_o.set_ylim(0, ymax * 1.05)
    lc2o = LineCollection(_make_segments_and_colors((out2, fixed2))[0], linewidths=0.35)
    lc2o.set_color(_make_segments_and_colors((out2, fixed2))[1])
    ax2_o.add_collection(lc2o)

    def _update_piles_axis(ax, piles: List[List[int]]):
        m = len(piles)
        max_h = max((len(p) for p in piles), default=1)
        ax.set_xlim(-1, max(1, m))
        ax.set_ylim(0, max_h * 1.15)

    def _piles_colors(piles: List[List[int]], highlight: Optional[int]):
        m = len(piles)
        colors = ["steelblue"] * m
        if highlight is not None and 0 <= highlight < m:
            colors[highlight] = "orange"
        return colors

    done1 = False
    done2 = False

    def update(_):
        nonlocal done1, done2

        if not done1:
            frame1 = None
            try:
                for _ in range(skip):
                    frame1 = next(gen1)
            except StopIteration:
                done1 = True

            if frame1 is not None:
                p = frame1["piles"]
                h = frame1["highlight_pile"]
                lc1p.set_segments(_piles_segments(p))
                lc1p.set_color(_piles_colors(p, h))
                _update_piles_axis(ax1_p, p)

                out = frame1["output"]
                fx = frame1["fixed"]
                segs, cols = _make_segments_and_colors((out, fx))
                lc1o.set_segments(segs)
                lc1o.set_color(cols)

                if frame1["phase"] == "done":
                    done1 = True

        if not done2:
            frame2 = None
            try:
                for _ in range(skip):
                    frame2 = next(gen2)
            except StopIteration:
                done2 = True

            if frame2 is not None:
                p = frame2["piles"]
                h = frame2["highlight_pile"]
                lc2p.set_segments(_piles_segments(p))
                lc2p.set_color(_piles_colors(p, h))
                _update_piles_axis(ax2_p, p)

                out = frame2["output"]
                fx = frame2["fixed"]
                segs, cols = _make_segments_and_colors((out, fx))
                lc2o.set_segments(segs)
                lc2o.set_color(cols)

                if frame2["phase"] == "done":
                    done2 = True

        if done1 and done2:
            print("\nDone! Both patience sorts completed.")
            plt.pause(0.2)

        return (lc1p, lc1o, lc2p, lc2o)

    ani = FuncAnimation(fig, update, interval=1, repeat=False)
    plt.tight_layout()
    plt.show()
    return ani



# Menus

def choose_input_type():
    print("\nInput types:")
    for i, t in enumerate(INPUT_TYPES, 1):
        print(f"{i}. {t}")
    return INPUT_TYPES[int(input("Choose input type (number): ").strip()) - 1]


def choose_algo(prompt):
    algos = [
        ("QuickSort (basic)", "bars", quicksort_basic_states),
        ("QuickSort (optimized)", "bars", quicksort_optimized_states),
        ("MergeSort (basic)", "bars", mergesort_basic_states),
        ("MergeSort (optimized)", "bars", mergesort_optimized_states),
        ("HeapSort (basic)", "bars", heapsort_basic_states),
        ("HeapSort (optimized)", "bars", heapsort_optimized_states),
        ("Patience Sort (basic)", "patience", patiencesort_basic_events),
        ("Patience Sort (optimized)", "patience", patiencesort_optimized_events),
    ]
    print(f"\n{prompt}")
    for i, (name, _, _) in enumerate(algos, 1):
        print(f"{i}. {name}")
    return algos[int(input("Choose algorithm (number): ").strip()) - 1]



# MAIN

if __name__ == "__main__":
    print("=== Sorting Visualization ===")

    print("\nMode:")
    print("1. Visualize ONE algorithm")
    print("2. Compare TWO algorithms")
    mode = int(input("Choose mode (1/2): ").strip())

    n = int(input("\nEnter size n (e.g. 10000): ").strip())
    input_type = choose_input_type()

    seed = derive_seed(2026, kind=input_type, n=n)
    data = make_input(kind=input_type, n=n, seed=seed)

    # For bars, skip based on n; for patience, a bit higher skip usually feels smoother.
    skip_bars = max(50, n // 200)
    skip_pat = max(100, n // 150)

    if mode == 1:
        name, kind, fn = choose_algo("Choose algorithm")
        if kind == "patience":
            animate_patience_subplot(fn(data), f"{name} | n={n} | {input_type}", skip=skip_pat)
        else:
            animate_stream_thin_bars(fn(data), f"{name} | n={n} | {input_type}", skip=skip_bars)

    else:
        name1, kind1, fn1 = choose_algo("Choose algorithm A")
        name2, kind2, fn2 = choose_algo("Choose algorithm B")

        title1 = f"{name1} | n={n} | {input_type}"
        title2 = f"{name2} | n={n} | {input_type}"

        # If both are patience, do the nice 2x2 subplot compare.
        if kind1 == "patience" and kind2 == "patience":
            animate_compare_patience_subplot(fn1(data), title1, fn2(data), title2, skip=skip_pat)
        else:
            # Mixed compare: use thin-bar compare (patience won't show piles here).
            # If you want, we can upgrade this later to show piles vs bars in one figure.
            if kind1 == "patience":
                # fallback: compare using output-only (still correct) by wrapping patience events into bar frames
                def patience_to_barframes(events):
                    for ev in events:
                        yield ev["output"], ev["fixed"]

                gen1 = patience_to_barframes(fn1(data))
            else:
                gen1 = fn1(data)

            if kind2 == "patience":
                def patience_to_barframes(events):
                    for ev in events:
                        yield ev["output"], ev["fixed"]

                gen2 = patience_to_barframes(fn2(data))
            else:
                gen2 = fn2(data)

            animate_compare_thin_bars(gen1, title1, gen2, title2, skip=skip_bars)