from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


# Benchmarks are only meaningful when inputs are reproducible. This module
# provides deterministic generators for several common "input distributions".
# A fixed base seed is used, and a derived per-(kind, n) seed is computed so the
# same configuration always produces the same list.


INPUT_TYPES = [
    "random",
    "sorted",
    "reversed",
    "few_unique",
    "nearly_sorted",
    "almost_reversed",
    "all_equal",
]


@dataclass(frozen=True)
class InputSpec:
    kind: str
    n: int
    seed: int


def derive_seed(base_seed: int, *, kind: str, n: int) -> int:
    """Derive a deterministic per-(kind, n) seed from a base seed.

    Implementation detail:
    - We avoid Python's built-in hash() because it is randomized between runs.
    """

    kind_index = INPUT_TYPES.index(kind) if kind in INPUT_TYPES else 999
    return int(base_seed + 100_000 * kind_index + n)


def make_input(*, kind: str, n: int, seed: int) -> List[int]:
    """Generate a list of integers with the requested property.

    Generators (deterministic given seed):
    - random: uniformly random integers in a fixed range
    - sorted: already sorted ascending sequence
    - reversed: descending sequence
    - few_unique: many duplicates (small value range)
    - nearly_sorted: sorted with ~1% random swaps
    - almost_reversed: reversed with ~1% random swaps
    - all_equal: all elements identical
    """

    rng = random.Random(seed)

    if n < 0:
        raise ValueError("n must be non-negative")

    if kind == "random":
        return [rng.randint(0, 10**6) for _ in range(n)]

    if kind == "sorted":
        return list(range(n))

    if kind == "reversed":
        return list(range(n, 0, -1))

    if kind == "few_unique":
        return [rng.randint(0, 20) for _ in range(n)]

    if kind == "nearly_sorted":
        a = list(range(n))
        swaps = max(1, n // 100)  # ~1% swaps
        for _ in range(swaps):
            i = rng.randrange(n) if n else 0
            j = rng.randrange(n) if n else 0
            if n:
                a[i], a[j] = a[j], a[i]
        return a

    if kind == "almost_reversed":
        a = list(range(n, 0, -1))
        swaps = max(1, n // 100)
        for _ in range(swaps):
            i = rng.randrange(n) if n else 0
            j = rng.randrange(n) if n else 0
            if n:
                a[i], a[j] = a[j], a[i]
        return a

    if kind == "all_equal":
        value = rng.randint(0, 10**6)
        return [value for _ in range(n)]

    raise ValueError(f"Unknown input type: {kind}")


def make_input_from_spec(spec: InputSpec) -> List[int]:
    return make_input(kind=spec.kind, n=spec.n, seed=spec.seed)
