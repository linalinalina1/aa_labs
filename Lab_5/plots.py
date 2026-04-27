from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from io_utils import read_csv_dicts, to_float, to_int


def _load_agg(agg_csv: Path) -> list[dict[str, Any]]:
    rows = read_csv_dicts(agg_csv)
    out: list[dict[str, Any]] = []

    for r in rows:
        out.append(
            {
                "algorithm": r["algorithm"],
                "variant": r["variant"],
                "density": r["density"],
                "n": to_int(r["n"]),
                "seconds_median": to_float(r["seconds_median"]),
            }
        )

    return out


def _plot_groups(
    *,
    groups: dict[tuple[str, str], list[tuple[int, float]]],
    out_path: Path,
    title: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for (algorithm, variant), pts in sorted(groups.items()):
        pts_sorted = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        plt.plot(xs, ys, marker="o", label=f"{algorithm}:{variant}")

    plt.xlabel("Number of nodes (n)")
    plt.ylabel("Median execution time (s)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_plots(agg_csv: Path, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_agg(agg_csv)
    densities = sorted({str(r["density"]) for r in rows})

    # Individual plots: one algorithm per graph type
    for density in densities:
        sub = [r for r in rows if r["density"] == density]

        for alg in sorted({str(r["algorithm"]) for r in sub}):
            groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
            for r in sub:
                if r["algorithm"] != alg:
                    continue
                groups[(str(r["algorithm"]), str(r["variant"]))].append(
                    (int(r["n"]), float(r["seconds_median"]))
                )

            if groups:
                _plot_groups(
                    groups=groups,
                    out_path=out_dir / f"{alg}__{density}.png",
                    title=f"{alg.title()} runtime vs n ({density})",
                )

    # Combined plots: all algorithms together for each graph type
    for density in densities:
        sub = [r for r in rows if r["density"] == density]
        groups: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)

        for r in sub:
            groups[(str(r["algorithm"]), str(r["variant"]))].append(
                (int(r["n"]), float(r["seconds_median"]))
            )

        if groups:
            _plot_groups(
                groups=groups,
                out_path=out_dir / f"all__{density}.png",
                title=f"All algorithms runtime vs n ({density})",
            )
