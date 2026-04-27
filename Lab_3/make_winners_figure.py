from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "results.csv"
DEFAULT_OUT = ROOT / "graphs" / "winners.png"

ALGO_KEYS = [
    ("BFS Basic", "bfs_basic_time"),
    ("DFS Basic", "dfs_basic_time"),
    ("BFS Optimized", "bfs_optimized_time"),
    ("DFS Optimized", "dfs_optimized_time"),
]


def compute_winners(results_csv: Path) -> Counter[str]:
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    winners: Counter[str] = Counter()
    for r in rows:
        timings = {name: float(r[key]) for name, key in ALGO_KEYS}
        winner = min(timings.items(), key=lambda kv: kv[1])[0]
        winners[winner] += 1

    # Ensure stable key presence and order.
    for name, _key in ALGO_KEYS:
        winners.setdefault(name, 0)

    return winners


def plot_winners(winners: Counter[str], out_path: Path) -> None:
    labels = [name for name, _key in ALGO_KEYS]
    values = [int(winners.get(name, 0)) for name in labels]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(labels, values)
    plt.title("Winner counts across all benchmark cases")
    plt.ylabel("Wins (lowest time)")
    plt.xticks(rotation=20, ha="right")

    for bar, val in zip(bars, values, strict=False):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(val),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    winners = compute_winners(RESULTS_CSV)
    plot_winners(winners, DEFAULT_OUT)
    print(f"Wrote: {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
