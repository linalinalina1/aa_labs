from __future__ import annotations

from pathlib import Path

from benchmark import run_benchmarks
from config import BenchmarkConfig
from plots import save_plots
from reporting import save_tables


def main() -> None:
    cfg = BenchmarkConfig()

    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "artifacts" / "results"
    graphs_dir = base_dir / "artifacts" / "graphs"
    tables_dir = base_dir / "artifacts" / "tables"

    agg_csv = run_benchmarks(cfg, out_dir=results_dir)
    save_plots(agg_csv, out_dir=graphs_dir)
    save_tables(agg_csv, out_dir=tables_dir)

    print("Done.")
    print(f"Aggregated results: {agg_csv}")
    print(f"Plots: {graphs_dir}")
    print(f"Tables: {tables_dir}")


if __name__ == "__main__":
    main()