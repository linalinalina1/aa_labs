from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from io_utils import read_csv_dicts, to_float, to_int, write_csv_dicts


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
                "seconds_mean": to_float(r["seconds_mean"]) if "seconds_mean" in r else to_float(r["seconds_median"]),
                "samples": to_int(r["samples"]) if "samples" in r else 0,
            }
        )

    return out


def _latex_table(headers: list[str], body_rows: list[list[str]]) -> str:
    cols = "l" + "r" * (len(headers) - 1)
    out: list[str] = []
    out.append("\\begin{tabular}{%s}" % cols)
    out.append("\\hline")
    out.append(" & ".join(headers) + " \\\\")
    out.append("\\hline")
    for row in body_rows:
        out.append(" & ".join(row) + " \\\\")
    out.append("\\hline")
    out.append("\\end{tabular}")
    return "\n".join(out) + "\n"


def _save_table_image(
    headers: list[str],
    body_rows: list[list[str]],
    title: str,
    out_path: Path,
) -> None:
    fig_height = max(2.5, 0.55 * (len(body_rows) + 3))
    fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(headers)), fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=body_rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.35)

    ax.set_title(title, pad=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _format_console_table(headers: list[str], body_rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]

    for row in body_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    lines = [sep, fmt_row(headers), sep]
    for row in body_rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


def save_tables(agg_csv: Path, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_rows = _load_agg(agg_csv)
    densities = sorted({str(r["density"]) for r in agg_rows})
    algorithms = sorted({str(r["algorithm"]) for r in agg_rows})

    # Preferred order: baseline then optimized
    preferred_variant_order = ["matrix", "heap", "naive", "dsu"]

    # Individual tables per algorithm and density
    for density in densities:
        for algorithm in algorithms:
            sub = [r for r in agg_rows if r["density"] == density and r["algorithm"] == algorithm]
            if not sub:
                continue

            variants = [v for v in preferred_variant_order if any(str(r["variant"]) == v for r in sub)]
            ns = sorted({int(r["n"]) for r in sub})

            by_n_variant: dict[tuple[int, str], float] = {}
            for r in sub:
                by_n_variant[(int(r["n"]), str(r["variant"]))] = float(r["seconds_median"])

            wide_rows: list[dict[str, Any]] = []
            for n in ns:
                row: dict[str, Any] = {"n": n}
                for v in variants:
                    row[v] = by_n_variant.get((n, v), float("nan"))
                wide_rows.append(row)

            csv_path = out_dir / f"times__{algorithm}__{density}.csv"
            write_csv_dicts(csv_path, wide_rows, fieldnames=["n", *variants])

            headers = ["n", *variants]
            body_rows: list[list[str]] = []
            for n in ns:
                row = [str(n)]
                for v in variants:
                    value = by_n_variant.get((n, v), float("nan"))
                    row.append(f"{value:.6f}")
                body_rows.append(row)

            tex_path = out_dir / f"times__{algorithm}__{density}.tex"
            tex_path.write_text(_latex_table(headers, body_rows), encoding="utf-8")

            png_path = out_dir / f"times__{algorithm}__{density}.png"
            title = f"Execution times for {algorithm} on {density} graph (seconds)"
            _save_table_image(headers, body_rows, title, png_path)

            print()
            print(title)
            print(_format_console_table(headers, body_rows))
            print(f"Saved CSV: {csv_path}")
            print(f"Saved TEX: {tex_path}")
            print(f"Saved PNG: {png_path}")

    # Combined tables for all algorithms together, per density
    for density in densities:
        sub = [r for r in agg_rows if r["density"] == density]
        if not sub:
            continue

        preferred_algo_variant_order = [
            "prim:matrix",
            "prim:heap",
            "kruskal:naive",
            "kruskal:dsu",
        ]

        algo_variants = [
            av
            for av in preferred_algo_variant_order
            if any(f"{r['algorithm']}:{r['variant']}" == av for r in sub)
        ]

        ns = sorted({int(r["n"]) for r in sub})

        by_n_algo_variant: dict[tuple[int, str], float] = {}
        for r in sub:
            key = (int(r["n"]), f"{r['algorithm']}:{r['variant']}")
            by_n_algo_variant[key] = float(r["seconds_median"])

        wide_rows: list[dict[str, Any]] = []
        for n in ns:
            row: dict[str, Any] = {"n": n}
            for av in algo_variants:
                row[av] = by_n_algo_variant.get((n, av), float("nan"))
            wide_rows.append(row)

        csv_path = out_dir / f"times__all__{density}.csv"
        write_csv_dicts(csv_path, wide_rows, fieldnames=["n", *algo_variants])

        headers = ["n", *algo_variants]
        body_rows: list[list[str]] = []
        for n in ns:
            row = [str(n)]
            for av in algo_variants:
                value = by_n_algo_variant.get((n, av), float("nan"))
                row.append(f"{value:.6f}")
            body_rows.append(row)

        tex_path = out_dir / f"times__all__{density}.tex"
        tex_path.write_text(_latex_table(headers, body_rows), encoding="utf-8")

        png_path = out_dir / f"times__all__{density}.png"
        title = f"Execution times for all algorithms on {density} graph (seconds)"
        _save_table_image(headers, body_rows, title, png_path)

        print()
        print(title)
        print(_format_console_table(headers, body_rows))
        print(f"Saved CSV: {csv_path}")
        print(f"Saved TEX: {tex_path}")
        print(f"Saved PNG: {png_path}")
