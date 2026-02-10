import time
from statistics import mean
import matplotlib.pyplot as plt

# Choose input set
RUN_MODE = "big"   # "small" or "big"

# Input sets
SMALL_INPUTS = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45]
BIG_INPUTS   = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849]

RUNS = 3                 # 3 runs
PRINT_DECIMALS = 8       # show microseconds, avoids 0.0000
SAVE_GRAPH = True        # saves png

# Fast doubling algorithm
def fib_fast_doubling(n: int) -> int:
    def helper(k: int):
        if k == 0:
            return (0, 1)
        a, b = helper(k // 2)
        c = a * (2 * b - a)
        d = a * a + b * b
        if k % 2 == 0:
            return (c, d)
        else:
            return (d, c + d)
    return helper(n)[0]

# Timing
def time_once(func, n: int) -> float:
    t0 = time.perf_counter()
    func(n)
    t1 = time.perf_counter()
    return t1 - t0

def benchmark_3runs(func, inputs):
    """
    Returns:
      runs_times: list of 3 lists, each list length=len(inputs)
      avg_times: list length=len(inputs)
    """
    runs_times = []
    for run_idx in range(RUNS):
        row = []
        for n in inputs:
            dt = time_once(func, n)
            row.append(dt)
        runs_times.append(row)

    avg_times = [mean([runs_times[r][i] for r in range(RUNS)]) for i in range(len(inputs))]
    return runs_times, avg_times

# Table printing
def fmt_time(x: float) -> str:
    return f"{x:.{PRINT_DECIMALS}f}"

def print_fancy_table(title: str, inputs, runs_times, avg_times):
    # Build columns: first col is row label, then each n value
    col_headers = ["Run"] + [str(n) for n in inputs]

    # Build data rows: "0", "1", "2", "AVG"
    data_rows = []
    for r in range(RUNS):
        data_rows.append([str(r)] + [fmt_time(v) for v in runs_times[r]])
    data_rows.append(["AVG"] + [fmt_time(v) for v in avg_times])

    # Compute column widths
    cols = list(zip(col_headers, *data_rows))  # columns as tuples
    col_widths = [max(len(cell) for cell in col) for col in cols]

    def pad(cell: str, w: int) -> str:
        # Right-align numeric columns, left-align first column
        if cell in ("Run", "0", "1", "2", "AVG"):
            return cell.ljust(w)
        # Numbers
        return cell.rjust(w)

    # Border parts
    def border(left, mid, right, fill="─"):
        pieces = [fill * (w + 2) for w in col_widths]  # +2 for spaces padding
        return left + mid.join(pieces) + right

    top = border("┌", "┬", "┐")
    sep = border("├", "┼", "┤")
    bottom = border("└", "┴", "┘")

    # Print
    print("\n" + title)
    print(top)

    # Header row
    header_cells = [pad(col_headers[i], col_widths[i]) for i in range(len(col_headers))]
    header_line = "│ " + " │ ".join(header_cells) + " │"
    print(header_line)
    print(sep)

    # Data rows
    for i, row in enumerate(data_rows):
        cells = [pad(row[j], col_widths[j]) for j in range(len(row))]
        line = "│ " + " │ ".join(cells) + " │"
        print(line)

        # Add a separator before AVG row for clarity
        if i == RUNS - 1:
            print(sep)

    print(bottom)

# Graph (average)
def plot_average_graph(inputs, avg_times, run_mode: str):
    plt.figure()
    plt.plot(inputs, avg_times, marker="o")
    plt.title("Fast Doubling Fibonacci Function (Average of 3 runs)")
    plt.xlabel("n-th Fibonacci Term")
    plt.ylabel("Time (s)")
    plt.grid(True)

    if SAVE_GRAPH:
        out_name = f"fast_doubling_avg_{run_mode}.png"
        plt.savefig(out_name, dpi=200, bbox_inches="tight")
        print(f"\nSaved graph as: {out_name}")

    plt.show()

# Main
def main():
    if RUN_MODE == "small":
        inputs = SMALL_INPUTS
        title = "Results for Fast Doubling Method (Small Inputs) — 3 runs + AVG"
    elif RUN_MODE == "big":
        inputs = BIG_INPUTS
        title = "Results for Fast Doubling Method (Big Inputs) — 3 runs + AVG"
    else:
        raise ValueError("RUN_MODE must be 'small' or 'big'")

    print(f"Running Fast Doubling benchmark with RUN_MODE='{RUN_MODE}', RUNS={RUNS}")
    runs_times, avg_times = benchmark_3runs(fib_fast_doubling, inputs)

    print_fancy_table(title, inputs, runs_times, avg_times)
    plot_average_graph(inputs, avg_times, RUN_MODE)

if __name__ == "__main__":
    main()
