import time
import math
from statistics import mean
import matplotlib.pyplot as plt

# Choose input set
RUN_MODE = "small"   # "small" or "big"

# Input sets (from report)
SMALL_INPUTS = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45]
BIG_INPUTS   = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849]

RUNS = 3
PRINT_DECIMALS = 8
SAVE_GRAPH = True

# Binet float method
def fib_binet_float(n: int) -> int:
    if n <= 1:
        return n
    phi = (1 + math.sqrt(5)) / 2
    return int(round((phi ** n) / math.sqrt(5)))

# Timing
def time_once(func, n: int) -> float:
    t0 = time.perf_counter()
    func(n)
    t1 = time.perf_counter()
    return t1 - t0

def benchmark_3runs(func, inputs):
    runs_times = []
    for run_idx in range(RUNS):
        row = []
        for n in inputs:
            print(f"{RUN_MODE.upper()} | Binet_Float | run {run_idx+1}/{RUNS} | computing for n = {n} ...")
            row.append(time_once(func, n))
        runs_times.append(row)

    avg_times = [
        mean(runs_times[r][i] for r in range(RUNS))
        for i in range(len(inputs))
    ]
    return runs_times, avg_times

# Fancy table
def fmt(x):
    return f"{x:.{PRINT_DECIMALS}f}"

def print_fancy_table(title, inputs, runs_times, avg_times):
    headers = ["Run"] + [str(n) for n in inputs]
    rows = []

    for r in range(RUNS):
        rows.append([str(r)] + [fmt(v) for v in runs_times[r]])
    rows.append(["AVG"] + [fmt(v) for v in avg_times])

    col_widths = [
        max(len(row[i]) for row in [headers] + rows)
        for i in range(len(headers))
    ]

    def line(left, mid, right):
        return left + mid.join("─" * (w + 2) for w in col_widths) + right

    print("\n" + title)
    print(line("┌", "┬", "┐"))
    print("│ " + " │ ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))) + " │")
    print(line("├", "┼", "┤"))

    for i, row in enumerate(rows):
        print("│ " + " │ ".join(row[j].rjust(col_widths[j]) for j in range(len(row))) + " │")
        if i == RUNS - 1:
            print(line("├", "┼", "┤"))

    print(line("└", "┴", "┘"))

# Graph (average)
def plot_average_graph(inputs, avg_times):
    plt.figure()
    plt.plot(inputs, avg_times, marker="o")
    plt.title("Binet Float Fibonacci Function (Average of 3 runs)")
    plt.xlabel("n-th Fibonacci Term")
    plt.ylabel("Time (s)")
    plt.grid(True)

    if SAVE_GRAPH:
        filename = f"binet_float_avg_{RUN_MODE}.png"
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        print(f"\nSaved graph as: {filename}")

    plt.show()

# Main
def main():
    if RUN_MODE == "small":
        inputs = SMALL_INPUTS
        title = "Results for Binet Float Method (Small Inputs) — 3 runs + AVG"
    elif RUN_MODE == "big":
        inputs = BIG_INPUTS
        title = "Results for Binet Float Method (Big Inputs) — 3 runs + AVG"
    else:
        raise ValueError("RUN_MODE must be 'small' or 'big'")

    print(f"Running Binet Float benchmark with RUN_MODE='{RUN_MODE}', RUNS={RUNS}")
    runs_times, avg_times = benchmark_3runs(fib_binet_float, inputs)

    print_fancy_table(title, inputs, runs_times, avg_times)
    plot_average_graph(inputs, avg_times)

if __name__ == "__main__":
    main()
