import time
from statistics import mean
from decimal import Decimal, getcontext, ROUND_HALF_UP
import matplotlib.pyplot as plt

# Choose input set
RUN_MODE = "big"   # "small" or "big"

# Input sets
SMALL_INPUTS = [5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45]
BIG_INPUTS   = [501, 631, 794, 1000, 1259, 1585, 1995, 2512, 3162, 3981, 5012, 6310, 7943, 10000, 12589, 15849]

RUNS = 3
PRINT_DECIMALS = 8

# Decimal Binet method
def fibonacci_decimal_binet(n: int, precision_digits: int) -> int:
    if n <= 1:
        return n

    getcontext().prec = precision_digits

    sqrt5 = Decimal(5).sqrt()
    phi = (Decimal(1) + sqrt5) / Decimal(2)

    value = (phi ** n) / sqrt5
    rounded = (value + Decimal("0.5")).to_integral_value(rounding=ROUND_HALF_UP)
    return int(rounded)

def required_precision(n: int) -> int:
    # Safe precision rule for Decimal-Binet
    return max(50, n + 20)

# Timing
def time_once(n: int) -> float:
    prec = required_precision(n)
    t0 = time.perf_counter()
    fibonacci_decimal_binet(n, prec)
    t1 = time.perf_counter()
    return t1 - t0

def benchmark_3_runs(inputs):
    runs_times = []
    for r in range(RUNS):
        row = []
        for n in inputs:
            row.append(time_once(n))
        runs_times.append(row)

    avg_times = [
        mean(runs_times[r][i] for r in range(RUNS))
        for i in range(len(inputs))
    ]
    return runs_times, avg_times

# Table with results printing
def fmt(x):
    return f"{x:.{PRINT_DECIMALS}f}"

def print_table(title, inputs, runs_times, avg_times):
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

# Graph (average only)
def plot_graph(inputs, avg_times):
    plt.figure()
    plt.plot(inputs, avg_times, marker="o")
    plt.title("Decimal Binet Fibonacci Function (Average of 3 runs)")
    plt.xlabel("n-th Fibonacci Term")
    plt.ylabel("Time (s)")
    plt.grid(True)

    filename = f"decimal_binet_avg_{RUN_MODE}.png"
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"\nSaved graph as: {filename}")

# Main
def main():
    if RUN_MODE == "small":
        inputs = SMALL_INPUTS
        title = "Results for Decimal Binet Method (Small Inputs)"
    elif RUN_MODE == "big":
        inputs = BIG_INPUTS
        title = "Results for Decimal Binet Method (Big Inputs)"
    else:
        raise ValueError("RUN_MODE must be 'small' or 'big'")

    print(f"Running Decimal-Binet with RUN_MODE='{RUN_MODE}', RUNS={RUNS}")
    runs_times, avg_times = benchmark_3_runs(inputs)

    print_table(title, inputs, runs_times, avg_times)
    plot_graph(inputs, avg_times)

if __name__ == "__main__":
    main()
