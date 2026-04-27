from collections import deque
import random
import time
import math
import csv
import os
import matplotlib.pyplot as plt


ALGO_STYLES = {
    "BFS Basic": {"color": "tab:blue", "marker": "o"},
    "DFS Basic": {"color": "tab:orange", "marker": "s"},
    "BFS Optimized": {"color": "tab:green", "marker": "^"},
    "DFS Optimized": {"color": "tab:red", "marker": "x"},
}


# Basic BFS
# Visits nodes level by level using a queue
# Stores the traversal order
def bfs_basic(graph, start):
    visited = set()
    order = []
    queue = deque([start])

    visited.add(start)

    while queue:
        node = queue.popleft()
        order.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return order


# Basic DFS
# Explores nodes deeply using a stack
# Stores the traversal order
def dfs_basic(graph, start):
    visited = set()
    order = []
    stack = [start]

    while stack:
        node = stack.pop()

        if node not in visited:
            visited.add(node)
            order.append(node)

            for neighbor in reversed(graph[node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return order


# Optimized BFS
# Counts only visited nodes instead of storing traversal order
def bfs_optimized(graph, start):
    # Optimized for graphs with nodes 0..n-1 (which all our generators produce):
    # - Use a compact bytearray for visited checks
    # - Mark visited on enqueue (like BFS basic) to avoid duplicates
    n = len(graph)
    visited = bytearray(n)
    visited[start] = 1

    queue = deque([start])
    graph_local = graph
    visited_local = visited
    queue_append = queue.append
    queue_popleft = queue.popleft
    count = 0

    while queue:
        node = queue_popleft()
        count += 1

        for neighbor in graph_local[node]:
            if not visited_local[neighbor]:
                visited_local[neighbor] = 1
                queue_append(neighbor)

    return count


# Optimized DFS
# Counts only visited nodes instead of storing traversal order
def dfs_optimized(graph, start):
    # Optimized for graphs with nodes 0..n-1 (which all our generators produce):
    # - Use a compact bytearray for visited checks
    # - Mark visited when pushing onto the stack to avoid duplicate pushes
    n = len(graph)
    visited = bytearray(n)
    visited[start] = 1

    stack = [start]
    graph_local = graph
    visited_local = visited
    stack_append = stack.append
    count = 0

    while stack:
        node = stack.pop()
        count += 1

        for neighbor in reversed(graph_local[node]):
            if not visited_local[neighbor]:
                visited_local[neighbor] = 1
                stack_append(neighbor)

    return count


# 1. Sparse graph
# Few edges per node
def generate_sparse_graph(n):
    graph_sets = [set() for _ in range(n)]

    for i in range(n):
        while len(graph_sets[i]) < 2 and n > 1:
            j = random.randint(0, n - 1)
            if j != i:
                graph_sets[i].add(j)
                graph_sets[j].add(i)

    return [list(neighbors) for neighbors in graph_sets]


# 2. Dense graph
# Almost every node is connected to all others
def generate_dense_graph(n):
    return [[j for j in range(n) if j != i] for i in range(n)]


# 3. Tree graph
# Connected graph without cycles
def generate_tree_graph(n):
    graph_sets = [set() for _ in range(n)]

    for i in range(1, n):
        parent = random.randint(0, i - 1)
        graph_sets[i].add(parent)
        graph_sets[parent].add(i)

    return [list(neighbors) for neighbors in graph_sets]


# 4. Chain graph
# Nodes connected in a straight line
def generate_chain_graph(n):
    graph_sets = [set() for _ in range(n)]

    for i in range(n - 1):
        graph_sets[i].add(i + 1)
        graph_sets[i + 1].add(i)

    return [list(neighbors) for neighbors in graph_sets]


# 5. Cycle graph
# Nodes form a closed loop
def generate_cycle_graph(n):
    graph_sets = [set() for _ in range(n)]

    for i in range(n):
        nxt = (i + 1) % n
        graph_sets[i].add(nxt)
        graph_sets[nxt].add(i)

    return [list(neighbors) for neighbors in graph_sets]


# 6. Disconnected graph
# Two separate connected components
def generate_disconnected_graph(n):
    graph_sets = [set() for _ in range(n)]
    half = n // 2

    for i in range(half - 1):
        graph_sets[i].add(i + 1)
        graph_sets[i + 1].add(i)

    for i in range(half, n - 1):
        graph_sets[i].add(i + 1)
        graph_sets[i + 1].add(i)

    return [list(neighbors) for neighbors in graph_sets]


# 7. Star graph
# One central node connected to all other nodes
def generate_star_graph(n):
    graph_sets = [set() for _ in range(n)]

    for i in range(1, n):
        graph_sets[0].add(i)
        graph_sets[i].add(0)

    return [list(neighbors) for neighbors in graph_sets]


# 8. Binary tree graph
# Each node can have up to two children
def generate_binary_tree_graph(n):
    graph_sets = [set() for _ in range(n)]

    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n:
            graph_sets[i].add(left)
            graph_sets[left].add(i)

        if right < n:
            graph_sets[i].add(right)
            graph_sets[right].add(i)

    return [list(neighbors) for neighbors in graph_sets]


# 9. Random graph
# Edges are created randomly with probability p
def generate_random_graph(n, p=0.3):
    graph_sets = [set() for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                graph_sets[i].add(j)
                graph_sets[j].add(i)

    return [list(neighbors) for neighbors in graph_sets]


# 10. Directed graph
# Edges go in one direction only
def generate_directed_graph(n):
    graph_sets = [set() for _ in range(n)]

    for i in range(n):
        while len(graph_sets[i]) < 2 and n > 1:
            j = random.randint(0, n - 1)
            if j != i:
                graph_sets[i].add(j)

    return [list(neighbors) for neighbors in graph_sets]


# 11. Weighted graph
# Each edge has a numerical weight
def generate_weighted_graph(n):
    graph = [[] for _ in range(n)]
    graph_sets = [set() for _ in range(n)]

    for i in range(n):
        while len(graph_sets[i]) < 2 and n > 1:
            j = random.randint(0, n - 1)
            if j != i and j not in graph_sets[i]:
                weight = random.randint(1, 10)
                graph[i].append((j, weight))
                graph[j].append((i, weight))
                graph_sets[i].add(j)
                graph_sets[j].add(i)

    return graph


# Converts a weighted graph into a normal adjacency list
# BFS and DFS do not use the weights
def simplify_weighted_graph(graph):
    simplified = []

    for neighbors in graph:
        row = []
        for neighbor, _weight in neighbors:
            row.append(neighbor)
        simplified.append(row)

    return simplified


# 12. Complete graph
# Every node is connected to every other node
def generate_complete_graph(n):
    return [[j for j in range(n) if j != i] for i in range(n)]


# 13. Bipartite graph
# Nodes are divided into two groups
def generate_bipartite_graph(n):
    graph_sets = [set() for _ in range(n)]
    half = n // 2

    for i in range(half):
        for j in range(half, n):
            graph_sets[i].add(j)
            graph_sets[j].add(i)

    return [list(neighbors) for neighbors in graph_sets]


# 14. Grid graph
# Nodes are arranged like a 2D matrix
def generate_grid_graph(rows, cols):
    n = rows * cols
    graph_sets = [set() for _ in range(n)]

    def idx(r, c):
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            node = idx(r, c)

            if r > 0:
                graph_sets[node].add(idx(r - 1, c))
            if r < rows - 1:
                graph_sets[node].add(idx(r + 1, c))
            if c > 0:
                graph_sets[node].add(idx(r, c - 1))
            if c < cols - 1:
                graph_sets[node].add(idx(r, c + 1))

    return [list(neighbors) for neighbors in graph_sets]


# 15. Ladder graph
# Two parallel chains connected like a ladder
def generate_ladder_graph(n):
    graph_sets = [set() for _ in range(n * 2)]

    for i in range(n):
        if i < n - 1:
            graph_sets[i].add(i + 1)
            graph_sets[i + 1].add(i)

            graph_sets[i + n].add(i + n + 1)
            graph_sets[i + n + 1].add(i + n)

        graph_sets[i].add(i + n)
        graph_sets[i + n].add(i)

    return [list(neighbors) for neighbors in graph_sets]


# Measures execution time for one run
def measure_single_run(function, graph, start_node):
    start_time = time.perf_counter()
    result = function(graph, start_node)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, result


# Measures average execution time over several runs
def measure_average_time(function, graph, start_node, repeats=5):
    total_time = 0.0
    last_result = None

    for _ in range(repeats):
        start_time = time.perf_counter()
        last_result = function(graph, start_node)
        total_time += time.perf_counter() - start_time

    return total_time / repeats, last_result


def density_benchmark_analysis(
    sizes: list[int] | None = None,
    densities: list[tuple[str, float]] | None = None,
    repeats: int = 3,
    start_node: int = 0,
):
    if sizes is None:
        
        sizes = [50, 100, 200, 400, 600, 1000, 2000]

    if densities is None:
        densities = [
            ("Rare (5%)", 0.05),
            ("20%", 0.2),
            ("40%", 0.4),
            ("60%", 0.6),
            ("80%", 0.8),
            ("Dense (95%)", 0.95),
        ]

    results = []

    for density_label, p in densities:
        sizes_for_density = list(sizes)
        # Extend only for low densities (includes Rare (5%) and 20% by default)
        if p <= 0.2:
            sizes_for_density = sorted(set(sizes_for_density + [5000, 10000]))

        for n in sizes_for_density:
            graph = generate_random_graph(n, p=p)

            bfs_basic_t, _ = measure_average_time(bfs_basic, graph, start_node, repeats=repeats)
            dfs_basic_t, _ = measure_average_time(dfs_basic, graph, start_node, repeats=repeats)

            bfs_opt_t, _ = measure_average_time(bfs_optimized, graph, start_node, repeats=repeats)
            dfs_opt_t, _ = measure_average_time(dfs_optimized, graph, start_node, repeats=repeats)

            results.append(
                {
                    "density": density_label,
                    "p": p,
                    "n": n,
                    "bfs_basic": bfs_basic_t,
                    "bfs_optimized": bfs_opt_t,
                    "dfs_basic": dfs_basic_t,
                    "dfs_optimized": dfs_opt_t,
                    "winner": None,
                    "loser": None,
                }
            )

    return results


def _winner_loser(row: dict) -> tuple[str, str]:
    timings: dict[str, float] = {
        "BFS Basic": float(row["bfs_basic"]),
        "DFS Basic": float(row["dfs_basic"]),
        "BFS Optimized": float(row["bfs_optimized"]),
        "DFS Optimized": float(row["dfs_optimized"]),
    }
    winner = min(timings.items(), key=lambda kv: kv[1])[0]
    loser = max(timings.items(), key=lambda kv: kv[1])[0]
    return winner, loser


def print_density_table(results):
    densities = sorted({row["density"] for row in results})

    print("\n=== Density benchmark (random graphs) ===")
    print("Times in seconds (avg).")

    for density in densities:
        sizes = sorted({row["n"] for row in results if row["density"] == density})
        print(f"\nDensity: {density}")
        header = (
            f"{'n':<8}"
            f"{'BFS Basic':<14}"
            f"{'BFS Opt':<14}"
            f"{'DFS Basic':<14}"
            f"{'DFS Opt':<14}"
            f"{'Winner':<16}"
            f"{'Loser':<16}"
        )
        print(header)
        print("-" * len(header))
        for n in sizes:
            row = next(r for r in results if r["density"] == density and r["n"] == n)
            winner, loser = _winner_loser(row)
            row["winner"] = winner
            row["loser"] = loser
            print(
                f"{n:<8}"
                f"{row['bfs_basic']:<14.6f}"
                f"{row['bfs_optimized']:<14.6f}"
                f"{row['dfs_basic']:<14.6f}"
                f"{row['dfs_optimized']:<14.6f}"
                f"{winner:<16}"
                f"{loser:<16}"
            )


def save_density_results_to_csv(results, filename="density_results.csv"):
    # Ensure winner/loser is filled even if user skips console printing
    for row in results:
        if not row.get("winner") or not row.get("loser"):
            winner, loser = _winner_loser(row)
            row["winner"] = winner
            row["loser"] = loser

    fieldnames = [
        "density",
        "p",
        "n",
        "bfs_basic",
        "bfs_optimized",
        "dfs_basic",
        "dfs_optimized",
        "winner",
        "loser",
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def save_density_intersection_plot(results, output_folder="graphs"):
    os.makedirs(output_folder, exist_ok=True)

    densities = []
    for row in results:
        if row["density"] not in densities:
            densities.append(row["density"])

    density_to_rows = {d: [r for r in results if r["density"] == d] for d in densities}

    ncols = 3
    nrows = math.ceil(len(densities) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.8 * nrows), squeeze=False)

    intersections = []

    def _find_crossing(xs, ys_a, ys_b):
        diffs = [a - b for a, b in zip(ys_a, ys_b)]
        for i in range(len(xs) - 1):
            d1 = diffs[i]
            d2 = diffs[i + 1]
            if d1 == 0:
                return float(xs[i])
            if d1 * d2 < 0:
                # linear interpolation for x where diff crosses 0
                x1, x2 = float(xs[i]), float(xs[i + 1])
                return x1 + (0.0 - d1) * (x2 - x1) / (d2 - d1)
        return None

    def _interp_y(xs, ys, x):
        for i in range(len(xs) - 1):
            x1 = float(xs[i])
            x2 = float(xs[i + 1])
            if x1 <= x <= x2:
                y1 = float(ys[i])
                y2 = float(ys[i + 1])
                if x2 == x1:
                    return y1
                return y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        return None

    for idx, density in enumerate(densities):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]

        sizes = sorted({row["n"] for row in density_to_rows[density]})

        rows = {row["n"]: row for row in density_to_rows[density]}
        bfs_basic_vals = [rows[n]["bfs_basic"] for n in sizes]
        bfs_opt_vals = [rows[n]["bfs_optimized"] for n in sizes]
        dfs_basic_vals = [rows[n]["dfs_basic"] for n in sizes]
        dfs_opt_vals = [rows[n]["dfs_optimized"] for n in sizes]

        ax.plot(
            sizes,
            bfs_basic_vals,
            marker=ALGO_STYLES["BFS Basic"]["marker"],
            color=ALGO_STYLES["BFS Basic"]["color"],
            label="BFS Basic",
        )
        ax.plot(
            sizes,
            dfs_basic_vals,
            marker=ALGO_STYLES["DFS Basic"]["marker"],
            color=ALGO_STYLES["DFS Basic"]["color"],
            label="DFS Basic",
        )
        ax.plot(
            sizes,
            bfs_opt_vals,
            marker=ALGO_STYLES["BFS Optimized"]["marker"],
            color=ALGO_STYLES["BFS Optimized"]["color"],
            label="BFS Optimized",
        )
        ax.plot(
            sizes,
            dfs_opt_vals,
            marker=ALGO_STYLES["DFS Optimized"]["marker"],
            color=ALGO_STYLES["DFS Optimized"]["color"],
            label="DFS Optimized",
        )

        series = [
            ("BFS Basic", bfs_basic_vals),
            ("DFS Basic", dfs_basic_vals),
            ("BFS Optimized", bfs_opt_vals),
            ("DFS Optimized", dfs_opt_vals),
        ]

        # Record intersections for ALL pairs (CSV output)
        for i in range(len(series)):
            for j in range(i + 1, len(series)):
                name_a, ys_a = series[i]
                name_b, ys_b = series[j]

                diffs = [a - b for a, b in zip(ys_a, ys_b)]
                for k in range(len(sizes) - 1):
                    d1 = diffs[k]
                    d2 = diffs[k + 1]

                    if d1 == 0:
                        x_cross = float(sizes[k])
                    elif d1 * d2 < 0:
                        x1, x2 = float(sizes[k]), float(sizes[k + 1])
                        x_cross = x1 + (0.0 - d1) * (x2 - x1) / (d2 - d1)
                    else:
                        continue

                    y_a = _interp_y(sizes, ys_a, x_cross)
                    y_b = _interp_y(sizes, ys_b, x_cross)
                    if y_a is None or y_b is None:
                        continue

                    intersections.append(
                        {
                            "density": density,
                            "pair": f"{name_a} vs {name_b}",
                            "intersection_n": x_cross,
                            "intersection_time_s": (float(y_a) + float(y_b)) / 2.0,
                        }
                    )

        # Mark intersections between basic and optimized versions (if any) on the plot
        # (we keep plot markings minimal to avoid clutter)
        bfs_x = _find_crossing(sizes, bfs_basic_vals, bfs_opt_vals)
        if bfs_x is not None:
            bfs_y = _interp_y(sizes, bfs_basic_vals, bfs_x)
            if bfs_y is not None:
                ax.scatter([bfs_x], [bfs_y], color="black", marker="x", zorder=10)
                ax.axvline(bfs_x, color="black", linestyle="--", alpha=0.25)
                ax.text(bfs_x, bfs_y, "  BFS cross", fontsize=9, va="bottom")

        dfs_x = _find_crossing(sizes, dfs_basic_vals, dfs_opt_vals)
        if dfs_x is not None:
            dfs_y = _interp_y(sizes, dfs_basic_vals, dfs_x)
            if dfs_y is not None:
                ax.scatter([dfs_x], [dfs_y], color="black", marker="x", zorder=10)
                ax.axvline(dfs_x, color="black", linestyle="--", alpha=0.25)
                ax.text(dfs_x, dfs_y, "  DFS cross", fontsize=9, va="bottom")

        ax.set_title(f"Random graph density: {density}")
        ax.set_xlabel("Input size (n)")
        ax.set_ylabel("Time (s)")
        ax.grid(True, alpha=0.25)

    # Hide any unused subplots
    for idx in range(len(densities), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(os.path.join(output_folder, "density_intersections.png"), bbox_inches="tight")
    plt.close(fig)

    if intersections:
        out_csv = os.path.join(output_folder, "density_intersections_table.csv")
        fieldnames = ["density", "pair", "intersection_n", "intersection_time_s"]
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(intersections)


def save_density_tables_image(results, output_folder="graphs"):
    os.makedirs(output_folder, exist_ok=True)

    densities = []
    for row in results:
        if row["density"] not in densities:
            densities.append(row["density"])

    # Ensure winner/loser computed
    for row in results:
        if not row.get("winner") or not row.get("loser"):
            winner, loser = _winner_loser(row)
            row["winner"] = winner
            row["loser"] = loser

    density_to_rows = {d: {r["n"]: r for r in results if r["density"] == d} for d in densities}

    ncols = 3
    nrows = math.ceil(len(densities) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5.2 * nrows), squeeze=False)

    col_labels = ["n", "BFS Basic", "BFS Opt", "DFS Basic", "DFS Opt", "Winner", "Loser"]

    for idx, density in enumerate(densities):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r][c]
        ax.axis("off")
        ax.set_title(f"Density: {density}")

        rows_for_density = density_to_rows[density]
        sizes = sorted(rows_for_density.keys())
        cell_text = []
        for n in sizes:
            row = rows_for_density[n]
            cell_text.append(
                [
                    str(n),
                    f"{row['bfs_basic']:.6f}",
                    f"{row['bfs_optimized']:.6f}",
                    f"{row['dfs_basic']:.6f}",
                    f"{row['dfs_optimized']:.6f}",
                    row["winner"],
                    row["loser"],
                ]
            )

        table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.1, 1.25)
        table.auto_set_column_width(col=list(range(len(col_labels))))

    for idx in range(len(densities), nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(output_folder, "density_tables.png"), bbox_inches="tight")
    plt.close(fig)


# Performs empirical analysis for all graph types and sizes
# Uses larger and more varied input sizes
def empirical_analysis():
    repeats = 5

    graph_types = {
        "Sparse": {
            "generator": generate_sparse_graph,
            "sizes": [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        },
        "Dense": {
            "generator": generate_dense_graph,
            "sizes": [10, 50, 100, 200, 300, 500]
        },
        "Tree": {
            "generator": generate_tree_graph,
            "sizes": [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        },
        "Chain": {
            "generator": generate_chain_graph,
            "sizes": [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        },
        "Cycle": {
            "generator": generate_cycle_graph,
            "sizes": [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        },
        "Disconnected": {
            "generator": generate_disconnected_graph,
            "sizes": [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        },
        "Star": {
            "generator": generate_star_graph,
            "sizes": [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        },
        "Binary_Tree": {
            "generator": generate_binary_tree_graph,
            "sizes": [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        },
        "Random": {
            "generator": lambda n: generate_random_graph(n, 0.3),
            "sizes": [10, 50, 100, 200, 500, 1000, 2000, 5000]
        },
        "Directed": {
            "generator": generate_directed_graph,
            "sizes": [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        },
        "Weighted": {
            "generator": generate_weighted_graph,
            "sizes": [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        },
        "Complete": {
            "generator": generate_complete_graph,
            "sizes": [10, 20, 50, 100, 200, 300]
        },
        "Bipartite": {
            "generator": generate_bipartite_graph,
            "sizes": [10, 20, 50, 100, 200, 500, 1000]
        },
        "Grid": {
            "generator": lambda n: generate_grid_graph(max(2, int(math.sqrt(n))), max(2, int(math.sqrt(n)))),
            "sizes": [16, 25, 100, 225, 400, 900, 2500, 10000]
        },
        "Ladder": {
            "generator": lambda n: generate_ladder_graph(max(2, n // 2)),
            "sizes": [10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        }
    }

    results = []

    for name, config in graph_types.items():
        generator = config["generator"]
        sizes = config["sizes"]

        for n in sizes:
            graph = generator(n)

            if name == "Weighted":
                graph = simplify_weighted_graph(graph)

            # Use the same timing method for all 4 algorithms (fair comparison)
            bfs_basic_time, bfs_basic_order = measure_average_time(bfs_basic, graph, 0, repeats)
            dfs_basic_time, dfs_basic_order = measure_average_time(dfs_basic, graph, 0, repeats)
            assert bfs_basic_order is not None
            assert dfs_basic_order is not None

            bfs_opt_time, bfs_opt_count = measure_average_time(bfs_optimized, graph, 0, repeats)
            dfs_opt_time, dfs_opt_count = measure_average_time(dfs_optimized, graph, 0, repeats)

            result = {
                "graph": name,
                "vertices": len(graph),
                "bfs_basic_time": bfs_basic_time,
                "dfs_basic_time": dfs_basic_time,
                "bfs_optimized_time": bfs_opt_time,
                "dfs_optimized_time": dfs_opt_time,
                "bfs_basic_visited": len(bfs_basic_order),
                "dfs_basic_visited": len(dfs_basic_order),
                "bfs_optimized_visited": bfs_opt_count,
                "dfs_optimized_visited": dfs_opt_count
            }

            results.append(result)

    return results


# Prints separate tables in the console for each graph type
def print_results_table(results):
    print("\nRESULTS TABLES BY GRAPH TYPE\n")

    graph_names = sorted(set(r["graph"] for r in results))

    for graph_name in graph_names:
        print(f"\nGraph Type: {graph_name}\n")

        header = (
            f"{'Vertices':<10}"
            f"{'BFS Basic':<15}"
            f"{'DFS Basic':<15}"
            f"{'BFS Opt':<15}"
            f"{'DFS Opt':<15}"
            f"{'Winner':<16}"
            f"{'Loser':<16}"
        )

        print(header)
        print("-" * len(header))

        data = [r for r in results if r["graph"] == graph_name]
        data.sort(key=lambda x: x["vertices"])

        for r in data:
            timings = {
                "BFS Basic": r["bfs_basic_time"],
                "DFS Basic": r["dfs_basic_time"],
                "BFS Optimized": r["bfs_optimized_time"],
                "DFS Optimized": r["dfs_optimized_time"],
            }
            winner = min(timings.items(), key=lambda kv: kv[1])[0]
            loser = max(timings.items(), key=lambda kv: kv[1])[0]
            print(
                f"{r['vertices']:<10}"
                f"{r['bfs_basic_time']:<15.6f}"
                f"{r['dfs_basic_time']:<15.6f}"
                f"{r['bfs_optimized_time']:<15.6f}"
                f"{r['dfs_optimized_time']:<15.6f}"
                f"{winner:<16}"
                f"{loser:<16}"
            )


# Saves results to CSV
def save_results_to_csv(results, filename="results.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "graph",
                "vertices",
                "bfs_basic_time",
                "dfs_basic_time",
                "bfs_optimized_time",
                "dfs_optimized_time",
                "bfs_basic_visited",
                "dfs_basic_visited",
                "bfs_optimized_visited",
                "dfs_optimized_visited"
            ]
        )
        writer.writeheader()
        writer.writerows(results)


# Saves one chart per graph type with all 4 versions together
def save_plots(results, folder="graphs"):
    os.makedirs(folder, exist_ok=True)

    graph_names = sorted(set(r["graph"] for r in results))

    for graph_name in graph_names:
        data = [r for r in results if r["graph"] == graph_name]
        data.sort(key=lambda x: x["vertices"])

        x = [r["vertices"] for r in data]
        bfs_basic = [r["bfs_basic_time"] for r in data]
        dfs_basic = [r["dfs_basic_time"] for r in data]
        bfs_opt = [r["bfs_optimized_time"] for r in data]
        dfs_opt = [r["dfs_optimized_time"] for r in data]

        plt.figure()
        plt.plot(
            x,
            bfs_basic,
            marker=ALGO_STYLES["BFS Basic"]["marker"],
            color=ALGO_STYLES["BFS Basic"]["color"],
            label="BFS Basic",
        )
        plt.plot(
            x,
            dfs_basic,
            marker=ALGO_STYLES["DFS Basic"]["marker"],
            color=ALGO_STYLES["DFS Basic"]["color"],
            label="DFS Basic",
        )
        plt.plot(
            x,
            bfs_opt,
            marker=ALGO_STYLES["BFS Optimized"]["marker"],
            color=ALGO_STYLES["BFS Optimized"]["color"],
            label="BFS Optimized",
        )
        plt.plot(
            x,
            dfs_opt,
            marker=ALGO_STYLES["DFS Optimized"]["marker"],
            color=ALGO_STYLES["DFS Optimized"]["color"],
            label="DFS Optimized",
        )
        plt.xlabel("Number of vertices")
        plt.ylabel("Execution time (seconds)")
        plt.title(f"{graph_name} - All 4 Versions")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{folder}/{graph_name}_all.png")
        plt.close()


def save_results_tables_image(results, output_folder="graphs"):
    os.makedirs(output_folder, exist_ok=True)

    graph_names = sorted(set(r["graph"] for r in results))

    for graph_name in graph_names:
        data = [r for r in results if r["graph"] == graph_name]
        data.sort(key=lambda x: x["vertices"])

        col_labels = ["Vertices", "BFS Basic", "DFS Basic", "BFS Opt", "DFS Opt", "Winner", "Loser"]
        cell_text = []

        for r in data:
            timings = {
                "BFS Basic": float(r["bfs_basic_time"]),
                "DFS Basic": float(r["dfs_basic_time"]),
                "BFS Optimized": float(r["bfs_optimized_time"]),
                "DFS Optimized": float(r["dfs_optimized_time"]),
            }
            winner = min(timings.items(), key=lambda kv: kv[1])[0]
            loser = max(timings.items(), key=lambda kv: kv[1])[0]

            cell_text.append(
                [
                    str(r["vertices"]),
                    f"{timings['BFS Basic']:.6f}",
                    f"{timings['DFS Basic']:.6f}",
                    f"{timings['BFS Optimized']:.6f}",
                    f"{timings['DFS Optimized']:.6f}",
                    winner,
                    loser,
                ]
            )

        fig_h = max(2.0, 0.42 * len(cell_text) + 1.2)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        ax.axis("off")
        ax.set_title(f"{graph_name} - Execution Time Results (all algorithms)")

        table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.05, 1.2)
        table.auto_set_column_width(col=list(range(len(col_labels))))

        fig.tight_layout()
        out_path = os.path.join(output_folder, f"{graph_name}_table.png")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


def main():
    random.seed(42)

    results = empirical_analysis()
    print_results_table(results)
    save_results_to_csv(results)
    save_plots(results)
    save_results_tables_image(results, output_folder="graphs")

    print("\nResults were saved to results.csv")
    print("Charts with all 4 versions were saved in the 'graphs' folder.")

    density_results = density_benchmark_analysis()
    print_density_table(density_results)
    save_density_results_to_csv(density_results, filename="density_results.csv")
    save_density_intersection_plot(density_results, output_folder="graphs")
    save_density_tables_image(density_results, output_folder="graphs")

    print("\nDensity benchmark results were saved to density_results.csv")
    print("Density intersection plot was saved to graphs/density_intersections.png")
    print("Density tables image was saved to graphs/density_tables.png")


if __name__ == "__main__":
    main()