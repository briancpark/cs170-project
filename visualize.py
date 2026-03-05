"""
Visualize CS 170 Breakout Room solver outputs.

Usage:
    python visualize.py                          # visualize all outputs
    python visualize.py outputs/small-1.out      # visualize a single output
    python visualize.py --size small             # visualize all small outputs
"""

import argparse
import glob
import os
from collections import Counter, defaultdict
from os.path import basename

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from parse import read_input_file
from utils import (
    calculate_happiness,
    calculate_happiness_for_room,
    calculate_stress_for_room,
)


def load_output(output_path):
    """Parse an output file into a student->room dict."""
    D = {}
    with open(output_path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 2:
                D[int(tokens[0])] = int(tokens[1])
    return D


def get_room_groups(D):
    """Convert student->room mapping to room->[students] mapping."""
    rooms = defaultdict(list)
    for student, room in D.items():
        rooms[room].append(student)
    return dict(rooms)


def visualize_single(input_path, output_path, save_dir=None):
    """Visualize a single input/output pair with a multi-panel figure."""
    G, s = read_input_file(input_path, 100)
    D = load_output(output_path)
    n = len(G.nodes)
    rooms = get_room_groups(D)
    k = len(rooms)
    budget_per_room = s / k
    total_happiness = calculate_happiness(D, G)

    name = basename(output_path).replace(".out", "")

    # Compute per-room stats
    room_ids = sorted(rooms.keys())
    room_happiness = []
    room_stress = []
    room_sizes = []
    for r in room_ids:
        students = rooms[r]
        room_happiness.append(calculate_happiness_for_room(students, G))
        room_stress.append(calculate_stress_for_room(students, G))
        room_sizes.append(len(students))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{name}  |  n={n}, k={k}, S_max={s:.3f}, budget/room={budget_per_room:.3f}, "
        f"total happiness={total_happiness:.2f}",
        fontsize=13,
        fontweight="bold",
    )

    # --- Panel 1: Graph with room coloring ---
    ax = axes[0, 0]
    cmap = plt.colormaps.get_cmap("tab20").resampled(max(k, 1))
    color_map = [cmap(D[node] % 20) for node in sorted(G.nodes)]
    pos = nx.spring_layout(G, seed=42, k=2.0 / np.sqrt(n))
    nx.draw_networkx(
        G,
        pos,
        ax=ax,
        node_color=color_map,
        node_size=300 if n <= 20 else 120,
        font_size=8 if n <= 20 else 5,
        edge_color="#cccccc",
        width=0.3,
        with_labels=n <= 20,
    )
    patches = [
        mpatches.Patch(color=cmap(i % 20), label=f"Room {r}")
        for i, r in enumerate(room_ids)
    ]
    if k <= 12:
        ax.legend(handles=patches, fontsize=7, loc="upper left", ncol=2)
    ax.set_title("Student Graph (colored by room)")

    # --- Panel 2: Happiness & Stress per room ---
    ax = axes[0, 1]
    x_pos = np.arange(len(room_ids))
    width = 0.35
    ax.bar(x_pos - width / 2, room_happiness, width, label="Happiness", color="#4CAF50")
    ax.bar(x_pos + width / 2, room_stress, width, label="Stress", color="#F44336")
    ax.axhline(
        y=budget_per_room,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Stress budget ({budget_per_room:.2f})",
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"R{r}" for r in room_ids], fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title("Happiness & Stress per Room")
    ax.legend(fontsize=8)

    # --- Panel 3: Room size distribution ---
    ax = axes[1, 0]
    ax.bar(
        x_pos,
        room_sizes,
        color=[cmap(i % 20) for i in range(len(room_ids))],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"R{r}" for r in room_ids], fontsize=8)
    ax.set_ylabel("Number of Students")
    ax.set_title("Room Sizes")
    for i, sz in enumerate(room_sizes):
        ax.text(i, sz + 0.1, str(sz), ha="center", fontsize=9, fontweight="bold")

    # --- Panel 4: Stress utilization (% of budget used) ---
    ax = axes[1, 1]
    utilization = [
        st / budget_per_room * 100 if budget_per_room > 0 else 0 for st in room_stress
    ]
    colors = [
        "#F44336" if u > 100 else "#FF9800" if u > 80 else "#4CAF50"
        for u in utilization
    ]
    ax.barh(x_pos, utilization, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(x=100, color="red", linestyle="--", linewidth=1.5, label="100% budget")
    ax.set_yticks(x_pos)
    ax.set_yticklabels([f"R{r}" for r in room_ids], fontsize=8)
    ax.set_xlabel("Stress Budget Utilization (%)")
    ax.set_title("Stress Budget Utilization per Room")
    ax.legend(fontsize=8)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def visualize_summary(input_dir, output_dir, save_dir=None):
    """Summary dashboard across all outputs."""
    output_files = sorted(glob.glob(os.path.join(output_dir, "*.out")))
    if not output_files:
        print("No output files found.")
        return

    records = {"small": [], "medium": [], "large": []}
    for out_path in output_files:
        name = basename(out_path).replace(".out", "")
        in_path = os.path.join(input_dir, name + ".in")
        if not os.path.exists(in_path):
            continue
        G, s = read_input_file(in_path, 100)
        D = load_output(out_path)
        n = len(G.nodes)
        k = len(set(D.values()))
        happiness = calculate_happiness(D, G)

        if name.startswith("small"):
            records["small"].append((name, n, k, happiness, s))
        elif name.startswith("medium"):
            records["medium"].append((name, n, k, happiness, s))
        else:
            records["large"].append((name, n, k, happiness, s))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "CS 170 Breakout Room Solver - Summary Dashboard",
        fontsize=15,
        fontweight="bold",
    )

    for col, size in enumerate(["small", "medium", "large"]):
        data = records[size]
        if not data:
            continue
        names, ns, ks, happs, budgets = zip(*data)

        # Top row: happiness distribution
        ax = axes[0, col]
        ax.hist(happs, bins=20, color="#4CAF50", edgecolor="black", alpha=0.8)
        ax.axvline(
            np.mean(happs),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(happs):.1f}",
        )
        ax.set_title(f"{size.capitalize()} (n={ns[0]}) - Happiness Distribution")
        ax.set_xlabel("Total Happiness")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

        # Bottom row: number of rooms distribution
        ax = axes[1, col]
        room_counts = Counter(ks)
        room_vals = sorted(room_counts.keys())
        ax.bar(
            room_vals,
            [room_counts[v] for v in room_vals],
            color="#2196F3",
            edgecolor="black",
            alpha=0.8,
        )
        ax.set_title(f"{size.capitalize()} - Rooms (k) Distribution")
        ax.set_xlabel("Number of Rooms (k)")
        ax.set_ylabel("Count")
        ax.set_xticks(room_vals)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "summary.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Summary saved to {os.path.join(save_dir, 'summary.png')}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize CS 170 breakout room outputs"
    )
    parser.add_argument(
        "output_file", nargs="?", default=None, help="Single output file to visualize"
    )
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        help="Visualize all outputs of this size",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary dashboard across all outputs",
    )
    parser.add_argument(
        "--save", default=None, help="Directory to save figures (instead of showing)"
    )
    args = parser.parse_args()

    input_dir = "inputs"
    output_dir = "outputs"

    if args.output_file:
        # Single file
        name = basename(args.output_file).replace(".out", "")
        in_path = os.path.join(input_dir, name + ".in")
        visualize_single(in_path, args.output_file, save_dir=args.save)
    elif args.size:
        # All of one size
        pattern = os.path.join(output_dir, f"{args.size}-*.out")
        files = sorted(glob.glob(pattern))
        print(f"Visualizing {len(files)} {args.size} outputs...")
        for f in files:
            name = basename(f).replace(".out", "")
            in_path = os.path.join(input_dir, name + ".in")
            if os.path.exists(in_path):
                visualize_single(in_path, f, save_dir=args.save or "viz")
        print(f"Saved to {'viz' if not args.save else args.save}/")
    elif args.summary:
        visualize_summary(input_dir, output_dir, save_dir=args.save)
    else:
        # Default: show summary + a few examples
        visualize_summary(input_dir, output_dir, save_dir=args.save or "viz")
        for size in ["small", "medium", "large"]:
            example = os.path.join(output_dir, f"{size}-1.out")
            if os.path.exists(example):
                name = basename(example).replace(".out", "")
                in_path = os.path.join(input_dir, name + ".in")
                if os.path.exists(in_path):
                    visualize_single(in_path, example, save_dir=args.save or "viz")
        print(f"Saved to {'viz' if not args.save else args.save}/")
