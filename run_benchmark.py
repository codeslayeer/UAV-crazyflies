"""
Systematic comparison of A* and RRT* on the default benchmark map.

Usage:
    python run_benchmark.py
    python run_benchmark.py --trials 20 --save benchmark_results.png
    python run_benchmark.py --csv-dir results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.grid import default_env
from core.random_map import random_env
from core.types import AggregatedMetrics
from metrics.benchmark import aggregate_results, run_benchmark
from metrics.export import export_latex_table, export_summary_csv, export_trials_csv
from planners.astar import astar
from planners.rrtstar import rrtstar

START: tuple[float, float] = (2, 2)
GOAL: tuple[float, float] = (47, 47)


def print_summary_table(summaries: list[AggregatedMetrics]) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'Planner':<10} {'Success':>8} {'Length':>14} {'Time (ms)':>14} "
        f"{'Nodes':>12} {'Smoothness':>14} {'Max curv.':>12}"
    )
    print("\n" + "=" * len(header))
    print("PLANNER BENCHMARK — default_env(), start=(2,2), goal=(47,47)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for s in summaries:
        length = f"{s.length_mean:.2f}±{s.length_std:.2f}"
        time_ms = f"{s.compute_time_mean * 1000:.1f}±{s.compute_time_std * 1000:.1f}"
        nodes = f"{s.nodes_mean:.0f}±{s.nodes_std:.0f}"
        smooth = f"{s.smoothness_mean:.2f}±{s.smoothness_std:.2f}"
        curv = f"{s.max_curvature_mean:.3f}±{s.max_curvature_std:.3f}"
        success = f"{s.success_rate * 100:.0f}%"
        print(
            f"{s.planner_name:<10} {success:>8} {length:>14} {time_ms:>14} "
            f"{nodes:>12} {smooth:>14} {curv:>12}"
        )

    print("=" * len(header))
    print("\nNotes:")
    print("  Length      — Euclidean path length (meters, cell_size=0.1)")
    print("  Smoothness  — sum of squared heading changes (rad²); lower = smoother")
    print("  Max curv.   — peak discrete curvature (1/units); lower = gentler turns")
    print("  RRT* std    — non-zero because the algorithm is stochastic\n")


def save_bar_chart(summaries: list[AggregatedMetrics], filepath: str) -> None:
    """Save a grouped bar chart comparing key metrics across planners."""
    names = [s.planner_name for s in summaries]
    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("A* vs RRT* — Benchmark Comparison", fontsize=13)

    metrics_config = [
        (axes[0, 0], "Path length (m)", [s.length_mean for s in summaries],
         [s.length_std for s in summaries], "steelblue"),
        (axes[0, 1], "Compute time (ms)", [s.compute_time_mean * 1000 for s in summaries],
         [s.compute_time_std * 1000 for s in summaries], "darkorange"),
        (axes[1, 0], "Waypoints explored", [s.nodes_mean for s in summaries],
         [s.nodes_std for s in summaries], "seagreen"),
        (axes[1, 1], "Smoothness (rad²)", [s.smoothness_mean for s in summaries],
         [s.smoothness_std for s in summaries], "indianred"),
    ]

    for ax, title, means, stds, color in metrics_config:
        ax.bar(x, means, width, yerr=stds, capsize=4, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    print(f"Chart saved: {filepath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark A* and RRT* planners.")
    parser.add_argument(
        "--trials", type=int, default=10,
        help="Number of runs per planner (default: 10)",
    )
    parser.add_argument(
        "--save", type=str, default="benchmark_results.png",
        help="Output path for the comparison chart",
    )
    parser.add_argument(
        "--csv-dir", type=str, default=None,
        help="If set, export trials CSV, summary CSV, and LaTeX table to this directory",
    )
    parser.add_argument("--random", action="store_true", help="Use random maps for benchmark")
    parser.add_argument("--width", type=int, default=50, help="Width for random maps")
    parser.add_argument("--height", type=int, default=50, help="Height for random maps")
    parser.add_argument("--density", type=float, default=0.15, help="Obstacle fraction for random maps")
    parser.add_argument("--seed", type=int, help="Seed for random map generator")
    args = parser.parse_args()

    # Allow random maps via CLI options
    if getattr(args, "random", False):
        env = random_env(width=getattr(args, "width", 50),
                         height=getattr(args, "height", 50),
                         cell_size=0.1,
                         obstacle_fraction=getattr(args, "density", 0.15),
                         seed=getattr(args, "seed", None),
                         keep_points=[START, GOAL])
    else:
        env = default_env()
    planners = {"A*": astar, "RRT*": rrtstar}

    print(f"Running benchmark ({args.trials} trials per planner)...")
    records = run_benchmark(planners, env, START, GOAL, num_trials=args.trials)
    summaries = aggregate_results(records)

    print_summary_table(summaries)
    save_bar_chart(summaries, args.save)

    if args.csv_dir:
        out = Path(args.csv_dir)
        export_trials_csv(records, out / "benchmark_trials.csv")
        export_summary_csv(summaries, out / "benchmark_summary.csv")
        export_latex_table(
            summaries,
            out / "benchmark_table.tex",
            caption="Comparison of A* and RRT* on the default benchmark map.",
            label="tab:planner_benchmark",
        )
        print(f"Exported CSV/LaTeX to: {out}/")


if __name__ == "__main__":
    main()
