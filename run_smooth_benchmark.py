"""
Benchmark A*, A* + B-spline, and RRT* — with CSV/LaTeX export.

Usage:
    python run_smooth_benchmark.py
    python run_smooth_benchmark.py --trials 10 --csv-dir results/
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
from smoothing.bspline import make_astar_bspline_planner

START: tuple[float, float] = (2, 2)
GOAL: tuple[float, float] = (47, 47)


def print_summary_table(summaries: list[AggregatedMetrics]) -> None:
    header = (
        f"{'Planner':<16} {'Success':>8} {'Length':>14} {'Time (ms)':>14} "
        f"{'Nodes':>12} {'Smoothness':>14} {'Max curv.':>12}"
    )
    print("\n" + "=" * len(header))
    print("SMOOTHING BENCHMARK — A* vs A*+B-spline vs RRT*")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for s in summaries:
        length = f"{s.length_mean:.2f}+/-{s.length_std:.2f}"
        time_ms = f"{s.compute_time_mean * 1000:.1f}+/-{s.compute_time_std * 1000:.1f}"
        nodes = f"{s.nodes_mean:.0f}+/-{s.nodes_std:.0f}"
        smooth = f"{s.smoothness_mean:.2f}+/-{s.smoothness_std:.2f}"
        curv = f"{s.max_curvature_mean:.3f}+/-{s.max_curvature_std:.3f}"
        success = f"{s.success_rate * 100:.0f}%"
        print(
            f"{s.planner_name:<16} {success:>8} {length:>14} {time_ms:>14} "
            f"{nodes:>12} {smooth:>14} {curv:>12}"
        )

    print("=" * len(header))
    print("\nKey comparison: A*+B-spline should reduce smoothness vs raw A*")
    print("while staying close in path length and remaining collision-free.\n")


def save_bar_chart(summaries: list[AggregatedMetrics], filepath: str) -> None:
    names = [s.planner_name for s in summaries]
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Smoothing Benchmark — A* vs A*+B-spline vs RRT*", fontsize=13)

    configs = [
        (axes[0, 0], "Path length (m)", [s.length_mean for s in summaries],
         [s.length_std for s in summaries], "steelblue"),
        (axes[0, 1], "Compute time (ms)", [s.compute_time_mean * 1000 for s in summaries],
         [s.compute_time_std * 1000 for s in summaries], "darkorange"),
        (axes[1, 0], "Smoothness (rad²)", [s.smoothness_mean for s in summaries],
         [s.smoothness_std for s in summaries], "indianred"),
        (axes[1, 1], "Max curvature", [s.max_curvature_mean for s in summaries],
         [s.max_curvature_std for s in summaries], "seagreen"),
    ]

    for ax, title, means, stds, color in configs:
        ax.bar(x, means, yerr=stds, capsize=4, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    print(f"Chart saved: {filepath}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark planners with B-spline smoothing on A* paths.",
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--save", type=str, default="smooth_benchmark.png")
    parser.add_argument(
        "--csv-dir", type=str, default="results",
        help="Directory for CSV and LaTeX exports",
    )
    parser.add_argument(
        "--num-points", type=int, default=100,
        help="Resampled points on the B-spline curve",
    )
    parser.add_argument("--random", action="store_true", help="Use random maps for benchmark")
    parser.add_argument("--width", type=int, default=50, help="Width for random maps")
    parser.add_argument("--height", type=int, default=50, help="Height for random maps")
    parser.add_argument("--density", type=float, default=0.15, help="Obstacle fraction for random maps")
    parser.add_argument("--seed", type=int, help="Seed for random map generator")
    args = parser.parse_args()

    if getattr(args, "random", False):
        env = random_env(width=args.width, height=args.height, cell_size=0.1,
                         obstacle_fraction=args.density, seed=args.seed,
                         keep_points=[START, GOAL])
    else:
        env = default_env()
    planners = {
        "A*": astar,
        "A*+B-spline": make_astar_bspline_planner(num_points=args.num_points),
        "RRT*": rrtstar,
    }

    print(f"Running smoothing benchmark ({args.trials} trials per planner)...")
    records = run_benchmark(planners, env, START, GOAL, num_trials=args.trials)
    summaries = aggregate_results(records)

    print_summary_table(summaries)
    save_bar_chart(summaries, args.save)

    out = Path(args.csv_dir)
    trials_csv = out / "smooth_benchmark_trials.csv"
    summary_csv = out / "smooth_benchmark_summary.csv"
    latex_tex = out / "smooth_benchmark_table.tex"

    export_trials_csv(records, trials_csv)
    export_summary_csv(summaries, summary_csv)
    export_latex_table(
        summaries,
        latex_tex,
        caption="Comparison of A*, A* with B-spline smoothing, and RRT*.",
        label="tab:smoothing_benchmark",
    )

    print(f"CSV (trials):  {trials_csv}")
    print(f"CSV (summary): {summary_csv}")
    print(f"LaTeX table:   {latex_tex}")


if __name__ == "__main__":
    main()
