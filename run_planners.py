"""
Visual comparison of A* and RRT* on the default environment.

Usage:
    python run_planners.py
"""
from __future__ import annotations

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from core.grid import default_env
from metrics.path_metrics import compute_path_metrics
from planners.astar import astar
from planners.rrtstar import rrtstar

START: tuple[float, float] = (2, 2)
GOAL: tuple[float, float] = (47, 47)


def draw_env(ax, env, title: str) -> None:
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    for x, y in env.obstacles:
        ax.add_patch(patches.Rectangle((x, y), 1, 1, color="gray"))

    ax.plot(*START, "go", ms=8, label="Start")
    ax.plot(*GOAL, "r*", ms=10, label="Goal")


def plot_path(ax, result, color: str, label: str) -> None:
    if not result.success:
        print(f"  [{label}] No path found ({result.compute_time * 1000:.1f} ms).")
        return

    path = result.path
    xs, ys = zip(*path)
    ax.plot(xs, ys, color=color, lw=1.5, label=label)

    metrics = compute_path_metrics(path, cell_size=1.0)
    assert metrics is not None
    print(
        f"  [{label}] {metrics.num_waypoints} waypoints | "
        f"length={metrics.length:.2f} | "
        f"smoothness={metrics.smoothness:.2f} | "
        f"time={result.compute_time * 1000:.1f} ms | "
        f"nodes={result.nodes_explored}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visual comparison of A* and RRT* on a map (default or random)."
    )
    parser.add_argument("--random", action="store_true", help="Use a randomly generated map")
    parser.add_argument("--width", type=int, default=50, help="Grid width for random map")
    parser.add_argument("--height", type=int, default=50, help="Grid height for random map")
    parser.add_argument(
        "--density",
        type=float,
        default=0.15,
        help="Target obstacle fraction for random map (0-1)",
    )
    parser.add_argument("--seed", type=int, help="RNG seed for random map")
    args = parser.parse_args()

    if args.random:
        # Generate a random environment and place start/goal near corners
        from core.random_map import random_env

        global START, GOAL
        START = (2, 2)
        GOAL = (args.width - 3, args.height - 3)
        env = random_env(
            width=args.width,
            height=args.height,
            cell_size=1.0,
            obstacle_fraction=args.density,
            seed=args.seed,
            keep_points=[START, GOAL],
        )
    else:
        env = default_env()

    print("=== A* ===")
    result_astar = astar(env, START, GOAL)

    print("=== RRT* ===")
    result_rrt = rrtstar(env, START, GOAL)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    draw_env(axes[0], env, "A*")
    plot_path(axes[0], result_astar, "blue", "A*")
    axes[0].legend()

    draw_env(axes[1], env, "RRT*")
    plot_path(axes[1], result_rrt, "orange", "RRT*")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("paths.png", dpi=120)
    print("\nImage saved: paths.png")
    plt.show()


if __name__ == "__main__":
    main()
