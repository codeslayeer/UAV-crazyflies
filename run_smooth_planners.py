"""
Visual comparison: raw A*, smoothed A*, and RRT* on the default map.

Usage:
    python run_smooth_planners.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from core.grid import default_env
from metrics.path_metrics import compute_path_metrics
from planners.astar import astar
from planners.rrtstar import rrtstar
from smoothing.bspline import smooth_planner_result

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


def plot_result(ax, result, color: str, label: str, cell_size: float) -> None:
    if not result.success or result.path is None:
        print(f"  [{label}] No path.")
        return
    xs, ys = zip(*result.path)
    ax.plot(xs, ys, color=color, lw=1.5, label=label)
    m = compute_path_metrics(result.path, cell_size)
    if m:
        print(
            f"  [{label}] waypoints={m.num_waypoints} | length={m.length:.2f} m | "
            f"smoothness={m.smoothness:.2f} | max_curv={m.max_curvature:.3f}"
        )


def main() -> None:
    env = default_env()

    print("=== A* (raw) ===")
    raw = astar(env, START, GOAL)

    print("=== A* + B-spline ===")
    smoothed = smooth_planner_result(raw, env, num_points=100)

    print("=== RRT* ===")
    rrt = rrtstar(env, START, GOAL)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    draw_env(axes[0], env, "A* (raw grid path)")
    plot_result(axes[0], raw, "blue", "A*", env.cell_size)
    axes[0].legend()

    draw_env(axes[1], env, "A* + B-spline")
    plot_result(axes[1], raw, "blue", "A* raw", env.cell_size)
    plot_result(axes[1], smoothed, "purple", "B-spline", env.cell_size)
    axes[1].legend()

    draw_env(axes[2], env, "RRT*")
    plot_result(axes[2], rrt, "orange", "RRT*", env.cell_size)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("smooth_paths.png", dpi=120)
    print("\nImage saved: smooth_paths.png")
    plt.show()


if __name__ == "__main__":
    main()
