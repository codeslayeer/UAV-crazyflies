"""B-spline trajectory smoothing for grid-based paths."""

from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import splev, splprep

from core.grid import Grid2D
from core.types import Path, PlannerResult, Point
from smoothing.validator import is_path_collision_free


def simplify_collinear(path: Path, eps: float = 1e-6) -> Path:
    """
    Remove intermediate waypoints that lie on a straight line.

    Grid-based A* produces many collinear cell centers; removing them
    before spline fitting prevents the curve from being forced through
    redundant points and yields smoother trajectories.
    """
    if len(path) <= 2:
        return list(path)

    simplified: Path = [path[0]]
    for i in range(1, len(path) - 1):
        ax, ay = simplified[-1]
        bx, by = path[i]
        cx, cy = path[i + 1]
        # Cross product magnitude → deviation from collinearity.
        cross = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        if cross > eps:
            simplified.append(path[i])
    simplified.append(path[-1])
    return simplified


def smooth_bspline(
    path: Path,
    num_points: int = 100,
    degree: int = 3,
    smoothing: float | None = None,
    simplify: bool = True,
) -> Path | None:
    """
    Fit a B-spline through waypoints and resample uniformly.

    Args:
        path: Input polyline (e.g. from A*).
        num_points: Number of points on the smoothed curve.
        degree: Spline degree (3 = cubic). Automatically clamped to len(path)-1.
        smoothing: Scipy smoothing factor (None = auto, 0 = interpolate).
        simplify: Remove collinear points before fitting.

    Returns:
        Smoothed path as list of (x, y), or None on failure.
    """
    if not path or len(path) < 2:
        return None

    work_path = simplify_collinear(path) if simplify else list(path)

    if len(work_path) == 2:
        t = np.linspace(0.0, 1.0, num_points)
        x0, y0 = work_path[0]
        x1, y1 = work_path[1]
        return [(float(x0 + t_i * (x1 - x0)), float(y0 + t_i * (y1 - y0))) for t_i in t]

    xs = [p[0] for p in work_path]
    ys = [p[1] for p in work_path]

    k = min(degree, len(work_path) - 1)
    # Small positive s approximates rather than interpolates → smoother curves.
    s = smoothing if smoothing is not None else max(len(work_path) * 0.5, 1.0)

    try:
        tck, _ = splprep([xs, ys], s=s, k=k)
    except (ValueError, TypeError):
        return None

    u = np.linspace(0.0, 1.0, num_points)
    sx, sy = splev(u, tck)
    return list(zip(sx.tolist(), sy.tolist()))


def smooth_planner_result(
    result: PlannerResult,
    env: Grid2D,
    num_points: int = 100,
    degree: int = 3,
    fallback_to_original: bool = True,
) -> PlannerResult:
    """
    Apply B-spline smoothing to a planner result and validate collision-free.

    If the smoothed path collides with obstacles, returns the original path
    when ``fallback_to_original=True`` (safe default for benchmarking).
    """
    if not result.success or result.path is None:
        return result

    t0 = time.perf_counter()
    smoothed = smooth_bspline(result.path, num_points=num_points, degree=degree)
    smooth_time = time.perf_counter() - t0

    if smoothed is None:
        return result

    if not is_path_collision_free(env, smoothed):
        if fallback_to_original:
            return PlannerResult(
                path=result.path,
                compute_time=result.compute_time + smooth_time,
                nodes_explored=result.nodes_explored,
            )
        return PlannerResult(
            path=None,
            compute_time=result.compute_time + smooth_time,
            nodes_explored=result.nodes_explored,
        )

    return PlannerResult(
        path=smoothed,
        compute_time=result.compute_time + smooth_time,
        nodes_explored=result.nodes_explored,
    )


def make_astar_bspline_planner(
    num_points: int = 100,
    degree: int = 3,
):
    """
    Factory: returns a planner callable that runs A* then B-spline smoothing.

    Used by the benchmark harness as a third planner entry.
    """
    from planners.astar import astar

    def astar_bspline(env: Grid2D, start: Point, goal: Point) -> PlannerResult:
        raw = astar(env, start, goal)
        return smooth_planner_result(
            raw, env, num_points=num_points, degree=degree,
        )

    return astar_bspline
