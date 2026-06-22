"""Path quality metrics for planner comparison and dataset labelling."""

from __future__ import annotations

import math

from core.types import Path, PathMetrics


def _segment_lengths(path: Path) -> list[float]:
    return [
        math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]


def _headings(path: Path) -> list[float]:
    """Heading angle (rad) of each segment, range (-pi, pi]."""
    return [math.atan2(path[i + 1][1] - path[i][1], path[i + 1][0] - path[i][0])
            for i in range(len(path) - 1)]


def path_length(path: Path | None, cell_size: float = 1.0) -> float:
    """
    Total Euclidean path length.

    Args:
        path: Sequence of (x, y) waypoints.
        cell_size: Meters per grid cell (1.0 = unitless grid coordinates).
    """
    if not path or len(path) < 2:
        return 0.0
    return sum(_segment_lengths(path)) * cell_size


def smoothness(path: Path | None) -> float:
    """
    Sum of squared heading changes between consecutive segments (rad²).

    Lower values indicate smoother trajectories. A straight line has smoothness 0.
    Grid-based A* paths typically score high due to 90°/45° turns.
    """
    if not path or len(path) < 3:
        return 0.0

    headings = _headings(path)
    total = 0.0
    for i in range(len(headings) - 1):
        delta = headings[i + 1] - headings[i]
        # Wrap to [-pi, pi] for the shortest angular difference.
        delta = (delta + math.pi) % (2 * math.pi) - math.pi
        total += delta ** 2
    return total


def max_curvature(path: Path | None) -> float:
    """
    Maximum discrete curvature along the path (1 / turning radius).

    Uses the Menger curvature formula for three consecutive points.
    Returns 0 for paths with fewer than 3 points.
    """
    if not path or len(path) < 3:
        return 0.0

    max_k = 0.0
    for i in range(1, len(path) - 1):
        ax, ay = path[i - 1]
        bx, by = path[i]
        cx, cy = path[i + 1]

        ab = math.hypot(bx - ax, by - ay)
        bc = math.hypot(cx - bx, cy - by)
        ca = math.hypot(ax - cx, ay - cy)

        # Triangle area via cross product → circumradius R = abc / (4 * area).
        area2 = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        if area2 < 1e-12:
            continue
        curvature = (4.0 * area2) / (ab * bc * ca)
        max_k = max(max_k, curvature)

    return max_k


def compute_path_metrics(path: Path | None, cell_size: float = 1.0) -> PathMetrics | None:
    """Compute all path metrics, or None if the path is invalid."""
    if not path or len(path) < 2:
        return None

    return PathMetrics(
        length=path_length(path, cell_size),
        num_waypoints=len(path),
        smoothness=smoothness(path),
        max_curvature=max_curvature(path),
    )
