"""Collision validation for smoothed trajectories."""

from __future__ import annotations

from core.grid import Grid2D
from core.types import Path


def is_path_collision_free(env: Grid2D, path: Path, segment_steps: int = 20) -> bool:
    """
    Check that every waypoint and segment lies in free space.

    Used after B-spline smoothing to ensure the curve did not cut through
    obstacles — a common issue when rounding sharp grid turns.
    """
    if len(path) < 2:
        return False

    for point in path:
        if not env.is_free_point(point):
            return False

    for i in range(len(path) - 1):
        if not env.is_segment_free(path[i], path[i + 1], steps=segment_steps):
            return False

    return True
