"""Random map generation utilities for dataset diversity."""

from __future__ import annotations

import random
from typing import Iterable

from core.grid import Grid2D
from core.types import Point


def random_env(
    width: int = 50,
    height: int = 50,
    cell_size: float = 1.0,
    obstacle_fraction: float = 0.15,
    block_min: int = 1,
    block_max: int = 5,
    seed: int | None = None,
    keep_points: Iterable[Point] | None = None,
) -> Grid2D:
    """Create a Grid2D populated with random rectangular obstacles.

    Parameters
    - obstacle_fraction: target fraction of occupied cells (0.0 - 1.0).
    - block_min/block_max: size range for random rectangular blocks (in cells).
    - seed: optional RNG seed for reproducibility.
    - keep_points: iterable of (x,y) points to keep free (rounded to ints).
    """
    if obstacle_fraction <= 0:
        return Grid2D(width, height, cell_size)

    rng = random.Random(seed)
    env = Grid2D(width, height, cell_size)

    total_cells = max(1, width * height)
    target = int(total_cells * min(max(obstacle_fraction, 0.0), 1.0))

    attempts = 0
    max_attempts = max(1000, target * 8)

    # Add random rectangles until we reach (or exceed) target obstacle count.
    while len(env.obstacles) < target and attempts < max_attempts:
        attempts += 1
        w = rng.randint(max(1, block_min), min(max(1, block_max), width))
        h = rng.randint(max(1, block_min), min(max(1, block_max), height))
        x = rng.randint(0, max(0, width - w))
        y = rng.randint(0, max(0, height - h))
        env.add_rectangle(x, y, w, h)

    # Ensure any requested keep_points are free
    if keep_points is not None:
        for p in keep_points:
            xi = int(round(p[0]))
            yi = int(round(p[1]))
            env.obstacles.discard((xi, yi))

    return env


def from_occupancy_array(arr, cell_size: float = 1.0) -> Grid2D:
    """Create a Grid2D from a 2D numpy-like occupancy array (1=obstacle).

    The input is expected to have shape (height, width).
    """
    try:
        import numpy as np
    except Exception:  # pragma: no cover - optional dependency
        np = None

    if np is None:
        raise RuntimeError("Numpy is required for from_occupancy_array")

    h, w = arr.shape
    env = Grid2D(w, h, cell_size)
    for y in range(h):
        for x in range(w):
            if arr[y, x]:
                env.add_obstacle(x, y)
    return env
