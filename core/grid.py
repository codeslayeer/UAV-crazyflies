"""2D occupancy grid for classical path planning."""

from __future__ import annotations

from core.types import Point


class Grid2D:
    """
    2D grid with rectangular cell obstacles.

    Coordinates are cell indices (integers). Use ``cell_size`` to convert
    to meters for Crazyflie deployment (e.g. cell_size=0.1 → 5 m × 5 m for 50×50).
    """

    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        cell_size: float = 1.0,
    ) -> None:
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.obstacles: set[tuple[int, int]] = set()

    def add_obstacle(self, x: int, y: int) -> None:
        self.obstacles.add((x, y))

    def add_rectangle(self, x: int, y: int, w: int, h: int) -> None:
        for i in range(x, x + w):
            for j in range(y, y + h):
                self.obstacles.add((i, j))

    def is_free(self, x: int, y: int) -> bool:
        return (
            0 <= x < self.width
            and 0 <= y < self.height
            and (x, y) not in self.obstacles
        )

    def is_free_point(self, point: Point) -> bool:
        return self.is_free(int(round(point[0])), int(round(point[1])))

    def is_segment_free(self, p1: Point, p2: Point, steps: int = 20) -> bool:
        """Check that a straight segment does not cross occupied cells."""
        for i in range(steps + 1):
            t = i / steps
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            if not self.is_free(int(round(x)), int(round(y))):
                return False
        return True

    def to_occupancy_array(self):
        """Return (height, width) numpy array: 0 = free, 1 = obstacle."""
        import numpy as np

        grid = np.zeros((self.height, self.width), dtype=np.uint8)
        for x, y in self.obstacles:
            grid[y, x] = 1
        return grid


def default_env() -> Grid2D:
    """Standard benchmark environment with three vertical walls."""
    env = Grid2D(50, 50, cell_size=0.1)
    env.add_rectangle(10, 5, 3, 20)
    env.add_rectangle(20, 25, 3, 20)
    env.add_rectangle(30, 5, 3, 15)
    return env
