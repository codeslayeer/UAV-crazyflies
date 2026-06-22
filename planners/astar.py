"""A* path planner on a 2D grid with 8-connectivity."""

from __future__ import annotations

import heapq
import math
import time

from core.grid import Grid2D
from core.types import Path, PlannerResult, Point

NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def heuristic(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def astar(env: Grid2D, start: Point, goal: Point) -> PlannerResult:
    """
    A* on a 2D grid (8-connected).

    Returns:
        PlannerResult with the path [(x, y), ...] or path=None if unreachable.
    """
    t0 = time.perf_counter()
    nodes_explored = 0

    open_heap: list[tuple[float, float, Point]] = []
    heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))

    came_from: dict[Point, Point] = {}
    g_score: dict[Point, float] = {start: 0.0}

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        if g > g_score.get(current, float("inf")):
            continue

        nodes_explored += 1

        if current == goal:
            path: Path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            elapsed = time.perf_counter() - t0
            return PlannerResult(
                path=path[::-1],
                compute_time=elapsed,
                nodes_explored=nodes_explored,
            )

        for dx, dy in NEIGHBORS_8:
            nb = (current[0] + dx, current[1] + dy)
            if not env.is_free(int(nb[0]), int(nb[1])):
                continue
            step = math.hypot(dx, dy)
            new_g = g_score[current] + step
            if new_g < g_score.get(nb, float("inf")):
                g_score[nb] = new_g
                f = new_g + heuristic(nb, goal)
                heapq.heappush(open_heap, (f, new_g, nb))
                came_from[nb] = current

    elapsed = time.perf_counter() - t0
    return PlannerResult(path=None, compute_time=elapsed, nodes_explored=nodes_explored)
