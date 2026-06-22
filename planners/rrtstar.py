"""RRT* path planner in continuous 2D space with grid collision checking."""

from __future__ import annotations

import math
import random
import time

from core.grid import Grid2D
from core.types import Path, PlannerResult, Point


class Node:
    """Tree node for RRT*."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.parent: Node | None = None
        self.cost: float = 0.0

    def pos(self) -> Point:
        return (self.x, self.y)


def _dist(a: Node, b: Node) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def rrtstar(
    env: Grid2D,
    start: Point,
    goal: Point,
    max_iter: int = 2000,
    step_size: float = 2.0,
    goal_radius: float = 2.0,
    rewire_radius: float = 5.0,
) -> PlannerResult:
    """
    RRT* in continuous 2D with obstacle checks on the occupancy grid.

    Returns:
        PlannerResult with the path [(x, y), ...] or path=None if no solution found.
    """
    t0 = time.perf_counter()

    root = Node(*start)
    nodes: list[Node] = [root]

    def nearest(pt: Point) -> Node:
        return min(nodes, key=lambda n: math.hypot(n.x - pt[0], n.y - pt[1]))

    def steer(n_near: Node, pt: Point) -> Node:
        dx, dy = pt[0] - n_near.x, pt[1] - n_near.y
        d = math.hypot(dx, dy)
        if d <= step_size:
            return Node(pt[0], pt[1])
        return Node(n_near.x + dx / d * step_size, n_near.y + dy / d * step_size)

    def near_nodes(n_new: Node) -> list[Node]:
        return [n for n in nodes if _dist(n, n_new) <= rewire_radius]

    best_goal_node: Node | None = None

    for _ in range(max_iter):
        rand_pt: Point = goal if random.random() < 0.1 else (
            random.uniform(0, env.width),
            random.uniform(0, env.height),
        )

        n_near = nearest(rand_pt)
        n_new = steer(n_near, rand_pt)

        if not env.is_segment_free(n_near.pos(), n_new.pos()):
            continue

        neighbors = near_nodes(n_new)
        best_parent = n_near
        best_cost = n_near.cost + _dist(n_near, n_new)

        for n in neighbors:
            c = n.cost + _dist(n, n_new)
            if c < best_cost and env.is_segment_free(n.pos(), n_new.pos()):
                best_parent = n
                best_cost = c

        n_new.parent = best_parent
        n_new.cost = best_cost
        nodes.append(n_new)

        for n in neighbors:
            new_cost = n_new.cost + _dist(n_new, n)
            if new_cost < n.cost and env.is_segment_free(n_new.pos(), n.pos()):
                n.parent = n_new
                n.cost = new_cost

        if math.hypot(n_new.x - goal[0], n_new.y - goal[1]) <= goal_radius:
            if best_goal_node is None or n_new.cost < best_goal_node.cost:
                best_goal_node = n_new

    elapsed = time.perf_counter() - t0
    nodes_explored = len(nodes)

    if best_goal_node is None:
        return PlannerResult(path=None, compute_time=elapsed, nodes_explored=nodes_explored)

    path: Path = []
    node: Node | None = best_goal_node
    while node is not None:
        path.append(node.pos())
        node = node.parent

    return PlannerResult(
        path=path[::-1],
        compute_time=elapsed,
        nodes_explored=nodes_explored,
    )
