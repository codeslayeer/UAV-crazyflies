"""Abstract planner interface."""

from __future__ import annotations

from typing import Protocol

from core.grid import Grid2D
from core.types import PlannerResult, Point


class BasePlanner(Protocol):
    """Protocol implemented by all path planners."""

    def plan(self, env: Grid2D, start: Point, goal: Point) -> PlannerResult: ...
