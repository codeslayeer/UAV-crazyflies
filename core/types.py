"""Shared types for planners, metrics, and future dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field

# Continuous 2D coordinate (grid units or meters when scaled by cell_size).
Point = tuple[float, float]
Path = list[Point]


@dataclass
class PlannerResult:
    """Unified output from any path planner."""

    path: Path | None
    compute_time: float  # seconds
    nodes_explored: int
    success: bool = field(init=False)

    def __post_init__(self) -> None:
        self.success = self.path is not None and len(self.path) >= 2


@dataclass
class PathMetrics:
    """Geometric and kinematic quality measures for a single path."""

    length: float
    num_waypoints: int
    smoothness: float  # sum of squared heading changes (rad²); lower is smoother
    max_curvature: float  # 1 / minimum turning radius (1/units)


@dataclass
class TrialRecord:
    """One planner run with path quality metrics."""

    planner_name: str
    trial_index: int
    result: PlannerResult
    metrics: PathMetrics | None  # None when planning failed


@dataclass
class AggregatedMetrics:
    """Mean ± std over multiple trials (used for stochastic planners like RRT*)."""

    planner_name: str
    num_trials: int
    success_rate: float
    length_mean: float
    length_std: float
    compute_time_mean: float
    compute_time_std: float
    nodes_mean: float
    nodes_std: float
    smoothness_mean: float
    smoothness_std: float
    max_curvature_mean: float
    max_curvature_std: float
