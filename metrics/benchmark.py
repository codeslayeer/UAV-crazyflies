"""Benchmark harness for comparing path planners."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Protocol

from core.grid import Grid2D
from core.types import AggregatedMetrics, Path, PlannerResult, Point, TrialRecord
from metrics.path_metrics import compute_path_metrics


class Planner(Protocol):
    """Interface every planner must satisfy for benchmarking."""

    def __call__(self, env: Grid2D, start: Point, goal: Point) -> PlannerResult: ...


def run_single_trial(
    planner: Planner,
    planner_name: str,
    env: Grid2D,
    start: Point,
    goal: Point,
    trial_index: int = 0,
) -> TrialRecord:
    """Run one planner invocation and attach path metrics."""
    result = planner(env, start, goal)
    metrics = compute_path_metrics(result.path, env.cell_size) if result.success else None
    return TrialRecord(
        planner_name=planner_name,
        trial_index=trial_index,
        result=result,
        metrics=metrics,
    )


def run_benchmark(
    planners: dict[str, Planner],
    env: Grid2D,
    start: Point,
    goal: Point,
    num_trials: int = 10,
) -> list[TrialRecord]:
    """
    Run each planner ``num_trials`` times and collect results.

    A* is deterministic; repeated trials confirm stability.
    RRT* is stochastic; multiple trials estimate mean performance.
    """
    records: list[TrialRecord] = []
    for name, planner in planners.items():
        for trial in range(num_trials):
            records.append(
                run_single_trial(planner, name, env, start, goal, trial_index=trial)
            )
    return records


def aggregate_results(records: list[TrialRecord]) -> list[AggregatedMetrics]:
    """Compute mean ± std per planner from trial records."""
    by_planner: dict[str, list[TrialRecord]] = {}
    for record in records:
        by_planner.setdefault(record.planner_name, []).append(record)

    summaries: list[AggregatedMetrics] = []
    for name, trials in by_planner.items():
        successes = [t for t in trials if t.result.success]
        success_rate = len(successes) / len(trials)

        def _stats(values: list[float]) -> tuple[float, float]:
            if not values:
                return 0.0, 0.0
            if len(values) == 1:
                return values[0], 0.0
            return statistics.mean(values), statistics.stdev(values)

        length_m, length_s = _stats([t.metrics.length for t in successes if t.metrics])
        time_m, time_s = _stats([t.result.compute_time for t in trials])
        nodes_m, nodes_s = _stats([float(t.result.nodes_explored) for t in trials])
        smooth_m, smooth_s = _stats([t.metrics.smoothness for t in successes if t.metrics])
        curv_m, curv_s = _stats([t.metrics.max_curvature for t in successes if t.metrics])

        summaries.append(
            AggregatedMetrics(
                planner_name=name,
                num_trials=len(trials),
                success_rate=success_rate,
                length_mean=length_m,
                length_std=length_s,
                compute_time_mean=time_m,
                compute_time_std=time_s,
                nodes_mean=nodes_m,
                nodes_std=nodes_s,
                smoothness_mean=smooth_m,
                smoothness_std=smooth_s,
                max_curvature_mean=curv_m,
                max_curvature_std=curv_s,
            )
        )

    return summaries


def timed_planner(
    planner_fn: Callable[[Grid2D, Point, Point], tuple[Path | None, int]],
) -> Planner:
    """
    Wrap a legacy planner function that returns (path, nodes_explored)
    into the Planner protocol with automatic timing.
    """

    def wrapper(env: Grid2D, start: Point, goal: Point) -> PlannerResult:
        t0 = time.perf_counter()
        path, nodes = planner_fn(env, start, goal)
        elapsed = time.perf_counter() - t0
        return PlannerResult(path=path, compute_time=elapsed, nodes_explored=nodes)

    return wrapper
