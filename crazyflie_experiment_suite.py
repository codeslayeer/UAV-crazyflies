"""Campagne expérimentale A* vs RRT* pour le rapport Crazyflie.

Exemples :
  python crazyflie_experiment_suite.py --profile quick
  python crazyflie_experiment_suite.py --profile standard --repetitions 5
  python crazyflie_experiment_suite.py --profile full --output rapport_resultats
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crazyflie_obstacles_benchmark import (Config, draw_scene, make_scene, path_length,
                                            plan_mission, tour)


PROFILES = {
    "quick": [("simple", 3, 5, 350), ("dense", 7, 6, 550)],
    "standard": [("simple", 3, 5, 450), ("moyen", 7, 6, 650), ("complexe", 11, 7, 850)],
    "full": [("simple", 3, 5, 550), ("moyen", 8, 6, 850), ("complexe", 13, 7, 1200)],
}


def point_box_distance(p: np.ndarray, box: np.ndarray) -> float:
    """Distance euclidienne entre un point et la surface d'une boîte."""
    low = np.array([box[0], box[2], box[4]]); high = np.array([box[1], box[3], box[5]])
    return float(np.linalg.norm(np.maximum(low - p, 0) + np.maximum(p - high, 0)))


def sampled_path(path: np.ndarray, spacing: float = .08) -> np.ndarray:
    samples = []
    for a, b in zip(path, path[1:]):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        samples.extend(a + (b-a) * t/n for t in range(n))
    return np.array(samples + [path[-1]])


def trajectory_metrics(path: np.ndarray, boxes: list[np.ndarray]) -> dict[str, float]:
    samples = sampled_path(path)
    min_clearance = min(point_box_distance(p, box) for p in samples for box in boxes)
    vectors = np.diff(path, axis=0)
    angles = []
    for a, b in zip(vectors, vectors[1:]):
        if np.linalg.norm(a) > 1e-9 and np.linalg.norm(b) > 1e-9:
            angles.append(np.degrees(np.arccos(np.clip(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))))
    return {"waypoints": len(path), "turn_angle_deg": float(sum(angles)), "min_clearance_m": min_clearance}


def save_summary(rows: list[dict], output: Path) -> list[dict]:
    fields = ["scenario", "method", "trials", "success_rate_pct", "distance_m_mean", "distance_m_std",
              "solve_ms_mean", "solve_ms_std", "clearance_m_mean", "turn_deg_mean", "waypoints_mean"]
    summary = []
    for scenario in dict.fromkeys(r["scenario"] for r in rows):
        for method in ("A*", "RRT*"):
            chosen = [r for r in rows if r["scenario"] == scenario and r["method"] == method]
            success = [r for r in chosen if r["success"]]
            values = lambda key: [r[key] for r in success]
            summary.append({"scenario": scenario, "method": method, "trials": len(chosen),
                            "success_rate_pct": 100*len(success)/len(chosen),
                            "distance_m_mean": np.mean(values("distance_m")) if success else np.nan,
                            "distance_m_std": np.std(values("distance_m")) if success else np.nan,
                            "solve_ms_mean": np.mean([r["solve_ms"] for r in chosen]),
                            "solve_ms_std": np.std([r["solve_ms"] for r in chosen]),
                            "clearance_m_mean": np.mean(values("min_clearance_m")) if success else np.nan,
                            "turn_deg_mean": np.mean(values("turn_angle_deg")) if success else np.nan,
                            "waypoints_mean": np.mean(values("waypoints")) if success else np.nan})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
    return summary


def summary_figure(summary: list[dict], output: Path) -> None:
    scenarios = list(dict.fromkeys(r["scenario"] for r in summary)); metrics = [
        ("success_rate_pct", "Réussite (%)"), ("distance_m_mean", "Distance (m)"),
        ("solve_ms_mean", "Calcul (ms)"), ("clearance_m_mean", "Marge min. (m)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7)); width = .34; x = np.arange(len(scenarios))
    for ax, (key, label) in zip(axes.ravel(), metrics):
        for offset, method, color in [(-width/2, "A*", "tab:blue"), (width/2, "RRT*", "tab:orange")]:
            vals = [next(r[key] for r in summary if r["scenario"] == s and r["method"] == method) for s in scenarios]
            ax.bar(x+offset, vals, width, label=method, color=color)
        ax.set_xticks(x, scenarios); ax.set_ylabel(label); ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(); fig.suptitle("Campagne comparative A* / RRT* — environnements 3D")
    fig.tight_layout(); fig.savefig(output / "summary.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--repetitions", type=int, default=3, help="missions indépendantes par scénario")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("rapport_resultats"))
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    rng, rows = np.random.default_rng(args.seed), []
    for scenario, obstacle_count, sensor_count, iterations in PROFILES[args.profile]:
        cfg = Config(sensors=sensor_count, obstacles=obstacle_count, rrt_iterations=iterations)
        for repetition in range(1, args.repetitions + 1):
            print(f"{scenario} — mission {repetition}/{args.repetitions}…", flush=True)
            points, boxes = make_scene(rng, cfg); order = tour(points); paths = {}
            for method in ("A*", "RRT*"):
                path, solve_ms = plan_mission(method, order, points, boxes, cfg, rng); paths[method] = path
                row = {"scenario": scenario, "obstacles": obstacle_count, "sensors": sensor_count,
                       "rrt_iterations": iterations, "mission": repetition, "method": method,
                       "success": path is not None, "solve_ms": solve_ms}
                if path is not None:
                    row.update({"distance_m": path_length(path), **trajectory_metrics(path, boxes)})
                else:
                    row.update({"distance_m": np.nan, "waypoints": np.nan, "turn_angle_deg": np.nan, "min_clearance_m": np.nan})
                rows.append(row)
            draw_scene(points, boxes, paths, args.output / Path("."))
            # Conserver toutes les scènes : l'image est renommée immédiatement.
            scene = args.output / "mission_3d.png"
            scene.replace(args.output / f"scene_{scenario}_{repetition:02d}.png")
    with (args.output / "all_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    summary = save_summary(rows, args.output); summary_figure(summary, args.output)
    print(f"\nTerminé. Données et figures : {args.output.resolve()}")


if __name__ == "__main__":
    main()
