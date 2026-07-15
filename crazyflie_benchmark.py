"""Banc d'essai de planification de mission Crazyflie.

Exemple :
    python crazyflie_benchmark.py --missions 100 --sensors 8 --output resultats

Compare les tournées de collecte sur exactement les mêmes missions aléatoires :
  - Aléatoire (référence volontairement faible)
  - Plus proche voisin (très rapide)
  - Plus proche voisin + 2-opt (bon compromis pour ce projet)
  - Held-Karp exact (optimum, seulement pour petits nombres de capteurs)
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class MissionConfig:
    sensors: int = 8
    area: float = 5.0
    altitude: float = 1.2
    max_speed: float = 1.0       # m/s
    max_accel: float = 1.2       # m/s²
    dwell: float = 2.0           # s / capteur
    hover_power: float = 5.0     # W, approximation à calibrer avec mesures réelles
    flight_power: float = 8.0    # W, approximation à calibrer avec mesures réelles


def length(route: list[int], points: np.ndarray) -> float:
    return sum(float(np.linalg.norm(points[a] - points[b])) for a, b in zip(route, route[1:]))


def flight_time(route: list[int], points: np.ndarray, cfg: MissionConfig) -> float:
    """Temps par segment, départ et arrivée à vitesse nulle au capteur."""
    total = 0.0
    switch_length = cfg.max_speed ** 2 / cfg.max_accel
    for a, b in zip(route, route[1:]):
        d = float(np.linalg.norm(points[a] - points[b]))
        total += (2 * math.sqrt(d / cfg.max_accel) if d <= switch_length
                  else 2 * cfg.max_speed / cfg.max_accel + (d - switch_length) / cfg.max_speed)
    return total


def nearest_neighbor(points: np.ndarray) -> list[int]:
    remaining, route, current = set(range(1, len(points))), [0], 0
    while remaining:
        nxt = min(remaining, key=lambda j: float(np.linalg.norm(points[current] - points[j])))
        route.append(nxt); remaining.remove(nxt); current = nxt
    return route + [0]


def two_opt(route: list[int], points: np.ndarray) -> list[int]:
    best, changed = route[:], True
    while changed:
        changed = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                candidate = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if length(candidate, points) + 1e-10 < length(best, points):
                    best, changed = candidate, True
    return best


def held_karp(points: np.ndarray) -> list[int]:
    """TSP exact par programmation dynamique : O(n² 2^n), limite pratique ~12."""
    n = len(points) - 1
    cost, parent = {}, {}
    for j in range(n):
        cost[(1 << j, j)] = float(np.linalg.norm(points[0] - points[j + 1]))
    for mask in range(1, 1 << n):
        for last in range(n):
            if not mask & (1 << last) or mask == 1 << last:
                continue
            previous = mask ^ (1 << last)
            best_value, best_prev = min(
                (cost[(previous, k)] + float(np.linalg.norm(points[k + 1] - points[last + 1])), k)
                for k in range(n) if previous & (1 << k))
            cost[(mask, last)], parent[(mask, last)] = best_value, best_prev
    full = (1 << n) - 1
    last = min(range(n), key=lambda j: cost[(full, j)] + float(np.linalg.norm(points[j + 1] - points[0])))
    reverse = []
    mask = full
    while True:
        reverse.append(last + 1)
        if mask == 1 << last:
            break
        last, mask = parent[(mask, last)], mask ^ (1 << last)
    return [0] + reverse[::-1] + [0]


def random_baseline(points: np.ndarray, rng: np.random.Generator, samples: int = 80) -> list[int]:
    """Meilleure parmi plusieurs tournées aléatoires : référence reproductible."""
    candidates = [[0] + list(rng.permutation(np.arange(1, len(points)))) + [0] for _ in range(samples)]
    return min(candidates, key=lambda r: length(r, points))


def evaluate(name: str, route: list[int], points: np.ndarray, cfg: MissionConfig, solve_seconds: float) -> dict:
    dist = length(route, points)
    flying = flight_time(route, points, cfg)
    total = flying + cfg.sensors * cfg.dwell
    energy = (flying * cfg.flight_power + cfg.sensors * cfg.dwell * cfg.hover_power) / 3600
    return {"method": name, "distance_m": dist, "flight_time_s": flying,
            "total_time_s": total, "energy_Wh": energy, "solve_ms": solve_seconds * 1000}


def one_mission(rng: np.random.Generator, cfg: MissionConfig) -> list[dict]:
    base = np.array([[0.0, 0.0, cfg.altitude]])
    sensors = np.column_stack((rng.uniform(-cfg.area, cfg.area, (cfg.sensors, 2)),
                               rng.uniform(.5, 2.0, cfg.sensors)))
    points, results = np.vstack((base, sensors)), []
    methods = [("Aléatoire (meilleur de 80)", lambda: random_baseline(points, rng)),
               ("Plus proche voisin", lambda: nearest_neighbor(points)),
               ("Plus proche voisin + 2-opt", lambda: two_opt(nearest_neighbor(points), points))]
    if cfg.sensors <= 12:
        methods.append(("Held-Karp exact", lambda: held_karp(points)))
    for name, planner in methods:
        tic = time.perf_counter(); route = planner(); elapsed = time.perf_counter() - tic
        results.append(evaluate(name, route, points, cfg, elapsed))
    optimum = next((r["distance_m"] for r in results if r["method"] == "Held-Karp exact"), min(r["distance_m"] for r in results))
    for row in results:
        row["gap_to_optimum_pct"] = 100 * (row["distance_m"] / optimum - 1)
    return results


def save_plot(rows: list[dict], output: Path) -> None:
    methods = list(dict.fromkeys(r["method"] for r in rows))
    metrics = [("distance_m", "Distance moyenne (m)"), ("total_time_s", "Temps total moyen (s)"),
               ("energy_Wh", "Énergie moyenne (Wh)"), ("solve_ms", "Calcul moyen (ms)")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7)); axes = axes.ravel()
    for ax, (key, label) in zip(axes, metrics):
        means = [np.mean([r[key] for r in rows if r["method"] == method]) for method in methods]
        ax.bar(range(len(methods)), means, color="tab:blue")
        ax.set_xticks(range(len(methods)), [m.replace("Plus proche voisin", "PPV").replace("Aléatoire (meilleur de 80)", "Aléatoire") for m in methods], rotation=17, ha="right")
        ax.set_ylabel(label); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Benchmark de planification — mêmes missions pour chaque méthode")
    fig.tight_layout(); fig.savefig(output / "benchmark.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--missions", type=int, default=50)
    parser.add_argument("--sensors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("resultats_benchmark"))
    args = parser.parse_args()
    if not 3 <= args.sensors <= 12:
        parser.error("--sensors doit être entre 3 et 12 (Held-Karp reste alors raisonnable).")
    cfg, rng, rows = MissionConfig(sensors=args.sensors), np.random.default_rng(args.seed), []
    for mission_id in range(args.missions):
        for row in one_mission(rng, cfg):
            row["mission"] = mission_id + 1; rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    save_plot(rows, args.output)
    print(f"\nBenchmark terminé : {args.missions} missions, {args.sensors} capteurs.")
    print(f"Résultats : {args.output.resolve()}")
    for method in dict.fromkeys(r["method"] for r in rows):
        selected = [r for r in rows if r["method"] == method]
        print(f"{method:28s} | distance {np.mean([r['distance_m'] for r in selected]):5.2f} m"
              f" | écart optimum {np.mean([r['gap_to_optimum_pct'] for r in selected]):4.1f} %"
              f" | calcul {np.mean([r['solve_ms'] for r in selected]):6.2f} ms")


if __name__ == "__main__":
    main()
