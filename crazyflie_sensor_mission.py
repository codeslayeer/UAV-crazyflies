"""Simulation 3D d'une mission Crazyflie de collecte de capteurs.

Lancez :  python crazyflie_sensor_mission.py
Dépendances : numpy, matplotlib (pip install numpy matplotlib)
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


@dataclass
class Config:
    n_sensors: int = 7
    seed: int = 14
    area_xy: float = 5.0          # m, zone [-area_xy, area_xy]
    altitude: float = 1.2         # m
    max_speed: float = 1.0        # m/s
    max_accel: float = 1.2        # m/s²
    max_lateral_accel: float = 1.0  # m/s², contrainte de virage
    dwell_time: float = 2.0       # s de collecte par capteur
    dt: float = 0.05              # s


def distance(a, b):
    return float(np.linalg.norm(a - b))


def nearest_neighbor(points: np.ndarray) -> list[int]:
    """Ordre glouton : rapide, mais pas forcément optimal."""
    remaining = set(range(1, len(points)))
    route, current = [0], 0
    while remaining:
        nxt = min(remaining, key=lambda i: distance(points[current], points[i]))
        route.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return route + [0]


def two_opt(route: list[int], points: np.ndarray) -> list[int]:
    """Améliore la tournée gloutonne par échanges de segments (2-opt)."""
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                proposal = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if route_length(proposal, points) < route_length(best, points) - 1e-9:
                    best, improved = proposal, True
    return best


def route_length(route: list[int], points: np.ndarray) -> float:
    return sum(distance(points[a], points[b]) for a, b in zip(route[:-1], route[1:]))


def rounded_path(route: list[int], points: np.ndarray, radius: float) -> np.ndarray:
    """Arrondit les coins avec un arc circulaire, sans sortir des segments.

    Le rayon minimal physique vaut v²/a_lat. Les coins proches sont réduits
    automatiquement pour rester réalisables.
    """
    pieces = [points[route[0]]]
    for k in range(1, len(route) - 1):
        a, b, c = points[route[k - 1]], points[route[k]], points[route[k + 1]]
        u, v = a - b, c - b
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if lu < 1e-8 or lv < 1e-8:
            continue
        u, v = u / lu, v / lv
        angle = math.acos(np.clip(np.dot(u, v), -1.0, 1.0))
        if angle < 0.08 or abs(math.pi - angle) < 0.08:
            pieces.append(b)
            continue
        tangent = min(radius / math.tan(angle / 2), 0.35 * lu, 0.35 * lv)
        p_in, p_out = b + u * tangent, b + v * tangent
        # Quadratic Bézier: simple, continuous, and visibly respects rounded turns.
        t = np.linspace(0, 1, 12)[:, None]
        curve = (1 - t) ** 2 * p_in + 2 * (1 - t) * t * b + t ** 2 * p_out
        pieces.extend(curve[1:])
    pieces.append(points[route[-1]])
    return np.array(pieces)


def sample_motion(route_points: np.ndarray, cfg: Config) -> np.ndarray:
    """Échantillonne chaque vol et un stationnaire de collecte à chaque capteur.

    Chaque arrivée se fait à vitesse nulle : le drone peut alors changer de cap
    pendant son stationnaire. C'est une contrainte réaliste pour une collecte
    radio, et évite d'imposer un virage instantané à vitesse de croisière.
    """
    out = [route_points[0]]
    for i, (p, q) in enumerate(zip(route_points[:-1], route_points[1:])):
        length = distance(p, q)
        if length < 1e-9:
            continue
        # vitesse possible sur un segment en tenant compte accélération/freinage
        v_peak = min(cfg.max_speed, math.sqrt(length * cfg.max_accel))
        travel = max(2 * v_peak / cfg.max_accel, length / v_peak)
        n = max(2, math.ceil(travel / cfg.dt))
        alpha = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.extend(p + alpha * (q - p))
        out.append(q)
        if i < len(route_points) - 2:  # pas de collecte au retour à la base
            out.extend([q] * round(cfg.dwell_time / cfg.dt))
    return np.array(out)


def benchmark(points: np.ndarray, cfg: Config) -> dict[str, tuple[list[int], float]]:
    nn = nearest_neighbor(points)
    opt = two_opt(nn, points)
    # Exponentially many permutations are avoided: 2-opt is suitable after ~8 points.
    return {"Plus proche voisin": (nn, route_length(nn, points)),
            "Plus proche voisin + 2-opt": (opt, route_length(opt, points))}


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)
    start = np.array([0.0, 0.0, cfg.altitude])
    sensors = np.column_stack((rng.uniform(-cfg.area_xy, cfg.area_xy, (cfg.n_sensors, 2)),
                               rng.uniform(0.5, 2.0, cfg.n_sensors)))
    points = np.vstack((start, sensors))
    results = benchmark(points, cfg)
    name, (route, length) = min(results.items(), key=lambda x: x[1][1])
    r_min = cfg.max_speed ** 2 / cfg.max_lateral_accel
    path = rounded_path(route, points, r_min)  # enveloppe indicative des virages en vol
    mission_points = points[route]
    motion = sample_motion(mission_points, cfg)

    print("\n--- Benchmark des tournées ---")
    for method, (_, d) in results.items():
        print(f"{method:30s}: {d:5.2f} m, durée estimée {(d / cfg.max_speed + cfg.n_sensors * cfg.dwell_time):5.1f} s")
    print(f"\nChoix : {name}; rayon minimal de virage à vitesse max : {r_min:.2f} m")
    print("Ordre de visite :", " → ".join("Base" if i == 0 else f"S{i}" for i in route))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"Crazyflie — {name} | collecte {cfg.dwell_time:.1f} s / capteur")
    ax.plot(mission_points[:, 0], mission_points[:, 1], mission_points[:, 2], color="tab:blue", lw=1.5, label="route de mission")
    ax.plot(path[:, 0], path[:, 1], path[:, 2], color="tab:blue", alpha=.35, ls="--", label="enveloppe de virage")
    ax.scatter(*start, s=80, marker="s", color="tab:green", label="base / retour")
    ax.scatter(sensors[:, 0], sensors[:, 1], sensors[:, 2], s=55, color="tab:orange", label="capteurs")
    for i, p in enumerate(sensors, 1):
        ax.text(*p, f" S{i}")
    drone, = ax.plot([], [], [], "o", color="crimson", ms=8, label="Crazyflie")
    ax.set(xlim=(-cfg.area_xy-1, cfg.area_xy+1), ylim=(-cfg.area_xy-1, cfg.area_xy+1), zlim=(0, 2.5),
           xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax.legend(loc="upper left")

    def update(frame):
        p = motion[frame]
        drone.set_data_3d([p[0]], [p[1]], [p[2]])
        return drone,

    FuncAnimation(fig, update, frames=len(motion), interval=1000 * cfg.dt, blit=False, repeat=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
