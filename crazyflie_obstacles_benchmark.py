"""Simulation 3D avec obstacles : comparaison A* et RRT*.

Exemple : python crazyflie_obstacles_benchmark.py --missions 30 --sensors 6
Les résultats sont sauvegardés dans resultats_obstacles/.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Config:
    sensors: int = 5
    obstacles: int = 5
    area: float = 5.0
    z_min: float = .4
    z_max: float = 2.5
    clearance: float = .28       # rayon drone + marge de sécurité (m)
    grid: float = .45             # résolution A* (m) : assez fine pour ce prototype
    rrt_step: float = .55         # pas RRT* (m)
    rrt_iterations: int = 600     # mode interactif ; augmenter pour une étude longue
    max_speed: float = 1.0
    dwell: float = 2.0


def inside_box(p: np.ndarray, box: np.ndarray, margin: float) -> bool:
    """box = [xmin, xmax, ymin, ymax, zmin, zmax]."""
    return (box[0] - margin <= p[0] <= box[1] + margin and
            box[2] - margin <= p[1] <= box[3] + margin and
            box[4] - margin <= p[2] <= box[5] + margin)


def free_point(p: np.ndarray, boxes: list[np.ndarray], cfg: Config) -> bool:
    return (-cfg.area <= p[0] <= cfg.area and -cfg.area <= p[1] <= cfg.area and
            cfg.z_min <= p[2] <= cfg.z_max and not any(inside_box(p, b, cfg.clearance) for b in boxes))


def free_segment(a: np.ndarray, b: np.ndarray, boxes: list[np.ndarray], cfg: Config) -> bool:
    steps = max(2, math.ceil(np.linalg.norm(b - a) / .08))
    return all(free_point(a + (b - a) * t / steps, boxes, cfg) for t in range(steps + 1))


def make_scene(rng: np.random.Generator, cfg: Config) -> tuple[np.ndarray, list[np.ndarray]]:
    base = np.array([0., 0., 1.2]); boxes: list[np.ndarray] = []
    # Boîtes verticales : bâtiments, étagères ou obstacles dans une salle.
    while len(boxes) < cfg.obstacles:
        x, y = rng.uniform(-4.3, 4.3, 2); sx, sy = rng.uniform(.45, 1.1, 2)
        h = rng.uniform(1.0, 2.25); candidate = np.array([x-sx/2, x+sx/2, y-sy/2, y+sy/2, 0., h])
        if not inside_box(base, candidate, cfg.clearance + .25): boxes.append(candidate)
    points = [base]
    while len(points) < cfg.sensors + 1:
        p = np.array([rng.uniform(-4.7, 4.7), rng.uniform(-4.7, 4.7), rng.uniform(.55, 2.25)])
        if free_point(p, boxes, cfg) and all(np.linalg.norm(p-q) > 1.0 for q in points): points.append(p)
    return np.array(points), boxes


def tour(points: np.ndarray) -> list[int]:
    remaining, order, current = set(range(1, len(points))), [0], 0
    while remaining:
        nxt = min(remaining, key=lambda j: np.linalg.norm(points[current]-points[j]))
        order.append(nxt); remaining.remove(nxt); current = nxt
    order.append(0)
    # 2-opt pour ne pas confondre l'ordre de visite avec la comparaison A*/RRT*.
    changed = True
    def cost(r): return sum(np.linalg.norm(points[a]-points[b]) for a,b in zip(r,r[1:]))
    while changed:
        changed = False
        for i in range(1, len(order)-2):
            for j in range(i+1, len(order)-1):
                candidate = order[:i]+order[i:j+1][::-1]+order[j+1:]
                if cost(candidate) < cost(order)-1e-9: order, changed = candidate, True
    return order


def astar(start: np.ndarray, goal: np.ndarray, boxes: list[np.ndarray], cfg: Config) -> np.ndarray | None:
    """A* 3D à connectivité 26, puis lissage par visibilité directe."""
    lo = np.array([-cfg.area, -cfg.area, cfg.z_min]); step = cfg.grid
    shape = np.floor((np.array([cfg.area, cfg.area, cfg.z_max])-lo)/step).astype(int)+1
    def cell(p): return tuple(np.clip(np.rint((p-lo)/step).astype(int), 0, shape-1))
    def pos(c): return lo + np.array(c)*step
    start_c, goal_c = cell(start), cell(goal)
    open_heap=[(0., start_c)]; g={start_c:0.}; parent={}; closed=set()
    moves=[(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1) if (i,j,k)!=(0,0,0)]
    while open_heap:
        _, cur=heapq.heappop(open_heap)
        if cur in closed: continue
        if cur == goal_c:
            cells=[cur]
            while cur in parent: cur=parent[cur]; cells.append(cur)
            path=np.vstack([start, *(pos(c) for c in cells[-2:0:-1]), goal])
            return smooth(path, boxes, cfg)
        closed.add(cur)
        for move in moves:
            nxt=tuple(np.array(cur)+move)
            if any(n<0 or n>=shape[i] for i,n in enumerate(nxt)): continue
            a,b=pos(cur),pos(nxt)
            if not free_point(b, boxes, cfg) or not free_segment(a,b,boxes,cfg): continue
            candidate=g[cur]+np.linalg.norm(b-a)
            if candidate < g.get(nxt, float('inf')):
                g[nxt]=candidate; parent[nxt]=cur
                heapq.heappush(open_heap,(candidate+np.linalg.norm(b-goal),nxt))
    return None


def smooth(path: np.ndarray, boxes: list[np.ndarray], cfg: Config) -> np.ndarray:
    out=[path[0]]; i=0
    while i < len(path)-1:
        j=len(path)-1
        while j>i+1 and not free_segment(path[i],path[j],boxes,cfg): j-=1
        out.append(path[j]); i=j
    return np.array(out)


def rrt_star(start: np.ndarray, goal: np.ndarray, boxes: list[np.ndarray], cfg: Config, rng: np.random.Generator) -> np.ndarray | None:
    """RRT* 3D : échantillonnage continu, sélection de parent et recâblage."""
    nodes=[start.copy()]; parents=[-1]; costs=[0.]; best_goal=None
    last_improvement = 0
    for iteration in range(cfg.rrt_iterations):
        sample=goal if rng.random()<.12 else np.array([rng.uniform(-cfg.area,cfg.area), rng.uniform(-cfg.area,cfg.area), rng.uniform(cfg.z_min,cfg.z_max)])
        nearest=min(range(len(nodes)), key=lambda i: np.linalg.norm(nodes[i]-sample))
        direction=sample-nodes[nearest]; norm=np.linalg.norm(direction)
        if norm < 1e-9: continue
        new=nodes[nearest]+direction/norm*min(cfg.rrt_step,norm)
        if not free_segment(nodes[nearest],new,boxes,cfg): continue
        near=[i for i,p in enumerate(nodes) if np.linalg.norm(p-new)<1.05 and free_segment(p,new,boxes,cfg)]
        parent=min(near, key=lambda i: costs[i]+np.linalg.norm(nodes[i]-new)) if near else nearest
        nodes.append(new); parents.append(parent); costs.append(costs[parent]+np.linalg.norm(nodes[parent]-new)); ni=len(nodes)-1
        for i in near:
            candidate=costs[ni]+np.linalg.norm(nodes[i]-new)
            if candidate+1e-9 < costs[i]: parents[i]=ni; costs[i]=candidate
        if np.linalg.norm(new-goal) <= cfg.rrt_step and free_segment(new,goal,boxes,cfg):
            goal_cost=costs[ni]+np.linalg.norm(new-goal)
            if best_goal is None or goal_cost < best_goal[0]:
                best_goal=(goal_cost,ni); last_improvement=iteration
        # Après une solution, 120 itérations supplémentaires suffisent ici pour
        # améliorer la trajectoire sans bloquer le benchmark inutilement.
        if best_goal is not None and iteration-last_improvement >= 120: break
    if best_goal is None: return None
    indices=[]; cur=best_goal[1]
    while cur != -1: indices.append(cur); cur=parents[cur]
    return smooth(np.vstack([*(nodes[i] for i in indices[::-1]), goal]), boxes, cfg)


def path_length(path: np.ndarray) -> float: return float(sum(np.linalg.norm(a-b) for a,b in zip(path,path[1:])))


def plan_mission(method: str, order: list[int], points: np.ndarray, boxes: list[np.ndarray], cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray | None, float]:
    all_paths=[]; tic=time.perf_counter()
    for a,b in zip(order,order[1:]):
        leg=astar(points[a],points[b],boxes,cfg) if method=='A*' else rrt_star(points[a],points[b],boxes,cfg,rng)
        if leg is None: return None, (time.perf_counter()-tic)*1000
        all_paths.append(leg if not all_paths else leg[1:])
    return np.vstack(all_paths), (time.perf_counter()-tic)*1000


def draw_scene(points: np.ndarray, boxes: list[np.ndarray], paths: dict[str,np.ndarray|None], output: Path) -> None:
    fig=plt.figure(figsize=(10,5)); axes=[fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
    for ax,(name,path) in zip(axes,paths.items()):
        for b in boxes: ax.bar3d(b[0],b[2],b[4],b[1]-b[0],b[3]-b[2],b[5]-b[4],alpha=.25,color='gray')
        ax.scatter(*points[0],marker='s',s=55,color='tab:green'); ax.scatter(points[1:,0],points[1:,1],points[1:,2],color='tab:orange')
        if path is not None: ax.plot(path[:,0],path[:,1],path[:,2],color='tab:blue',lw=2)
        ax.set(title=name,xlim=(-5,5),ylim=(-5,5),zlim=(0,2.6),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
    fig.tight_layout(); fig.savefig(output/'mission_3d.png',dpi=160); plt.close(fig)


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--missions',type=int,default=3); p.add_argument('--sensors',type=int,default=5); p.add_argument('--obstacles',type=int,default=5); p.add_argument('--rrt-iterations',type=int,default=600); p.add_argument('--seed',type=int,default=7); p.add_argument('--output',type=Path,default=Path('resultats_obstacles')); args=p.parse_args()
    cfg=Config(sensors=args.sensors,obstacles=args.obstacles,rrt_iterations=args.rrt_iterations); rng=np.random.default_rng(args.seed); rows=[]; args.output.mkdir(parents=True,exist_ok=True)
    for mission in range(1,args.missions+1):
        print(f'Mission {mission}/{args.missions}…', flush=True)
        points,boxes=make_scene(rng,cfg); order=tour(points); paths={}
        for method in ('A*','RRT*'):
            path,solve=plan_mission(method,order,points,boxes,cfg,rng); paths[method]=path
            dist=path_length(path) if path is not None else float('nan')
            rows.append({'mission':mission,'method':method,'success':path is not None,'distance_m':dist,'total_time_s':dist/cfg.max_speed+cfg.sensors*cfg.dwell if path is not None else float('nan'),'solve_ms':solve})
        if mission==1: draw_scene(points,boxes,paths,args.output)
    with (args.output/'benchmark_obstacles.csv').open('w',newline='',encoding='utf-8') as f: writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print(f'\nRésultats : {args.output.resolve()}')
    for method in ('A*','RRT*'):
        r=[x for x in rows if x['method']==method]; ok=[x for x in r if x['success']]
        print(f"{method:4s} | réussite {len(ok)/len(r):.0%} | distance {np.mean([x['distance_m'] for x in ok]):.2f} m | calcul {np.mean([x['solve_ms'] for x in r]):.1f} ms")


if __name__=='__main__': main()
