"""Prototype RL : Q-learning tabulaire pour une liaison 3D Crazyflie.

Il apprend à rejoindre un capteur dans une carte fixe avec obstacles, puis compare
le chemin obtenu à A*. C'est une base de travail RL, pas encore un contrôleur de vol.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from crazyflie_obstacles_benchmark import Config, astar, free_point, make_scene, path_length


MOVES = np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])


def make_grid(boxes, cfg, resolution):
    lo=np.array([-cfg.area,-cfg.area,cfg.z_min]); hi=np.array([cfg.area,cfg.area,cfg.z_max])
    shape=np.floor((hi-lo)/resolution).astype(int)+1
    free=np.zeros(shape,dtype=bool)
    for idx in np.ndindex(*shape): free[idx]=free_point(lo+np.array(idx)*resolution,boxes,cfg)
    return lo, shape, free


def closest_cell(point, lo, resolution, shape, free):
    candidate=np.clip(np.rint((point-lo)/resolution).astype(int),0,shape-1)
    if free[tuple(candidate)]: return tuple(candidate)
    choices=np.argwhere(free); return tuple(choices[np.argmin(np.linalg.norm(choices-candidate,axis=1))])


def train(start, goal, free, episodes, rng):
    q=np.zeros(free.shape+(len(MOVES),)); rewards=[]; epsilon=.95
    for _ in range(episodes):
        state=start; total=0.
        for _step in range(350):
            action=int(rng.integers(len(MOVES))) if rng.random()<epsilon else int(np.argmax(q[state]))
            nxt=tuple(np.array(state)+MOVES[action])
            valid=all(0<=n<free.shape[i] for i,n in enumerate(nxt)) and free[nxt]
            if not valid: nxt=state; reward=-5.
            elif nxt==goal: reward=100.
            else: reward=-1. - .02*np.linalg.norm(np.array(nxt)-np.array(goal))
            q[state+(action,)]+= .25*(reward+.96*np.max(q[nxt])-q[state+(action,)])
            total+=reward; state=nxt
            if state==goal: break
        rewards.append(total); epsilon=max(.05,epsilon*.9985)
    return q, rewards


def extract(start, goal, q, lo, resolution, free):
    state=start; cells=[state]; seen={state}
    for _ in range(400):
        if state==goal: break
        action=int(np.argmax(q[state])); nxt=tuple(np.array(state)+MOVES[action])
        if not (all(0<=n<free.shape[i] for i,n in enumerate(nxt)) and free[nxt]) or nxt in seen: return None
        cells.append(nxt); seen.add(nxt); state=nxt
    return np.array([lo+np.array(c)*resolution for c in cells]) if state==goal else None


def main():
    p=argparse.ArgumentParser(); p.add_argument('--episodes',type=int,default=5000); p.add_argument('--seed',type=int,default=9); p.add_argument('--output',type=Path,default=Path('rl_resultats')); args=p.parse_args()
    cfg=Config(sensors=1,obstacles=5); rng=np.random.default_rng(args.seed); points,boxes=make_scene(rng,cfg); start,goal=points
    resolution=.5; lo,shape,free=make_grid(boxes,cfg,resolution); s=closest_cell(start,lo,resolution,shape,free); g=closest_cell(goal,lo,resolution,shape,free)
    q,rewards=train(s,g,free,args.episodes,rng); rl_path=extract(s,g,q,lo,resolution,free); a_path=astar(start,goal,boxes,cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'rl_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['method','success','distance_m','episodes']);w.writeheader();w.writerow({'method':'A*','success':a_path is not None,'distance_m':path_length(a_path) if a_path is not None else '', 'episodes':0});w.writerow({'method':'Q-learning','success':rl_path is not None,'distance_m':path_length(rl_path) if rl_path is not None else '', 'episodes':args.episodes})
    fig,ax=plt.subplots(figsize=(8,3));ax.plot(rewards);ax.set(xlabel='Épisode',ylabel='Récompense cumulée',title='Apprentissage Q-learning');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(args.output/'rl_training.png',dpi=170);plt.close(fig)
    fig=plt.figure(figsize=(9,5));ax=fig.add_subplot(111,projection='3d')
    for b in boxes: ax.bar3d(b[0],b[2],b[4],b[1]-b[0],b[3]-b[2],b[5]-b[4],alpha=.25,color='gray')
    ax.scatter(*start,s=55,marker='s',color='tab:green',label='Départ');ax.scatter(*goal,s=55,color='tab:orange',label='Capteur')
    if a_path is not None: ax.plot(a_path[:,0],a_path[:,1],a_path[:,2],label='A*',color='tab:blue')
    if rl_path is not None: ax.plot(rl_path[:,0],rl_path[:,1],rl_path[:,2],label='Q-learning',color='tab:red')
    ax.set(xlim=(-5,5),ylim=(-5,5),zlim=(0,2.6),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)');ax.legend();fig.tight_layout();fig.savefig(args.output/'rl_scene.png',dpi=170);plt.close(fig)
    print(f'Terminé : {args.output.resolve()}')


if __name__=='__main__': main()
