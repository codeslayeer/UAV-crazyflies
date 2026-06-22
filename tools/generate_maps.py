"""Utility to generate and save random map datasets.

Usage:
    python tools/generate_maps.py --out-dir data/maps --count 20 --width 50 --height 50 --density 0.15

Saves each map as a .npy occupancy array and a small metadata .json with seed and params.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `from core.*` works when running the script
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from core.random_map import random_env


def save_map(env, path: Path):
    arr = env.to_occupancy_array()
    np.save(path.with_suffix(".npy"), arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random occupancy maps dataset.")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--density", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cell-size", type=float, default=0.1)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    base_seed = args.seed
    for i in range(args.count):
        seed = None if base_seed is None else base_seed + i
        env = random_env(width=args.width, height=args.height, cell_size=args.cell_size,
                         obstacle_fraction=args.density, seed=seed)
        fname = out / f"map_{i:03d}"
        save_map(env, fname)
        meta = {
            "index": i,
            "seed": seed,
            "width": args.width,
            "height": args.height,
            "density": args.density,
            "cell_size": args.cell_size,
        }
        with open(fname.with_suffix('.json'), 'w') as f:
            json.dump(meta, f)

    print(f"Saved {args.count} maps to: {out}")


if __name__ == '__main__':
    main()
