"""Measure ABIC (shared optimizer, no mask) runtime vs graph size, to set benchmark budgets.

    python scripts/time_abic.py --dims 10 20 30 35 --n 500 --budget 300 --out results/abic_timing.json
One restart per run, 2 seeds, z-scored and raw data.  ``timed_out`` = hit the per-run budget.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgm.abic import run_abic  # noqa: E402
from pgm.dcd import DCDConfig  # noqa: E402
from skeleton.simulator import random_admg, simulate_linear_gaussian  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[10, 20, 30, 35])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--out", default="results/abic_timing.json")
    a = ap.parse_args()
    rows = []
    for d in a.dims:
        for std in (True, False):
            for seed in range(a.seeds):
                rng = np.random.default_rng(seed)
                D, B = random_admg(d, int(1.2 * d), 0.1, rng)
                X, _, _ = simulate_linear_gaussian(D, B, a.n, rng, standardize=std)
                r = run_abic(X, [f"X{i}" for i in range(d)], DCDConfig(seed=seed, time_budget=a.budget))
                row = dict(d=d, n=a.n, standardized=std, seed=seed, elapsed=r.elapsed, t_optimize=r.t_optimize,
                           t_select=r.t_select, n_fev=r.n_fev, n_solves=r.n_solves, n_outer=r.n_outer,
                           converged=r.converged, timed_out=r.timed_out, h=r.h)
                rows.append(row)
                print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
