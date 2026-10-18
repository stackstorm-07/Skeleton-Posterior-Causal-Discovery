"""Calibrate SPOT's hard floor for *our* posterior, on simulated graphs disjoint from every benchmark graph.

The released SPOT code pins pairs with p <= 1e-3 to zero.  That number belongs to the released xgboost cascade; our
HistGradientBoosting cascade is less extreme (it rarely outputs p < 1e-3), so 1e-3 prunes little.  Rule used here:
the largest floor in a fixed grid that pins at most 1 % of the TRUE edges of held-out simulated graphs (seed 777,
d 25-40, AR rho in {0, .3, .6}); the benchmark graphs use other seeds and are never looked at.

    python scripts/calibrate_floor.py        ->  results/floor_calibration.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skeleton.posterior import SkeletonPosterior          # noqa: E402
from skeleton.simulator import make_instances             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3, 1e-2, 2e-2, 3e-2, 5e-2]


def main(n=40, seed=777, max_true_pinned=0.01):
    post = SkeletonPosterior.load(os.path.join(ROOT, "models", "skeleton_posterior.pkl"))
    pe, pn = [], []
    for it in make_instances(n, d_range=(25, 40), ar_rhos=(0.0, 0.3, 0.6), seed=seed):
        P = post.predict(it.X).P
        iu = np.triu_indices(P.shape[0], 1)
        pe.append(P[iu][it.skeleton[iu]])
        pn.append(P[iu][~it.skeleton[iu]])
    pe, pn = np.concatenate(pe), np.concatenate(pn)
    table = [dict(floor=f, true_edges_pinned=float(np.mean(pe <= f)), non_edges_pinned=float(np.mean(pn <= f))) for f in GRID]
    ok = [r["floor"] for r in table if r["true_edges_pinned"] <= max_true_pinned]
    out = dict(seed=seed, n_graphs=n, n_true_edges=int(len(pe)), n_non_edges=int(len(pn)), rule=f"largest floor in grid pinning <= {max_true_pinned:.0%} of true edges",
               chosen=max(ok), released_floor=1e-3, table=table)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "floor_calibration.json"), "w") as f:
        json.dump(out, f, indent=1)
    for r in table:
        print(f"floor {r['floor']:.3f}: true edges pinned {r['true_edges_pinned']:6.2%}   non-edges pinned {r['non_edges_pinned']:6.1%}")
    print("chosen:", out["chosen"])


if __name__ == "__main__":
    main()
