"""M4: hard-floor calibration for the WIDE posterior, on calibration graphs (seed namespace 10000-19999), never on benchmark data.

    python scripts/calibrate_floor_wide.py --model models/skeleton_posterior_wide.pkl   ->  results/floor_calibration_wide.json

Same rule as calibrate_floor.py (largest floor in the grid pinning <= 1 % of the TRUE edges), applied per size bucket and pooled.
``chosen`` is the largest floor that satisfies the rule in EVERY bucket (one conservative number for all d), so the floor
never depends on the d of a benchmark graph.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skeleton.posterior import SkeletonPosterior       # noqa: E402
from skeleton.wide import one_instance                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 8e-3, 1e-2, 2e-2, 3e-2, 5e-2]
BUCKETS = {"m3": (25, 40, 30), "d60": (60, 60, 20), "d100": (100, 100, 15), "d150": (150, 150, 8), "d200": (200, 200, 8)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "skeleton_posterior_wide.pkl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "floor_calibration_wide.json"))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-true-pinned", type=float, default=0.01)
    a = ap.parse_args(argv)
    post = SkeletonPosterior.load(a.model)
    per = {}
    for b, (lo, hi, cnt) in BUCKETS.items():
        pe, pn = [], []
        for i in range(max(1, int(round(cnt * a.scale)))):
            it = one_instance("calib", b, i, lo, hi)
            P = post.predict(it.X).P
            iu = np.triu_indices(P.shape[0], 1)
            pe.append(P[iu][it.skeleton[iu]]); pn.append(P[iu][~it.skeleton[iu]])
        per[b] = (np.concatenate(pe), np.concatenate(pn))
        print(f"{b}: {len(per[b][0])} true edges, {len(per[b][1])} non-edges", flush=True)
    pooled = (np.concatenate([v[0] for v in per.values()]), np.concatenate([v[1] for v in per.values()]))
    per["pooled"] = pooled
    table = {b: [dict(floor=f, true_edges_pinned=float(np.mean(pe <= f)), non_edges_pinned=float(np.mean(pn <= f))) for f in GRID]
             for b, (pe, pn) in per.items()}
    ok = [f for f in GRID if all(r["true_edges_pinned"] <= a.max_true_pinned for b in per if b != "pooled" for r in table[b] if r["floor"] == f)]
    chosen = max(ok) if ok else None
    out = dict(rule=f"largest floor in grid pinning <= {a.max_true_pinned:.0%} of true edges in EVERY bucket", model=a.model,
               seed_namespace="calib 10000-19999", chosen=chosen, released_floor=1e-3, table=table,
               n_true_edges={b: int(len(v[0])) for b, v in per.items()})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    for b in per:
        print(b, " ".join(f"{r['floor']:.3f}:{r['true_edges_pinned']:.1%}/{r['non_edges_pinned']:.0%}" for r in table[b]))
    print("chosen:", chosen)
    return out


if __name__ == "__main__":
    main()
