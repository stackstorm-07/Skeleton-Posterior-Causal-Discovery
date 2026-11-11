"""M4 / WP1: train the skeleton posterior on graphs with d = 25 ... 200 (the 'wide' posterior, used as the primary SPOT).

    python scripts/train_posterior_wide.py                      # full recipe, models/skeleton_posterior_wide.pkl
    python scripts/train_posterior_wide.py --scale 0.1 --out /tmp/w.pkl --results /tmp/w.json   # smoke run

Training graphs: see skeleton/wide.py (buckets, counts, seed namespaces).  At most ``--max-neg`` negative pairs per graph
are used to FIT the classifiers (inverse-sampling weights keep p calibrated); stage inputs are still computed for ALL pairs.
The narrow M3 model (models/skeleton_posterior.pkl) is untouched and stays available as the 'spot_narrow' ablation.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skeleton.posterior import PosteriorConfig, SkeletonPosterior      # noqa: E402
from skeleton.wide import TRAIN_BUCKETS, wide_instances                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="multiply the graph counts (smoke runs)")
    ap.add_argument("--max-neg", type=int, default=6000, help="negatives per graph used for fitting (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "skeleton_posterior_wide.pkl"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "posterior_training_wide.json"))
    a = ap.parse_args(argv)

    t = time.perf_counter()
    items = wide_instances("train", scale=a.scale)
    train = [it for _, it in items]
    t_sim = time.perf_counter() - t
    counts = {b: sum(1 for nm, _ in items if nm == b) for b in TRAIN_BUCKETS}
    print(f"simulated {len(train)} training graphs in {t_sim:.0f}s: {counts}", flush=True)

    cfg = PosteriorConfig(seed=a.seed, max_neg_per_graph=(a.max_neg or None))
    post = SkeletonPosterior(cfg).fit(train, verbose=True)
    post.info["recipe"] = dict(kind="wide", buckets={k: list(v) for k, v in TRAIN_BUCKETS.items()}, scale=a.scale, counts=counts,
                               max_neg=a.max_neg, seed_namespace="train 0-9999")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    post.save(a.out)
    out = dict(config=vars(a), counts=counts, t_simulate=t_sim, **{k: post.info[k] for k in ("t_features", "t_fit", "t_train_total")},
               stage_rows=post.info["rows"])
    os.makedirs(os.path.dirname(a.results), exist_ok=True)
    json.dump(out, open(a.results, "w"), indent=1)
    print(f"wide posterior: {len(train)} graphs, {post.info['t_train_total']:.0f}s "
          f"(features {post.info['t_features']:.0f}s, classifiers {post.info['t_fit']:.0f}s) -> {a.out}")


if __name__ == "__main__":
    main()
