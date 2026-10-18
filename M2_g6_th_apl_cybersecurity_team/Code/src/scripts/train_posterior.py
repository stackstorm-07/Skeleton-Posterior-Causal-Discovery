"""Offline training of the skeleton posterior (the one-time cost in the runtime breakdown).

    python scripts/train_posterior.py --n-train 400 --out models/skeleton_posterior.pkl

Writes the model and ``results/posterior_training.json`` (training time split into feature building and
classifier fitting, plus held-out AUROC / AUPRC / KL on freshly simulated graphs that were never trained on).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skeleton.simulator import make_instances                      # noqa: E402
from skeleton.posterior import PosteriorConfig, SkeletonPosterior, evaluate   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def summarize(rows):
    keys = ["auroc", "auprc", "kl", "brier", "ece", "recall_p05", "recall_p10",
            "auroc_stage0", "auroc_stage1", "auroc_stage2", "t_features", "t_model"]
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys if any(k in r for r in rows)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=60)
    ap.add_argument("--d-min", type=int, default=25)
    ap.add_argument("--d-max", type=int, default=40)
    ap.add_argument("--n-min", type=int, default=300)
    ap.add_argument("--n-max", type=int, default=2000)
    ap.add_argument("--ar", type=float, nargs="+", default=[0.0, 0.3, 0.6])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "skeleton_posterior.pkl"))
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "posterior_training.json"))
    a = ap.parse_args(argv)

    kw = dict(d_range=(a.d_min, a.d_max), n_range=(a.n_min, a.n_max), ar_rhos=tuple(a.ar))
    t = time.perf_counter()
    train = make_instances(a.n_train, seed=a.seed, **kw)
    test = make_instances(a.n_test, seed=a.seed + 10_000, **kw)          # disjoint seed: held-out graphs
    t_sim = time.perf_counter() - t

    post = SkeletonPosterior(PosteriorConfig(seed=a.seed)).fit(train, verbose=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    post.save(a.out)
    rows = evaluate(post, test)
    out = dict(config=vars(a), t_simulate=t_sim, **{k: post.info[k] for k in ("t_features", "t_fit", "t_train_total")},
               stage_rows=post.info["rows"], heldout=summarize(rows), heldout_rows=rows)
    os.makedirs(os.path.dirname(a.results), exist_ok=True)
    json.dump(out, open(a.results, "w"), indent=1)
    print(f"trained on {a.n_train} graphs in {post.info['t_train_total']:.0f}s "
          f"(features {post.info['t_features']:.0f}s, classifiers {post.info['t_fit']:.0f}s); model -> {a.out}")
    print("held-out:", {k: round(v, 3) for k, v in out["heldout"].items()})


if __name__ == "__main__":
    main()
