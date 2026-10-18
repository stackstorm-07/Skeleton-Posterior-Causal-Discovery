"""Stage 2c: does SPOT beat ABIC on synthetic linear-Gaussian ancestral ADMGs (the setting the paper claims)?

    python scripts/spot_vs_abic_synthetic.py --d 25 --n 1000 --seeds 0 1 2 3 --restarts 1 --out results/syn_2c_main.jsonl

For every graph (seed) the *same* data, optimizer settings, restart count and optimizer seed are given to every
variant; the variants differ only in what the skeleton posterior contributes:

  abic            no mask, no pruning                                  (the baseline)
  spot            Eq. 9 mask + hard floor p <= 1e-3 (the released value)  (SPOT as released)
  spot_cal        Eq. 9 mask + hard floor p <= 6e-3 (calibrated for our posterior, see calibrate_floor.py)
  spot_nofloor    Eq. 9 mask only                                       (ablation: is it the stochastic mask?)
  spot_floor_only calibrated floor only, no mask (ABIC on pruned pairs) (ablation: is it just pruning?)
  spot_random     spot_cal with the same probabilities on random pairs  (ablation: does the *information* matter?)
  spot_oracle     SPOT with the true skeleton as posterior              (upper bound)
  spot_released   the other released script (sign-blind, 0.1 scaling)

Results are appended to a JSONL file (one line per graph x variant) and the script resumes where it stopped.
Metrics are skeleton / arrowhead / tail F1, TPR, FDR against the true PAG, corrected and legacy (bug compatible).
Wall-clock: ``t_optimize`` is the time inside the optimizer, ``t_posterior`` the online posterior inference,
``t_total`` their sum plus thresholding/BIC (``elapsed``).  Offline posterior training is a separate, one-time cost.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgm.abic import run_abic                                         # noqa: E402
from pgm.admg_model import masks_to_admg                              # noqa: E402
from pgm.dcd import DCDConfig, discover_admg                          # noqa: E402
from pgm.graphs import admg_to_pag                                    # noqa: E402
from pgm.metrics import compare_pag, compare_pag_legacy, flat         # noqa: E402
from pgm.spot import CALIBRATED_FLOOR, SpotConfig, allowed_pairs, run_spot              # noqa: E402
from skeleton.posterior import SkeletonPosterior, oracle_posterior, posterior_metrics, random_posterior  # noqa: E402
from skeleton.simulator import mag_skeleton, random_admg, simulate_linear_gaussian  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_VARIANTS = ["abic", "spot", "spot_cal", "spot_nofloor", "spot_floor_only", "spot_random", "spot_oracle", "spot_released"]


def make_graph(seed: int, d: int, n: int, data: str):
    rng = np.random.default_rng(10_000 + seed)
    edge_ratio = float(rng.uniform(1.0, 1.5))
    bi_frac = float(rng.uniform(0.05, 0.15))                      # paper: 5-15 % bidirected
    D, B = random_admg(d, int(round(edge_ratio * d)), bi_frac, rng)
    X, _, _ = simulate_linear_gaussian(D, B, n, rng, standardize=(data == "zscore"))
    names = [f"X{i}" for i in range(d)]
    return D, B, X, names, mag_skeleton(D, B)


def score(admg, true_pag):
    est = admg_to_pag(admg)
    row = {}
    row.update(flat(compare_pag(est, true_pag), "c_"))
    row.update(flat(compare_pag_legacy(est, true_pag), "l_"))
    return row


def run_variant(name, X, names, post_res, skeleton, cfg, rng_seed, budget):
    if budget is not None:
        cfg = DCDConfig(**{**cfg.__dict__, "time_budget": budget})
    t0 = time.perf_counter()
    if name == "abic":
        res = run_abic(X, names, cfg)
        extra = dict(t_posterior=0.0, n_pairs_allowed=-1)
    else:
        P = post_res.P
        sc = SpotConfig(mask_seed=rng_seed)
        if name in ("spot_cal", "spot_floor_only"):
            sc = SpotConfig(mask_seed=rng_seed, floor=CALIBRATED_FLOOR)
        elif name == "spot_nofloor":
            sc = SpotConfig(mask_seed=rng_seed, floor=None)
        elif name == "spot_random":
            sc = SpotConfig(mask_seed=rng_seed, floor=CALIBRATED_FLOOR)
            P = random_posterior(post_res.P, np.random.default_rng(rng_seed))
        elif name == "spot_oracle":
            P = oracle_posterior(skeleton)
        elif name == "spot_released":
            sc = SpotConfig(variant="released_stochastic", mask_seed=rng_seed)
        t_post = post_res.t_total
        if name == "spot_floor_only":
            allowed = allowed_pairs(P, sc)
            res = discover_admg(X, names, cfg, mask=None, allowed=allowed)
            extra = dict(t_posterior=t_post, n_pairs_allowed=int(np.triu(allowed, 1).sum()))
        else:
            # the posterior was already computed once per graph; its time is charged below, not re-measured here
            res = run_spot(X, names, P=P, cfg=cfg, spot_cfg=sc)
            extra = dict(t_posterior=t_post, n_pairs_allowed=res.n_pairs_allowed,
                         mask_kept_frac_by_step={str(k): v["kept_fraction"] for k, v in (res.mask_summary or {}).items()})
    wall = time.perf_counter() - t0
    return res, extra, wall


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=25)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--data", choices=["zscore", "raw"], default="zscore")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=["abic", "spot"], choices=ALL_VARIANTS)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--budget", type=float, default=None, help="equal wall-clock budget (s, all restarts) for every variant")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "skeleton_posterior.pkl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["seed"], r["variant"]))
    post = SkeletonPosterior.load(a.model)
    cfg = DCDConfig(num_restarts=a.restarts, seed=0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for seed in a.seeds:
        todo = [v for v in a.variants if (seed, v) not in done]
        if not todo:
            continue
        D, B, X, names, skel = make_graph(seed, a.d, a.n, a.data)
        Xp = (X - X.mean(0)) / X.std(0)                                # the posterior always sees z-scored data
        post_res = post.predict(Xp)
        pm = posterior_metrics(post_res.P, skel)
        pinned_true = float((np.triu(skel, 1) & ~np.triu(allowed_pairs(post_res.P, SpotConfig(floor=CALIBRATED_FLOOR)), 1)).sum()
                            / max(np.triu(skel, 1).sum(), 1))
        true_pag = admg_to_pag(masks_to_admg(names, D, B))
        for v in todo:
            res, extra, wall = run_variant(v, X, names, post_res, skel, cfg, seed, a.budget)
            row = dict(seed=seed, variant=v, d=a.d, n=a.n, data=a.data, restarts=a.restarts, budget=a.budget, tag=a.tag,
                       n_true_edges=int(np.triu(skel, 1).sum()), n_true_bidirected=int(np.triu(B, 1).sum()),
                       n_est_edges=res.admg.n_edges(), n_est_bidirected=len(res.admg.bi_edges),
                       bic=res.bic, converged=bool(res.converged), timed_out=bool(res.timed_out), h=res.h,
                       n_fev=res.n_fev, n_solves=res.n_solves, n_outer=res.n_outer,
                       t_optimize=res.t_optimize, t_select=res.t_select, elapsed=res.elapsed, wall=wall,
                       t_posterior_features=post_res.t_features, t_posterior_model=post_res.t_model,
                       post_auroc=pm["auroc"], post_auprc=pm["auprc"], floor_pins_true_edge_frac_cal=pinned_true)
            row.update(extra)
            row.update(score(res.admg, true_pag))
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"seed {seed} {v:16s} skel {row['c_skeleton_f1']:.3f} head {row['c_arrowhead_f1']:.3f} "
                  f"tail {row['c_tail_f1']:.3f} | opt {res.t_optimize:6.1f}s fev {res.n_fev:6d} bic {res.bic:9.1f} "
                  f"conv {res.converged}", flush=True)


if __name__ == "__main__":
    main()
