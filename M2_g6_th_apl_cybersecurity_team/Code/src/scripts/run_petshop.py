"""Stage 2d: ABIC vs SPOT (and ablations) on PetShop.

    python scripts/run_petshop.py --prep gauss_rank --seeds 0 1 2 3 4 --variants abic spot spot_cal --out results/ps_2d_main_gauss_rank.jsonl

One JSONL line per (scenario, preprocessing, seed, variant, budget); resumable (finished lines are skipped).

What is held fixed / what varies
  * The PetShop data of a scenario is ONE dataset, so ``seed`` varies only the optimizer's random initialisation and SPOT's
    mask random stream.  The spread over seeds is optimizer variance, NOT data variance; it is reported as such.
  * Every variant receives identical preprocessed data, DCDConfig (restarts, L-BFGS settings, seed) and, if --budget is given,
    the same total wall-clock budget (SPOT's online posterior time is charged to it, ABIC pays nothing).
  * The skeleton posterior always sees z-scored data (that is what it was trained on), whatever the learner's preprocessing is.
  * The posterior is computed once per (scenario, preprocessing); its time is charged to every SPOT-type variant.
Variants: abic | spot (floor 1e-3, as released) | spot_cal (floor 6e-3) | spot_nofloor | spot_floor_only | spot_random |
          spot_oracle (true skeleton from the ground-truth PAG) | spot_released (other released script).
Metrics: skeleton / arrowhead / tail F1, TPR, FDR against the ground-truth PAG, corrected (c_*) and legacy (l_*).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from pgm.dcd import DCDConfig                                                   # noqa: E402
from pgm.petshop import PREPROCESSINGS, SCENARIOS, load_petshop, lag1_autocorr, preprocess, skeleton_matrix, zscore   # noqa: E402
from pgm.spot import CALIBRATED_FLOOR, SpotConfig, allowed_pairs                # noqa: E402
from skeleton.posterior import SkeletonPosterior, posterior_metrics              # noqa: E402
from spot_vs_abic_synthetic import ALL_VARIANTS, run_variant, score             # noqa: E402  (identical code path to 2c)


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            keys.add((r["scenario"], r["prep"], r["seed"], r["variant"], r.get("budget")))
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="PetShop `dataset` dir (else $PETSHOP_ROOT)")
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    ap.add_argument("--prep", default="gauss_rank", choices=PREPROCESSINGS,
                    help="learner preprocessing: gauss_rank (headline) | none (plain z-score of raw latency) | raw_centered | log")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=["abic", "spot", "spot_cal"], choices=ALL_VARIANTS)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--budget", type=float, default=None, help="equal wall-clock budget in seconds (all restarts) for every variant")
    ap.add_argument("--max-rows", type=int, default=None, help="use only the first N rows (quick tests)")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "skeleton_posterior.pkl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--lbfgs-maxiter", type=int, default=None, help="override DCDConfig.lbfgs_maxiter (tests only)")
    ap.add_argument("--max-outer", type=int, default=None, help="override DCDConfig.max_outer (tests only)")
    a = ap.parse_args(argv)

    done = done_keys(a.out)
    post = SkeletonPosterior.load(a.model)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for scn in a.scenarios:
        todo_any = any((scn, a.prep, s, v, a.budget) not in done for s in a.seeds for v in a.variants)
        if not todo_any:
            continue
        ps = load_petshop(scn, a.root)
        data = ps.data[:a.max_rows] if a.max_rows else ps.data
        X = preprocess(data, a.prep)
        Xp = zscore(X)                                    # the posterior always sees z-scored data
        names, skel = ps.names, skeleton_matrix(ps.pag, ps.names)
        post_res = post.predict(Xp)
        pm = posterior_metrics(post_res.P, skel)
        pinned_true = float((np.triu(skel, 1) & ~np.triu(allowed_pairs(post_res.P, SpotConfig(floor=CALIBRATED_FLOOR)), 1)).sum()
                            / max(np.triu(skel, 1).sum(), 1))
        n_di, n_bi = ps.admg.n_edges()[0], len(ps.admg.bi_edges)
        print(f"[{scn}/{a.prep}] n={X.shape[0]} d={X.shape[1]} latent={len(ps.latents)} truth skeleton edges={int(np.triu(skel, 1).sum())} "
              f"| posterior AUROC {pm['auroc']:.3f} AUPRC {pm['auprc']:.3f} ({post_res.t_total:.2f}s) | cal floor pins "
              f"{100 * pinned_true:.0f}% of true edges", flush=True)
        for seed in a.seeds:
            base = DCDConfig(num_restarts=a.restarts, seed=seed)
            if a.lbfgs_maxiter:
                base = DCDConfig(**{**base.__dict__, "lbfgs_maxiter": a.lbfgs_maxiter})
            if a.max_outer:
                base = DCDConfig(**{**base.__dict__, "max_outer": a.max_outer})
            for v in a.variants:
                if (scn, a.prep, seed, v, a.budget) in done:
                    continue
                res, extra, wall = run_variant(v, X, names, post_res, skel, base, seed, a.budget)
                row = dict(scenario=scn, prep=a.prep, seed=seed, variant=v, n=int(X.shape[0]), d=int(X.shape[1]),
                           n_latent=len(ps.latents), restarts=a.restarts, budget=a.budget, tag=a.tag,
                           n_true_edges=int(np.triu(skel, 1).sum()), n_true_directed=n_di, n_true_bidirected=n_bi,
                           n_est_edges=res.admg.n_edges(), n_est_bidirected=len(res.admg.bi_edges),
                           bic=res.bic, converged=bool(res.converged), timed_out=bool(res.timed_out), h=res.h,
                           n_fev=res.n_fev, n_solves=res.n_solves, n_outer=res.n_outer,
                           t_optimize=res.t_optimize, t_select=res.t_select, elapsed=res.elapsed, wall=wall,
                           t_posterior_features=post_res.t_features, t_posterior_model=post_res.t_model,
                           post_auroc=pm["auroc"], post_auprc=pm["auprc"], floor_pins_true_edge_frac_cal=pinned_true,
                           lag1_median=float(np.median(lag1_autocorr(X))))
                row.update(extra)
                row.update(score(res.admg, ps.pag))
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  seed {seed} {v:16s} skel {row['c_skeleton_f1']:.3f} head {row['c_arrowhead_f1']:.3f} tail {row['c_tail_f1']:.3f} "
                      f"| est {res.admg.n_edges()[0]}+{len(res.admg.bi_edges)}bi | opt {res.t_optimize:6.1f}s fev {res.n_fev:6d} conv {res.converged}",
                      flush=True)


if __name__ == "__main__":
    main()
