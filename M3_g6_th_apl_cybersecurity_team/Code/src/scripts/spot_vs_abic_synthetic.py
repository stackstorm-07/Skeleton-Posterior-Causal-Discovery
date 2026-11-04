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
  <variant>_dir   (M3, E10) any of the above with the bidirected block forced to 0: the same optimizer learning a DAG

M3 robustness options (E1/E2 of the plan), all applied to the *observed* data only, the ground truth is the clean graph:
  --density E   exactly E edges per node (default: the M2 draw, uniform 1.0-1.5)
  --snr S       measurement noise, variance = var(column) / S (default: none)
  --ar RHO      AR(1) autocorrelated structural errors (default: 0)
  --missing P   fraction of cells MCAR-removed and mean-imputed (default: 0)
  --design e1|e2  the plan's job lists (see ``design_conditions``); every option may also be given by hand.
A condition changes only what the corruption/option says: for a given seed the base graph and the SEM parameters are the
same at every setting (the density option changes the graph by definition), and the default call is bit-identical to M2.

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
from pgm.metrics import compare_pag, compare_pag_legacy, flat, shd_flat        # noqa: E402
from pgm.spot import CALIBRATED_FLOOR, SpotConfig, allowed_pairs, run_spot              # noqa: E402
from skeleton.posterior import SkeletonPosterior, oracle_posterior, posterior_metrics, random_posterior  # noqa: E402
from skeleton.simulator import corrupt_data, mag_skeleton, random_admg, simulate_linear_gaussian  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_VARIANTS = ["abic", "spot", "spot_cal", "spot_nofloor", "spot_floor_only", "spot_random", "spot_oracle", "spot_released"]
# "<variant>_dir" (E10): the same variant with the bidirected block forced to 0 (DCDConfig.directed_only), everything else identical
ALL_VARIANTS = BASE_VARIANTS + [v + "_dir" for v in BASE_VARIANTS]


def make_graph(seed: int, d: int, n: int, data: str, density=None, snr=None, ar=0.0, missing=0.0):
    """One synthetic problem.  Defaults reproduce the M2 generator exactly (same rng stream, same numbers).

    The graph / SEM stream is ``default_rng(10_000 + seed)`` as before; the edge ratio and bidirected share are always
    drawn (so the stream is consumed identically) and ``density`` then overrides the edge ratio.  Corruptions use their
    own streams (20_000 + seed for noise and 30_000 + seed for missingness), so they never disturb the graph."""
    rng = np.random.default_rng(10_000 + seed)
    edge_ratio = float(rng.uniform(1.0, 1.5))
    bi_frac = float(rng.uniform(0.05, 0.15))                      # paper: 5-15 % bidirected
    if density is not None:
        edge_ratio = float(density)
    D, B = random_admg(d, int(round(edge_ratio * d)), bi_frac, rng)
    corrupted = snr is not None or missing > 0.0
    if not corrupted:
        X, _, _ = simulate_linear_gaussian(D, B, n, rng, standardize=(data == "zscore"), ar_rho=ar)
    else:
        X, _, _ = simulate_linear_gaussian(D, B, n, rng, standardize=False, ar_rho=ar)
        if snr is not None:
            X = corrupt_data(X, np.random.default_rng(20_000 + seed), snr=snr)
        if missing > 0.0:
            X = corrupt_data(X, np.random.default_rng(30_000 + seed), missing=missing)
        X = X - X.mean(axis=0)
        if data == "zscore":
            X = X / X.std(axis=0)
    names = [f"X{i}" for i in range(d)]
    return D, B, X, names, mag_skeleton(D, B)


def score(admg, true_pag):
    return score_pag(admg_to_pag(admg), true_pag)


def score_pag(est, true_pag):
    """Metric columns for an estimated PAG (shared by the ADMG learners and the FCI/RFCI baselines)."""
    row = {}
    row.update(flat(compare_pag(est, true_pag), "c_"))
    row.update(flat(compare_pag_legacy(est, true_pag), "l_"))
    row.update(shd_flat(est, true_pag, "c_"))              # structural Hamming distances (M3), corrected convention only
    return row


def run_variant(name, X, names, post_res, skeleton, cfg, rng_seed, budget):
    if name.endswith("_dir"):                                       # E10: directed-only twin of any variant
        name = name[:-4]
        cfg = DCDConfig(**{**cfg.__dict__, "directed_only": True})
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


COND_KEYS = ("d", "n", "data", "density", "snr", "ar", "missing")


def cond_key(r: dict, seed=None, variant=None):
    """Resume key of a result line: graph seed, variant and the whole condition.  Lines written before M3 lack the new
    fields; their defaults (None / 0) are the M2 setting, so old files resume exactly as before."""
    return (r["seed"] if seed is None else seed, r["variant"] if variant is None else variant,
            r["d"], r["n"], r["data"], r.get("density"), r.get("snr"), float(r.get("ar") or 0.0), float(r.get("missing") or 0.0))


# E2 levels (plan section 3).  The SNR levels are our choice (the plan only says "SNR levels").
E1_NS = (200, 500, 1000, 2000)
E2_LEVELS = dict(density=(1.0, 2.0, 3.0), snr=(10.0, 3.0, 1.0), ar=(0.3, 0.6), missing=(0.05, 0.20))


def design_conditions(design: str, d: int = 25, graphs=None, data: str = "zscore"):
    """Job lists of the plan.  Each entry: dict(seeds, variants, d, n, data, density, snr, ar, missing).

    e1: n in {200, 500, 1000, 2000}, ``graphs`` (default 10) graphs per n, ABIC vs SPOT-cal, plus SPOT-1e-3 at n = 1000.
    e2: n = 1000, ``graphs`` (default 8) graphs per cell, ABIC vs SPOT-cal, one factor at a time: density 1/2/3 edges per
        node, SNR 10/3/1, AR 0.3/0.6, missing 5/20 %.  The default cell (no corruption, M2 density draw, AR 0) is
        written as part of E1 (n = 1000) and is not repeated here; the AR 0 level of the AR curve is that same cell.
    Seeds are 0..graphs-1 in both, so a graph at n = 1000 in E1 is the same graph as in every E2 cell (except density,
    which redraws the edge count by definition)."""
    out = []
    if design == "e1":
        seeds = list(range(10 if graphs is None else graphs))
        for n in E1_NS:
            variants = ["abic", "spot_cal"] + (["spot"] if n == 1000 else [])
            out.append(dict(seeds=seeds, variants=variants, d=d, n=n, data=data, density=None, snr=None, ar=0.0, missing=0.0))
    elif design == "e2":
        seeds = list(range(8 if graphs is None else graphs))
        for factor, levels in E2_LEVELS.items():
            for lv in levels:
                c = dict(seeds=seeds, variants=["abic", "spot_cal"], d=d, n=1000, data=data,
                         density=None, snr=None, ar=0.0, missing=0.0)
                c[factor] = lv
                out.append(c)
    else:
        raise ValueError(design)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=25)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--data", choices=["zscore", "raw"], default="zscore")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=["abic", "spot"], choices=ALL_VARIANTS)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--budget", type=float, default=None, help="equal wall-clock budget (s, all restarts) for every variant")
    ap.add_argument("--density", type=float, default=None, help="edges per node (default: M2 draw, uniform 1.0-1.5)")
    ap.add_argument("--snr", type=float, default=None, help="measurement-noise SNR (variance ratio); default none")
    ap.add_argument("--ar", type=float, default=0.0, help="AR(1) coefficient of the structural errors (0 = iid)")
    ap.add_argument("--missing", type=float, default=0.0, help="MCAR fraction of cells, mean-imputed")
    ap.add_argument("--design", choices=["custom", "e1", "e2"], default="custom",
                    help="the plan's job list (overrides --n/--seeds/--variants/--density/--snr/--ar/--missing); "
                         "--graphs sets the number of graphs per cell, --d the size")
    ap.add_argument("--graphs", type=int, default=None, help="graphs per cell for --design (default: e1 10, e2 8)")
    ap.add_argument("--list-jobs", action="store_true", help="print the conditions and the number of jobs, run nothing")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "skeleton_posterior.pkl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    if a.design == "custom":
        conds = [dict(seeds=a.seeds, variants=a.variants, d=a.d, n=a.n, data=a.data,
                      density=a.density, snr=a.snr, ar=a.ar, missing=a.missing)]
    else:
        conds = design_conditions(a.design, d=a.d, graphs=a.graphs, data=a.data)
    if a.list_jobs:
        for c in conds:
            print({k: v for k, v in c.items() if k != "seeds"}, "graphs:", len(c["seeds"]))
        print("jobs:", sum(len(c["seeds"]) * len(c["variants"]) for c in conds))
        return

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            done.add(cond_key(json.loads(line)))
    post = SkeletonPosterior.load_or_retrain(a.model)
    cfg = DCDConfig(num_restarts=a.restarts, seed=0)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for c in conds:
        for seed in c["seeds"]:
            key = lambda v: (seed, v, c["d"], c["n"], c["data"], c["density"], c["snr"], float(c["ar"]), float(c["missing"]))
            todo = [v for v in c["variants"] if key(v) not in done]
            if not todo:
                continue
            for row in run_graph_job(c, seed, todo, post, cfg, a.budget, a.restarts, a.tag, a.design):
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                cond = f"n={c['n']} dens={c['density']} snr={c['snr']} ar={c['ar']} mis={c['missing']}"
                print(f"seed {seed} {row['variant']:16s} [{cond}] skel {row['c_skeleton_f1']:.3f} head {row['c_arrowhead_f1']:.3f} "
                      f"tail {row['c_tail_f1']:.3f} shd {row.get('c_shd_pag', float('nan'))} | opt {row['t_optimize']:6.1f}s "
                      f"fev {row['n_fev']:6d} bic {row['bic']:9.1f} conv {row['converged']}", flush=True)


def run_graph_job(c, seed, variants, post, cfg, budget, restarts, tag="", design="custom"):
    """One synthetic graph (condition ``c``, graph ``seed``): data, posterior once, then every requested variant on the SAME data.
    Returns the result rows (one per variant).  Used by ``main`` and by the parallel queue (scripts/run_queue_parallel.py)."""
    D, B, X, names, skel = make_graph(seed, c["d"], c["n"], c["data"], density=c["density"], snr=c["snr"],
                                      ar=c["ar"], missing=c["missing"])
    Xp = (X - X.mean(0)) / X.std(0)                                    # the posterior always sees z-scored data
    post_res = post.predict(Xp)
    pm = posterior_metrics(post_res.P, skel)
    pinned_true = float((np.triu(skel, 1) & ~np.triu(allowed_pairs(post_res.P, SpotConfig(floor=CALIBRATED_FLOOR)), 1)).sum()
                        / max(np.triu(skel, 1).sum(), 1))
    true_pag = admg_to_pag(masks_to_admg(names, D, B))
    rows = []
    for v in variants:
        res, extra, wall = run_variant(v, X, names, post_res, skel, cfg, seed, budget)
        row = dict(seed=seed, variant=v, d=c["d"], n=c["n"], data=c["data"], restarts=restarts, budget=budget, tag=tag,
                   density=c["density"], snr=c["snr"], ar=c["ar"], missing=c["missing"], design=design,
                   n_true_edges=int(np.triu(skel, 1).sum()), n_true_bidirected=int(np.triu(B, 1).sum()),
                   n_est_edges=res.admg.n_edges(), n_est_bidirected=len(res.admg.bi_edges),
                   bic=res.bic, converged=bool(res.converged), timed_out=bool(res.timed_out), h=res.h,
                   n_fev=res.n_fev, n_solves=res.n_solves, n_outer=res.n_outer,
                   t_optimize=res.t_optimize, t_select=res.t_select, elapsed=res.elapsed, wall=wall,
                   t_posterior_features=post_res.t_features, t_posterior_model=post_res.t_model,
                   post_auroc=pm["auroc"], post_auprc=pm["auprc"], floor_pins_true_edge_frac_cal=pinned_true)
        row.update(extra)
        row.update(score(res.admg, true_pag))
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
