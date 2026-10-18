"""Stage 2d, step 0: is PetShop inside the skeleton posterior's training distribution?

    python scripts/petshop_data_check.py [--root DIR] [--out results/petshop_data_check.json]

For every scenario: rows n, observed d, latent nodes, ground-truth size, per-variable skew, lag-1 autocorrelation
(raw and after Gaussian-rank), and a verdict against the ranges the posterior was trained on
(``results/posterior_training.json``: d 25-40, n 300-2000, AR(1) rho in {0, .3, .6}, edge ratio 1-1.5 per node).
If anything is outside, it prints the exact retraining command (``scripts/train_posterior.py``).
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import skew

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgm.petshop import SCENARIOS, lag1_autocorr, load_petshop, preprocess, skeleton_matrix   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check(scn, root=None, train_cfg=None):
    ps = load_petshop(scn, root)
    X = ps.data
    r_raw, r_gr = lag1_autocorr(X), lag1_autocorr(preprocess(X, "gauss_rank"))
    n, d = X.shape
    S = skeleton_matrix(ps.pag, ps.names)
    n_skel = int(np.triu(S, 1).sum())
    n_di, n_bi = ps.admg.n_edges()[0], len(ps.admg.bi_edges)
    rho_hi = float(np.max(np.abs(r_gr)))
    rho_med = float(np.median(r_gr))
    n_eff = n * (1 - max(rho_med, 0)) / (1 + max(rho_med, 0))
    out = dict(scenario=scn, n=n, d=d, n_latent=len(ps.latents), latents=ps.latents,
               truth=dict(directed=n_di, bidirected=n_bi, pag_edges=len(ps.pag), skeleton_edges=n_skel,
                          edges_per_node=n_skel / d),
               skew_median=float(np.median(skew(X, axis=0))), skew_max=float(np.max(skew(X, axis=0))),
               lag1_raw=dict(median=float(np.median(r_raw)), mean=float(r_raw.mean()), max=float(r_raw.max()), min=float(r_raw.min())),
               lag1_gauss_rank=dict(median=rho_med, mean=float(r_gr.mean()), max=float(r_gr.max()), min=float(r_gr.min())),
               n_eff_from_median_rho=n_eff)
    c = train_cfg or {}
    d_lo, d_hi = c.get("d_min", 25), c.get("d_max", 40)
    n_lo, n_hi = c.get("n_min", 300), c.get("n_max", 2000)
    rho_max_train = max(c.get("ar", [0.0, 0.3, 0.6]))
    issues = []
    if not d_lo <= d <= d_hi:
        issues.append(f"d={d} outside the trained range {d_lo}-{d_hi}")
    if not n_lo <= n <= n_hi:
        issues.append(f"n={n} outside the trained range {n_lo}-{n_hi}")
    if rho_med > rho_max_train + 0.1:
        issues.append(f"median lag-1 autocorrelation {rho_med:.2f} above the trained maximum {rho_max_train}")
    if not 0.8 <= out["truth"]["edges_per_node"] <= 1.7:
        issues.append(f"true skeleton has {out['truth']['edges_per_node']:.2f} edges/node (trained 1.0-1.5, bidirected 5-15 %)")
    out["issues"] = issues
    out["in_distribution"] = not issues
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--train-json", default=os.path.join(ROOT, "results", "posterior_training.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "petshop_data_check.json"))
    a = ap.parse_args(argv)
    train_cfg = json.load(open(a.train_json))["config"] if os.path.exists(a.train_json) else None
    rows = [check(s, a.root, train_cfg) for s in a.scenarios]
    for r in rows:
        t = r["truth"]
        print(f"{r['scenario']:18s} n={r['n']:5d} d={r['d']:2d} latent={r['n_latent']:2d} | truth: {t['directed']} directed, "
              f"{t['bidirected']} bidirected, {t['skeleton_edges']} skeleton edges ({t['edges_per_node']:.2f}/node) | "
              f"lag1 raw median {r['lag1_raw']['median']:.2f}, gauss-rank median {r['lag1_gauss_rank']['median']:.2f} "
              f"(max {r['lag1_gauss_rank']['max']:.2f}) | n_eff~{r['n_eff_from_median_rho']:.0f} | skew median {r['skew_median']:.1f}")
        print("   ->", "inside the posterior's training distribution" if r["in_distribution"] else "OUTSIDE: " + "; ".join(r["issues"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rows, open(a.out, "w"), indent=1)
    bad = [r for r in rows if not r["in_distribution"]]
    if bad:
        n_all = [r["n"] for r in rows]
        d_all = [r["d"] for r in rows]
        rho = max(max(r["lag1_gauss_rank"]["median"], 0.0) for r in rows)
        print("\nSuggested retraining (about 4 min per 400 graphs on one core), then re-run this check:\n"
              f"  python scripts/train_posterior.py --d-min {min(min(d_all), 25)} --d-max {max(max(d_all), 40)} "
              f"--n-min {min(min(n_all), 300)} --n-max {max(max(n_all), 2000)} --ar 0.0 0.3 0.6 {min(round(rho + 0.15, 1), 0.95)}")
        print("(The edge-density mismatch cannot be fixed by --ar/--n flags; it is a limitation to report.)")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
