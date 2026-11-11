"""E7: probability quality of the skeleton posterior.  Headline metric: log-loss; companion: reliability diagram + ECE.

    python scripts/posterior_quality.py --out results/post_quality.jsonl                                  # simulated only
    PETSHOP_ROOT=... python scripts/posterior_quality.py --petshop --out results/post_quality.jsonl      # + the PetShop scenarios

Datasets
  simulated   60 fresh graphs (seed 4242, d 25-40, AR 0 / 0.3 / 0.6), disjoint from the training graphs (seed 0) and the floor
              calibration graphs (seed 777).  Pooled into one row.
  petshop     one row per scenario: posterior on the gauss-rank preprocessed series (same input as the learners), labels = true skeleton
              of the ground-truth PAG.
Calibrators (E7 "effect of calibration"): ``none``, ``isotonic`` and ``platt`` (logistic regression on logit p), both fitted on 40
SIMULATED graphs (seed 9001) only, then applied unchanged to every dataset: nothing is fitted on PetShop labels.
One JSON row per (dataset, calibrator): AUROC, AUPRC, log-loss, Brier, ECE, reliability bins, and the share of true edges whose p is
below each hard floor (1e-3 and the calibrated 6e-3) -- the "fraction of true edges below floor" of the plan.
Only a per-dataset summary is stored.  Brier is reported next to log-loss (M4: the guidelines name NLL, log-loss, Brier and calibration), with a bootstrap CI.
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
from skeleton.posterior import default_model_path, SkeletonPosterior, pool_reliability                          # noqa: E402
from skeleton.simulator import make_instances                                            # noqa: E402

CALIBRATORS = ("none", "isotonic", "platt")
FLOORS = (1e-3, 6e-3)


def _upper(M):
    return M[np.triu_indices(M.shape[0], 1)]


def fit_calibrators(post, n=40, seed=9001):
    """Isotonic and Platt maps fitted on simulated graphs only.  Returns {name: f(p) -> p'}."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    ps, ys = [], []
    for it in make_instances(n, d_range=(25, 40), ar_rhos=(0.0, 0.3, 0.6), seed=seed):
        P = post.predict(it.X).P
        ps.append(np.clip(_upper(P), 1e-6, 1 - 1e-6)); ys.append(_upper(it.skeleton.astype(float)))
    p, y = np.concatenate(ps), np.concatenate(ys)
    iso = IsotonicRegression(y_min=1e-6, y_max=1 - 1e-6, out_of_bounds="clip").fit(p, y)
    logit = lambda q: np.log(q / (1 - q))
    pl = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
    return {"none": lambda q: q,
            "isotonic": lambda q: iso.predict(q),
            "platt": lambda q: pl.predict_proba(logit(np.clip(q, 1e-6, 1 - 1e-6))[:, None])[:, 1]}


def _ece(p, y, n_bins=10):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)) - 1, 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(n_bins) if (idx == b).any()))


def overconfidence(rel):
    """Signed calibration gap in the direction of extremity, averaged over pairs: for bins with predicted p >= 0.5 it is (predicted - observed
    fraction of edges), for bins below 0.5 it is (observed - predicted).  > 0: the posterior is MORE extreme than the data justify
    (over-confident); < 0: under-confident; 0: calibrated."""
    n = max(sum(rel["count"]), 1)
    g = 0.0
    for c, mp, ft in zip(rel["count"], rel["mean_pred"], rel["frac_true"]):
        if c:
            g += c / n * ((mp - ft) if mp >= 0.5 else (ft - mp))
    return float(g)


def _boot(groups, B, seed=0):
    """Bootstrap CIs (2.5 / 97.5 %) of log-loss, Brier, ECE and over-confidence.  ``groups`` = list of (p, y), one per graph: whole graphs are
    resampled; with a single graph (PetShop) the pairs are resampled instead (pairs are then treated as independent: optimistic)."""
    rng = np.random.default_rng(seed)
    out = dict(log_loss=[], brier=[], ece=[], overconfidence=[])
    for _ in range(B):
        if len(groups) > 1:
            pick = rng.integers(0, len(groups), len(groups))
            p = np.concatenate([groups[i][0] for i in pick]); y = np.concatenate([groups[i][1] for i in pick])
        else:
            p0, y0 = groups[0]
            i = rng.integers(0, len(p0), len(p0)); p, y = p0[i], y0[i]
        out["log_loss"].append(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))))
        out["brier"].append(float(np.mean((p - y) ** 2)))
        out["ece"].append(_ece(p, y))
        out["overconfidence"].append(overconfidence(pool_reliability([(p, y)])))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}


def summarize_pairs(pairs, label, cal_name, n_graphs, boot=300):
    """pairs: list of (p, y), one entry per graph, pooled into one row (CIs by resampling graphs; see _boot)."""
    pairs = [(np.clip(a, 1e-6, 1 - 1e-6), b) for a, b in pairs]
    p = np.concatenate([a for a, _ in pairs]); y = np.concatenate([b for _, b in pairs])
    from sklearn.metrics import average_precision_score, roc_auc_score
    rel = pool_reliability([(p, y)])
    row = dict(dataset=label, calibrator=cal_name, n_graphs=n_graphs, n_pairs=int(len(y)), n_true_edges=int(y.sum()),
               auroc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
               auprc=float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
               log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), brier=float(np.mean((p - y) ** 2)),
               ece=rel["ece"], overconfidence=overconfidence(rel), reliability=rel, mean_pred=float(p.mean()), base_rate=float(y.mean()))
    if boot:
        row["ci95"] = _boot(pairs, boot)
        row["ci_unit"] = "graphs" if len(pairs) > 1 else "pairs (treated as independent)"
    for fl in FLOORS:
        row[f"true_edges_below_{fl:g}"] = float(np.mean(p[y > 0] <= fl)) if y.sum() else float("nan")
    return row


def simulated_rows(post, cals, n=60, seed=4242):
    raw = []
    for it in make_instances(n, d_range=(25, 40), ar_rhos=(0.0, 0.3, 0.6), seed=seed):
        raw.append((np.clip(_upper(post.predict(it.X).P), 1e-6, 1 - 1e-6), _upper(it.skeleton.astype(float))))
    return [summarize_pairs([(cals[c](p), y) for p, y in raw], "simulated", c, n) for c in CALIBRATORS]


def petshop_rows(post, cals, scenario, root, prep="gauss_rank", max_rows=None):
    from pgm.petshop import load_petshop, preprocess, skeleton_matrix, zscore
    ps = load_petshop(scenario, root)
    data = ps.data[:max_rows] if max_rows else ps.data
    P = post.predict(zscore(preprocess(data, prep))).P
    p = np.clip(_upper(P), 1e-6, 1 - 1e-6)
    y = _upper(skeleton_matrix(ps.pag, ps.names).astype(float))
    return [dict(summarize_pairs([(cals[c](p), y)], scenario, c, 1), prep=prep, n=int(len(data)), d=len(ps.names)) for c in CALIBRATORS]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--petshop", action="store_true")
    ap.add_argument("--root", default=None)
    ap.add_argument("--scenarios", nargs="+", default=None)
    ap.add_argument("--n-sim", type=int, default=60)
    ap.add_argument("--n-cal", type=int, default=40)
    ap.add_argument("--model", default=default_model_path(ROOT))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    post = SkeletonPosterior.load_or_retrain(a.model)
    cals = fit_calibrators(post, a.n_cal)
    rows = simulated_rows(post, cals, a.n_sim)
    if a.petshop:
        from pgm.petshop import SCENARIOS, default_root
        for scn in (a.scenarios or SCENARIOS):
            rows += petshop_rows(post, cals, scn, a.root or default_root())
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for r in rows:
        print(f"{r['dataset']:18s} {r['calibrator']:9s} log-loss {r['log_loss']:.4f} ECE {r['ece']:.4f} AUROC {r['auroc']:.3f} "
              f"true edges below 6e-3: {r['true_edges_below_0.006']:.3f}")


if __name__ == "__main__":
    main()
