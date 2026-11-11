"""M4 item 2: predictive calibration scores (coverage, PIT, z^2, CRPS), Brier / CI columns in the aggregators."""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)
from pgm.inference import ADMGInference, predictive_scores                       # noqa: E402
from skeleton.simulator import random_admg, simulate_linear_gaussian              # noqa: E402


def _fit(seed=4, n=40000, ntr=20000):
    rng = np.random.default_rng(seed)
    D, B = random_admg(7, 9, 0.4, rng)
    X, _, _ = simulate_linear_gaussian(D, B, n, rng, standardize=False)
    names = [f"v{i}" for i in range(7)]
    inf = ADMGInference.fit(X[:ntr], names, D, B)
    t = int(np.argmax(D.sum(0)))
    return inf, X[ntr:], t, list(np.flatnonzero(D[:, t]))


def test_a_calibrated_model_is_calibrated_and_an_overconfident_one_is_caught():
    inf, Xte, t, pa = _fit()
    s = predictive_scores(inf, Xte, t, pa)
    assert abs(s["coverage90"] - 0.90) < 0.012 and abs(s["coverage50"] - 0.50) < 0.015
    assert abs(s["z2"] - 1.0) < 0.05 and s["pit_ks"] < 0.015
    # the same conditional mean with half the variance: over-confident
    coef, var = inf.conditional_map(t, inf._ix(pa))
    orig = inf.conditional_map
    inf.conditional_map = lambda tt, ee: (coef, var / 2)
    bad = predictive_scores(inf, Xte, t, pa)
    inf.conditional_map = orig
    assert bad["coverage90"] < s["coverage90"] - 0.07 and bad["z2"] > 1.8 and bad["pit_ks"] > 0.05
    assert bad["logpdf"] < s["logpdf"] and bad["rmse"] == s["rmse"]        # worse NLL, identical mean prediction
    # and under-confident (double variance): coverage above nominal, z2 < 1
    inf.conditional_map = lambda tt, ee: (coef, var * 2)
    wide = predictive_scores(inf, Xte, t, pa)
    inf.conditional_map = orig
    assert wide["coverage90"] > 0.95 and wide["z2"] < 0.6


def test_gaussian_crps_matches_numerical_integration():
    from scipy import stats
    inf, Xte, t, pa = _fit(n=4000, ntr=3000)
    sc = predictive_scores(inf, Xte[:5], t, pa)
    coef, var = inf.conditional_map(t, inf._ix(pa))
    e = inf._ix(pa)
    mu = inf.mean[t] + (Xte[:5][:, e] - inf.mean[e]) @ coef
    sd = np.sqrt(max(var, 1e-9))
    grid = np.linspace(-12, 12, 40001)
    num = []
    for m, y in zip(mu, Xte[:5, t]):
        F = stats.norm.cdf(grid, m, sd)
        num.append(np.trapezoid((F - (grid >= y)) ** 2, grid))
    assert abs(np.mean(num) - sc["crps"]) < 2e-3


def test_rolling_rows_carry_the_calibration_columns():
    import validate_inference as vi
    rng = np.random.default_rng(0)
    D, B = random_admg(6, 7, 0.3, rng)
    X, _, _ = simulate_linear_gaussian(D, B, 600, rng, standardize=False)
    X = X - X.min() + 1.0                                                    # positive: the "log" transform needs it
    names = [f"v{i}" for i in range(6)]
    rows = vi.rolling_rows(X, names, {"truth": (D, B)}, {"truth": 0.0}, int(np.argmax(D.sum(0))))
    assert rows and all(k in rows[0] for k in ("logpdf", "rmse", "coverage90", "coverage50", "z2", "pit_ks", "crps"))
    assert {r["query"] for r in rows} == {"A", "B"} and all(0 <= r["coverage90"] <= 1 for r in rows)


def test_aggregate_tables_show_brier_ci_and_coverage():
    import aggregate_m3 as am
    with tempfile.TemporaryDirectory() as tmp:
        res = os.path.join(tmp, "res")
        os.makedirs(res)
        rng = np.random.default_rng(1)
        rows = []
        for sc in ("low_traffic", "high_traffic"):
            for gs in range(3):
                for sp in range(5):
                    for m, cov in (("saturated", 0.85), ("truth/admg", 0.90)):
                        rows.append(dict(scenario=sc, graph_seed=gs, split=sp, model=m, source="none", structure=m.split("/")[-1], query="B",
                                         logpdf=-1.0 + 0.1 * rng.normal(), rmse=0.5, cond_sd=0.5, coverage90=cov + 0.02 * rng.normal(),
                                         coverage50=0.5, z2=1.0, pit_ks=0.03, crps=0.3, t_fit=0.01, t_graph_learn=0.0))
        with open(os.path.join(res, "inference_m3.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        base = dict(dataset="simulated", calibrator="none", n_graphs=5, n_pairs=100, n_true_edges=10, auroc=.9, auprc=.8, log_loss=.2, brier=.05, ece=.01,
                    overconfidence=0.0, reliability=dict(mean_pred=[.1, .9], frac_true=[.1, .9], count=[50, 50], ece=.01), ci95=dict(log_loss=[.1, .3], brier=[.04, .06], ece=[0, .02], overconfidence=[-.01, .01]),
                    ci_unit="graph")
        with open(os.path.join(res, "post_quality_m3.jsonl"), "w") as f:
            f.write(json.dumps(base) + "\n")
        agg = am.Agg(res, os.path.join(res, "aggregate"))
        agg.sec_E8()
        agg.sec_E7()
        t5 = agg.tables["T5_inference"][0]
        assert {"coverage_90", "NLL_CI95", "CRPS"} <= set(t5.columns)
        t7 = agg.tables["T7_posterior_quality"][0]
        assert t7["Brier"].iloc[0] == "0.0500" and t7["Brier_CI"].iloc[0] == "[0.0400, 0.0600]"


def test_aggregate_quick_paired_tables_have_working_ci_columns():
    import pandas as pd
    import aggregate_quick as aq
    from scipy import stats
    rng = np.random.default_rng(0)
    rows = []
    for sc in ("a", "b"):
        for seed in range(8):
            base = rng.normal(0.5, 0.05)
            rows.append(dict(scenario=sc, seed=seed, variant="abic", c_skeleton_f1=base, c_shd_pag=40.0, t_optimize=10.0))
            rows.append(dict(scenario=sc, seed=seed, variant="spot_cal", c_skeleton_f1=base + 0.05 + 0.01 * rng.normal(), c_shd_pag=38.0, t_optimize=5.0))
    df = pd.DataFrame(rows)
    out = aq.paired(df, ["scenario", "seed"], "abic", "spot_cal", ["c_skeleton_f1"])
    r = out.iloc[0]
    d = df[df.variant == "spot_cal"].c_skeleton_f1.values - df[df.variant == "abic"].c_skeleton_f1.values
    h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    assert abs(r["ci95_lo"] - (d.mean() - h)) < 1e-9 and abs(r["ci95_hi"] - (d.mean() + h)) < 1e-9
    assert r["ci_excludes_0"] and r["decision"] == "supported"
    ms = aq.msd(df, ["variant"], ["c_skeleton_f1"])
    assert "skel F1 CI95" in ms.columns and ms["skel F1 CI95"].iloc[0].startswith("[")
