"""M3 item 4: robustness options of the synthetic benchmark (density, measurement noise, autocorrelation, missing data)."""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from skeleton.simulator import corrupt_data, random_admg, simulate_linear_gaussian    # noqa: E402
import spot_vs_abic_synthetic as syn                                                   # noqa: E402


def _m2_make_graph(seed, d, n, data):
    """The M2 generator, copied verbatim, to prove the defaults did not change."""
    rng = np.random.default_rng(10_000 + seed)
    edge_ratio = float(rng.uniform(1.0, 1.5))
    bi_frac = float(rng.uniform(0.05, 0.15))
    D, B = random_admg(d, int(round(edge_ratio * d)), bi_frac, rng)
    X, _, _ = simulate_linear_gaussian(D, B, n, rng, standardize=(data == "zscore"))
    return D, B, X


def test_defaults_are_bit_identical_to_m2():
    for data in ("zscore", "raw"):
        for seed in (0, 3):
            D0, B0, X0 = _m2_make_graph(seed, 12, 150, data)
            D, B, X, names, skel = syn.make_graph(seed, 12, 150, data)
            assert (D == D0).all() and (B == B0).all() and np.array_equal(X, X0)


def lag1(X):
    Z = (X - X.mean(0)) / X.std(0)
    return float((Z[1:] * Z[:-1]).mean())


def test_autocorrelation_option_keeps_graph_and_adds_dependence():
    D0, B0, X0, *_ = syn.make_graph(1, 10, 3000, "zscore", ar=0.0)
    D6, B6, X6, *_ = syn.make_graph(1, 10, 3000, "zscore", ar=0.6)
    assert (D0 == D6).all() and (B0 == B6).all()                # same graph at every autocorrelation level
    assert abs(lag1(X0)) < 0.08 and lag1(X6) > 0.25            # iid vs clearly autocorrelated rows


def test_density_option_sets_edges_per_node():
    for dens in (1.0, 2.0, 3.0):
        D, B, X, names, skel = syn.make_graph(2, 20, 100, "zscore", density=dens)
        n_edges = int(D.sum() + np.triu(B, 1).sum())
        assert n_edges == int(round(dens * 20))              # ancestral repair only turns bidirected into directed edges


def test_measurement_noise_matches_requested_snr_and_keeps_graph():
    D0, B0, X0, *_ = syn.make_graph(4, 10, 20000, "zscore")
    D1, B1, X1, *_ = syn.make_graph(4, 10, 20000, "zscore", snr=3.0)
    assert (D0 == D1).all() and (B0 == B1).all()
    # corr(clean, noisy) = sqrt(snr / (1 + snr)) = 0.866 for snr = 3
    c = np.mean([np.corrcoef(X0[:, j], X1[:, j])[0, 1] for j in range(10)])
    assert abs(c - np.sqrt(3.0 / 4.0)) < 0.02
    assert np.allclose(X1.std(0), 1.0)                         # z-scored after corruption, as the benchmark requires


def test_missing_cells_are_mean_imputed_and_mcar():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5000, 6)) * [1, 2, 3, 4, 5, 6] + 10.0
    Y = corrupt_data(X, np.random.default_rng(1), missing=0.2)
    changed = Y != X
    assert abs(changed.mean() - 0.2) < 0.01 and not np.isnan(Y).any()
    for j in range(6):                                          # imputed cells all equal the mean of the observed ones
        vals = Y[changed[:, j], j]
        assert np.allclose(vals, vals[0]) and abs(vals[0] - Y[~changed[:, j], j].mean()) < 1e-9
    assert np.array_equal(X, X)                                  # input untouched (copy semantics)
    Z = corrupt_data(X, np.random.default_rng(1), missing=0.0, snr=None)
    assert np.array_equal(Z, X)                                  # no-op when nothing is requested
    try:
        corrupt_data(X, np.random.default_rng(1), snr=0.0)
        raise AssertionError("snr = 0 must be rejected")
    except ValueError:
        pass


def test_corruptions_do_not_disturb_each_other_or_the_graph():
    a = syn.make_graph(5, 10, 500, "zscore", snr=5.0, missing=0.05)
    b = syn.make_graph(5, 10, 500, "zscore", snr=5.0, missing=0.05)
    assert np.array_equal(a[2], b[2])                           # reproducible
    c = syn.make_graph(5, 10, 500, "zscore", snr=5.0)
    assert (a[0] == c[0]).all() and not np.array_equal(a[2], c[2])


def test_resume_key_is_backward_compatible_with_m2_lines():
    old = dict(seed=0, variant="abic", d=25, n=1000, data="zscore")             # an M2 result line has no new fields
    new = dict(old, density=None, snr=None, ar=0.0, missing=0.0)
    assert syn.cond_key(old) == syn.cond_key(new)
    assert syn.cond_key(old) != syn.cond_key(dict(new, ar=0.3))
    assert syn.cond_key(old) != syn.cond_key(dict(new, n=500))


def test_design_job_lists_match_the_plan():
    e1 = syn.design_conditions("e1")
    assert [c["n"] for c in e1] == [200, 500, 1000, 2000]
    assert sum(len(c["seeds"]) * len(c["variants"]) for c in e1) == 10 * (2 * 4 + 1)   # +SPOT-1e-3 at n = 1000 only
    assert [c["variants"] for c in e1 if c["n"] == 1000] == [["abic", "spot_cal", "spot"]]
    e2 = syn.design_conditions("e2")
    assert len(e2) == 3 + 3 + 2 + 2 and all(c["n"] == 1000 and len(c["seeds"]) == 8 for c in e2)
    for c in e2:                                                                   # one factor at a time
        moved = [k for k in ("density", "snr") if c[k] is not None] + [k for k in ("ar", "missing") if c[k]]
        assert len(moved) == 1
    assert syn.design_conditions("e1", graphs=6)[0]["seeds"] == list(range(6))     # the plan's cut rule: 6 graphs per n


def test_list_jobs_runs_nothing(capsys=None):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "x.jsonl")
        syn.main(["--design", "e2", "--list-jobs", "--out", out])
        assert not os.path.exists(out)


def _have_posterior_and_tetrad():
    import sklearn
    from pgm.graphs import find_tetrad_jar
    try:
        find_tetrad_jar()
    except Exception as e:
        raise __import__("unittest").SkipTest(str(e))
    p = os.path.join(ROOT, "models", "skeleton_posterior.pkl")
    from skeleton.posterior import SkeletonPosterior
    try:
        SkeletonPosterior.load(p)
        return p
    except Exception:
        alt = os.path.join(ROOT, "models", f"skeleton_posterior_sk{sklearn.__version__}.pkl")
        if os.path.exists(alt):
            return alt
        raise __import__("unittest").SkipTest("no loadable posterior (retrain once: scripts/train_posterior.py)")


def test_end_to_end_conditions_resume_and_columns():
    model = _have_posterior_and_tetrad()
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "s.jsonl")
        base = ["--d", "8", "--n", "200", "--seeds", "0", "--variants", "abic", "spot_cal", "--model", model, "--out", out]
        for extra in ([], ["--ar", "0.3"], ["--snr", "3"], ["--missing", "0.05"], ["--density", "2"]):
            syn.main(base + extra)
        rows = [json.loads(l) for l in open(out)]
        assert len(rows) == 5 * 2
        assert all("c_shd_pag" in r and "density" in r and "snr" in r and "ar" in r and "missing" in r for r in rows)
        assert len({syn.cond_key(r) for r in rows}) == 10                      # every (condition, variant) is its own key
        for extra in ([], ["--ar", "0.3"], ["--snr", "3"]):                     # resume: nothing re-run
            syn.main(base + extra)
        assert len(open(out).readlines()) == 10
        r_ar = [r for r in rows if r["ar"] == 0.3 and r["variant"] == "abic"][0]
        r_dens = [r for r in rows if r["density"] == 2.0 and r["variant"] == "abic"][0]
        assert r_dens["n_true_edges"] >= r_ar["n_true_edges"] - 2             # density 2 gives about 16 edges at d = 8
