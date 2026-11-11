"""Inference layer: Gaussian conditioning, marginals, do-queries (checked against Monte Carlo of the SCM)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pgm.inference import ADMGInference, predictive_scores   # noqa: E402
from skeleton.simulator import random_admg, simulate_linear_gaussian   # noqa: E402


def _model(seed=0, d=7, n=60000):
    rng = np.random.default_rng(seed)
    D, B = random_admg(d, 9, 0.4, rng)
    X, W1, Om = simulate_linear_gaussian(D, B, n, rng, standardize=False)
    names = [f"v{i}" for i in range(d)]
    return rng, D, B, X, W1, Om, names


def test_joint_matches_sample_covariance():
    rng, D, B, X, W1, Om, names = _model(0)
    inf = ADMGInference(names, W1, Om)
    assert np.abs(inf.Sigma - np.cov(X.T, bias=True)).max() < 0.06 * max(1.0, np.abs(inf.Sigma).max())


def test_condition_equals_regression_on_samples():
    rng, D, B, X, W1, Om, names = _model(1)
    inf = ADMGInference(names, W1, Om)
    ev = [0, 1, 2]
    law = inf.condition([5], {i: float(X[10, i]) for i in ev})
    # OLS of x5 on x0..x2 (population) from the large sample
    A = np.c_[X[:, ev], np.ones(len(X))]
    beta = np.linalg.lstsq(A, X[:, 5], rcond=None)[0]
    pred = np.r_[X[10, ev], 1.0] @ beta
    assert abs(law.mean[0] - pred) < 0.05
    resid_var = np.var(X[:, 5] - A @ beta)
    assert abs(law.cov[0, 0] - resid_var) < 0.05 * max(1.0, resid_var)


def test_condition_rejects_overlap_and_handles_empty_evidence():
    rng, D, B, X, W1, Om, names = _model(2, n=500)
    inf = ADMGInference(names, W1, Om)
    try:
        inf.condition([0], {0: 1.0})
        raise AssertionError("should have raised")
    except ValueError:
        pass
    law = inf.condition([0, 1], {})
    assert np.allclose(law.cov, inf.Sigma[:2, :2])


def test_do_chain_effect_is_product_of_coefficients():
    names = ["a", "b", "c"]
    W1 = np.zeros((3, 3)); W1[0, 1] = 2.0; W1[1, 2] = -0.5      # a -> b -> c
    Om = np.eye(3)
    inf = ADMGInference(names, W1, Om)
    assert abs(inf.total_effect("a", "c") - (-1.0)) < 1e-12
    law = inf.do({"a": 3.0})
    assert np.allclose(law.mean, [3.0, 6.0, -3.0])
    assert np.allclose(np.diag(law.cov), [0.0, 1.0, 1.0 + 0.25], atol=1e-6)


def test_do_cuts_confounding_unlike_conditioning():
    # x <- (shared noise) -> y with NO directed edge: conditioning on x moves y, intervening on x does not
    names = ["x", "y"]
    W1 = np.zeros((2, 2))
    Om = np.array([[1.0, 0.8], [0.8, 1.0]])
    inf = ADMGInference(names, W1, Om)
    assert abs(inf.condition(["y"], {"x": 2.0}).mean[0] - 1.6) < 1e-6
    assert abs(inf.do({"x": 2.0}).mean[1]) < 1e-9
    assert inf.total_effect("x", "y") == 0.0


def test_do_matches_monte_carlo_of_scm():
    rng, D, B, X, W1, Om, names = _model(3, d=6, n=10)
    d = 6
    inf = ADMGInference(names, W1, Om)
    i, a = 1, 2.5
    # simulate the intervened SCM directly: eps ~ N(0, Om) (cut row/col i), x_i = a
    Om2 = Om.copy(); Om2[i, :] = 0; Om2[:, i] = 0
    W2 = W1.copy(); W2[:, i] = 0
    eps = rng.multivariate_normal(np.zeros(d), Om2, size=200000)
    eps[:, i] = a
    Xi = eps @ np.linalg.inv(np.eye(d) - W2)
    law = inf.do({i: a})
    assert np.abs(Xi.mean(0) - law.mean).max() < 0.02
    assert np.abs(np.cov(Xi.T) - law.cov).max() < 0.03


def test_predictive_scores_beat_marginal_on_dependent_data():
    rng, D, B, X, W1, Om, names = _model(4, n=4000)
    inf = ADMGInference.fit(X[:3000], names, D, B)
    t = int(np.argmax(D.sum(0)))            # a node with parents
    pa = list(np.flatnonzero(D[:, t]))
    s_graph = predictive_scores(inf, X[3000:], t, pa)
    s_marg = predictive_scores(inf, X[3000:], t, [])
    assert s_graph["logpdf"] > s_marg["logpdf"]
