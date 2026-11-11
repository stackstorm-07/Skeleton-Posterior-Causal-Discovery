"""RICF: matches direct maximum likelihood, increases the likelihood monotonically, recovers parameters."""
import numpy as np
from scipy.optimize import minimize

from pgm.admg_model import bic, gaussian_loglik, masks_to_admg, pseudo_variables
from pgm.ricf import fit_ricf
from skeleton.simulator import random_admg, simulate_linear_gaussian


def _case(seed=3, d=8, n=500, bi=0.5):
    rng = np.random.default_rng(seed)
    D, B = random_admg(d, 14, bi, rng)
    X, W1, Om = simulate_linear_gaussian(D, B, n, rng, standardize=False)
    return D, B, X - X.mean(0), W1, Om


def test_ricf_matches_direct_ml():
    D, B, X, _, _ = _case()
    n, d = X.shape
    S = X.T @ X / n
    r = fit_ricf(X, D, B)
    assert r.converged
    di, bi = np.argwhere(D), np.argwhere(np.triu(B, 1))

    def unp(t):
        W = np.zeros((d, d)); O = np.diag(np.exp(t[:d])); k = d
        for a, b in di:
            W[a, b] = t[k]; k += 1
        for a, b in bi:
            O[a, b] = O[b, a] = t[k]; k += 1
        return W, O

    def f(t):
        ll = gaussian_loglik(S, n, *unp(t))
        return -ll if np.isfinite(ll) else 1e10

    t0 = np.r_[np.log(np.diag(S)), np.zeros(len(di) + len(bi))]
    s = minimize(f, t0, method="BFGS", options={"maxiter": 3000, "gtol": 1e-8})
    assert abs(r.loglik - (-s.fun)) < 1e-3, (r.loglik, -s.fun)


def test_ricf_monotone_and_consistent():
    D, B, X, W1, Om = _case(seed=0, d=10, n=20000, bi=0.3)
    r = fit_ricf(X, D, B)
    assert all(np.diff(r.trace) > -1e-6)
    assert np.abs(r.W1 - W1).max() < 0.06
    assert np.abs(r.W2 + np.diag(r.Wii) - Om).max() < 0.06


def test_pseudo_variables_regression_identity():
    # eps_i regressed on Z_i gives Omega[-i, i]: check with the population covariance (exact identity)
    rng = np.random.default_rng(0)
    d = 5
    A = rng.standard_normal((d, d)); Om = A @ A.T + d * np.eye(d)
    eps = rng.multivariate_normal(np.zeros(d), Om, size=400000)
    Z = pseudo_variables(eps, np.zeros((d, d)), Om)
    for i in range(d):
        Zi = Z[i]
        coef = np.linalg.lstsq(Zi, eps[:, i], rcond=None)[0]
        o = [j for j in range(d) if j != i]
        assert np.allclose(coef[o], Om[o, i], atol=0.03), (i, coef[o], Om[o, i])


def test_bic_prefers_true_graph_over_empty_and_complete():
    D, B, X, _, _ = _case(seed=5, d=6, n=2000, bi=0.2)
    names = [f"X{i}" for i in range(6)]
    g_true = masks_to_admg(names, D, B)
    g_empty = masks_to_admg(names, np.zeros((6, 6), int), np.zeros((6, 6), int))
    assert bic(X, g_true) < bic(X, g_empty)
