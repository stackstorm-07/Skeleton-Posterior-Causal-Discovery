"""Analytic vs central-finite-difference gradients of every differentiable piece in pgm/dcd.py.

    python scripts/gradcheck.py            # prints the errors, exit code 1 if any exceeds the tolerance

Checks, for both structure penalties ('released' = code of the paper, 'expm' = paper Eq. 4):
  * h(W1, W2) and its gradients w.r.t. W1 and W2,
  * the full primal loss (data fit + augmented Lagrangian + smooth L0) over the free coordinates,
    through the fast Gram-matrix path *and* the direct pseudo-variable path, which must agree,
  * that pinned coordinates (outside the allowed skeleton) carry exactly zero gradient.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgm.admg_model import pseudo_variables, unpack  # noqa: E402
from pgm.dcd import PENALTIES, DCDConfig, Problem  # noqa: E402


def _rel(a, b):
    return np.abs(a - b) / np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))


def check_structure_penalty(name="released", d=6, seed=0, eps=1e-6):
    fn = PENALTIES[name]
    rng = np.random.default_rng(seed)
    W1 = rng.uniform(-1, 1, (d, d))
    W2 = rng.uniform(-1, 1, (d, d))
    h, g1, g2 = fn(W1, W2)
    e1 = np.zeros((d, d))
    e2 = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            for M, G, E in ((W1, g1, e1), (W2, g2, e2)):
                M[i, j] += eps
                hp = fn(W1, W2)[0]
                M[i, j] -= 2 * eps
                hm = fn(W1, W2)[0]
                M[i, j] += eps
                E[i, j] = _rel(G[i, j], (hp - hm) / (2 * eps))
    return float(e1.max()), float(e2.max())


def check_primal_loss(name="released", d=6, n=60, seed=1, eps=1e-6, rho=2.3, alpha=0.7, allowed=None):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    X = X @ rng.standard_normal((d, d)) * 0.5
    prob = Problem(X, DCDConfig(penalty=name), allowed=allowed)
    W1 = rng.uniform(-1, 1, (d, d)) * (1 - np.eye(d))
    L = np.tril(rng.uniform(-0.5, 0.5, (d, d)), -1)
    theta = np.concatenate([W1.ravel(), L.ravel()])
    W1m, W2m = unpack(theta, d)
    Z = pseudo_variables(prob.X, W1m, W2m + np.diag(np.var(prob.X - prob.X @ W1m, axis=0)))
    quad = prob.quad(Z)
    f, g = prob.value_grad(theta, rho, alpha, quad)
    f_direct, g_direct = prob.value_grad(theta, rho, alpha, Z)
    path_gap = max(abs(f - f_direct), float(np.abs(g - g_direct).max()))
    errs = []
    for k in np.flatnonzero(prob.free):
        t = theta.copy()
        t[k] += eps
        fp = prob.value_grad(t, rho, alpha, quad)[0]
        t[k] -= 2 * eps
        fm = prob.value_grad(t, rho, alpha, quad)[0]
        errs.append(_rel(g[k], (fp - fm) / (2 * eps)))
    pinned = float(np.abs(g[~prob.free]).max()) if (~prob.free).any() else 0.0
    return float(np.max(errs)), pinned, path_gap


def main(tol=1e-5) -> int:
    ok = True
    for name in PENALTIES:
        e1, e2 = check_structure_penalty(name)
        allowed = np.ones((6, 6), dtype=bool)
        allowed[0, 3] = allowed[3, 0] = allowed[1, 4] = allowed[4, 1] = False
        ep, pinned, gap = check_primal_loss(name, allowed=allowed)
        print(f"[{name:8s}] h: max rel err dW1 = {e1:.2e}, dW2 = {e2:.2e} | primal loss: {ep:.2e} | "
              f"pinned |grad| = {pinned:.1e} | Gram-vs-direct gap = {gap:.1e}")
        ok &= max(e1, e2, ep) < tol and pinned == 0.0 and gap < 1e-9
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
