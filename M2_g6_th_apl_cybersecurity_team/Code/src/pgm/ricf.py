"""Residual Iterative Conditional Fitting (Drton, Eichler & Richardson 2009) for a *fixed* ADMG.

Used (i) to refit a thresholded graph so it can be scored with BIC, and (ii) in tests as an
independent reference for the optimizer's parameter estimates.

One sweep visits each variable i in turn and, with every other variable's residual held fixed,

    regress X_i on [ X_pa(i) , Z_i[:, sib(i)] ]            (Z_i = pseudo-variables, see admg_model)

* coefficients on X_pa(i)     -> W1[pa(i), i]
* coefficients on Z_i[:, sib] -> Omega[i, sib(i)]
* Omega_ii = s^2 + Omega_{i,-i} Omega_{-i,-i}^{-1} Omega_{-i,i}        (s^2 = residual variance)

Each coordinate update can only increase the likelihood for ancestral ADMGs (the class used here).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .admg_model import ADMGParams, gaussian_loglik


@dataclass
class RICFResult:
    W1: np.ndarray
    W2: np.ndarray
    Wii: np.ndarray
    loglik: float
    n_iter: int
    converged: bool
    trace: list

    def params(self) -> ADMGParams:
        return ADMGParams(self.W1, self.W2, self.Wii)


def fit_ricf(X: np.ndarray, D: np.ndarray, B: np.ndarray, max_iter: int = 200, tol: float = 1e-9) -> RICFResult:
    """Maximum-likelihood fit of the ADMG with directed pattern D (D[i,j]: i->j) and bidirected pattern B."""
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0)
    n, d = X.shape
    S = X.T @ X / n
    D = np.asarray(D) != 0
    B = (np.asarray(B) != 0) | (np.asarray(B) != 0).T
    np.fill_diagonal(B, False)

    W1 = np.zeros((d, d))
    Om = np.diag(np.diag(S)).copy()
    trace = []
    ll_old = -np.inf
    converged = False
    ll = -np.inf
    for it in range(1, max_iter + 1):
        for i in range(d):
            pa = np.flatnonzero(D[:, i])
            sib = np.flatnonzero(B[i])
            o = np.r_[0:i, i + 1:d]
            cols = [X[:, pa]]
            Om_oo = Om[np.ix_(o, o)]
            if len(sib):
                eps = X - X @ W1
                Z = np.linalg.solve(Om_oo, eps[:, o].T).T                 # n x (d-1)
                pos = np.searchsorted(o, sib)                              # sib positions inside o
                cols.append(Z[:, pos])
            A = np.hstack(cols)
            if A.shape[1]:
                coef = np.linalg.lstsq(A, X[:, i], rcond=None)[0]
                res = X[:, i] - A @ coef
            else:
                coef = np.zeros(0)
                res = X[:, i]
            s2 = float(np.mean(res ** 2))
            W1[:, i] = 0.0
            W1[pa, i] = coef[: len(pa)]
            gam = np.zeros(d - 1)
            if len(sib):
                gam[pos] = coef[len(pa):]
                Om[i, o] = gam
                Om[o, i] = gam
                Om[i, i] = s2 + gam @ np.linalg.solve(Om_oo, gam)
            else:
                Om[i, o] = 0.0
                Om[o, i] = 0.0
                Om[i, i] = s2
        ll = gaussian_loglik(S, n, W1, Om)
        trace.append(ll)
        if not np.isfinite(ll):
            break
        if abs(ll - ll_old) / n < tol:
            converged = True
            break
        ll_old = ll
    W2 = Om - np.diag(np.diag(Om))
    return RICFResult(W1, W2, np.diag(Om).copy(), float(ll), len(trace), converged, trace)
