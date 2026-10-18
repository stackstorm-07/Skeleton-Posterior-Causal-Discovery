"""Linear-Gaussian ADMG model: parameters, vector layout, bounds, likelihood, thresholding, BIC.

Conventions (identical to the original SPOT/ABIC code, transposed relative to the paper's delta):

* ``W1[i, j] != 0``  <=>  edge  V_i -> V_j   (so the SEM is  X = X @ W1 + E  for row-vector samples)
* ``W2`` is symmetric with zero diagonal; ``W2[i, j] != 0`` <=> edge V_i <-> V_j, and it is the
  *off-diagonal of the noise covariance* Omega.
* ``Wii`` are the noise variances, so ``Omega = W2 + diag(Wii)``.
* Implied covariance of the observed variables:  Sigma = (I - W1)^{-T} Omega (I - W1)^{-1}.

Optimizer vector layout ``theta = [W1.ravel(), L.ravel()]`` (length 2 d^2).  ``L`` is a d x d matrix
whose *strictly lower triangle* holds the free bidirected parameters (``W2 = L_low + L_low^T``).
Every other entry of the vector is pinned to 0 by the bounds, exactly as in the original code, so
that a pair-level quantity (e.g. a skeleton posterior ``P[i, j]``) maps onto coordinates trivially.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .graphs import ADMG, has_directed_cycle

LOG2PI = np.log(2 * np.pi)


# --------------------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------------------
@dataclass
class ADMGParams:
    W1: np.ndarray
    W2: np.ndarray
    Wii: np.ndarray

    @property
    def d(self) -> int:
        return self.W1.shape[0]

    def omega(self) -> np.ndarray:
        return self.W2 + np.diag(self.Wii)

    def sigma(self) -> np.ndarray:
        Minv = np.linalg.inv(np.eye(self.d) - self.W1)
        return Minv.T @ self.omega() @ Minv

    def copy(self) -> "ADMGParams":
        return ADMGParams(self.W1.copy(), self.W2.copy(), self.Wii.copy())


# --------------------------------------------------------------------------------------
# vector layout
# --------------------------------------------------------------------------------------
def pack(W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    return np.concatenate([W1.ravel(), np.tril(W2, -1).ravel()])


def unpack(theta: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    W1 = theta[: d * d].reshape(d, d)
    L = np.tril(theta[d * d:].reshape(d, d), -1)
    return W1, L + L.T


def free_mask(d: int, allowed: Optional[np.ndarray] = None) -> np.ndarray:
    """Boolean mask over theta of the coordinates that may be non-zero.

    ``allowed`` is an optional symmetric (d, d) boolean matrix of pairs that are allowed to be
    adjacent (a *hard* skeleton constraint); ``None`` allows every pair.
    """
    if allowed is None:
        allowed = np.ones((d, d), dtype=bool)
    allowed = np.asarray(allowed, dtype=bool) & np.asarray(allowed, dtype=bool).T
    off = ~np.eye(d, dtype=bool)
    m1 = allowed & off
    m2 = allowed & np.tril(np.ones((d, d), dtype=bool), -1)
    return np.concatenate([m1.ravel(), m2.ravel()])


def make_bounds(d: int, allowed: Optional[np.ndarray] = None, bound: float = 4.0) -> List[Tuple[float, float]]:
    fm = free_mask(d, allowed)
    return [(-bound, bound) if f else (0.0, 0.0) for f in fm]


# --------------------------------------------------------------------------------------
# linear algebra helpers
# --------------------------------------------------------------------------------------
def safe_inverse(Omega: np.ndarray, floor_rel: float = 1e-3) -> np.ndarray:
    """Inverse of a symmetric matrix; if it is not positive definite, clip its eigenvalues first."""
    Om = (Omega + Omega.T) / 2
    try:
        np.linalg.cholesky(Om)
        return np.linalg.inv(Om)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(Om)
        floor = floor_rel * max(np.mean(np.abs(np.diag(Om))), 1e-12)
        w = np.maximum(w, floor)
        return (V / w) @ V.T


def pseudo_variables(X: np.ndarray, W1: np.ndarray, Omega: np.ndarray) -> np.ndarray:
    """RICF pseudo-variables.  ``Z[i]`` is an (n, d) array with column i equal to 0 and, for j != i,

        Z[i][:, -i] = eps[:, -i] @ Omega_{-i,-i}^{-1},     eps = X - X @ W1

    Regressing eps_i on Z[i] gives the coefficients Omega[i, -i]  (E[eps_i | eps_-i] =
    Omega_{i,-i} Omega_{-i,-i}^{-1} eps_{-i}), which is why the loss in dcd.py is linear in W2.
    """
    n, d = X.shape
    eps = X - X @ W1
    K = safe_inverse(Omega)
    Z = np.zeros((d, n, d))
    for i in range(d):
        o = np.r_[0:i, i + 1:d]
        Koo = K[np.ix_(o, o)] - np.outer(K[o, i], K[i, o]) / K[i, i]   # = Omega_{-i,-i}^{-1}
        Z[i][:, o] = eps[:, o] @ Koo
    return Z


# --------------------------------------------------------------------------------------
# likelihood and BIC
# --------------------------------------------------------------------------------------
def gaussian_loglik(S: np.ndarray, n: int, W1: np.ndarray, Omega: np.ndarray) -> float:
    """Gaussian log-likelihood of n centred samples with sample covariance S under (W1, Omega).

    -n/2 [ d log 2pi + log det Omega - 2 log|det(I - W1)| + tr( (I-W1) Omega^{-1} (I-W1)^T S ) ].
    Returns -inf when Omega is not positive definite or I - W1 is singular.
    """
    d = S.shape[0]
    M = np.eye(d) - W1
    s1, ld_om = np.linalg.slogdet(Omega)
    s2, ld_m = np.linalg.slogdet(M)
    if s1 <= 0 or s2 == 0:
        return -np.inf
    try:
        np.linalg.cholesky((Omega + Omega.T) / 2)
    except np.linalg.LinAlgError:
        return -np.inf
    K = M @ np.linalg.solve(Omega, M.T)
    return float(-0.5 * n * (d * LOG2PI + ld_om - 2 * ld_m + np.sum(K * S)))


def n_parameters(admg: ADMG) -> int:
    return len(admg.di_edges) + len(admg.bi_edges) + len(admg.vertices)


def bic_from_loglik(ll: float, k: int, n: int) -> float:
    """Lower is better (as in the original code)."""
    return -2.0 * ll + k * np.log(n)


def bic(X: np.ndarray, admg: ADMG) -> float:
    """BIC of the ADMG after refitting it with RICF.  ``inf`` for cyclic graphs / failed fits."""
    from .ricf import fit_ricf  # local import: ricf depends on this module
    D, B = admg_to_masks(admg)
    if has_directed_cycle(D):
        return np.inf
    res = fit_ricf(X, D, B)
    if not np.isfinite(res.loglik):
        return np.inf
    return bic_from_loglik(res.loglik, n_parameters(admg), X.shape[0])


# --------------------------------------------------------------------------------------
# ADMG <-> matrices
# --------------------------------------------------------------------------------------
def admg_to_masks(admg: ADMG) -> Tuple[np.ndarray, np.ndarray]:
    """(D, B) binary matrices: D[i, j] = 1 iff i -> j, B symmetric with B[i, j] = 1 iff i <-> j."""
    idx = {v: i for i, v in enumerate(admg.vertices)}
    d = len(admg.vertices)
    D = np.zeros((d, d), dtype=int)
    B = np.zeros((d, d), dtype=int)
    for a, b in admg.di_edges:
        D[idx[a], idx[b]] = 1
    for e in admg.bi_edges:
        a, b = tuple(e)
        B[idx[a], idx[b]] = B[idx[b], idx[a]] = 1
    return D, B


def masks_to_admg(names: Sequence[str], D: np.ndarray, B: np.ndarray) -> ADMG:
    G = ADMG(names)
    d = len(names)
    for i in range(d):
        for j in range(d):
            if D[i, j]:
                G.add_diedge(names[i], names[j])
            if i < j and B[i, j]:
                G.add_biedge(names[i], names[j])
    return G


def acyclic_threshold(W1: np.ndarray, thr0: float, grow: float = 1.05, thr_max: float = 2.0) -> Tuple[np.ndarray, float]:
    """Raise the magnitude threshold on W1 from ``thr0`` until its support has no directed cycle.

    Returns (support-thresholded W1, threshold used).  If even ``thr_max`` leaves a cycle the empty
    matrix is returned (the original code also falls back to an empty graph).
    """
    thr = thr0
    while thr <= thr_max:
        Wt = np.where(np.abs(W1) > thr, W1, 0.0)
        if not has_directed_cycle(Wt):
            return Wt, thr
        thr *= grow
    return np.zeros_like(W1), thr


def to_admg(names: Sequence[str], W1: np.ndarray, W2: np.ndarray, thr: float) -> ADMG:
    D = (np.abs(W1) > thr).astype(int)
    np.fill_diagonal(D, 0)
    B = (np.abs(W2) > thr).astype(int)
    np.fill_diagonal(B, 0)
    B = np.maximum(B, B.T)
    return masks_to_admg(names, D, B)
