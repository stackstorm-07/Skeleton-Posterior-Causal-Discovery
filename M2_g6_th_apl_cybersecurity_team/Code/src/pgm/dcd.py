"""Shared differentiable ADMG learner: the ABIC backbone, with one optional hook for SPOT.

ABIC  = this optimizer with ``mask=None``.
SPOT  = this optimizer with a gradient mask built from a skeleton posterior (see spot.py).
Everything else (loss, constraint, solver settings, restarts, thresholds, budget) is identical, so a
timing/accuracy comparison between the two isolates the effect of the mask.

Objective for one RICF step (Z = pseudo-variables, held fixed during the solve):

    f(W1, W2) = 1/(2n) sum_i || x_i - X W1[:, i] - Z_i W2[:, i] ||^2        (pseudo-variable least squares)
              + lam * sum tanh(l0_scale * log(n) * |theta|)                   (smooth L0 sparsity)
              + 0.5 rho h^2 + alpha h                                         (augmented Lagrangian)

h = 0  <=>  the support of (W1, W2) is an ancestral ADMG.  Two interchangeable forms of h:
  * ``penalty="released"`` (default): exactly the code released with ABIC/SPOT (baselines/original_dcd.py):
    NOTEARS-poly acyclicity  tr((I + W1*W1/d)^d) - d  plus the ancestrality term sum(S o W2*W2) with the
    truncated series S = sum_{k<d} (W1*W1)^k / k!.
  * ``penalty="expm"``: the paper's Eq. 4, tr(e^{W1*W1}) - d + sum(e^{W1*W1} o W2*W2), exact Frechet gradient.
All gradients are analytic (no autodiff library) and verified against finite differences in
scripts/gradcheck.py / tests/test_gradients.py.  The data-fit term is evaluated through Gram matrices
(``QuadLoss``), which is exactly equal to the direct pseudo-variable form but several times cheaper.

Known behaviour (measured in stage 2a, see STAGE2A_NOTES.md): the pseudo-variable loss counts a bidirected
edge in *both* endpoint equations, so on z-scored data the optimizer over-uses bidirected edges; restarts
chosen by BIC matter a lot; a (hard or soft) skeleton prior helps accuracy and speed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm_frechet
from scipy.optimize import minimize

from .admg_model import (ADMGParams, acyclic_threshold, admg_to_masks, bic, free_mask, make_bounds,
                         pack, pseudo_variables, to_admg, unpack)
from .graphs import ADMG

GradientMask = Callable[[np.ndarray, np.ndarray, int], np.ndarray]
"""mask(grad, theta, step) -> grad.   ``step`` is the 1-based augmented-Lagrangian outer iteration."""

BIG = 1e30


@dataclass
class DCDConfig:
    lam: float = 0.05                 # L0 strength
    penalty: str = "released"         # structure penalty h: 'released' (code of the paper) or 'expm' (paper Eq. 4)
    l0_scale: float = 0.5             # smooth L0 = tanh(l0_scale * log(n) * |theta|); 0.5 reproduces the released code
    max_outer: int = 100              # augmented-Lagrangian outer iterations
    h_tol: float = 1e-8
    rho_max: float = 1e16
    w_threshold: float = 0.05         # initial magnitude threshold for edges
    ricf_increment: int = 1           # RICF iterations added per outer iteration
    ricf_tol: float = 1e-4
    lbfgs_maxiter: int = 1000
    bound: float = 4.0
    num_restarts: int = 1
    init_w1: float = 0.5
    init_w2: float = 0.05
    seed: int = 0
    time_budget: Optional[float] = None     # total wall-clock seconds over all restarts
    verbose: bool = False


@dataclass
class DCDResult:
    admg: ADMG
    W1: np.ndarray            # thresholded directed weights
    W2: np.ndarray            # thresholded bidirected (covariance) weights
    W1_raw: np.ndarray
    W2_raw: np.ndarray
    threshold: float          # threshold actually used (raised if the support was cyclic)
    h: float
    bic: float
    converged: bool           # h <= h_tol at termination
    timed_out: bool
    n_outer: int
    n_solves: int
    n_fev: int
    elapsed: float            # total wall clock of discover_admg
    t_optimize: float = 0.0   # seconds inside the augmented-Lagrangian solves (what time_budget limits)
    t_select: float = 0.0     # seconds spent thresholding + BIC-scoring the restarts
    restart_bics: List[float] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)


class _Budget(Exception):
    pass


# --------------------------------------------------------------------------------------
# analytic loss pieces
# --------------------------------------------------------------------------------------
def structure_penalty(W1: np.ndarray, W2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """h and its gradients w.r.t. W1 and (the symmetric, entry-wise independent) W2."""
    d = W1.shape[0]
    P1, P2 = W1 * W1, W2 * W2
    C = np.eye(d) + P2
    expA, F = expm_frechet(P1.T, C)       # expA = expm(P1^T); F = L(P1^T, C) = d<C, expm(P1)>/dP1
    E = expA.T
    h = float(np.sum(C * E)) - d
    return h, 2.0 * W1 * F, 2.0 * W2 * E


def structure_penalty_released(W1: np.ndarray, W2: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """The penalty of the *released* ABIC/SPOT code (baselines/original_dcd.py), with analytic gradients.

    h = cycle_loss + ancestrality_loss with
        cycle_loss        = tr( (I + W1*W1 / d)^d ) - d                       (NOTEARS-poly acyclicity)
        ancestrality_loss = sum( S o (W2*W2) ),  S = sum_{k=0}^{d-1} (W1*W1)^k / k!   (truncated e^{W1*W1})
    The truncated series is differentiated by reverse-mode through its Horner recursion
    T_k = I + P T_{k+1} / (k+1), which is exact (checked against finite differences).
    """
    d = W1.shape[0]
    P, Q = W1 * W1, W2 * W2
    I = np.eye(d)
    M = I + P / d
    E = np.linalg.matrix_power(M, d - 1)
    h_c = float(np.sum(E.T * M)) - d
    g1 = 2.0 * W1 * E.T
    K = d - 1
    Ts = [None] * (K + 1)
    Ts[K] = I
    for k in range(K - 1, -1, -1):
        Ts[k] = I + P @ Ts[k + 1] / (k + 1)
    S = Ts[0]
    h_a = float(np.sum(S * Q))
    G = Q.copy()
    gP = np.zeros((d, d))
    for k in range(K):
        gP += G @ Ts[k + 1].T / (k + 1)
        G = P.T @ G / (k + 1)
    g1 = g1 + 2.0 * W1 * gP
    return h_c + h_a, g1, 2.0 * W2 * S


PENALTIES = {"expm": structure_penalty, "released": structure_penalty_released}


def ls_loss_grad(X: np.ndarray, W1: np.ndarray, W2: np.ndarray, Z: np.ndarray):
    n = X.shape[0]
    R = X - X @ W1 - np.einsum("ind,di->ni", Z, W2)
    f = 0.5 / n * float(np.sum(R * R))
    gW1 = -(X.T @ R) / n
    gW2 = -np.einsum("ind,ni->di", Z, R) / n
    return f, gW1, gW2


class QuadLoss:
    """The pseudo-variable least-squares loss for FIXED pseudo-variables Z, via Gram matrices.

    f = 1/(2n) sum_i || x_i - X w1_i - Z_i w2_i ||^2  is quadratic in (w1_i, w2_i), so with
        S = X'X/n,  G_i = X'Z_i/n,  H_i = Z_i'Z_i/n,  b_i = Z_i'x_i/n
    f_i = 1/2 [ S_ii - 2 (S[:,i].w1 + b_i.w2) + w1'S w1 + 2 w1'G_i w2 + w2'H_i w2 ],
    and the gradient is (S w1 + G_i w2 - S[:,i],  G_i' w1 + H_i w2 - b_i).  Exactly equal to
    ``ls_loss_grad`` (tested) but O(d^3) per evaluation instead of O(n d^2): the Grams are built once
    per RICF step and reused for the several hundred L-BFGS evaluations of that step.
    """

    def __init__(self, X: np.ndarray, S: np.ndarray, Z: np.ndarray):
        n = X.shape[0]
        self.S = S
        self.G = np.einsum("nk,ind->ikd", X, Z, optimize=True) / n        # G[i] = X' Z_i / n   (d, d, d)
        self.H = np.einsum("inx,iny->ixy", Z, Z, optimize=True) / n       # H[i] = Z_i' Z_i / n
        self.b = np.einsum("ind,ni->di", Z, X, optimize=True) / n         # b[:, i] = Z_i' x_i / n
        self.c = float(np.trace(S))

    def value_grad(self, W1: np.ndarray, W2: np.ndarray):
        S = self.S
        GW2 = np.einsum("ikd,di->ki", self.G, W2)       # G_i w2_i  (columns)
        HW2 = np.einsum("ixy,yi->xi", self.H, W2)       # H_i w2_i
        GTW1 = np.einsum("ikd,ki->di", self.G, W1)      # G_i' w1_i
        SW1 = S @ W1
        f = 0.5 * (self.c - 2.0 * (np.sum(S * W1) + np.sum(self.b * W2)) + np.sum(W1 * SW1)
                   + 2.0 * np.sum(W1 * GW2) + np.sum(W2 * HW2))
        return float(f), SW1 + GW2 - S, GTW1 + HW2 - self.b


class Problem:
    """Data + the analytic value/gradient of the primal (augmented-Lagrangian) loss."""

    def __init__(self, X: np.ndarray, cfg: DCDConfig, allowed: Optional[np.ndarray] = None):
        self.X = np.asarray(X, dtype=float) - np.mean(X, axis=0)
        self.n, self.d = self.X.shape
        self.cfg = cfg
        self.logn = float(np.log(self.n))
        self.free = free_mask(self.d, allowed)
        self.bounds = make_bounds(self.d, allowed, cfg.bound)
        self.S = self.X.T @ self.X / self.n
        self.penalty_fn = PENALTIES[cfg.penalty]

    def quad(self, Z: np.ndarray) -> QuadLoss:
        return QuadLoss(self.X, self.S, Z)

    def value_grad(self, theta: np.ndarray, rho: float, alpha: float, Z) -> Tuple[float, np.ndarray]:
        """``Z`` is either a QuadLoss (fast path used by the optimizer) or the raw pseudo-variables."""
        d = self.d
        W1, W2 = unpack(theta, d)
        if isinstance(Z, QuadLoss):
            f_ls, gW1_ls, gW2_ls = Z.value_grad(W1, W2)
        else:
            f_ls, gW1_ls, gW2_ls = ls_loss_grad(self.X, W1, W2, Z)
        h, gW1_h, gW2_h = self.penalty_fn(W1, W2)
        if not np.isfinite(h) or not np.isfinite(f_ls):
            return BIG, np.zeros_like(theta)
        c = rho * h + alpha
        gW1 = gW1_ls + c * gW1_h
        G2 = gW2_ls + c * gW2_h
        gL = np.tril(G2 + G2.T, -1)                       # W2[i,j] = W2[j,i] = L[i,j] (i > j)
        g = np.concatenate([gW1.ravel(), gL.ravel()])
        k = self.cfg.l0_scale * self.logn
        t = np.tanh(k * np.abs(theta))
        l0 = self.cfg.lam * float(np.sum(t))
        g = g + self.cfg.lam * k * (1.0 - t * t) * np.sign(theta)
        g = np.where(self.free, g, 0.0)
        return f_ls + 0.5 * rho * h * h + alpha * h + l0, g


# --------------------------------------------------------------------------------------
# optimizer
# --------------------------------------------------------------------------------------
class _Run:
    """One random restart of the augmented-Lagrangian / RICF procedure."""

    def __init__(self, prob: Problem, cfg: DCDConfig, mask: Optional[GradientMask], deadline: Optional[float]):
        self.p, self.cfg, self.mask, self.deadline = prob, cfg, mask, deadline
        self.n_fev = 0
        self.n_solves = 0
        self.timed_out = False

    def _solve(self, theta, rho, alpha, Z, step):
        best = {"f": np.inf, "theta": theta.copy()}

        def fun(th):
            if self.deadline is not None and time.perf_counter() > self.deadline:
                raise _Budget
            f, g = self.p.value_grad(th, rho, alpha, Z)
            self.n_fev += 1
            if self.mask is not None:
                g = self.mask(g, th, step)
            if f < best["f"]:
                best["f"], best["theta"] = f, th.copy()
            return f, g

        self.n_solves += 1
        try:
            sol = minimize(fun, theta, jac=True, method="L-BFGS-B", bounds=self.p.bounds,
                           options={"maxiter": self.cfg.lbfgs_maxiter, "maxfun": 5 * self.cfg.lbfgs_maxiter})
            return sol.x
        except _Budget:
            self.timed_out = True
            return best["theta"]

    def run(self, rng: np.random.Generator):
        p, cfg = self.p, self.cfg
        d = p.d
        fm = p.free
        W1 = rng.uniform(-cfg.init_w1, cfg.init_w1, (d, d))
        L = np.tril(rng.uniform(-cfg.init_w2, cfg.init_w2, (d, d)), -1)
        theta = np.concatenate([W1.ravel(), L.ravel()]) * fm
        W1h, W2h = unpack(theta, d)
        Wii = np.diag(p.S).copy()
        rho, alpha, h = 1.0, 0.0, np.inf
        ricf_max = 1
        history = []
        n_outer = 0
        for outer in range(cfg.max_outer):
            if self.timed_out:
                break
            n_outer = outer + 1
            W1n = W2n = Wiin = h_new = None
            while rho < cfg.rho_max:
                W1n, W2n, Wiin = W1h.copy(), W2h.copy(), Wii.copy()
                for _ in range(ricf_max):
                    W1o, W2o, Wo = W1n.copy(), W2n.copy(), Wiin.copy()
                    Z = pseudo_variables(p.X, W1n, W2n + np.diag(Wiin))
                    th = self._solve(pack(W1n, W2n), rho, alpha, p.quad(Z), n_outer)
                    W1n, W2n = unpack(th, d)
                    Wiin = np.var(p.X - p.X @ W1n, axis=0)
                    if self.timed_out:
                        break
                    if (np.abs(W1o - W1n).sum() + np.abs((W2o + np.diag(Wo)) - (W2n + np.diag(Wiin))).sum()) < cfg.ricf_tol:
                        break
                h_new = p.penalty_fn(W1n, W2n)[0]
                if cfg.verbose:
                    print(f"  outer {n_outer:3d} rho={rho:9.2e} h={h_new:11.4e}")
                if self.timed_out or h_new < 0.25 * h:
                    break
                rho *= 10
            W1h, W2h, Wii = W1n.copy(), W2n.copy(), Wiin.copy()
            h = h_new
            alpha += rho * h
            ricf_max += cfg.ricf_increment
            history.append(dict(outer=n_outer, rho=rho, alpha=alpha, h=float(h), t=time.perf_counter()))
            if self.timed_out or h <= cfg.h_tol or rho >= cfg.rho_max:
                break
        return W1h, W2h, Wii, float(h), n_outer, history


def discover_admg(X: np.ndarray, names: Sequence[str], cfg: Optional[DCDConfig] = None,
                  mask: Optional[GradientMask] = None, allowed: Optional[np.ndarray] = None) -> DCDResult:
    """Learn an ancestral ADMG from data.  ``mask``/``allowed`` are the only SPOT hooks."""
    cfg = cfg or DCDConfig()
    t0 = time.perf_counter()
    prob = Problem(X, cfg, allowed)
    rng = np.random.default_rng(cfg.seed)
    best = None
    bics: List[float] = []
    totals = dict(n_solves=0, n_fev=0, n_outer=0)
    t_opt = t_sel = 0.0
    timed_out = False
    for r in range(cfg.num_restarts):
        deadline = None
        if cfg.time_budget is not None:
            left = cfg.time_budget - (time.perf_counter() - t0)
            deadline = time.perf_counter() + max(left, 0.0) / (cfg.num_restarts - r)
        run = _Run(prob, cfg, mask, deadline)
        t_a = time.perf_counter()
        W1h, W2h, Wii, h, n_outer, hist = run.run(rng)
        t_b = time.perf_counter()
        W1t, thr = acyclic_threshold(W1h, cfg.w_threshold)
        G = to_admg(names, W1t, W2h, thr)
        b = bic(prob.X, G)
        bics.append(float(b))
        t_opt += t_b - t_a
        t_sel += time.perf_counter() - t_b
        totals["n_solves"] += run.n_solves
        totals["n_fev"] += run.n_fev
        totals["n_outer"] += n_outer
        timed_out |= run.timed_out
        if best is None or b < best[0]:
            best = (b, G, W1t, np.where(np.abs(W2h) > thr, W2h, 0.0), W1h, W2h, thr, h, hist)
        if cfg.verbose:
            print(f"restart {r + 1}/{cfg.num_restarts}: h={h:.2e} bic={b:.1f} edges={G.n_edges()}")
    b, G, W1t, W2t, W1h, W2h, thr, h, hist = best
    return DCDResult(admg=G, W1=W1t, W2=W2t, W1_raw=W1h, W2_raw=W2h, threshold=thr, h=h, bic=float(b),
                     converged=bool(h <= cfg.h_tol), timed_out=timed_out, n_outer=totals["n_outer"],
                     n_solves=totals["n_solves"], n_fev=totals["n_fev"], elapsed=time.perf_counter() - t0,
                     t_optimize=t_opt, t_select=t_sel, restart_bics=bics, history=hist)
