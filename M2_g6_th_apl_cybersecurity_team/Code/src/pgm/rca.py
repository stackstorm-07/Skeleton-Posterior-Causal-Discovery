"""Stage 3: root-cause ranking with a learned (or true) ancestral ADMG.

Setting (PetShop, Hardt et al. 2023): a metric of a target service (PetSite) violates its SLO; we have normal
data (``noissue``) and abnormal data (rows of the issue file after the injection time) for all services and must
rank the services by how likely they are the injected root cause.

Model.  Fit a linear-Gaussian ADMG (W1: directed, Omega: noise covariance whose off-diagonal support is the set of
bidirected edges) to the *normal* data with RICF.  In row convention  x = x W1 + eps,  eps ~ N(0, Omega), so
  Sigma = (I - W1)^-T Omega (I - W1)^-1.
A root cause is a service whose own mechanism changed.  We model that as a mean shift ``a`` of its structural noise
eps_i.  The abnormal mean shift of the observed vector is  dmu = mean(abnormal) - mean(normal)  and in noise space
  d_eps = (I - W1)^T dmu .
(For a node whose parents account for its own shift, d_eps_i ~ 0: a service that merely *transmits* an upstream
anomaly is not blamed.)  The evidence for "only node i shifted" is the reduction of the Mahalanobis distance
d_eps' Omega^-1 d_eps obtained by allowing a shift a at i (generalised least squares):

      score_i = (Omega^-1 d_eps)_i ^ 2 / (Omega^-1)_ii .

With Omega diagonal this is d_eps_i^2 / Omega_ii (standardised residual shift given the parents, the CIRCA-style score).
With bidirected edges Omega^-1 accounts for the noise correlation that a shared hidden cause creates, so a shift at one
service is not smeared onto its confounded neighbours.  Comparing the two variants on the same graph isolates what the
bidirected part of the ADMG buys.

Methods (all share data, transform and candidate set):
  anomaly  no graph: score_i = dmu_i^2 (standardised mean shift; associational baseline)
  dag      directed edges only (Omega diagonal)
  admg     directed + bidirected edges (full Omega)
Optional ``ancestors_only``: restrict candidates to the directed ancestors of the target in the graph (falls back to all
nodes when the target has none).  The target itself is never a candidate: in PetShop the root cause is always another node.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

from .admg_model import admg_to_masks, safe_inverse
from .graphs import ADMG
from .ricf import fit_ricf

METHODS = ("anomaly", "dag", "admg")
TRANSFORMS = ("log", "zscore", "gauss_rank")


# --------------------------------------------------------------------------------------------------
# transform fitted on the NORMAL data only, applied unchanged to abnormal data
# --------------------------------------------------------------------------------------------------
class Transformer:
    """``log``: log1p then standardise; ``zscore``: standardise; ``gauss_rank``: normal scores via the normal-period ECDF
    (saturates at about +-3 for ~600 normal rows, which compresses very large shifts)."""

    def __init__(self, kind: str = "log"):
        if kind not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}")
        self.kind = kind

    def _pre(self, X):
        return np.log1p(np.maximum(X, 0.0)) if self.kind == "log" else np.asarray(X, dtype=float)

    def fit(self, X_normal: np.ndarray) -> "Transformer":
        Z = self._pre(X_normal)
        self.mu = Z.mean(axis=0)
        sd = Z.std(axis=0)
        sd[sd < 1e-12] = 1.0
        self.sd = sd
        self.sorted = np.sort(X_normal, axis=0) if self.kind == "gauss_rank" else None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.kind != "gauss_rank":
            return (self._pre(X) - self.mu) / self.sd
        n, d = self.sorted.shape
        out = np.empty(X.shape)
        for j in range(d):
            lo = np.searchsorted(self.sorted[:, j], X[:, j], side="left")
            hi = np.searchsorted(self.sorted[:, j], X[:, j], side="right")
            u = ((lo + hi) / 2.0 + 0.5) / (n + 1.0)
            out[:, j] = norm.ppf(np.clip(u, 0.5 / (n + 1), 1 - 0.5 / (n + 1)))
        base = norm.ppf((np.arange(n) + 1.0) / (n + 1.0))
        return (out - base.mean()) / max(base.std(), 1e-12)


# --------------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------------
@dataclass
class RCAModel:
    names: List[str]
    W1: np.ndarray          # (d, d) directed coefficients, W1[i, j]: i -> j
    Omega: np.ndarray       # (d, d) noise covariance (diagonal for the 'dag' variant)
    D: np.ndarray           # directed support used
    B: np.ndarray           # bidirected support used
    loglik: float

    @property
    def d(self):
        return len(self.names)


def fit_rca_model(Z_normal: np.ndarray, names: Sequence[str], D: np.ndarray, B: np.ndarray) -> RCAModel:
    """RICF fit of the ADMG (D, B) on transformed normal data."""
    D = (np.asarray(D) != 0).astype(int)
    B = (np.asarray(B) != 0).astype(int)
    res = fit_ricf(Z_normal, D, B)
    Om = res.W2 + np.diag(res.Wii)
    return RCAModel(list(names), res.W1, (Om + Om.T) / 2, D, B, res.loglik)


def directed_ancestors(D: np.ndarray, target: int) -> np.ndarray:
    """Boolean vector: nodes with a directed path to ``target`` (excluding the target)."""
    d = D.shape[0]
    seen = np.zeros(d, dtype=bool)
    stack = [target]
    while stack:
        v = stack.pop()
        for p in np.flatnonzero(D[:, v]):
            if not seen[p]:
                seen[p] = True
                stack.append(p)
    seen[target] = False
    return seen


# --------------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------------
def noise_shift(W1: np.ndarray, dmu: np.ndarray) -> np.ndarray:
    """d_eps = (I - W1)^T dmu : the part of each node's shift its parents do not explain."""
    return dmu - W1.T @ dmu


def mechanism_scores(model: RCAModel, dmu: np.ndarray, use_bidirected: bool = True) -> np.ndarray:
    """score_i = (Omega^-1 d_eps)_i^2 / (Omega^-1)_ii   (see module docstring)."""
    de = noise_shift(model.W1, dmu)
    Om = model.Omega if use_bidirected else np.diag(np.diag(model.Omega))
    K = safe_inverse(Om)
    return (K @ de) ** 2 / np.maximum(np.diag(K), 1e-12)


def rank_root_causes(dmu: np.ndarray, target: int, method: str, model: Optional[RCAModel] = None,
                     ancestors_only: bool = False, exclude_target: bool = True) -> np.ndarray:
    """Candidate indices sorted from most to least likely root cause."""
    d = len(dmu)
    if method == "anomaly":
        s = dmu ** 2
    elif method in ("dag", "admg"):
        if model is None:
            raise ValueError("graph methods need a model")
        s = mechanism_scores(model, dmu, use_bidirected=(method == "admg"))
    else:
        raise ValueError(f"method must be one of {METHODS}")
    cand = np.ones(d, dtype=bool)
    if exclude_target:
        cand[target] = False
    if ancestors_only and model is not None:
        anc = directed_ancestors(model.D, target)
        if (anc & cand).any():
            cand &= anc
    idx = np.flatnonzero(cand)
    return idx[np.argsort(-s[idx], kind="stable")]


def rank_of(order: np.ndarray, truth: int) -> Optional[int]:
    """1-based rank of ``truth`` in ``order`` (None if it is not a candidate)."""
    w = np.flatnonzero(order == truth)
    return int(w[0]) + 1 if len(w) else None


# --------------------------------------------------------------------------------------------------
# graph (de)serialisation, so that learned graphs are computed once and reused
# --------------------------------------------------------------------------------------------------
def admg_to_json(G: ADMG) -> Dict:
    return dict(vertices=list(G.vertices), di=sorted([list(e) for e in G.di_edges]),
                bi=sorted([sorted(e) for e in G.bi_edges]))


def admg_from_json(blob: Dict) -> ADMG:
    return ADMG(blob["vertices"], [tuple(e) for e in blob["di"]], [tuple(e) for e in blob["bi"]])


def masks_of(G: ADMG, names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """(D, B) of ``G`` in the order of ``names`` (G's vertex order may differ)."""
    D, B = admg_to_masks(G)
    pos = {v: i for i, v in enumerate(G.vertices)}
    perm = [pos[v] for v in names]
    return D[np.ix_(perm, perm)], B[np.ix_(perm, perm)]
