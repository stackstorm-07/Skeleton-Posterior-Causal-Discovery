"""Probabilistic queries on a fitted linear-Gaussian ancestral ADMG (the 'inference layer').

Model (row convention, as in the rest of the package):   x = x W1 + eps,   eps ~ N(m_eps, Omega)
    =>  x ~ N(mu, Sigma),   mu = m_eps (I-W1)^-1,   Sigma = (I-W1)^-T Omega (I-W1)^-1.
W1[i, j] != 0 is a directed edge i -> j; the off-diagonal support of Omega are the bidirected edges.

Queries
-------
* ``marginal(idx)``                         joint law of a subset of services.
* ``condition(target, evidence)``           law of ``target`` given observed values of other services (Gaussian
                                            conditioning: Schur complement of Sigma). Observational.
* ``do(interventions)``                     law of the whole system after *setting* some services to fixed values:
                                            incoming directed edges and bidirected edges of an intervened node
                                            are cut, then the same formulas apply (graph surgery).
* ``total_effect(i, j)``                    change in E[x_j] per unit setting of x_i (do-operator).
* ``predictive_logpdf``                     held-out log density of conditional predictions (used for validation).

Identifiability. Ancestral ADMGs have no bidirected edge between a node and its ancestor, i.e. they are bow-free, and
for a linear model on a bow-free ADMG every edge coefficient is identified (Brito & Pearl 2002). Interventional
effects below are therefore identified *if the graph and the linear model are right*; with a learned graph they are
estimates whose quality is bounded by the graph's quality (see ADMG_JUSTIFICATION.md for what our graphs get wrong).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from .admg_model import safe_inverse
from .rca import RCAModel, fit_rca_model

Idx = Union[int, str]


@dataclass
class GaussianLaw:
    """A multivariate normal over ``names`` (mean, covariance)."""
    names: List[str]
    mean: np.ndarray
    cov: np.ndarray

    def sd(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.diag(self.cov), 0.0))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log density of the rows of ``x`` (n, k) under this law."""
        x = np.atleast_2d(x) - self.mean
        k = len(self.mean)
        L = np.linalg.cholesky(self.cov + 1e-12 * np.eye(k))
        z = np.linalg.solve(L, x.T)
        return -0.5 * (z ** 2).sum(axis=0) - np.log(np.diag(L)).sum() - 0.5 * k * np.log(2 * np.pi)


class ADMGInference:
    """Query object built from fitted parameters. ``mean`` is the mean of the observed variables (default 0)."""

    def __init__(self, names: Sequence[str], W1: np.ndarray, Omega: np.ndarray, mean: Optional[np.ndarray] = None,
                 ridge: float = 1e-9):
        self.names = list(names)
        self.W1 = np.asarray(W1, dtype=float)
        self.Omega = np.asarray(Omega, dtype=float)
        d = len(self.names)
        self.mean = np.zeros(d) if mean is None else np.asarray(mean, dtype=float)
        self._pos = {n: i for i, n in enumerate(self.names)}
        self._ridge = ridge
        A = np.eye(d) - self.W1
        self._Ainv = np.linalg.inv(A)
        self.Sigma = self._Ainv.T @ self.Omega @ self._Ainv
        self.Sigma = (self.Sigma + self.Sigma.T) / 2 + ridge * np.eye(d)

    # ---- construction -----------------------------------------------------------------------------------
    @classmethod
    def fit(cls, Z: np.ndarray, names: Sequence[str], D: np.ndarray, B: np.ndarray) -> "ADMGInference":
        """RICF fit of the graph (D, B) on data ``Z`` (n, d), then wrap the parameters."""
        m = fit_rca_model(Z, names, D, B)
        return cls(names, m.W1, m.Omega, mean=np.asarray(Z).mean(axis=0))

    @classmethod
    def from_model(cls, model: RCAModel, mean: Optional[np.ndarray] = None) -> "ADMGInference":
        return cls(model.names, model.W1, model.Omega, mean)

    # ---- helpers ----------------------------------------------------------------------------------------
    def _ix(self, items: Sequence[Idx]) -> np.ndarray:
        return np.array([self._pos[i] if isinstance(i, str) else int(i) for i in items], dtype=int)

    def joint(self) -> GaussianLaw:
        return GaussianLaw(self.names, self.mean.copy(), self.Sigma.copy())

    # ---- observational queries -------------------------------------------------------------------------
    def marginal(self, idx: Sequence[Idx]) -> GaussianLaw:
        ix = self._ix(idx)
        return GaussianLaw([self.names[i] for i in ix], self.mean[ix], self.Sigma[np.ix_(ix, ix)])

    def condition(self, target: Sequence[Idx], evidence: Mapping[Idx, float]) -> GaussianLaw:
        """Law of ``target`` given ``evidence`` {service: observed value} (values on the data's own scale)."""
        t = self._ix(target)
        e = self._ix(list(evidence.keys()))
        if len(set(t) & set(e)):
            raise ValueError("a service cannot be both target and evidence")
        if len(e) == 0:
            return self.marginal(target)
        xe = np.array([evidence[k] for k in evidence], dtype=float)
        Stt, Ste, See = self.Sigma[np.ix_(t, t)], self.Sigma[np.ix_(t, e)], self.Sigma[np.ix_(e, e)]
        G = np.linalg.solve(See, Ste.T).T                       # Ste See^-1
        mean = self.mean[t] + G @ (xe - self.mean[e])
        cov = Stt - G @ Ste.T
        return GaussianLaw([self.names[i] for i in t], mean, (cov + cov.T) / 2)

    def conditional_map(self, target: Idx, evidence_idx: Sequence[Idx]):
        """(intercept-free) linear predictor of ``target`` from the evidence set: returns (coef, cond_var)."""
        t = int(self._ix([target])[0])
        e = self._ix(evidence_idx)
        See = self.Sigma[np.ix_(e, e)]
        Ste = self.Sigma[t, e]
        coef = np.linalg.solve(See, Ste)
        var = self.Sigma[t, t] - Ste @ coef
        return coef, float(var)

    # ---- interventional queries ------------------------------------------------------------------------
    def do(self, interventions: Mapping[Idx, float]) -> GaussianLaw:
        """Law of all services after setting the given services to fixed values (graph surgery)."""
        d = len(self.names)
        iv = self._ix(list(interventions.keys()))
        vals = np.array([interventions[k] for k in interventions], dtype=float)
        W = self.W1.copy()
        Om = self.Omega.copy()
        W[:, iv] = 0.0                      # no parents for intervened nodes
        Om[iv, :] = 0.0
        Om[:, iv] = 0.0                     # and no shared noise with anyone
        m_eps = self.mean @ (np.eye(d) - self.W1)     # original noise means
        m_eps[iv] = vals
        Ainv = np.linalg.inv(np.eye(d) - W)
        mean = m_eps @ Ainv
        cov = Ainv.T @ Om @ Ainv
        return GaussianLaw(self.names, mean, (cov + cov.T) / 2)

    def total_effect(self, cause: Idx, effect: Idx) -> float:
        """d E[x_effect] / d (do x_cause = a)."""
        i = int(self._ix([cause])[0])
        j = int(self._ix([effect])[0])
        W = self.W1.copy()
        W[:, i] = 0.0
        return float(np.linalg.inv(np.eye(len(self.names)) - W)[i, j])

    def effects_of(self, cause: Idx, top: Optional[int] = None):
        """[(service, total effect)] of intervening on ``cause``, largest |effect| first (cause itself excluded)."""
        i = int(self._ix([cause])[0])
        W = self.W1.copy()
        W[:, i] = 0.0
        row = np.linalg.inv(np.eye(len(self.names)) - W)[i]
        out = [(self.names[j], float(row[j])) for j in range(len(row)) if j != i]
        out.sort(key=lambda t: -abs(t[1]))
        return out[:top] if top else out


# ---------------------------------------------------------------------------------------------------------
# validation helper: held-out conditional prediction
# ---------------------------------------------------------------------------------------------------------
def predictive_scores(inf: ADMGInference, X_test: np.ndarray, target: Idx, evidence_idx: Sequence[Idx]) -> Dict[str, float]:
    """Mean held-out log density (nats/row) and RMSE of predicting ``target`` from the evidence columns."""
    t = int(inf._ix([target])[0])
    e = inf._ix(evidence_idx)
    coef, var = inf.conditional_map(t, e)
    mu = inf.mean[t] + (X_test[:, e] - inf.mean[e]) @ coef
    y = X_test[:, t]
    var = max(var, 1e-9)
    lp = -0.5 * np.log(2 * np.pi * var) - 0.5 * (y - mu) ** 2 / var
    return dict(logpdf=float(lp.mean()), rmse=float(np.sqrt(np.mean((y - mu) ** 2))), cond_sd=float(np.sqrt(var)))
