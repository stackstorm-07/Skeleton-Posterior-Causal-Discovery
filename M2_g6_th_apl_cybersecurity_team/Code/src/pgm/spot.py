"""SPOT (Ma et al., KDD'24): the shared optimizer in dcd.py plus a skeleton-posterior gradient mask.

ABIC = ``discover_admg(mask=None)``.   SPOT = the *same call* with ``mask=SpotMask(P, ...)`` (and, optionally,
the hard floor below).  Nothing else differs (loss, penalty, L-BFGS settings, restarts, thresholds, budget),
so a timing/accuracy difference between the two isolates the effect of the posterior.

What SPOT does (paper Alg. 2 / Eq. 9; released code ``baselines/sa_ml4s_dcd.py``)
---------------------------------------------------------------------------------
At every L-BFGS-B function evaluation the gradient g of the augmented-Lagrangian objective is masked
coordinate by coordinate.  For a coordinate theta_k that belongs to the vertex pair (i, j):

    theta_k * g_k > 0   ("weakening": a descent step moves theta_k towards 0)   -> always kept
    otherwise           ("strengthening": theta_k moves away from 0)           -> kept with probability
                                                                                  (p_ij + c) ** (1 / (t + 1))

where ``p_ij`` is the skeleton posterior, ``c`` = 0.1 the floor constant and ``t`` the 1-based augmented-Lagrangian
outer iteration.  A rejected coordinate gets gradient 0 for that evaluation.  As ``t`` grows the exponent goes to 0,
the probability to 1, and SPOT turns into plain ABIC; sparsifying moves are never blocked.  The same ``p_ij`` is used
for the directed coordinates W1[i, j], W1[j, i] and for the bidirected coordinate (lower triangle) of the pair.

Two details of the *released* code that the paper text does not mention (both are switches here):

* ``floor``: pairs with p <= 0.001 are pinned to 0 through the optimizer's box bounds (a hard constraint) in every
  released SPOT script.  We keep it as the default (it is part of the code that produced the paper's numbers) and
  ablate it (``floor=None``) so the report can say how much of the effect is the stochastic mask and how much is pruning.
* ``variant="released_stochastic"`` is the *other* released script (``stochastic_ml4s_dcd.py``): sign-blind, rejected
  coordinates are scaled by 0.1 instead of zeroed, the acceptance probability is ``min(1, 0.1 + sqrt(p))`` and does
  not anneal, and the mask is only used during the first 5 outer iterations.  It does not match the paper's Eq. 9,
  so it is not the default; it is here for completeness.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, fields
from typing import Dict, Optional, Sequence

import numpy as np

from .admg_model import free_mask
from .dcd import DCDConfig, DCDResult, GradientMask, discover_admg

VARIANTS = ("paper", "released_stochastic")
RELEASED_FLOOR = 1e-3        # the value hard-coded in every released SPOT script (tuned to the released xgboost cascade)
CALIBRATED_FLOOR = 6e-3      # largest floor pinning <= 1 % of true edges for OUR cascade, on held-out simulated graphs
                             # (scripts/calibrate_floor.py -> results/floor_calibration.json); never tuned on benchmark data


@dataclass
class SpotConfig:
    variant: str = "paper"            # 'paper' (Eq. 9, default) | 'released_stochastic'
    c: float = 0.1                    # additive constant of Eq. 9
    floor: Optional[float] = RELEASED_FLOOR   # pairs with p <= floor are hard-pinned to 0 (released code); None = no pinning
    mask_seed: int = 0                # seed of the mask's own random stream (independent of the optimizer's init)
    charge_posterior: bool = True     # subtract the posterior's online inference time from a wall-clock budget

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")


def _effective_p(P: np.ndarray, variant: str) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    return np.sqrt(np.clip(P, 0.0, 1.0)) if variant == "released_stochastic" else np.clip(P, 0.0, 1.0)


def acceptance_probability(P: np.ndarray, step: int, cfg: SpotConfig) -> np.ndarray:
    """Probability that a *strengthening* update of a pair with posterior ``P`` is kept at outer iteration ``step``."""
    if cfg.variant == "paper":
        return np.minimum((np.clip(P, 0.0, 1.0) + cfg.c) ** (1.0 / (step + 1.0)), 1.0)
    return np.minimum(cfg.c + _effective_p(P, cfg.variant), 1.0)


def allowed_pairs(P: np.ndarray, cfg: SpotConfig) -> Optional[np.ndarray]:
    """Symmetric boolean matrix of the pairs that may be adjacent (p > floor), or None when no pinning is used."""
    if cfg.floor is None:
        return None
    A = _effective_p(P, cfg.variant) > cfg.floor
    A = A & A.T
    np.fill_diagonal(A, False)
    return A


class SpotMask:
    """The gradient mask ``mask(grad, theta, step) -> grad`` expected by ``discover_admg``.

    ``P`` is the (d, d) symmetric skeleton posterior.  Counters (``stats``) record, per outer iteration, how many
    strengthening proposals were made and how many were kept, over the coordinates that are actually free.
    """
    warmup_outer_released = 5

    def __init__(self, P: np.ndarray, cfg: Optional[SpotConfig] = None, allowed: Optional[np.ndarray] = None):
        self.cfg = cfg or SpotConfig()
        P = np.asarray(P, dtype=float)
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("P must be a square matrix")
        self.d = P.shape[0]
        self.P = 0.5 * (P + P.T)
        # one posterior value per optimizer coordinate: [W1.ravel(), L.ravel()] -> the pair's p twice
        self.p_vec = np.concatenate([self.P.ravel(), self.P.ravel()])
        self.free = free_mask(self.d, allowed)
        self.rng = np.random.default_rng(self.cfg.mask_seed)
        self.stats: Dict[int, list] = {}              # step -> [n_strengthening_proposals, n_kept, n_weakening]
        self._p_cache: Dict[int, np.ndarray] = {}

    def _accept_prob(self, step: int) -> np.ndarray:
        if step not in self._p_cache:
            self._p_cache[step] = acceptance_probability(self.p_vec, step, self.cfg)
        return self._p_cache[step]

    def __call__(self, grad: np.ndarray, theta: np.ndarray, step: int) -> np.ndarray:
        cfg = self.cfg
        if cfg.variant == "released_stochastic":
            if step > self.warmup_outer_released:
                return grad
            keep = self._accept_prob(step) > self.rng.random(grad.shape)
            return np.where(keep, 1.0, 0.1) * grad
        weakening = theta * grad > 0.0
        keep = weakening | (self._accept_prob(step) > self.rng.random(grad.shape))
        rec = self.stats.setdefault(step, [0, 0, 0])
        strengthen = self.free & ~weakening
        rec[0] += int(strengthen.sum())
        rec[1] += int((strengthen & keep).sum())
        rec[2] += int((self.free & weakening).sum())
        return np.where(keep, grad, 0.0)

    def summary(self) -> Dict[int, Dict[str, float]]:
        out = {}
        for s, (prop, kept, weak) in sorted(self.stats.items()):
            out[s] = dict(strengthening_proposals=prop, strengthening_kept=kept, weakening=weak,
                          kept_fraction=(kept / prop) if prop else float("nan"))
        return out


@dataclass
class SpotResult(DCDResult):
    """DCDResult plus the SPOT-specific bookkeeping (three-number runtime: offline / online / optimization)."""
    t_posterior: float = 0.0                 # online posterior inference (seconds); 0 when P was supplied
    P: Optional[np.ndarray] = None
    n_pairs_allowed: int = -1                # number of unordered pairs left free by the floor (-1 = no floor)
    mask_summary: Optional[Dict] = None
    spot_cfg: Optional[SpotConfig] = None


def run_spot(X: np.ndarray, names: Sequence[str], P: Optional[np.ndarray] = None, posterior=None,
             cfg: Optional[DCDConfig] = None, spot_cfg: Optional[SpotConfig] = None) -> SpotResult:
    """Run SPOT on preprocessed data ``X`` (n, d).

    Give either ``P`` (a precomputed (d, d) skeleton posterior; this is how the oracle / random ablations are run)
    or ``posterior`` (a fitted ``SkeletonPosterior``; its online inference time is measured and, if ``cfg.time_budget``
    is set and ``spot_cfg.charge_posterior``, deducted from the optimizer's wall-clock budget so that ABIC and SPOT get
    the same total time).
    """
    cfg = cfg or DCDConfig()
    spot_cfg = spot_cfg or SpotConfig()
    t_post = 0.0
    if P is None:
        if posterior is None:
            raise ValueError("run_spot needs P or posterior")
        t0 = time.perf_counter()
        P = posterior.predict(X).P
        t_post = time.perf_counter() - t0
    P = np.asarray(P, dtype=float)
    if P.shape != (X.shape[1], X.shape[1]):
        raise ValueError(f"P has shape {P.shape}, expected {(X.shape[1],) * 2}")
    if cfg.time_budget is not None and spot_cfg.charge_posterior and t_post > 0:
        cfg = DCDConfig(**{**cfg.__dict__, "time_budget": max(cfg.time_budget - t_post, 0.0)})
    allowed = allowed_pairs(P, spot_cfg)
    mask = SpotMask(P, spot_cfg, allowed)
    res = discover_admg(X, names, cfg, mask=mask, allowed=allowed)
    base = {f.name: getattr(res, f.name) for f in fields(res)}
    n_allowed = -1 if allowed is None else int(np.triu(allowed, 1).sum())
    return SpotResult(**base, t_posterior=t_post, P=P, n_pairs_allowed=n_allowed,
                      mask_summary=mask.summary(), spot_cfg=spot_cfg)
