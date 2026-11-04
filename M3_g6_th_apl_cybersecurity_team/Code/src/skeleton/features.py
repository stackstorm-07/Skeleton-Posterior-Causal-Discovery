"""Features for the skeleton posterior (ML4S-style, after Ma et al. KDD'22 / the SPOT release).

For a pair (i, j) the evidence for "adjacent in the MAG" is how *strongly* i and j stay dependent when
we condition on subsets of other variables:  a non-adjacent pair has some separating set that drives
the Fisher-z statistic to ~0, an adjacent pair never does.  The features are summaries (moments +
11 percentiles) of the Fisher-z statistic / p-value over conditioning sets of a given size, where the
candidate conditioning variables come from either a Markov-blanket estimate (stage 0) or from the
skeleton predicted by the previous stage of the cascade (stages 1, 2).

Speed: every partial correlation of order 0, 1, 2 is obtained for *all* (i, j, conditioning set) at once
by the standard recursion, so a feature row is just an array gather (``CITable``).

Direction of the tests (the "p > alpha" check):  a small p-value means DEPENDENT.  So in the Markov
blanket a variable is *added* when p <= alpha (dependent) and *removed* when p > alpha (independent), and
a pair survives the order-0 screen when p <= alpha.  ``legacy_inverted=True`` flips all three comparisons;
that is what the released SPOT/ML4S ``gen_feature.py`` appears to do (``prepare_zero_order`` keeps a pair
when ``pval > alpha1``; ``get_markov_blanket`` adds when ``cit > alpha1`` and drops when ``cit <= alpha2``).
It exists only so the effect can be measured (scripts/check_pvalue_direction.py).
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import erfc

PCT = np.linspace(0.0, 100.0, 11)
EMPTY15 = [-1.0] * 15


def summarize(v: np.ndarray) -> List[float]:
    """mean, std, max, min and 11 percentiles (15 numbers); -1 padding when there is nothing to summarise."""
    if v.size == 0:
        return EMPTY15
    return [float(v.mean()), float(v.std()), float(v.max()), float(v.min())] + np.percentile(v, PCT).tolist()


def fisher_sev(r, n: int, m: int):
    """|Fisher z| scaled to a N(0,1) statistic under independence, for partial correlation r given m variables."""
    dof = max(n - m - 3, 1)
    r = np.clip(r, -1 + 1e-9, 1 - 1e-9)
    return np.sqrt(dof) * np.abs(np.arctanh(r))


def pvalue(sev):
    return erfc(np.asarray(sev) / np.sqrt(2.0))      # = 2 (1 - Phi(sev))


class CITable:
    """Fisher-z conditional-independence tests for one dataset, with order-0/1/2 partial correlations tabulated."""

    def __init__(self, X: np.ndarray, orders: int = 2):
        X = np.asarray(X, dtype=float)
        self.n, self.d = X.shape
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        R = np.corrcoef(X.T) if self.d > 1 else np.ones((1, 1))
        R = np.nan_to_num(R)
        np.fill_diagonal(R, 1.0)
        self.R = np.clip(R, -1 + 1e-9, 1 - 1e-9)
        np.fill_diagonal(self.R, 1.0)                     # the clip must not shrink the diagonal (matters for pcorr, m >= 3)
        den = np.sqrt(1.0 - self.R ** 2)
        den[den < 1e-9] = 1e-9
        self.P1 = self.P2 = None
        with np.errstate(all="ignore"):
            if orders >= 1:
                # P1[k, i, j] = corr(i, j | k)
                P1 = (self.R[None, :, :] - self.R[:, :, None] * self.R[:, None, :]) / (den[:, :, None] * den[:, None, :])
                self.P1 = np.clip(np.nan_to_num(P1), -1 + 1e-9, 1 - 1e-9)
            if orders >= 2:
                T = self.P1
                g = np.sqrt(np.maximum(1.0 - T ** 2, 1e-18))                 # (k, i, j)
                Tt, gt = T.transpose(0, 2, 1), g.transpose(0, 2, 1)          # Tt[k, l, i] = T[k, i, l]
                # P2[k, l, i, j] = corr(i, j | k, l)
                P2 = (T[:, None, :, :] - Tt[:, :, :, None] * Tt[:, :, None, :]) / (gt[:, :, :, None] * gt[:, :, None, :])
                self.P2 = np.clip(np.nan_to_num(P2), -1 + 1e-9, 1 - 1e-9)
        # lag-1 autocorrelation of the rows (meaningful only if rows are time ordered, as in PetShop)
        Xc = (X - X.mean(0)) / sd
        self.ac1 = float(np.mean(np.sum(Xc[1:] * Xc[:-1], axis=0) / max(self.n - 1, 1))) if self.n > 2 else 0.0
        self._cache: Dict[Tuple[int, int, FrozenSet[int]], float] = {}

    # -- arbitrary-size conditioning (used by the Markov-blanket search) -----------------------------
    def pcorr(self, i: int, j: int, Z: Sequence[int]) -> float:
        Z = list(Z)
        m = len(Z)
        if m == 0:
            return float(self.R[i, j])
        if m == 1 and self.P1 is not None:
            return float(self.P1[Z[0], i, j])
        if m == 2 and self.P2 is not None:
            return float(self.P2[Z[0], Z[1], i, j])
        v = [i, j] + Z
        sub = self.R[np.ix_(v, v)]
        try:
            inv = np.linalg.inv(sub)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(sub)
        den = inv[0, 0] * inv[1, 1]
        return float(-inv[0, 1] / np.sqrt(den)) if den > 0 else 0.0

    def pval(self, i: int, j: int, Z: Sequence[int] = ()) -> float:
        key = (i, j, frozenset(Z)) if i < j else (j, i, frozenset(Z))
        p = self._cache.get(key)
        if p is None:
            m = len(Z)
            if m >= self.n - 3:
                p = 1.0
            else:
                p = float(pvalue(fisher_sev(self.pcorr(i, j, Z), self.n, m)))
            self._cache[key] = p
        return p

    # -- vectorised gathers for the feature rows ------------------------------------------------------
    def sev_o0(self, i: int, j: int) -> float:
        return float(fisher_sev(self.R[i, j], self.n, 0))

    def sev_o1(self, i: int, j: int, C: Sequence[int]) -> np.ndarray:
        if len(C) == 0:
            return np.empty(0)
        return fisher_sev(self.P1[np.asarray(C), i, j], self.n, 1)

    def sev_o2(self, i: int, j: int, C: Sequence[int], max_sub: int = 300) -> np.ndarray:
        C = np.asarray(C)
        if len(C) < 2:
            return np.empty(0)
        a, b = np.triu_indices(len(C), 1)
        if len(a) > max_sub:                                   # deterministic thinning, keeps the spread
            sel = np.linspace(0, len(a) - 1, max_sub).astype(int)
            a, b = a[sel], b[sel]
        return fisher_sev(self.P2[C[a], C[b], i, j], self.n, 2)


# ------------------------------------------------------------------------------------------------
# Markov blanket (grow-shrink)
# ------------------------------------------------------------------------------------------------
def markov_blanket(ci: CITable, alpha_grow: float = 0.05, alpha_shrink: float = 0.05,
                   legacy_inverted: bool = False, max_size: int = 12) -> List[List[int]]:
    """Grow-shrink Markov-blanket estimate for every variable (a superset-ish of its neighbours).

    grow:   add j while  i and j are DEPENDENT given the current blanket   (p <= alpha_grow)
    shrink: drop j when  i and j are INDEPENDENT given the rest of blanket (p >  alpha_shrink)
    ``legacy_inverted`` flips both comparisons (see the module docstring).  ``max_size`` bounds the blanket
    (needed because the inverted variant grows to the whole graph).
    """
    d = ci.d
    out = []
    for i in range(d):
        mb: List[int] = []
        changed = True
        while changed and len(mb) < max_size:
            changed = False
            for j in range(d):
                if j == i or j in mb:
                    continue
                p = ci.pval(i, j, mb)
                if (p > alpha_grow) if legacy_inverted else (p <= alpha_grow):
                    mb.append(j)
                    changed = True
                    break
        changed = True
        while changed and mb:
            changed = False
            for j in list(mb):
                rest = [k for k in mb if k != j]
                p = ci.pval(i, j, rest)
                if (p <= alpha_shrink) if legacy_inverted else (p > alpha_shrink):
                    mb.remove(j)
                    changed = True
                    break
        out.append(mb)
    return out


def zero_order_adjacency(ci: CITable, alpha: float, legacy_inverted: bool = False) -> np.ndarray:
    """Order-0 screen: pairs kept as possible edges.  Correct: marginally DEPENDENT (p <= alpha).
    Legacy (released ``prepare_zero_order``): p > alpha, i.e. the marginally INDEPENDENT pairs."""
    d = ci.d
    A = np.zeros((d, d), dtype=bool)
    for i in range(d):
        for j in range(i + 1, d):
            p = pvalue(fisher_sev(ci.R[i, j], ci.n, 0))
            keep = (p > alpha) if legacy_inverted else (p <= alpha)
            A[i, j] = A[j, i] = bool(keep)
    return A


# ------------------------------------------------------------------------------------------------
# Feature rows
# ------------------------------------------------------------------------------------------------
def _base(ci: CITable, i: int, j: int) -> List[float]:
    s0 = ci.sev_o0(i, j)
    return [s0, float(pvalue(s0)), abs(float(ci.R[i, j])), float(np.log(ci.n)), float(ci.d), ci.ac1]


def _order_block(ci: CITable, i: int, j: int, C: Sequence[int], order: int, max_sub: int) -> List[float]:
    sev = ci.sev_o1(i, j, C) if order == 1 else ci.sev_o2(i, j, C, max_sub)
    return summarize(sev) + summarize(pvalue(sev) if sev.size else sev) + [float(sev.size)]


def stage0_features(ci: CITable, pairs, mb: List[List[int]], max_sub: int = 300) -> np.ndarray:
    """Stage 0 (no earlier prediction): base + blanket membership/size + order-1 and order-2 summaries over
    the conditioning candidates MB(i) u MB(j) minus {i, j}."""
    mbs = [set(m) for m in mb]
    rows = []
    for i, j in pairs:
        C = sorted((mbs[i] | mbs[j]) - {i, j})
        rows.append(_base(ci, i, j)
                    + [float(j in mbs[i]), float(i in mbs[j]), float(len(mbs[i])), float(len(mbs[j])), float(len(C))]
                    + _order_block(ci, i, j, C, 1, max_sub) + _order_block(ci, i, j, C, 2, max_sub))
    return np.asarray(rows, dtype=float).reshape(len(pairs), -1)


def neighbours(adj: np.ndarray) -> List[List[int]]:
    return [list(np.nonzero(adj[i])[0]) for i in range(adj.shape[0])]


def stage_features(ci: CITable, pairs, stage: int, nbrs: List[List[int]], prev: Sequence[np.ndarray],
                   max_sub: int = 300) -> np.ndarray:
    """Stages 1 and 2: summaries over conditioning subsets of size ``stage`` drawn from the neighbours that the
    previous stage predicted for i and j, plus the previous stages' probabilities for the pair
    (``prev`` = list of d x d probability matrices, one per earlier stage)."""
    nb = [set(n) for n in nbrs]
    rows = []
    for i, j in pairs:
        C = sorted((nb[i] | nb[j]) - {i, j})
        rows.append(_base(ci, i, j) + [float(P[i, j]) for P in prev] + [float(len(C))]
                    + _order_block(ci, i, j, C, stage, max_sub))
    return np.asarray(rows, dtype=float).reshape(len(pairs), -1)


def all_pairs(d: int):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]
