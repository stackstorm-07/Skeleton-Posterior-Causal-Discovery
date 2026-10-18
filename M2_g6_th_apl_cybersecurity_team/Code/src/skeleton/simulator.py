"""Static simulator: random ancestral ADMGs + linear-Gaussian data (recipe of the SPOT/ABIC papers).

Stage 2a only needs the graph/data generation (used by the optimizer tests and benchmarks);
the true-PAG labelling used to train the skeleton posterior is added in stage 2b.

Recipe (paper Sec. 4.1): Erdos-Renyi DAG, a fraction ``bi_frac`` of its edges turned into bidirected
edges, directed weights ~ +-U[0.5, 2.0], noise covariance off-diagonals ~ +-U[0.4, 0.7] and diagonal
~ U[0.7, 1.2] + sum_j |Omega_ij| (so Omega is diagonally dominant, hence positive definite).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.linalg import expm


def random_admg(d: int, n_edges: int, bi_frac: float, rng: np.random.Generator,
                ancestral: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return (D, B): D[i, j] = 1 for i -> j, B symmetric with B[i, j] = 1 for i <-> j."""
    order = rng.permutation(d)
    pairs = [(order[a], order[b]) for a in range(d) for b in range(a + 1, d)]
    pick = rng.choice(len(pairs), size=min(n_edges, len(pairs)), replace=False)
    D = np.zeros((d, d), dtype=int)
    B = np.zeros((d, d), dtype=int)
    for k in pick:
        i, j = pairs[k]
        if rng.random() < bi_frac:
            B[i, j] = B[j, i] = 1
        else:
            D[i, j] = 1
    if ancestral:
        # a bidirected edge between a node and one of its ancestors makes the ADMG non-ancestral;
        # turn such edges back into directed ones (consistent with the DAG order, so still acyclic).
        reach = expm(D.astype(float)) > 1e-9
        for i in range(d):
            for j in range(i + 1, d):
                if B[i, j] and (reach[i, j] or reach[j, i]):
                    a, b = (i, j) if reach[i, j] else (j, i)
                    B[i, j] = B[j, i] = 0
                    D[a, b] = 1
    return D, B


def simulate_linear_gaussian(D: np.ndarray, B: np.ndarray, n: int, rng: np.random.Generator,
                             standardize: bool = True):
    """Sample n rows from the linear-Gaussian SEM of (D, B).  Returns (X, W1, Omega)."""
    d = D.shape[0]
    sgn = lambda shape: rng.choice([-1.0, 1.0], size=shape)
    W1 = rng.uniform(0.5, 2.0, (d, d)) * sgn((d, d)) * D
    off = np.triu(rng.uniform(0.4, 0.7, (d, d)) * sgn((d, d)), 1)
    off = (off + off.T) * B
    diag = rng.uniform(0.7, 1.2, d) + np.abs(off).sum(axis=1)
    Omega = off + np.diag(diag)
    Minv = np.linalg.inv(np.eye(d) - W1)
    Sigma = Minv.T @ Omega @ Minv
    X = rng.multivariate_normal(np.zeros(d), Sigma, size=n)
    X = X - X.mean(axis=0)
    if standardize:
        X = X / X.std(axis=0)
    return X, W1, Omega


def skeleton_of(D: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = (D != 0) | (D.T != 0) | (B != 0) | (B.T != 0)
    np.fill_diagonal(A, False)
    return A


# ================================================================================================
# Stage 2b: labelled instances for training / evaluating the skeleton posterior
# ================================================================================================
from dataclasses import dataclass, field          # noqa: E402
from typing import Dict, List                      # noqa: E402


@dataclass
class Instance:
    """One simulated dataset with its labels.

    ``skeleton``       : adjacency of the true MAG (maximal ancestral projection of the ADMG) = skeleton of the
                         true PAG.  This is what the benchmark scores against and what the posterior is trained on.
    ``skeleton_admg``  : plain adjacency of the ADMG (D or B edge).  Differs from ``skeleton`` only when the ADMG
                         has an inducing path between non-adjacent nodes (kept for diagnostics).
    """
    X: np.ndarray
    D: np.ndarray
    B: np.ndarray
    skeleton: np.ndarray
    skeleton_admg: np.ndarray
    meta: Dict = field(default_factory=dict)


def mag_skeleton(D: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Skeleton of the MAG of the ADMG (D, B): i - j iff there is an inducing path between i and j."""
    from pgm.graphs import ADMG, mag_projection
    d = D.shape[0]
    names = [str(i) for i in range(d)]
    G = ADMG(names,
             [(names[i], names[j]) for i, j in zip(*np.nonzero(D))],
             [(names[i], names[j]) for i, j in zip(*np.nonzero(np.triu(B)))])
    M = mag_projection(G)
    A = np.zeros((d, d), dtype=bool)
    for a, b in M.di_edges:
        A[int(a), int(b)] = A[int(b), int(a)] = True
    for e in M.bi_edges:
        a, b = tuple(e)
        A[int(a), int(b)] = A[int(b), int(a)] = True
    return A


def _ar1_filter(E: np.ndarray, rho: float) -> np.ndarray:
    """Make the rows of E (n, d) an AR(1) process with unit-variance marginals (rows are time ordered)."""
    if rho == 0.0:
        return E
    out = np.empty_like(E)
    out[0] = E[0]
    s = np.sqrt(1.0 - rho ** 2)
    for t in range(1, E.shape[0]):
        out[t] = rho * out[t - 1] + s * E[t]
    return out


def simulate_instance(d: int, n: int, rng: np.random.Generator, edge_ratio: Optional[float] = None,
                      bi_frac: Optional[float] = None, ar_rho: float = 0.0) -> Instance:
    """One labelled instance.  ``edge_ratio`` = #edges / #nodes (paper: 1.0 to 1.5), ``bi_frac`` = share of
    bidirected edges (paper: 5-15 %).  ``ar_rho`` > 0 makes the rows autocorrelated (PetShop is a time series);
    the data are returned z-scored, which is what the posterior (correlation based) and ABIC see."""
    edge_ratio = float(rng.uniform(1.0, 1.5)) if edge_ratio is None else edge_ratio
    bi_frac = float(rng.uniform(0.0, 0.15)) if bi_frac is None else bi_frac
    D, B = random_admg(d, int(round(edge_ratio * d)), bi_frac, rng)
    if ar_rho > 0.0:
        # autocorrelated noise: sample the (iid) SEM errors with AR(1) dynamics, then push them through the SEM
        sgn = lambda shape: rng.choice([-1.0, 1.0], size=shape)
        W1 = rng.uniform(0.5, 2.0, (d, d)) * sgn((d, d)) * D
        off = np.triu(rng.uniform(0.4, 0.7, (d, d)) * sgn((d, d)), 1)
        off = (off + off.T) * B
        Omega = off + np.diag(rng.uniform(0.7, 1.2, d) + np.abs(off).sum(axis=1))
        E = rng.multivariate_normal(np.zeros(d), Omega, size=n)
        E = _ar1_filter(E, ar_rho)
        X = E @ np.linalg.inv(np.eye(d) - W1)
        X = X - X.mean(axis=0)
        X = X / X.std(axis=0)
    else:
        X, _, _ = simulate_linear_gaussian(D, B, n, rng, standardize=True)
    return Instance(X=X, D=D, B=B, skeleton=mag_skeleton(D, B), skeleton_admg=skeleton_of(D, B),
                    meta=dict(d=d, n=n, edge_ratio=edge_ratio, bi_frac=bi_frac, ar_rho=ar_rho))


def make_instances(num: int, d_range=(20, 40), n_range=(300, 2000), ar_rhos=(0.0,), seed: int = 0) -> List[Instance]:
    """``num`` independent instances; size d uniform in ``d_range`` (inclusive), n log-uniform in ``n_range``,
    ``ar_rho`` drawn uniformly from the list ``ar_rhos``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        d = int(rng.integers(d_range[0], d_range[1] + 1))
        n = int(np.exp(rng.uniform(np.log(n_range[0]), np.log(n_range[1]))))
        rho = float(ar_rhos[rng.integers(len(ar_rhos))])
        out.append(simulate_instance(d, n, rng, ar_rho=rho))
    return out
