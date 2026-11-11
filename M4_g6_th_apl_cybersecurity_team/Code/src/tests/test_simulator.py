"""Static simulator: graphs are ancestral / acyclic, covariance matches the SEM, skeleton helper."""
import numpy as np
from scipy.linalg import expm

from skeleton.simulator import random_admg, simulate_linear_gaussian, skeleton_of
from pgm.graphs import has_directed_cycle


def test_random_admg_is_ancestral_and_acyclic():
    rng = np.random.default_rng(0)
    for _ in range(30):
        D, B = random_admg(15, 20, 0.3, rng)
        assert not has_directed_cycle(D)
        assert (B == B.T).all() and not np.diag(B).any()
        reach = expm(D.astype(float)) > 1e-9
        assert not (B.astype(bool) & (reach | reach.T)).any()      # no bidirected edge between an ancestor pair


def test_simulated_covariance_matches_model():
    rng = np.random.default_rng(1)
    D, B = random_admg(6, 8, 0.3, rng)
    X, W1, Om = simulate_linear_gaussian(D, B, 200000, rng, standardize=False)
    Minv = np.linalg.inv(np.eye(6) - W1)
    Sigma = Minv.T @ Om @ Minv
    assert np.abs(np.cov(X.T) - Sigma).max() < 0.1 * np.abs(Sigma).max()
    assert (np.linalg.eigvalsh(Om) > 0).all()
    # structure: W1 / Omega are supported exactly on D / B
    assert ((W1 != 0) == (D != 0)).all()
    assert (((Om - np.diag(np.diag(Om))) != 0) == (B != 0)).all()


def test_skeleton_of():
    D = np.zeros((3, 3), int); D[0, 1] = 1
    B = np.zeros((3, 3), int); B[1, 2] = B[2, 1] = 1
    A = skeleton_of(D, B)
    assert A[0, 1] and A[1, 0] and A[1, 2] and A[2, 1] and not A[0, 2] and not A.diagonal().any()
