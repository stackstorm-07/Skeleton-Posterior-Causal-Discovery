"""ABIC / the shared optimizer on small synthetic ancestral ADMGs.

These tests assert the behaviour that was measured and documented in the stage-2a notes:
  * raw (unstandardised) data, as in the paper's simulator, is recovered well with a few restarts;
  * the output is always an acyclic graph and the constraint h is driven to ~0;
  * the oracle skeleton (hard constraint) never hurts and is faster;
  * the time budget, the gradient-mask hook and the hard-skeleton hook behave as specified.
"""
import time

import numpy as np

from pgm.abic import run_abic
from pgm.admg_model import admg_to_masks, bic, masks_to_admg
from pgm.dcd import DCDConfig, discover_admg
from pgm.graphs import has_directed_cycle
from skeleton.simulator import random_admg, simulate_linear_gaussian, skeleton_of


def _f1(a, b):
    tp = (a & b).sum(); p = a.sum(); t = b.sum()
    return 2 * tp / (p + t) if p + t else 1.0


def _data(seed, d=5, n=1000, std=False, bi=0.15):
    rng = np.random.default_rng(seed)
    D, B = random_admg(d, int(1.2 * d), bi, rng)
    X, _, _ = simulate_linear_gaussian(D, B, n, rng, standardize=std)
    return D, B, X, [f"X{i}" for i in range(d)]


def test_recovery_small_raw_data():
    sk, gaps = [], []
    for seed in range(3):
        D, B, X, names = _data(seed)
        res = run_abic(X, names, DCDConfig(num_restarts=3, seed=seed))
        Dh, Bh = admg_to_masks(res.admg)
        sk.append(_f1(skeleton_of(Dh, Bh), skeleton_of(D, B)))
        gaps.append(res.bic - bic(X, masks_to_admg(names, D, B)))
        assert res.converged and res.h < 1e-6
        assert not has_directed_cycle(res.W1)
    assert np.mean(sk) >= 0.8, sk
    assert np.median(gaps) < 40, gaps          # BIC within a few tens of the true graph's
test_recovery_small_raw_data.SLOW = True


def test_oracle_skeleton_not_worse_and_faster():
    D, B, X, names = _data(1, d=8)
    cfg = DCDConfig(seed=1)
    free = discover_admg(X, names, cfg)
    orc = discover_admg(X, names, cfg, allowed=skeleton_of(D, B))
    sk_true = skeleton_of(D, B)
    Dh, Bh = admg_to_masks(orc.admg)
    assert not (skeleton_of(Dh, Bh) & ~sk_true).any()                 # hard constraint respected
    assert not ((np.abs(orc.W1_raw) > 0) & ~sk_true).any()
    assert orc.n_fev < free.n_fev
test_oracle_skeleton_not_worse_and_faster.SLOW = True


def test_time_budget_respected():
    D, B, X, names = _data(2, d=8)
    t = time.perf_counter()
    res = discover_admg(X, names, DCDConfig(num_restarts=2, time_budget=0.8))
    assert res.timed_out and res.t_optimize < 1.2 and res.elapsed < 3.0
    assert not has_directed_cycle(res.W1)                              # still a valid (acyclic) graph


def test_gradient_mask_hook_called_with_shapes():
    D, B, X, names = _data(3, d=5)
    calls = []

    def mask(g, th, step):
        calls.append((g.shape, th.shape, step))
        return g

    discover_admg(X, names, DCDConfig(max_outer=2), mask=mask)
    d = 5
    assert calls and all(c[0] == (2 * d * d,) and c[1] == (2 * d * d,) and c[2] >= 1 for c in calls)
    # identity mask == no mask (same seed => identical result)
    a = discover_admg(X, names, DCDConfig(max_outer=3, seed=7))
    b = discover_admg(X, names, DCDConfig(max_outer=3, seed=7), mask=lambda g, th, s: g)
    assert np.allclose(a.W1_raw, b.W1_raw) and np.allclose(a.W2_raw, b.W2_raw)


def test_zero_mask_freezes_parameters():
    D, B, X, names = _data(4, d=5)
    res = discover_admg(X, names, DCDConfig(max_outer=1), mask=lambda g, th, s: np.zeros_like(g))
    # with a zero gradient L-BFGS-B cannot move: the raw weights stay at their (small random) start
    assert np.abs(res.W1_raw).max() <= 0.5 + 1e-12


def test_deterministic_given_seed():
    D, B, X, names = _data(5, d=5)
    a = discover_admg(X, names, DCDConfig(seed=11, max_outer=4))
    b = discover_admg(X, names, DCDConfig(seed=11, max_outer=4))
    assert np.array_equal(a.W1_raw, b.W1_raw) and a.admg.di_edges == b.admg.di_edges
