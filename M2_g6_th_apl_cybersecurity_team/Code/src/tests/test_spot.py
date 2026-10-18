"""SPOT = shared optimizer + skeleton-posterior mask (Eq. 9).  Tests pin down the exact semantics."""
import numpy as np

from pgm.abic import run_abic
from pgm.dcd import DCDConfig, discover_admg
from pgm.spot import SpotConfig, SpotMask, acceptance_probability, allowed_pairs, run_spot
from skeleton.posterior import oracle_posterior, random_posterior
from skeleton.simulator import random_admg, simulate_linear_gaussian, skeleton_of


def _data(seed, d=6, n=800, std=False, bi=0.15):
    rng = np.random.default_rng(seed)
    D, B = random_admg(d, int(1.2 * d), bi, rng)
    X, _, _ = simulate_linear_gaussian(D, B, n, rng, standardize=std)
    return D, B, X, [f"X{i}" for i in range(d)]


# ---------------------------------------------------------------- the acceptance rule of Eq. 9
def test_acceptance_probability_matches_equation_9():
    cfg = SpotConfig(c=0.1)
    P = np.array([0.0, 0.2, 0.9, 1.0])
    for t in (1, 3, 10, 20):
        got = acceptance_probability(P, t, cfg)
        want = np.minimum((P + 0.1) ** (1.0 / (t + 1)), 1.0)
        assert np.allclose(got, want)
    # t = 1, p = 0 -> sqrt(0.1); grows with t; p + c >= 1 is always accepted
    assert np.isclose(acceptance_probability(np.array([0.0]), 1, cfg)[0], 0.1 ** 0.5)
    assert acceptance_probability(np.array([0.9, 1.0]), 5, cfg).min() == 1.0
    assert acceptance_probability(np.array([0.0]), 20, cfg)[0] > acceptance_probability(np.array([0.0]), 1, cfg)[0]


def test_weakening_always_kept_strengthening_stochastic():
    d = 4
    P = np.zeros((d, d))                               # posterior says "no edge" everywhere, c = 0.1 only
    m = SpotMask(P, SpotConfig(c=0.1, mask_seed=0))
    K = 2 * d * d
    theta = np.where(np.arange(K) % 2 == 0, 0.7, -0.7)
    g_weak = theta * 2.0                               # theta * g > 0  -> moving towards 0
    g_strong = -theta * 2.0                            # theta * g < 0  -> moving away from 0
    for _ in range(50):
        assert np.array_equal(m(g_weak, theta, 1), g_weak)           # always kept, even though p = 0
    kept = np.zeros(K)
    reps = 4000
    for _ in range(reps):
        kept += (m(g_strong, theta, 1) != 0)
    frac = kept / reps
    assert abs(frac.mean() - 0.1 ** 0.5) < 0.02, frac.mean()          # (0 + 0.1)^(1/2) at t = 1
    # a coordinate sitting exactly at 0 counts as strengthening, never as weakening
    theta0 = np.zeros(K)
    out = np.mean([m(np.ones(K), theta0, 1) != 0 for _ in range(2000)])
    assert abs(out - 0.1 ** 0.5) < 0.02


def test_acceptance_frequency_follows_p_and_t():
    d = 3
    P = np.array([[0, .8, .3], [.8, 0, .0], [.3, .0, 0]], dtype=float)
    K = 2 * d * d
    theta = np.full(K, 0.5)
    g = -np.ones(K)                                    # all strengthening
    for t in (1, 4, 12):
        m = SpotMask(P, SpotConfig(c=0.1, mask_seed=t))
        reps = 6000
        kept = sum((m(g, theta, t) != 0).astype(float) for _ in range(reps)) / reps
        want = acceptance_probability(np.concatenate([P.ravel(), P.ravel()]), t, m.cfg)
        assert np.abs(kept - want).max() < 0.03, (t, np.abs(kept - want).max())


def test_pair_to_coordinate_mapping():
    """p[i, j] governs W1[i, j], W1[j, i] and the bidirected coordinate of the pair (lower triangle)."""
    d = 5
    P = np.zeros((d, d))
    P[1, 3] = P[3, 1] = 1.0                            # only the pair {1, 3} is believed adjacent
    cfg = SpotConfig(c=0.0)                            # c = 0: p = 0 pairs are rejected with certainty at large t
    m = SpotMask(P, cfg)
    theta = np.full(2 * d * d, 0.5)
    g = -np.ones(2 * d * d)
    ok = np.zeros(2 * d * d, dtype=bool)
    for _ in range(20):
        ok |= (m(g, theta, 1) != 0)
    never = ~ok
    W1_kept = ok[: d * d].reshape(d, d)
    L_kept = ok[d * d:].reshape(d, d)
    L_low = np.tril(L_kept, -1)                                        # only the strictly lower triangle is free
    assert W1_kept[1, 3] and W1_kept[3, 1] and L_low[3, 1]             # lower triangle: row index > col index
    assert W1_kept.sum() == 2 and L_low.sum() == 1
    assert (~ok[d * d:].reshape(d, d)[np.tril_indices(d, -1)]).sum() == d * (d - 1) // 2 - 1


def test_p_one_is_identity_on_any_gradient():
    d = 4
    P = np.ones((d, d))
    m = SpotMask(P, SpotConfig(c=0.1))
    rng = np.random.default_rng(0)
    for _ in range(20):
        theta, g = rng.normal(size=2 * d * d), rng.normal(size=2 * d * d)
        assert np.array_equal(m(g, theta, 1), g)


def test_mask_stats_and_reproducibility():
    d = 4
    P = np.full((d, d), 0.3)
    theta = np.random.default_rng(1).normal(size=2 * d * d)
    g = np.random.default_rng(2).normal(size=2 * d * d)
    a, b = SpotMask(P, SpotConfig(mask_seed=5)), SpotMask(P, SpotConfig(mask_seed=5))
    for s in (1, 1, 2):
        assert np.array_equal(a(g, theta, s), b(g, theta, s))
    summ = a.summary()
    assert set(summ) == {1, 2} and 0.0 <= summ[1]["kept_fraction"] <= 1.0
    assert summ[1]["weakening"] + summ[1]["strengthening_proposals"] == 2 * a.free.sum()


# ---------------------------------------------------------------- the hard floor
def test_floor_defines_allowed_pairs_and_pins_coordinates():
    P = np.array([[0, .9, .0005, .5], [.9, 0, .2, .001], [.0005, .2, 0, .0], [.5, .001, .0, 0]])
    A = allowed_pairs(P, SpotConfig(floor=1e-3))
    assert A[0, 1] and A[1, 2] and A[0, 3]
    assert not A[0, 2] and not A[1, 3] and not A[2, 3]                 # p <= floor is pinned (released code: <=)
    assert A.diagonal().sum() == 0 and np.array_equal(A, A.T)
    assert allowed_pairs(P, SpotConfig(floor=None)) is None


# ---------------------------------------------------------------- the whole optimizer
def test_spot_with_p_one_and_no_floor_equals_abic_exactly():
    """If the posterior accepts everything, SPOT must reproduce ABIC bit for bit (same seed)."""
    D, B, X, names = _data(3, d=5)
    cfg = DCDConfig(seed=4, max_outer=4)
    a = run_abic(X, names, cfg)
    s = run_spot(X, names, P=np.ones((5, 5)), cfg=cfg, spot_cfg=SpotConfig(floor=None))
    assert np.array_equal(a.W1_raw, s.W1_raw) and np.array_equal(a.W2_raw, s.W2_raw)
    assert a.n_fev == s.n_fev and a.admg.di_edges == s.admg.di_edges


def test_oracle_spot_respects_skeleton_and_is_cheaper_than_abic():
    D, B, X, names = _data(1, d=8)
    sk = skeleton_of(D, B)
    cfg = DCDConfig(seed=1)
    abic = run_abic(X, names, cfg)
    s = run_spot(X, names, P=oracle_posterior(sk), cfg=cfg)
    assert not ((np.abs(s.W1_raw) > 0) & ~sk).any()                    # hard floor pins non-edges
    assert s.n_pairs_allowed == int(np.triu(sk, 1).sum())
    assert s.n_fev < abic.n_fev


def test_spot_is_deterministic_given_seeds():
    D, B, X, names = _data(5, d=6)
    P = np.clip(skeleton_of(D, B) * 0.7 + 0.1, 0, 1)
    np.fill_diagonal(P, 0)
    kw = dict(P=P, cfg=DCDConfig(seed=2, max_outer=4), spot_cfg=SpotConfig(mask_seed=9))
    a, b = run_spot(X, names, **kw), run_spot(X, names, **kw)
    assert np.array_equal(a.W1_raw, b.W1_raw) and a.admg.di_edges == b.admg.di_edges
    c = run_spot(X, names, P=P, cfg=DCDConfig(seed=2, max_outer=4), spot_cfg=SpotConfig(mask_seed=10))
    assert not np.array_equal(a.W1_raw, c.W1_raw)                      # the mask's own stream matters


def test_spot_output_is_acyclic_and_budget_is_respected():
    from pgm.graphs import has_directed_cycle
    D, B, X, names = _data(2, d=8)
    P = np.clip(skeleton_of(D, B) * 0.8 + 0.05, 0, 1)
    np.fill_diagonal(P, 0)
    res = run_spot(X, names, P=P, cfg=DCDConfig(num_restarts=2, time_budget=0.8))
    assert res.timed_out and res.t_optimize < 1.2
    assert not has_directed_cycle(res.W1)


def test_posterior_time_is_charged_to_the_budget():
    class Fake:
        def predict(self, X):
            import time as _t
            _t.sleep(0.3)
            class R: pass
            r = R(); r.P = np.full((X.shape[1],) * 2, 0.5)
            return r
    D, B, X, names = _data(2, d=6)
    res = run_spot(X, names, posterior=Fake(), cfg=DCDConfig(num_restarts=1, time_budget=1.0))
    assert res.t_posterior >= 0.3 and res.t_optimize <= 1.0 - 0.3 + 0.35
    res2 = run_spot(X, names, posterior=Fake(), cfg=DCDConfig(num_restarts=1, time_budget=1.0),
                    spot_cfg=SpotConfig(charge_posterior=False))
    assert res2.t_posterior >= 0.3


def test_random_posterior_ablation_runs_and_released_variant_runs():
    D, B, X, names = _data(6, d=6)
    sk = skeleton_of(D, B)
    P = np.clip(sk * 0.8 + 0.05, 0, 1)
    np.fill_diagonal(P, 0)
    Pr = random_posterior(P, np.random.default_rng(0))
    cfg = DCDConfig(seed=0, max_outer=5)
    for Q, sc in [(Pr, SpotConfig()), (P, SpotConfig(variant="released_stochastic", floor=1e-6))]:
        r = run_spot(X, names, P=Q, cfg=cfg, spot_cfg=sc)
        assert r.admg is not None and np.isfinite(r.bic)
