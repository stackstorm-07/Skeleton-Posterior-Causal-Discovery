"""Stage 2b: simulator labels, CI tables, p-value direction, skeleton posterior (train / predict / save / load)."""
import os
import tempfile

import numpy as np

from skeleton.simulator import (_ar1_filter, make_instances, mag_skeleton, random_admg, simulate_instance,
                                skeleton_of)
from skeleton.features import (CITable, all_pairs, markov_blanket, pvalue, fisher_sev, stage0_features,
                               zero_order_adjacency)
from skeleton.posterior import (PosteriorConfig, SkeletonPosterior, evaluate, oracle_posterior, posterior_metrics,
                                random_posterior)


# ----------------------------------------------------------------------------- simulator labels
def test_mag_skeleton_adds_inducing_path_edges():
    # found by search: a chain of bidirected edges makes the MAG adjacent where the ADMG is not
    D = np.zeros((5, 5), int); D[2, 0] = 1; D[3, 1] = 1
    B = np.zeros((5, 5), int)
    for a, b in [(0, 3), (0, 4), (1, 2), (2, 3), (3, 4)]:
        B[a, b] = B[b, a] = 1
    S, A = mag_skeleton(D, B), skeleton_of(D, B)
    assert (S >= A).all() and (S != A).any()
    assert (S == S.T).all() and not S.diagonal().any()


def test_instance_labels_and_shapes():
    ins = make_instances(5, d_range=(10, 12), n_range=(200, 300), seed=0)
    for i in ins:
        d = i.X.shape[1]
        assert 10 <= d <= 12 and 200 <= i.X.shape[0] <= 300
        assert i.skeleton.shape == (d, d) and (i.skeleton == i.skeleton.T).all() and not i.skeleton.diagonal().any()
        assert (i.skeleton >= i.skeleton_admg).all()
        assert np.allclose(i.X.mean(0), 0, atol=1e-8) and np.allclose(i.X.std(0), 1, atol=1e-8)


def test_make_instances_deterministic_and_disjoint_seeds():
    a = make_instances(3, d_range=(10, 10), seed=5)
    b = make_instances(3, d_range=(10, 10), seed=5)
    c = make_instances(3, d_range=(10, 10), seed=6)
    assert all(np.array_equal(x.X, y.X) for x, y in zip(a, b))
    assert not np.array_equal(a[0].X, c[0].X)


def test_ar1_filter_has_requested_autocorrelation():
    rng = np.random.default_rng(0)
    E = _ar1_filter(rng.standard_normal((20000, 3)), 0.6)
    ac = np.mean(np.sum(E[1:] * E[:-1], 0) / np.sum(E ** 2, 0))
    assert abs(ac - 0.6) < 0.03 and abs(E.std() - 1) < 0.05
    ins = simulate_instance(8, 3000, np.random.default_rng(1), ar_rho=0.6)
    assert CITable(ins.X).ac1 > 0.2


# ----------------------------------------------------------------------------- CI tables
def test_citable_matches_direct_partial_correlation():
    ins = make_instances(1, d_range=(12, 12), n_range=(500, 500), seed=2)[0]
    ci, R = CITable(ins.X), np.corrcoef(ins.X.T)

    def pc(a, b, Z):
        v = [a, b] + list(Z)
        inv = np.linalg.inv(R[np.ix_(v, v)])
        return -inv[0, 1] / np.sqrt(inv[0, 0] * inv[1, 1])
    rng = np.random.default_rng(0)
    for _ in range(60):
        a, b, k, l, m = rng.choice(12, 5, replace=False)        # five distinct variables
        assert abs(ci.P1[k, a, b] - pc(a, b, [k])) < 1e-9
        assert abs(ci.P2[k, l, a, b] - pc(a, b, [k, l])) < 1e-9
        assert abs(ci.pcorr(a, b, [k, l, m]) - pc(a, b, [k, l, m])) < 1e-9     # general-order path (Markov blanket)


# ----------------------------------------------------------------------------- p-value direction
def test_small_p_means_dependent():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    X = np.c_[x, x + 0.5 * rng.standard_normal(2000), rng.standard_normal(2000)]
    ci = CITable(X)
    assert ci.pval(0, 1) < 1e-6 and ci.pval(0, 2) > 0.01
    assert pvalue(fisher_sev(0.0, 100, 0)) > 0.99


def test_order0_screen_keeps_true_edges_and_legacy_does_not():
    ins = make_instances(8, d_range=(20, 25), n_range=(800, 1000), seed=3)
    kept, legacy, total = 0, 0, 0
    for i in ins:
        ci = CITable(i.X, orders=0)
        iu = np.triu_indices(i.X.shape[1], 1)
        s = i.skeleton[iu]
        kept += (zero_order_adjacency(ci, 0.15)[iu] & s).sum()
        legacy += (zero_order_adjacency(ci, 0.15, legacy_inverted=True)[iu] & s).sum()
        total += s.sum()
    assert kept / total > 0.9           # correct direction: almost every true edge survives the screen
    assert legacy / total < 0.2         # released direction: almost every true edge is thrown away


def test_markov_blanket_direction():
    # chain 0 -> 1 -> 2 -> 3: blanket of node 1 must contain its neighbours and not be empty
    rng = np.random.default_rng(0)
    n = 3000
    x0 = rng.standard_normal(n)
    x1 = x0 + 0.5 * rng.standard_normal(n)
    x2 = x1 + 0.5 * rng.standard_normal(n)
    x3 = x2 + 0.5 * rng.standard_normal(n)
    ci = CITable(np.c_[x0, x1, x2, x3])
    mb = markov_blanket(ci, 0.05, 0.05)
    assert {0, 2} <= set(mb[1]) and 3 not in mb[1]
    assert markov_blanket(ci, 0.05, 0.05, legacy_inverted=True)[1] != mb[1]


# ----------------------------------------------------------------------------- features
def test_feature_rows_have_fixed_width():
    ins = make_instances(2, d_range=(10, 14), n_range=(300, 500), seed=4)
    widths = set()
    for i in ins:
        ci = CITable(i.X)
        F = stage0_features(ci, all_pairs(i.X.shape[1]), markov_blanket(ci))
        assert np.isfinite(F).all()
        widths.add(F.shape[1])
    assert len(widths) == 1


# ----------------------------------------------------------------------------- posterior
def _small(seed, n):
    return make_instances(n, d_range=(12, 16), n_range=(400, 800), ar_rhos=(0.0, 0.4), seed=seed)


def _small_cfg(**kw):
    return PosteriorConfig(max_iter=60, cv_folds=2, min_samples_leaf=20, seed=0, **kw)


def test_posterior_predict_shape_symmetry_and_range():
    post = SkeletonPosterior(_small_cfg()).fit(_small(0, 12))
    X = _small(100, 1)[0].X
    r = post.predict(X)
    d = X.shape[1]
    assert r.P.shape == (d, d) and np.allclose(r.P, r.P.T) and not r.P.diagonal().any()
    assert r.P.min() >= 0 and r.P.max() <= 1
    assert len(r.stage_P) == 3 and r.t_total > 0 and r.n_scored[0] == d * (d - 1) // 2


def test_posterior_save_load_roundtrip_and_determinism():
    train = _small(1, 12)
    post = SkeletonPosterior(_small_cfg()).fit(train)
    X = _small(101, 1)[0].X
    P1 = post.predict(X).P
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "p.pkl")
        post.save(path)
        P2 = SkeletonPosterior.load(path).predict(X).P
    assert np.array_equal(P1, P2)
    assert np.array_equal(P1, SkeletonPosterior(_small_cfg()).fit(train).predict(X).P)      # same seed, same model


def test_posterior_heldout_beats_chance_by_far():
    post = SkeletonPosterior(_small_cfg()).fit(_small(2, 60))
    rows = evaluate(post, _small(900, 10))
    assert np.mean([r["auroc"] for r in rows]) > 0.93
    assert np.mean([r["auprc"] for r in rows]) > 0.7
    assert np.mean([r["recall_p10"] for r in rows]) > 0.8       # recall-oriented, as the paper asks
test_posterior_heldout_beats_chance_by_far.SLOW = True


def test_posterior_metrics_known_cases():
    S = np.zeros((6, 6), bool); S[0, 1] = S[1, 0] = S[2, 3] = S[3, 2] = True
    m = posterior_metrics(oracle_posterior(S, smooth=0.0), S)
    assert m["auroc"] == 1.0 and m["auprc"] == 1.0 and m["recall"] == 1.0 and m["precision"] == 1.0
    P = np.full((6, 6), 0.5); np.fill_diagonal(P, 0)
    assert abs(posterior_metrics(P, S)["auroc"] - 0.5) < 1e-9


def test_random_posterior_is_uninformative_but_same_profile():
    ins = _small(3, 1)[0]
    post = SkeletonPosterior(_small_cfg()).fit(_small(4, 12))
    P = post.predict(ins.X).P
    Q = random_posterior(P, np.random.default_rng(0))
    iu = np.triu_indices(P.shape[0], 1)
    assert np.allclose(np.sort(P[iu]), np.sort(Q[iu])) and np.allclose(Q, Q.T) and not Q.diagonal().any()
