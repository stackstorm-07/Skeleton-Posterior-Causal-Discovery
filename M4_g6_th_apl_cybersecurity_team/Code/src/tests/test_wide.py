"""M4 WP1: wide-d training set, negative subsampling, seed namespaces."""
import numpy as np

from skeleton import wide
from skeleton.posterior import PosteriorConfig, SkeletonPosterior, _subsample


def test_seed_namespaces_are_disjoint_and_asserted():
    tr = {wide.seed_for("train", b, i) for b in wide._BUCKET_IDX for i in (0, 50, 999)}
    ca = {wide.seed_for("calib", b, i) for b in wide._BUCKET_IDX for i in (0, 50, 999)}
    te = {wide.seed_for("test", b, i) for b in wide._BUCKET_IDX for i in (0, 50, 999)}
    assert max(tr) < 10_000 <= min(ca) and max(ca) < 20_000 <= min(te)
    try:
        wide.seed_for("train", "d200", 1000)
        raise RuntimeError("expected an AssertionError")
    except AssertionError:
        pass


def test_instances_are_reproducible_and_in_range():
    a = wide.one_instance("train", "d60", 3, 60, 60); b = wide.one_instance("train", "d60", 3, 60, 60)
    assert a.X.shape == (a.meta["n"], 60) and np.array_equal(a.X, b.X) and np.array_equal(a.skeleton, b.skeleton)
    assert 300 <= a.meta["n"] <= 2000


def test_subsample_keeps_all_positives_and_reweights_negatives():
    rng = np.random.default_rng(0)
    y = np.array([1] * 5 + [0] * 1000); F = np.arange(len(y), dtype=float)[:, None]
    X, yy, w = _subsample(F, y, 100, rng)
    assert yy.sum() == 5 and (yy == 0).sum() == 100 and np.isclose(w[yy == 0].sum(), 1000) and (w[yy == 1] == 1).all()
    assert _subsample(F, y, None, rng)[2] is None and _subsample(F, y, 5000, rng)[2] is None


def test_default_config_is_the_m3_behaviour_and_capped_fit_runs():
    assert PosteriorConfig().max_neg_per_graph is None
    insts = [it for _, it in wide.wide_instances("train", {"t": (12, 14, 6)})]
    p = SkeletonPosterior(PosteriorConfig(max_iter=20, max_neg_per_graph=20)).fit(insts)
    P = p.predict(insts[0].X).P
    assert P.shape == (insts[0].X.shape[1],) * 2 and np.isfinite(P).all() and ((P >= 0) & (P <= 1)).all()
