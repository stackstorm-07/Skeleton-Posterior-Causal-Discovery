"""M3 item 5: directed-only learner (E10) and the FCI / RFCI baselines through Tetrad (E9)."""
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

from pgm.dcd import DCDConfig, Problem, discover_admg                      # noqa: E402
from pgm.spot import SpotConfig, run_spot                                    # noqa: E402
from skeleton.posterior import oracle_posterior                              # noqa: E402
from skeleton.simulator import random_admg, simulate_linear_gaussian         # noqa: E402


def _need_tetrad():
    from pgm.graphs import find_tetrad_jar
    try:
        find_tetrad_jar()
    except Exception as e:
        raise __import__("unittest").SkipTest(str(e))


def _data(seed, d=6, n=800, bi=0.3):
    rng = np.random.default_rng(seed)
    D, B = random_admg(d, int(1.4 * d), bi, rng)
    X, _, _ = simulate_linear_gaussian(D, B, n, rng, standardize=True)
    return D, B, X, [f"X{i}" for i in range(d)]


# ------------------------------------------------------------------------------ E10: directed-only learner
def test_directed_only_pins_the_bidirected_block_and_nothing_else():
    X = np.random.default_rng(0).normal(size=(100, 5))
    p0, p1 = Problem(X, DCDConfig()), Problem(X, DCDConfig(directed_only=True))
    d2 = 25
    assert (p0.free == p1.free)[:d2].all()                       # W1 coordinates untouched
    assert p0.free[d2:].sum() == 10 and p1.free[d2:].sum() == 0  # the 10 lower-triangular W2 coordinates are pinned
    assert all(b == (0.0, 0.0) for b in p1.bounds[d2:]) and p1.bounds[:d2] == p0.bounds[:d2]
    g = p1.value_grad(np.random.default_rng(1).normal(size=2 * d2) * 0.3, 1.0, 0.0, np.zeros((5, 100, 5)))[1]
    assert (g[d2:] == 0).all()                                   # and the gradient there is exactly zero


def test_directed_only_learns_a_dag_and_default_is_unchanged():
    D, B, X, names = _data(3)
    base = discover_admg(X, names, DCDConfig(seed=0, max_outer=12))
    same = discover_admg(X, names, DCDConfig(seed=0, max_outer=12, directed_only=False))
    assert base.admg.n_edges() == same.admg.n_edges() and np.array_equal(base.W1, same.W1)   # flag off = M2 behaviour
    dag = discover_admg(X, names, DCDConfig(seed=0, max_outer=12, directed_only=True))
    assert len(dag.admg.bi_edges) == 0 and (dag.W2_raw == 0).all() and (dag.W2 == 0).all()
    assert dag.admg.n_edges()[0] > 0                              # it still finds directed structure


def test_spot_directed_only_counts_only_directed_coordinates():
    from pgm.spot import SpotMask
    d = 6
    P = np.full((d, d), 0.5)
    rng = np.random.default_rng(0)
    g, th = rng.normal(size=2 * d * d), rng.normal(size=2 * d * d)
    full, dag = SpotMask(P, SpotConfig(floor=None, mask_seed=0)), SpotMask(P, SpotConfig(floor=None, mask_seed=0), directed_only=True)
    full(g, th, 1), dag(g, th, 1)
    n_full = sum(full.stats[1][k] for k in (0, 2))              # strengthening proposals + weakening moves over free coordinates
    n_dag = sum(dag.stats[1][k] for k in (0, 2))
    assert n_full == d * (d - 1) + d * (d - 1) // 2             # W1 off-diagonal + lower-triangular W2
    assert n_dag == d * (d - 1)                                  # bidirected coordinates are not counted
    D, B, X, names = _data(4)
    P = oracle_posterior((D | D.T | B).astype(bool))
    res = run_spot(X, names, P=P, cfg=DCDConfig(seed=0, max_outer=8, directed_only=True), spot_cfg=SpotConfig(mask_seed=0))
    assert len(res.admg.bi_edges) == 0 and res.mask_summary


def test_dir_variants_in_the_benchmark_scripts():
    import spot_vs_abic_synthetic as syn
    assert "abic_dir" in syn.ALL_VARIANTS and "spot_cal_dir" in syn.ALL_VARIANTS and "abic" in syn.ALL_VARIANTS
    D, B, X, names, skel = syn.make_graph(2, 8, 300, "zscore")
    cfg = DCDConfig(seed=0, max_outer=10)
    post_res = SimpleNamespace(P=oracle_posterior(skel), t_total=0.0)
    for v in ("abic_dir", "spot_oracle_dir"):
        res, extra, wall = syn.run_variant(v, X, names, post_res, skel, cfg, 0, None)
        assert len(res.admg.bi_edges) == 0, v
    res, _, _ = syn.run_variant("abic", X, names, post_res, skel, cfg, 0, None)
    assert cfg.directed_only is False                             # the caller's config is never mutated


# ------------------------------------------------------------------------------ E9: FCI / RFCI
def test_fci_finds_the_collider_and_maps_names_back():
    _need_tetrad()
    from pgm.fci import run_tetrad
    rng = np.random.default_rng(0)
    n = 4000
    a, b = rng.normal(size=n), rng.normal(size=n)
    c = a + b + 0.5 * rng.normal(size=n)                          # A -> C <- B, A and B independent
    X = np.column_stack([a, b, c])
    names = ["svc a.v1", "svc-b", "CEE"]                          # awkward names: the jar only ever sees X1..X3
    for alg in ("fci", "rfci"):
        r = run_tetrad(X, names, algorithm=alg, alpha=0.01)
        assert r.pag.skeleton() == {frozenset(["svc a.v1", "CEE"]), frozenset(["svc-b", "CEE"])}, alg
        assert r.pag.arrowheads() == {("svc a.v1", "CEE"), ("svc-b", "CEE")}, alg   # both arrowheads at the collider
        assert r.t_wall > 0 and r.n_edges == 2


def test_fci_errors_are_explicit():
    _need_tetrad()
    from pgm.fci import TetradError, run_tetrad
    X = np.random.default_rng(1).normal(size=(500, 6))
    for bad in (lambda: run_tetrad(X, ["a"] * 5), lambda: run_tetrad(X, list("abcdef"), algorithm="pc"),
                lambda: run_tetrad(np.where(X > 2, np.nan, X), list("abcdef"))):
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    try:
        run_tetrad(X, list("abcdef"), timeout=0.001)
        raise AssertionError("expected a timeout")
    except TetradError as e:
        assert "timed out" in str(e)


def test_run_fci_synthetic_scores_resumes_and_retries_failures():
    _need_tetrad()
    import run_fci
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "f.jsonl")
        args = ["--source", "synthetic", "--d", "8", "--n", "400", "--seeds", "0", "1", "--alphas", "0.05",
                "--algorithms", "fci", "rfci", "--out", out]
        run_fci.main(args)
        rows = [json.loads(l) for l in open(out)]
        assert len(rows) == 4 and not any(r["failed"] for r in rows)
        assert all("c_shd_pag" in r and "c_skeleton_f1" in r and r["t_wall"] > 0 for r in rows)
        assert {(r["seed"], r["algorithm"]) for r in rows} == {(0, "fci"), (0, "rfci"), (1, "fci"), (1, "rfci")}
        run_fci.main(args)                                         # resume: nothing re-run
        assert len(open(out).readlines()) == 4
        # a failed line is not "done": it is retried, and a good line is appended
        with open(out, "a") as f:
            f.write(json.dumps(dict(source="synthetic", seed=2, d=8, n=400, data="zscore", density=None, snr=None, ar=0.0,
                                    missing=0.0, algorithm="fci", alpha=0.05, depth=-1, variant="fci@0.05", failed=True,
                                    error="boom")) + "\n")
        run_fci.main(args[:args.index("--seeds")] + ["--seeds", "2"] + args[args.index("--alphas"):])
        last = [json.loads(l) for l in open(out)]
        assert sum(1 for r in last if r["seed"] == 2 and not r["failed"]) == 2   # fci and rfci both ran for seed 2
        # the same graphs as the ADMG benchmark (same seed -> same truth size)
        import spot_vs_abic_synthetic as syn
        D, B, X, names, skel = syn.make_graph(0, 8, 400, "zscore")
        assert rows[0]["n_true_edges"] == int(np.triu(skel, 1).sum())


def test_run_fci_petshop_mode_on_fake_data():
    _need_tetrad()
    import run_fci
    from fake_petshop import make_fake_petshop
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_petshop(tmp, "low_traffic", n=300)
        out = os.path.join(tmp, "p.jsonl")
        run_fci.main(["--source", "petshop", "--root", tmp, "--scenarios", "low_traffic", "--algorithms", "fci",
                      "--alphas", "0.01", "0.05", "--out", out])
        rows = [json.loads(l) for l in open(out)]
        assert len(rows) == 2 and all(r["scenario"] == "low_traffic" and r["prep"] == "gauss_rank" and not r["failed"] for r in rows)
        assert all(r["d"] == 8 and "c_arrowhead_f1" in r and "c_shd_pag" in r for r in rows)
        run_fci.main(["--source", "petshop", "--root", tmp, "--scenarios", "low_traffic", "--algorithms", "fci",
                      "--alphas", "0.01", "0.05", "--out", out])
        assert len(open(out).readlines()) == 2
