"""Stage 3: scoring math, issue loader, and the runner on FAKE PetShop-format data (checks plumbing and the scoring logic only)."""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    sys.path.insert(0, p)
from fake_petshop import make_fake_issues, make_fake_petshop        # noqa: E402
from pgm.rca import (Transformer, fit_rca_model, mechanism_scores, noise_shift, rank_of, rank_root_causes,   # noqa: E402
                     directed_ancestors)
from skeleton.simulator import random_admg, simulate_linear_gaussian   # noqa: E402


def _sem(seed, d=8, n=4000):
    rng = np.random.default_rng(seed)
    D, B = random_admg(d, 10, 0.3, rng)
    X, W1, Om = simulate_linear_gaussian(D, B, n, rng, standardize=False)
    return rng, D, B, X, W1, Om


def test_score_equals_gls_reduction():
    """score_i is exactly the drop of the Mahalanobis distance when a shift at node i is allowed (brute force)."""
    rng, D, B, X, W1, Om = _sem(0)
    d = len(D)
    from pgm.rca import RCAModel
    model = RCAModel([str(i) for i in range(d)], W1, Om, D, B, 0.0)
    dmu = rng.normal(size=d)
    de = noise_shift(W1, dmu)
    K = np.linalg.inv(Om)
    s = mechanism_scores(model, dmu, use_bidirected=True)
    for i in range(d):
        # brute force: best single-node shift a_star by GLS, then the Mahalanobis drop it buys
        a_star = (K @ de)[i] / K[i, i]
        red = de @ K @ de - (de - a_star * np.eye(d)[i]) @ K @ (de - a_star * np.eye(d)[i])
        assert abs(red - s[i]) < 1e-8 * max(1.0, s[i])


def test_true_admg_finds_the_shifted_mechanism():
    hits_admg = hits_anom = 0
    trials = 0
    for seed in range(6):
        rng, D, B, X, W1, Om = _sem(seed)
        d = len(D)
        names = [str(i) for i in range(d)]
        model = fit_rca_model(X, names, D, B)
        Minv = np.linalg.inv(np.eye(d) - W1)
        for k in rng.choice(d, 2, replace=False):
            eps = rng.multivariate_normal(np.zeros(d), Om, size=60)
            eps[:, k] += 5 * np.sqrt(Om[k, k])
            dmu = (eps @ Minv).mean(0) - X.mean(0)
            target = int((k + 1) % d)
            hits_admg += rank_root_causes(dmu, target, "admg", model)[0] == k
            hits_anom += rank_root_causes(dmu, target, "anomaly")[0] == k
            trials += 1
    assert hits_admg >= trials - 1, (hits_admg, trials)
    assert hits_admg >= hits_anom


def test_helpers():
    D = np.zeros((4, 4), int); D[0, 1] = D[1, 2] = 1
    assert list(directed_ancestors(D, 2)) == [True, True, False, False]
    assert rank_of(np.array([3, 1, 2]), 2) == 3 and rank_of(np.array([3, 1]), 2) is None
    X = np.random.default_rng(0).lognormal(size=(500, 3))
    for kind in ("log", "zscore", "gauss_rank"):
        Z = Transformer(kind).fit(X).transform(X)
        assert np.abs(Z.mean(0)).max() < 0.1 and np.abs(Z.std(0) - 1).max() < 0.1, kind


def test_issue_loader_and_matching():
    from pgm.petshop_issues import list_issue_cases, load_issue_metrics, match_node
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_petshop(tmp, "low_traffic")
        make_fake_issues(tmp, "low_traffic")
        cases = list_issue_cases(os.path.join(tmp, "low_traffic"))
        assert len(cases) == 4 and {c.split for c in cases} == {"train", "test"}
        names = ["frontend", "api", "svc_a", "svc_b", "db", "queue", "cache", "ext"]
        base = np.ones((10, len(names)))
        X, info = load_issue_metrics(cases[0], names, base)
        assert info["window_used"] == "after" and X.shape == (40, len(names)), info
        X2, info2 = load_issue_metrics(cases[0], names, base, window="all")
        assert X2.shape[0] == 60
        assert match_node("svc_a", names) == (2, "exact") and match_node("SVC_A", names)[1] == "ci"
        assert match_node("svc", names)[1] == "ambiguous" and match_node("nothere", names)[1] == "none"
        assert match_node("fron", names) == (0, "fuzzy")


def test_run_rca_on_fake_data():
    from pgm.graphs import find_tetrad_jar
    try:
        find_tetrad_jar()
    except Exception as e:
        raise __import__("unittest").SkipTest(str(e))
    import json
    import run_rca
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_petshop(tmp, "low_traffic")
        make_fake_issues(tmp, "low_traffic")
        out = os.path.join(tmp, "rca.jsonl")
        rows = run_rca.main(["--root", tmp, "--scenarios", "low_traffic", "--sources", "abic", "truth", "--out", out,
                             "--graphs-dir", os.path.join(tmp, "g"), "--graph-seeds", "0", "--lbfgs-maxiter", "60", "--max-outer", "4"])
        assert len(rows) == 4 * (2 + 2 * 8)   # per case: no-graph {anomaly, corr} + per graph {anomaly x2, dag x3, admg x3}
        t = [r for r in rows if r["source"] == "truth" and r["method"] == "admg" and r["ancestor_mode"] == "none"]
        assert len(t) == 4 and all(r["root_observed"] for r in t)
        assert np.mean([r["top3"] for r in t]) >= 0.5
        import summarize_rca
        summarize_rca.main([out, "--out", os.path.join(tmp, "s")])
        assert os.path.exists(os.path.join(tmp, "s.md"))


def test_correlation_scores_pick_the_coupled_service():
    from pgm.rca import correlation_scores
    rng = np.random.default_rng(0)
    base = rng.normal(size=60)
    Za = np.c_[rng.normal(size=60), 2 * base + 0.1 * rng.normal(size=60), rng.normal(size=60), base]   # col 3 = target
    s = correlation_scores(Za, 3)
    assert np.argmax(s[:3]) == 1 and s[1] > 0.9 and s[3] > 0.99
    assert correlation_scores(Za[:2], 3).max() == 0.0          # too few abnormal rows -> no evidence


def test_ancestor_modes():
    d = 5
    D = np.zeros((d, d), dtype=int)
    D[0, 4] = 1; D[1, 0] = 1                 # 1 -> 0 -> 4 (target 4); nodes 2, 3 are not ancestors
    from pgm.rca import RCAModel
    model = RCAModel([str(i) for i in range(d)], np.zeros((d, d)), np.eye(d), D, np.zeros((d, d), dtype=int), 0.0)
    dmu = np.array([0.1, 0.2, 5.0, 3.0, 0.0])        # non-ancestors 2, 3 have the biggest shifts
    none = rank_root_causes(dmu, 4, "anomaly", model, ancestor_mode="none")
    first = rank_root_causes(dmu, 4, "anomaly", model, ancestor_mode="first")
    only = rank_root_causes(dmu, 4, "anomaly", model, ancestor_mode="only")
    assert list(none) == [2, 3, 1, 0]
    assert list(first) == [1, 0, 2, 3]       # ancestors first (by score), then the rest (by score): nothing is excluded
    assert list(only) == [1, 0]
    assert list(rank_root_causes(dmu, 4, "anomaly", model, ancestors_only=True)) == [1, 0]   # old spelling


def test_summarizer_handles_subset_of_sources():
    import pandas as pd, subprocess, tempfile, json
    rows = [dict(scenario="low_traffic", split="test", issue=f"i{k}", source="truth", graph_seed=-1, method="admg", ancestor_mode="first",
                 ancestors_only=False, rank=k % 3 + 1, top1=(k % 3 == 0), top3=True, rr=1.0 / (k % 3 + 1), root_observed=True,
                 root_match="exact", window="after", transform="log") for k in range(4)]
    with tempfile.TemporaryDirectory() as tmp:
        f = os.path.join(tmp, "r.jsonl")
        open(f, "w").write("\n".join(json.dumps(r) for r in rows))
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "summarize_rca.py"), f], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr[-500:]
        assert "ancestors first" in out.stdout
