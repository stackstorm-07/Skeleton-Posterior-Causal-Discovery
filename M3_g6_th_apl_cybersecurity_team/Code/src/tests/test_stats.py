import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from pgm.stats import block_bootstrap_indices, default_block_length, paired_summary


def test_block_bootstrap_indices_structure():
    rng = np.random.default_rng(0)
    n, l = 103, 8
    idx = block_bootstrap_indices(n, l, rng)
    assert len(idx) == n and idx.min() >= 0 and idx.max() < n
    # inside every block of l the indices are consecutive (time order preserved)
    for s in range(0, n - l, l):
        assert np.all(np.diff(idx[s:s + l]) == 1)
    assert default_block_length(589) == 8 and default_block_length(8) == 2
    assert len(block_bootstrap_indices(5, 50, rng)) == 5            # block longer than the series is clipped


def test_block_bootstrap_keeps_autocorrelation_iid_destroys_it():
    rng = np.random.default_rng(1)
    n = 3000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.7 * x[t - 1] + rng.normal()
    ac = lambda v: np.corrcoef(v[1:], v[:-1])[0, 1]
    blk = np.mean([ac(x[block_bootstrap_indices(n, 30, rng)]) for _ in range(20)])
    iid = np.mean([ac(x[rng.integers(0, n, n)]) for _ in range(20)])
    assert blk > 0.55 and abs(iid) < 0.1, (blk, iid)


def test_paired_summary_known_values():
    a = np.array([0.55, 0.60, 0.58, 0.62, 0.57, 0.61])
    b = a - np.array([0.03, 0.02, 0.04, 0.05, 0.01, 0.02])
    r = paired_summary(a, b)
    assert abs(r["mean_diff"] - 0.0283333) < 1e-6 and (r["wins"], r["losses"], r["ties"]) == (6, 0, 0)
    assert abs(r["p_wilcoxon"] - 2 / 2 ** 6) < 1e-12              # exact two-sided minimum for 6 pairs: 0.03125
    assert r["p_ttest"] < 0.01 and r["ci_mean_lo"] > 0
    # lower-is-better flips wins and losses; identical inputs give p = 1
    assert paired_summary(a, b, higher_is_better=False)["losses"] == 6
    z = paired_summary(a, a)
    assert z["p_wilcoxon"] == 1.0 and z["p_ttest"] == 1.0 and z["ties"] == 6
    assert paired_summary([np.nan], [1.0])["n"] == 0


def test_run_bootstrap_end_to_end_on_fake_data():
    import json, tempfile
    from pgm.graphs import find_tetrad_jar
    try:
        find_tetrad_jar()
    except Exception as e:
        raise __import__("unittest").SkipTest(str(e))
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts")); sys.path.insert(0, HERE)
    from fake_petshop import make_fake_petshop
    import run_bootstrap
    with tempfile.TemporaryDirectory() as tmp:
        make_fake_petshop(tmp, "low_traffic", n=300)
        out = os.path.join(tmp, "b.jsonl")
        args = ["--root", tmp, "--scenarios", "low_traffic", "--B", "2", "--workers", "2", "--out", out,
                "--lbfgs-maxiter", "60", "--max-outer", "4"]
        run_bootstrap.main(args)
        rows = [json.loads(l) for l in open(out)]
        assert len(rows) == 4 and {r["variant"] for r in rows} == {"abic", "spot_cal"}
        assert all("c_shd_pag" in r and r["block_len"] == 7 for r in rows)          # round(300 ** (1/3)) = 7
        run_bootstrap.main(args)                                                      # resume: nothing is re-run
        assert len(open(out).readlines()) == 4
        s = run_bootstrap.summarize(out)
        assert set(s.metric) >= {"c_skeleton_f1", "c_shd_pag"} and (s.n == 2).all()
        # paired resamples are reproducible and independent of the worker count
        i1 = run_bootstrap.resample_indices(2026, "low_traffic", 1, 300, 7)
        assert (i1 == run_bootstrap.resample_indices(2026, "low_traffic", 1, 300, 7)).all()
        assert not (i1 == run_bootstrap.resample_indices(2026, "low_traffic", 0, 300, 7)).all()
