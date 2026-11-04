"""Stage 2d plumbing test on a FAKE PetShop-format dataset (see fake_petshop.py): loader, truth, data check, runner,
resume, summary.  Optimizer settings are shrunk so it runs in about a minute; it checks the plumbing, not accuracy."""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "scripts"), HERE):
    sys.path.insert(0, p)
from fake_petshop import make_fake_petshop                      # noqa: E402


class SkipTest(Exception):
    pass


def _need_tools():
    from pgm.graphs import find_tetrad_jar
    try:
        find_tetrad_jar()
    except Exception as e:
        try:
            import pytest
            pytest.skip(str(e))
        except ImportError:
            raise SkipTest(str(e))
    if not os.path.exists(os.path.join(ROOT, "models", "skeleton_posterior.pkl")):
        raise SkipTest("no trained posterior")


def test_fake_petshop_end_to_end():
    _need_tools()
    import run_petshop
    import summarize_2d
    import petshop_data_check
    from pgm.petshop import load_petshop, preprocess
    with tempfile.TemporaryDirectory() as td:
        root = make_fake_petshop(td, "low_traffic", n=300)
        ps = load_petshop("low_traffic", root)
        assert ps.meta["d"] == 8 and set(ps.latents) == {"lat1", "lat2"}
        assert len(ps.pag) > 0
        for how in ("gauss_rank", "none", "raw_centered"):
            Z = preprocess(ps.data, how)
            assert Z.shape == ps.data.shape and abs(Z.mean()) < 1e-8
        chk = petshop_data_check.check("low_traffic", root)
        assert chk["d"] == 8 and not chk["in_distribution"]          # d=8 is outside the trained 25-40: must be flagged
        out = os.path.join(td, "r.jsonl")
        args = ["--root", root, "--scenarios", "low_traffic", "--seeds", "0", "1", "--variants", "abic", "spot_cal", "spot_oracle",
                "--out", out, "--lbfgs-maxiter", "60", "--max-outer", "6"]
        run_petshop.main(args)
        rows = [json.loads(l) for l in open(out)]
        assert len(rows) == 6
        for r in rows:
            assert 0.0 <= r["c_skeleton_f1"] <= 1.0 and "l_tail_f1" in r and r["prep"] == "gauss_rank"
            assert r["t_optimize"] >= 0 and r["n_true_edges"] == chk["truth"]["skeleton_edges"]
        assert all(r["t_posterior"] == 0 for r in rows if r["variant"] == "abic")
        assert all(r["t_posterior"] > 0 for r in rows if r["variant"] != "abic")
        run_petshop.main(args)                                       # resume: nothing new may be appended
        assert len(open(out).read().strip().splitlines()) == 6
        txt = os.path.join(td, "s.md")
        summarize_2d.main([out, "--out", txt])
        s = open(txt).read()
        assert "low_traffic" in s and "Runtime breakdown" in s and "spot_oracle" in s
