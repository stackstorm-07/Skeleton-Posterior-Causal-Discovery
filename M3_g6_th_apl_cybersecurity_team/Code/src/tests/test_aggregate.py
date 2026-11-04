"""M3 item 6: scripts/aggregate_m3.py and the pgm/stats.py helpers it uses.  Rows with DESIGNED effects: a real effect must be
"supported", a null effect must not, and a missing experiment must be skipped, not crash."""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

import aggregate_m3 as am                                                         # noqa: E402
from pgm.stats import bootstrap_mean_ci, bootstrap_paired_ci, holm, two_way_anova, welch_diff   # noqa: E402


def _syn_rows(effect, graphs=20, ns=(200, 1000), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for n in ns:
        for g in range(graphs):
            base = 0.6 + 0.05 * rng.normal()
            for v, add in (("abic", 0.0), ("spot_cal", effect)):
                f1 = base + add + 0.01 * rng.normal()
                rows.append(dict(seed=g, variant=v, d=25, n=n, data="zscore", density=None, snr=None, ar=0.0, missing=0.0,
                                 c_skeleton_f1=f1, c_arrowhead_f1=f1 / 2, c_tail_f1=f1 / 3, c_shd_pag=float(30 - 20 * f1),
                                 t_optimize=2.0 if v == "abic" else 1.0, t_posterior_features=0.1, t_posterior_model=0.05))
    return rows


def _write(res, name, rows):
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_decision_rule_needs_both_tests_and_the_interval():
    f = lambda **k: dict(n=20, p_ttest=0.01, p_wilcoxon=0.01, ci_mean_lo=0.01, ci_mean_hi=0.05, ci_pct_lo=-1, ci_pct_hi=1, mean_diff=0.03, **k)
    assert am.decide(f()) == "supported: first better"
    assert am.decide(f(), higher=False) == "supported: first worse"
    assert am.decide(dict(f(), p_wilcoxon=0.2)) == "not distinguishable from noise"
    assert am.decide(dict(f(), p_ttest=0.2)) == "not distinguishable from noise"
    assert am.decide(dict(f(), ci_mean_lo=-0.01)) == "not distinguishable from noise"
    assert am.decide(f(), ci="pct") == "not distinguishable from noise"          # the percentile interval is used for bootstrap rows
    assert am.decide(dict(n=1)) == "n too small"


def test_real_effect_supported_null_not_and_no_hand_typed_numbers():
    with tempfile.TemporaryDirectory() as tmp:
        for name, eff in (("real", 0.08), ("null", 0.0)):
            res = os.path.join(tmp, name, "res")
            _write(res, "syn_m3_e1.jsonl", _syn_rows(eff))
            agg = am.Agg(res, os.path.join(res, "aggregate"))
            agg.sec_E1()
            agg.paired_table()
            rows = [r for r in agg.paired_rows if r["metric"] == "c_skeleton_f1" and r["exp"] == "E1"]
            assert {r["n"] for r in rows} >= {200, 1000, "pooled"}
            dec = {r["decision"] for r in rows}
            if name == "real":
                assert dec == {"supported: first better"} and all(abs(r["mean_diff"] - 0.08) < 0.01 for r in rows)
                assert all(r["ci_lo"] > 0 for r in rows)
            else:
                assert "supported: first better" not in {r["decision"] for r in rows if r["n"] == "pooled"}
                assert all(r["ci_lo"] < 0 < r["ci_hi"] for r in rows if r["n"] == "pooled")
            # time: spot_cal pays the posterior, abic does not
            tt = agg.get("E1").groupby("variant")["t_total"].mean()
            assert abs(tt["abic"] - 2.0) < 1e-9 and abs(tt["spot_cal"] - 1.15) < 1e-9
            assert os.path.exists(os.path.join(res, "aggregate", "tables", "T1a_synthetic_sample_size.csv"))
            assert os.path.exists(os.path.join(res, "aggregate", "figs", "F1_sample_size_vs_shd.png"))
        # a table cell equals the number computed from the file
        import pandas as pd
        t = pd.read_csv(os.path.join(tmp, "real", "res", "aggregate", "tables", "T1a_synthetic_sample_size.csv"))
        raw = pd.DataFrame(_syn_rows(0.08))
        m = raw[(raw.n == 1000) & (raw.variant == "abic")].c_skeleton_f1.mean()
        assert abs(t[(t.n == 1000) & (t.method == "ABIC")]["_c_skeleton_f1_mean"].iloc[0] - m) < 1e-9


def test_duplicate_rows_are_counted_once_and_missing_experiments_do_not_crash():
    with tempfile.TemporaryDirectory() as tmp:
        res = os.path.join(tmp, "res")
        rows = _syn_rows(0.05, graphs=8)
        _write(res, "syn_m3_e1.jsonl", rows + rows[:10])                      # a re-run appended 10 duplicates
        agg = am.Agg(res, os.path.join(res, "aggregate"))
        m = agg.run()                                                         # every other file is missing
        assert m["inputs"]["syn_m3_e1.jsonl"]["rows"] == len(rows) + 10 and m["inputs"]["syn_m3_e1.jsonl"]["rows_after_dedup"] == len(rows)
        assert len(m["inputs"]["syn_m3_e1.jsonl"]["sha256"]) == 64
        assert len(agg.get("E1")) == len(rows)
        verdicts = {r["hypothesis"]: r["verdict"] for r in agg.scorecard}
        assert set(verdicts) == {"H1", "H2", "H3", "H4", "H5"} and verdicts["H2"] == "not run / incomplete"
        t0 = agg.tables["T0_completeness"][0].set_index("experiment")
        assert t0.loc["E3", "status"] == "not run" and t0.loc["E11", "status"].startswith("skipped")
        assert os.path.exists(os.path.join(res, "aggregate", "manifest.json")) and os.path.exists(os.path.join(res, "aggregate", "scorecard.md"))


def test_stats_helpers():
    rng = np.random.default_rng(0)
    a, b = np.repeat(["x", "y"], 40), np.tile(np.repeat(["p", "q"], 20), 2)
    y = rng.normal(size=80) + 2.0 * (a == "x")
    r = {e["effect"]: e for e in two_way_anova(y, a, b)}
    assert r["a"]["p"] < 0.001 and r["b"]["p"] > 0.01 and set(r) >= {"a", "b", "a:b"}
    adj = holm(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5]) and (adj >= np.array([0.01, 0.04, 0.03, 0.5])).all()
    x = rng.normal(0.5, 0.1, size=200)
    mu, lo, hi = bootstrap_mean_ci(x)
    assert lo < 0.5 < hi and abs(mu - x.mean()) < 1e-12
    mu, lo, hi, p = bootstrap_paired_ci(x + 0.2, x)
    assert lo > 0 and p < 0.01
    w = welch_diff(rng.normal(0, 1, 50), rng.normal(1, 1, 50))
    assert w["ci_hi"] < 0 and w["p"] < 0.01
