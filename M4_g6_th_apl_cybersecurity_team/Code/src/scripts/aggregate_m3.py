"""M3 item 6: read every result file, write the report's tables, figures and the hypothesis scorecard.  Nothing is typed by hand.

    python scripts/aggregate_m3.py                                   # results/ -> results/aggregate/
    python scripts/aggregate_m3.py --results-dir results --out-dir results/aggregate [--cut] [--petshop-root <dataset>]

Outputs in ``--out-dir``
  tables/<name>.csv|md|tex   every table (columns starting with "_" exist only in the CSV: raw numbers behind a formatted "mean ± sd" cell)
  figs/F1..F7*.png|pdf       F1 sample size vs SHD, F2 robustness, F3 accuracy/inference error vs time, F4 ablation bars, F5 bootstrap
                             distribution, F6 reliability diagrams, F7 edge-level error map (needs --petshop-root and the cached graphs)
  tables.md                  all tables in one file;   scorecard.md|csv  hypotheses H1-H5;   manifest.json  inputs (sha256, rows), outputs, settings
A missing or unfinished experiment never stops the run: its tables/figures are skipped and the completeness table says how many jobs finished.

Statistics (fixed here, before any number exists)
  paired unit      synthetic: graph;  PetShop optimizer seeds: (scenario, seed);  bootstrap: (scenario, resample);  RCA: case
  "supported"      paired t-test p < 0.05 AND Wilcoxon signed-rank p < 0.05 AND the 95 % CI of the mean paired difference excludes 0
                   (t-interval; for the block bootstrap the percentile interval of the differences). Otherwise "not distinguishable from noise".
                   With fewer than 6 units the Wilcoxon test cannot reach 0.05, so such comparisons are never "supported" (they are descriptive).
  p-values         no multiplicity correction in the decision (plan); the Holm-adjusted t-test p-values over the whole paired-test table are listed
  ANOVA            two-way (method x sample size) with interaction on E1, graphs treated as independent observations
  PetShop seeds    seeds vary only the optimizer: their spread is optimizer variance, not data variance (the bootstrap, E5, is the data-variance test)
  time             t_total = optimizer time + online posterior time for SPOT variants, optimizer time for ABIC; FCI/RFCI: whole java run.
                   Times from runs with timing_quiet = False are listed but only ratios within the same run are meaningful.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from pgm.stats import bootstrap_mean_ci, bootstrap_paired_ci, holm, paired_summary, two_way_anova, welch_diff   # noqa: E402

# ----------------------------------------------------------------------------------------------------------------- labels / files
LABEL = {"abic": "ABIC", "spot": "SPOT (floor 1e-3)", "spot_cal": "SPOT-cal (floor 6e-3)", "spot_nofloor": "mask only",
         "spot_floor_only": "floor only", "spot_random": "random posterior", "spot_oracle": "oracle posterior",
         "spot_released": "SPOT (other released)", "abic_dir": "ABIC directed-only", "spot_cal_dir": "SPOT-cal directed-only",
         "spot_dir": "SPOT directed-only"}
for _a in ("fci", "rfci"):
    for _al in (0.01, 0.05):
        LABEL[f"{_a}@{_al}"] = f"{_a.upper()} (alpha={_al:g})"
COLOR = {"abic": "#0072B2", "spot_cal": "#D55E00", "spot": "#E69F00", "spot_nofloor": "#CC79A7", "spot_floor_only": "#009E73",
         "spot_random": "#999999", "spot_oracle": "#000000", "abic_dir": "#56B4E9", "spot_cal_dir": "#F0A07A"}
FILES = dict(E1="syn_m3_e1.jsonl", E2="syn_m3_e2.jsonl", E3="ps_m3_e3.jsonl", E4="ps_m3_e4.jsonl", E5="boot_main.jsonl", E6="ps_m3_e6.jsonl",
             E7="post_quality_m3.jsonl", E8="inference_m3.jsonl", E10="ps_m3_e10.jsonl", E12="rca_m3.jsonl", E13="ps_m3_e13.jsonl",
             E9_ps="fci_petshop.jsonl", E9_syn="fci_syn_e1.jsonl")
KEYS = dict(E1=["seed", "variant", "d", "n", "data", "density", "snr", "ar", "missing"], E2=["seed", "variant", "d", "n", "data", "density", "snr", "ar", "missing"],
            E3=["scenario", "prep", "seed", "variant", "budget"], E4=["scenario", "prep", "seed", "variant", "budget"],
            E6=["scenario", "prep", "seed", "variant", "budget"], E10=["scenario", "prep", "seed", "variant", "budget"],
            E13=["scenario", "prep", "seed", "variant", "budget"], E5=["scenario", "boot_b", "variant"],
            E9_ps=["scenario", "prep", "algorithm", "alpha"], E9_syn=["seed", "d", "n", "density", "snr", "ar", "missing", "algorithm", "alpha"],
            E7=["dataset", "calibrator"], E8=["scenario", "graph_seed", "split", "model", "query"],
            E12=["scenario", "split", "issue", "source", "graph_seed", "method", "ancestor_mode"])
SCEN_LABEL = {"low_traffic": "low", "high_traffic": "high", "temporal_traffic1": "temporal 1", "temporal_traffic2": "temporal 2"}
M_SKEL, M_ARR, M_TAIL, M_SHD, M_T = "c_skeleton_f1", "c_arrowhead_f1", "c_tail_f1", "c_shd_pag", "t_total"
METRICS = [(M_SKEL, True, "skeleton F1"), (M_ARR, True, "arrowhead F1"), (M_TAIL, True, "tail F1"), (M_SHD, False, "SHD (PAG)"), (M_T, False, "time (s)")]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 110, "savefig.dpi": 200, "legend.frameon": False})


def lab(v):
    return LABEL.get(v, v)


def col(v):
    return COLOR.get(v, "#555555")


def ms(x, nd=3):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "n/a"
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}" if len(x) > 1 else f"{x.mean():.{nd}f}"


def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan"), float("nan")
    from scipy import stats
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def decide(s, higher=True, ci="mean"):
    """The plan's decision rule (see the module docstring)."""
    if s.get("n", 0) < 2 or "p_ttest" not in s:
        return "n too small"
    lo, hi = (s["ci_pct_lo"], s["ci_pct_hi"]) if ci == "pct" else (s["ci_mean_lo"], s["ci_mean_hi"])
    if not (s["p_ttest"] < 0.05 and s["p_wilcoxon"] < 0.05 and (lo > 0 or hi < 0)):
        return "not distinguishable from noise"
    return "supported: first better" if (s["mean_diff"] > 0) == higher else "supported: first worse"


def to_latex(df, caption, label):
    cols = [c for c in df.columns if not c.startswith("_")]
    esc = lambda x: str(x).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("±", r"$\pm$").replace("≥", r"$\geq$").replace("≤", r"$\leq$")
    lines = [r"\begin{table}[t]", r"\centering\footnotesize", rf"\caption{{{esc(caption)}}}\label{{{label}}}", r"\begin{tabular}{@{}" + "l" * len(cols) + r"@{}}",
             r"\toprule", " & ".join(esc(c) for c in cols) + r"\\", r"\midrule"]
    lines += [" & ".join(esc(v) for v in row) + r"\\" for row in df[cols].astype(object).where(df[cols].notna(), "").values.tolist()]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table}"]) + "\n"


# ----------------------------------------------------------------------------------------------------------------- the aggregator
class Agg:
    def __init__(self, results, out, cut=False, petshop_root=None):
        self.results, self.out, self.cut, self.petshop_root = results, out, cut, petshop_root
        for sub in ("tables", "figs"):
            os.makedirs(os.path.join(out, sub), exist_ok=True)
        self.frames, self.tables, self.figures, self.inputs, self.notes, self.facts, self.scorecard = {}, {}, [], {}, [], {}, []
        self.paired_rows = []

    # ---- input
    def load(self, key):
        path = os.path.join(self.results, FILES[key])
        if not os.path.exists(path):
            self.frames[key] = None
            return None
        raw = open(path, "rb").read()
        df = pd.DataFrame([json.loads(l) for l in raw.decode().splitlines() if l.strip()])
        n0 = len(df)
        sub = [c for c in KEYS.get(key, []) if c in df.columns]
        if sub and len(df):
            df = df.fillna({c: -999 for c in sub if df[c].dtype != object}).drop_duplicates(subset=sub, keep="last").replace(-999, np.nan)
        self.inputs[FILES[key]] = dict(sha256=hashlib.sha256(raw).hexdigest(), rows=n0, rows_after_dedup=len(df))
        if len(df) and "variant" in df and "t_optimize" in df:
            post = df.get("t_posterior_features", 0) + df.get("t_posterior_model", 0) if "t_posterior_features" in df else df.get("t_posterior", 0)
            df["t_total"] = df["t_optimize"] + np.where(df["variant"].astype(str).str.startswith("spot"), post, 0.0)
        if len(df) and "t_wall" in df:
            df["t_total"] = df["t_wall"]
        if key.startswith("E9") and "algorithm" in df:
            df["variant"] = df["algorithm"] + "@" + df["alpha"].astype(float).map(lambda a: f"{a:g}")
            df = df[~df.get("failed", False).astype(bool)] if "failed" in df else df
        self.frames[key] = df if len(df) else None
        return self.frames[key]

    def get(self, key):
        return self.frames.get(key)

    # ---- output
    def add_table(self, name, df, caption, sources):
        if df is None or not len(df):
            return
        df = df.reset_index(drop=True)
        self.tables[name] = (df, caption, sources)
        base = os.path.join(self.out, "tables", name)
        df.to_csv(base + ".csv", index=False)
        show = df[[c for c in df.columns if not c.startswith("_")]]
        with open(base + ".md", "w") as f:
            f.write(show.to_markdown(index=False, floatfmt=".3f") + "\n")
        with open(base + ".tex", "w") as f:
            f.write(to_latex(show.map(lambda v: f"{v:.3f}" if isinstance(v, float) else v), caption, f"tab:{name}"))

    def save_fig(self, fig, name, caption, sources):
        base = os.path.join(self.out, "figs", name)
        fig.savefig(base + ".png", bbox_inches="tight")
        fig.savefig(base + ".pdf", bbox_inches="tight")
        plt.close(fig)
        self.figures.append(dict(name=name, caption=caption, sources=sources))

    def compare(self, df, unit, a, b, metric, higher, exp, ctx, ci="mean", note=""):
        """One paired comparison of variant ``a`` against variant ``b`` over the units; appended to the paired-test rows and returned."""
        if df is None or metric not in df:
            return None
        sub = df[df["variant"].isin([a, b])]
        piv = sub.pivot_table(index=unit, columns="variant", values=metric, aggfunc="mean")
        if a not in piv or b not in piv:
            return None
        t = piv[[a, b]].dropna()
        if len(t) == 0:
            return None
        s = paired_summary(t[a].values, t[b].values, higher_is_better=higher)
        s["decision"] = decide(s, higher, ci)
        lo, hi = (s.get("ci_pct_lo"), s.get("ci_pct_hi")) if ci == "pct" else (s.get("ci_mean_lo"), s.get("ci_mean_hi"))
        self.paired_rows.append(dict(exp=exp, **ctx, first=lab(a), second=lab(b), metric=metric, unit="/".join(unit) if isinstance(unit, list) else unit,
                                     units=s["n"], mean_first=s["mean_a"], mean_second=s["mean_b"], mean_diff=s.get("mean_diff"), ci_lo=lo, ci_hi=hi,
                                     ci_kind="percentile" if ci == "pct" else "t", wins=s.get("wins"), losses=s.get("losses"), ties=s.get("ties"),
                                     p_ttest=s.get("p_ttest"), p_wilcoxon=s.get("p_wilcoxon"), decision=s["decision"], note=note,
                                     _a=a, _b=b))
        s["ci_lo"], s["ci_hi"] = lo, hi
        return s

    # ============================================================================================ sections
    def completeness(self, queue_failures):
        import run_queue_parallel as rq
        ns = argparse.Namespace(experiments=rq.ALL_EXPS, scenarios=rq.SCENARIOS, seeds=None, graph_seeds=None, B=None, d=25, syn_graphs=None,
                                e1_ns=None, e2_factors=None, pq_n_sim=60, pq_n_cal=40, cut=self.cut, root=None, model=None, results_dir=self.results,
                                lbfgs_maxiter=None, max_outer=None, max_rows=None)
        jobs = rq.build_jobs(ns, self.results)
        found = {}
        for fn in sorted({j["file"] for j in jobs}):
            ids = rq.done_ids(os.path.join(self.results, fn))
            found[fn] = ids
        rows = []
        for exp in rq.ALL_EXPS:
            js = [j for j in jobs if j["exp"] == exp]
            n_done = sum(j["id"] in found[j["file"]] for j in js)
            rows.append(dict(experiment=exp, files=", ".join(sorted({j["file"] for j in js})), jobs_planned=len(js), jobs_finished=n_done,
                             rows_in_files=sum(len(found[f]) for f in {j["file"] for j in js}),
                             status="complete" if n_done == len(js) else ("not run" if n_done == 0 else f"UNFINISHED ({n_done}/{len(js)})")))
        rows.append(dict(experiment="E11", files="(analysis of the files above)", jobs_planned=0, jobs_finished=0,
                         status="figure F7 and table from cached graphs" if self.petshop_root else "skipped (no --petshop-root)"))
        fails = []
        if os.path.exists(queue_failures):
            fails = [json.loads(l) for l in open(queue_failures) if l.strip()]
        df = pd.DataFrame(rows)
        df["failed_logged"] = [sum(f.get("exp") == r for f in fails) for r in df.experiment]
        self.add_table("T0_completeness", df, "Experiment completeness: planned jobs (default sizes of the queue) against finished jobs found in the result files", list(FILES.values()))
        self.facts["completeness"] = {r["experiment"]: r["status"] for r in rows}

    # ---- E1 / E2 : synthetic
    def sec_E1(self):
        df = self.load("E1")
        if df is None:
            return
        src = [FILES["E1"]]
        rows = []
        for (n, v), g in df.groupby(["n", "variant"]):
            r = dict(n=int(n), method=lab(v), graphs=len(g))
            for m, _, name in METRICS:
                r[name] = ms(g[m].values, 2 if m == M_SHD or m == M_T else 3)
                r[f"_{m}_mean"], r[f"_{m}_ci_lo"], r[f"_{m}_ci_hi"] = ci95(g[m].values)
            rows.append(r)
        self.add_table("T1a_synthetic_sample_size", pd.DataFrame(rows), "Synthetic graphs (d=25, z-scored): mean ± SD over graphs, by sample size", src)
        for n, g in df.groupby("n"):
            for m, hib, _ in METRICS:
                self.compare(g, ["seed"], "spot_cal", "abic", m, hib, "E1", dict(n=int(n)))
            if n == 1000:
                for m, hib, _ in METRICS:
                    self.compare(g, ["seed"], "spot", "abic", m, hib, "E1", dict(n=int(n)))
        # pooled over n and the ANOVA
        two = df[df["variant"].isin(["abic", "spot_cal"])]
        self.compare(two, ["n", "seed"], "spot_cal", "abic", M_SKEL, True, "E1", dict(n="pooled"))
        an_rows = []
        for m, name in ((M_SKEL, "skeleton F1"), (M_SHD, "SHD (PAG)")):
            if two["n"].nunique() >= 2:
                for r in two_way_anova(two[m].values, two["variant"].values, two["n"].astype(int).values):
                    an_rows.append(dict(metric=name, effect={"a": "method", "b": "sample size", "a:b": "method x sample size"}.get(r["effect"], r["effect"]),
                                        df=r["df"], F=r["F"], p=r["p"], partial_eta2=r["partial_eta2"]))
        self.add_table("T4_anova_E1", pd.DataFrame(an_rows), "Two-way ANOVA on E1 (ABIC vs SPOT-cal x sample size)", src)
        # gain vs n (for H1)
        piv = two.pivot_table(index=["n", "seed"], columns="variant", values=M_SKEL)
        if {"abic", "spot_cal"} <= set(piv.columns):
            d = (piv["spot_cal"] - piv["abic"]).dropna().reset_index(name="diff")
            from scipy import stats
            if d["n"].nunique() >= 2 and len(d) > 3:
                lr = stats.linregress(np.log2(d["n"].values.astype(float)), d["diff"].values)
                h = stats.t.ppf(0.975, len(d) - 2) * lr.stderr
                self.facts["E1_slope"] = dict(slope=float(lr.slope), lo=float(lr.slope - h), hi=float(lr.slope + h), p=float(lr.pvalue))
            self.facts["E1_gain_by_n"] = {int(n): float(g["diff"].mean()) for n, g in d.groupby("n")}
            self.facts["E1_diffs_n1000"] = d[d["n"] == 1000]["diff"].values if (d["n"] == 1000).any() else np.array([])
        self.facts["E1_anova"] = {(r["metric"], r["effect"]): r["p"] for r in an_rows}
        # F1
        fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
        for k, (m, name) in enumerate(((M_SHD, "SHD (PAG), lower is better"), (M_SKEL, "skeleton F1"))):
            for v in ("abic", "spot_cal", "spot"):
                g = df[df["variant"] == v]
                if not len(g):
                    continue
                ns = sorted(g["n"].unique())
                stat = [ci95(g[g["n"] == n][m].values) for n in ns]
                x = np.array(ns, dtype=float) * (1.0 + 0.02 * ["abic", "spot_cal", "spot"].index(v))
                mu, lo, hi = (np.array([s[i] for s in stat]) for i in range(3))
                ax[k].errorbar(x, mu, yerr=[mu - lo, hi - mu], marker="o", ms=4, capsize=3, color=col(v), label=lab(v), lw=1.3)
            ax[k].set_xscale("log"); ax[k].set_xticks(sorted(df["n"].unique())); ax[k].set_xticklabels(sorted(df["n"].unique()))
            ax[k].set_xlabel("sample size n"); ax[k].set_ylabel(name)
        ax[0].legend(fontsize=7)
        self.save_fig(fig, "F1_sample_size_vs_shd", "Sample size against SHD and skeleton F1 on synthetic graphs (mean ± 95% CI over graphs).", src)

    def sec_E2(self):
        df = self.load("E2")
        base = self.get("E1")
        if df is None:
            return
        src = [FILES["E2"], FILES["E1"]]
        seeds = set(df["seed"])
        b = base[(base["n"] == 1000) & base["variant"].isin(["abic", "spot_cal"]) & base["seed"].isin(seeds)].copy() if base is not None else None
        cells = []
        for fac, key in (("density", "density"), ("measurement noise (SNR)", "snr"), ("autocorrelation (AR)", "ar"), ("missing data (MCAR)", "missing")):
            d = df[df[key].notna() & (df[key] != 0)] if key in ("ar", "missing") else df[df[key].notna()]
            if key in ("ar", "missing"):
                d = d[(d["density"].isna()) & (d["snr"].isna())] if key == "ar" else d
            for lv, g in d.groupby(key):
                cells.append((fac, key, lv, g))
        rows, plot = [], {}
        for fac, key, lv, g in cells:
            tag = dict(factor=fac, level=float(lv))
            for v in ("abic", "spot_cal"):
                gv = g[g["variant"] == v]
                r = dict(tag, method=lab(v), graphs=len(gv), **{name: ms(gv[m].values, 2 if m in (M_SHD, M_T) else 3) for m, _, name in METRICS})
                rows.append(r)
                plot.setdefault(fac, {}).setdefault(v, []).append((lv, ci95(gv[M_SKEL].values)))
            for m, hib, _ in METRICS[:4]:
                self.compare(g, ["seed"], "spot_cal", "abic", m, hib, "E2", dict(factor=fac, level=float(lv)))
        if b is not None and len(b):                                   # the default cell
            for v in ("abic", "spot_cal"):
                gv = b[b["variant"] == v]
                rows.append(dict(factor="default cell (E1, n=1000)", level=np.nan, method=lab(v), graphs=len(gv),
                                 **{name: ms(gv[m].values, 2 if m in (M_SHD, M_T) else 3) for m, _, name in METRICS}))
        self.add_table("T1b_synthetic_robustness", pd.DataFrame(rows), "Synthetic robustness (n=1000, d=25): mean ± SD over graphs per cell", src)
        fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.8), sharey=True)
        for ax, (fac, byv) in zip(axes, plot.items()):
            for v, pts in byv.items():
                pts = sorted(pts, key=lambda t: (-t[0] if "SNR" in fac else t[0]))
                x = np.arange(len(pts)) + (0.04 if v == "spot_cal" else 0)
                mu = np.array([p[1][0] for p in pts]); lo = np.array([p[1][1] for p in pts]); hi = np.array([p[1][2] for p in pts])
                if b is not None and len(b):                           # prepend the default cell
                    gv = b[b["variant"] == v][M_SKEL].values
                    m0, l0, h0 = ci95(gv)
                    mu, lo, hi, x = np.r_[m0, mu], np.r_[l0, lo], np.r_[h0, hi], np.r_[-1 + (0.04 if v == "spot_cal" else 0), x]
                ax.errorbar(x, mu, yerr=[np.nan_to_num(mu - lo), np.nan_to_num(hi - mu)], marker="o", ms=4, capsize=3, color=col(v), label=lab(v), lw=1.3)
            lv = [p[0] for p in sorted(list(byv.values())[0], key=lambda t: (-t[0] if "SNR" in fac else t[0]))]
            ax.set_xticks(([-1] if b is not None and len(b) else []) + list(range(len(lv))))
            ax.set_xticklabels((["default"] if b is not None and len(b) else []) + [f"{x:g}" for x in lv], fontsize=7)
            ax.set_title(fac, fontsize=8)
        axes[0].set_ylabel("skeleton F1"); axes[0].legend(fontsize=7)
        self.save_fig(fig, "F2_robustness", "Robustness of ABIC and SPOT-cal at n=1000: skeleton F1 against graph density, measurement noise (SNR 10, 3, 1), autocorrelation and missing data (mean ± 95% CI over graphs).", src)

    # ---- PetShop families
    def ps_frame(self):
        parts = [self.get(k) for k in ("E3", "E4", "E10") if self.get(k) is not None]
        return pd.concat(parts, ignore_index=True) if parts else None

    def sec_PetShop(self):
        for k in ("E3", "E4", "E10", "E13", "E6", "E5", "E9_ps", "E9_syn"):
            self.load(k)
        ps = self.ps_frame()
        e3 = self.get("E3")
        if e3 is not None:
            rows = []
            for (scn, v), g in e3.groupby(["scenario", "variant"]):
                rows.append(dict(scenario=SCEN_LABEL.get(scn, scn), method=lab(v), seeds=len(g), n=int(g["n"].iloc[0]), d=int(g["d"].iloc[0]),
                                 **{name: ms(g[m].values, 2 if m in (M_SHD, M_T) else 3) for m, _, name in METRICS},
                                 est_directed=f"{np.mean([x[0] for x in g['n_est_edges']]):.1f}", est_bidirected=f"{g['n_est_bidirected'].mean():.1f}",
                                 true_edges=int(g["n_true_edges"].iloc[0]), converged=f"{int(g['converged'].sum())}/{len(g)}"))
            self.add_table("T2_petshop_per_scenario", pd.DataFrame(rows), "PetShop, per scenario: mean ± SD over optimizer seeds (E3). SD is optimizer variance only.", [FILES["E3"]])
            for m, hib, _ in METRICS:
                for scn, g in e3.groupby("scenario"):
                    self.compare(g, ["seed"], "spot_cal", "abic", m, hib, "E3", dict(scenario=SCEN_LABEL.get(scn, scn)), note="optimizer seeds only (descriptive)")
                self.compare(e3, ["scenario", "seed"], "spot_cal", "abic", m, hib, "E3", dict(scenario="pooled"), note="optimizer seeds only (descriptive)")
                self.compare(e3, ["scenario", "seed"], "spot", "abic", m, hib, "E3", dict(scenario="pooled"), note="optimizer seeds only (descriptive)")
            # W/L/T over scenarios
            wl = []
            sm = e3.groupby(["scenario", "variant"])[[M_SKEL, M_ARR, M_TAIL, M_SHD]].mean()
            for v in ("spot_cal", "spot"):
                for m, hib, name in METRICS[:4]:
                    d = [(sm.loc[(s, v), m] - sm.loc[(s, "abic"), m]) * (1 if hib else -1) for s in sm.index.get_level_values(0).unique() if (s, v) in sm.index and (s, "abic") in sm.index]
                    wl.append(dict(method=lab(v), metric=name, vs="ABIC", scenarios=len(d), wins=int(sum(x > 0 for x in d)), losses=int(sum(x < 0 for x in d)), ties=int(sum(x == 0 for x in d)),
                                   mean_gain=float(np.mean(d)) if d else float("nan")))
            self.add_table("T2b_petshop_wlt", pd.DataFrame(wl), "PetShop (E3): wins/losses/ties over scenarios (scenario means), mean gain in the favourable direction", [FILES["E3"]])
        self.sec_E4(ps)
        self.sec_E5()
        self.sec_E6()
        self.sec_E9_E10_E13(ps)
        self.sec_T1(ps)

    def sec_E4(self, ps):
        e4, e3 = self.get("E4"), self.get("E3")
        if e4 is None or e3 is None:
            return
        src = [FILES["E3"], FILES["E4"]]
        both = ps[ps["variant"].isin(["abic", "spot_cal", "spot", "spot_nofloor", "spot_floor_only", "spot_random", "spot_oracle"])]
        common = set(map(tuple, e4[["scenario", "seed"]].values)) & set(map(tuple, e3[["scenario", "seed"]].values))
        both = both[[(s, k) in common for s, k in zip(both["scenario"], both["seed"])]]
        rows = []
        for v in ("abic", "spot_floor_only", "spot_nofloor", "spot_cal", "spot", "spot_random", "spot_oracle"):
            g = both[both["variant"] == v]
            if not len(g):
                continue
            rows.append(dict(method=lab(v), units=len(g), **{name: ms(g[m].values, 2 if m in (M_SHD, M_T) else 3) for m, _, name in METRICS},
                             free_pairs=f"{g['n_pairs_allowed'].replace(-1, np.nan).mean():.0f}" if g["n_pairs_allowed"].replace(-1, np.nan).notna().any() else "all",
                             _skel=float(g[M_SKEL].mean()), _t=float(g["t_optimize"].mean())))
        self.add_table("T3_ablation", pd.DataFrame(rows), "PetShop ablation (E4 with the matching E3 runs): mean ± SD over (scenario, seed) units", src)
        for ref in ("abic", "spot_cal"):
            for v in ("spot_floor_only", "spot_nofloor", "spot_random", "spot_oracle", "spot_cal" if ref == "abic" else "abic"):
                if v == ref:
                    continue
                for m, hib, _ in METRICS[:4] + [("t_optimize", False, "")]:
                    self.compare(both, ["scenario", "seed"], v, ref, m, hib, "E4", dict(scenario="pooled"), note="optimizer seeds only (descriptive)")
        # H3 facts: paired log optimizer time against ABIC
        lt = both.copy()
        lt["log_t"] = np.log(lt["t_optimize"].clip(lower=1e-3))
        f = {}
        for v in ("spot_floor_only", "spot_nofloor", "spot_cal"):
            s = self.compare(lt, ["scenario", "seed"], v, "abic", "log_t", False, "E4-H3", dict(scenario="pooled"), note="log optimizer time, ratio = exp(diff)")
            if s:
                f[v] = dict(ratio=float(np.exp(s["mean_diff"])), lo=float(np.exp(s["ci_lo"])) if s.get("ci_lo") is not None else np.nan,
                            hi=float(np.exp(s["ci_hi"])) if s.get("ci_hi") is not None else np.nan, decision=s["decision"], n=s["n"])
        piv = both.pivot_table(index=["scenario", "seed"], columns="variant", values="t_optimize")
        if {"abic", "spot_cal", "spot_floor_only"} <= set(piv.columns):
            q = piv.dropna(subset=["abic", "spot_cal", "spot_floor_only"])
            q = q[q["abic"] > q["spot_cal"]]
            if len(q):
                f["share_floor"] = float(np.median((q["abic"] - q["spot_floor_only"]) / (q["abic"] - q["spot_cal"])))
        self.facts["H3"] = f
        # F4
        fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
        for k, (m, name) in enumerate(((M_SKEL, "skeleton F1"), ("t_optimize", "optimizer time (s)"))):
            for i, v in enumerate(order):
                g = both[both["variant"] == v]
                mu, lo, hi = ci95(g[m].values)
                ax[k].bar(i, mu, color=col(v), alpha=0.85, yerr=[[max(mu - lo, 0) if np.isfinite(lo) else 0], [max(hi - mu, 0) if np.isfinite(hi) else 0]], capsize=3)
                ax[k].scatter(np.full(len(g), i) + np.linspace(-0.18, 0.18, max(len(g), 1))[:len(g)], g[m].values, s=7, color="k", alpha=0.45, zorder=3)
            ax[k].set_xticks(range(len(order))); ax[k].set_xticklabels([lab(v).replace(" posterior", "\nposterior").replace(" (floor", "\n(floor") for v in order], rotation=35, ha="right", fontsize=7)
            ax[k].set_ylabel(name)
        self.save_fig(fig, "F4_ablation", "PetShop ablation: ABIC, floor only, mask only, SPOT-cal, random and oracle posterior. Bars: mean ± 95% CI over (scenario, seed) units; dots: units.", src)

    def sec_E5(self):
        df = self.get("E5")
        if df is None:
            return
        src = [FILES["E5"]]
        rows = []
        for (scn, v), g in df.groupby(["scenario", "variant"]):
            rows.append(dict(scenario=SCEN_LABEL.get(scn, scn), method=lab(v), resamples=len(g), block_len=int(g["block_len"].iloc[0]),
                             **{name: ms(g[m].values, 2 if m in (M_SHD, M_T) else 3) for m, _, name in METRICS}))
        self.add_table("T2c_petshop_bootstrap", pd.DataFrame(rows), "PetShop block bootstrap (E5): mean ± SD over resamples", src)
        for m, hib, _ in METRICS:
            for scn, g in df.groupby("scenario"):
                self.compare(g, ["boot_b"], "spot_cal", "abic", m, hib, "E5", dict(scenario=SCEN_LABEL.get(scn, scn)), ci="pct")
            s = self.compare(df, ["scenario", "boot_b"], "spot_cal", "abic", m, hib, "E5", dict(scenario="pooled"), ci="pct")
            if m == M_SKEL and s:
                self.facts["E5_pooled"] = s
        piv = df.pivot_table(index=["scenario", "boot_b"], columns="variant", values=M_SKEL)
        if {"abic", "spot_cal"} <= set(piv.columns):
            d = (piv["spot_cal"] - piv["abic"]).dropna().reset_index(name="diff")
            self.facts["E5_diffs"] = d["diff"].values
            fig, ax = plt.subplots(figsize=(4.6, 3.0))
            scs = list(d["scenario"].unique())
            data = [d[d["scenario"] == s]["diff"].values for s in scs]
            vp = ax.violinplot(data, showextrema=False, widths=0.8)
            for b_ in vp["bodies"]:
                b_.set_facecolor(col("spot_cal")); b_.set_alpha(0.35)
            for i, x in enumerate(data):
                mu, lo, hi = np.mean(x), np.percentile(x, 2.5), np.percentile(x, 97.5)
                ax.scatter(np.full(len(x), i + 1) + np.linspace(-0.1, 0.1, len(x)), x, s=6, color="k", alpha=0.45)
                ax.errorbar(i + 1, mu, yerr=[[mu - lo], [hi - mu]], color=col("spot_cal"), capsize=4, marker="D", ms=5)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(range(1, len(scs) + 1)); ax.set_xticklabels([SCEN_LABEL.get(s, s) for s in scs])
            ax.set_ylabel("paired Δ skeleton F1 (SPOT-cal − ABIC)")
            self.save_fig(fig, "F5_bootstrap_delta", "Block bootstrap: paired difference in skeleton F1 per resample (dots), mean and 95% percentile interval (diamond), by scenario.", src)

    def sec_E6(self):
        df = self.get("E6")
        if df is None:
            return
        src = [FILES["E6"]]
        rows = []
        for (bd, v), g in df.groupby(["budget", "variant"]):
            rows.append(dict(budget_s=bd, method=lab(v), units=len(g), skeleton_F1=ms(g[M_SKEL].values), SHD=ms(g[M_SHD].values, 2),
                             converged=f"{int(g['converged'].sum())}/{len(g)}", timed_out=f"{int(g['timed_out'].sum())}/{len(g)}", timing_quiet=bool(g["timing_quiet"].all()) if "timing_quiet" in g else "n/a"))
        self.add_table("T2d_equal_budget", pd.DataFrame(rows), "Equal wall-clock budget (E6): mean ± SD over (scenario, seed) units", src)
        for bd, g in df.groupby("budget"):
            for m, hib, _ in METRICS[:4]:
                self.compare(g, ["scenario", "seed"], "spot_cal", "abic", m, hib, "E6", dict(budget=bd), note="optimizer seeds only (descriptive)")

    def sec_E9_E10_E13(self, ps):
        syn = self.get("E1")
        fs = self.get("E9_syn")
        if fs is not None and syn is not None:
            both = pd.concat([syn[syn["variant"].isin(["abic", "spot_cal"])][["seed", "n", "variant", M_SKEL, M_ARR, M_TAIL, M_SHD, M_T]],
                              fs[["seed", "n", "variant", M_SKEL, M_ARR, M_TAIL, M_SHD, M_T]]], ignore_index=True)
            for n, g in both.groupby("n"):
                for v in ("fci@0.05", "fci@0.01", "rfci@0.05", "rfci@0.01"):
                    for ref in ("abic", "spot_cal"):
                        for m, hib, _ in METRICS[:4]:
                            self.compare(g, ["seed"], v, ref, m, hib, "E9", dict(n=int(n)))
            self.frames["E9_syn_all"] = both
        if ps is not None:
            e3 = self.get("E3")
            e10 = self.get("E10")
            if e10 is not None and e3 is not None:
                comb = pd.concat([e3, e10], ignore_index=True)
                common = set(map(tuple, e10[["scenario", "seed"]].values)) & set(map(tuple, e3[["scenario", "seed"]].values))
                comb = comb[[(s, k) in common for s, k in zip(comb["scenario"], comb["seed"])]]
                rows = []
                for v in ("abic", "abic_dir", "spot_cal", "spot_cal_dir"):
                    g = comb[comb["variant"] == v]
                    if len(g):
                        rows.append(dict(method=lab(v), units=len(g), **{name: ms(g[m].values, 2 if m in (M_SHD, M_T) else 3) for m, _, name in METRICS},
                                         est_bidirected=f"{g['n_est_bidirected'].mean():.1f}"))
                self.add_table("T2e_directed_only", pd.DataFrame(rows), "Directed-only learner (E10) against the ADMG learners on the same (scenario, seed) units", [FILES["E3"], FILES["E10"]])
                for a, b in (("abic_dir", "abic"), ("spot_cal_dir", "spot_cal")):
                    for m, hib, _ in METRICS[:4]:
                        self.compare(comb, ["scenario", "seed"], a, b, m, hib, "E10", dict(scenario="pooled"), note="optimizer seeds only (descriptive)")
            e13 = self.get("E13")
            if e13 is not None and e3 is not None:
                comb = pd.concat([e3.assign(prep="gauss_rank"), e13], ignore_index=True)
                rows = []
                for (pp, v), g in comb[comb["variant"].isin(["abic", "spot_cal"])].groupby(["prep", "variant"]):
                    rows.append(dict(preprocessing=pp, method=lab(v), units=len(g), **{name: ms(g[m].values, 2 if m in (M_SHD, M_T) else 3) for m, _, name in METRICS[:4]}))
                self.add_table("T2f_preprocessing", pd.DataFrame(rows), "Preprocessing sensitivity (E13): gauss-rank (E3) against raw-centred", [FILES["E3"], FILES["E13"]])
        e9 = self.get("E9_ps")
        if e9 is not None:
            rows = [dict(scenario=SCEN_LABEL.get(r.scenario, r.scenario), method=lab(r.variant), skeleton_F1=f"{r.c_skeleton_f1:.3f}", arrowhead_F1=f"{r.c_arrowhead_f1:.3f}",
                         tail_F1=f"{r.c_tail_f1:.3f}", SHD=int(r.c_shd_pag), est_edges=int(r.n_est_edges), time_s=f"{r.t_wall:.1f}") for r in e9.itertuples()]
            self.add_table("T2g_fci_petshop", pd.DataFrame(rows), "FCI / RFCI on PetShop (E9): one deterministic run per scenario, algorithm and alpha (no seed spread)", [FILES["E9_ps"]])

    def sec_T1(self, ps):
        """Algorithm-wise KPI table over the settings."""
        rows = []

        def add(setting, g, unit):
            for v in g["variant"].unique():
                x = g[g["variant"] == v]
                r = dict(setting=setting, method=lab(v), unit=unit, units=len(x))
                for m, _, name in METRICS:
                    if m in x:
                        r[name] = ms(x[m].values, 2 if m in (M_SHD, M_T) else 3)
                        r[f"_{m}_mean"] = float(x[m].mean())
                rows.append(r)
        allsyn = self.get("E9_syn_all")
        if allsyn is not None:
            add("synthetic, n=1000, d=25", allsyn[allsyn["n"] == 1000], "graph")
        elif self.get("E1") is not None:
            e1 = self.get("E1")
            add("synthetic, n=1000, d=25", e1[e1["n"] == 1000], "graph")
        if ps is not None:
            e3, e4, e10 = self.get("E3"), self.get("E4"), self.get("E10")
            seeds = set(map(tuple, e3[["scenario", "seed"]].values)) if e3 is not None else set()
            use = ps[[(s, k) in seeds for s, k in zip(ps["scenario"], ps["seed"])]] if seeds else ps
            add("PetShop, optimizer seeds", use, "(scenario, seed)")
        e5 = self.get("E5")
        if e5 is not None:
            add("PetShop, block bootstrap", e5, "(scenario, resample)")
        e9 = self.get("E9_ps")
        if e9 is not None:
            add("PetShop, FCI / RFCI", e9, "scenario")
        self.add_table("T1_algorithm_kpi", pd.DataFrame(rows), "Algorithm-wise KPI table: mean ± SD over the units named in the 'unit' column", list(self.inputs))

    # ---- E7 / E8 / E12
    def sec_E7(self):
        df = self.load("E7")
        if df is None:
            return
        rows = []
        for r in df.itertuples():
            ci = r.ci95 if isinstance(r.ci95, dict) else {}
            rows.append(dict(dataset=SCEN_LABEL.get(r.dataset, r.dataset), calibrator=r.calibrator, pairs=int(r.n_pairs), true_edges=int(r.n_true_edges),
                             log_loss=f"{r.log_loss:.4f}", log_loss_CI=("[%.4f, %.4f]" % tuple(ci["log_loss"])) if "log_loss" in ci else "",
                             Brier=f"{r.brier:.4f}" if hasattr(r, "brier") else "", Brier_CI=("[%.4f, %.4f]" % tuple(ci["brier"])) if "brier" in ci else "",
                             ECE=f"{r.ece:.4f}", ECE_CI=("[%.4f, %.4f]" % tuple(ci["ece"])) if "ece" in ci else "", overconfidence=f"{r.overconfidence:+.4f}",
                             AUROC=f"{r.auroc:.3f}", AUPRC=f"{r.auprc:.3f}", below_1e3=f"{getattr(r, 'true_edges_below_0.001'):.3f}" if hasattr(r, "true_edges_below_0.001") else "",
                             below_6e3=f"{getattr(r, 'true_edges_below_0.006'):.3f}" if hasattr(r, 'true_edges_below_0.006') else "", ci_unit=r.ci_unit if isinstance(r.ci_unit, str) else ""))
        self.add_table("T7_posterior_quality", pd.DataFrame(rows), "Skeleton-posterior probability quality (E7): log-loss and Brier with ECE (CIs by bootstrap); overconfidence > 0 means more extreme than the data justify; below_x = share of true edges with p <= x", [FILES["E7"]])
        raw = df[df["calibrator"] == "none"]
        self.facts["E7"] = {r.dataset: dict(ece=r.ece, oc=r.overconfidence, ci=r.ci95 if isinstance(r.ci95, dict) else None, auroc=r.auroc, log_loss=r.log_loss) for r in raw.itertuples()}
        # F6
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharex=True, sharey=True)
        for ax, cal, title in zip(axes, ("none", "platt"), ("posterior as trained", "Platt-calibrated on simulated graphs only")):
            ax.plot([0, 1], [0, 1], color="k", lw=0.8, ls=":")
            for r in df[df["calibrator"] == cal].itertuples():
                rel = r.reliability
                m = np.array(rel["count"]) > 0
                x, y = np.array(rel["mean_pred"])[m], np.array(rel["frac_true"])[m]
                ax.plot(x, y, marker="o", ms=3.5, lw=1.3, label=f"{SCEN_LABEL.get(r.dataset, r.dataset)} (ECE {r.ece:.3f})",
                        color="#000000" if r.dataset == "simulated" else None, ls="-" if r.dataset == "simulated" else "--")
            ax.set_title(title, fontsize=8); ax.set_xlabel("mean predicted p")
            ax.legend(fontsize=6.5)
        axes[0].set_ylabel("observed fraction of true edges")
        self.save_fig(fig, "F6_reliability", "Reliability diagrams of the skeleton posterior: simulated held-out graphs (black) against PetShop scenarios. Above the diagonal = under-predicts edges.", [FILES["E7"]])

    def sec_E8(self):
        df = self.load("E8")
        if df is None:
            return
        src = [FILES["E8"]]
        rows = []
        for (q, m), g in df.groupby(["query", "model"]):
            extra = {}
            for col_, name_, nd_ in (("coverage90", "coverage_90", 3), ("coverage50", "coverage_50", 3), ("z2", "z2 (1=calibrated)", 3), ("pit_ks", "PIT_KS", 3), ("crps", "CRPS", 3)):
                if col_ in g:
                    extra[name_] = ms(g[col_].values, nd_)
            mean_, lo_, hi_ = ci95(-g["logpdf"].values)
            rows.append(dict(query=q, model=m, units=len(g), held_out_NLL=ms(-g["logpdf"].values), NLL_CI95=("[%.3f, %.3f]" % (lo_, hi_)) if lo_ == lo_ else "n/a", logpdf=ms(g["logpdf"].values), RMSE=ms(g["rmse"].values), **extra,
                             fit_time_s=ms(g["t_fit"].values, 4), graph_learn_time_s=ms(g["t_graph_learn"].values, 1),
                             _nll=float(-g["logpdf"].mean()), _rmse=float(g["rmse"].mean()), _t=float((g["t_fit"] + g["t_graph_learn"]).mean())))
        self.add_table("T5_inference", pd.DataFrame(rows), "Inference validation (E8): held-out negative log density (nats/row, lower is better) and RMSE over (scenario, split) units; query A = sink given its true ancestors, B = every service given the rest", src)
        d = df.copy()
        d["variant"] = d["model"]
        unit = ["scenario", "graph_seed", "split"]
        facts = {}
        for q in ("A", "B"):
            g = d[d["query"] == q]
            for src_ in ("truth", "abic", "spot_cal"):
                a_ = f"{src_}/admg"
                if not (g["variant"] == a_).any():
                    continue
                for b_ in (f"{src_}/dag", "saturated", "marginal"):
                    s = self.compare(g, unit, a_, b_, "logpdf", True, "E8", dict(query=q), note="held-out log density, expanding windows (windows overlap: units are not independent)")
                    facts[(q, src_, b_)] = s
        self.facts["E8"] = facts
        # F3 right panel (also drawn in sec_F3)

    def sec_E12(self):
        df = self.load("E12")
        if df is None:
            return
        src = [FILES["E12"]]
        df = df.copy()
        df["case"] = df["scenario"] + "/" + df["split"] + "/" + df["issue"].astype(str)
        df["cfg"] = df["source"] + "|" + df["method"] + "|" + df["ancestor_mode"].astype(str)
        per = df.groupby(["split", "cfg", "case"])[["top1", "top3", "rr"]].mean().reset_index()      # average over graph seeds first
        rows = []
        for split in ("all", "train", "test"):
            sub = per if split == "all" else per[per["split"] == split]
            for cfg, g in sub.groupby("cfg"):
                srcn, meth, anc = cfg.split("|")
                r = dict(split=split, graph=srcn, method=meth, ancestors=anc, cases=len(g))
                for k, name in (("top1", "top-1"), ("top3", "top-3"), ("rr", "MRR")):
                    mu, lo, hi = bootstrap_mean_ci(g[k].values)
                    r[name] = f"{mu:.2f} [{lo:.2f}, {hi:.2f}]"
                    r[f"_{k}"] = mu
                rows.append(r)
        out = pd.DataFrame(rows).sort_values(["split", "_rr"], ascending=[True, False])
        self.add_table("T6_rca", out, "Root-cause ranking (E12): top-1, top-3 and MRR with bootstrap 95% CI over cases (graph seeds averaged per case); split 'test' is the held-out split", src)
        # paired comparisons (bootstrap over cases)
        pivr = per.pivot_table(index=["split", "case"], columns="cfg", values="rr")
        pr = []

        def cmp(a, b, label):
            if a not in pivr or b not in pivr:
                return None
            t = pivr[[a, b]].dropna()
            if len(t) < 3:
                return None
            res = {}
            for split in ("all", "test"):
                tt = t if split == "all" else t[t.index.get_level_values(0) == "test"]
                if len(tt) < 3:
                    continue
                mu, lo, hi, p = bootstrap_paired_ci(tt[a].values, tt[b].values)
                res[split] = (mu, lo, hi, p, len(tt))
                pr.append(dict(split=split, comparison=label, cases=len(tt), mean_diff_MRR=mu, ci_lo=lo, ci_hi=hi, p_bootstrap=p,
                               decision="supported: first better" if lo > 0 else ("supported: first worse" if hi < 0 else "not distinguishable from noise")))
            return res
        facts = {}
        for s_ in ("truth", "abic", "spot_cal"):
            facts[("admg_vs_dag", s_)] = cmp(f"{s_}|admg|none", f"{s_}|dag|none", f"{s_} graph: ADMG vs directed-only")
            facts[("admg_vs_anomaly", s_)] = cmp(f"{s_}|admg|none", "none|anomaly|none", f"{s_} graph ADMG vs no graph (anomaly size)")
            facts[("admg_vs_corr", s_)] = cmp(f"{s_}|admg|none", "none|corr|none", f"{s_} graph ADMG vs correlation baseline")
        facts[("truth_vs_abic", "")] = cmp("truth|admg|none", "abic|admg|none", "ADMG ranking: true graph vs ABIC graph")
        facts[("truth_vs_spot", "")] = cmp("truth|admg|none", "spot_cal|admg|none", "ADMG ranking: true graph vs SPOT-cal graph")
        facts[("spot_vs_abic", "")] = cmp("spot_cal|admg|none", "abic|admg|none", "ADMG ranking: SPOT-cal graph vs ABIC graph")
        self.add_table("T6b_rca_paired", pd.DataFrame(pr), "Root-cause ranking, paired differences in reciprocal rank (bootstrap over cases; first minus second)", src)
        self.facts["E12"] = facts

    # ---- F3 and H-facts across experiments
    def sec_F3(self):
        e6, e3, e8 = self.get("E6"), self.get("E3"), self.get("E8")
        if (e6 is None or e3 is None) and e8 is None:
            return
        fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.1))
        if e6 is not None and e3 is not None:
            for v in ("abic", "spot_cal"):
                pts = [(bd, ci95(g[M_SKEL].values), g["wall"].mean() if "wall" in g else bd) for bd, g in e6[e6["variant"] == v].groupby("budget")]
                g3 = e3[e3["variant"] == v]
                pts.append((np.inf, ci95(g3[M_SKEL].values), float(g3["t_total"].mean())))
                x = [p[2] for p in pts]; mu = np.array([p[1][0] for p in pts]); lo = np.array([p[1][1] for p in pts]); hi = np.array([p[1][2] for p in pts])
                ax[0].errorbar(x, mu, yerr=[np.nan_to_num(mu - lo), np.nan_to_num(hi - mu)], marker="o", capsize=3, color=col(v), label=lab(v), lw=1.3)
            ax[0].set_xlabel("wall-clock time per run (s): budgets 60, 120 s and unconstrained")
            ax[0].set_ylabel("skeleton F1"); ax[0].legend(fontsize=7)
        if e8 is not None:
            d = e8[e8["query"] == "B"]
            for (m, st, srcn), g in d.groupby(["model", "structure", "source"]):
                t = float((g["t_fit"] + g["t_graph_learn"]).mean())
                ax[1].scatter(t + 1e-3, -g["logpdf"].mean(), color={"admg": col("spot_cal"), "dag": col("abic"), "saturated": "#009E73", "marginal": "#999999"}.get(st, "k"),
                              marker={"truth": "s", "abic": "o", "spot_cal": "^", "none": "D"}.get(srcn, "o"), s=26, label=m)
            ax[1].set_xscale("log"); ax[1].set_xlabel("fit + graph-learning time (s)"); ax[1].set_ylabel("held-out NLL (query B), lower is better")
            ax[1].legend(fontsize=5.5, ncol=2)
        self.save_fig(fig, "F3_accuracy_and_inference_vs_time", "Left: skeleton F1 against wall-clock time at equal budgets (E6) and unconstrained (E3). Right: held-out inference error against computational time (E8).", [FILES["E6"], FILES["E3"], FILES["E8"]])

    # ---- E11
    def sec_E11(self):
        if not self.petshop_root:
            return
        try:
            from pgm.petshop import load_petshop
            from pgm.rca import admg_from_json
        except Exception as e:                                           # noqa: BLE001
            self.notes.append(f"E11 skipped: {e}")
            return
        gdir = os.path.join(self.results, "rca_graphs")
        if not os.path.isdir(gdir):
            self.notes.append("E11 skipped: results/rca_graphs missing (run E12 first)")
            return
        rows, shown = [], None
        for scn in ("low_traffic", "high_traffic", "temporal_traffic1", "temporal_traffic2"):
            try:
                ps = load_petshop(scn, self.petshop_root)
            except Exception as e:                                       # noqa: BLE001
                self.notes.append(f"E11: PetShop scenario {scn} could not be loaded ({type(e).__name__}: {str(e)[:120]}); skipped")
                continue
            names = ps.names
            idx = {v: i for i, v in enumerate(names)}
            tr_skel = set(ps.pag.skeleton())
            tr_dir = {(a, b) for a, b in ps.mag.di_edges}
            tr_bi = {frozenset(e) for e in ps.mag.bi_edges}
            import networkx as nx
            G = nx.Graph([(a, b) for a, b in ps.dag_edges])
            for src in ("abic", "spot_cal"):
                p = os.path.join(gdir, f"{scn}_{src}_gauss_rank_seed0.json")
                if not os.path.exists(p):
                    continue
                A = admg_from_json(json.load(open(p))["admg"])
                est_dir = {(a, b) for a, b in A.di_edges}
                est_bi = {frozenset(e) for e in A.bi_edges}
                est_skel = {frozenset(e) for e in est_dir} | est_bi
                def dist(e):
                    a, b = tuple(e)
                    try:
                        return nx.shortest_path_length(G, a, b)
                    except Exception:                                   # noqa: BLE001
                        return -1
                spurious = est_skel - tr_skel
                hit = est_skel & tr_skel
                missed = tr_skel - est_skel
                disguised = [e for e in est_bi if e in {frozenset(x) for x in tr_dir}]
                rows.append(dict(scenario=SCEN_LABEL.get(scn, scn), graph=lab(src), true_edges=len(tr_skel), found=len(hit), missed=len(missed), spurious=len(spurious),
                                 est_directed=len(est_dir), est_bidirected=len(est_bi),
                                 bidirected_on_true_directed_pair=len(disguised), bidirected_spurious=len([e for e in est_bi if e not in tr_skel]),
                                 directed_correct=len([e for e in est_dir if e in tr_dir]), directed_reversed=len([e for e in est_dir if (e[1], e[0]) in tr_dir]),
                                 median_call_distance_spurious=float(np.median([dist(e) for e in spurious])) if spurious else float("nan"),
                                 median_call_distance_missed=float(np.median([dist(e) for e in missed])) if missed else float("nan")))
                if shown is None and src == "spot_cal":
                    shown = (scn, names, tr_skel, est_dir, est_bi, tr_dir)
        self.add_table("T8_error_analysis", pd.DataFrame(rows), "Edge-level error analysis (E11), seed-0 graphs: found / missed / spurious adjacencies, edge types, bidirected edges that sit on true directed pairs, and call-graph distance (hops in the undirected call graph, -1 = unreachable)", ["rca_graphs/*.json", "PetShop graph.csv"])
        if shown:
            scn, names, tr_skel, est_dir, est_bi, tr_dir = shown
            d = len(names); idx = {v: i for i, v in enumerate(names)}
            M = np.zeros((d, d))
            for e in tr_skel:
                a, b = tuple(e); M[idx[a], idx[b]] = M[idx[b], idx[a]] = 1                    # missed (until found)
            est = {frozenset(e) for e in est_dir} | est_bi
            for e in est:
                a, b = tuple(e)
                M[idx[a], idx[b]] = M[idx[b], idx[a]] = 2 if e in tr_skel else 3                # 2 found, 3 spurious
            from matplotlib.colors import ListedColormap
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            ax.imshow(M, cmap=ListedColormap(["#ffffff", "#E69F00", "#009E73", "#D55E00"]), vmin=0, vmax=3)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"{SCEN_LABEL.get(scn, scn)}, SPOT-cal graph (seed 0)", fontsize=8)
            from matplotlib.patches import Patch
            ax.legend(handles=[Patch(color="#009E73", label="found"), Patch(color="#E69F00", label="missed"), Patch(color="#D55E00", label="spurious")], fontsize=7, loc="upper right")
            self.save_fig(fig, "F7_error_map", "Edge-level error map of the SPOT-cal graph for one scenario (services on both axes, same order): found, missed and spurious adjacencies.", ["rca_graphs/*.json"])

    # ============================================================================================ scorecard
    def decide_row(self, label, value, verdict):
        return dict(part=label, evidence=value, verdict=verdict)

    def sc(self, hid, statement, parts, verdict):
        self.scorecard.append(dict(hypothesis=hid, statement=statement, verdict=verdict, evidence=" | ".join(f"{p['part']}: {p['evidence']} [{p['verdict']}]" for p in parts)))

    @staticmethod
    def rl(v):
        return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"

    def fmt_s(self, s):
        if not s:
            return "no data"
        lo, hi = s.get("ci_lo"), s.get("ci_hi")
        return (f"diff {s['mean_diff']:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}], n={s['n']}, p_t={s.get('p_ttest', float('nan')):.3g}, p_W={s.get('p_wilcoxon', float('nan')):.3g}")

    def build_scorecard(self):
        NR = "not run / incomplete"
        # H1
        parts = []
        s1 = next((r for r in self.paired_rows if r["exp"] == "E1" and r["n"] == "pooled"), None)
        if s1:
            sd = dict(n=s1["units"], mean_diff=s1["mean_diff"], ci_lo=s1["ci_lo"], ci_hi=s1["ci_hi"], p_ttest=s1["p_ttest"], p_wilcoxon=s1["p_wilcoxon"])
            parts.append(self.decide_row("SPOT-cal vs ABIC skeleton F1, pooled over n (E1)", self.fmt_s(sd), s1["decision"]))
            slope = self.facts.get("E1_slope")
            p_int = self.facts.get("E1_anova", {}).get(("skeleton F1", "method x sample size"))
            shrink = bool(slope and slope["hi"] < 0 and p_int is not None and p_int < 0.05)
            parts.append(self.decide_row("gain shrinks with n (slope of paired diff on log2 n; ANOVA interaction)",
                                         (f"slope {slope['slope']:+.4f} per doubling, 95% CI [{slope['lo']:+.4f}, {slope['hi']:+.4f}]; interaction p={p_int:.3g}; gain by n {self.facts.get('E1_gain_by_n')}" if slope and p_int is not None else "n/a"),
                                         "supported" if shrink else "not distinguishable from noise"))
            if s1["decision"] == "supported: first better":
                v = "supported" if shrink else "partly supported (gain shown, shrinkage not shown)"
            elif s1["decision"] == "supported: first worse":
                v = "contradicted"
            else:
                v = "not distinguishable from noise"
        else:
            v = NR
        self.sc("H1", "SPOT improves skeleton accuracy over ABIC on simulated graphs, and the gain shrinks as n grows", parts, v)
        # H2
        parts, v = [], NR
        e5 = self.facts.get("E5_pooled")
        d1 = self.facts.get("E1_diffs_n1000")
        d5 = self.facts.get("E5_diffs")
        if e5 and d1 is not None and len(d1) > 1 and d5 is not None and len(d5) > 1:
            parts.append(self.decide_row("PetShop gain, skeleton F1 (E5 pooled, percentile CI)", self.fmt_s(dict(e5, ci_lo=e5.get("ci_pct_lo"), ci_hi=e5.get("ci_pct_hi"))), e5["decision"]))
            parts.append(self.decide_row("synthetic gain at n=1000 (E1)", f"mean {np.mean(d1):+.3f} over {len(d1)} graphs", "descriptive"))
            w = welch_diff(d5, d1)
            sup = bool(w.get("p", 1) < 0.05 and w.get("ci_hi", 1) < 0)
            parts.append(self.decide_row("PetShop gain minus synthetic gain (Welch)", f"diff {w['mean_diff']:+.3f}, 95% CI [{w['ci_lo']:+.3f}, {w['ci_hi']:+.3f}], p={w['p']:.3g}", "supported" if sup else "not distinguishable from noise"))
            e7 = self.facts.get("E7", {})
            ps_auc = [x["auroc"] for k, x in e7.items() if k != "simulated"]
            if ps_auc and "simulated" in e7:
                parts.append(self.decide_row("mechanism: posterior AUROC PetShop vs simulated (E7, descriptive)", f"PetShop mean {np.mean(ps_auc):.3f} vs simulated {e7['simulated']['auroc']:.3f}", "descriptive"))
            v = "supported" if sup else "not distinguishable from noise"
        self.sc("H2", "On PetShop the gain is smaller than on synthetic data, because the posterior is out of distribution there", parts, v)
        # H3
        h3 = self.facts.get("H3", {})
        parts, v = [], NR
        if "spot_floor_only" in h3 and "spot_nofloor" in h3:
            for k, name in (("spot_floor_only", "floor only vs ABIC, optimizer time ratio"), ("spot_nofloor", "mask only vs ABIC, optimizer time ratio"), ("spot_cal", "full SPOT-cal vs ABIC, optimizer time ratio")):
                if k in h3:
                    x = h3[k]
                    parts.append(self.decide_row(name, f"ratio {x['ratio']:.2f}, 95% CI [{x['lo']:.2f}, {x['hi']:.2f}], n={x['n']}", x["decision"]))
            if "share_floor" in h3:
                parts.append(self.decide_row("median share of the speed-up recovered by the floor alone", f"{h3['share_floor']:.2f}", "descriptive"))
            fo, mo = h3["spot_floor_only"], h3["spot_nofloor"]
            floor_fast = fo["decision"] == "supported: first better"
            mask_fast = mo["decision"] == "supported: first better"
            v = "supported" if (floor_fast and not mask_fast) else ("partly supported (both contribute)" if (floor_fast and mask_fast) else
                                                                    ("contradicted (mask alone is faster, floor alone is not)" if mask_fast else "not distinguishable from noise"))
        self.sc("H3", "SPOT's speed-up comes mainly from the hard floor (pruning), not from the stochastic mask", parts, v)
        # H4
        e7 = self.facts.get("E7", {})
        parts, v = [], NR
        if "simulated" in e7 and len(e7) > 1:
            sim = e7["simulated"]
            sim_ok = sim["ci"] is not None and sim["ci"]["ece"][1] <= 0.02
            parts.append(self.decide_row("simulated: ECE (upper CI bound <= 0.02 counts as well calibrated)", f"ECE {sim['ece']:.4f}, CI {self.rl(sim['ci']['ece']) if sim['ci'] else 'n/a'}, over-confidence {sim['oc']:+.4f}", "calibrated" if sim_ok else "not shown"))
            over = []
            for k, x in e7.items():
                if k == "simulated" or not x["ci"]:
                    continue
                oc = x["ci"]["overconfidence"][0] > 0 and x["ci"]["ece"][0] > sim["ci"]["ece"][1] if sim["ci"] else False
                over.append(oc)
                parts.append(self.decide_row(f"PetShop {SCEN_LABEL.get(k, k)}: ECE above simulated CI and over-confidence CI > 0", f"ECE {x['ece']:.4f}, CI {self.rl(x['ci']['ece'])}, over-confidence {x['oc']:+.4f}, CI {self.rl(x['ci']['overconfidence'])}", "over-confident" if oc else "not shown"))
            v = "supported" if (sim_ok and over and all(over)) else ("partly supported" if (over and any(over)) or (sim_ok and not any(over)) else "not distinguishable from noise")
            if sim_ok and over and not any(over):
                v = "not supported on PetShop"
        self.sc("H4", "The posterior is well calibrated on simulated data and over-confident on PetShop", parts, v)
        # H5
        f8, f12 = self.facts.get("E8", {}), self.facts.get("E12", {})
        parts, v, ok_a, ok_b, ok_c = [], NR, None, None, None
        if f8:
            oks, noninf = [], []
            for src_ in ("truth", "abic", "spot_cal"):
                s = f8.get(("B", src_, "marginal"))
                if s:
                    parts.append(self.decide_row(f"E8 query B, {lab(src_) if src_ != 'truth' else 'true graph'} ADMG vs marginal (held-out log density)", self.fmt_s(s), s["decision"]))
                    oks.append(s["decision"] == "supported: first better")
                s = f8.get(("B", src_, "saturated"))
                if s:
                    ni = s["ci_lo"] is not None and s["ci_lo"] > -0.05
                    parts.append(self.decide_row(f"E8 query B, {src_} ADMG vs saturated Gaussian, non-inferiority margin -0.05 nats", self.fmt_s(s), "no worse" if ni else "worse or inconclusive"))
                    noninf.append(ni)
            ok_a, ok_b = (bool(oks) and all(oks)), (bool(noninf) and all(noninf))
        if f12:
            def get(k, split="all"):
                r = f12.get(k)
                return r.get(split) if r else None
            t_an = get(("admg_vs_anomaly", "truth"))
            l_an = [get(("admg_vs_anomaly", s_)) for s_ in ("abic", "spot_cal")]
            tl = get(("truth_vs_abic", ""))
            if t_an:
                parts.append(self.decide_row("E12 all cases: true-graph ADMG vs no graph, MRR diff", f"{t_an[0]:+.3f} [{t_an[1]:+.3f}, {t_an[2]:+.3f}], n={t_an[4]}", "better" if t_an[1] > 0 else "not distinguishable"))
            for s_, l in zip(("ABIC", "SPOT-cal"), l_an):
                if l:
                    parts.append(self.decide_row(f"E12 all cases: {s_}-graph ADMG vs no graph, MRR diff", f"{l[0]:+.3f} [{l[1]:+.3f}, {l[2]:+.3f}], n={l[4]}", "better" if l[1] > 0 else ("worse" if l[2] < 0 else "not distinguishable")))
            if tl:
                parts.append(self.decide_row("E12 all cases: true-graph vs ABIC-graph ADMG, MRR diff", f"{tl[0]:+.3f} [{tl[1]:+.3f}, {tl[2]:+.3f}], n={tl[4]}", "true better" if tl[1] > 0 else "not distinguishable"))
            if t_an:
                ok_c = bool(t_an[1] > 0 and all((l is None) or (l[1] <= 0) for l in l_an))
        if parts:
            known = [x for x in (ok_a, ok_b, ok_c) if x is not None]
            v = "supported" if (known and all(known) and len(known) == 3) else ("partly supported" if any(known) else "not distinguishable from noise")
            if len(known) < 3 and all(known):
                v = f"partly supported ({3 - len(known)} of 3 parts not run)"
        self.sc("H5", "A fitted ADMG predicts held-out data better than a no-structure baseline and is no worse than a saturated Gaussian; structure helps RCA only if it is accurate", parts, v)
        df = pd.DataFrame(self.scorecard)
        self.add_table("T9_scorecard", df[["hypothesis", "statement", "verdict", "evidence"]], "Hypothesis scorecard, computed from the result files with the decision rule fixed in the plan", list(self.inputs))
        with open(os.path.join(self.out, "scorecard.md"), "w") as f:
            for r in self.scorecard:
                f.write(f"## {r['hypothesis']}: {r['statement']}\n\n**{r['verdict']}**\n\n")
                for p in r["evidence"].split(" | "):
                    f.write(f"- {p}\n")
                f.write("\n")
        df.to_csv(os.path.join(self.out, "scorecard.csv"), index=False)

    def paired_table(self):
        df = pd.DataFrame(self.paired_rows)
        if not len(df):
            return
        df["p_ttest_holm"] = holm(df["p_ttest"].values)
        self.add_table("T3_paired_tests", df, "All paired comparisons. Decision rule: paired t p<0.05 and Wilcoxon p<0.05 and the 95% CI excludes 0 (Holm-adjusted t-test p listed)", list(self.inputs))

    # ============================================================================================ run
    def run(self):
        t0 = time.time()
        self.sec_E1()
        self.sec_E2()
        self.sec_PetShop()
        self.sec_E7()
        self.sec_E8()
        self.sec_E12()
        self.sec_F3()
        self.sec_E11()
        self.paired_table()
        self.build_scorecard()
        self.completeness(os.path.join(self.results, "queue_failures.jsonl"))
        with open(os.path.join(self.out, "tables.md"), "w") as f:
            for name in sorted(self.tables):
                df, cap, src = self.tables[name]
                f.write(f"### {name}: {cap}\n\n_source: {', '.join(src)}_\n\n")
                f.write(df[[c for c in df.columns if not c.startswith('_')]].to_markdown(index=False, floatfmt=".3f") + "\n\n")
            f.write("### Figures\n\n" + "\n".join(f"- **{x['name']}**: {x['caption']} (source: {', '.join(x['sources'])})" for x in self.figures) + "\n")
        manifest = dict(created=time.strftime("%F %T"), results_dir=self.results, cut=self.cut, inputs=self.inputs, tables=sorted(self.tables),
                        figures=[x["name"] for x in self.figures], notes=self.notes, seconds=round(time.time() - t0, 1),
                        decision_rule="paired t p<0.05 and Wilcoxon p<0.05 and 95% CI excludes 0")
        json.dump(manifest, open(os.path.join(self.out, "manifest.json"), "w"), indent=1, default=str)
        return manifest


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cut", action="store_true", help="the queue was run with --cut (completeness is judged against the reduced sizes)")
    ap.add_argument("--petshop-root", default=None, help="PetShop dataset dir: enables the E11 error analysis and figure F7")
    a = ap.parse_args(argv)
    agg = Agg(a.results_dir, a.out_dir or os.path.join(a.results_dir, "aggregate"), cut=a.cut, petshop_root=a.petshop_root or os.environ.get("PETSHOP_ROOT"))
    m = agg.run()
    print(f"wrote {len(m['tables'])} tables and {len(m['figures'])} figures to {agg.out}")
    print(pd.DataFrame(agg.scorecard)[["hypothesis", "verdict"]].to_string(index=False) if agg.scorecard else "")
    t0 = agg.tables.get("T0_completeness")
    if t0 is not None:
        print(t0[0][["experiment", "jobs_planned", "jobs_finished", "status"]].to_string(index=False))
    return agg


if __name__ == "__main__":
    main()
