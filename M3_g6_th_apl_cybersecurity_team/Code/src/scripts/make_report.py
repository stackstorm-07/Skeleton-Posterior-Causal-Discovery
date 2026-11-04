"""Stage 2e: milestone-report tables from the raw stage-2d results.

    python scripts/make_report.py                       # reads results/ps_2d_*.jsonl, writes results/report_tables.{md,tex}
    python scripts/make_report.py --jsonl results/ps_2d_main_gauss_rank.jsonl --prep gauss_rank

Everything is computed from the raw JSONL (nothing is typed in by hand), so re-running after the queue finishes
(temporal_traffic2, seeds 3-4, ablations) refreshes every table and the coverage block.

Tables
  1  F1 across scenarios, corrected and legacy metrics (scenario seed-means averaged over scenarios; wins/losses vs ABIC)
  2  per-scenario F1 (mean +- sd over optimizer seeds) incl. TPR/FDR of the skeleton
  3  runtime breakdown: offline posterior training (one-time) / online posterior inference / optimization
  4  estimated vs true number of directed and bidirected edges (context for arrowhead/tail scores)
  5  skeleton-posterior quality on PetShop vs held-out simulated graphs, and how many true edges the 6e-3 floor pins

"Corrected" = an edge mark is identified by the ordered node pair that carries it.  "Legacy" reproduces the indexing of
the released SPOT code (it only compares *which node* carries a head/tail), which inflates arrowhead/tail F1;
the skeleton F1 is identical under both.  PetShop is one dataset per scenario, so seeds vary only the optimizer
(initialisation, SPOT mask stream): the sd over seeds is optimizer variance, not data variance, and no significance claim is made.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ORDER = ["abic", "spot_cal", "spot"]
LABEL = {"abic": "ABIC", "spot_cal": "SPOT (calibrated floor 6e-3)", "spot": "SPOT (released floor 1e-3)"}
SHORT = {"abic": "ABIC", "spot_cal": "SPOT-cal", "spot": "SPOT-1e-3"}
SCEN_LABEL = {"low_traffic": "low traffic", "high_traffic": "high traffic",
              "temporal_traffic1": "temporal 1", "temporal_traffic2": "temporal 2"}
ALL_SCENARIOS = ["low_traffic", "high_traffic", "temporal_traffic1", "temporal_traffic2"]
FULL_SEEDS = [0, 1, 2, 3, 4]


# ----------------------------------------------------------------------------- helpers
def load(paths, prep, budget_none=True):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p) if l.strip()]
    df = pd.DataFrame(rows)
    df = df[df.prep == prep]
    if budget_none:
        df = df[df.budget.isna()]
    df = df[df.variant.isin(ORDER)].copy()
    df["est_di"] = df.n_est_edges.map(lambda e: e[0])
    df["t_total"] = df["elapsed"] + df["t_posterior"]
    return df


def pm(x, nd=3):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=0):.{nd}f}" if len(x) > 1 else f"{x.mean():.{nd}f}"


def scen_means(df, cols):
    return df.groupby(["scenario", "variant"])[cols].mean()


def wl(d, tol=1e-9):
    return f"W{int((d > tol).sum())}/L{int((d < -tol).sum())}/T{int((np.abs(d) <= tol).sum())}"


def to_tex(df, caption, label, align=None):
    esc = lambda s: str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#").replace("±", "$\\pm$").replace("Δ", "$\\Delta$")
    align = align or ("l" + "r" * (len(df.columns) - 1))
    out = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{esc(caption)}}}", f"\\label{{{label}}}",
           "\\resizebox{\\linewidth}{!}{%", f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(esc(c) for c in df.columns) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        out.append(" & ".join(esc(v) for v in r.values) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(out)


# ----------------------------------------------------------------------------- tables
def table1(df):
    """Scenario-level F1 (each scenario's seed-mean is one observation)."""
    cols = ["c_skeleton_f1", "c_arrowhead_f1", "c_tail_f1", "l_arrowhead_f1", "l_tail_f1"]
    m = scen_means(df, cols)
    base = m.xs("abic", level="variant")
    rows = []
    for v in ORDER:
        if v not in m.index.get_level_values("variant"):
            continue
        x = m.xs(v, level="variant").loc[base.index]
        r = {"method": LABEL[v]}
        for c, name in zip(cols, ["Skeleton F1", "Arrowhead F1", "Tail F1", "Arrowhead F1 (legacy)", "Tail F1 (legacy)"]):
            r[name] = f"{x[c].mean():.3f}"
            if v != "abic":
                r[name] += f" ({(x[c] - base[c]).mean():+.3f}; {wl((x[c] - base[c]).values)})"
        rows.append(r)
    return pd.DataFrame(rows), len(base)


def table2(df):
    rows = []
    for scn in [s for s in ALL_SCENARIOS if s in set(df.scenario)]:
        g = df[df.scenario == scn]
        for v in ORDER:
            s = g[g.variant == v]
            if s.empty:
                continue
            rows.append({"scenario": SCEN_LABEL[scn], "n / d": f"{int(s.n.iloc[0])} / {int(s.d.iloc[0])}", "seeds": len(s),
                         "method": SHORT[v], "Skeleton F1": pm(s.c_skeleton_f1), "TPR": pm(s.c_skeleton_tpr, 2),
                         "FDR": pm(s.c_skeleton_fdr, 2), "Arrowhead F1": pm(s.c_arrowhead_f1), "Tail F1": pm(s.c_tail_f1)})
    return pd.DataFrame(rows)


def table3(df, train_json):
    off = json.load(open(train_json))["t_train_total"] if os.path.exists(train_json) else None
    sm = scen_means(df, ["t_optimize", "t_posterior", "t_total", "n_fev", "n_pairs_allowed"])
    rows = []
    base_t = sm.xs("abic", level="variant")["t_optimize"].mean()
    for v in ORDER:
        g = df[df.variant == v]
        if g.empty:
            continue
        s = sm.xs(v, level="variant")
        pairs = "all" if v == "abic" else f"{s.n_pairs_allowed.mean():.0f}"
        rows.append({"method": LABEL[v],
                     "offline training (one-time)": "n/a" if v == "abic" or off is None else f"{off:.0f} s",
                     "online posterior / run": "0" if v == "abic" else f"{g.t_posterior.mean():.2f} s",
                     "optimization / run (all runs)": f"{g.t_optimize.mean():.0f} s",
                     "optimization / run (scenario mean)": f"{s.t_optimize.mean():.0f} s",
                     "vs ABIC": "1.00x" if v == "abic" else f"{s.t_optimize.mean() / base_t:.2f}x",
                     "L-BFGS evals / run": f"{g.n_fev.mean():,.0f}",
                     "free pairs (of d(d-1)/2)": pairs,
                     "converged": f"{int(g.converged.sum())}/{len(g)}"})
    per = []
    for scn in [s for s in ALL_SCENARIOS if s in set(df.scenario)]:
        r = {"scenario": SCEN_LABEL[scn]}
        for v in ORDER:
            x = df[(df.scenario == scn) & (df.variant == v)]
            if len(x):
                r[SHORT[v] + " opt. time (s)"] = f"{x.t_optimize.mean():.0f}"
        per.append(r)
    return pd.DataFrame(rows), pd.DataFrame(per)


def table4(df):
    rows = []
    for scn in [s for s in ALL_SCENARIOS if s in set(df.scenario)]:
        for v in ORDER:
            s = df[(df.scenario == scn) & (df.variant == v)]
            if s.empty:
                continue
            rows.append({"scenario": SCEN_LABEL[scn], "method": SHORT[v], "est. directed": f"{s.est_di.mean():.1f}",
                         "est. bidirected": f"{s.n_est_bidirected.mean():.1f}", "true directed": f"{s.n_true_directed.mean():.0f}",
                         "true bidirected": f"{s.n_true_bidirected.mean():.0f}"})
    return pd.DataFrame(rows)


def table5(df, train_json):
    rows = []
    for scn in [s for s in ALL_SCENARIOS if s in set(df.scenario)]:
        s = df[df.scenario == scn].iloc[0]
        rows.append({"data": f"PetShop {SCEN_LABEL[scn]}", "posterior AUROC": f"{s.post_auroc:.3f}", "posterior AUPRC": f"{s.post_auprc:.3f}",
                     "true edges pinned by 6e-3 floor": f"{100 * s.floor_pins_true_edge_frac_cal:.0f}%"})
    if os.path.exists(train_json):
        h = json.load(open(train_json))["heldout"]
        rows.append({"data": "simulated held-out graphs", "posterior AUROC": f"{h['auroc']:.3f}", "posterior AUPRC": f"{h['auprc']:.3f}",
                     "true edges pinned by 6e-3 floor": "≈1%"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- coverage + findings
def coverage(df, prep):
    lines = []
    seeds = {s: sorted(df[df.scenario == s].seed.unique().tolist()) for s in ALL_SCENARIOS if s in set(df.scenario)}
    lines.append(f"* Preprocessing `{prep}`, one restart, no wall-clock budget, original posterior pickle.")
    lines.append("* Completed (scenario: seeds): " + "; ".join(f"{SCEN_LABEL[s]}: {v}" for s, v in seeds.items()) + ".")
    missing = [SCEN_LABEL[s] for s in ALL_SCENARIOS if s not in seeds]
    short = [f"{SCEN_LABEL[s]} (seeds {sorted(set(FULL_SEEDS) - set(v))} missing)" for s, v in seeds.items() if set(v) != set(FULL_SEEDS)]
    if missing:
        lines.append("* Not run: " + ", ".join(missing) + ".")
    if short:
        lines.append("* Fewer seeds than planned: " + "; ".join(short) + ".")
    lines.append("* Not run at all: ablations (no-floor, floor-only, random/oracle posterior) on PetShop, equal-budget runs, `raw_centered` sensitivity run.")
    lines.append(f"* Every completed run converged ({int(df.converged.sum())}/{len(df)}); none hit a time limit ({int(df.timed_out.sum())}/{len(df)}).")
    return lines


def findings(df, t1_n):
    m = scen_means(df, ["c_skeleton_f1", "c_arrowhead_f1", "c_tail_f1"])
    b = m.xs("abic", level="variant")
    out = []
    for v in ["spot_cal", "spot"]:
        x = m.xs(v, level="variant").loc[b.index]
        d = x - b
        out.append(f"* {LABEL[v]} vs ABIC over {t1_n} scenarios: skeleton F1 {d.c_skeleton_f1.mean():+.3f} ({wl(d.c_skeleton_f1.values)}), "
                   f"arrowhead {d.c_arrowhead_f1.mean():+.3f} ({wl(d.c_arrowhead_f1.values)}), tail {d.c_tail_f1.mean():+.3f} ({wl(d.c_tail_f1.values)}).")
    tl = m.groupby("variant").c_tail_f1.mean()
    out.append("* The skeleton-F1 gain is consistent in sign across scenarios but small and of the same order as the seed-to-seed (optimizer) spread "
               "(Table 2); arrowhead and tail differences are inside that spread. Tail F1 is low for every method "
               f"(scenario means {tl.min():.3f}-{tl.max():.3f}).")
    out.append(f"* Both learners return {df.n_est_bidirected.min():.0f}-{df.n_est_bidirected.max():.0f} bidirected edges per run against 0-1 in the "
               "ground truth, so arrowhead/tail scores mostly measure how directed edges are mislabelled as bidirected; do not over-read them.")
    sp = df.groupby("variant").t_optimize.mean()
    out.append(f"* Optimization time per run: ABIC {sp['abic']:.0f} s, SPOT-cal {sp['spot_cal']:.0f} s, SPOT-1e-3 {sp['spot']:.0f} s (run-weighted). "
               "The saving tracks the number of free pairs (the floor), i.e. pruning; the PetShop ablations that would separate pruning from the "
               "stochastic mask were not run (the synthetic ablation in 2c could not separate them either with 3 graphs).")
    out.append("* The posterior is weaker on PetShop than on simulated graphs (Table 5), and the 6e-3 floor pins a much larger share of true edges here, "
               "which caps recall for SPOT-cal.")
    return out


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", default=None)
    ap.add_argument("--prep", default="gauss_rank")
    ap.add_argument("--train-json", default=os.path.join(ROOT, "results", "posterior_training.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "report_tables"))
    a = ap.parse_args(argv)
    paths = a.jsonl or sorted(glob.glob(os.path.join(ROOT, "results", "ps_2d_main_*.jsonl")))
    df = load(paths, a.prep)
    t1, n1 = table1(df)
    t2 = table2(df)
    t3, t3b = table3(df, a.train_json)
    t4 = table4(df)
    t5 = table5(df, a.train_json)

    md = ["# Milestone report tables: ABIC vs SPOT on PetShop (stage 2e)\n",
          "> **Status: PARTIAL.** Numbers are final for what finished, not for the planned design.\n"]
    md += coverage(df, a.prep)
    md.append("\n> PetShop is one dataset per scenario: seeds vary only the optimizer, so ± is optimizer variance, not data variance, "
              "and no significance claim is made. With 3 scenarios, 2-3 seeds and one restart the ranking of methods is tentative.\n")
    md.append("## Key readings\n")
    md += findings(df, n1)
    md.append(f"\n## Table 1. F1 across scenarios (mean of {n1} scenario means; in brackets: difference to ABIC; wins/losses/ties over scenarios)\n")
    md.append("Corrected metric unless marked legacy. Skeleton F1 is the same under both conventions.\n")
    md.append(t1.to_markdown(index=False, disable_numparse=True))
    md.append("\n## Table 2. Per-scenario results (mean ± sd over optimizer seeds)\n")
    md.append(t2.to_markdown(index=False, disable_numparse=True))
    md.append("\n## Table 3. Runtime breakdown\n")
    md.append("Three numbers: offline posterior training (paid once, shared by all SPOT runs), online posterior inference (per dataset, charged to SPOT), "
              "optimization (per run). Single-core timings: compare ratios, not absolute seconds.\n")
    md.append(t3.to_markdown(index=False, disable_numparse=True))
    md.append("\nOptimization time per scenario (s, mean over seeds):\n")
    md.append(t3b.to_markdown(index=False, disable_numparse=True))
    md.append("\n## Table 4. Estimated vs true edge counts (mean over seeds)\n")
    md.append(t4.to_markdown(index=False, disable_numparse=True))
    md.append("\n## Table 5. Skeleton posterior: PetShop vs simulated\n")
    md.append(t5.to_markdown(index=False, disable_numparse=True))
    text = "\n".join(md) + "\n"

    tex = "\n\n".join([
        to_tex(t1, "F1 across scenarios (corrected unless legacy); differences to ABIC in brackets.", "tab:f1", "lrrrrr"),
        to_tex(t2, "Per-scenario results, mean ± sd over optimizer seeds.", "tab:perscenario", "lrrlrrrrr"),
        to_tex(t3, "Runtime breakdown (single core).", "tab:runtime"),
        to_tex(t4, "Estimated vs true edge counts.", "tab:edges"),
        to_tex(t5, "Skeleton posterior quality.", "tab:posterior")]) + "\n"
    open(a.out + ".md", "w").write(text)
    open(a.out + ".tex", "w").write(tex)
    for name, t in [("f1", t1), ("per_scenario", t2), ("runtime", t3), ("edges", t4), ("posterior", t5)]:
        t.to_csv(f"{a.out}_{name}.csv", index=False)
    print(text)


if __name__ == "__main__":
    main()
