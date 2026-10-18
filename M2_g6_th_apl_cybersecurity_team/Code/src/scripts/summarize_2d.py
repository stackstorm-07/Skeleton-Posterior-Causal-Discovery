"""Aggregate the stage-2d PetShop JSONL results into markdown + CSV tables.

    python scripts/summarize_2d.py results/ps_2d_*.jsonl --out results/ps_2d_summary.md

Per (preprocessing, budget): one block per scenario (mean +- sd over optimizer seeds, paired difference to ABIC), then a
scenario-level block (each scenario's seed-mean is one observation; with 4 scenarios no p-value is meaningful, so only the
mean difference and wins/losses over scenarios are given), and the three-number runtime breakdown
(offline posterior training, online posterior inference, optimization).

IMPORTANT: PetShop is one dataset per scenario, so the spread over seeds is optimizer variance only.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from summarize_2c import LEGACY, METRICS, ms, runtime_table, table      # noqa: E402

KEYS = ["c_skeleton_f1", "c_arrowhead_f1", "c_tail_f1", "c_skeleton_tpr", "c_skeleton_fdr", "c_arrowhead_tpr",
        "c_arrowhead_fdr", "c_tail_tpr", "c_tail_fdr"]


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    df = pd.DataFrame(rows)
    df["t_total"] = df["elapsed"] + df["t_posterior"]
    return df


def scenario_level(df, cols, base="abic"):
    """Each scenario's seed-mean is one observation; paired difference to ABIC over scenarios."""
    m = df.groupby(["scenario", "variant"])[[c for c, _ in cols]].mean()
    out = []
    for v in [base] + [x for x in df.variant.unique() if x != base]:
        row = {"variant": v}
        sc = sorted(set(m.xs(v, level="variant").index) & set(m.xs(base, level="variant").index))
        row["n_scenarios"] = len(sc)
        for c, label in cols:
            x = m.xs(v, level="variant").loc[sc, c].values
            y = m.xs(base, level="variant").loc[sc, c].values
            row[label] = f"{x.mean():.3f}"
            if v != base:
                d = x - y
                row[label + " Δ vs ABIC"] = f"{d.mean():+.3f} (W {int((d > 0).sum())}/L {int((d < 0).sum())}/T {int((d == 0).sum())})"
        out.append(row)
    return pd.DataFrame(out)


def breakdown(df, train_json):
    off = None
    if train_json and os.path.exists(train_json):
        off = json.load(open(train_json))["t_train_total"]
    rows = []
    for v in df.variant.unique():
        g = df[df.variant == v]
        rows.append(dict(variant=v, offline_training_s=("n/a" if v == "abic" or off is None else f"{off:.0f} (one-time, shared)"),
                         online_posterior_s=f"{g.t_posterior.mean():.2f}", optimization_s=f"{g.t_optimize.mean():.1f}",
                         total_per_run_s=f"{g.t_total.mean():.1f}", fev=f"{g.n_fev.mean():.0f}",
                         converged=f"{int(g.converged.sum())}/{len(g)}", timed_out=f"{int(g.timed_out.sum())}/{len(g)}"))
    return pd.DataFrame(rows)


def graph_size(df):
    rows = []
    for v in df.variant.unique():
        g = df[df.variant == v]
        rows.append(dict(variant=v, est_directed=f"{g.n_est_edges.map(lambda e: e[0]).mean():.1f}",
                         est_bidirected=f"{g.n_est_bidirected.mean():.1f}", true_directed=f"{g.n_true_directed.mean():.1f}",
                         true_bidirected=f"{g.n_true_bidirected.mean():.1f}", true_skeleton_edges=f"{g.n_true_edges.mean():.1f}"))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--train-json", default=os.path.join(ROOT, "results", "posterior_training.json"))
    a = ap.parse_args(argv)
    df = load(a.paths)
    parts = ["> PetShop = one dataset per scenario: the sd over seeds is optimizer variance, not data variance. "
             "Corrected metrics unless marked legacy.\n"]
    for (prep, budget), g in df.groupby(["prep", "budget"], dropna=False):
        bt = "no wall-clock budget" if pd.isna(budget) else f"equal budget {budget:.0f}s per run"
        parts.append(f"## Preprocessing `{prep}`, {bt}\n")
        parts.append("### Across scenarios (scenario seed-means)\n")
        parts.append(scenario_level(g, METRICS).to_markdown(index=False) + "\n")
        parts.append(scenario_level(g, LEGACY).to_markdown(index=False) + "\n")
        for scn, s in g.groupby("scenario"):
            parts.append(f"### {scn} (n={int(s.n.iloc[0])}, d={int(s.d.iloc[0])}, {int(s.n_latent.iloc[0])} latent nodes; "
                         f"{s.seed.nunique()} seeds)\n")
            parts.append("Corrected\n\n" + table(s, METRICS).to_markdown(index=False) + "\n")
            parts.append("Legacy\n\n" + table(s, LEGACY).to_markdown(index=False) + "\n")
            parts.append("Estimated vs true graph size\n\n" + graph_size(s).to_markdown(index=False) + "\n")
            parts.append("Runtime (paired means)\n\n" + runtime_table(s).to_markdown(index=False) + "\n")
        parts.append("### Runtime breakdown (all scenarios)\n")
        parts.append(breakdown(g, a.train_json).to_markdown(index=False) + "\n")
    text = "\n".join(parts)
    print(text)
    if a.out:
        open(a.out, "w").write(text)
        df.to_csv(os.path.splitext(a.out)[0] + ".csv", index=False)


if __name__ == "__main__":
    main()
