"""Aggregate the stage-2c JSONL results into paired SPOT-vs-ABIC tables (markdown + CSV).

    python scripts/summarize_2c.py results/syn_2c_main_zscore.jsonl [more.jsonl ...] [--out results/syn_2c_summary.md]

Pairing is by graph (seed): every variant saw the same data, so differences are paired.  With only a handful of graphs
the Wilcoxon p-values are coarse (n = 6 cannot go below 0.031 two-sided); they are reported for orientation, not as proof.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METRICS = [("c_skeleton_f1", "skel F1"), ("c_arrowhead_f1", "head F1"), ("c_tail_f1", "tail F1")]
LEGACY = [("l_skeleton_f1", "skel F1 (legacy)"), ("l_arrowhead_f1", "head F1 (legacy)"), ("l_tail_f1", "tail F1 (legacy)")]


def load(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in open(p)]
    df = pd.DataFrame(rows)
    df["t_total"] = df["elapsed"] + df["t_posterior"]
    return df


def ms(x):
    return f"{np.mean(x):.3f} ± {np.std(x, ddof=1) if len(x) > 1 else 0.0:.3f}"


def table(df, cols, base="abic"):
    out = []
    b = df[df.variant == base].set_index("seed")
    for v in [base] + [v for v in df.variant.unique() if v != base]:
        g = df[df.variant == v].set_index("seed")
        common = sorted(set(g.index) & set(b.index))
        if not common:
            continue
        row = {"variant": v, "n_graphs": len(common)}
        for c, label in cols:
            x, y = g.loc[common, c].values, b.loc[common, c].values
            row[label] = ms(x)
            if v != base:
                d = x - y
                try:
                    p = wilcoxon(x, y).pvalue if np.any(d != 0) else 1.0
                except ValueError:
                    p = float("nan")
                row[label + " Δ vs ABIC"] = f"{d.mean():+.3f} (W {int((d > 0).sum())}/L {int((d < 0).sum())}, p={p:.3f})"
        out.append(row)
    return pd.DataFrame(out)


def runtime_table(df, base="abic"):
    b = df[df.variant == base].set_index("seed")
    out = []
    for v in [base] + [v for v in df.variant.unique() if v != base]:
        g = df[df.variant == v].set_index("seed")
        common = sorted(set(g.index) & set(b.index))
        if not common:
            continue
        row = dict(variant=v, n_graphs=len(common),
                   t_optimize_s=f"{g.loc[common, 't_optimize'].mean():.1f}",
                   t_posterior_s=f"{g.loc[common, 't_posterior'].mean():.2f}",
                   t_total_s=f"{g.loc[common, 't_total'].mean():.1f}",
                   n_fev=f"{g.loc[common, 'n_fev'].mean():.0f}",
                   converged=f"{int(g.loc[common, 'converged'].sum())}/{len(common)}",
                   timed_out=f"{int(g.loc[common, 'timed_out'].sum())}/{len(common)}",
                   est_directed=f"{g.loc[common, 'n_est_edges'].map(lambda e: e[0]).mean():.1f}",
                   est_bidirected=f"{g.loc[common, 'n_est_bidirected'].mean():.1f}",
                   pairs_free=(f"{g.loc[common, 'n_pairs_allowed'].mean():.0f}" if (g.loc[common, 'n_pairs_allowed'] >= 0).all() else "all"))
        if v != base:
            row["time_ratio_vs_abic"] = f"{(g.loc[common, 't_total'] / b.loc[common, 't_total']).mean():.2f}x"
            row["fev_ratio_vs_abic"] = f"{(g.loc[common, 'n_fev'] / b.loc[common, 'n_fev']).mean():.2f}x"
        out.append(row)
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    df = load(a.paths)
    parts = []
    for key, g in df.groupby(["data", "d", "n", "budget"], dropna=False):
        data, d, n, budget = key
        title = f"data={data}, d={d}, n={n}, " + ("no wall-clock budget" if pd.isna(budget) else f"equal budget {budget:.0f}s")
        parts.append(f"### {title}\n")
        parts.append("**Accuracy (corrected metrics; mean ± sd over graphs; paired difference to ABIC)**\n")
        parts.append(table(g, METRICS).to_markdown(index=False) + "\n")
        parts.append("**Accuracy (legacy, bug-compatible metrics)**\n")
        parts.append(table(g, LEGACY).to_markdown(index=False) + "\n")
        parts.append("**Runtime and graph size (means over the paired graphs)**\n")
        parts.append(runtime_table(g).to_markdown(index=False) + "\n")
    text = "\n".join(parts)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
        df.to_csv(os.path.splitext(a.out)[0] + ".csv", index=False)


if __name__ == "__main__":
    main()
