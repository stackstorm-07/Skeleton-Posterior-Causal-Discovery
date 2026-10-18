"""Aggregate stage-3 RCA rows into paste-ready tables.

    python scripts/summarize_rca.py results/rca_stage3.jsonl --out results/rca_summary

Metrics follow the PetShop benchmark: top-1 and top-3 recall of the (unique) ground-truth root-cause node over the injected
latency issues; MRR (mean of 1/rank, 0 if the node is not ranked) is added because top-k on a few dozen cases is coarse.
A case whose root cause is not an observed variable counts as a miss for every method (it is listed in the ``unobs.`` column).
The number of cases per scenario is small (the benchmark has 14 / 14 / 8 latency issues for low / high / temporal traffic), so
differences of one or two cases are noise; no significance test is attempted.
"""
import argparse
import os

import pandas as pd

SRC = {"none": "no graph", "abic": "ABIC graph", "spot": "SPOT-1e-3 graph", "spot_cal": "SPOT-cal graph", "truth": "true MAG"}
MTH = {"anomaly": "anomaly size", "dag": "directed edges only", "admg": "ADMG (directed + bidirected)"}


def label(r):
    s = f"{SRC[r.source]}, {MTH[r.method]}"
    return s + (", ancestors only" if r.ancestors_only else "")


def agg(g):
    n = len(g)
    return pd.Series(dict(cases=n, top1=g.top1.mean(), top3=g.top3.mean(), mrr=g.rr.mean(),
                          unobs=int((~g.root_observed).sum())))


def fmt(df):
    out = df.copy()
    for c in ("top1", "top3", "mrr"):
        out[c] = out[c].map(lambda x: f"{x:.2f}")
    return out.rename(columns={"cases": "cases", "top1": "top-1", "top3": "top-3", "mrr": "MRR", "unobs": "unobs."})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="all", choices=["all", "train", "test"])
    a = ap.parse_args(argv)
    df = pd.concat([pd.read_json(p, lines=True) for p in a.paths], ignore_index=True)
    if a.split != "all":
        df = df[df.split == a.split]
    df["label"] = df.apply(label, axis=1)
    order = list(dict.fromkeys(df.label))
    parts = [f"# Stage 3: root-cause ranking on PetShop latency issues (split: {a.split})\n",
             "> Few cases per scenario: differences of one or two cases are noise. Root-cause labels were never used to learn graphs or choose settings.\n"]
    pooled = df.groupby("label").apply(agg, include_groups=False).loc[order].reset_index()
    parts.append("## Pooled over scenarios\n")
    parts.append(fmt(pooled).to_markdown(index=False, disable_numparse=True) + "\n")
    scn_tabs = []
    for scn, g in df.groupby("scenario"):
        t = g.groupby("label").apply(agg, include_groups=False).loc[order]
        t = t.rename(columns=lambda c: f"{scn}: {c}")[[f"{scn}: top1", f"{scn}: top3"]]
        scn_tabs.append(t)
    per = pd.concat(scn_tabs, axis=1).reset_index()
    for c in per.columns[1:]:
        per[c] = per[c].map(lambda x: f"{x:.2f}")
    per.columns = [c.replace("top1", "top-1").replace("top3", "top-3") for c in per.columns]
    parts.append("## Per scenario (top-1 / top-3 recall)\n")
    parts.append(per.to_markdown(index=False, disable_numparse=True) + "\n")
    cases = df[df.label == order[0]].groupby(["scenario", "split"]).size().unstack(fill_value=0)
    parts.append("## Cases scored\n")
    parts.append(cases.to_markdown() + "\n")
    notes = df[df.label == order[0]]
    parts.append(f"Abnormal window: { {k: int(v) for k, v in notes.window.value_counts().items()} }; node matching of root causes: { {k: int(v) for k, v in notes.root_match.value_counts().items()} }; "
                 f"transform: {df["transform"].iloc[0]}.\n")
    text = "\n".join(parts)
    print(text)
    if a.out:
        open(a.out + ".md", "w").write(text)
        pooled.to_csv(a.out + "_pooled.csv", index=False)
        per.to_csv(a.out + "_per_scenario.csv", index=False)


if __name__ == "__main__":
    main()
