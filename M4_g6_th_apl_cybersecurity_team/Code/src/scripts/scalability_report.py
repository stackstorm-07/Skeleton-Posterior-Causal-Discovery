"""Table and figures for the scalability sweep:  python scripts/scalability_report.py [--in results/scalability.jsonl] [--out results/scalability]

Per (d, variant): runs, mean skeleton F1 and PAG-SHD (± std), share of runs that hit the budget, share that diverged (no edges and h > 1),
mean total time, mean peak memory, mean allowed pairs.  Figures: skeleton F1 vs d, total time vs d, peak memory vs d.
"""
import argparse
import json
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/scalability.jsonl")
    ap.add_argument("--out", default="results/scalability")
    a = ap.parse_args(argv)
    rows = [r for r in (json.loads(l) for l in open(a.inp) if l.strip()) if "error" not in r]
    ds = sorted({r["d"] for r in rows}); vs = [v for v in ("abic", "abic_s", "spot_cal", "spot_wide") if any(r["variant"] == v for r in rows)]
    cols = ["d", "variant", "runs", "skel_F1", "PAG_SHD", "timed_out", "diverged", "t_total_s", "peak_rss_MB", "allowed_pairs"]
    lines, tab = [], {}
    ms = lambda x: f"{np.mean(x):.3f} ± {np.std(x, ddof=1) if len(x) > 1 else float('nan'):.3f}"
    for d in ds:
        for v in vs:
            s = [r for r in rows if r["d"] == d and r["variant"] == v]
            if not s:
                continue
            f1 = [r["c_skeleton_f1"] for r in s]; shd = [r.get("c_shd_pag", np.nan) for r in s]
            div = [(r["n_est_edges"] == 0 and (r["h"] or 0) > 1) for r in s]
            ap_ = [r.get("n_pairs_allowed", r["n_candidate_pairs"]) for r in s]
            tab[(d, v)] = dict(f1=f1, t=[r["t_total"] for r in s], rss=[r["peak_rss_mb"] for r in s])
            lines.append([d, v, len(s), ms(f1), ms(shd), f"{np.mean([r['timed_out'] for r in s]):.0%}", f"{np.mean(div):.0%}",
                          f"{np.mean(tab[(d, v)]['t']):.0f}", f"{np.mean(tab[(d, v)]['rss']):.0f}", f"{np.mean(ap_):.0f}"])
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(map(str, l)) + " |" for l in lines)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out + ".md", "w").write(md + "\n"); print(md)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for key, lab, fn in (("f1", "skeleton F1 (PAG)", "f1"), ("t", "total time per run (s)", "time"), ("rss", "peak memory (MB)", "mem")):
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for v in vs:
            xs = [d for d in ds if (d, v) in tab]
            ax.errorbar(xs, [np.mean(tab[(d, v)][key]) for d in xs], [np.std(tab[(d, v)][key], ddof=1) if len(tab[(d, v)][key]) > 1 else 0 for d in xs],
                        marker="o", capsize=3, label=v)
        ax.set_xlabel("number of nodes d"); ax.set_ylabel(lab); ax.set_xscale("log"); ax.set_xticks(ds); ax.set_xticklabels(ds)
        ax.legend(); fig.tight_layout(); fig.savefig(f"{a.out}_{fn}.png", dpi=200); plt.close(fig)
    print("figures ->", a.out + "_{f1,time,mem}.png")


if __name__ == "__main__":
    main()
