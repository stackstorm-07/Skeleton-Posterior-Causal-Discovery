"""E14 (M4): does the posterior still work when d grows?  Narrow (M3, trained on d 25-40) vs wide (trained on d 25-200).

    python scripts/posterior_by_d.py --wide models/skeleton_posterior_wide.pkl --out results/posterior_by_d.jsonl

Fresh test graphs (seed namespace >= 20000, never used for training or floor calibration), d in {25, 40, 60, 100, 150, 200}.
Per graph and model: log-loss (headline), AUROC, AUPRC, ECE, share of TRUE edges pinned by the floor, feature / model time, peak RSS.
Rows are appended to the jsonl, one per (model, graph); re-running skips finished rows.  Then:  --summarize  prints the table
(mean and 95 % CI of the mean per d) and writes results/posterior_by_d.png.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pgm.envinfo import JobMeter                                           # noqa: E402
from skeleton.posterior import SkeletonPosterior, default_model_path, posterior_metrics       # noqa: E402
from skeleton.wide import one_instance                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID = {"d25": (25, 30), "d40": (40, 30), "d60": (60, 30), "d100": (100, 30), "d150": (150, 15), "d200": (200, 15)}   # d -> (d, graphs)


def run(models, floors, out, scale=1.0, grid=None):
    grid = grid or GRID
    done = set()
    if os.path.exists(out):
        for ln in open(out):
            r = json.loads(ln); done.add((r["model"], r["bucket"], r["idx"]))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "a") as f:
        for bucket, (d, cnt) in grid.items():
            for i in range(max(1, int(round(cnt * scale)))):
                todo = [m for m in models if (m, bucket, i) not in done]
                if not todo:
                    continue
                inst = one_instance("test", bucket, i, d, d)
                iu = np.triu_indices(d, 1)
                for name in todo:
                    with JobMeter() as meter:
                        res = models[name].predict(inst.X)
                    row = dict(model=name, bucket=bucket, idx=i, d=d, n=inst.X.shape[0], ar_rho=inst.meta["ar_rho"],
                               n_true=int(inst.skeleton.sum() // 2), t_features=res.t_features, t_model=res.t_model,
                               n_scored=res.n_scored, **posterior_metrics(res.P, inst.skeleton))
                    pe = res.P[iu][inst.skeleton[iu]]
                    row["true_pinned"] = float(np.mean(pe <= (floors[name] or 1e-3))) if len(pe) else float("nan")
                    row["floor"] = floors[name]          # None: calibration found no safe floor; pinned share is then shown at the released 1e-3
                    row.update(peak_rss_mb=meter.fields()["peak_rss_mb"], rss_scope=meter.fields()["rss_scope"])
                    f.write(json.dumps(row) + "\n"); f.flush()
                print(f"{bucket} #{i} done", flush=True)


def summarize(path, png=None):
    rows = [json.loads(l) for l in open(path)]
    keys = ["log_loss", "auroc", "auprc", "ece", "true_pinned", "t_features", "peak_rss_mb"]
    models = sorted({r["model"] for r in rows})
    ds = sorted({r["d"] for r in rows})
    tab = {}
    print(f"{'d':>4} {'model':<8} {'graphs':>6} " + " ".join(f"{k:>16}" for k in keys))
    for d in ds:
        for m in models:
            sel = [r for r in rows if r["d"] == d and r["model"] == m]
            if not sel:
                continue
            cells = {}
            for k in keys:
                v = np.array([r[k] for r in sel], float)
                v = v[np.isfinite(v)]
                half = 1.96 * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
                cells[k] = (float(v.mean()), float(half))
            tab[(d, m)] = cells
            print(f"{d:>4} {m:<8} {len(sel):>6} " + " ".join(f"{cells[k][0]:>9.4f}±{cells[k][1]:<6.4f}" for k in keys))
    if png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for m in models:
            xs = [d for d in ds if (d, m) in tab]
            ax.errorbar(xs, [tab[(d, m)]["log_loss"][0] for d in xs], [tab[(d, m)]["log_loss"][1] for d in xs], marker="o", capsize=3, label=m)
        ax.set_xlabel("number of nodes d"); ax.set_ylabel("posterior log-loss (lower is better)"); ax.set_xscale("log")
        ax.set_xticks(ds); ax.set_xticklabels(ds); ax.legend(); fig.tight_layout(); fig.savefig(png, dpi=200)
        print("figure ->", png)
    return tab


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--narrow", default=default_model_path(ROOT))
    ap.add_argument("--wide", default=os.path.join(ROOT, "models", "skeleton_posterior_wide.pkl"))
    ap.add_argument("--floor-narrow", type=float, default=6e-3)
    ap.add_argument("--floor-wide", type=float, default=None, help="default: 'chosen' of results/floor_calibration_wide.json")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "posterior_by_d.jsonl"))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(argv)
    if a.summarize:
        return summarize(a.out, os.path.splitext(a.out)[0] + ".png")
    fw = a.floor_wide
    if fw is None and os.path.exists(os.path.join(ROOT, "results", "floor_calibration_wide.json")):
        fw = json.load(open(os.path.join(ROOT, "results", "floor_calibration_wide.json")))["chosen"]
    models = {"narrow": SkeletonPosterior.load_or_retrain(a.narrow), "wide": SkeletonPosterior.load(a.wide)}
    run(models, {"narrow": a.floor_narrow, "wide": fw}, a.out, a.scale)
    summarize(a.out, os.path.splitext(a.out)[0] + ".png")


if __name__ == "__main__":
    main()
