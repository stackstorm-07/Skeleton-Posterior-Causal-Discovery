"""Aggregate whatever M3 result files exist into tables (markdown + csv) and figures.  Works on partial results.

    python scripts/aggregate_quick.py --results results --out results/m3_report

Reads results/{ps_m3_e3,ps_m3_e4,ps_m3_e10,ps_m3_e13,boot_main,syn_m3_e1,syn_m3_e2,fci_*,post_quality_m3,rca_m3}.jsonl
Every table states how many units (seeds / resamples / graphs) it is based on.  Missing files are skipped, not errors.
"""
import argparse, json, os, sys, warnings
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from pgm.stats import paired_summary
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

METRICS = ["c_skeleton_f1", "c_arrowhead_f1", "c_tail_f1"]
SHORT = {"c_skeleton_f1": "skel F1", "c_arrowhead_f1": "arrow F1", "c_tail_f1": "tail F1", "c_shd_pag": "SHD(PAG)", "t_optimize": "opt time s"}
MD = []; CSV = {}

def load(path):
    if not os.path.exists(path): return None
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try: rows.append(json.loads(line))
            except Exception: pass
    return pd.DataFrame(rows) if rows else None

def emit(title, df, name, note=""):
    MD.append(f"\n### {title}\n")
    if note: MD.append(note + "\n")
    MD.append(df.to_markdown(index=False, floatfmt=".3f") if len(df) else "(no data)"); CSV[name] = df

def dedup(df, keys):
    keys = [k for k in keys if k in df.columns]
    return df.drop_duplicates(subset=keys, keep="last")

def cols(df, want): return [c for c in want if c in df.columns]

def msd(df, by, metrics):
    g = df.groupby(by)
    out = g.size().rename("n_runs").reset_index()
    for m in metrics:
        s = g[m].agg(["mean", "std"]).reset_index()
        out[SHORT.get(m, m)] = [f"{a:.3f} ± {0 if pd.isna(b) else b:.3f}" for a, b in zip(s["mean"], s["std"])]
        # 95 % t interval of the mean over the runs in the group (n = 1 -> no interval)
        from scipy import stats as _st
        n = g[m].count().values
        h = np.where(n > 1, _st.t.ppf(0.975, np.maximum(n - 1, 1)) * s["std"].fillna(0).values / np.sqrt(np.maximum(n, 1)), np.nan)
        out[SHORT.get(m, m) + " CI95"] = [("[%.3f, %.3f]" % (a - hh, a + hh)) if hh == hh else "n/a" for a, hh in zip(s["mean"], h)]
    return out

def paired(df, unit, a, b, metrics, by=None):
    """paired differences b - a (a = reference) over `unit`, optionally within `by` groups"""
    rows = []
    groups = [((), df)] if not by else list(df.groupby(by))
    for key, sub in groups:
        key = key if isinstance(key, tuple) else (key,)
        A = sub[sub.variant == a].set_index(unit); B = sub[sub.variant == b].set_index(unit)
        A = A[~A.index.duplicated()]; B = B[~B.index.duplicated()]
        common = A.index.intersection(B.index)
        for m in metrics:
            if m not in sub.columns or len(common) == 0: continue
            hib = m not in ("c_shd_pag", "t_optimize")
            s = paired_summary(B.loc[common, m].values, A.loc[common, m].values)
            d = B.loc[common, m].values - A.loc[common, m].values
            r = dict(zip(by or [], key)); r.update(metric=SHORT.get(m, m), compare=f"{b} - {a}", n_pairs=len(common),
                mean_diff=float(np.mean(d)), sd_diff=float(np.std(d, ddof=1)) if len(d) > 1 else float("nan"),
                wins=int(((d > 0) if hib else (d < 0)).sum()), losses=int(((d < 0) if hib else (d > 0)).sum()), ties=int((d == 0).sum()),
                p_wilcoxon=s.get("p_wilcoxon", float("nan")), p_ttest=s.get("p_ttest", float("nan")))
            # 95 % CI of the mean paired difference: t interval and percentile interval (paired_summary's key names)
            for k_out, k in (("ci95_lo", "ci_mean_lo"), ("ci95_hi", "ci_mean_hi"), ("ci95_pct_lo", "ci_pct_lo"), ("ci95_pct_hi", "ci_pct_hi")):
                r[k_out] = s.get(k, float("nan"))
            lo, hi = r["ci95_lo"], r["ci95_hi"]
            r["ci_excludes_0"] = bool(lo > 0 or hi < 0) if lo == lo and hi == hi else False
            r["decision"] = ("supported" if (r["p_ttest"] < 0.05 and r["p_wilcoxon"] < 0.05 and r["ci_excludes_0"]) else "not distinguishable from noise")
            rows.append(r)
    return pd.DataFrame(rows)

def bar(df, x, hue, y, fn, title, ylabel):
    p = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
    e = df.pivot_table(index=x, columns=hue, values=y, aggfunc="std").fillna(0)
    ax = p.plot.bar(yerr=e, capsize=3, figsize=(7, 4)); ax.set_ylabel(ylabel); ax.set_title(title); plt.xticks(rotation=20)
    plt.tight_layout(); plt.savefig(fn, dpi=160); plt.close()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--out", default="results/m3_report")
    a = ap.parse_args(); R = lambda f: os.path.join(a.results, f); os.makedirs(a.out, exist_ok=True)
    MD.append("# M3 aggregated results\n\nAll numbers are generated from raw result lines by scripts/aggregate_m3.py. ± is the SD over the stated units. "
              "PetShop has ONE dataset per scenario: seeds change the optimiser start only, so these p-values describe optimiser variability, not new datasets.\n")

    # ---------------- E3 main PetShop
    e3 = load(R("ps_m3_e3.jsonl"))
    if e3 is not None:
        e3 = dedup(e3, ["scenario", "seed", "variant", "prep"]); m = cols(e3, METRICS + ["c_shd_pag", "t_optimize"])
        emit("E3. PetShop: ABIC vs SPOT (per scenario)", msd(e3, ["scenario", "variant"], m), "E3_per_scenario", "gauss_rank, one restart, no budget; seeds = optimiser seeds.")
        scn = e3.groupby(["scenario", "variant"])[m].mean().reset_index()
        pooled = scn.groupby("variant")[m].mean().reset_index(); pooled.insert(1, "n_scenarios", scn.groupby("variant").size().values)
        emit("E3. Mean of scenario means", pooled, "E3_pooled")
        pr = pd.concat([paired(e3, "seed", "abic", v, m, by=["scenario"]) for v in ["spot_cal", "spot"] if v in set(e3.variant)] or [pd.DataFrame()])
        emit("E3. Paired difference vs ABIC over the same optimiser seeds", pr, "E3_paired", "difference = SPOT variant - ABIC. For SHD and time, negative is better.")
        # scenario-level win/loss
        if len(pr):
            wl = pr[pr.metric == "skel F1"].groupby("compare").apply(lambda g: f"W{(g.mean_diff>0).sum()}/L{(g.mean_diff<0).sum()}/T{(g.mean_diff==0).sum()} of {len(g)} scenarios").reset_index(name="scenario wins/losses/ties (skel F1)")
            emit("E3. Scenario-level record", wl, "E3_wl")
        bar(e3, "scenario", "variant", "c_skeleton_f1", os.path.join(a.out, "F_e3_skeleton.png"), "PetShop skeleton F1 (mean, SD over seeds)", "skeleton F1")
        if "t_optimize" in e3: bar(e3, "scenario", "variant", "t_optimize", os.path.join(a.out, "F_e3_time.png"), "Optimisation time (contended machine: use ratios)", "seconds")
        if "n_est_bidirected" in e3:
            e3["n_est_dir"] = e3.n_est_edges.apply(lambda x: x[0] if isinstance(x, list) else np.nan)
            emit("E3. Estimated edge counts", e3.groupby(["scenario", "variant"])[["n_est_dir", "n_est_bidirected"]].mean().reset_index(), "E3_edgecounts", "Truth has 0-1 bidirected edges and 30-38 directed.")

    # ---------------- E4 ablations (+E3 reference)
    e4 = load(R("ps_m3_e4.jsonl"))
    if e4 is not None:
        e4 = dedup(e4, ["scenario", "seed", "variant", "prep"]); m = cols(e4, METRICS + ["c_shd_pag", "t_optimize"])
        ref = e3[e3.variant.isin(["abic", "spot", "spot_cal"])] if e3 is not None else None
        allv = pd.concat([e4, ref]) if ref is not None else e4
        allv = allv[allv.seed.isin(sorted(e4.seed.unique()))]
        emit("E4. Ablations on PetShop (same seeds as the ablation runs)", msd(allv, ["variant"], m), "E4_ablation",
             "spot_nofloor = mask only; spot_floor_only = hard floor only; spot_random = mask from a random posterior; spot_oracle = mask from the TRUE skeleton (upper bound, not a method).")
        emit("E4. Ablations per scenario", msd(allv, ["scenario", "variant"], m), "E4_ablation_scn")
        bar(allv, "variant", "scenario", "c_skeleton_f1", os.path.join(a.out, "F_e4_ablation.png"), "Ablations: skeleton F1", "skeleton F1")

    # ---------------- E10 directed only
    e10 = load(R("ps_m3_e10.jsonl"))
    if e10 is not None:
        e10 = dedup(e10, ["scenario", "seed", "variant"]); m = cols(e10, METRICS + ["c_shd_pag", "t_optimize"])
        emit("E10. Directed-only models (no bidirected edges)", msd(e10, ["scenario", "variant"], m), "E10")
        v = sorted(set(e10.variant))
        if len(v) == 2: emit("E10. Paired", paired(e10, "seed", v[0], v[1], m, by=["scenario"]), "E10_paired")

    # ---------------- E13 raw_centered sensitivity
    e13 = load(R("ps_m3_e13.jsonl"))
    if e13 is not None:
        e13 = dedup(e13, ["scenario", "seed", "variant"]); m = cols(e13, METRICS + ["t_optimize"])
        emit("E13. raw_centered preprocessing (sensitivity)", msd(e13, ["scenario", "variant"], m), "E13",
             "Compare with E3 (gauss_rank). Large drops mean the result depends on preprocessing.")

    # ---------------- E5 block bootstrap
    bm = load(R("boot_main.jsonl"))
    if bm is not None:
        bm = dedup(bm, ["scenario", "boot_b", "variant"]); m = cols(bm, METRICS + ["c_shd_pag"])
        bm["unit"] = list(zip(bm.scenario, bm.boot_b)); b = bm.copy()
        emit("E5. Moving-block bootstrap: per scenario", msd(bm, ["scenario", "variant"], m), "E5_per_scenario", "Resamples of the DATA (block length ~ n^(1/3)); this is dataset-level variability.")
        rows = []
        for v in [x for x in set(bm.variant) if x != "abic"]:
            rows.append(paired(bm.assign(unit=bm.boot_b.astype(str)), "unit", "abic", v, m, by=["scenario"]))
        emit("E5. Paired difference vs ABIC on the same resamples", pd.concat(rows) if rows else pd.DataFrame(), "E5_paired")
        if "spot_cal" in set(bm.variant):
            fig, axs = plt.subplots(1, bm.scenario.nunique(), figsize=(4 * bm.scenario.nunique(), 3.4), squeeze=False)
            for ax, (s, sub) in zip(axs[0], bm.groupby("scenario")):
                A = sub[sub.variant == "abic"].set_index("boot_b").c_skeleton_f1; B = sub[sub.variant == "spot_cal"].set_index("boot_b").c_skeleton_f1
                d = (B - A).dropna(); ax.hist(d, bins=10); ax.axvline(0, color="k"); ax.axvline(d.mean(), color="r"); ax.set_title(f"{s}\nSPOT-cal - ABIC skel F1 (n={len(d)})", fontsize=9)
            plt.tight_layout(); plt.savefig(os.path.join(a.out, "F_e5_bootstrap_diff.png"), dpi=160); plt.close()

    # ---------------- E1 / E2 synthetic
    for nm, fn in [("E1", "syn_m3_e1.jsonl"), ("E2", "syn_m3_e2.jsonl")]:
        s = load(R(fn))
        if s is None: continue
        keys = [k for k in ["seed", "n", "variant", "density", "snr", "ar", "missing", "design", "data"] if k in s.columns]
        s = dedup(s, keys); m = cols(s, METRICS + ["c_shd_pag", "t_optimize"])
        by = [k for k in (["n"] if nm == "E1" else ["density", "snr", "ar", "missing", "n"]) if k in s.columns]
        s2 = s.copy()
        for k in by: s2[k] = s2[k].fillna(-1) if k != "n" else s2[k]
        emit(f"{nm}. Synthetic: mean ± SD over graphs", msd(s2, by + ["variant"], m), f"{nm}_summary", "Raw-z data as configured in the experiment; graphs differ by seed.")
        vs = [v for v in ["spot_cal", "spot"] if v in set(s2.variant)]
        if "abic" in set(s2.variant) and vs:
            emit(f"{nm}. Paired vs ABIC over graphs", pd.concat([paired(s2, "seed", "abic", v, m, by=by) for v in vs]), f"{nm}_paired")
        if nm == "E1" and "n" in s.columns:
            fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
            for ax, mm in zip(axs, ["c_skeleton_f1", "c_shd_pag" if "c_shd_pag" in s.columns else "c_arrowhead_f1"]):
                for v, g in s.groupby("variant"):
                    t = g.groupby("n")[mm].agg(["mean", "std"]).reset_index(); ax.errorbar(t.n, t["mean"], yerr=t["std"], marker="o", capsize=3, label=v)
                ax.set_xscale("log"); ax.set_xlabel("sample size n"); ax.set_ylabel(SHORT.get(mm, mm)); ax.legend(fontsize=7)
            plt.tight_layout(); plt.savefig(os.path.join(a.out, "F_e1_sample_size.png"), dpi=160); plt.close()

    # ---------------- E9 FCI
    fp = load(R("fci_petshop.jsonl"))
    if fp is not None:
        fp = fp[~fp.get("failed", False).astype(bool)] if "failed" in fp else fp
        keep = cols(fp, ["scenario", "algorithm", "alpha"] + METRICS + ["t_wall"])
        emit("E9. FCI/RFCI on PetShop (deterministic: one run per setting)", fp[keep].groupby([c for c in ["scenario", "algorithm", "alpha"] if c in keep]).mean(numeric_only=True).reset_index(), "E9_petshop",
             "Single data set per scenario; compare with the E3 table. Do not over-read small gaps.")
    fs = load(R("fci_syn_e1.jsonl"))
    if fs is not None:
        fs = fs[~fs.get("failed", False).astype(bool)] if "failed" in fs else fs
        emit("E9. FCI on synthetic (E1 design)", msd(fs, [c for c in ["n", "variant"] if c in fs.columns], cols(fs, METRICS)), "E9_syn")

    # ---------------- E7 posterior quality
    pq = load(R("post_quality_m3.jsonl"))
    if pq is not None:
        keep = cols(pq, ["dataset", "scenario", "calibrator", "n_graphs", "n_pairs", "n_true_edges", "auroc", "auprc", "log_loss", "ece", "brier", "overconfidence"])
        emit("E7. Skeleton posterior quality", pq[keep], "E7", "log-loss is the headline probability-quality metric; AUROC/AUPRC are ranking metrics.")
        try:
            fig, ax = plt.subplots(figsize=(4.6, 4.4)); ax.plot([0, 1], [0, 1], "k--", lw=1)
            for _, r in pq.iterrows():
                rel = r.get("reliability")
                if isinstance(rel, dict):
                    if "mean_pred" in rel and "frac_true" in rel:
                        ax.plot(rel["mean_pred"], rel["frac_true"], marker="o", ms=3, label=f"{r.get('dataset')} {r.get('scenario', '')} {r.get('calibrator', '')}")
            ax.set_xlabel("predicted edge probability"); ax.set_ylabel("observed edge frequency"); ax.legend(fontsize=6); plt.tight_layout()
            plt.savefig(os.path.join(a.out, "F_e7_reliability.png"), dpi=160); plt.close()
        except Exception as ex: MD.append(f"\n(reliability figure skipped: {ex})\n")

    # ---------------- E12 RCA
    rc = load(R("rca_m3.jsonl"))
    if rc is not None:
        g = [c for c in ["source", "method", "ancestor_mode"] if c in rc.columns]
        t = rc.groupby(g + (["split"] if "split" in rc else [])).agg(n_cases=("top1", "size"), top1=("top1", "mean"), top3=("top3", "mean"), mrr=("rr", "mean")).reset_index()
        emit("E12. Root-cause ranking (latency cases; all graph seeds pooled)", t, "E12",
             "n_cases counts case x graph-seed rows. Cases are few: a difference of one or two cases is noise. 'train'/'test' split is reported separately.")
        t2 = rc.groupby(g).agg(n_rows=("top1", "size"), top1=("top1", "mean"), top3=("top3", "mean"), mrr=("rr", "mean")).reset_index()
        emit("E12. Same, splits pooled", t2, "E12_pooled")

    open(os.path.join(a.out, "tables.md"), "w").write("\n".join(MD))
    for k, d in CSV.items(): d.to_csv(os.path.join(a.out, k + ".csv"), index=False)
    print("wrote", a.out, "tables:", len(CSV))

if __name__ == "__main__": main()
