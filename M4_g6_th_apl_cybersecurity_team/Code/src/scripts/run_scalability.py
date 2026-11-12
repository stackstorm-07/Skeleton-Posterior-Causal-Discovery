"""M4 scalability check: ABIC vs SPOT as the number of variables d grows, under a FIXED wall-clock budget per run.

    python scripts/run_scalability.py --list                                   # plan, runs nothing
    nohup python scripts/run_scalability.py --workers 3 > results/scal.log 2>&1 &   # overnight

Synthetic linear-Gaussian ancestral ADMGs (same generator as E1), n = 1000, d in {25, 50, 100, 200} by default.  For every graph the SAME data
and optimizer settings go to every variant:
    abic        no posterior, default dense start W1 ~ U(-0.5, 0.5)  (at d >= 100 its acyclicity term overflows: see the report note)
    abic_s      ABIC with the start scaled to the d = 25 spectral radius, init_w1 = 0.5 * sqrt(25 / d)   (the fair large-d baseline)
    spot_cal    narrow posterior (trained on d 25-40), floor 6e-3         (the M3 SPOT)
    spot_wide   wide posterior (trained on d 25-200), floor from results/floor_calibration_wide.json (null -> mask only)
Each run gets ``--budget`` seconds of wall clock; SPOT's posterior inference time is deducted from it (equal total time).  A run that
hits the budget is flagged ``timed_out`` and is scored on its current iterate, so at large d the question is "what accuracy does each
method reach in the same time", not "who converges".  Every row stores d, n, seed, budget, accuracy (F1, PAG-SHD), posterior time,
allowed pairs (= number of free parameter pairs), peak memory of the run, and the environment stamps.  Absolute times are only meaningful
when the machine was otherwise idle (``load1_end`` is stored); ratios between variants of the same job are robust.
Results: results/scalability.jsonl (resumable).  Table + figures: python scripts/scalability_report.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import argparse
import json
import multiprocessing as mp
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "scalability.jsonl")
VARIANTS = ("abic", "abic_s", "spot_cal", "spot_wide")


def _job(job):
    from pgm.dcd import DCDConfig
    from pgm.envinfo import JobMeter
    from pgm.graphs import admg_to_pag
    from pgm.admg_model import masks_to_admg
    from skeleton.posterior import SkeletonPosterior, posterior_metrics
    from scripts.spot_vs_abic_synthetic import make_graph, run_variant, score

    d, seed, n, budget = job["d"], job["seed"], job["n"], job["budget"]
    D, B, X, names, skel = make_graph(seed, d, n, "zscore")
    rows = []
    try:
        true_pag = admg_to_pag(masks_to_admg(names, D, B))
    except Exception as e:                                        # scoring problem: report it, do not hide it
        return [dict(job_id=job["id"], error=f"true PAG failed: {e!r}", d=d, seed=seed)]
    cfg = DCDConfig(num_restarts=1, seed=0)
    post = {}
    for v in job["variants"]:
        if v in ("abic", "abic_s"):
            res_p, floor, pm, t_post, pfields = None, None, {}, 0.0, None
        else:
            key = "narrow" if v == "spot_cal" else "wide"
            if key not in post:
                m = SkeletonPosterior.load(job["models"][key])
                with JobMeter() as pmeter:
                    pr = m.predict(X)
                post[key] = (pr, pmeter.fields())
            res_p, pfields = post[key]
            floor = job["floors"][key]
            t_post = res_p.t_total
            pm = posterior_metrics(res_p.P, skel)
        b = None if budget is None else max(budget - t_post, 1.0)
        cfg_v = DCDConfig(num_restarts=1, seed=0, init_w1=0.5 * min(1.0, float(np.sqrt(25.0 / d)))) if v == "abic_s" else cfg
        with JobMeter() as meter:
            res, extra, wall = run_variant({"spot_wide": "spot_cal", "abic_s": "abic"}.get(v, v), X, names, res_p, skel, cfg_v, seed, b, floor=floor)
        t0 = time.perf_counter()
        sc = score(res.admg, true_pag)
        row = dict(job_id=f"{job['id']}|{v}", variant=v, d=d, n=n, seed=seed, budget=budget, floor=floor,
                   n_true_edges=int(np.triu(skel, 1).sum()), n_est_edges=res.admg.n_edges(), n_est_bidirected=len(res.admg.bi_edges),
                   n_candidate_pairs=int(d * (d - 1) // 2), converged=bool(res.converged), timed_out=bool(res.timed_out), h=res.h,
                   n_fev=res.n_fev, n_outer=res.n_outer, t_optimize=res.t_optimize, elapsed=res.elapsed, wall=wall,
                   t_posterior=t_post, t_total=wall + t_post, t_scoring=time.perf_counter() - t0,
                   post_log_loss=pm.get("log_loss"), post_auroc=pm.get("auroc"), post_auprc=pm.get("auprc"))
        row.update(extra); row.update(sc); row.update(meter.fields())
        if pfields is not None:
            row["posterior_peak_rss_mb"] = pfields["peak_rss_mb"]
        rows.append(row)
    return rows


def _run(job):
    try:
        return _job(job)
    except Exception as e:
        import traceback
        return [dict(job_id=job["id"], error=repr(e), trace=traceback.format_exc()[-1500:], d=job["d"], seed=job["seed"])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="+", default=[25, 50, 100, 200])
    ap.add_argument("--graphs", type=int, nargs="+", default=[4, 3, 3, 2], help="graphs per d (same order as --ds)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--budget", type=float, default=600.0, help="seconds per run (SPOT: minus its posterior time)")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--narrow", default=None)
    ap.add_argument("--wide", default=os.path.join(ROOT, "models", "skeleton_posterior_wide.pkl"))
    ap.add_argument("--floor-wide", default="file", help="'file' (results/floor_calibration_wide.json), 'none' or a number")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--max-jobs", type=int, default=None)
    a = ap.parse_args(argv)
    from skeleton.posterior import default_model_path
    narrow = a.narrow or default_model_path(ROOT)
    if a.floor_wide == "file":
        fj = os.path.join(ROOT, "results", "floor_calibration_wide.json")
        fw = json.load(open(fj))["chosen"] if os.path.exists(fj) else None
    else:
        fw = None if a.floor_wide == "none" else float(a.floor_wide)
    from pgm.spot import CALIBRATED_FLOOR as CF
    variants = [v for v in a.variants if v != "spot_wide" or os.path.exists(a.wide)]
    if "spot_wide" in a.variants and "spot_wide" not in variants:
        print(f"[warn] {a.wide} missing: spot_wide skipped", flush=True)
    done = set()
    if os.path.exists(a.out):
        for ln in open(a.out):
            r = json.loads(ln)
            if "error" not in r:
                done.add(r["job_id"])
    jobs = []
    for d, g in zip(a.ds, a.graphs):
        for seed in range(g):
            jid = f"scal|d{d}|n{a.n}|s{seed}|b{int(a.budget)}"
            todo = [v for v in variants if f"{jid}|{v}" not in done]
            if todo:
                jobs.append(dict(id=jid, d=d, seed=seed, n=a.n, budget=a.budget, variants=todo,
                                 models=dict(narrow=narrow, wide=a.wide), floors=dict(narrow=CF, wide=fw)))
    jobs.sort(key=lambda j: -j["d"])                             # longest first
    est = sum(len(j["variants"]) * a.budget for j in jobs) / 3600
    print(f"{len(jobs)} jobs ({sum(len(j['variants']) for j in jobs)} runs), up to {est:.1f} core-hours, {a.workers} workers, "
          f"about {est / a.workers:.1f} h wall; variants {variants}; wide floor {fw}; narrow model {os.path.basename(narrow)}", flush=True)
    if a.list:
        return
    if a.max_jobs:
        jobs = jobs[:a.max_jobs]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(_run, jobs):
            with open(a.out, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            for r in rows:
                if "error" in r:
                    print("FAILED", r.get("job_id"), r["error"], flush=True)
                else:
                    print(f"d={r['d']:>3} s{r['seed']} {r['variant']:<9} skelF1 {r['c_skeleton_f1']:.3f} shd {r.get('c_shd_pag')} "
                          f"t {r['t_total']:.0f}s timed_out {r['timed_out']} rss {r['peak_rss_mb']:.0f}MB  [{(time.time() - t0) / 60:.0f} min]", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
