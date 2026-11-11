"""M3 / E5: moving-block bootstrap of the PetShop series, ABIC vs SPOT on the SAME resample.

    export PETSHOP_ROOT=<dataset dir>   CAUSAL_CMD_JAR=<causal-cmd jar>
    python scripts/run_bootstrap.py --scenarios low_traffic high_traffic temporal_traffic1 --B 20 --workers 7 \
        --variants abic spot_cal --out results/boot_main.jsonl
    python scripts/run_bootstrap.py --summarize results/boot_main.jsonl         # paired tables, no re-running

Why: PetShop is ONE recording per scenario, so optimizer seeds only measure optimizer noise.  Here the data themselves are
resampled: overlapping blocks of ``--block-len`` consecutive rows (default n^(1/3)) are drawn with replacement, which keeps
the autocorrelation inside a block.  For every resample b the data, preprocessing, posterior, optimizer settings and
optimizer seed (= b) are identical for every variant, so differences between variants are paired.  The ground-truth PAG is
the same for all resamples.  Resample b depends only on (--boot-seed, scenario, b), not on the worker count or the order.

One line of JSON per finished (scenario, b, variant); resumable (finished lines are skipped).
Use ``--workers N`` to run N jobs in parallel (one thread each; set N = physical cores - 1).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")         # one BLAS thread per job: otherwise parallel jobs fight over the cores and timings are meaningless

import argparse
import json
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from pgm.dcd import DCDConfig                                                   # noqa: E402
from pgm.petshop import SCENARIOS, load_petshop, preprocess, skeleton_matrix, zscore   # noqa: E402
from pgm.stats import block_bootstrap_indices, default_block_length, paired_summary    # noqa: E402
from skeleton.posterior import default_model_path, SkeletonPosterior, posterior_metrics              # noqa: E402
from spot_vs_abic_synthetic import ALL_VARIANTS, run_variant, score             # noqa: E402

_CACHE = {}          # per worker process: dataset per scenario and the posterior


def resample_indices(boot_seed: int, scenario: str, b: int, n: int, block_len: int) -> np.ndarray:
    sidx = SCENARIOS.index(scenario)
    rng = np.random.default_rng([boot_seed, sidx, b])
    return block_bootstrap_indices(n, block_len, rng)


def run_job(spec: dict) -> dict:
    scn, b, v = spec["scenario"], spec["b"], spec["variant"]
    if "post" not in _CACHE:
        _CACHE["post"] = SkeletonPosterior.load_or_retrain(spec["model"], verbose=False)
    if scn not in _CACHE:
        _CACHE[scn] = load_petshop(scn, spec["root"])
    ps, post = _CACHE[scn], _CACHE["post"]
    data = ps.data[:spec["max_rows"]] if spec["max_rows"] else ps.data
    n = data.shape[0]
    bl = spec["block_len"] or default_block_length(n)
    idx = resample_indices(spec["boot_seed"], scn, b, n, bl)
    X = preprocess(data[idx], spec["prep"])
    names, skel = ps.names, skeleton_matrix(ps.pag, ps.names)
    post_res = post.predict(zscore(X))
    pm = posterior_metrics(post_res.P, skel)
    cfg = DCDConfig(num_restarts=spec["restarts"], seed=b)
    if spec["lbfgs_maxiter"]:
        cfg = DCDConfig(**{**cfg.__dict__, "lbfgs_maxiter": spec["lbfgs_maxiter"]})
    if spec["max_outer"]:
        cfg = DCDConfig(**{**cfg.__dict__, "max_outer": spec["max_outer"]})
    res, extra, wall = run_variant(v, X, names, post_res, skel, cfg, b, spec["budget"])
    row = dict(scenario=scn, boot_b=b, boot_seed=spec["boot_seed"], block_len=bl, variant=v, prep=spec["prep"], n=int(n),
               d=int(X.shape[1]), budget=spec["budget"], n_est_edges=res.admg.n_edges(), n_est_bidirected=len(res.admg.bi_edges),
               n_true_edges=int(np.triu(skel, 1).sum()), converged=bool(res.converged), timed_out=bool(res.timed_out),
               n_fev=res.n_fev, t_optimize=res.t_optimize, wall=wall, t_posterior=post_res.t_total,
               post_auroc=pm["auroc"], post_log_loss=pm["log_loss"], post_ece=pm["ece"], worker_pid=os.getpid())
    row.update(extra)
    row.update(score(res.admg, ps.pag))
    return row


def done_keys(path):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                keys.add((r["scenario"], r["boot_b"], r["variant"]))
    return keys


METRICS = [("c_skeleton_f1", True), ("c_arrowhead_f1", True), ("c_tail_f1", True), ("c_shd_pag", False), ("t_optimize", False)]


def summarize(path, ref="abic", out_csv=None):
    import pandas as pd
    df = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
    rows = []
    for scn, g in df.groupby("scenario"):
        piv = {m: g.pivot_table(index="boot_b", columns="variant", values=m) for m, _ in METRICS if m in g}
        for v in sorted(set(g.variant) - {ref}):
            for m, hib in METRICS:
                if m not in piv or ref not in piv[m] or v not in piv[m]:
                    continue
                t = piv[m][[v, ref]].dropna()
                s = paired_summary(t[v].values, t[ref].values, higher_is_better=hib)
                rows.append(dict(scenario=scn, variant=v, vs=ref, metric=m, **s))
    out = pd.DataFrame(rows)
    if out_csv:
        out.to_csv(out_csv, index=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="PetShop `dataset` dir (else $PETSHOP_ROOT)")
    ap.add_argument("--scenarios", nargs="+", default=["low_traffic", "high_traffic", "temporal_traffic1"], choices=SCENARIOS)
    ap.add_argument("--variants", nargs="+", default=["abic", "spot_cal"], choices=ALL_VARIANTS)
    ap.add_argument("--B", type=int, default=20, help="number of bootstrap resamples (b = 0..B-1)")
    ap.add_argument("--block-len", type=int, default=None, help="block length (default round(n^(1/3)))")
    ap.add_argument("--boot-seed", type=int, default=2026)
    ap.add_argument("--prep", default="gauss_rank")
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=None, help="tests only")
    ap.add_argument("--model", default=default_model_path(ROOT))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "boot_main.jsonl"))
    ap.add_argument("--lbfgs-maxiter", type=int, default=None, help="tests only")
    ap.add_argument("--max-outer", type=int, default=None, help="tests only")
    ap.add_argument("--summarize", default=None, help="only print the paired summary of an existing result file")
    a = ap.parse_args(argv)

    if a.summarize:
        import pandas as pd
        pd.set_option("display.width", 220, "display.max_columns", 30)
        s = summarize(a.summarize, out_csv=os.path.splitext(a.summarize)[0] + "_paired.csv")
        cols = ["scenario", "variant", "metric", "n", "mean_a", "mean_b", "mean_diff", "ci_pct_lo", "ci_pct_hi", "wins", "losses", "p_wilcoxon", "p_ttest"]
        print(s[[c for c in cols if c in s]].round(4).to_string(index=False))
        return s

    from pgm.petshop import default_root
    root = a.root or default_root()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    SkeletonPosterior.load_or_retrain(a.model)             # once, in the parent: avoids several workers retraining at the same time
    done = done_keys(a.out)
    base = dict(root=root, model=a.model, prep=a.prep, restarts=a.restarts, budget=a.budget, block_len=a.block_len,
                boot_seed=a.boot_seed, max_rows=a.max_rows, lbfgs_maxiter=a.lbfgs_maxiter, max_outer=a.max_outer)
    jobs = [dict(base, scenario=s, b=b, variant=v) for b in range(a.B) for s in a.scenarios for v in a.variants
            if (s, b, v) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done), workers={a.workers}", flush=True)
    t0, n_done = time.time(), 0
    with open(a.out, "a") as f:
        def write(row):
            nonlocal n_done
            f.write(json.dumps(row) + "\n"); f.flush(); n_done += 1
            el = time.time() - t0
            eta = el / n_done * (len(jobs) - n_done)
            print(f"[{n_done}/{len(jobs)}] {row['scenario']} b={row['boot_b']} {row['variant']:9s} skel {row['c_skeleton_f1']:.3f} "
                  f"shd {row['c_shd_pag']:3d} opt {row['t_optimize']:6.1f}s | elapsed {el / 60:5.1f} min, ETA {eta / 60:5.1f} min", flush=True)
        if a.workers <= 1:
            for j in jobs:
                write(run_job(j))
        else:
            # 'spawn', not fork: forking after scikit-learn/OpenMP has run in the parent can deadlock the children
            with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
                for fut in as_completed([ex.submit(run_job, j) for j in jobs]):
                    write(fut.result())
    print("done ->", a.out)


if __name__ == "__main__":
    main()
