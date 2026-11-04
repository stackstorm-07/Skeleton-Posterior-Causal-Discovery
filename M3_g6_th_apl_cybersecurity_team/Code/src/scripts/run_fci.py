"""E9: classical FCI / RFCI baselines (Fisher-z, Tetrad via the causal-cmd jar), scored with the same PAG metrics as ABIC / SPOT.

    # PetShop: same data and preprocessing as run_petshop.py, one deterministic run per (scenario, algorithm, alpha)
    PETSHOP_ROOT=... python scripts/run_fci.py --source petshop --out results/fci_petshop.jsonl
    # synthetic: the same graphs and data as spot_vs_abic_synthetic.py (same seeds, same options / designs)
    python scripts/run_fci.py --source synthetic --design e1 --out results/fci_syn_e1.jsonl
    python scripts/run_fci.py --source synthetic --d 25 --n 1000 --seeds 0 1 2 --out results/fci_syn.jsonl

FCI/RFCI have no random initialisation, so there is no optimizer seed: PetShop gives ONE row per (scenario, algorithm, alpha)
and the spread over seeds that ABIC/SPOT report has no counterpart.  Do not compare an FCI value with a mean +- SD of
optimizer seeds as if the SD were comparable; on synthetic data the spread over graphs is comparable.

Defaults: alpha in {0.01, 0.05}, algorithms fci and rfci (the plan).  One JSON line per job, resumable; a job that failed
(Java error or timeout) is written with ``failed = true`` and is retried on the next start, never silently dropped.
Run time ``t_wall`` is the whole java subprocess (JVM start-up is about 0.9 s of it); see pgm/fci.py.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from pgm.admg_model import masks_to_admg                                         # noqa: E402
from pgm.fci import ALGORITHMS, TetradError, run_tetrad                           # noqa: E402
from pgm.graphs import admg_to_pag                                                # noqa: E402
from pgm.petshop import PREPROCESSINGS, SCENARIOS, lag1_autocorr, load_petshop, preprocess   # noqa: E402
from spot_vs_abic_synthetic import cond_key, design_conditions, make_graph, score_pag          # noqa: E402


def n_bidirected(pag) -> int:
    return sum(1 for m in pag.edges.values() if m == (">", ">"))


def fci_row(X, names, true_pag, algorithm, alpha, timeout, depth=-1):
    """Run one job; returns the metric/diagnostic columns (``failed`` is set on error)."""
    try:
        r = run_tetrad(X, names, algorithm=algorithm, alpha=alpha, depth=depth, timeout=timeout)
    except TetradError as e:
        return dict(failed=True, error=str(e)[:300])
    row = dict(failed=False, t_wall=r.t_wall, n_est_edges=r.n_edges, n_est_bidirected=n_bidirected(r.pag),
               n_est_circle_marks=sum(m.count("o") for m in r.pag.edges.values()))
    row.update(score_pag(r.pag, true_pag))
    return row


def fci_petshop_rows(ps, X, scn, prep, jobs, timeout, depth=-1, tag=""):
    """Rows for FCI/RFCI on one PetShop scenario (X = preprocessed data); ``jobs`` = [(algorithm, alpha), ...]."""
    rows = []
    for alg, al in jobs:
        row = dict(source="petshop", scenario=scn, prep=prep, algorithm=alg, alpha=al, depth=depth, n=int(X.shape[0]),
                   d=int(X.shape[1]), n_latent=len(ps.latents), n_true_edges=len(ps.pag.edges), tag=tag,
                   lag1_median=float(np.median(lag1_autocorr(X))))
        row.update(fci_row(X, ps.names, ps.pag, alg, al, timeout, depth))
        rows.append(row)
    return rows


def fci_synthetic_rows(c, seed, jobs, timeout, depth=-1, design="custom", tag=""):
    """Rows for FCI/RFCI on one synthetic graph (condition ``c`` as in spot_vs_abic_synthetic.design_conditions)."""
    D, B, X, names, skel = make_graph(seed, c["d"], c["n"], c["data"], density=c["density"], snr=c["snr"], ar=c["ar"], missing=c["missing"])
    true_pag = admg_to_pag(masks_to_admg(names, D, B))
    rows = []
    for alg, al in jobs:
        row = dict(source="synthetic", seed=seed, d=c["d"], n=c["n"], data=c["data"], density=c["density"], snr=c["snr"],
                   ar=c["ar"], missing=c["missing"], algorithm=alg, alpha=al, depth=depth, design=design,
                   variant=f"{alg}@{al}", n_true_edges=int(np.triu(skel, 1).sum()), tag=tag)
        row.update(fci_row(X, names, true_pag, alg, al, timeout, depth))
        rows.append(row)
    return rows


def done_keys(path, keyf):
    keys = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if not r.get("failed"):
                keys.add(keyf(r))
    return keys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["petshop", "synthetic"], required=True)
    ap.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.05])
    ap.add_argument("--depth", type=int, default=-1, help="max conditioning-set size (-1 = unlimited)")
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per job; a timeout is recorded as failed")
    # petshop
    ap.add_argument("--root", default=None, help="PetShop `dataset` dir (else $PETSHOP_ROOT)")
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    ap.add_argument("--prep", default="gauss_rank", choices=PREPROCESSINGS)
    ap.add_argument("--max-rows", type=int, default=None)
    # synthetic (same meaning as in spot_vs_abic_synthetic.py)
    ap.add_argument("--design", choices=["custom", "e1", "e2"], default="custom")
    ap.add_argument("--graphs", type=int, default=None)
    ap.add_argument("--d", type=int, default=25)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--data", choices=["zscore", "raw"], default="zscore")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--density", type=float, default=None)
    ap.add_argument("--snr", type=float, default=None)
    ap.add_argument("--ar", type=float, default=0.0)
    ap.add_argument("--missing", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    jobs = [(alg, al) for alg in a.algorithms for al in a.alphas]

    def write(row):
        with open(a.out, "a") as f:
            f.write(json.dumps(row) + "\n")
        tag = "FAILED " + row.get("error", "")[:60] if row.get("failed") else \
            f"skel {row['c_skeleton_f1']:.3f} head {row['c_arrowhead_f1']:.3f} tail {row['c_tail_f1']:.3f} shd {row['c_shd_pag']} {row['t_wall']:.1f}s"
        print(f"  {row['algorithm']:5s} alpha={row['alpha']:<5} {tag}", flush=True)

    if a.source == "petshop":
        keyf = lambda r: (r["scenario"], r["prep"], r["algorithm"], r["alpha"], r.get("depth", -1))
        done = done_keys(a.out, keyf)
        for scn in a.scenarios:
            todo = [(alg, al) for alg, al in jobs if (scn, a.prep, alg, al, a.depth) not in done]
            if not todo:
                continue
            ps = load_petshop(scn, a.root)
            data = ps.data[:a.max_rows] if a.max_rows else ps.data
            X = preprocess(data, a.prep)
            print(f"[{scn}/{a.prep}] n={X.shape[0]} d={X.shape[1]} truth: {len(ps.pag.edges)} PAG edges", flush=True)
            for row in fci_petshop_rows(ps, X, scn, a.prep, todo, a.timeout, a.depth, a.tag):
                write(row)
        return

    # synthetic
    if a.design == "custom":
        conds = [dict(seeds=a.seeds, d=a.d, n=a.n, data=a.data, density=a.density, snr=a.snr, ar=a.ar, missing=a.missing)]
    else:
        conds = design_conditions(a.design, d=a.d, graphs=a.graphs, data=a.data)
    keyf = lambda r: cond_key(r, variant=f"{r['algorithm']}@{r['alpha']}") + (r.get("depth", -1),)
    done = done_keys(a.out, keyf)
    seen = set()                                            # e1 lists the same (graph, n) for several variants: run each once
    for c in conds:
        for seed in c["seeds"]:
            ck = (seed, c["d"], c["n"], c["data"], c["density"], c["snr"], float(c["ar"]), float(c["missing"]))
            if ck in seen:
                continue
            seen.add(ck)
            todo = [(alg, al) for alg, al in jobs if ck[:1] + (f"{alg}@{al}",) + ck[1:] + (a.depth,) not in done]
            if not todo:
                continue
            print(f"[synthetic seed {seed} n={c['n']} dens={c['density']} snr={c['snr']} ar={c['ar']} mis={c['missing']}]", flush=True)
            for row in fci_synthetic_rows(c, seed, todo, a.timeout, a.depth, a.design, a.tag):
                write(row)


if __name__ == "__main__":
    main()
