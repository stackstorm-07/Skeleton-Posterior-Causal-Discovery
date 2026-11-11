"""Stage 3: root-cause analysis on PetShop with the learned ADMGs.

    python scripts/run_rca.py --scenarios low_traffic high_traffic temporal_traffic1 temporal_traffic2 \
        --sources abic spot_cal truth --out results/rca_stage3.jsonl

For every scenario:
  1. graphs: the ADMG learned by ABIC and by SPOT (same data, preprocessing, optimizer settings and seed as stage 2d; learned once and
     cached under ``--graphs-dir``), plus the ground-truth MAG ('truth', the reference upper bound);
  2. a linear-Gaussian model of the NORMAL period for each graph (RICF), in a transformed space fitted on the normal data only;
  3. for every injected *latency* issue (train and test splits are both scored and tagged): the abnormal-window mean shift is ranked
     with  anomaly (no graph) | dag (directed edges only) | admg (directed + bidirected)  x  with/without the ancestor filter.
The root-cause labels are never used to learn graphs or to choose settings.  Output: one JSONL row per (case, source, method).

Only the latency graph is learned (stage 2d used latency), so availability issues are out of scope here.
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
from pgm.dcd import DCDConfig                                                             # noqa: E402
from pgm.petshop import SCENARIOS, load_petshop, preprocess, skeleton_matrix, zscore, default_root   # noqa: E402
from pgm.petshop_issues import list_issue_cases, load_issue_metrics, match_node          # noqa: E402
from pgm.rca import (ANCESTOR_MODES, Transformer, admg_from_json, admg_to_json, fit_rca_model, masks_of, rank_of,   # noqa: E402
                     rank_root_causes)
from skeleton.posterior import default_model_path, SkeletonPosterior                                          # noqa: E402

LEARNED = ("abic", "spot", "spot_cal")


def get_graph(source, scn, ps, X, names, post, post_res, skel, a):
    """(D, B, meta) of a graph source; learned graphs are cached as JSON."""
    if source == "truth":
        D, B = masks_of(ps.mag, names)
        return D, B, dict(kind="truth")
    path = os.path.join(a.graphs_dir, f"{scn}_{source}_{a.prep}_seed{a.graph_seed}.json")
    if os.path.exists(path) and not a.relearn:
        blob = json.load(open(path))
        G = admg_from_json(blob["admg"])
        D, B = masks_of(G, names)
        return D, B, dict(kind="learned-cached", **blob["meta"])
    from spot_vs_abic_synthetic import run_variant
    cfg = DCDConfig(num_restarts=a.restarts, seed=a.graph_seed)
    if a.lbfgs_maxiter:
        cfg = DCDConfig(**{**cfg.__dict__, "lbfgs_maxiter": a.lbfgs_maxiter})
    if a.max_outer:
        cfg = DCDConfig(**{**cfg.__dict__, "max_outer": a.max_outer})
    t0 = time.perf_counter()
    res, extra, _ = run_variant(source, X, names, post_res, skel, cfg, a.graph_seed, None)
    meta = dict(source=source, seed=a.graph_seed, prep=a.prep, restarts=a.restarts, t_optimize=res.t_optimize,
                converged=bool(res.converged), n_di=len(res.admg.di_edges), n_bi=len(res.admg.bi_edges),
                t_learn=time.perf_counter() - t0)
    os.makedirs(a.graphs_dir, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"                       # atomic: parallel queue jobs may learn the same graph at the same time
    json.dump(dict(admg=admg_to_json(res.admg), meta=meta), open(tmp, "w"))
    os.replace(tmp, path)
    D, B = masks_of(res.admg, names)
    return D, B, dict(kind="learned", **meta)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="PetShop `dataset` dir (else $PETSHOP_ROOT)")
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS, choices=SCENARIOS)
    ap.add_argument("--sources", nargs="+", default=["abic", "spot_cal", "truth"], choices=list(LEARNED) + ["truth"])
    ap.add_argument("--transform", default="log", choices=["log", "zscore", "gauss_rank"],
                    help="space in which shifts are measured (fitted on the normal data only)")
    ap.add_argument("--metric", default="latency", help="only issues whose target metric is this are scored")
    ap.add_argument("--statistic", default="Average")
    ap.add_argument("--window", default="after", choices=["after", "all"])
    ap.add_argument("--prep", default="gauss_rank", help="preprocessing used to LEARN the graphs (as in stage 2d)")
    ap.add_argument("--graph-seeds", type=int, nargs="+", default=[0], help="optimizer seeds of the learned graphs (truth is seed-free)")
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--graphs-dir", default=os.path.join(ROOT, "results", "rca_graphs"))
    ap.add_argument("--relearn", action="store_true")
    ap.add_argument("--model", default=default_model_path(ROOT))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "rca_stage3.jsonl"))
    ap.add_argument("--max-rows", type=int, default=None, help="use only the first N normal rows (tests)")
    ap.add_argument("--lbfgs-maxiter", type=int, default=None, help="tests only")
    ap.add_argument("--max-outer", type=int, default=None, help="tests only")
    a = ap.parse_args(argv)

    root = a.root or default_root()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    post = SkeletonPosterior.load_or_retrain(a.model) if any(s in LEARNED and s != "abic" for s in a.sources) else None
    rows = []
    for scn in a.scenarios:
        ps = load_petshop(scn, root)
        names = ps.names
        raw = ps.data[:a.max_rows] if a.max_rows else ps.data
        X = preprocess(raw, a.prep)
        skel = skeleton_matrix(ps.pag, names)
        post_res = post.predict(zscore(X)) if post is not None else None
        tr = Transformer(a.transform).fit(raw)
        Zn = tr.transform(raw)
        cases = [c for c in list_issue_cases(os.path.join(root, scn)) if c.target_metric == a.metric]
        print(f"[{scn}] {len(cases)} {a.metric} issues (train {sum(c.split == 'train' for c in cases)}, "
              f"test {sum(c.split == 'test' for c in cases)}); n_normal={len(raw)} d={len(names)}", flush=True)
        # one (source, graph_seed) per graph; the truth graph does not depend on the seed
        gspecs = [(s, -1) for s in a.sources if s == "truth"] + [(s, gs) for s in a.sources if s != "truth" for gs in a.graph_seeds]
        graphs, models = {}, {}
        for s, gs in gspecs:
            a.graph_seed = gs
            D, B, meta = get_graph(s, scn, ps, X, names, post, post_res, skel, a)
            graphs[(s, gs)] = (D, B, meta)
            models[(s, gs, "dag")] = fit_rca_model(Zn, names, D, np.zeros_like(B))
            models[(s, gs, "admg")] = fit_rca_model(Zn, names, D, B)
            print(f"[{scn}] graph {s:9s} seed {gs:2d}: {int(D.sum())} directed + {int(np.triu(B).sum())} bidirected  ({meta.get('kind')})", flush=True)
        for c in cases:
            t_idx, t_how = match_node(c.target_node, names)
            r_idx, r_how = match_node(c.root_node, names)
            if t_idx is None:
                print(f"   skip {c.split}/{c.issue}: target '{c.target_node}' not among the observed nodes ({t_how})", flush=True)
                continue
            Xa, info = load_issue_metrics(c, names, raw, a.metric, a.statistic, a.window)
            Za = tr.transform(Xa)
            dmu = Za.mean(axis=0)
            variants = [("none", -1, "anomaly", "none"), ("none", -1, "corr", "none")]
            for s, gs in gspecs:
                variants += [(s, gs, "anomaly", "first"), (s, gs, "anomaly", "only")]
                variants += [(s, gs, m, anc) for m in ("dag", "admg") for anc in ANCESTOR_MODES]
            for s, gs, m, anc in variants:
                model = None
                if m in ("dag", "admg"):
                    model = models[(s, gs, m)]
                elif s != "none":
                    model = models[(s, gs, "admg")]          # only used for the ancestor set
                order = rank_root_causes(dmu, t_idx, m, model, ancestor_mode=anc, Za=Za)
                rk = rank_of(order, r_idx) if r_idx is not None else None
                rows.append(dict(scenario=scn, split=c.split, issue=c.issue, source=s, graph_seed=gs, method=m,
                                 ancestor_mode=anc, ancestors_only=(anc == "only"),
                                 rank=rk, top1=bool(rk == 1), top3=bool(rk is not None and rk <= 3),
                                 rr=(1.0 / rk if rk else 0.0), n_candidates=int(len(order)),
                                 root_node=c.root_node, root_observed=r_idx is not None, root_match=r_how,
                                 target_node=c.target_node, target_match=t_how, transform=a.transform, window=info["window_used"],
                                 n_abnormal=info["n_abnormal"], n_rows=info["n_rows"], missing_cols=info["missing_cols"],
                                 graph_meta=graphs[(s, gs)][2] if s != "none" else None,
                                 top_pred=names[order[0]] if len(order) else None))
            if r_idx is None:
                print(f"   note {c.split}/{c.issue}: root cause '{c.root_node}' is not an observed node ({r_how}); counted as a miss", flush=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {a.out}")
    return rows


if __name__ == "__main__":
    main()
