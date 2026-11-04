"""Validate the inference layer on PetShop: do the fitted ADMGs predict held-out data?

    python scripts/validate_inference.py --scenarios low_traffic high_traffic --sources truth abic spot_cal --out results/inference_validation.json

Protocol (time-ordered, no leakage): first 70% of the normal rows fit the transform and every model, last 30% score it.
Queries
  A  target = the root service (PetSite / frontend, the sink) given its TRUE directed ancestors
  B  every service given ALL other services (averaged over services)
Models: marginal (no dependence) | saturated Gaussian (full covariance, no graph) | ADMG / DAG from each graph source.
Metric: mean held-out log density of the target (nats/row; higher is better) and RMSE (log-standardised units).
Learned graphs are read from the cache written by scripts/run_rca.py (``--graphs-dir``); missing ones are skipped.
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
from pgm.inference import ADMGInference, predictive_scores                      # noqa: E402
from pgm.petshop import SCENARIOS, load_petshop, default_root                     # noqa: E402
from pgm.rca import Transformer, admg_from_json, directed_ancestors, masks_of      # noqa: E402


def _flat(Z, names, Omega):
    return ADMGInference(names, np.zeros((len(names),) * 2), Omega, mean=Z.mean(axis=0))


def evaluate(Ztr, Zte, names, graphs, sink):
    """graphs: {source: (D, B)}. Returns {model: {A: {...}, B: {...}}}."""
    d = len(names)
    models = {"marginal": _flat(Ztr, names, np.diag(Ztr.var(axis=0) + 1e-6)),
              "saturated": _flat(Ztr, names, np.cov(Ztr.T) + 1e-6 * np.eye(d))}
    for s, (D, B) in graphs.items():
        models[f"{s}/admg"] = ADMGInference.fit(Ztr, names, D, B)
        models[f"{s}/dag"] = ADMGInference.fit(Ztr, names, D, np.zeros_like(B))
    Dt = graphs["truth"][0] if "truth" in graphs else None
    anc = [int(i) for i in np.flatnonzero(directed_ancestors(Dt, sink))] if Dt is not None else []
    out = {}
    for m, inf in models.items():
        r = {}
        if anc:
            r["A"] = predictive_scores(inf, Zte, sink, anc)
        rows = [predictive_scores(inf, Zte, j, [k for k in range(d) if k != j]) for j in range(d)]
        r["B"] = {k: float(np.mean([x[k] for x in rows])) for k in ("logpdf", "rmse")}
        out[m] = r
    return out, [names[i] for i in anc]


ROLLING_FRACS = (0.5, 0.6, 0.7, 0.8, 0.9)


def rolling_rows(Zfull_raw, names, graphs, graph_t_opt, sink, transform="log", fracs=ROLLING_FRACS, test_frac=0.1):
    """E8: expanding-window, time-ordered evaluation.  Split k trains on rows [0, f_k * n) and tests on the next ``test_frac * n`` rows
    (the last split tests on the final rows).  The transform is fitted on the training rows only.  One row per (split, model, query).

    ``graphs``: {source: (D, B)} (include "truth" for query A).  ``graph_t_opt``: {source: seconds the graph took to learn} (reported
    next to the inference error: the E8 "error vs computational time" plot uses fit time + graph learning time).
    Metric: mean held-out log density (nats/row, higher is better) and RMSE; the model fit time ``t_fit`` is measured here."""
    n, d = Zfull_raw.shape
    Dt = graphs["truth"][0] if "truth" in graphs else None
    anc = [int(i) for i in np.flatnonzero(directed_ancestors(Dt, sink))] if Dt is not None else []
    rows = []
    for k, f in enumerate(fracs):
        cut = int(f * n)
        end = min(n, cut + max(int(test_frac * n), 10)) if f < fracs[-1] else n
        tr = Transformer(transform).fit(Zfull_raw[:cut])
        Ztr, Zte = tr.transform(Zfull_raw[:cut]), tr.transform(Zfull_raw[cut:end])
        models = {}                                    # name -> (inference object, t_fit, source, structure)
        t0 = time.perf_counter()
        models["marginal"] = (_flat(Ztr, names, np.diag(Ztr.var(axis=0) + 1e-6)), time.perf_counter() - t0, "none", "marginal")
        t0 = time.perf_counter()
        models["saturated"] = (_flat(Ztr, names, np.cov(Ztr.T) + 1e-6 * np.eye(d)), time.perf_counter() - t0, "none", "saturated")
        for s, (D, B) in graphs.items():
            for structure, Bm in (("admg", B), ("dag", np.zeros_like(B))):
                t0 = time.perf_counter()
                inf = ADMGInference.fit(Ztr, names, D, Bm)
                models[f"{s}/{structure}"] = (inf, time.perf_counter() - t0, s, structure)
        for m, (inf, t_fit, src, structure) in models.items():
            base = dict(split=k, train_frac=f, n_train=int(cut), n_test=int(end - cut), model=m, source=src, structure=structure,
                        t_fit=float(t_fit), t_graph_learn=float(graph_t_opt.get(src, 0.0)), transform=transform)
            if anc:
                rows.append(dict(base, query="A", n_evidence=len(anc), **predictive_scores(inf, Zte, sink, anc)))
            sc = [predictive_scores(inf, Zte, j, [q for q in range(d) if q != j]) for j in range(d)]
            rows.append(dict(base, query="B", n_evidence=d - 1,
                             **{k_: float(np.mean([x[k_] for x in sc])) for k_ in ("logpdf", "rmse", "cond_sd")}))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--scenarios", nargs="+", default=["low_traffic", "high_traffic"], choices=SCENARIOS)
    ap.add_argument("--sources", nargs="+", default=["truth", "abic", "spot_cal"])
    ap.add_argument("--graph-seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--prep", default="gauss_rank")
    ap.add_argument("--transform", default="log")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--graphs-dir", default=os.path.join(ROOT, "results", "rca_graphs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "inference_validation.json"))
    a = ap.parse_args(argv)
    root = a.root or default_root()
    res = {}
    for scn in a.scenarios:
        ps = load_petshop(scn, root)
        names = ps.names
        n = len(ps.data)
        cut = int(a.train_frac * n)
        tr = Transformer(a.transform).fit(ps.data[:cut])
        Ztr, Zte = tr.transform(ps.data[:cut]), tr.transform(ps.data[cut:])
        sink = names.index("PetSite") if "PetSite" in names else 0
        D, B = masks_of(ps.mag, names)
        for gs in a.graph_seeds:
            graphs = {"truth": (D, B)}
            for s in a.sources:
                if s == "truth":
                    continue
                p = os.path.join(a.graphs_dir, f"{scn}_{s}_{a.prep}_seed{gs}.json")
                if not os.path.exists(p):
                    print(f"[{scn}] no cached graph {p}; skipping {s}")
                    continue
                graphs[s] = masks_of(admg_from_json(json.load(open(p))["admg"]), names)
            out, anc = evaluate(Ztr, Zte, names, graphs, sink)
            res[f"{scn}/seed{gs}"] = dict(n_train=cut, n_test=n - cut, sink=names[sink], sink_ancestors=anc, models=out)
            print(f"\n[{scn} seed {gs}] train {cut} / test {n - cut}; query A = {names[sink]} | {len(anc)} true ancestors")
            print(f"{'model':16s} {'A logp':>8s} {'A rmse':>8s} {'B logp':>8s} {'B rmse':>8s}")
            for m, r in out.items():
                A = r.get("A", {})
                print(f"{m:16s} {A.get('logpdf', float('nan')):8.3f} {A.get('rmse', float('nan')):8.3f} "
                      f"{r['B']['logpdf']:8.3f} {r['B']['rmse']:8.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    return res


if __name__ == "__main__":
    main()
