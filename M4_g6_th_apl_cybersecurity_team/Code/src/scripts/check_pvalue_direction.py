"""The "p > alpha" check.

A small p-value from a Fisher-z test means the pair is DEPENDENT.  The released SPOT/ML4S ``gen_feature.py``
keeps a pair in the order-0 screen when ``pval > alpha1`` (``prepare_zero_order``), adds a variable to the Markov
blanket when ``cit(i, j, mb) > alpha1`` and removes one when ``cit(...) <= alpha2``.  Read literally that selects
INDEPENDENT pairs / variables.  This script measures what that does, in three steps, all on freshly simulated
held-out graphs that no model was trained on:

  1. order-0 screen:   share of TRUE edges and of NON-edges kept by ``p <= alpha`` (correct) vs ``p > alpha`` (legacy)
  2. Markov blanket:   recall of true neighbours and blanket size, correct vs legacy comparisons
  3. whole cascade:    four posteriors trained on identical data, differing only in the comparison direction
                       (and in whether the stage-0 classifier or the order-0 screen starts the cascade);
                       held-out AUROC / AUPRC / KL / recall at p>=0.05.

    python scripts/check_pvalue_direction.py [--n-train 120 --n-test 40]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from skeleton.simulator import make_instances                                  # noqa: E402
from skeleton.features import CITable, markov_blanket, zero_order_adjacency     # noqa: E402
from skeleton.posterior import PosteriorConfig, SkeletonPosterior, evaluate     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def screen_and_blanket(test, alpha=0.15):
    rows = []
    for legacy in (False, True):
        kept_true = kept_non = n_true = n_non = 0
        mb_rec, mb_size = [], []
        for inst in test:
            ci = CITable(inst.X, orders=1)
            A = zero_order_adjacency(ci, alpha, legacy_inverted=legacy)
            S = inst.skeleton
            iu = np.triu_indices(S.shape[0], 1)
            s, a = S[iu], A[iu]
            kept_true += int((a & s).sum()); n_true += int(s.sum())
            kept_non += int((a & ~s).sum()); n_non += int((~s).sum())
            mb = markov_blanket(ci, alpha, alpha, legacy_inverted=legacy, max_size=12)
            for i in range(S.shape[0]):
                nb = np.nonzero(S[i])[0]
                if len(nb):
                    mb_rec.append(np.mean([j in mb[i] for j in nb]))
                mb_size.append(len(mb[i]))
        rows.append(dict(variant="legacy (p > alpha)" if legacy else "correct (p <= alpha)",
                         screen_keeps_true_edges=kept_true / n_true, screen_keeps_non_edges=kept_non / n_non,
                         mb_recall_of_true_neighbours=float(np.mean(mb_rec)), mb_mean_size=float(np.mean(mb_size))))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--d-min", type=int, default=25)
    ap.add_argument("--d-max", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pvalue_direction.json"))
    a = ap.parse_args(argv)
    kw = dict(d_range=(a.d_min, a.d_max), n_range=(300, 2000), ar_rhos=(0.0, 0.3, 0.6))
    train = make_instances(a.n_train, seed=a.seed, **kw)
    test = make_instances(a.n_test, seed=a.seed + 10_000, **kw)

    out = dict(config=vars(a))
    t = time.time()
    out["step1_2_screen_and_blanket"] = screen_and_blanket(test)
    for r in out["step1_2_screen_and_blanket"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    print(f"  ({time.time() - t:.0f}s)")

    variants = {
        "A  corrected, MB-classifier stage 0":        PosteriorConfig(seed=a.seed),
        "B  legacy-inverted, MB-classifier stage 0":  PosteriorConfig(seed=a.seed, legacy_inverted=True),
        "C  corrected, order-0 screen stage 0":       PosteriorConfig(seed=a.seed, stage0="alpha"),
        "D  legacy (released), order-0 screen":       PosteriorConfig(seed=a.seed, stage0="alpha", legacy_inverted=True),
    }
    out["step3_cascade"] = []
    for name, cfg in variants.items():
        t = time.time()
        post = SkeletonPosterior(cfg).fit(train)
        rows = evaluate(post, test)
        m = {k: float(np.mean([r[k] for r in rows])) for k in ("auroc", "auprc", "kl", "recall_p05", "recall_p10")}
        m.update(variant=name, train_seconds=time.time() - t,
                 auroc_sd=float(np.std([r["auroc"] for r in rows])), n_test=len(rows),
                 per_graph_auroc=[r["auroc"] for r in rows], per_graph_auprc=[r["auprc"] for r in rows])
        out["step3_cascade"].append(m)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()
               if not k.startswith("per_graph")}, flush=True)
    # paired comparison against variant A on the same held-out graphs
    base = out["step3_cascade"][0]
    out["paired_vs_A"] = []
    for m in out["step3_cascade"][1:]:
        for metric in ("auroc", "auprc"):
            diff = np.array(base[f"per_graph_{metric}"]) - np.array(m[f"per_graph_{metric}"])
            out["paired_vs_A"].append(dict(variant=m["variant"], metric=metric, mean_gain_of_A=float(diff.mean()),
                                           se=float(diff.std(ddof=1) / np.sqrt(len(diff))),
                                           graphs_where_A_better=int((diff > 0).sum()), n=len(diff)))
            print(out["paired_vs_A"][-1], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
