"""Metrics: skeleton / arrowhead / tail precision-recall-F1 against a true PAG.

`compare_pag` is the corrected metric (an edge mark is identified by the *ordered* pair
(from, to) of the node that carries it).  `compare_pag_legacy` reproduces the (node2, node2)
indexing bug of the released SPOT code (utils/graph.py: compare_pag) so numbers can be
compared with the paper's convention.
"""
from __future__ import annotations
from typing import Dict, Set
from .graphs import PAG


def _prf(est: Set, tru: Set) -> Dict[str, float]:
    tp = len(est & tru)
    prec = tp / len(est) if est else 0.0
    rec = tp / len(tru) if tru else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return dict(f1=f1, tpr=rec, fdr=(1 - prec) if est else 0.0, precision=prec, recall=rec,
                n_est=len(est), n_true=len(tru), tp=tp)


def compare_pag(est: PAG, true: PAG) -> Dict[str, Dict[str, float]]:
    return {
        "skeleton": _prf(est.skeleton(), true.skeleton()),
        "arrowhead": _prf(est.arrowheads(), true.arrowheads()),
        "tail": _prf(est.tails(), true.tails()),
    }


def _legacy_items(P: PAG, mark: str) -> Set:
    """Bug-compatible: item is (node2, node2) so only the multiset of head/tail *endpoints* matters."""
    out = set()
    for (u, v), (mu, mv) in P.edges.items():
        if mu == mark:
            out.add((u, u))
        if mv == mark:
            out.add((v, v))
    return out


def compare_pag_legacy(est: PAG, true: PAG) -> Dict[str, Dict[str, float]]:
    return {
        "skeleton": _prf(est.skeleton(), true.skeleton()),
        "arrowhead": _prf(_legacy_items(est, '>'), _legacy_items(true, '>')),
        "tail": _prf(_legacy_items(est, '-'), _legacy_items(true, '-')),
    }


def shd_pag(est: PAG, true: PAG) -> Dict[str, int]:
    """Structural Hamming distances between two PAGs over the same node set (lower is better, 0 = identical).

    * ``shd_skeleton``  adjacencies present in exactly one of the two graphs (extra + missing edges).
    * ``shd_pag``       node pairs whose edge differs in ANY way: absent in one graph, or present in both with a
                        different pair of end marks ('o', '>' , '-').  Every such pair costs 1 (the usual PAG-SHD).
    * ``shd_marks``     finer version: skeleton errors, plus the number of wrong end marks (0, 1 or 2 per shared edge).
    Extra/missing split is returned too (``extra``, ``missing``) so error analysis can reuse it.
    The nodes of ``est`` and ``true`` must be the same set; marks are compared per node, not per stored order.
    """
    if set(est.nodes) != set(true.nodes):
        raise ValueError("PAGs must be defined over the same nodes")

    def marks(P):
        out = {}
        for (u, v), (mu, mv) in P.edges.items():
            out[frozenset((u, v))] = {u: mu, v: mv}      # mark found AT each endpoint
        return out
    E, T = marks(est), marks(true)
    extra = len(E.keys() - T.keys())
    missing = len(T.keys() - E.keys())
    wrong_pairs = wrong_marks = 0
    for k in E.keys() & T.keys():
        bad = sum(E[k][n] != T[k][n] for n in k)
        wrong_pairs += bad > 0
        wrong_marks += bad
    return dict(shd_skeleton=extra + missing, shd_pag=extra + missing + wrong_pairs,
                shd_marks=extra + missing + wrong_marks, extra=extra, missing=missing,
                wrong_mark_pairs=wrong_pairs, n_pairs=len(est.nodes) * (len(est.nodes) - 1) // 2)


def shd_flat(est: PAG, true: PAG, prefix: str = "") -> Dict[str, float]:
    """The SHD numbers as flat result-row entries, plus SHD normalised by the number of true edges."""
    r = shd_pag(est, true)
    out = {f"{prefix}{k}": r[k] for k in ("shd_skeleton", "shd_pag", "shd_marks", "extra", "missing")}
    n_true = max(len(true.edges), 1)
    out[f"{prefix}shd_pag_norm"] = r["shd_pag"] / n_true
    return out


def flat(res: Dict[str, Dict[str, float]], prefix: str = "") -> Dict[str, float]:
    return {f"{prefix}{k}_{m}": v for k, d in res.items() for m, v in d.items() if m in ("f1", "tpr", "fdr")}


def compare_admg(est, true, legacy: bool = False) -> Dict[str, Dict[str, float]]:
    """Score an estimated ADMG against a true ADMG the way the paper does: both are projected to a
    MAG, converted to a PAG with Tetrad, and the PAGs are compared (skeleton / arrowhead / tail)."""
    from .graphs import admg_to_pag
    cmp = compare_pag_legacy if legacy else compare_pag
    return cmp(admg_to_pag(est), admg_to_pag(true))
