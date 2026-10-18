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


def flat(res: Dict[str, Dict[str, float]], prefix: str = "") -> Dict[str, float]:
    return {f"{prefix}{k}_{m}": v for k, d in res.items() for m, v in d.items() if m in ("f1", "tpr", "fdr")}


def compare_admg(est, true, legacy: bool = False) -> Dict[str, Dict[str, float]]:
    """Score an estimated ADMG against a true ADMG the way the paper does: both are projected to a
    MAG, converted to a PAG with Tetrad, and the PAGs are compared (skeleton / arrowhead / tail)."""
    from .graphs import admg_to_pag
    cmp = compare_pag_legacy if legacy else compare_pag
    return cmp(admg_to_pag(est), admg_to_pag(true))
