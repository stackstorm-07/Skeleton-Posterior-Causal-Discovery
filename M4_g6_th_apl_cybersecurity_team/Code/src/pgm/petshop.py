"""PetShop loader + ground truth.

Data: Hardt et al., "The PetShop Dataset" (Amazon, CC-BY-4.0, Apache-2.0 code).
Ground truth: the call graph (caller -> callee in graph.csv) is *inverted* (callee -> caller),
because latency of a callee propagates to its callers.  Call-graph nodes that are not usable
observed variables (no metric column, mostly missing, or constant) are treated as latent;
the DAG over V u L is latent-projected to an ADMG, then MAG, then PAG (Tetrad).
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from .graphs import ADMG, PAG, dag_with_latents_to_pag

SCENARIOS = ["low_traffic", "high_traffic", "temporal_traffic1", "temporal_traffic2"]


def _scenario_dirs_ok(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "low_traffic", "graph.csv"))


def resolve_root(root: Optional[str] = None) -> str:
    """The PetShop ``dataset`` directory (the folder that contains ``low_traffic/``, ``high_traffic/``, ...).

    Order: the ``root`` argument, then $PETSHOP_ROOT, then two fallbacks.  ``~`` and ``$VARS`` are expanded, relative paths are made absolute,
    and a path to the cloned repository (which contains ``dataset/``) is accepted.  On failure the error says WHAT was tried and WHY it was
    rejected (unset / does not exist / exists but has no ``low_traffic/graph.csv``), so a variable that "is set" but not found can be diagnosed."""
    tried, notes = [], []
    cands = [("argument root", root), ("$PETSHOP_ROOT", os.environ.get("PETSHOP_ROOT")),
             ("fallback /home/claude/ps_data", "/home/claude/ps_data/petshop-root-cause-analysis/dataset"),
             ("fallback data/petshop/dataset", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "petshop", "dataset"))]
    for label, raw in cands:
        if raw is None or raw == "":
            if label in ("argument root", "$PETSHOP_ROOT"):
                notes.append(f"{label}: not set" + (" (an exported variable is needed: `export PETSHOP_ROOT=...`; a plain `PETSHOP_ROOT=...` line is invisible to child processes)" if label == "$PETSHOP_ROOT" else ""))
            continue
        path = os.path.abspath(os.path.expanduser(os.path.expandvars(raw)))
        for cand in (path, os.path.join(path, "dataset")):
            if _scenario_dirs_ok(cand):
                return cand
        tried.append(path)
        why = "does not exist" if not os.path.isdir(path) else "exists but contains no low_traffic/graph.csv (point at the folder that holds low_traffic/, high_traffic/, ...)"
        notes.append(f"{label} = {raw!r} -> {path}: {why}")
    raise FileNotFoundError("PetShop dataset directory not found.\n  " + "\n  ".join(notes) +
                            "\nSet PETSHOP_ROOT to the `dataset` directory of https://github.com/amazon-science/petshop-root-cause-analysis")


def default_root() -> str:
    return resolve_root(None)


@dataclass
class PetShop:
    scenario: str
    names: List[str]            # observed variables (ordered)
    short: List[str]            # readable short names
    data: np.ndarray            # (n, d) raw latency (nan-imputed)
    latents: List[str]
    call_edges: List[tuple]     # caller -> callee, all nodes
    dag_edges: List[tuple]      # inverted: callee -> caller
    admg: ADMG
    mag: ADMG
    pag: PAG
    meta: Dict = field(default_factory=dict)


def _short(n: str) -> str:
    n = n.replace("https://sqs.us-west-2.amazonaws.com/Services-sqspetadoption_AWS::SQS::Queue", "SQS_petadoption")
    n = n.replace(".us-west-2.elb.amazonaws.com", "").replace(".us-west-2.rds.amazonaws.com", "")
    return n.replace("AWS::", "").replace("_remote", "_rem")


def load_petshop(scenario: str = "low_traffic", root: Optional[str] = None, max_missing: float = 0.05,
                 statistic: str = "Average", metric: str = "latency") -> PetShop:
    root = resolve_root(root)
    path = os.path.join(root, scenario)
    G = pd.read_csv(os.path.join(path, "graph.csv"), index_col=0)
    all_nodes = list(G.index)
    call_edges = [(a, b) for a in all_nodes for b in all_nodes if G.loc[a, b] > 0]
    m = pd.read_csv(os.path.join(path, "noissue", "metrics.csv"), header=[0, 1, 2], index_col=0)
    L = m.xs((metric, statistic), axis=1, level=[1, 2])
    obs, lat = [], []
    for v in all_nodes:
        if v in L.columns and L[v].isna().mean() <= max_missing and L[v].nunique() > 1:
            obs.append(v)
        else:
            lat.append(v)
    X = L[obs].interpolate(limit_direction="both").ffill().bfill().values.astype(float)
    dag_edges = [(b, a) for a, b in call_edges]  # invert
    admg, mag, pag = dag_with_latents_to_pag(dag_edges, all_nodes, lat)
    return PetShop(scenario, obs, [_short(v) for v in obs], X, lat, call_edges, dag_edges, admg, mag, pag,
                   meta=dict(n=X.shape[0], d=X.shape[1], n_latent=len(lat)))


# ----- preprocessing -----------------------------------------------------------------

def gauss_rank(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    R = np.apply_along_axis(rankdata, 0, X)
    Z = norm.ppf((R - 0.5) / n)
    return Z


def preprocess(X: np.ndarray, how: str = "gauss_rank") -> np.ndarray:
    if how == "gauss_rank":
        Z = gauss_rank(X)
    elif how == "log":
        Z = np.log1p(np.maximum(X, 0))
    elif how == "none":
        Z = X.copy()
    elif how == "raw_centered":
        # sensitivity check: keep the variables' relative scales (var-sortability), only centre and divide by ONE global
        # scalar so the overall magnitude is ~1 and the optimizer is numerically comfortable
        Z = X - X.mean(0)
        return Z / max(float(Z.std()), 1e-12)
    else:
        raise ValueError(how)
    Z = Z - Z.mean(0)
    sd = Z.std(0)
    sd[sd == 0] = 1
    return Z / sd


PREPROCESSINGS = ("gauss_rank", "none", "raw_centered", "log")   # 'none' = plain z-scoring of raw latency


def zscore(X: np.ndarray) -> np.ndarray:
    sd = X.std(0)
    sd[sd == 0] = 1
    return (X - X.mean(0)) / sd


def lag1_autocorr(X: np.ndarray) -> np.ndarray:
    """Per-variable lag-1 autocorrelation of the rows (rows are assumed to be in time order)."""
    Z = zscore(X)
    return (Z[1:] * Z[:-1]).mean(0)


def skeleton_matrix(pag: PAG, names) -> np.ndarray:
    """Boolean symmetric adjacency matrix of a PAG over ``names`` (the true skeleton for the oracle / posterior check)."""
    idx = {v: i for i, v in enumerate(names)}
    S = np.zeros((len(names), len(names)), dtype=bool)
    for (u, v) in pag.edges:
        S[idx[u], idx[v]] = S[idx[v], idx[u]] = True
    return S
