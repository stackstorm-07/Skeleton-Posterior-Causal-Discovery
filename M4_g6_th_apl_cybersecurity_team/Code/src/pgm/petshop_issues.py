"""PetShop injected-issue cases.

Layout (Hardt et al. 2023, App. B; arXiv 2311.04806):

    <scenario>/graph.csv
    <scenario>/noissue/metrics.csv
    <scenario>/{train,test}/issue<k>/metrics.csv
    <scenario>/{train,test}/issue<k>/target.json

``target.json`` holds ``{"target": {node, metric, agg, timestamp}, "root_cause": {node, metric}, "metadata": ...}``: the SLO-violating
target (PetSite) with the unix time the issue was injected, and exactly one ground-truth root-cause node.  ``metrics.csv`` has the
same 3-level header (node, metric, statistic) as the normal data.  The issues are split into train/test by the microservice they
originate from.

Assumptions that could not be checked without the dataset (each is one parameter, and the runner prints what it found):
  * node names in ``target.json`` equal the names in ``graph.csv``; if not, a unique case-insensitive prefix/substring match is used and flagged;
  * the abnormal window is the set of rows with time >= the injection timestamp (``window='after'``); ``window='all'`` uses the whole file;
  * the row index of ``metrics.csv`` is a datetime or unix seconds.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class IssueCase:
    scenario: str
    split: str
    issue: str
    path: str
    target_node: str
    target_metric: str
    target_agg: str
    timestamp: Optional[float]
    root_node: str
    root_metric: Optional[str]


def list_issue_cases(scenario_dir: str, splits: Sequence[str] = ("train", "test")) -> List[IssueCase]:
    scenario = os.path.basename(os.path.normpath(scenario_dir))
    out: List[IssueCase] = []
    for sp in splits:
        sdir = os.path.join(scenario_dir, sp)
        if not os.path.isdir(sdir):
            continue
        for name in sorted(os.listdir(sdir), key=lambda s: (len(s), s)):
            p = os.path.join(sdir, name)
            tj = os.path.join(p, "target.json")
            if not (os.path.isdir(p) and os.path.exists(tj) and os.path.exists(os.path.join(p, "metrics.csv"))):
                continue
            t = json.load(open(tj))
            tg, rc = t["target"], t["root_cause"]
            out.append(IssueCase(scenario, sp, name, p, tg["node"], tg.get("metric", ""), tg.get("agg", ""),
                                 tg.get("timestamp"), rc["node"], rc.get("metric")))
    return out


def match_node(name: str, names: Sequence[str]) -> Tuple[Optional[int], str]:
    """Index of ``name`` among ``names`` and how it matched: 'exact' | 'ci' | 'fuzzy' | 'none' | 'ambiguous'."""
    if name in names:
        return list(names).index(name), "exact"
    low = [n.lower() for n in names]
    if name.lower() in low:
        return low.index(name.lower()), "ci"
    key = name.lower()
    hits = [i for i, n in enumerate(low) if n.startswith(key) or key in n]
    if len(hits) == 1:
        return hits[0], "fuzzy"
    return None, "ambiguous" if hits else "none"


def _to_seconds(index) -> Optional[np.ndarray]:
    try:
        if np.issubdtype(np.asarray(index).dtype, np.number):
            v = np.asarray(index, dtype=float)
            return v / 1000.0 if v.max() > 1e11 else v
        dt = pd.to_datetime(pd.Index(index), utc=True)
        # resolution-independent (the integer view of a DatetimeIndex is in ns, us or s depending on the pandas version)
        return np.asarray((dt - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds(), dtype=float)
    except Exception:
        return None


def load_issue_metrics(case: IssueCase, names: Sequence[str], baseline: np.ndarray, metric: str = "latency",
                       statistic: str = "Average", window: str = "after", min_rows: int = 3
                       ) -> Tuple[np.ndarray, dict]:
    """Abnormal-window matrix (m, d) of ``metric``/``statistic`` for the variables ``names`` (graph variables).

    Missing columns / values are imputed from the issue file's own interpolation, then with the normal-period median.
    Returns (X_abn, info) with the window actually used and counts, so fallbacks are visible in the results."""
    m = pd.read_csv(os.path.join(case.path, "metrics.csv"), header=[0, 1, 2], index_col=0)
    L = m.xs((metric, statistic), axis=1, level=[1, 2])
    secs = _to_seconds(m.index)
    cols, n_missing_cols = [], 0
    for j, v in enumerate(names):
        if v in L.columns:
            cols.append(L[v].astype(float).interpolate(limit_direction="both").to_numpy())
        else:
            n_missing_cols += 1
            cols.append(np.full(len(L), np.nan))
    X = np.column_stack(cols)
    med = np.median(baseline, axis=0)
    nan = np.isnan(X)
    X[nan] = np.broadcast_to(med, X.shape)[nan]
    info = dict(n_rows=int(len(X)), missing_cols=n_missing_cols, window_used="all")
    if window == "after" and case.timestamp is not None and secs is not None:
        keep = secs >= float(case.timestamp)
        info["rows_after"] = int(keep.sum())
        if keep.sum() >= min_rows:
            X, info["window_used"] = X[keep], "after"
        else:
            info["window_note"] = "fewer than min_rows rows after the injection time; used the whole file"
    elif window == "after":
        info["window_note"] = "could not parse the time index or timestamp; used the whole file"
    info["n_abnormal"] = int(len(X))
    return X, info
