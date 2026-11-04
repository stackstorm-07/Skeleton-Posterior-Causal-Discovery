"""FCI / RFCI baselines through Tetrad's ``causal-cmd`` jar (the jar that ships with the SPOT artifact; Java is required).

    from pgm.fci import run_tetrad
    r = run_tetrad(X, names, algorithm="fci", alpha=0.05)        # X: (n, d) array
    r.pag, r.t_wall, r.n_edges

What is run: ``java -jar causal-cmd.jar --algorithm {fci|rfci} --data-type continuous --test fisher-z-test --alpha A``
on the same preprocessed matrix the ADMG learners get.  The output is already a PAG, so it is scored with exactly the same
PAG metrics as the learners (``scripts/spot_vs_abic_synthetic.score_pag``), against the same true PAG.

Notes that matter for reading the numbers
* Node names are replaced by ``X1..Xd`` for the jar (PetShop service names contain characters the jar's reader may not accept)
  and mapped back, so ``r.pag`` is over ``names``.
* ``t_wall`` is the whole subprocess (JVM start-up about 0.9 s, reading the file, search); the jar's own timestamps have 1 s
  resolution and are not used.  Compare it with the learners' ``t_optimize`` only as an order of magnitude.
* One thread (``--thread 1``) so that a parallel queue does not oversubscribe cores.
* FCI/RFCI in Tetrad can return edge types such as ``o-o``, ``o->``, ``<->``, ``-->``; all are parsed.  Tetrad may also
  emit a graph with no edges; that is a valid PAG here (all F1 = 0).
* Not run through any Python causal-discovery library: nothing here depends on causal-learn or XGBoost.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .graphs import PAG, _parse_tetrad_graph, find_tetrad_jar

ALGORITHMS = ("fci", "rfci")


class TetradError(RuntimeError):
    pass


@dataclass
class TetradResult:
    pag: PAG
    algorithm: str
    alpha: float
    t_wall: float                 # seconds, the whole java subprocess
    n_edges: int
    command: str


def _write_table(path: str, X: np.ndarray, tags: Sequence[str]) -> None:
    with open(path, "w") as f:
        f.write("\t".join(tags) + "\n")
        np.savetxt(f, np.asarray(X, dtype=float), fmt="%.10g", delimiter="\t")


def run_tetrad(X: np.ndarray, names: Sequence[str], algorithm: str = "fci", alpha: float = 0.05, depth: int = -1,
               max_path_length: int = -1, complete_rules: bool = False, timeout: Optional[float] = None,
               jar: Optional[str] = None, java_opts: Sequence[str] = ()) -> TetradResult:
    """Run FCI or RFCI (Fisher-z test) on ``X`` and return the PAG.  Raises ``TetradError`` on failure or timeout."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("X must be (n, d) with d = len(names)")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/inf; impute first")
    jar = jar or find_tetrad_jar()
    tags = [f"X{i + 1}" for i in range(len(names))]
    back = dict(zip(tags, names))
    with tempfile.TemporaryDirectory() as tmp:
        data = os.path.join(tmp, "data.txt")
        _write_table(data, X, tags)
        cmd = ["java", *java_opts, "-jar", jar, "--algorithm", algorithm, "--data-type", "continuous",
               "--dataset", data, "--delimiter", "tab", "--test", "fisher-z-test", "--alpha", repr(float(alpha)),
               "--depth", str(int(depth)), "--maxPathLength", str(int(max_path_length)),
               "--thread", "1", "--skip-latest", "--prefix", "out", "--out", tmp]
        if complete_rules:
            cmd += ["--completeRuleSetUsed"]
        t0 = time.perf_counter()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise TetradError(f"{algorithm} timed out after {timeout:.0f}s") from e
        t_wall = time.perf_counter() - t0
        out = os.path.join(tmp, "out.txt")
        if r.returncode != 0 or not os.path.exists(out):
            raise TetradError(f"{algorithm} failed (exit {r.returncode}):\n{(r.stdout + r.stderr)[-800:]}")
        text = open(out).read()
        if "Graph Edges" not in text:
            raise TetradError(f"{algorithm}: no graph in the output:\n{text[-600:]}")
        pag = _parse_tetrad_graph(out, back, list(names))
    return TetradResult(pag=pag, algorithm=algorithm, alpha=float(alpha), t_wall=t_wall,
                        n_edges=len(pag.edges), command=" ".join(cmd[:2] + ["..."] + cmd[4:10]))
