"""
Graph utilities: ADMG / MAG / PAG containers, latent projection, MAG projection,
and the Tetrad bridge (MAG -> PAG) used for scoring.

Credits
-------
`inducing_path` / `mag_projection` are ported from the original SPOT artifact
(Ma et al., KDD'24, utils/graph.py, Apache-2.0), which itself follows the DCD/ABIC
code of Bhattacharya et al. (AISTATS'21).  The original relies on the `ananke`
and `causal-learn` packages; this module is dependency-free (numpy + networkx only)
so it runs in restricted environments.  The MAG -> PAG step still uses Tetrad
(through the `causal-cmd` jar that ships with the original artifact), exactly as
the original code does.
"""
from __future__ import annotations

import itertools
import os
import subprocess
import tempfile
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# ADMG
# --------------------------------------------------------------------------------------


class ADMG:
    """Acyclic directed mixed graph: directed edges (a -> b) and bidirected edges (a <-> b)."""

    def __init__(self, vertices: Sequence[str], di_edges: Iterable[Tuple[str, str]] = (),
                 bi_edges: Iterable[Tuple[str, str]] = ()):
        self.vertices: List[str] = list(vertices)
        self.di_edges: Set[Tuple[str, str]] = set()
        self.bi_edges: Set[FrozenSet[str]] = set()
        self._cache: Dict[str, Dict[str, Set[str]]] = {}
        for a, b in di_edges:
            self.add_diedge(a, b)
        for a, b in bi_edges:
            self.add_biedge(a, b)

    # -- mutation ------------------------------------------------------------------
    def add_diedge(self, a: str, b: str):
        self.di_edges.add((a, b))
        self._cache.clear()

    def add_biedge(self, a: str, b: str):
        if a != b:
            self.bi_edges.add(frozenset((a, b)))
            self._cache.clear()

    def has_biedge(self, a: str, b: str) -> bool:
        return frozenset((a, b)) in self.bi_edges

    # -- adjacency helpers -----------------------------------------------------------
    def _maps(self):
        if not self._cache:
            par = {v: set() for v in self.vertices}
            chi = {v: set() for v in self.vertices}
            sib = {v: set() for v in self.vertices}
            for a, b in self.di_edges:
                par[b].add(a)
                chi[a].add(b)
            for e in self.bi_edges:
                a, b = tuple(e)
                sib[a].add(b)
                sib[b].add(a)
            self._cache = {"par": par, "chi": chi, "sib": sib}
        return self._cache

    def parents(self, vs: Iterable[str]) -> Set[str]:
        m = self._maps()["par"]
        return set().union(*[m[v] for v in vs]) if vs else set()

    def children(self, vs: Iterable[str]) -> Set[str]:
        m = self._maps()["chi"]
        return set().union(*[m[v] for v in vs]) if vs else set()

    def siblings(self, vs: Iterable[str]) -> Set[str]:
        m = self._maps()["sib"]
        return set().union(*[m[v] for v in vs]) if vs else set()

    def ancestors(self, vs: Iterable[str]) -> Set[str]:
        """Ancestors *including* the vertices themselves (as in Ananke)."""
        vs = list(vs)
        seen = set(vs)
        stack = list(vs)
        par = self._maps()["par"]
        while stack:
            v = stack.pop()
            for p in par[v]:
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def n_edges(self) -> Tuple[int, int]:
        return len(self.di_edges), len(self.bi_edges)

    def __repr__(self):
        return f"ADMG(|V|={len(self.vertices)}, di={len(self.di_edges)}, bi={len(self.bi_edges)})"


def has_directed_cycle(W: np.ndarray) -> bool:
    """True if the (non-zero pattern of the) matrix W, with W[i, j] != 0 meaning i -> j, has a cycle."""
    A = (np.abs(W) > 0).astype(np.int8)
    d = A.shape[0]
    indeg = A.sum(axis=0).astype(int)
    stack = [i for i in range(d) if indeg[i] == 0]
    removed = 0
    while stack:
        v = stack.pop()
        removed += 1
        for j in np.nonzero(A[v])[0]:
            indeg[j] -= 1
            if indeg[j] == 0:
                stack.append(j)
    return removed != d


# --------------------------------------------------------------------------------------
# Latent projection:  DAG over V u L   ->  ADMG over V
# --------------------------------------------------------------------------------------


def latent_projection(dag_edges: Iterable[Tuple[str, str]], nodes: Sequence[str],
                      latents: Iterable[str]) -> ADMG:
    """Verma-Pearl latent projection of a DAG with latent variables onto the observed variables.

    a -> b  iff there is a directed path a -> l1 -> ... -> b whose interior nodes are all latent;
    a <-> b iff some latent l has directed paths (all interior nodes latent) to both a and b.
    """
    latents = set(latents)
    nodes = list(nodes)
    obs = [v for v in nodes if v not in latents]
    children: Dict[str, Set[str]] = {v: set() for v in nodes}
    for a, b in dag_edges:
        children[a].add(b)

    def reach_through_latents(src: str) -> Set[str]:
        """Observed nodes reachable from src by a directed path whose interior nodes are latent."""
        out: Set[str] = set()
        seen = {src}
        stack = [src]
        while stack:
            u = stack.pop()
            for c in children[u]:
                if c in latents:
                    if c not in seen:
                        seen.add(c)
                        stack.append(c)
                else:
                    out.add(c)
        return out

    G = ADMG(obs)
    for a in obs:
        for b in reach_through_latents(a):
            G.add_diedge(a, b)
    for l in latents:
        R = sorted(reach_through_latents(l))
        for a, b in itertools.combinations(R, 2):
            G.add_biedge(a, b)
    return G


# --------------------------------------------------------------------------------------
# MAG projection (inducing paths), ported from the original artifact
# --------------------------------------------------------------------------------------


def inducing_path(G: ADMG, Vi: str, Vj: str) -> bool:
    if Vi in G.parents([Vj]) or Vj in G.parents([Vi]) or Vi in G.siblings([Vj]):
        return True
    anc = G.ancestors([Vi, Vj])
    stack = [s for s in G.siblings([Vi]) if s in anc]
    stack += [c for c in G.children([Vi]) if c in anc]
    visited: Set[str] = set()
    while stack:
        if Vj in stack or Vj in G.parents(stack):
            return True
        v = stack.pop()
        visited.add(v)
        stack.extend(set(s for s in G.siblings([v]) if s in anc) - visited)
    return False


def mag_projection(G: ADMG) -> ADMG:
    """Maximal ancestral graph (as an ADMG) that encodes the m-separations of the ADMG G."""
    M = ADMG(G.vertices)
    for Vi, Vj in itertools.combinations(G.vertices, 2):
        if inducing_path(G, Vi, Vj):
            if Vi in G.ancestors([Vj]):
                M.add_diedge(Vi, Vj)
            elif Vj in G.ancestors([Vi]):
                M.add_diedge(Vj, Vi)
            else:
                M.add_biedge(Vi, Vj)
    return M


# --------------------------------------------------------------------------------------
# PAG container
# --------------------------------------------------------------------------------------


class PAG:
    """Partial ancestral graph.

    edges[(u, v)] = (mark_at_u, mark_at_v) with u before v in `nodes`;  marks are
    'o' (circle), '>' (arrowhead) or '-' (tail).  For the edge ``u o-> v`` the value is ('o', '>').
    """

    def __init__(self, nodes: Sequence[str]):
        self.nodes = list(nodes)
        self.idx = {v: i for i, v in enumerate(self.nodes)}
        self.edges: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def add_edge(self, u: str, v: str, mark_u: str, mark_v: str):
        if self.idx[u] > self.idx[v]:
            u, v, mark_u, mark_v = v, u, mark_v, mark_u
        self.edges[(u, v)] = (mark_u, mark_v)

    # canonical item sets used by the metrics -------------------------------------------
    def skeleton(self) -> Set[FrozenSet[str]]:
        return {frozenset(e) for e in self.edges}

    def arrowheads(self) -> Set[Tuple[str, str]]:
        """{(x, y)}: the edge between x and y has an arrowhead at y."""
        out = set()
        for (u, v), (mu, mv) in self.edges.items():
            if mu == '>':
                out.add((v, u))
            if mv == '>':
                out.add((u, v))
        return out

    def tails(self) -> Set[Tuple[str, str]]:
        out = set()
        for (u, v), (mu, mv) in self.edges.items():
            if mu == '-':
                out.add((v, u))
            if mv == '-':
                out.add((u, v))
        return out

    def edge_string(self, u: str, v: str) -> str:
        mu, mv = self.edges[(u, v)]
        left = {'o': 'o', '-': '-', '>': '<'}[mu]
        right = {'o': 'o', '-': '-', '>': '>'}[mv]
        return f"{u} {left}-{right} {v}"

    def __len__(self):
        return len(self.edges)

    def __repr__(self):
        return f"PAG(|V|={len(self.nodes)}, |E|={len(self.edges)})"

    def to_lines(self) -> List[str]:
        return [self.edge_string(u, v) for (u, v) in sorted(self.edges)]


def pag_from_mag_marks(mag: ADMG) -> PAG:
    """MAG -> 'PAG' with no equivalence-class completion (every mark determined by the MAG itself)."""
    P = PAG(mag.vertices)
    for a, b in mag.di_edges:
        P.add_edge(a, b, '-', '>')
    for e in mag.bi_edges:
        a, b = tuple(e)
        P.add_edge(a, b, '>', '>')
    return P


# --------------------------------------------------------------------------------------
# Tetrad bridge
# --------------------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DEFAULT_JAR_CANDIDATES = [
    os.environ.get("CAUSAL_CMD_JAR", ""),
    os.path.join(_ROOT, "third_party", "Spot_original", "learn", "bin", "causal-cmd-1.3.0-jar-with-dependencies.jar"),
    os.path.join(_ROOT, "java", "causal-cmd-1.3.0-jar-with-dependencies.jar"),
]


def find_tetrad_jar() -> str:
    for p in DEFAULT_JAR_CANDIDATES:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "causal-cmd jar not found. Set CAUSAL_CMD_JAR or put causal-cmd-1.3.0-jar-with-dependencies.jar "
        "into third_party/Spot_original/learn/bin/ (it ships with the original SPOT artifact).")


def _run_java(main_class: str, path: str):
    jar = find_tetrad_jar()
    cp = os.pathsep.join([jar, os.path.join(_ROOT, "java")])
    r = subprocess.run(["java", "-classpath", cp, main_class, path], capture_output=True, text=True)
    return r


def _parse_tetrad_graph(path: str, names: Dict[str, str], order: Sequence[str]) -> PAG:
    lines = open(path).read().strip().split("\n")
    P = PAG(order)
    start = [i for i, l in enumerate(lines) if l.startswith("Graph Edges")][0]
    for line in lines[start + 1:]:
        line = line.strip()
        if not line:
            continue
        body = line.split(". ", 1)[1].split(" ")
        u, typ, v = body[0], body[1], body[2]
        left = {'o': 'o', '-': '-', '<': '>'}[typ[0]]
        right = {'o': 'o', '-': '-', '>': '>'}[typ[2]]
        P.add_edge(names[u], names[v], left, right)
    return P


def mag_to_pag(mag: ADMG) -> PAG:
    """Convert a MAG into its PAG with Tetrad's MagToPag (complete rule set), as in the original code."""
    order = list(mag.vertices)
    tag = {v: f"X{i + 1}" for i, v in enumerate(order)}
    back = {t: v for v, t in tag.items()}
    with tempfile.TemporaryDirectory() as tmp:
        mag_path = os.path.join(tmp, "G.mag")
        with open(mag_path, "w") as f:
            f.write("Graph Nodes:\n" + ",".join(tag[v] for v in order) + "\n\nGraph Edges:\n")
            k = 1
            for a, b in sorted(mag.di_edges):
                f.write(f"{k}. {tag[a]} --> {tag[b]}\n")
                k += 1
            for e in sorted(tuple(sorted(e)) for e in mag.bi_edges):
                f.write(f"{k}. {tag[e[0]]} <-> {tag[e[1]]}\n")
                k += 1
        r = _run_java("convertMag2Pag", mag_path)
        out = mag_path + ".pag"
        if not os.path.exists(out):
            raise RuntimeError("Tetrad MAG->PAG conversion failed:\n" + r.stdout + r.stderr)
        return _parse_tetrad_graph(out, back, order)


def admg_to_pag(G: ADMG) -> PAG:
    """ADMG -> MAG (inducing-path projection) -> PAG (Tetrad)."""
    return mag_to_pag(mag_projection(G))


def dag_with_latents_to_pag(dag_edges: Iterable[Tuple[str, str]], nodes: Sequence[str],
                            latents: Iterable[str]) -> Tuple[ADMG, ADMG, PAG]:
    """Ground-truth builder: DAG over V u L  ->  (projected ADMG, MAG, PAG) over the observed V."""
    admg = latent_projection(dag_edges, nodes, latents)
    mag = mag_projection(admg)
    return admg, mag, mag_to_pag(mag)
