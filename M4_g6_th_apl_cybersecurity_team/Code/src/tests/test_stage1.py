import itertools, networkx as nx
from pgm.graphs import *
from pgm.metrics import compare_pag, compare_pag_legacy
from pgm.petshop import load_petshop as _load_petshop


class SkipTest(Exception):
    pass


def load_petshop(*a, **k):
    # the PetShop dataset is an external download (PETSHOP_ROOT); skip, don't fail, when it is absent
    try:
        return _load_petshop(*a, **k)
    except FileNotFoundError as e:
        try:
            import pytest
            pytest.skip(str(e))
        except ImportError:
            raise SkipTest(str(e))

def test_latent_projection():
    g = latent_projection([("L","A"),("L","B"),("A","C"),("B","M"),("M","C")], list("LABMC"), ["L","M"])
    assert ("B","C") in g.di_edges and ("A","C") in g.di_edges and g.has_biedge("A","B")

def test_tetrad_toy():
    m = ADMG(["X1","X2","X3","X4"], [("X1","X2"),("X3","X2")], [("X3","X4")])
    try:
        p = mag_to_pag(mag_projection(m))
    except FileNotFoundError as e:           # the Tetrad jar is an external file (see README); skip, don't fail
        try:
            import pytest
            pytest.skip(str(e))
        except ImportError:
            raise SkipTest(str(e))
    s = set(p.to_lines())
    assert "X1 o-> X2" in s and "X2 <-o X3" in s and "X3 o-o X4" in s, s

def test_metrics_identity():
    p = load_petshop("low_traffic").pag
    r = compare_pag(p, p)
    assert all(abs(r[k]["f1"] - 1) < 1e-9 for k in r)
    assert compare_pag_legacy(p, p)["tail"]["f1"] == 1.0

def test_ground_truth_inverted():
    p = load_petshop("low_traffic")
    # PetSite calls PetSearch fargate -> true edge fargate -> PetSite
    assert ("PetSearch_AWS::ECS::Fargate", "PetSite") in p.admg.di_edges


def test_shd_pag():
    from pgm.metrics import shd_pag, shd_flat
    nodes = list("ABCD")
    t = PAG(nodes); t.add_edge("A", "B", "-", ">"); t.add_edge("B", "C", "o", "o"); t.add_edge("C", "D", ">", ">")
    assert shd_pag(t, t) == dict(shd_skeleton=0, shd_pag=0, shd_marks=0, extra=0, missing=0, wrong_mark_pairs=0, n_pairs=6)
    e = PAG(nodes)
    e.add_edge("A", "B", ">", "-")          # reversed orientation: both marks wrong
    e.add_edge("B", "C", "o", ">")          # one mark wrong
    e.add_edge("A", "D", "o", "o")          # extra edge;  C-D is missing
    r = shd_pag(e, t)
    assert (r["extra"], r["missing"], r["wrong_mark_pairs"]) == (1, 1, 2)
    assert r["shd_skeleton"] == 2 and r["shd_pag"] == 4 and r["shd_marks"] == 2 + 2 + 1
    # symmetric in which node is stored first
    e2 = PAG(list("DCBA")); e2.add_edge("B", "A", ">", "-"); e2.add_edge("C", "B", ">", "o"); e2.add_edge("D", "A", "o", "o")
    r2 = shd_pag(e2, t)            # nodes listed in another order; A-B now correct, B-C has one wrong mark
    assert (r2["extra"], r2["missing"], r2["wrong_mark_pairs"], r2["shd_pag"]) == (1, 1, 1, 3)
    f = shd_flat(e, t, "c_")
    assert f["c_shd_pag"] == 4 and abs(f["c_shd_pag_norm"] - 4 / 3) < 1e-12
    try:
        shd_pag(PAG(["A"]), t); assert False
    except ValueError:
        pass
