import itertools, networkx as nx
from pgm.graphs import *
from pgm.metrics import compare_pag, compare_pag_legacy
from pgm.petshop import load_petshop

def test_latent_projection():
    g = latent_projection([("L","A"),("L","B"),("A","C"),("B","M"),("M","C")], list("LABMC"), ["L","M"])
    assert ("B","C") in g.di_edges and ("A","C") in g.di_edges and g.has_biedge("A","B")

def test_tetrad_toy():
    m = ADMG(["X1","X2","X3","X4"], [("X1","X2"),("X3","X2")], [("X3","X4")])
    p = mag_to_pag(mag_projection(m))
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
