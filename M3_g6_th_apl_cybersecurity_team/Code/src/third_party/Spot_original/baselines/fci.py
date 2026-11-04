import os, sys, dill
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.cit import fisherz


sys.path.append("./")
from learn.gen_feature import Dataset
from utils.fci import fci
from utils.graph import compare_pag

def end2end_fci_test(g_path: str, verbose:bool=False):
    dataset = Dataset(g_path)
    g, _ = fci(dataset.X, independence_test_method=fisherz)

    with open(os.path.join(g_path, "cpag.pkl"), "rb") as f:
        cpag : GeneralGraph= dill.load(f) 
    
    if verbose: print("FCI", g_path, compare_pag(g.get_graph_edges(), cpag.get_graph_edges()))
    return compare_pag(g.get_graph_edges(), cpag.get_graph_edges())

