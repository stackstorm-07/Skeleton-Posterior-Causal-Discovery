import os, sys, dill
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.search.ScoreBased.GES import ges
import p_tqdm

sys.path.append("./")
from learn.gen_feature import Dataset
from utils.graph import compare_pag

def end2end_ges_test(g_path: str, verbose:bool=False):
    dataset = Dataset(g_path)
    g: GeneralGraph = ges(dataset.X)["G"]

    with open(os.path.join(g_path, "cpag.pkl"), "rb") as f:
        cpag : GeneralGraph= dill.load(f) 
    
    if verbose: print("GES", g_path, compare_pag(g.get_graph_edges(), cpag.get_graph_edges()))
    return compare_pag(g.get_graph_edges(), cpag.get_graph_edges())

if __name__ == "__main__":
    # for idx in list(range(10)):
    #     g_p = f"data/test/large_{idx}"
    #     # print(soft_order_ml4s_dcd_test(g_p, 2, True))
    #     rlt = end2end_ges_test(g_p, True)
    #     rlt["g"] = idx
    #     perf.append(rlt)
    perf = list(map(lambda idx: end2end_ges_test(f"data/test/large_{idx}", True), range(10)))
    import pandas as pd
    df = pd.DataFrame(perf)
    df.to_csv("ges_large.csv", index=False)