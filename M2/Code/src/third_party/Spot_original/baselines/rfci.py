from functools import partial
import numpy as np
import pandas as pd
import os, sys, dill, tempfile
from causallearn.graph.GeneralGraph import GeneralGraph
sys.path.append("./")
from utils.graph import compare_pag, read_causal_cmd_graph
from learn.gen_feature import Dataset

def rfci(g_path: str, verbose:bool=False):
    dataset = Dataset(g_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        df = pd.DataFrame(dataset.X)
        data_path = os.path.join(tmpdir, "data.csv")
        df.to_csv(data_path, index=False)
        cmd = f"java -jar learn/bin/causal-cmd-1.3.0-jar-with-dependencies.jar  --algorithm rfci --data-type continuous --dataset {data_path} "
        cmd = cmd + f"--delimiter comma --test fisher-z-test -out {tmpdir}"
        os.system(cmd)
        graph_file = [f for f in os.listdir(tmpdir) if "rfci" in f][0]
        pag = read_causal_cmd_graph(os.path.join(tmpdir, graph_file))
    with open(os.path.join(g_path, "cpag.pkl"), "rb") as f:
        cpag : GeneralGraph= dill.load(f) 
    
    if verbose: print("RFCI", g_path, compare_pag(pag.get_graph_edges(), cpag.get_graph_edges()))
    return compare_pag(pag.get_graph_edges(), cpag.get_graph_edges())

if __name__ == "__main__":
    perf = []
    for idx in list(range(10)):
        np.random.seed(42)
        g_p = f"data/test/large_{idx}"
        rlt = gfci(g_p, True)
        rlt["g"] = idx
        perf.append(rlt)
    import pandas as pd
    df = pd.DataFrame(perf)
    df.to_csv("rfci_large.csv", index=False)