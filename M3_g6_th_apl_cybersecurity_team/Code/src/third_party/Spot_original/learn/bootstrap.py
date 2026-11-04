from functools import partial
import numpy as np
import pandas as pd
import os, sys, p_tqdm, tempfile
from sklearn import metrics
sys.path.append("./")
from utils.fci import fci
from utils.graph import sample_mag, read_causal_cmd_graph
from learn.gen_feature import Dataset
from learn.cmp_skl import compute_aucroc

BOOTSTRAP_SUBSAMPLE_RATE = 0.5

def fci(dataset):
    subsample_idx = np.random.choice(dataset.X.shape[0], size=int(dataset.X.shape[0] * BOOTSTRAP_SUBSAMPLE_RATE), replace=False)
    sub_X = dataset.X[subsample_idx,:]
    fci_g, _ = fci(sub_X)
    mag = sample_mag(fci_g)
    D = np.zeros_like(dataset.D)
    B = np.zeros_like(dataset.B)
    for Vi, Vj in mag.di_edges:
        idx_i, idx_j = int(Vi[1:]) -1, int(Vj[1:]) -1
        D[idx_i, idx_j] = 1
    for Vi, Vj in mag.bi_edges:
        idx_i, idx_j = int(Vi[1:]) -1, int(Vj[1:]) -1
        B[idx_i, idx_j] = 1
        B[idx_j, idx_i] = 1
    return D, B

def rfci(dataset:Dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        subsample_idx = np.random.choice(dataset.X.shape[0], size=int(dataset.X.shape[0] * BOOTSTRAP_SUBSAMPLE_RATE), replace=False)
        sub_X = dataset.X[subsample_idx,:]
        df = pd.DataFrame(sub_X)
        data_path = os.path.join(tmpdir, "data.csv")
        df.to_csv(data_path, index=False)
        cmd = f"java -jar learn/bin/causal-cmd-1.3.0-jar-with-dependencies.jar  --algorithm rfci --data-type continuous --dataset {data_path} "
        cmd = cmd + f"--delimiter comma --test fisher-z-test -out {tmpdir}"
        os.system(cmd)
        try:
            graph_file = [f for f in os.listdir(tmpdir) if "rfci" in f][0]
            pag = read_causal_cmd_graph(os.path.join(tmpdir, graph_file))
        except:
            print(os.listdir(tmpdir))
            exit()
    mag = sample_mag(pag)
    D = np.zeros_like(dataset.D)
    B = np.zeros_like(dataset.B)
    for Vi, Vj in mag.di_edges:
        idx_i, idx_j = int(Vi[1:]) -1, int(Vj[1:]) -1
        D[idx_i, idx_j] = 1
    for Vi, Vj in mag.bi_edges:
        idx_i, idx_j = int(Vi[1:]) -1, int(Vj[1:]) -1
        B[idx_i, idx_j] = 1
        B[idx_j, idx_i] = 1
    return D, B


def bootstrap(g_path, algo=fci):
    dataset = Dataset(g_path)
    D_list, B_list = [], []
    for i in range(100):
        D, B = algo(dataset)
        D_list.append(D)
        B_list.append(B)
    D_mean = np.zeros_like(D_list[0])
    B_mean = np.zeros_like(D_list[0])
    S_mean = np.zeros_like(D_mean)
    for (Vi, Vj), _ in np.ndenumerate(D_list[0]):
        D_mean[Vi,Vj] = np.mean([D[Vi,Vj] for D in D_list])
        B_mean[Vi,Vj] = np.mean([B[Vi,Vj] for B in B_list])
    for (Vi, Vj), _ in np.ndenumerate(S_mean):
        if Vi < Vj:
            S_mean[Vi,Vj] = min(1, D_mean[Vi,Vj] + D_mean[Vj,Vi] + B_mean[Vi,Vj])
            S_mean[Vj,Vi] = min(1, D_mean[Vi,Vj] + D_mean[Vj,Vi] + B_mean[Vi,Vj])
    return S_mean    

if __name__ == "__main__":
    def _helper(i):
        g_p = f"data/test/large_{i}"
        S_mean = bootstrap(g_p, rfci)
        print(g_p, compute_aucroc(g_p, S_mean))
        np.save(os.path.join(g_p, "bootstrap_rfci.npy"), S_mean)
    list(map(_helper, [8,9]))
        
