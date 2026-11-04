from functools import partial
import numpy as np
import os, sys
from p_tqdm import p_umap
sys.path.append("./")
from utils.fci import fci
from utils.ricf import estimate_covariance
from utils.graph import sample_mag
from learn.gen_feature import Dataset

BOOTSTRAP_SUBSAMPLE_RATE = 0.5

def generate_one_vicinal(idx, vicinal_path, dataset, proxy_alg_func):
    vg_path = os.path.join(vicinal_path, f"v_{idx}")
    if os.path.exists(os.path.join(vg_path, "X.npy")): return
    os.makedirs(vg_path, exist_ok=True)
    subsample_idx = np.random.choice(dataset.X.shape[0], size=int(dataset.X.shape[0] * BOOTSTRAP_SUBSAMPLE_RATE), replace=False)
    sub_X = dataset.X[subsample_idx,:]
    fci_g, _ = proxy_alg_func(sub_X)
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
    sigma, delta, beta = estimate_covariance(dataset.X, mag)
    dim = dataset.B.shape[0]
    new_X = np.random.multivariate_normal([0] * dim, sigma, size=dataset.X.shape[0])
    new_X = new_X - np.mean(new_X, axis=0)
    
    np.savetxt(os.path.join(vg_path, "D.txt"), D, fmt='%i')
    np.savetxt(os.path.join(vg_path, "B.txt"), B, fmt='%i')
    np.save(os.path.join(vg_path, "X.npy"), new_X)

def generate_vicinal(g_path:str, num:int=256, proxy_alg:str="fci"):
    if proxy_alg == "fci":
        proxy_alg_func = fci
    else:
        raise NotImplementedError()

    dataset = Dataset(g_path)
    vicinal_path = os.path.join(g_path, "vicinal")
    os.makedirs(vicinal_path, exist_ok=True)
    
    list(map(partial(generate_one_vicinal, vicinal_path=vicinal_path, dataset=dataset, proxy_alg_func=proxy_alg_func), range(num)))

def generate_vicinal_sachs(g_path:str, num:int=32, proxy_alg:str="fci"):
    if proxy_alg == "fci":
        proxy_alg_func = fci
    else:
        raise NotImplementedError()

    dataset = Dataset(g_path, True)
    vicinal_path = os.path.join(g_path, "vicinal")
    os.makedirs(vicinal_path, exist_ok=True)
    
    list(map(partial(generate_one_vicinal, vicinal_path=vicinal_path, dataset=dataset, proxy_alg_func=proxy_alg_func), range(num)))

# def parallel_generate_vicinal(g_path:str, num:int=32, proxy_alg:str="fci"):
#     if proxy_alg == "fci":
#         proxy_alg_func = fci
#     else:
#         raise NotImplementedError()
#     dataset = Dataset(g_path)
#     vicinal_path = os.path.join(g_path, "vicinal")
#     os.makedirs(vicinal_path, exist_ok=True)

#     subsample_idx_list = [np.random.choice(dataset.X.shape[0], size=int(dataset.X.shape[0] * BOOTSTRAP_SUBSAMPLE_RATE), replace=False) for i in range(num)]
#     sub_X_list = [dataset.X[subsample_idx,:].copy() for subsample_idx in subsample_idx_list]
#     mags = [sample_mag(i[0]) for i in p_umap(fci, sub_X_list)]
#     D_list = []
#     B_list = []
#     for mag in mags:
#         D = np.zeros_like(dataset.D)
#         B = np.zeros_like(dataset.B)
#         for Vi, Vj in mag.di_edges:
#             idx_i, idx_j = int(Vi[1:]) -1, int(Vj[1:]) -1
#             D[idx_i, idx_j] = 1
#         for Vi, Vj in mag.bi_edges:
#             idx_i, idx_j = int(Vi[1:]) -1, int(Vj[1:]) -1
#             B[idx_i, idx_j] = 1
#             B[idx_j, idx_i] = 1
#         D_list.append(D)
#         B_list.append(B)
#     sigma_list = [i[0] for i in p_umap(lambda g: estimate_covariance(dataset.X, g),mags)]
#     dim = dataset.B.shape[0]
#     new_X_list = [np.random.multivariate_normal([0] * dim, sigma, size=dataset.X.shape[0]) for sigma in sigma_list]
#     for idx in range(num):
#         new_X_list[idx] = new_X_list[idx] - np.mean(new_X_list[idx], axis=0)

#         vg_path = os.path.join(vicinal_path, f"v_{idx}")
#         np.savetxt(os.path.join(vg_path, "D.txt"), D_list[idx], fmt='%i')
#         np.savetxt(os.path.join(vg_path, "B.txt"), B_list[idx], fmt='%i')
#         np.save(os.path.join(vg_path, "X.npy"), new_X_list[idx])
    


if __name__  == "__main__":
    # for i in range(10):
    #     g_p = f"data/test-sf/large_{i}"

    p_umap(generate_vicinal, [f"data/test-sf/large_{i}" for i in range(10)])