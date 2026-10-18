import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import os, json, dill, sys
from p_tqdm import p_umap
sys.path.append("./")
from utils.graph import adj2admg, admg_to_pag, pag2cpag

random.seed(42)

# path
data_path = "./data"

# general parameter
nodenumclass = {'vsmall': (5, 10), 'small': (11, 20), 'medium': (21, 50), 'large': (51, 100), 'vlarge': (101, 200)}
avgInDegree = (1.0, 1.5) # == avgOutDegree == #Edges/#Nodes, lower/ upper bound
avgBiRate = (0.05, 0.15)

# training data parameter
train_path = os.path.join(data_path, "train")
train_numeachclass = {'vsmall': 0, 'small': 0, 'medium': 1000, 'large': 1000, 'vlarge': 0}
os.makedirs(train_path, exist_ok=True)

# test data parameter
test_path = os.path.join(data_path, "test-sf")
test_numeachclass = {'vsmall': 0, 'small': 0, 'medium': 0, 'large': 10, 'vlarge': 0}
os.makedirs(test_path, exist_ok=True)

def simulate_mag(d, edge, bi, graph_type):
    """Simulate random DAG with some expected number of edges.
    Args:
        d (int): num of nodes
        edge (int): expected num of edges
        bi (float): expected proportion of bidirected edges
        graph_type (str): ER, SF, BP
    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=edge)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(edge / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=edge, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()

    # generate directed edges and bidirected edges
    B_adj_mat = np.random.binomial(1, bi, B_perm.shape) * B_perm
    D_adj_mat = B_perm - B_adj_mat
    for iy in range(d):
        for ix in range(iy):
            has_edge = B_adj_mat[iy, ix] + B_adj_mat[ix, iy]
            B_adj_mat[iy, ix] = has_edge
            B_adj_mat[ix, iy] = has_edge

    return D_adj_mat, B_adj_mat

def simulate_linear_sem(D: np.ndarray, B: np.ndarray):
    beta = np.random.uniform(0.5, 2.0, size=D.shape) * np.random.choice([-1, 1], size=D.shape)
    beta *= D.T
    omega = np.random.uniform(0.4, 0.7, size=D.shape) * np.random.choice([-1, 1], size=D.shape)
    omega = np.triu(omega)
    omega += omega.T
    omega *= B
    row_wise_sum = np.sum(np.abs(omega), axis=1)
    diagonal = np.random.uniform(0.7, 1.2) + row_wise_sum
    # print(diagonal, omega)
    np.fill_diagonal(omega, np.abs(diagonal))
    # print(omega)
    return beta, omega

def simulate_data(beta: np.ndarray, omega: np.ndarray, size:int):
    dim = beta.shape[0]
    true_sigma = np.linalg.inv(np.eye(dim) - beta) @ omega @ np.linalg.inv((np.eye(dim) - beta).T)
    X = np.random.multivariate_normal([0] * dim, true_sigma, size=size)
    X = X - np.mean(X, axis=0)  # centre the data
    return X

def is_adjacent(D: np.ndarray, B: np.ndarray, i: int, j: int):
    return D[i,j] == 1 or D[j,i] == 1 or B[i,j] ==1 or B[j,i] == 1

def has_di_edge(D: np.ndarray, B: np.ndarray, i: int, j: int):
    return D[i,j] == 1 or B[i,j] ==1 or B[j,i] == 1

def extract_instance(D: np.ndarray, B: np.ndarray, g_path: str):
    node_size = D.shape[0]
    tforks = set()
    vstrucs = set()
    for j in range(node_size):
        for i in range(node_size):
            for k in range(node_size):
                if k <= i or j == i or j == k: continue
                # i-j-k
                if is_adjacent(D, B, i, j) and is_adjacent(D, B, k, j) and not is_adjacent(D, B, i, k):
                    tforks.add((j, i, k))
                    # i->j<-k
                    if has_di_edge(D, B, i, j) and has_di_edge(D, B, k, j):
                        vstrucs.add((j, i, k))
    np.savetxt(os.path.join(g_path, "vstrucs.txt"), np.array(sorted(list(vstrucs)), dtype=int), fmt='%i')
    np.savetxt(os.path.join(g_path, "tforks.txt"), np.array(sorted(list(tforks)), dtype=int), fmt='%i')

def mag2pag(D, B):
    admg = adj2admg(D, B)
    pag = admg_to_pag(admg)
    cpag = pag2cpag(pag)
    return cpag

def generate_data():

    def generate_by_conf(conf: dict):
        g_path = conf["path"]
        os.makedirs(g_path, exist_ok=True)
        with open(os.path.join(g_path, "conf.json"), "w") as f:
            json.dump(conf, f, indent=0)
        D, B = simulate_mag(conf["node"], conf["edge"], conf["bi"], "ER")
        np.savetxt(os.path.join(g_path, "D.txt"), D, fmt='%i')
        np.savetxt(os.path.join(g_path, "B.txt"), B, fmt='%i')
        extract_instance(D, B, g_path)
        beta, omega = simulate_linear_sem(D,B)
        X = simulate_data(beta, omega, 1000)
        np.save(os.path.join(g_path, "X.npy"), X)
        cpag = mag2pag(D, B)
        with open(os.path.join(g_path, "cpag.pkl"), "wb") as f:
            dill.dump(cpag, f)

    # generate conf
    conf_list = []
    # for graph_class in train_numeachclass.keys():
    #     for graph_idx in range(train_numeachclass[graph_class]):
    #         node_num = np.random.randint(nodenumclass[graph_class][0], nodenumclass[graph_class][1])
    #         edge_num = int(node_num * (np.random.random() * .7 + 1))
    #         bi_rate = 0.05 + np.random.random() * 0.15
    #         conf = {
    #             "node": node_num, 
    #             "edge": edge_num, 
    #             "bi": bi_rate,
    #             "path": os.path.join(train_path, f"{graph_class}_{graph_idx}")
    #         }
    #         conf_list.append(conf)
    for graph_class in test_numeachclass.keys():
        for graph_idx in range(test_numeachclass[graph_class]):
            node_num = np.random.randint(nodenumclass[graph_class][0], nodenumclass[graph_class][1])
            edge_num = int(node_num * (np.random.random() * .7 + 1))
            bi_rate = 0.05 + np.random.random() * 0.15
            conf = {
                "node": node_num, 
                "edge": edge_num, 
                "bi": bi_rate,
                "path": os.path.join(test_path, f"{graph_class}_{graph_idx}")
            }
            conf_list.append(conf)
    
    p_umap(generate_by_conf, conf_list)

if __name__ == "__main__":
    # D_adj_mat, B_adj_mat = simulate_mag(4,6,0.3,"ER")
    # beta, omega = simulate_linear_sem(D_adj_mat,B_adj_mat)
    # print(simulate_data(beta, omega, 10))
    generate_data()