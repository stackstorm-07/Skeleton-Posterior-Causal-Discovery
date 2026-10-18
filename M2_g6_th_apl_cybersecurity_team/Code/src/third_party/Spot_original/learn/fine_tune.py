import time
import numpy as np
import os, sys, random, dill
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edges import Edges
import xgboost as xgb
from p_tqdm import p_umap
from typing import List
sys.path.append("./")
from utils.fci import fci
from utils.ricf import estimate_covariance
from utils.graph import compare_skl, sample_mag
from learn.gen_feature import Dataset, ml4s_generate_vicinal_order_feature
from learn.espi import ESPIModel, end2end_order_test
from learn.vicinal import generate_vicinal

data_path = "./data"
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

def retrain_order(order:int, vicinal_graph_paths: List[str]):
    def _get_g_feature(g_path):
        dataset = Dataset(g_path)
        feature = np.load(os.path.join(dataset.order_path, f"{order}_order_feature.npy"))
        X = feature[:,3:]
        y = feature[:,0]
        return X,y
    # all = p_umap(_get_g_feature, vicinal_graph_paths)
    all = []
    for vgp in vicinal_graph_paths:
        all.append(_get_g_feature(vgp))
    train_features = np.vstack([i[0] for i in all])
    train_labels = np.hstack([i[1] for i in all])
    positive_instance = [(train_labels[i], train_features[i]) for i in range(train_features.shape[0]) if train_labels[i] == 1]
    negative_instance = [(train_labels[i], train_features[i]) for i in range(train_features.shape[0]) if train_labels[i] == 0]
    random.shuffle(negative_instance)
    trimmed_negative_instance = negative_instance[:len(positive_instance)]
    train_instances = positive_instance + trimmed_negative_instance
    random.shuffle(train_instances)
    train_labels = [i[0] for i in train_instances]
    train_features = [i[1] for i in train_instances]
    fine_tune_model = ESPIModel.fine_tune(order, train_features, train_labels)
    return fine_tune_model

def prepare_next_order_sf(order:int, vicinal_graph_paths: List[str], model_path: str):
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    if order == 1: thres = 0.05
    else: raise NotImplementedError()
    for vg_path in vicinal_graph_paths:
        dataset = Dataset(vg_path)
        feature: np.ndarray = np.load(os.path.join(dataset.order_path, f"{order}_order_feature.npy"))
        X = ESPIModel.feature_augmentation(order, feature[:,3:])
        _ = feature[:,0].astype(int)
        pair = feature[:,1:3].astype(int)

        adjacency = [(pair[i], pred_score) for i, pred_score in np.ndenumerate(model.predict_proba(X)[:, 1]) if pred_score >= thres]
        curr_skl = {i:set() for i in range(dataset.D.shape[0])}
        curr_pred = {}
        for instance in adjacency:
            adj, pred_score = instance
            curr_skl[adj[0]].add(adj[1])
            curr_skl[adj[1]].add(adj[0])
            curr_pred[(adj[0], adj[1])] = pred_score

        with open(os.path.join(dataset.order_path, f"{order}_order_skl.pkl"), "wb") as f:
            dill.dump(curr_skl, f)
        with open(os.path.join(dataset.order_path, f"{order}_order_pred.pkl"), "wb") as f:
            dill.dump(curr_pred, f)

def prepare_next_order(order:int, vicinal_graph_paths: List[str]):
    model = xgb.XGBClassifier()
    model.load_model(f"./model/{order}_ml4s.model")
    if order == 1: thres = 0.05
    else: raise NotImplementedError()
    for vg_path in vicinal_graph_paths:
        dataset = Dataset(vg_path)
        feature: np.ndarray = np.load(os.path.join(dataset.order_path, f"{order}_order_feature.npy"))
        X = feature[:,3:]
        _ = feature[:,0].astype(int)
        pair = feature[:,1:3].astype(int)

        adjacency = [(pair[i], pred_score) for i, pred_score in np.ndenumerate(model.predict_proba(X)[:, 1]) if pred_score >= thres]
        curr_skl = {i:set() for i in range(dataset.D.shape[0])}
        curr_pred = {}
        for instance in adjacency:
            adj, pred_score = instance
            curr_skl[adj[0]].add(adj[1])
            curr_skl[adj[1]].add(adj[0])
            curr_pred[(adj[0], adj[1])] = pred_score

        with open(os.path.join(dataset.order_path, f"{order}_order_skl.pkl"), "wb") as f:
            dill.dump(curr_skl, f)
        with open(os.path.join(dataset.order_path, f"{order}_order_pred.pkl"), "wb") as f:
            dill.dump(curr_pred, f)

def retrain_sf(g_path:str):
    vicinal_path = os.path.join(g_path, "vicinal")
    os.makedirs(vicinal_path, exist_ok=True)
    # generate features for 1 order model
    # generate_vicinal(g_path)
    vicinal_graph_paths = [os.path.join(vicinal_path, dir_name) for dir_name in os.listdir(vicinal_path)]    
    # ml4s_generate_vicinal_order_feature(g_path, 1)
    two_order_model = retrain_order(1, vicinal_graph_paths)
    two_order_model.save_model(os.path.join(g_path,"vicinal_1_order.model"))
    # # generate features for 2 order model
    prepare_next_order_sf(1, vicinal_graph_paths, os.path.join(g_path,"vicinal_1_order.model"))
    ml4s_generate_vicinal_order_feature(g_path, 2)
    # retrain
    two_order_model = retrain_order(2, vicinal_graph_paths)
    two_order_model.save_model(os.path.join(g_path,"vicinal_2_order.model"))


def retrain(g_path:str):
    start = time.time()
    vicinal_path = os.path.join(g_path, "vicinal")
    os.makedirs(vicinal_path, exist_ok=True)
    # generate features for 1 order model
    # generate_vicinal(g_path)
    vicinal_graph_paths = [os.path.join(vicinal_path, dir_name) for dir_name in os.listdir(vicinal_path)]    
    ml4s_generate_vicinal_order_feature(g_path, 1)
    # # generate features for 2 order model
    prepare_next_order(1, vicinal_graph_paths)
    ml4s_generate_vicinal_order_feature(g_path, 2)
    # retrain
    two_order_model = retrain_order(2, vicinal_graph_paths)
    two_order_model.save_model(os.path.join(g_path,"vicinal_2_order.model"))
    consumption = time.time() - start
    np.savetxt(os.path.join(g_path,"vicinal_time.txt"), np.array([consumption]))

def end2end_test(g_path: str, order:int, thres: float):
    dataset = Dataset(g_path)
    feature: np.ndarray = np.load(os.path.join(dataset.order_path, f"{order}_order_feature.npy"))
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(g_path,f"vicinal_{order}_order.model"))
    X = ESPIModel.feature_augmentation(order, feature[:,3:])
    _ = feature[:,0].astype(int)
    pair = feature[:,1:3].astype(int)
    
    adjacency = [(pair[i], pred_score) for i, pred_score in np.ndenumerate(clf.predict_proba(X)[:, 1]) if pred_score >= thres]

    curr_skl = {i:set() for i in range(dataset.D.shape[0])}
    curr_pred = {}
    nodes = []
    for i in range(dataset.X.shape[1]):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)
    est_skl = GeneralGraph(nodes)
    for instance in adjacency:
        adj, pred_score = instance
        est_skl.add_edge(Edges().undirected_edge(nodes[adj[0]], nodes[adj[1]]))
        curr_skl[adj[0]].add(adj[1])
        curr_skl[adj[1]].add(adj[0])
        curr_pred[(adj[0], adj[1])] = pred_score

    with open(os.path.join(g_path, "cpag.pkl"), "rb") as f:
        cpag : GeneralGraph= dill.load(f)

    return compare_skl(est_skl.get_graph_edges(), cpag.get_graph_edges())

def end2end_espi_sachs():    
    dataset = Dataset("data/sachs", True)
    feature: np.ndarray = np.load(os.path.join(dataset.order_path, f"{2}_order_feature.npy"))
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join("data/sachs", f"vicinal_{2}_order.model"))
    X = ESPIModel.feature_augmentation(2, feature[:,3:])
    _ = feature[:,0].astype(int)
    pair = feature[:,1:3].astype(int)
    
    adjacency = [(pair[i], pred_score) for i, pred_score in np.ndenumerate(clf.predict_proba(X)[:, 1])]

    S = np.zeros_like(dataset.D)
    for pair in adjacency:
        (i,j), score = pair
        S[i,j] = score
        S[j,i] = score
    return S

def end2end_espi(g_path: str):    
    dataset = Dataset(g_path)
    feature: np.ndarray = np.load(os.path.join(dataset.order_path, f"{2}_order_feature.npy"))
    clf = xgb.XGBClassifier()
    clf.load_model(os.path.join(g_path,f"vicinal_{2}_order.model"))
    X = ESPIModel.feature_augmentation(2, feature[:,3:])
    _ = feature[:,0].astype(int)
    pair = feature[:,1:3].astype(int)
    
    adjacency = [(pair[i], pred_score) for i, pred_score in np.ndenumerate(clf.predict_proba(X)[:, 1])]

    S = np.zeros_like(dataset.D)
    for pair in adjacency:
        (i,j), score = pair
        S[i,j] = score
        S[j,i] = score
    return S


def main():
    if sys.argv[1] == "finetune":
        retrain(sys.argv[2])

if __name__ == "__main__":
    # retrain("data/sachs")

    p_umap(retrain_sf, [f"data/test-sf/large_{i}" for i in range(10)])
    
    # # generate_vicinal(g_p, 10)
    # # retrain(g_p)
    # g_p = f"data/test/large_0"
    # for thres in np.linspace(0.1, 0.99, num=40):
    #     print(thres)
    #     print(end2end_test(g_p, 2, thres))
    #     print(end2end_order_test(g_p, 2, thres, False))