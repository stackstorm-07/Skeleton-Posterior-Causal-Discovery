import numpy as np
import os, sys, dill
from p_tqdm import p_umap
import xgboost as xgb
from sklearn import metrics
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edges import Edges
from causallearn.utils.cit import fisherz


sys.path.append("./")
from learn.gen_feature import Dataset
from utils.fci import post_rule0_orient, fci
from utils.graph import compare_pag
from baselines.fci import end2end_fci_test
from baselines.soft_fci_dcd import end2end_dcd_test, soft_dcd_test

data_path = "./data"
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

model_path = "./model/xbg.model"

def train():

    def _get_g_feature(g_path):
        dataset = Dataset(g_path)
        feature = np.load(dataset.ml4c_featurePath)
        X = feature[:,4:]
        y = feature[:,0]
        return X,y

    graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path)]
    all = p_umap(_get_g_feature, graph_paths)
    train_features = np.vstack([i[0] for i in all])
    print(train_features.shape)
    train_labels = np.hstack([i[1] for i in all])
    clf = xgb.XGBClassifier()
    clf.fit(train_features, train_labels)
    clf.save_model(model_path)

def test():

    def _get_g_feature(g_path):
        dataset = Dataset(g_path)
        feature = np.load(dataset.ml4c_featurePath)
        X = feature[:,4:]
        y = feature[:,0]
        return X,y

    graph_paths = [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path)]
    all = p_umap(_get_g_feature, graph_paths)
    test_features = np.vstack([i[0] for i in all])
    test_labels = np.hstack([i[1] for i in all])
    clf = xgb.XGBClassifier(use_label_encoder=False)
    clf.load_model(model_path)

    y_pred = clf.predict(test_features)
    print("roc_auc_score", metrics.roc_auc_score(test_labels, y_pred))
    print("precision_score", metrics.precision_score(test_labels, y_pred))
    print("recall_score", metrics.recall_score(test_labels, y_pred))
    print("f1_score", metrics.f1_score(test_labels, y_pred))
    print("accuracy_score", metrics.accuracy_score(test_labels, y_pred))

def end2end_ml4mag_test(g_path: str, thres: float=0.5):
    dataset = Dataset(g_path)
    feature: np.ndarray = np.load(dataset.ml4c_featurePath)
    X = feature[:,4:]
    _ = feature[:,0].astype(int)
    tforks = feature[:,1:4].astype(int)

    clf = xgb.XGBClassifier(use_label_encoder=False)
    clf.load_model(model_path)
    vstrucs = [tforks[i] for i, pred_score in np.ndenumerate(clf.predict_proba(X)[:, 1]) if pred_score >= thres]
    nodes = []
    for i in range(dataset.X.shape[1]):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)
    skeleton = GeneralGraph(nodes)
    for (i,j), is_adj in np.ndenumerate(dataset.D):
        if is_adj == 1 and not skeleton.is_adjacent_to(nodes[i], nodes[j]):
            skeleton.add_edge(Edges().undirected_edge(nodes[i], nodes[j]))
    for (i,j), is_adj in np.ndenumerate(dataset.B):
        if is_adj == 1 and not skeleton.is_adjacent_to(nodes[i], nodes[j]):
            skeleton.add_edge(Edges().undirected_edge(nodes[i], nodes[j]))
    g, _ = post_rule0_orient(skeleton, vstrucs, dataset.X, independence_test_method=fisherz)

    with open(os.path.join(g_path, "cpag.pkl"), "rb") as f:
        cpag : GeneralGraph= dill.load(f) 
    
    return compare_pag(g.get_graph_edges(), cpag.get_graph_edges())



if __name__ == "__main__":
    # train()
    # test()
    g_p = "data/test/small_0"
    # for thres in np.linspace(0.1, 0.9, num=20):
    #     print(thres, end2end_ml4mag_test(g_p,thres))
    # print(end2end_fci_test(g_p))
    print(soft_dcd_test(g_p))
    print(end2end_dcd_test(g_p, False))
    # print(end2end_dcd_test(g_p, False))