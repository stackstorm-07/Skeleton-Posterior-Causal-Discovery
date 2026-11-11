import numpy as np
import os, sys, dill, random
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
from utils.graph import compare_skl

data_path = "./data"
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

model_path = "./model/ml4s.model"

def oneshot_train():

    def _get_g_feature(g_path):
        dataset = Dataset(g_path)
        feature = np.load(dataset.ml4s_featurePath)
        X = feature[:,3:]
        y = feature[:,0]
        return X,y

    graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path)]
    all = p_umap(_get_g_feature, graph_paths)
    train_features = np.vstack([i[0] for i in all])
    train_labels = np.hstack([i[1] for i in all])
    positive_instance = [(train_labels[i], train_features[i]) for i in range(train_features.shape[0]) if train_labels[i] == 1]
    negative_instance = [(train_labels[i], train_features[i]) for i in range(train_features.shape[0]) if train_labels[i] == 0]
    print(positive_instance[:5])
    print(negative_instance[:5])
    print(len(positive_instance), len(negative_instance))
    random.shuffle(negative_instance)
    trimmed_negative_instance = negative_instance[:len(positive_instance)]
    train_instances = positive_instance + trimmed_negative_instance
    random.shuffle(train_instances)
    train_labels = [i[0] for i in train_instances]
    train_features = [i[1] for i in train_instances]
    clf = xgb.XGBClassifier()
    clf.fit(train_features, train_labels)
    clf.save_model(model_path)

def order_train(order:int):

    def _get_g_feature(g_path):
        dataset = Dataset(g_path)
        feature = np.load(os.path.join(dataset.order_path, f"{order}_order_feature.npy"))
        X = feature[:,3:]
        y = feature[:,0]
        return X,y

    graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path)]
    all = p_umap(_get_g_feature, graph_paths)
    train_features = np.vstack([i[0] for i in all])
    train_labels = np.hstack([i[1] for i in all])
    positive_instance = [(train_labels[i], train_features[i]) for i in range(train_features.shape[0]) if train_labels[i] == 1]
    negative_instance = [(train_labels[i], train_features[i]) for i in range(train_features.shape[0]) if train_labels[i] == 0]
    print(positive_instance[:5])
    print(negative_instance[:5])
    print(len(positive_instance), len(negative_instance))
    random.shuffle(negative_instance)
    trimmed_negative_instance = negative_instance[:len(positive_instance)]
    train_instances = positive_instance + trimmed_negative_instance
    random.shuffle(train_instances)
    train_labels = [i[0] for i in train_instances]
    train_features = [i[1] for i in train_instances]
    clf = xgb.XGBClassifier()
    clf.fit(train_features, train_labels)
    clf.save_model(f"./model/{order}_ml4s.model")

def oneshot_test():

    def _get_g_feature(g_path):
        dataset = Dataset(g_path)
        feature = np.load(dataset.ml4s_featurePath)
        X = feature[:,3:]
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

def order_test(order:int):
    def _get_g_feature(g_path):
        dataset = Dataset(g_path)
        feature = np.load(os.path.join(dataset.order_path, f"{order}_order_feature.npy"))
        X = feature[:,3:]
        y = feature[:,0]
        return X,y

    graph_paths = [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path)]
    all = p_umap(_get_g_feature, graph_paths)
    test_features = np.vstack([i[0] for i in all])
    test_labels = np.hstack([i[1] for i in all])
    clf = xgb.XGBClassifier(use_label_encoder=False)
    clf.load_model(f"./model/{order}_ml4s.model")

    y_pred = clf.predict(test_features)
    print("roc_auc_score", metrics.roc_auc_score(test_labels, y_pred))
    print("precision_score", metrics.precision_score(test_labels, y_pred))
    print("recall_score", metrics.recall_score(test_labels, y_pred))
    print("f1_score", metrics.f1_score(test_labels, y_pred))
    print("accuracy_score", metrics.accuracy_score(test_labels, y_pred))

def end2end_ml4s_test(g_path: str, thres: float=0.5):
    dataset = Dataset(g_path)
    feature: np.ndarray = np.load(dataset.ml4s_featurePath)
    X = feature[:,3:]
    _ = feature[:,0].astype(int)
    pair = feature[:,1:3].astype(int)

    clf = xgb.XGBClassifier(use_label_encoder=False)
    clf.load_model(model_path)
    adjacency = [pair[i] for i, pred_score in np.ndenumerate(clf.predict_proba(X)[:, 1]) if pred_score >= thres]
    nodes = []
    for i in range(dataset.X.shape[1]):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)
    skeleton = GeneralGraph(nodes)
    for adj in adjacency:
        skeleton.add_edge(Edges().undirected_edge(nodes[adj[0]], nodes[adj[1]]))

    with open(os.path.join(g_path, "cpag.pkl"), "rb") as f:
        cpag : GeneralGraph= dill.load(f)

    return compare_skl(skeleton.get_graph_edges(), cpag.get_graph_edges())

def end2end_order_ml4s_test(g_path: str, order:int, thres: float=0.5, save_file:bool=True):
    dataset = Dataset(g_path)
    feature: np.ndarray = np.load(os.path.join(dataset.order_path, f"{order}_order_feature.npy"))
    X = feature[:,3:]
    _ = feature[:,0].astype(int)
    pair = feature[:,1:3].astype(int)

    clf = xgb.XGBClassifier(use_label_encoder=False)
    clf.load_model(f"./model/{order}_ml4s.model")
    adjacency = [(pair[i], pred_score) for i, pred_score in np.ndenumerate(clf.predict_proba(X)[:, 1]) if pred_score >= thres]

    curr_skl = {i:set() for i in range(dataset.D.shape[0])}
    curr_pred = {}
    nodes = []
    for i in range(dataset.X.shape[1]):
        node = GraphNode(f"X{i + 1}")
        node.add_attribute("id", i)
        nodes.append(node)
    skeleton = GeneralGraph(nodes)
    for instance in adjacency:
        adj, pred_score = instance
        skeleton.add_edge(Edges().undirected_edge(nodes[adj[0]], nodes[adj[1]]))
        curr_skl[adj[0]].add(adj[1])
        curr_skl[adj[1]].add(adj[0])
        curr_pred[(adj[0], adj[1])] = pred_score

    with open(os.path.join(g_path, "cpag.pkl"), "rb") as f:
        cpag : GeneralGraph= dill.load(f)
    if save_file:
        with open(os.path.join(dataset.order_path, f"{order}_order_skl.pkl"), "wb") as f:
            dill.dump(curr_skl, f)
        with open(os.path.join(dataset.order_path, f"{order}_order_pred.pkl"), "wb") as f:
            dill.dump(curr_pred, f)

    return compare_skl(skeleton.get_graph_edges(), cpag.get_graph_edges())

def prepare_next_order(order:int, thres: float=0.5, verbose:bool=False):
    graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path)] + [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path)]
    for g_path in graph_paths:
        if verbose: print(g_path, end2end_order_ml4s_test(g_path,order,thres))
        else: end2end_order_ml4s_test(g_path,order,thres)


def ml4s_soft(g_path: str):
    dataset = Dataset(g_path)
    feature: np.ndarray = np.load(dataset.ml4s_featurePath)
    X = feature[:,3:]
    _ = feature[:,0].astype(int)
    pair = feature[:,1:3].astype(int)

    clf = xgb.XGBClassifier(use_label_encoder=False)
    clf.load_model(model_path)
    adjacency = [(pair[i], pred_score) for i, pred_score in np.ndenumerate(clf.predict_proba(X)[:, 1])]
    S = np.zeros_like(dataset.D)
    for pair in adjacency:
        (i,j), score = pair
        S[i,j] = score
        S[j,i] = score
    return S

def order_ml4s_soft(g_path: str, order:int):
    dataset = Dataset(g_path)
    feature: np.ndarray = np.load(os.path.join(dataset.order_path, f"{order}_order_feature.npy"))
    X = feature[:,3:]
    _ = feature[:,0].astype(int)
    pair = feature[:,1:3].astype(int)

    clf = xgb.XGBClassifier(use_label_encoder=False)
    clf.load_model(f"./model/{order}_ml4s.model")
    adjacency = [(pair[i], pred_score) for i, pred_score in np.ndenumerate(clf.predict_proba(X)[:, 1])]
    S = np.zeros_like(dataset.D)
    for pair in adjacency:
        (i,j), score = pair
        S[i,j] = score
        S[j,i] = score
    return S

def ml4s_hard(g_path: str, thres:float=0.6):
    dataset = Dataset(g_path)
    feature: np.ndarray = np.load(dataset.ml4s_featurePath)
    X = feature[:,3:]
    _ = feature[:,0].astype(int)
    pair = feature[:,1:3].astype(int)

    clf = xgb.XGBClassifier(use_label_encoder=False)
    clf.load_model(model_path)
    adjacency = [(pair[i], pred_score) for i, pred_score in np.ndenumerate(clf.predict_proba(X)[:, 1]) if pred_score >= thres]
    S = np.zeros_like(dataset.D)
    for pair in adjacency:
        (i,j), _ = pair
        S[i,j] = 1
        S[j,i] = 1
    return S

if __name__ == "__main__":
    order_train(2)
    # order_test(2)
    # thres = 0.1
    # prepare_next_order(1, thres,True)
    g_p = "data/test/medium_0"
    for thres in np.linspace(0.1, 0.8, num=100):
        print(thres, end2end_order_ml4s_test(g_p,2,thres,False))
    # for thres in np.linspace(0.3, 0.99, num=50):
    #     print(thres, end2end_ml4s_test(g_p,thres))
    # print(end2end_fci_test(g_p))
    # print(soft_dcd_test(g_p))
    # print(end2end_dcd_test(g_p, False))
    # print(end2end_dcd_test(g_p, False))