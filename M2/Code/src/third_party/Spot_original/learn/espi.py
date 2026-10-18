import numpy as np
import os, sys, dill, random
from p_tqdm import p_umap
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edges import Edges
import xgboost as xgb

sys.path.append("./")
from learn.gen_feature import Dataset
from utils.fci import post_rule0_orient, fci
from utils.graph import compare_skl

data_path = "./data"
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

class ESPIModel:

    @staticmethod
    def train_model(training_feature, training_label):
        model = xgb.XGBClassifier()
        model.fit(training_feature, training_label)
        return model
    
    # @staticmethod
    # def fine_tune(order:int, training_feature, training_label):
    #     pretrain_feature = np.load(f"data/train/train_features_{order}.npy")
    #     pretrain_label = np.load(f"data/train/train_labels_{order}.npy")
    #     features = np.vstack([pretrain_feature, training_feature])
    #     labels = np.hstack([pretrain_label, training_label])
    #     print(pretrain_feature.shape, features.shape)
    #     model = xgb.XGBClassifier()
    #     model.fit(features, labels)
    #     return model

    @staticmethod
    def fine_tune(order:int, training_feature, training_label):
        augmented_train_feature = ESPIModel.feature_augmentation(order, training_feature)
        tar_model = xgb.XGBClassifier()
        tar_model.fit(augmented_train_feature, training_label)
        return tar_model
    
    @staticmethod
    def fine_tune_predict_proba(order:int, tar_model: xgb.XGBClassifier, features):
        return tar_model.predict_proba(ESPIModel.feature_augmentation(order, features))
    
    @staticmethod
    def feature_augmentation(order:int, features):
        src_model = xgb.XGBClassifier()
        src_model.load_model(f"./model/{order}_ml4s.model")
        src_pred_leaf = src_model.get_booster().predict(xgb.DMatrix(features), pred_leaf=True)
        src_pred_margin = src_model.get_booster().predict(xgb.DMatrix(features), output_margin=True)
        return np.array([np.hstack([features[i], src_pred_leaf[i], src_pred_margin[i]]) for i in range(len(features))])

def order_train(order:int):

    def _get_g_feature(g_path):
        dataset = Dataset(g_path)
        feature = np.load(os.path.join(dataset.order_path, f"{order}_order_feature.npy"))
        X = feature[:,3:]
        y = feature[:,0]
        return X,y

    graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, dir_name))] 
    all = p_umap(_get_g_feature, graph_paths)

    train_features = np.vstack([i[0] for i in all])
    train_labels = np.hstack([i[1] for i in all])
    positive_instance = [(train_labels[i], train_features[i]) for i in range(train_features.shape[0]) if train_labels[i] == 1]
    negative_instance = [(train_labels[i], train_features[i]) for i in range(train_features.shape[0]) if train_labels[i] == 0]
    random.shuffle(negative_instance)
    trimmed_negative_instance = negative_instance[:len(positive_instance)]
    train_instances = positive_instance + trimmed_negative_instance
    random.shuffle(train_instances)

    train_labels = np.array([i[0] for i in train_instances])
    train_features = np.array([i[1] for i in train_instances])
    src_model = ESPIModel.train_model(train_features, train_labels)
    src_model.save_model(f"./model/{order}_espi.model")
    np.save(f"data/train/train_features_{order}.npy", train_features)
    np.save(f"data/train/train_labels_{order}.npy", train_labels)

def end2end_order_test(g_path: str, order:int, thres: float=0.5, save_file:bool=True, src_model=None):
    dataset = Dataset(g_path)
    feature: np.ndarray = np.load(os.path.join(dataset.order_path, f"{order}_order_feature.npy"))
    X = feature[:,3:]
    _ = feature[:,0].astype(int)
    pair = feature[:,1:3].astype(int)
    if src_model == None:
        src_model = xgb.XGBClassifier()
        src_model.load_model(f"./model/{order}_espi.model")

    adjacency = [(pair[i], pred_score) for i, pred_score in np.ndenumerate(src_model.predict_proba(X)[:, 1]) if pred_score >= thres]
        
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

def end2end_espi_pretrain(g_path: str):
    dataset = Dataset(g_path)
    feature: np.ndarray = np.load(os.path.join(dataset.order_path, f"2_order_feature.npy"))
    X = feature[:,3:]
    _ = feature[:,0].astype(int)
    pair = feature[:,1:3].astype(int)
    model = xgb.XGBClassifier()
    model.load_model(f"./model/2_espi.model")
    adjacency = [(pair[i], pred_score) for i, pred_score in np.ndenumerate(model.predict_proba(X)[:, 1])]
    S = np.zeros_like(dataset.D)
    for pair in adjacency:
        (i,j), score = pair
        S[i,j] = score
        S[j,i] = score
    return S

def prepare_next_order(order:int, thres: float=0.5, verbose:bool=False):
    graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, dir_name))] 
    graph_paths += [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, dir_name))]
    src_model = xgb.XGBClassifier()
    src_model.load_model(f"./model/{order}_espi.model")
    for g_path in graph_paths:
        if verbose: print(g_path, end2end_order_test(g_path,order,thres,src_model))
        else: end2end_order_test(g_path,order,thres,src_model)

def prepare_next_order_sf(order:int, thres: float=0.5, verbose:bool=False):
    test_path = os.path.join(data_path, "test-sf-pt")
    graph_paths = [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, dir_name))]
    src_model = xgb.XGBClassifier()
    src_model.load_model(f"./model/{order}_espi.model")
    for g_path in graph_paths:
        if verbose: print(g_path, end2end_order_test(g_path,order,thres,src_model))
        else: end2end_order_test(g_path,order,thres,src_model)

def prepare_next_order_sachs(order:int, thres: float=0.5):
    dataset = Dataset("data/sachs",True)
    feature: np.ndarray = np.load(os.path.join(dataset.order_path, f"{order}_order_feature.npy"))
    X = feature[:,3:]
    _ = feature[:,0].astype(int)
    pair = feature[:,1:3].astype(int)
    src_model = xgb.XGBClassifier()
    src_model.load_model(f"./model/{order}_espi.model")

    adjacency = [(pair[i], pred_score) for i, pred_score in np.ndenumerate(src_model.predict_proba(X)[:, 1]) if pred_score >= thres]
        
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

    with open(os.path.join(dataset.order_path, f"{order}_order_skl.pkl"), "wb") as f:
        dill.dump(curr_skl, f)
    with open(os.path.join(dataset.order_path, f"{order}_order_pred.pkl"), "wb") as f:
        dill.dump(curr_pred, f)

if __name__ == "__main__":
    prepare_next_order_sf(1, 0.05)
    # order_train(2)
    # order_test(2)
    # thres = 0.05
    # prepare_next_order_sachs(1, thres)
    # g_p = "data/test/large_0"
    # for thres in np.linspace(0.05, 0.99, num=50):
    #     print(thres, end2end_order_test(g_p,2,thres,False))
    # for thres in np.linspace(0.3, 0.99, num=50):
    #     print(thres, end2end_ml4s_test(g_p,thres))
    # print(end2end_fci_test(g_p))
    # print(soft_dcd_test(g_p))
    # print(end2end_dcd_test(g_p, False))
    # print(end2end_dcd_test(g_p, False))