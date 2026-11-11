from functools import partial
import numpy as np
from math import log, sqrt
from scipy.stats import norm
import os, random, copy, sys, warnings, dill
from typing import FrozenSet, Set, List, Tuple, Dict
from p_tqdm import p_umap
from itertools import combinations, product
from causallearn.utils.cit import CIT
sys.path.append("./")
from utils.kernel_embedding import percentile_embedding, meanstdmaxmin, Kernel_Embedding

data_path = "./data"
train_path = os.path.join(data_path, "train")
test_path = os.path.join(data_path, "test")

class Dataset:
    def __init__(self, g_path: str, is_sachs=False) -> None:
        if is_sachs:
            self.X:np.ndarray = np.load(os.path.join(g_path, "X.npy"))
            self.D = np.zeros((self.X.shape[1],self.X.shape[1]))
            self.B = np.zeros_like(self.D)
        else:
            self.D:np.ndarray = np.loadtxt(os.path.join(g_path, "D.txt"))
            self.B:np.ndarray = np.loadtxt(os.path.join(g_path, "B.txt"))
            self.X:np.ndarray = np.load(os.path.join(g_path, "X.npy"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.vstrucs = set(map(tuple, np.loadtxt(os.path.join(g_path, "vstrucs.txt"), dtype=int).reshape((-1, 3))))
                self.tforks = set(map(tuple, np.loadtxt(os.path.join(g_path, "tforks.txt"), dtype=int).reshape((-1, 3))))
            except:
                self.vstrucs = set()
                self.tforks = set()
        self.CIT = CIT(self.X)
        self.sampleSize = self.X.shape[0]
        self.correlation_matrix = np.corrcoef(self.X.T)
        self.cache = {}
        self.skeleton = {i:self.get_pc(i) for i in range(self.D.shape[0])}
        self.mb = {}
        self.mb:Dict[int, Set[int]]

        self.maxConSizeForForwardSearch = 3
        self.maxConSizeForBackSearch = 2
        self.confidenceLevel = 0.99
        self.maxCountOfSepsets = 50
        self.maxPairsCount = 100
        self.maxEstimandCondCount = 1000
        self.maxMBCombCount = 1000

        self.alpha1 = 0.15
        self.alpha2 = 0.15

        self.ml4c_featurePath = os.path.join(g_path, "ml4cfeature.npy")
        self.ml4s_featurePath = os.path.join(g_path, "ml4sfeature.npy")
        self.order_path = os.path.join(g_path, "order")
        os.makedirs(self.order_path, exist_ok=True)

    def cit(self, i: int, j: int, z: Tuple[int]):
        cache_key = (i, j, frozenset(z)) if i< j else (j, i, frozenset(z))
        if cache_key in self.cache: return self.cache[cache_key][0]
        return self.cache.setdefault(cache_key, self.fisherz(i, j, z))[0]
    
    def cit_with_sev(self, i: int, j: int, z: List[int]):
        cache_key = (i, j, frozenset(z)) if i< j else (j, i, frozenset(z))
        if cache_key in self.cache: return self.cache[cache_key]
        return self.cache.setdefault(cache_key, self.fisherz(i, j, z))
    
    def fisherz(self, X, Y, condition_set):
        """
        Perform an independence test using Fisher-Z's test

        Parameters
        ----------
        data : data matrices
        X, Y and condition_set : column indices of data

        Returns
        -------
        p : the p-value of the test
        sev: severity
        """
        var = list((X, Y) + condition_set)
        sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
        inv = np.linalg.inv(sub_corr_matrix)
        r = -inv[0, 1] / sqrt(inv[0, 0] * inv[1, 1])
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sampleSize - len(condition_set) - 3) * abs(Z)
        sev = abs(X)
        p = 2 * (1 - norm.cdf(sev))
        return p, sev
    
    def get_pc(self, i: int) -> FrozenSet[int]:
        neighbors = list(np.argwhere(self.D[:,i] != 0)) + list(np.argwhere(self.D[i,:] != 0)) + list(np.argwhere(self.B[:,i] != 0)) + list(np.argwhere(self.B[i,:] != 0))
        return frozenset(np.array(neighbors).flatten())
    
    def get_sepset_from_skeleton(self, i: int, k: int):
        i_PC = self.skeleton[i] - {k}
        k_PC = self.skeleton[k] - {i}
        search_from_i = set.union(*[set(combinations(i_PC, condsize)) for condsize in
                                      range(1 + min(self.maxConSizeForBackSearch, len(i_PC)))])
        search_from_k = set.union(*[set(combinations(k_PC, condsize)) for condsize in
                                      range(1 + min(self.maxConSizeForBackSearch, len(k_PC)))])
        search_from = list(search_from_i.union(search_from_k))
        random.shuffle(search_from)

        valid_sepsets = set()
        maximum_invalid_sepset = (None, -1)
        # though not pval > 0.01, save the most nearest (subset, pval), e.g. pval=0.009
        for subset in search_from:
            pValue = self.cit(i, k, list(subset))
            if pValue > 1. - self.confidenceLevel:
                valid_sepsets.add(subset)
            elif pValue > maximum_invalid_sepset[1]:
                maximum_invalid_sepset = (subset, pValue)
            if len(valid_sepsets) == self.maxCountOfSepsets:
                return valid_sepsets
        return valid_sepsets if valid_sepsets else {maximum_invalid_sepset[0]}
    
    def get_markov_blanket(self, i: int) -> Set[int]:
        if i in self.mb: return self.mb[i]
        mb = []
        # grow
        while True:
            no_change = True
            for j in range(self.D.shape[0]):
                if i == j: continue
                if j in mb: continue
                # sub_mb_set = set.union(*[set(combinations(mb, condsize)) for condsize in
                #                       range(1 + min(self.maxConSizeForForwardSearch, len(mb)))])
                # for sub_mb in sub_mb_set:
                #     if self.cit(i, j, tuple(sub_mb)) > self.alpha1:
                #         mb.append(j)
                #         no_change = False
                #         break
                if self.cit(i, j, tuple(mb)) > self.alpha1:
                    # print("add",i, mb, j, self.cit(i, j, tuple(mb)))
                    mb.append(j)
                    no_change = False
                    break
                if not no_change: break
            if no_change: break
        # shrink
        while True:
            no_change = True
            for j in mb:
                sub_mb_set = set.union(*[set(combinations(mb, condsize)) for condsize in
                                      range(1 + min(self.maxConSizeForBackSearch, len(mb)))])
                for sub_mb in sub_mb_set:
                    if self.cit(i, j, tuple(set(sub_mb)-{j})) <= self.alpha2:
                        # print("rm",i, mb, j, self.cit(i, j, tuple(set(sub_mb)-{j})))
                        mb.remove(j)
                        no_change = False
                        break
                # if self.cit(i, j, tuple(set(mb)-{j})) <= self.alpha1:
                #     mb.remove(j)
                #     no_change = False
                #     break
                if not no_change: break
            if no_change: break
        return self.mb.setdefault(i, set(mb))
    
    def and_rule(self):
        for i in range(self.D.shape[0]):
            for j in range(self.D.shape[0]):
                if i in self.get_markov_blanket(j) and j not in self.get_markov_blanket(i):
                    self.mb[j] = self.mb[j] - {i}
                if j in self.get_markov_blanket(i) and i not in self.get_markov_blanket(j):
                    self.mb[i] = self.mb[i] - {j}
        # print(self.mb)
    
    def ml4s_gen_pair_raw_feature(self, i:int, j:int):

        # if self.cit(i,j,tuple()) <= self.alpha1: return np.zeros(15)
        # if j not in self.get_markov_blanket(i): return np.zeros(15)
        # if i not in self.get_markov_blanket(j): return np.zeros(15)
        zero_order = self.cit_with_sev(i,j,tuple())[0]
        mb_i = self.get_markov_blanket(i) - {j}
        mb_j = self.get_markov_blanket(j) - {i}
        # for k in mb_i.union(mb_j):
        #     if self.cit(i,j,(k,)) <= self.alpha1: return np.zeros(15)
        sepset_candidate_i = set.union(*[set(combinations(mb_i, condsize)) for condsize in
                                      range(1 + min(self.maxConSizeForBackSearch, len(mb_i)))])
        sepset_candidate_j = set.union(*[set(combinations(mb_j, condsize)) for condsize in
                                      range(1 + min(self.maxConSizeForBackSearch, len(mb_j)))])
        search_from = list(sepset_candidate_i.union(sepset_candidate_j))
        # mb = self.get_markov_blanket(i) - {j} if len(self.get_markov_blanket(i)) < len(self.get_markov_blanket(j)) else self.get_markov_blanket(j) - {i}
        
        # search_from = list(set.union(*[set(combinations(mb, condsize)) for condsize in
        #                               range(1 + min(self.maxConSizeForBackSearch, len(mb)))]))
        random.shuffle(search_from)
        search_from.sort(key=lambda x:len(x))
        if len(search_from) > self.maxMBCombCount: 
            print("cutoff")
            search_from = search_from[:self.maxMBCombCount]
        sevs = {i:[] for i in range(1 + self.maxConSizeForBackSearch)}
        for candidate in search_from:
            sevs[len(candidate)].append(self.cit_with_sev(i,j,candidate)[1])
        feature = [[zero_order]]
        for i in range(1, 1 + self.maxConSizeForBackSearch):
            feature.append(np.hstack([meanstdmaxmin(sevs[i]), percentile_embedding(sevs[i])]))
        # sevs = [self.cit_with_sev(i,j,candidate)[1] for candidate in search_from]
        feature = np.hstack(feature)
        return feature
    
    def ml4s_gen_order_pair_raw_feature(self, i:int, j:int, order:int, curr_skl:Dict[int, Set[int]]):
        search_from = list(combinations((curr_skl[i] - {j}).union((curr_skl[j] - {i})), order))
        random.shuffle(search_from)
        search_from.sort(key=lambda x:len(x))
        if len(search_from) > self.maxMBCombCount: 
            # print("cutoff")
            search_from = search_from[:self.maxMBCombCount]
        pvals, sevs = [], []
        for candidate in search_from:
            pval, sev = self.cit_with_sev(i,j,candidate)
            pvals.append(pval)
            sevs.append(sev)
        feature = np.hstack([meanstdmaxmin(pvals), percentile_embedding(pvals), meanstdmaxmin(sevs), percentile_embedding(sevs)])
        return feature
    
    def ml4s_gen_all_feature(self):
        self.and_rule()
        all_features = []
        for i,j in np.ndindex(self.D.shape):
            if i>=j: continue
            feature = self.ml4s_gen_pair_raw_feature(i,j)
            label = 1 if j in self.get_pc(i) else 0
            all_features.append(np.hstack([[label], [i, j], feature]))
        all_features = np.array(all_features)
        np.save(self.ml4s_featurePath, all_features)
        return all_features
    
    def ml4s_gen_order_feature(self, order:int):
        sklPath = os.path.join(self.order_path, f"{order-1}_order_skl.pkl")
        predPath = os.path.join(self.order_path, f"{order-1}_order_pred.pkl")
        with open(sklPath,"rb") as f:
            curr_skl = dill.load(f)
        if order > 1:
            with open(predPath,"rb") as f:
                    pred_dict = dill.load(f)
        all_features = []
        
        if order == 1:
            for i,j in np.ndindex(self.D.shape):
                if i>=j: continue
                feature = self.ml4s_gen_order_pair_raw_feature(i,j,order,curr_skl)
                label = 1 if j in self.get_pc(i) else 0
                all_features.append(np.hstack([[label], [i, j], feature]))
        else:
            for adj in pred_dict:
                i,j = adj
                feature = self.ml4s_gen_order_pair_raw_feature(i,j,order,curr_skl)
                label = 1 if j in self.get_pc(i) else 0
                pred = pred_dict[adj]
                all_features.append(np.hstack([[label], [i, j], [pred],feature]))
        all_features = np.array(all_features)
        order_featurePath = os.path.join(self.order_path, f"{order}_order_feature.npy")
        np.save(order_featurePath, all_features)
        return all_features
    
    def prepare_zero_order(self):
        zero_order_skl = {i:set() for i in range(self.D.shape[0])}
        for i,j in np.ndindex(self.D.shape):
            if i>=j: continue
            pval, sev = self.cit_with_sev(i,j,tuple())
            if pval > self.alpha1: 
                zero_order_skl[i].add(j)
                zero_order_skl[j].add(i)
        zero_order_sklPath = os.path.join(self.order_path, "0_order_skl.pkl")
        with open(zero_order_sklPath,"wb") as f:
            dill.dump(zero_order_skl, f)
        return zero_order_skl
    
    def ml4c_gen_tfork_raw_feature(self, i: int, j: int, k: int):

        def _overlap(set1, set2):
            set1, set2 = set(set1), set(set2) # maybe input is list
            min_size = min(len(set1), len(set2)) # if minsize=0, return 1 because ∅∈anyset
            return 1. if min_size == 0 else len(set1.intersection(set2)) / min_size

        def _avg_overlap(set1, list_of_set2):
            if len(list_of_set2) == 0: return 0.
            return np.mean([_overlap(set1, set2) for set2 in list_of_set2])

        def _condon(estimands: List[Tuple[int]], conditions: List[Tuple[int]]):
            '''
            :param estimands (bivariable): list or set of tuples, and these tuples are all with len=2
            :param conditions: list or set of tuples, and these tuples can be various in length, e.g. 0, 1, 2, 3,...
            :return: list of tuples (flatten, 1D)
            '''
            est_cond_pairs = [(e0, e1, c) for ((e0, e1), c) in product(estimands, conditions) if e0 not in c and e1 not in c]
            if len(est_cond_pairs) > self.maxEstimandCondCount: est_cond_pairs = random.sample(est_cond_pairs, self.maxEstimandCondCount)
            return [self.cit(e0, e1, list(c)) for (e0, e1, c) in est_cond_pairs]

        i_PC_minus_j = self.skeleton[i] - {j}
        k_PC_minus_j = self.skeleton[k] - {j}
        i_pcypairs = [_ for _ in product({i}, k_PC_minus_j) if _[0] not in self.skeleton[_[1]]]
        k_pcypairs = [_ for _ in product({k}, i_PC_minus_j) if _[0] not in self.skeleton[_[1]]]
        ik_pcpairs = [_ for _ in {tuple(sorted(tp)) for tp in product(i_PC_minus_j, k_PC_minus_j)} # sort and set to remove repeat
                         if _[0] != _[1] and _[0] not in self.skeleton[_[1]]] # add this restrict: no adjacent pair

        ik_sepsets = list(self.get_sepset_from_skeleton(i, j))
        ik_sepsets_uT = [tuple(set(_).union({j})) for _ in ik_sepsets]  # has order, corresponding to XY_sepsets

        scalings = [
            len(self.skeleton[j]),
            len(self.skeleton[i]),
            len(self.skeleton[k]),
            len(ik_sepsets),
            np.average([len(s) for s in ik_sepsets])
        ]

        overlaps = [
            _overlap(self.skeleton[i], self.skeleton[k]),
            _overlap(self.skeleton[i], self.skeleton[j]),
            _overlap(self.skeleton[k], self.skeleton[j]),
            _avg_overlap({j}, ik_sepsets),
            _avg_overlap(self.skeleton[i], ik_sepsets), # how many of sepsets are from PCX?
            _avg_overlap(self.skeleton[k], ik_sepsets), # how many of sepsets are from PCY? (sum >= 1, bcs of repeats e.g. ∅)
            _avg_overlap(self.skeleton[j], ik_sepsets),
        ]

        estimands_catagories = [
            [(i, k)],
            i_pcypairs,
            k_pcypairs,
            ik_pcpairs
        ]

        conditions_categories = [
            [(j,)],
            ik_sepsets,
            ik_sepsets_uT,
            [(pcj,) for pcj in self.skeleton[j] - {i, k}],
            [tuple(set(s).union({pcj})) for pcj in self.skeleton[j] - {i, k} for s in ik_sepsets if pcj not in s]
        ]

        return scalings + overlaps + [[_condon(ests, conds) for ests in estimands_catagories] for conds in conditions_categories]

    def ml4c_generate_all_feature(self):
        all_features = []
        for tf in self.tforks:
            j, i, k = tf
            feature = self.ml4c_gen_tfork_raw_feature(i, j, k)
            all_features.append([[j, i, k], feature])
        
        ke =  Kernel_Embedding()
        processed_all_features = []
        for tf, raw_feature in all_features:
            j, i, k = tf
            scalings_overlaps = raw_feature[:12] # 12 = 5 scalings + 7 overlaps
            est_conds = raw_feature[12:]
            unitary_XY_T = list(est_conds[0][0])
            processed_fts = copy.copy(scalings_overlaps + unitary_XY_T)

            unitary_flag = True
            for condlist in est_conds:
                for estlist in condlist:
                    if unitary_flag: # jump the first unitary, no need to percentile
                        unitary_flag = False; continue
                    pvals = [_ for _ in estlist]
                    meanstdmaxmin_values = [len(pvals)] + meanstdmaxmin(pvals) # length=1+4=5
                    pval_embd = ke.get_empirical_embedding(pvals) # length=15
                    processed_fts.extend(meanstdmaxmin_values + pval_embd)

            processed_all_features.append([1 if tuple(tf) in self.vstrucs else 0] + [j, i, k] + processed_fts) # 0: label, 1-4: tfork, 4-: feature
        processed_all_features = np.array(processed_all_features)
        np.save(self.ml4c_featurePath, processed_all_features)
        return processed_all_features

def ml4c_generate_feature():
    # helper func
    def _helper(g_path):
        dataset = Dataset(g_path)
        dataset.ml4c_generate_all_feature()
    # generate feature
    graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path)] + [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path)]
    p_umap(_helper, graph_paths)

def ml4s_generate_feature():
    # helper func
    def _helper(g_path):
        dataset = Dataset(g_path)
        dataset.ml4s_gen_all_feature()
    # generate feature
    # graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path) if "medium_0" in dir_name]# + [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path)]
    graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path)] + [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path)]
    p_umap(_helper, graph_paths)

def ml4s_generate_order_feature(order:int):
    # helper func
    def _helper(g_path):
        dataset = Dataset(g_path)
        if order == 1:
            dataset.prepare_zero_order()
        dataset.ml4s_gen_order_feature(order)
    # generate feature
    # graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path) if "medium_0" in dir_name]# + [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path)]
    graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, dir_name))] 
    graph_paths += [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, dir_name))]
    p_umap(_helper, graph_paths)

def ml4s_generate_order_feature_sf(order:int):
    # helper func
    def _helper(g_path):
        dataset = Dataset(g_path)
        if order == 1:
            dataset.prepare_zero_order()
        dataset.ml4s_gen_order_feature(order)
    # generate feature
    # graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path) if "medium_0" in dir_name]# + [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path)]
    test_path = os.path.join(data_path, "test-sf")
    graph_paths = [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path) if os.path.isdir(os.path.join(test_path, dir_name))]
    p_umap(_helper, graph_paths)

def ml4s_generate_order_feature_sachs(order:int):
    # helper func
    def _helper(g_path):
        dataset = Dataset(g_path, is_sachs=True)
        if order == 1:
            dataset.prepare_zero_order()
        dataset.ml4s_gen_order_feature(order)
    # generate feature
    # graph_paths = [os.path.join(train_path, dir_name) for dir_name in os.listdir(train_path) if "medium_0" in dir_name]# + [os.path.join(test_path, dir_name) for dir_name in os.listdir(test_path)]
    _helper("data/sachs")

# helper func
def _ml4s_generate_vicinal_order_feature_helper(g_p, order):
    dataset = Dataset(g_p)
    if order == 1:
        dataset.prepare_zero_order()
    dataset.ml4s_gen_order_feature(order)

def ml4s_generate_vicinal_order_feature(g_path, order:int):
    # generate feature
    vicinal_path = os.path.join(g_path, "vicinal")
    vicinal_graph_paths = [os.path.join(vicinal_path, dir_name) for dir_name in os.listdir(vicinal_path)]
    list(map(partial(_ml4s_generate_vicinal_order_feature_helper, order=order), vicinal_graph_paths))


if __name__ == "__main__":
    # p_umap(lambda gp: Dataset(gp).prepare_zero_order(), [f"data/test-sf-pt/large_{i}" for i in range(10)])
    p_umap(lambda gp: Dataset(gp).ml4s_gen_order_feature(2), [f"data/test-sf-pt/large_{i}" for i in range(10)])