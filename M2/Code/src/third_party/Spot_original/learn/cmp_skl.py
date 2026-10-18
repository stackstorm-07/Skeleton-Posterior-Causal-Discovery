
from functools import partial
import numpy as np
import os, sys, dill, random, tqdm, copy
from sklearn import metrics
sys.path.append("./")
from learn.fine_tune import end2end_espi as espi_ft
from learn.gen_feature import Dataset
from learn.espi import end2end_espi_pretrain as espi_pt

def compute_aucroc(g_path, S_mean):
    dataset = Dataset(g_path)
    labels = []
    preds = []
    for (Vi, Vj), pred in np.ndenumerate(S_mean):
        if Vi < Vj:
            preds.append(pred)
            if dataset.D[Vi,Vj] == 1 or dataset.D[Vj,Vi] == 1 or dataset.B[Vi,Vj] == 1:
                labels.append(1)
            else:
                labels.append(0)
    return metrics.roc_auc_score(labels, preds)

def compute_aucprc(g_path, S_mean):
    dataset = Dataset(g_path)
    labels = []
    preds = []
    for (Vi, Vj), pred in np.ndenumerate(S_mean):
        if Vi < Vj:
            preds.append(pred)
            if dataset.D[Vi,Vj] == 1 or dataset.D[Vj,Vi] == 1 or dataset.B[Vi,Vj] == 1:
                labels.append(1)
            else:
                labels.append(0)
    return metrics.average_precision_score(labels, preds)

def compute_kl(g_path, S_mean, adjust=False):
    dataset = Dataset(g_path)
    labels = []
    preds = []
    for (Vi, Vj), pred in np.ndenumerate(S_mean):
        if Vi < Vj:
            preds.append(pred)
            if dataset.D[Vi,Vj] == 1 or dataset.D[Vj,Vi] == 1 or dataset.B[Vi,Vj] == 1:
                labels.append(1)
            else:
                labels.append(0)
    preds = np.array(preds)
    labels = np.array(labels)
    if adjust:
        normalizing_ratio = np.sum(labels)/np.sum(preds)
        normalized_preds = preds * normalizing_ratio
        normalized_preds[normalized_preds > 1] = 1 - 1e-10
        normalized_preds[normalized_preds == 0] = 1e-10
        
        return metrics.log_loss(labels, normalized_preds)
    else:
        return metrics.log_loss(labels, preds)

if __name__ == "__main__":
    algs = ["espi", "nbfci", "avici", "nbrfci"]
    algs = ["espi", "espi+ft"]
    metrics_list = {"auroc": compute_aucroc, "auprc": compute_aucprc, "kl": compute_kl}
    result = {m:{alg:[] for alg in algs} for m in metrics_list}

    for i in range(10):
        try:
            skl = {
                "espi": espi_pt(f"data/test-sf-pt/large_{i}"),
                "espi+ft": espi_ft(f"data/test-sf/large_{i}"),
            }

            for metric in metrics_list:
                for alg in skl:
                    result[metric][alg].append(metrics_list[metric](f"data/test-sf/large_{i}", skl[alg]))
        except:
            pass
    print(result)
    print("\t", "\t".join(metrics_list))
    for alg in skl:
        print(alg, "\t".join(map(str,map(partial(round, ndigits=4) ,map(np.mean, [result[m][alg] for m in result])))))
