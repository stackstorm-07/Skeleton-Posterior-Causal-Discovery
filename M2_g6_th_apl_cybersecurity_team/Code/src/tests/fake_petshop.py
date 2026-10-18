"""A FAKE PetShop-format dataset for pipeline tests only (graph.csv + noissue/metrics.csv with the 3-level header).
Its numbers mean nothing; it exists so the loader -> Tetrad ground truth -> posterior -> ABIC/SPOT -> metrics -> summary
chain can be exercised end to end without the real dataset."""
import os
import numpy as np
import pandas as pd


def make_fake_petshop(root, scenario="low_traffic", n=300, seed=0, rho=0.4):
    rng = np.random.default_rng(seed)
    nodes = ["frontend", "api", "svc_a", "svc_b", "db", "queue", "cache", "ext", "lat1", "lat2"]
    calls = [("frontend", "api"), ("api", "svc_a"), ("api", "svc_b"), ("svc_a", "db"), ("svc_b", "db"), ("svc_b", "cache"),
             ("svc_a", "queue"), ("frontend", "ext"), ("svc_a", "lat1"), ("svc_b", "lat1"), ("frontend", "lat2"), ("ext", "lat2")]
    latent = {"lat1", "lat2"}                      # no metric column -> treated as latent by the loader
    A = pd.DataFrame(0, index=nodes, columns=nodes)
    for a, b in calls:
        A.loc[a, b] = 1
    order = ["db", "queue", "cache", "lat1", "lat2", "ext", "svc_a", "svc_b", "api", "frontend"]
    series = {}
    for v in order:                                # callee latency propagates to its callers
        e = rng.standard_t(5, n)
        for i in range(1, n):
            e[i] += rho * e[i - 1]
        x = np.exp(0.3 * e) + 1.0
        for a, b in calls:
            if a == v:
                x = x + 0.6 * series[b]
        series[v] = x
    obs = [v for v in nodes if v not in latent]
    cols = pd.MultiIndex.from_tuples([(v, "latency", "Average") for v in obs] + [(v, "error", "Sum") for v in obs])
    M = pd.DataFrame(np.column_stack([series[v] for v in obs] + [rng.random(n) for _ in obs]), columns=cols,
                     index=pd.date_range("2023-01-01", periods=n, freq="min"))
    p = os.path.join(root, scenario)
    os.makedirs(os.path.join(p, "noissue"), exist_ok=True)
    A.to_csv(os.path.join(p, "graph.csv"))
    M.to_csv(os.path.join(p, "noissue", "metrics.csv"))
    return root


def make_fake_issues(root, scenario="low_traffic", n=300, seed=1, rho=0.4, roots=("svc_a", "svc_b", "db", "ext"),
                     n_pre=20, n_post=40, delta=3.0, target="frontend"):
    """FAKE issue folders in the real PetShop layout (<scenario>/{train,test}/issueK/{metrics.csv,target.json}).
    Each issue shifts the *own mechanism* of one node (extra latency) which then propagates to its callers."""
    nodes = ["frontend", "api", "svc_a", "svc_b", "db", "queue", "cache", "ext", "lat1", "lat2"]
    calls = [("frontend", "api"), ("api", "svc_a"), ("api", "svc_b"), ("svc_a", "db"), ("svc_b", "db"), ("svc_b", "cache"),
             ("svc_a", "queue"), ("frontend", "ext"), ("svc_a", "lat1"), ("svc_b", "lat1"), ("frontend", "lat2"), ("ext", "lat2")]
    latent = {"lat1", "lat2"}
    order = ["db", "queue", "cache", "lat1", "lat2", "ext", "svc_a", "svc_b", "api", "frontend"]
    obs = [v for v in nodes if v not in latent]
    import json
    for k, rc in enumerate(roots):
        rng = np.random.default_rng(seed * 100 + k)
        m = n_pre + n_post
        series = {}
        for v in order:
            e = rng.standard_t(5, m)
            for i in range(1, m):
                e[i] += rho * e[i - 1]
            x = np.exp(0.3 * e) + 1.0
            if v == rc:
                x[n_pre:] += delta
            for a, b in calls:
                if a == v:
                    x = x + 0.6 * series[b]
            series[v] = x
        t0 = pd.Timestamp("2023-04-13 12:00:00", tz="UTC")
        idx = pd.date_range(t0, periods=m, freq="5min")
        cols = pd.MultiIndex.from_tuples([(v, "latency", "Average") for v in obs] + [(v, "error", "Sum") for v in obs])
        M = pd.DataFrame(np.column_stack([series[v] for v in obs] + [rng.random(m) for _ in obs]), columns=cols, index=idx)
        split = "train" if k % 2 == 0 else "test"
        p = os.path.join(root, scenario, split, f"issue{k}")
        os.makedirs(p, exist_ok=True)
        M.to_csv(os.path.join(p, "metrics.csv"))
        ts = int(idx[n_pre].timestamp())
        json.dump({"target": {"node": target, "metric": "latency", "agg": "Average", "timestamp": ts},
                   "root_cause": {"node": rc, "metric": None}, "metadata": {}}, open(os.path.join(p, "target.json"), "w"))
    return root
