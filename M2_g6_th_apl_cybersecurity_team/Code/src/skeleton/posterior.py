"""Skeleton posterior p_ij = P(i and j are adjacent in the MAG | data): a cascade of gradient-boosted classifiers.

Cascade (ML4S-like, paper Sec. 3.2 / App. B), one classifier per conditioning-set order:

  stage 0  Markov-blanket features (conditioning candidates MB(i) u MB(j))               -> p0 for every pair
  stage 1  order-1 CI summaries, candidates = neighbours of i, j in {p0 >= tau0}          -> p1 for pairs in S0
  stage 2  order-2 CI summaries, candidates = neighbours of i, j in {p1 >= tau1}          -> p2 for pairs in S1

The final posterior of a pair is the output of the *last stage that scored it*; pairs dropped early keep their
(low) early probability.  The thresholds tau0 < tau1 are deliberately low: the paper prefers recall over
precision because a missed edge cannot be recovered by the optimizer while a spurious one is pruned.

Training uses graphs from the static simulator, labelled with the skeleton of the true MAG.  The inputs of
stage k+1 are *out-of-fold* predictions of stage k (K-fold over graphs), so the next stage is trained on the same
kind of (imperfect) candidate sets it will see at test time.

``stage0='alpha'`` replaces the stage-0 classifier by the plain order-0 test (the released code's design) and
``legacy_inverted=True`` flips the p-value comparisons as in the released code; both exist only for the
p > alpha check (scripts/check_pvalue_direction.py).

sklearn's HistGradientBoostingClassifier is used because xgboost is unavailable here; it is the same model family.
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from .features import (CITable, all_pairs, markov_blanket, neighbours, stage0_features, stage_features,
                       zero_order_adjacency)


@dataclass
class PosteriorConfig:
    n_stages: int = 3                 # 1 = stage 0 only, 2 = + order 1, 3 = + order 2
    tau0: float = 0.05                # candidate threshold after stage 0   (recall oriented)
    tau1: float = 0.10                # candidate threshold after stage 1
    alpha_mb: float = 0.05            # Markov-blanket test level
    max_mb: int = 12
    max_sub: int = 300                # cap on conditioning subsets per pair and order
    stage0: str = "classifier"        # 'classifier' | 'alpha'
    alpha0: float = 0.15              # level of the order-0 screen when stage0 == 'alpha'
    legacy_inverted: bool = False     # flip the p-value comparisons (released-code behaviour), diagnostic only
    cv_folds: int = 3
    max_iter: int = 200
    learning_rate: float = 0.08
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 40
    l2: float = 1.0
    seed: int = 0


def _clf(cfg: PosteriorConfig, seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_iter=cfg.max_iter, learning_rate=cfg.learning_rate,
                                          max_leaf_nodes=cfg.max_leaf_nodes, min_samples_leaf=cfg.min_samples_leaf,
                                          l2_regularization=cfg.l2, early_stopping=False, random_state=seed)


@dataclass
class PosteriorResult:
    P: np.ndarray                     # (d, d) symmetric, zero diagonal: the skeleton posterior
    t_features: float                 # seconds building CI tables / Markov blankets / feature rows
    t_model: float                    # seconds inside classifier predict_proba
    n_scored: List[int]               # number of pairs scored at each stage
    stage_P: List[np.ndarray] = field(default_factory=list)

    @property
    def t_total(self) -> float:
        return self.t_features + self.t_model


class _LazyCI:
    """``cis[k]`` builds the CI table of instance k (cheap, ~0.1 s) instead of holding hundreds of d^4 tensors."""

    def __init__(self, instances):
        self.instances = instances
        self._k, self._ci = None, None

    def __getitem__(self, k):
        if self._k != k:
            self._k, self._ci = k, CITable(self.instances[k].X)
        return self._ci


class SkeletonPosterior:
    def __init__(self, cfg: Optional[PosteriorConfig] = None):
        self.cfg = cfg or PosteriorConfig()
        self.models: List[Optional[HistGradientBoostingClassifier]] = []
        self.info: Dict = {}

    # ------------------------------------------------------------------ helpers
    def _stage0_adj_alpha(self, ci: CITable) -> np.ndarray:
        return zero_order_adjacency(ci, self.cfg.alpha0, self.cfg.legacy_inverted)

    def _mb(self, ci: CITable):
        c = self.cfg
        return markov_blanket(ci, c.alpha_mb, c.alpha_mb, c.legacy_inverted, c.max_mb)

    @staticmethod
    def _sym(d: int, pairs, vals, fill=0.0) -> np.ndarray:
        P = np.full((d, d), fill, dtype=float)
        for (i, j), v in zip(pairs, vals):
            P[i, j] = P[j, i] = v
        np.fill_diagonal(P, 0.0)
        return P

    # ------------------------------------------------------------------ training
    def fit(self, instances: Sequence, verbose: bool = False) -> "SkeletonPosterior":
        """``instances``: objects with ``.X`` (n, d) and ``.skeleton`` (d, d bool), e.g. simulator.Instance."""
        c = self.cfg
        t_all = time.perf_counter()
        t_feat = t_fit = 0.0
        N = len(instances)
        folds = np.arange(N) % c.cv_folds
        np.random.default_rng(c.seed).shuffle(folds)
        pairs = [all_pairs(inst.X.shape[1]) for inst in instances]
        labels = [np.array([inst.skeleton[i, j] for i, j in pr], dtype=int) for inst, pr in zip(instances, pairs)]
        self.models = [None] * c.n_stages
        self.info = dict(n_instances=N, rows=[])

        # ---- stage 0 ---------------------------------------------------------------------------
        t = time.perf_counter()
        cis = _LazyCI(instances)                                  # CI tables are rebuilt on demand (each is ~d^4 floats)
        stage_P: List[List[np.ndarray]] = [[] for _ in range(N)]  # per instance, per finished stage: d x d probabilities
        adj_prev: List[np.ndarray] = [None] * N                   # candidate adjacency entering the next stage
        if c.stage0 == "classifier":
            F0 = [stage0_features(cis[k], pr, self._mb(cis[k]), c.max_sub) for k, pr in enumerate(pairs)]
            t_feat += time.perf_counter() - t
            P0 = self._fit_stage(0, F0, labels, folds, [np.ones(len(p), bool) for p in pairs])
            t_fit += self._last_fit_time
            for k, inst in enumerate(instances):
                d = inst.X.shape[1]
                stage_P[k].append(self._sym(d, pairs[k], P0[k]))
                adj_prev[k] = stage_P[k][0] >= c.tau0
        else:                                                      # 'alpha': order-0 test is the first stage
            for k in range(N):
                adj_prev[k] = self._stage0_adj_alpha(cis[k])
                stage_P[k].append(adj_prev[k].astype(float))
            t_feat += time.perf_counter() - t

        # ---- stages 1 .. n_stages-1 ----------------------------------------------------------------
        for s in range(1, c.n_stages):
            t = time.perf_counter()
            # released-code behaviour: in 'alpha' mode the order-0 screen only supplies conditioning candidates and
            # order 1 scores ALL pairs; otherwise only pairs that survived the previous stage are scored
            all_at_1 = (s == 1 and c.stage0 == "alpha")
            sel_pairs = [[(i, j) for (i, j) in pr if all_at_1 or adj_prev[k][i, j]] for k, pr in enumerate(pairs)]
            F = [stage_features(cis[k], sel_pairs[k], s, neighbours(adj_prev[k]), stage_P[k], c.max_sub)
                 if sel_pairs[k] else np.zeros((0, 1)) for k in range(N)]
            ys = [np.array([instances[k].skeleton[i, j] for i, j in sel_pairs[k]], dtype=int) for k in range(N)]
            t_feat += time.perf_counter() - t
            Pk = self._fit_stage(s, F, ys, folds, [np.ones(len(sp), bool) for sp in sel_pairs])
            t_fit += self._last_fit_time
            for k, inst in enumerate(instances):
                d = inst.X.shape[1]
                M = stage_P[k][-1].copy()                          # pairs not scored here keep their earlier probability
                for (i, j), v in zip(sel_pairs[k], Pk[k]):
                    M[i, j] = M[j, i] = v
                stage_P[k].append(M)
                adj_prev[k] = (M >= c.tau1) if all_at_1 else ((M >= c.tau1) & adj_prev[k])
            if verbose:
                kept = np.mean([adj_prev[k].sum() / 2 for k in range(N)])
                rec = np.mean([(adj_prev[k] & instances[k].skeleton).sum() / max(instances[k].skeleton.sum(), 1)
                               for k in range(N)])
                print(f"  stage {s}: candidate pairs/graph {kept:.0f}, candidate recall {rec:.3f}")
        self.info.update(t_features=t_feat, t_fit=t_fit, t_train_total=time.perf_counter() - t_all)
        return self

    def _fit_stage(self, s: int, F: List[np.ndarray], y: List[np.ndarray], folds: np.ndarray, use) -> List[np.ndarray]:
        """Fit the stage-s classifier on all graphs and return OUT-OF-FOLD probabilities per graph."""
        c = self.cfg
        N = len(F)
        t = time.perf_counter()
        oof = [None] * N
        for f in range(c.cv_folds):
            tr = [k for k in range(N) if folds[k] != f and len(y[k])]
            te = [k for k in range(N) if folds[k] == f]
            m = _clf(c, c.seed + 10 * s + f).fit(np.vstack([F[k] for k in tr]), np.concatenate([y[k] for k in tr]))
            for k in te:
                oof[k] = m.predict_proba(F[k])[:, 1] if len(y[k]) else np.zeros(0)
        allk = [k for k in range(N) if len(y[k])]
        model = _clf(c, c.seed + 10 * s + 99).fit(np.vstack([F[k] for k in allk]), np.concatenate([y[k] for k in allk]))
        if s >= len(self.models):
            self.models.extend([None] * (s + 1 - len(self.models)))
        self.models[s] = model
        self.info["rows"].append(dict(stage=s, rows=int(sum(len(v) for v in y)), positives=int(sum(v.sum() for v in y))))
        self._last_fit_time = time.perf_counter() - t
        return oof

    # ------------------------------------------------------------------ inference
    def predict(self, X: np.ndarray) -> PosteriorResult:
        c = self.cfg
        X = np.asarray(X, dtype=float)
        d = X.shape[1]
        pairs = all_pairs(d)
        t_feat = t_mod = 0.0
        n_scored: List[int] = []
        t = time.perf_counter()
        ci = CITable(X)
        stage_P: List[np.ndarray] = []
        if c.stage0 == "classifier":
            F0 = stage0_features(ci, pairs, self._mb(ci), c.max_sub)
            t_feat += time.perf_counter() - t
            t = time.perf_counter()
            p0 = self.models[0].predict_proba(F0)[:, 1]
            t_mod += time.perf_counter() - t
            stage_P.append(self._sym(d, pairs, p0))
            adj = stage_P[0] >= c.tau0
            n_scored.append(len(pairs))
        else:
            adj = self._stage0_adj_alpha(ci)
            stage_P.append(adj.astype(float))
            t_feat += time.perf_counter() - t
            n_scored.append(len(pairs))
        for s in range(1, c.n_stages):
            all_at_1 = (s == 1 and c.stage0 == "alpha")
            sel = [(i, j) for (i, j) in pairs if all_at_1 or adj[i, j]]
            n_scored.append(len(sel))
            M = stage_P[-1].copy()
            if sel:
                t = time.perf_counter()
                F = stage_features(ci, sel, s, neighbours(adj), stage_P, c.max_sub)
                t_feat += time.perf_counter() - t
                t = time.perf_counter()
                p = self.models[s].predict_proba(F)[:, 1]
                t_mod += time.perf_counter() - t
                for (i, j), v in zip(sel, p):
                    M[i, j] = M[j, i] = v
            stage_P.append(M)
            adj = (M >= c.tau1) if all_at_1 else ((M >= c.tau1) & adj)
        return PosteriorResult(P=stage_P[-1], t_features=t_feat, t_model=t_mod, n_scored=n_scored, stage_P=stage_P)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(dict(cfg=asdict(self.cfg), models=self.models, info=self.info), f)

    @classmethod
    def load(cls, path: str) -> "SkeletonPosterior":
        with open(path, "rb") as f:
            blob = pickle.load(f)
        obj = cls(PosteriorConfig(**blob["cfg"]))
        obj.models, obj.info = blob["models"], blob["info"]
        return obj


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def _upper(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices(M.shape[0], 1)]


def posterior_metrics(P: np.ndarray, skeleton: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """AUROC / AUPRC (as in paper Table 3), KL, Brier and ECE of one posterior against the true skeleton.

    KL(point-mass truth || Bernoulli(p)) is mean(-log p_true-label), i.e. the log-loss, which is how a KL
    divergence to a 0/1 skeleton reduces; reported per pair."""
    p = np.clip(_upper(P), 1e-6, 1 - 1e-6)
    y = _upper(skeleton.astype(float))
    out = dict(
        auroc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        auprc=float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
        kl=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        brier=float(np.mean((p - y) ** 2)),
    )
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    ece = sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(10) if (idx == b).any())
    pred = p >= thr
    tp = float((pred & (y > 0)).sum())
    out.update(ece=float(ece), recall=tp / max(y.sum(), 1.0), precision=tp / max(pred.sum(), 1.0),
               recall_p05=float(((p >= 0.05) & (y > 0)).sum() / max(y.sum(), 1.0)),
               recall_p10=float(((p >= 0.10) & (y > 0)).sum() / max(y.sum(), 1.0)))
    return out


def evaluate(post: SkeletonPosterior, instances: Sequence) -> List[Dict]:
    rows = []
    for inst in instances:
        res = post.predict(inst.X)
        m = posterior_metrics(res.P, inst.skeleton)
        m.update(t_features=res.t_features, t_model=res.t_model, d=inst.X.shape[1], n=inst.X.shape[0],
                 ar_rho=inst.meta.get("ar_rho", 0.0), n_true=int(inst.skeleton.sum() // 2))
        for s, Ps in enumerate(res.stage_P):               # AUROC after each stage, for the cascade ablation
            m[f"auroc_stage{s}"] = posterior_metrics(Ps, inst.skeleton)["auroc"]
        rows.append(m)
    return rows


# ----------------------------------------------------------------------------------------------
# Reference posteriors for the SPOT ablations (used in stage 2c/2d)
# ----------------------------------------------------------------------------------------------
def oracle_posterior(skeleton: np.ndarray, smooth: float = 0.0) -> np.ndarray:
    """Posterior that knows the true skeleton: 1 on true edges, ``smooth`` elsewhere (0 = hard)."""
    P = np.where(skeleton, 1.0, smooth).astype(float)
    np.fill_diagonal(P, 0.0)
    return P


def random_posterior(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uninformative control: the same multiset of probabilities as ``P`` assigned to random pairs
    (same density and confidence profile, no information about the graph)."""
    d = P.shape[0]
    iu = np.triu_indices(d, 1)
    v = P[iu].copy()
    rng.shuffle(v)
    Q = np.zeros_like(P)
    Q[iu] = v
    return Q + Q.T
