# Stage 2b notes: skeleton posterior

## What exists now
| file | content |
|---|---|
| `skeleton/simulator.py` | 2a generators plus `Instance`, `simulate_instance`, `make_instances`, `mag_skeleton`. Labels = skeleton of the true **MAG** (= skeleton of the true PAG, which is what the benchmark scores). Optional AR(1) rows (`ar_rho`) because PetShop is a time series. |
| `skeleton/features.py` | `CITable` (Fisher-z partial correlations of order 0, 1, 2 for all variable sets at once, plus a general-order path), grow-shrink Markov blanket, order-0 screen, stage-0 / stage-k feature rows (summaries of the Fisher-z statistic and p-value over conditioning sets: 4 moments + 11 percentiles). |
| `skeleton/posterior.py` | `SkeletonPosterior`: cascade of 3 `HistGradientBoostingClassifier`s (orders 0, 1, 2), out-of-fold inputs between stages, `fit / predict / save / load`, timing split, metrics (AUROC, AUPRC, log-loss "KL", Brier, ECE, recall at p>=0.05 / 0.10), and the SPOT ablation posteriors `oracle_posterior`, `random_posterior`. |
| `scripts/train_posterior.py` | offline training with timing; writes `models/skeleton_posterior.pkl` and `results/posterior_training.json` |
| `scripts/check_pvalue_direction.py` | the p > alpha check (`results/pvalue_direction.json/.log`) |
| `tests/test_posterior.py` | 14 tests. Whole suite: `python tests/run_tests.py` -> 32 passed, 2 skipped (the two PetShop-data tests, dataset not present here) |

The draft `features.py` / `posterior.py` in the 2a zip sat in `pgm/` and had no simulator labelling, no training script and no tests; they are now in `skeleton/` as in the plan, finished and tested. Two fixes while testing: the clip in `CITable` also shrank the correlation-matrix diagonal (error 3e-9 in order >= 3 partial correlations, now exact to 1e-9), and the `stage0='alpha'` diagnostic mode now scores all pairs at order 1 like the released code does (before, it silently scored only the screened pairs, which would have confounded the check).

## Held-out result (60 freshly simulated graphs, never trained on; 400 training graphs, d in 25-40)
| | AUROC | AUPRC | KL (log-loss/pair) | Brier | ECE | true edges with p >= 0.05 / 0.10 |
|---|---|---|---|---|---|---|
| ours | 0.989 | 0.946 | 0.057 | 0.015 | 0.013 | 95.7% / 92.6% |
| paper Table 3 (SPOT, 50-100 nodes) | 0.99 | 0.97 | 0.03 | | | |

AUROC by cascade stage: 0.983 -> 0.988 -> 0.989 (each order helps a little). By data: AR rho 0 / 0.3 / 0.6 -> 0.989 / 0.989 / 0.988; n < 600 -> 0.983, n >= 1200 -> 0.992. The paper's AUPRC and KL are somewhat better; graph sizes and settings differ and I do not know how the paper computes its KL (mine is the log-loss against the 0/1 skeleton), so do not read that gap as a defect or as a win.

**Runtime (for the three-number breakdown):** offline training 203 s one time (CI features 152 s, classifier fitting 51 s, simulation 3 s, single core, 400 graphs); online inference 0.29 s per dataset (features 0.28 s, model 0.009 s).

## The p > alpha check: confirmed, fixed, and smaller in effect than I expected
A small Fisher-z p-value means *dependent*. The released `learn/gen_feature.py` does: `prepare_zero_order` keeps a pair when `pval > alpha1`; `get_markov_blanket` adds a variable when `cit(...) > alpha1` and removes one when `cit(...) <= alpha2` (alpha = 0.15). Read literally, that selects independent pairs. I only read the source (the released pipeline needs xgboost, causal-learn and ananke, which are not available here), so what is established is: the code says this, and its effect in *our* re-implementation is as below. I cannot say what it did to the paper's numbers.

Same 40 held-out graphs (d 25-40), models trained on identical 120 graphs, differing only in comparison direction:

| step | correct (`p <= alpha`) | legacy (`p > alpha`) |
|---|---|---|
| order-0 screen keeps true edges | **98.1%** | **1.9%** |
| order-0 screen keeps non-edges | 41.3% | 58.7% |
| Markov blanket recall of true neighbours | 65.9% | **1.1%** |

At the level of the tests the inversion is decisive. End to end it is much milder, because the later classifiers re-derive adjacency from many conditioning sets:

| cascade (held-out) | AUROC | AUPRC | KL |
|---|---|---|---|
| A corrected, Markov-blanket classifier at stage 0 (**default**) | 0.988 | 0.936 | 0.062 |
| B legacy comparisons, same design | 0.976 | 0.915 | 0.076 |
| C corrected, order-0 screen at stage 0 | 0.981 | 0.920 | 0.072 |
| D legacy comparisons, order-0 screen (the released design) | 0.975 | 0.903 | 0.080 |

Paired over the 40 graphs, A beats D by 0.0122 AUROC (SE 0.0017, better on 36/40 graphs) and 0.034 AUPRC (SE 0.004, 38/40); A beats B by 0.012 AUROC (34/40). One training seed per variant. **Decision:** the correct direction is the default; `legacy_inverted=True` stays only for this diagnostic. In the report I would say "the released code appears to invert the test; in our re-implementation fixing it improves held-out AUROC by about 0.012", not that it invalidates the paper.

## Assumptions to know about (and what I could not check)
1. **PetShop is out of distribution for this posterior.** It is trained on linear-Gaussian simulated ancestral ADMGs; PetShop is skewed, autocorrelated and call-graph shaped. The held-out numbers above say nothing about PetShop accuracy. The real test is SPOT vs ABIC in 2c/2d, and I cannot run the posterior on PetShop here because the dataset is not in this environment.
2. **Training ranges are guesses:** d 25-40, n log-uniform 300-2000, AR(1) rho in {0, 0.3, 0.6}. `log n`, `d` and lag-1 autocorrelation are classifier inputs, and trees do not extrapolate. If PetShop's n or its lag-1 autocorrelation after Gaussian-rank preprocessing falls outside these ranges, tell me the numbers (or run `CITable(X).ac1` and `X.shape`) and I will retrain on matching ranges, or drop `ac1` as a feature.
3. **Recall:** 4.3% of true edges get p < 0.05 and about 8% candidate pairs are lost by stage 2 (candidate recall 0.918). Those are the edges SPOT's mask can hardly recover; the `c = 0.1` floor in Eq. 9 helps a little. If 2c shows missing edges, lowering `tau0`, `tau1` is the first thing to try.
4. sklearn `HistGradientBoostingClassifier` replaces xgboost (not installed); the same model family, not identical numbers.
5. The optional dynamic simulator with domain adaptation is deferred, as planned.

## Interface for 2c
```python
from skeleton.posterior import SkeletonPosterior, oracle_posterior, random_posterior
post = SkeletonPosterior.load("models/skeleton_posterior.pkl")
res = post.predict(X)        # X: preprocessed (n, d); res.P symmetric (d, d), zero diagonal; res.t_features, res.t_model
P_oracle = oracle_posterior(true_skeleton)            # ablation: oracle
P_random = random_posterior(res.P, rng)               # ablation: same probabilities, random pairs
```
Re-run: `python scripts/train_posterior.py` (about 4 min on one core), `python scripts/check_pvalue_direction.py` (about 6 min).
