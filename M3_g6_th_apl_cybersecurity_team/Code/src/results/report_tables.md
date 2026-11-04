# Milestone report tables: ABIC vs SPOT on PetShop (stage 2e)

> **Status: PARTIAL.** Numbers are final for what finished, not for the planned design.

* Preprocessing `gauss_rank`, one restart, no wall-clock budget, original posterior pickle.
* Completed (scenario: seeds): low traffic: [0, 1, 2]; high traffic: [0, 1, 2]; temporal 1: [0, 1].
* Not run: temporal 2.
* Fewer seeds than planned: low traffic (seeds [3, 4] missing); high traffic (seeds [3, 4] missing); temporal 1 (seeds [2, 3, 4] missing).
* Not run at all: ablations (no-floor, floor-only, random/oracle posterior) on PetShop, equal-budget runs, `raw_centered` sensitivity run.
* Every completed run converged (24/24); none hit a time limit (0/24).

> PetShop is one dataset per scenario: seeds vary only the optimizer, so ± is optimizer variance, not data variance, and no significance claim is made. With 3 scenarios, 2-3 seeds and one restart the ranking of methods is tentative.

## Key readings

* SPOT (calibrated floor 6e-3) vs ABIC over 3 scenarios: skeleton F1 +0.036 (W3/L0/T0), arrowhead +0.016 (W2/L1/T0), tail +0.049 (W2/L0/T1).
* SPOT (released floor 1e-3) vs ABIC over 3 scenarios: skeleton F1 +0.021 (W3/L0/T0), arrowhead -0.004 (W2/L1/T0), tail +0.007 (W1/L1/T1).
* The skeleton-F1 gain is consistent in sign across scenarios but small and of the same order as the seed-to-seed (optimizer) spread (Table 2); arrowhead and tail differences are inside that spread. Tail F1 is low for every method (scenario means 0.056-0.105).
* Both learners return 17-41 bidirected edges per run against 0-1 in the ground truth, so arrowhead/tail scores mostly measure how directed edges are mislabelled as bidirected; do not over-read them.
* Optimization time per run: ABIC 115 s, SPOT-cal 46 s, SPOT-1e-3 58 s (run-weighted). The saving tracks the number of free pairs (the floor), i.e. pruning; the PetShop ablations that would separate pruning from the stochastic mask were not run (the synthetic ablation in 2c could not separate them either with 3 graphs).
* The posterior is weaker on PetShop than on simulated graphs (Table 5), and the 6e-3 floor pins a much larger share of true edges here, which caps recall for SPOT-cal.

## Table 1. F1 across scenarios (mean of 3 scenario means; in brackets: difference to ABIC; wins/losses/ties over scenarios)

Corrected metric unless marked legacy. Skeleton F1 is the same under both conventions.

| method                       | Skeleton F1              | Arrowhead F1             | Tail F1                  | Arrowhead F1 (legacy)    | Tail F1 (legacy)         |
|:-----------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|:-------------------------|
| ABIC                         | 0.514                    | 0.301                    | 0.056                    | 0.479                    | 0.271                    |
| SPOT (calibrated floor 6e-3) | 0.550 (+0.036; W3/L0/T0) | 0.317 (+0.016; W2/L1/T0) | 0.105 (+0.049; W2/L0/T1) | 0.488 (+0.009; W2/L1/T0) | 0.385 (+0.115; W2/L1/T0) |
| SPOT (released floor 1e-3)   | 0.535 (+0.021; W3/L0/T0) | 0.297 (-0.004; W2/L1/T0) | 0.063 (+0.007; W1/L1/T1) | 0.471 (-0.007; W1/L2/T0) | 0.350 (+0.079; W2/L1/T0) |

## Table 2. Per-scenario results (mean ± sd over optimizer seeds)

| scenario     | n / d     | seeds   | method    | Skeleton F1   | TPR         | FDR         | Arrowhead F1   | Tail F1       |
|:-------------|:----------|:--------|:----------|:--------------|:------------|:------------|:---------------|:--------------|
| low traffic  | 589 / 29  | 3       | ABIC      | 0.484 ± 0.019 | 0.49 ± 0.02 | 0.52 ± 0.02 | 0.288 ± 0.018  | 0.000 ± 0.000 |
| low traffic  | 589 / 29  | 3       | SPOT-cal  | 0.521 ± 0.028 | 0.51 ± 0.04 | 0.46 ± 0.01 | 0.243 ± 0.015  | 0.000 ± 0.000 |
| low traffic  | 589 / 29  | 3       | SPOT-1e-3 | 0.514 ± 0.017 | 0.51 ± 0.02 | 0.47 ± 0.04 | 0.298 ± 0.013  | 0.000 ± 0.000 |
| high traffic | 589 / 35  | 3       | ABIC      | 0.513 ± 0.018 | 0.61 ± 0.04 | 0.56 ± 0.01 | 0.299 ± 0.012  | 0.167 ± 0.059 |
| high traffic | 589 / 35  | 3       | SPOT-cal  | 0.537 ± 0.014 | 0.60 ± 0.03 | 0.51 ± 0.00 | 0.301 ± 0.059  | 0.181 ± 0.166 |
| high traffic | 589 / 35  | 3       | SPOT-1e-3 | 0.536 ± 0.019 | 0.62 ± 0.03 | 0.53 ± 0.02 | 0.258 ± 0.014  | 0.123 ± 0.003 |
| temporal 1   | 1652 / 31 | 2       | ABIC      | 0.544 ± 0.004 | 0.61 ± 0.00 | 0.51 ± 0.01 | 0.317 ± 0.054  | 0.000 ± 0.000 |
| temporal 1   | 1652 / 31 | 2       | SPOT-cal  | 0.590 ± 0.019 | 0.62 ± 0.02 | 0.44 ± 0.02 | 0.407 ± 0.024  | 0.133 ± 0.000 |
| temporal 1   | 1652 / 31 | 2       | SPOT-1e-3 | 0.554 ± 0.017 | 0.58 ± 0.03 | 0.47 ± 0.01 | 0.334 ± 0.010  | 0.067 ± 0.067 |

## Table 3. Runtime breakdown

Three numbers: offline posterior training (paid once, shared by all SPOT runs), online posterior inference (per dataset, charged to SPOT), optimization (per run). Single-core timings: compare ratios, not absolute seconds.

| method                       | offline training (one-time)   | online posterior / run   | optimization / run (all runs)   | optimization / run (scenario mean)   | vs ABIC   | L-BFGS evals / run   | free pairs (of d(d-1)/2)   | converged   |
|:-----------------------------|:------------------------------|:-------------------------|:--------------------------------|:-------------------------------------|:----------|:---------------------|:---------------------------|:------------|
| ABIC                         | n/a                           | 0                        | 115 s                           | 107 s                                | 1.00x     | 97,916               | all                        | 8/8         |
| SPOT (calibrated floor 6e-3) | 203 s                         | 0.32 s                   | 46 s                            | 45 s                                 | 0.42x     | 40,331               | 105                        | 8/8         |
| SPOT (released floor 1e-3)   | 203 s                         | 0.32 s                   | 58 s                            | 57 s                                 | 0.53x     | 49,784               | 198                        | 8/8         |

Optimization time per scenario (s, mean over seeds):

| scenario     | ABIC opt. time (s)   | SPOT-cal opt. time (s)   | SPOT-1e-3 opt. time (s)   |
|:-------------|:---------------------|:-------------------------|:--------------------------|
| low traffic  | 83                   | 19                       | 32                        |
| high traffic | 192                  | 78                       | 92                        |
| temporal 1   | 47                   | 39                       | 46                        |

## Table 4. Estimated vs true edge counts (mean over seeds)

| scenario     | method    | est. directed   | est. bidirected   | true directed   | true bidirected   |
|:-------------|:----------|:----------------|:------------------|:----------------|:------------------|
| low traffic  | ABIC      | 7.3             | 25.0              | 30              | 1                 |
| low traffic  | SPOT-cal  | 9.3             | 19.7              | 30              | 1                 |
| low traffic  | SPOT-1e-3 | 9.3             | 20.7              | 30              | 1                 |
| high traffic | ABIC      | 12.3            | 39.3              | 38              | 0                 |
| high traffic | SPOT-cal  | 14.3            | 32.0              | 38              | 0                 |
| high traffic | SPOT-1e-3 | 13.3            | 37.0              | 38              | 0                 |
| temporal 1   | ABIC      | 6.5             | 34.0              | 32              | 1                 |
| temporal 1   | SPOT-cal  | 8.5             | 28.0              | 32              | 1                 |
| temporal 1   | SPOT-1e-3 | 7.5             | 28.0              | 32              | 1                 |

## Table 5. Skeleton posterior: PetShop vs simulated

| data                      | posterior AUROC   | posterior AUPRC   | true edges pinned by 6e-3 floor   |
|:--------------------------|:------------------|:------------------|:----------------------------------|
| PetShop low traffic       | 0.830             | 0.515             | 32%                               |
| PetShop high traffic      | 0.931             | 0.590             | 16%                               |
| PetShop temporal 1        | 0.876             | 0.570             | 15%                               |
| simulated held-out graphs | 0.989             | 0.946             | ≈1%                               |
