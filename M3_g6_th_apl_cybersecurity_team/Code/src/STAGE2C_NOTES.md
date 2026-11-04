# Stage 2c notes: SPOT (skeleton-posterior-guided optimizer) and the synthetic check vs ABIC

## What exists now
| file | content |
|---|---|
| `pgm/spot.py` | `SpotMask`, `SpotConfig`, `run_spot`, `allowed_pairs`. ABIC and SPOT call the same `discover_admg`; SPOT only adds the mask (and optional floor). `run_spot` reports posterior inference time separately and can charge it to a wall-clock budget. |
| `tests/test_spot.py` | 13 tests (exact Eq. 9 rule, acceptance frequencies, pair->coordinate mapping, floor, p==1 & no floor == ABIC bit for bit, determinism, budget, oracle). Whole suite: **45 passed, 2 skipped** (PetShop-data tests). |
| `scripts/calibrate_floor.py` | floor calibration -> `results/floor_calibration.json` |
| `scripts/spot_vs_abic_synthetic.py`, `run_2c_queue.sh`, `summarize_2c.py` | benchmark (resumable), aggregation (paired, corrected + legacy metrics) |
| `results/syn_2c_*.jsonl`, `syn_2c_summary.md/.csv` | raw per-run results and aggregated tables |

## What SPOT is here (and where it differs from the paper text)
* Mask = paper Eq. 9, identical to the released `baselines/sa_ml4s_dcd.py`: at every L-BFGS evaluation, a coordinate with theta*g>0 (weakening) is always kept; otherwise it is kept with probability (p_ij + c)^(1/(t+1)), c=0.1, t = 1-based outer ALM iteration; rejected coordinates get gradient 0. p_ij drives W1[i,j], W1[j,i] and the pair's bidirected coordinate.
* **Hard floor (not in the paper text, in every released script):** pairs with p <= 1e-3 are pinned to 0 via bounds. Here a switch (`floor`).
* The released repo has a **second, different script** (`stochastic_ml4s_dcd.py`: sign-blind, rejected gradient x0.1, sqrt(p), only first 5 outer iterations). Available as `variant="released_stochastic"`; it is not the paper's Eq. 9 and was the worst variant here.
* **Floor calibration.** The released 1e-3 is tied to the released xgboost cascade; our HistGradientBoosting cascade is less extreme (1e-3 leaves 160 of 300 pairs free). We use 6e-3 = largest floor in a grid pinning <=1% of true edges on held-out simulated graphs (seed 777, disjoint from every benchmark graph; pins 0.93% of true edges and about 77% of non-edges). Both floors are reported: `spot` (1e-3, "as released") and `spot_cal` (6e-3). This is a one-time choice made without looking at benchmark results.

## Results (d=25, n=1000, 1 restart, same data/optimizer seed/settings for all variants; corrected metrics)
Per-graph pairing; Wilcoxon p is coarse with 6 graphs (min 0.031).

**Raw data (the paper's simulator; ABIC works properly), 6 graphs**
| | skeleton F1 | arrowhead F1 | tail F1 | total time (s) |
|---|---|---|---|---|
| ABIC | 0.894 | 0.806 | 0.704 | 75.7 |
| SPOT (1e-3) | 0.949 (+0.055, 4W/2L, p=0.16) | 0.878 | 0.751 | 55.5 |
| SPOT-cal (6e-3) | **0.961 (+0.067, 6W/0L, p=0.031)** | 0.916 (+0.110, 5W/1L, p=0.062) | 0.892 | **24.0 (0.32x)** |
In this setting the paper's qualitative claim (SPOT > ABIC, and not slower) is reproduced, with large per-graph timing variance (ABIC 33-130 s).

**z-scored data (what PetShop preprocessing will look like), 6 graphs**
| | skeleton F1 | arrowhead F1 | tail F1 | total time (s) |
|---|---|---|---|---|
| ABIC | 0.687 | 0.390 | 0.104 | 82.9 |
| SPOT (1e-3) | 0.718 (+0.031, 4W/2L, p=0.56) | 0.387 | 0.117 | 96.1 (1.32x) |
| SPOT-cal | 0.730 (+0.042, 5W/1L, p=0.094) | 0.419 | 0.106 | 59.4 (0.80x) |
Small, non-significant skeleton gain; no arrowhead/tail gain. Both methods return about 27 bidirected edges against 1-4 true ones (the 2a finding: the pseudo-variable loss over-uses bidirected edges on z-scored data; our objective matches the released `primal_loss`, so this is a property of the method, not a bug).

**Ablations (z-scored, 3 graphs; skeleton / arrowhead / tail F1; total s)**
| variant | skel | head | tail | time |
|---|---|---|---|---|
| ABIC | 0.635 | 0.308 | 0.065 | 83.6 |
| SPOT (1e-3) | 0.722 | 0.379 | 0.107 | 105.1 |
| SPOT-cal | 0.688 | 0.340 | 0.132 | 72.8 |
| mask only (no floor) | 0.671 | 0.345 | 0.064 | 81.9 |
| floor only (no mask) | 0.667 | 0.321 | 0.104 | 73.0 |
| released-script variant | 0.624 | 0.332 | 0.078 | 74.8 |
| random posterior (same probabilities, random pairs) | 0.330 | 0.149 | 0.088 | 37.6 |
| oracle posterior | 0.922 | 0.586 | 0.206 | 20.7 |
Reading: the information in the posterior matters (random posterior is far worse than ABIC; oracle is far better). The gain of the real posterior over ABIC is a fraction of the oracle's headroom: pairs the posterior leaves free let the optimizer keep adding spurious (mostly bidirected) edges. With 3 graphs, mask-only vs floor-only cannot be separated.

**Equal 60 s wall-clock budget (z-scored, 4 graphs)**
ABIC skeleton F1 0.668 (converged on 1/4), SPOT-cal 0.689 (converged on 2/4). Per graph: 0.545/0.491, 0.711/0.720, 0.676/0.722, 0.740/0.824. Mildly favourable, 3 of 4 graphs, not conclusive.

## Honest conclusion for 2c
* The implementation behaves as specified (tests, limiting cases, sensible controls). No sign of a bug: random posterior hurts, oracle helps, SPOT with p==1 equals ABIC exactly.
* SPOT beats ABIC clearly on raw synthetic data (calibrated floor: 6/6 graphs on skeleton F1, about 3x faster) and only marginally and non-significantly on z-scored data, where ABIC's own objective is the bottleneck.
* The speed benefit comes mostly from pruning (the floor), not from the stochastic mask: nearly every L-BFGS solve hits the 1000-iteration cap, so time is proportional to the number of solves. The paper's own Table 7 reports only about 8% less time.
* Caveats: 3-6 graphs per setting, one restart, d=25 only, one optimizer seed; the posterior was trained on d 25-40 data of the same simulator family (it is in-distribution here, but PetShop is not).

## For 2d (PetShop)
1. Decide preprocessing: z-scored is the planned default, but expect both methods to over-use bidirected edges, so arrowhead/tail F1 will be low for both. Raw/centered latency is the sensitivity check (and may favour ABIC for var-sortability reasons that do not hold in PetShop). Run both.
2. Report both floors, labelled "as released" (1e-3) and "calibrated" (6e-3).
3. Report the three-number runtime: offline posterior training (203 s), online inference (about 0.3 s), optimization.
4. Check PetShop n and lag-1 autocorrelation against the posterior's training ranges (n 300-2000, rho in {0, .3, .6}); retrain if outside.
5. Do not use the single-core timings here as absolute numbers; they were measured on 1 core.
