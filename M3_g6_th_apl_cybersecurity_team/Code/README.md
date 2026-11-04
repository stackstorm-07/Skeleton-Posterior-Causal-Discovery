# Code — PetShop: SPOT vs ABIC

All code lives in `src/` (the `petshop_spot` package). Run every command from inside `src/`.

## What is in `src/`
| Path | Contents |
|---|---|
| `pgm/graphs.py`, `pgm/petshop.py` | ADMG/MAG/PAG utilities, latent projection, Tetrad MAG->PAG bridge; PetShop loader and inverted-call-graph ground truth |
| `pgm/admg_model.py`, `pgm/ricf.py` | Linear-Gaussian ADMG model and RICF fitting |
| `pgm/dcd.py` | The shared differentiable optimizer (augmented Lagrangian + L-BFGS-B, analytic gradients). **ABIC = no mask**, **SPOT = gradient mask** |
| `pgm/abic.py`, `pgm/spot.py` | ABIC wrapper; SPOT mask (Eq. 9) and optional hard floor |
| `pgm/metrics.py` | Skeleton/arrowhead/tail F1, TPR, FDR (corrected and legacy indexing) |
| `pgm/rca.py`, `pgm/petshop_issues.py` | Root-cause ranking (anomaly / dag / admg scores) and issue-file loader |
| `skeleton/` | Simulator, Fisher-z CI features, 3-stage gradient-boosting skeleton posterior |
| `models/skeleton_posterior.pkl` | Trained posterior used in all SPOT runs |
| `scripts/` | Entry points (below) |
| `tests/` | Unit tests (plain-Python runner) |
| `results/` | Outputs of the runs reported in the paper |
| `third_party/Spot_original` | The released SPOT code (Apache-2.0), used for the Tetrad jar and reference |
| `*_NOTES.md`, `ADMG_JUSTIFICATION.md` | Design notes per stage and the written ADMG argument |

## Setup
- Python 3 with `numpy`, `scipy`, `networkx`, `scikit-learn`, `pandas`. No autodiff library is needed.
- Java, for the Tetrad MAG->PAG step. Put `causal-cmd-1.3.0-jar-with-dependencies.jar` (ships in the original `Spot.zip`) in `third_party/Spot_original/learn/bin/`, or set `CAUSAL_CMD_JAR=/path/to/jar`.
- PetShop data: see `../Data/README.md`, then `export PETSHOP_ROOT=/path/to/dataset`.

## Quick checks (no data needed)
```
python scripts/gradcheck.py            # analytic vs finite-difference gradients
python tests/run_tests.py              # all tests; add --fast to skip slow ones
python tests/run_tests.py -k test_rca  # inference-layer tests only
```
Tests needing PetShop or the Tetrad jar are skipped if those are absent.

## Minimal use
```python
from pgm.abic import run_abic
from pgm.dcd import DCDConfig
res = run_abic(X, names, DCDConfig(num_restarts=3, seed=0, time_budget=600))  # X: (n, d)
res.admg, res.bic, res.t_optimize, res.converged
```
SPOT is run through `pgm/spot.py` with the same `DCDConfig`; see `STAGE2C_NOTES.md`.

## Reproducing the reported results
| Step | Command | Time (1 core) |
|---|---|---|
| Train skeleton posterior | `python scripts/train_posterior.py` | ~4 min |
| Calibrate the hard floor | `python scripts/calibrate_floor.py` | — |
| p-value direction check | `python scripts/check_pvalue_direction.py` | ~6 min |
| ABIC runtime vs size | `python scripts/time_abic.py` | — |
| Synthetic SPOT vs ABIC + ablations | `scripts/run_2c_queue.sh`, then `python scripts/summarize_2c.py results/syn_2c_*.jsonl --out results/syn_2c_summary.md` | ~2 h, resumable |
| PetShop data check | `PETSHOP_ROOT=... python scripts/petshop_data_check.py` | short |
| PetShop SPOT vs ABIC | `PETSHOP_ROOT=... scripts/run_2d_queue.sh` | ~5–8 h, resumable |
| Report tables | `python scripts/make_report.py` -> `results/report_tables.{md,tex,*.csv}` | seconds |
| Root-cause ranking | `PETSHOP_ROOT=... python scripts/run_rca.py --scenarios low_traffic high_traffic temporal_traffic1 temporal_traffic2 --sources abic spot_cal truth --out results/rca_stage3.jsonl`, then `python scripts/summarize_rca.py results/rca_stage3.jsonl --out results/rca_summary` | graphs ~1–3 min each, scoring seconds |

Runs are seeded and resumable; timings are single-core, so compare ratios rather than absolute seconds.

## Status and caveats
- PetShop results in the report come from `results/ps_2d_main_gauss_rank.jsonl` (3 scenarios, 2–3 seeds, one restart). `temporal_traffic2` and the PetShop ablations are not run.
- `run_rca.py` is implemented and unit-tested on fake PetShop-format data (`tests/fake_petshop.py`); it has not been run on the real issue folders yet. Check node-name matching (`root_match` counts) when you do.
- The released SPOT code has an inverted p-value test and a metric-indexing bug; we use corrected versions and report legacy metrics alongside (see report, Section IV-E).
- XGBoost is not used; the posterior uses scikit-learn's `HistGradientBoostingClassifier`.
