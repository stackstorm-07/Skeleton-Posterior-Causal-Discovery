# PetShop: SPOT vs ABIC (CSE 516)

* **Stage 1** — graphs (ADMG/MAG/PAG, latent projection, Tetrad bridge), corrected + legacy F1 metrics, PetShop loader with inverted call-graph ground truth.
* **Stage 2a** — linear-Gaussian ADMG model, RICF, the shared differentiable optimizer (`pgm/dcd.py`; ABIC = no mask, SPOT = gradient mask), analytic gradients with finite-difference checks, synthetic simulator. See `STAGE2A_NOTES.md`.
* **Stage 2b** — skeleton posterior: labelled static simulator, Fisher-z CI features, 3-stage gradient-boosting cascade, offline training script, the p > alpha check, 14 tests. See `STAGE2B_NOTES.md`. Trained model: `models/skeleton_posterior.pkl`.
* **Stage 2c** — SPOT (`pgm/spot.py`, Eq. 9 mask + optional hard floor), 13 tests, synthetic SPOT-vs-ABIC benchmark and ablations (`results/syn_2c_*`). See `STAGE2C_NOTES.md`.
* **Stage 2d** — PetShop runner, data check, queue and summary built and tested on a fake PetShop-format dataset; **real runs pending the dataset** (`STAGE2D_NOTES.md`).
* 2e report tables: not yet (`scripts/summarize_2d.py` already produces the raw tables).

Based on SPOT (Ma et al., KDD'24, Apache-2.0, `third_party/Spot_original`), ABIC (Bhattacharya et al. 2021), PetShop (Hardt et al., CC-BY-4.0).

## Setup
* numpy, scipy, networkx, scikit-learn; Java for the Tetrad MAG->PAG step. No autodiff library is needed.
* Put `causal-cmd-1.3.0-jar-with-dependencies.jar` (ships in the original `Spot.zip`) in `third_party/Spot_original/learn/bin/` or set `CAUSAL_CMD_JAR`.
* PetShop data: set `PETSHOP_ROOT` to the dataset directory (tests that need it are skipped otherwise).

## Use
```python
from pgm.abic import run_abic
from pgm.dcd import DCDConfig
res = run_abic(X, names, DCDConfig(num_restarts=3, seed=0, time_budget=600))   # X: (n, d) array
res.admg, res.bic, res.t_optimize, res.converged
```

## Checks
```
python scripts/gradcheck.py          # analytic vs finite-difference gradients
python tests/run_tests.py            # all tests (plain Python);  --fast skips the slow ones;  pytest also works
python scripts/time_abic.py          # ABIC runtime vs graph size -> results/abic_timing.json
python scripts/train_posterior.py    # offline skeleton-posterior training + held-out metrics (~4 min)
python scripts/check_pvalue_direction.py   # p > alpha check, 4 cascade variants (~6 min)
python scripts/calibrate_floor.py          # SPOT hard-floor calibration
scripts/run_2c_queue.sh                    # synthetic SPOT vs ABIC (~2 h, one core, resumable)
python scripts/summarize_2c.py results/syn_2c_*.jsonl --out results/syn_2c_summary.md
PETSHOP_ROOT=/path/to/dataset scripts/run_2d_queue.sh   # Stage 2d on PetShop (~5-8 h one core; resumable)
python scripts/petshop_data_check.py       # n / d / autocorrelation vs the posterior's training range
```

## Stage 2e
`python scripts/make_report.py` -> `results/report_tables.{md,tex,*.csv}` (see `STAGE2E_NOTES.md`).

## Stage 3
`scripts/run_rca.py` + `scripts/summarize_rca.py` (root-cause ranking with learned ADMGs), `ADMG_JUSTIFICATION.md`, `STAGE3_NOTES.md`. Built and tested on fake PetShop-format data; real-data results not yet produced.
