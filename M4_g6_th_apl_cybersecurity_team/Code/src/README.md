# PetShop: SPOT vs ABIC (CSE 516)

* **Stage 1** — graphs (ADMG/MAG/PAG, latent projection, Tetrad bridge), corrected + legacy F1 metrics, PetShop loader with inverted call-graph ground truth.
* **Stage 2a** — linear-Gaussian ADMG model, RICF, the shared differentiable optimizer (`pgm/dcd.py`; ABIC = no mask, SPOT = gradient mask), analytic gradients with finite-difference checks, synthetic simulator. See `STAGE2A_NOTES.md`.
* **Stage 2b** — skeleton posterior: labelled static simulator, Fisher-z CI features, 3-stage gradient-boosting cascade, offline training script, the p > alpha check, 14 tests. See `STAGE2B_NOTES.md`. Trained model: `models/skeleton_posterior.pkl`.
* **Stage 2c** — SPOT (`pgm/spot.py`, Eq. 9 mask + optional hard floor), 13 tests, synthetic SPOT-vs-ABIC benchmark and ablations (`results/syn_2c_*`). See `STAGE2C_NOTES.md`.
* **Stage 2d** — PetShop runner, data check, queue and summary (`STAGE2D_NOTES.md`). Real runs: M2 (`results/ps_2d_*`, 24 runs, 3 scenarios, older posterior) and M3 (`results/ps_m3_*`, 4 scenarios, 5 seeds). **Never pool M2 and M3 rows.**
* 2e report tables: `scripts/make_report.py` (M2); M3 tables come from `scripts/aggregate_quick.py` (what the submitted M3 report used) and `scripts/aggregate_m3.py` (the fuller aggregator with the hypothesis scorecard).

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

## M3 additions (see CHANGES_M3.md)
`pgm/stats.py`, `scripts/run_bootstrap.py` (E5); `--design e1|e2` and `--density/--snr/--ar/--missing` in `spot_vs_abic_synthetic.py` (E1/E2);
`<variant>_dir` directed-only variants (E10); `pgm/fci.py` + `scripts/run_fci.py` (E9).
`scripts/run_queue_parallel.py` runs the plan's experiment list in one resumable process pool (CHANGES_M3.md, step 8); `scripts/posterior_quality.py` (E7).

## M4 additions (see CHANGES_M4.md)
WP0: `skeleton/features.py` no longer stores a d^4 tensor (`CITable.p2` evaluates order-2 partial correlations on demand; predictions are bit-identical to M3);
`pgm/envinfo.py` stamps every queue row with peak memory, CPU time, load average, CPU model, library versions and a code hash (`timing_clean` flag);
`DCDConfig.checkpoint_cb` is the anytime hook. `scripts/aggregate_m3.py` is restored to the full aggregator; the short script is `scripts/aggregate_quick.py`.

**Posterior loading is strict (M4).** A scikit-learn pickle written by another version is refused, not loaded with a warning; scripts default to `models/skeleton_posterior_sk<installed version>.pkl` (see CHANGES_M4.md, WP-repro-1).


## Running everything (final submission)
Two passes, one script (`run_all.sh`); details of the experiments and decision rules are in `CHANGES_M3.md` / `CHANGES_M4.md`.
```bash
cd Code/src
export CAUSAL_CMD_JAR=/abs/path/causal-cmd-1.3.0-jar-with-dependencies.jar     # Java 17+ needed
export PETSHOP_ROOT=/abs/path/to/petshop-root-cause-analysis/dataset
python tests/run_tests.py --fast                      # 101 passed, 17 skipped without PetShop
bash run_all.sh main  <physical cores - 1>            # pass 1 (E1-E5, E7, E9, E10, E12, E13) + tables: results/aggregate/, results/m3_report/
bash run_all.sh quiet <cores / 2>                     # pass 2, machine otherwise idle: E6, E8, scalability, wide posterior + figures
```
* `run_scalability.py` (E15-lite): synthetic d = 25, 50, 100, 200, n = 1000, fixed time budget per run, variants `abic`, `abic_s` (scaled start), `spot_cal` (M3 posterior), `spot_wide`;
  report with `scalability_report.py` -> `results/scalability.{md,_f1.png,_time.png,_mem.png}`. `posterior_by_d.py` (E14): posterior quality narrow vs wide by d.
* The wide posterior (`train_posterior_wide.py`, `calibrate_floor_wide.py`) is new in M4 and was not trained at full size before submission-day; read `results/train_wide.log` and `results/floor_wide.log` before quoting anything from it.
* Do not mix M2 files (`ps_2d_*`, `syn_2c_*`) or rows with different `code_hash` in one table; `timing_clean = false` rows are not for absolute-time claims.
* Experiments not run in M4: E16-E19 (exact vs approximate inference, structure-to-inference error, PetShop prediction/representation table, effect error) and the PetShop rerun with the wide posterior (E20).
