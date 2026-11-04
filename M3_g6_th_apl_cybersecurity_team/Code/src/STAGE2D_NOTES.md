# Stage 2d notes: PetShop runs

## Status: pipeline built; PARTIAL real runs done (see RESULTS at the end)
The PetShop dataset was not available where this stage was built (not in the uploads, no network), so **no PetShop result
exists yet and none is reported here.** What exists is the complete, tested path from data to tables. Run it where the
dataset is (`git clone https://github.com/amazon-science/petshop-root-cause-analysis`, `PETSHOP_ROOT=<repo>/dataset`).

| file | purpose |
|---|---|
| `scripts/petshop_data_check.py` | step 0: n, d, latents, truth size, skew, lag-1 autocorrelation per scenario vs the posterior's training ranges; flags what is outside and prints the retraining command |
| `scripts/run_petshop.py` | scenarios x preprocessing x seeds x variants (abic, spot, spot_cal, ablations), equal-budget option, resumable JSONL |
| `scripts/run_2d_queue.sh` | the full queue below, then the summary |
| `scripts/summarize_2d.py` | tables (corrected + legacy), per scenario and across scenarios, graph size, three-number runtime |
| `pgm/petshop.py` | added `raw_centered` preprocessing, `lag1_autocorr`, `skeleton_matrix` |
| `tests/fake_petshop.py`, `tests/test_petshop_pipeline.py` | FAKE PetShop-format data; loader -> Tetrad truth -> posterior -> ABIC/SPOT/oracle -> metrics -> resume -> summary. Checks plumbing only. |

Test suite: **46 passed, 0 failed, 2 skipped** (the 2 Stage-1 tests that need the real PetShop data still skip; run them with
`PETSHOP_ROOT` set as the first check on the real data).

## Decisions (your "proceed" accepted my recommendation)
* Headline = Gaussian-rank + z-score (`gauss_rank`). Sensitivity = `raw_centered`: centered, divided by ONE global scalar so the
  variables keep their relative scales (the case where var-sortability could help ABIC) but the optimizer is well scaled.
  `none` (plain per-variable z-score of raw latency) is also available (`--prep none`).
* The skeleton posterior always sees z-scored data, whatever the learner's preprocessing.
* Seeds: PetShop is one dataset per scenario, so seeds vary the optimizer initialisation and SPOT's mask stream only. The sd over
  seeds is optimizer variance, not data variance; the summary says so. (No claim of statistical significance from seeds.)
  Across the 4 scenarios only mean difference and wins/losses are reported; with 4 observations a p-value is not meaningful.
* Floors: `spot` = 1e-3 ("as released"), `spot_cal` = 6e-3 ("calibrated", fixed in 2c before any benchmark). Both reported.
* Equal-budget runs use 120 s per run (ABIC at d~30 needs 80-175 s unconstrained in the 2a timing); SPOT's online posterior time is charged to it.
* Same code path as 2c: `run_petshop.py` imports `run_variant`/`score` from the 2c script, so synthetic and PetShop numbers are comparable.
* The oracle ablation uses the skeleton of the ground-truth PAG.

## Things to expect / check when the real numbers arrive
1. `petshop_data_check.py` first. Retrain the posterior if n, d or autocorrelation is outside its range; report the density mismatch
   (call-graph truth may have fewer edges per node than the simulator's 1.0-1.5) as a limitation if flagged.
2. 2c showed that on z-scored data both learners over-use bidirected edges (about 27 estimated vs 1-4 true). PetShop truth has only
   1-2 bidirected edges, so expect low arrowhead/tail F1 for both and SPOT's skeleton gain to be small; the `est_bidirected` column makes this visible.
3. If the posterior's AUROC against the PetShop skeleton (printed per scenario and stored as `post_auroc`) is much lower than the
   0.99 held-out value, the domain shift is the story, not the optimizer.
4. Timings are single-core; do not quote them as absolute numbers.

## PARTIAL real-data snapshot (queue still running when this zip was cut)
Real PetShop (commit as supplied, dataset/*). Data check: all four scenarios inside the posterior's training range; Stage-1 tests 4/4 pass.
`results/ps_2d_*.jsonl` hold whatever finished; `results/ps_2d_summary_partial.md` aggregates the main run only. Incomplete scenarios/seeds are NOT final.
Key early findings: posterior AUROC on PetShop 0.83 (low_traffic) / 0.93 (high_traffic) vs 0.99 simulated; calibrated floor pins 32% / 16% of true edges
(vs ~1% simulated); skeleton F1 of SPOT ~ ABIC; both over-use bidirected edges (17-41 est vs 0-1 true); SPOT faster mainly via floor pruning.


## RESULTS (partial; stopped early on request)
Data: PetShop commit 2e96f93 files, Gaussian-rank + z-score, 1 restart, original posterior pickle (sklearn 1.8.0, loads fine here).
Completed (seeds per scenario): {'low_traffic': [0, 1, 2], 'high_traffic': [0, 1, 2], 'temporal_traffic1': [0, 1]}. NOT run: temporal_traffic2, ablations, equal-budget, raw_centered, seeds 3-4.
Seeds vary only the optimizer (one dataset per scenario): no significance claims.

Posterior vs true PetShop skeleton (AUROC, share of true edges pinned by the 6e-3 floor): {'low_traffic': (0.83, 0.32), 'high_traffic': (0.931, 0.16), 'temporal_traffic1': (0.876, 0.15)}
(synthetic held-out: AUROC 0.99, ~1% pinned) -> clear domain shift; the calibrated floor caps recall here.

Scenario-level means (3 scenarios, corrected): ABIC skel/head/tail F1 0.514/0.301/0.056; SPOT-cal 0.550/0.317/0.105;
SPOT (1e-3) 0.535/0.297/0.063. SPOT-cal skeleton F1 higher in 3/3 scenarios (+0.036 mean), arrowhead 2 of 3, small differences.
Both learners estimate ~17-41 bidirected edges against 0-1 true. Runtime per run (optimization): ABIC 115 s, SPOT-cal 46 s, SPOT 58 s,
online posterior 0.3 s, offline training 203 s (one-time). Speedup is mostly from the floor pruning pairs. All runs converged.
Full tables: results/ps_2d_summary.md (.csv). Raw: results/ps_2d_main_gauss_rank.jsonl, results/petshop_data_check.json, results/run_2d_queue.log.
