# CHANGES_M4

Base: the submitted M3 code (`M3_g6_th_apl_cybersecurity_team/Code/src`). Plan: "M4 Code Update Plan" (WP0 to WP7).

## WP0 - foundations  (done; nothing here changes a result)

**WP0.1 lazy order-2 features** (`skeleton/features.py`).  `CITable` stored `P2`, a dense d x d x d x d float64 array (about 0.8 GB at d = 100, 13 GB at d = 200).
It is gone; `CITable.p2(k, l, i, j)` evaluates the same formula (same clipping and NaN handling) from the order-1 table (O(d^3)).
 * Tests: `tests/test_posterior.py::test_lazy_order2_equals_the_old_dense_tensor` (max difference < 1e-12 on 4000 random indices and one full slice), `::test_features_do_not_need_a_d4_tensor` (tracemalloc at d = 100).
 * Whole-pipeline check, shipped posterior `models/skeleton_posterior_sk1.9.1.pkl`, one simulated dataset per d (n = 600), old tree vs new tree in separate processes:
   `max |P_old - P_new| = 0.0` at d = 30 and d = 60.  Peak process memory at d = 60: 458 MB -> 169 MB.
 * New code only (predict time includes nothing but features + the cascade; peak RSS includes generating the dataset): d = 100, 150, 200 complete; d = 200 took about 10 s and 343 MB.
   (Measured once on the 2-core sandbox; re-time on the run machine.)  The posterior is therefore no longer the memory wall for the scalability sweep; its quality at d > 40 is a separate question (WP1).

**WP0.2 environment and resource stamps** (`pgm/envinfo.py`, `scripts/run_queue_parallel.py`).  Every row written by the queue now carries
`peak_rss_mb` (peak of that job: VmHWM reset through /proc/self/clear_refs; `rss_scope` says `job` or `process_lifetime`), `rss_start_mb`, `job_cpu_s`, `job_wall_s`,
`load1_start`, `load1_end`, `py_version`, `numpy_version`, `scipy_version`, `sklearn_version`, `cpu_model`, `logical_cpus`, `host`, `blas_threads`, `code_hash`, and `timing_clean`
(quiet group AND the 1-minute load average at the end of the job <= workers + 1).  `load1` lags by a minute, so `timing_clean` is a screen, not a proof.
 * Tests: `tests/test_envinfo.py`; the end-to-end queue test checks the new fields.

**WP0.3 anytime hook** (`pgm/dcd.py`).  `DCDConfig.checkpoint_cb(elapsed_s, W1, W2, h, outer, restart)` is called once per outer iteration with copies of the raw iterates; default `None`.
 * Tests: `tests/test_checkpoint_hook.py` (a seeded run with and without the hook gives identical weights, h and n_fev; one call per outer iteration; a callback that overwrites its copy cannot corrupt the run).

**WP0.4 quiet timing mode.**  The queue from M3 already had the quiet group and the E6 refusal rules; WP0.2 adds the measurements that make a "clean" claim checkable. No further runner change.

**WP0.5 hygiene.**
 * `scripts/aggregate_m3.py` in the M3 submission contained the full 940-line aggregator with every line commented out and the short script appended below it, so `tests/test_aggregate.py` failed (3 tests, also on the untouched M3 tree).
   Split back into `scripts/aggregate_m3.py` (full aggregator, uncommented; its tests pass) and `scripts/aggregate_quick.py` (the short script that produced the tables of the submitted M3 report).
 * Removed `results.zip`, `causal-cmd.log`, `results/_wrong_pickle/`, `results/_queue_tmp/`, `__pycache__`; added `.gitignore`; README status lines corrected.

## Not done yet
WP1 (wide posterior), WP2 (scalability), WP3 to WP6 (inference suite), WP7 (runner and aggregator for M4). Nothing in the M4 plan has produced a result yet.

## WP-repro-1 - a stale posterior can no longer be used silently  (done; changes no result when the right pickle is used)

**Problem found on the M3 run.** `--model` defaulted to `models/skeleton_posterior.pkl`, the M2 file written by scikit-learn 1.8.0. Under 1.9.1 it unpickled with only an
`InconsistentVersionWarning` ("might lead to breaking code or invalid results"), so `load_or_retrain` never fell back and the queue ran on the stale model. It was caught by reading the log.

**Fix** (`skeleton/posterior.py`).
 * `SkeletonPosterior.load` is strict: that warning is turned into `PosteriorIncompatible`, and a pickle whose recorded `info["sklearn_version"]` differs from the installed version is also refused.
   `load_or_retrain` therefore switches to `skeleton_posterior_sk<version>.pkl` (shipped for 1.9.1) or retrains and caches it. The loaded object records `source_path`.
 * `default_model_path(root)` returns `models/skeleton_posterior_sk<installed version>.pkl` if it exists, else the base file. All scripts that take `--model` (and `calibrate_floor.py`) use it as the default:
   `run_queue_parallel.py`, `run_petshop.py`, `run_bootstrap.py`, `run_rca.py`, `spot_vs_abic_synthetic.py`, `posterior_quality.py`.
 * Tests (`tests/test_posterior.py`): the matching pickle loads with warnings turned into errors; a pickle stamped with another version is refused; the M2 file is refused on any version except 1.8.0;
   `default_model_path` and `load_or_retrain` pick the matching pickle with no warning.
 * Checked by hand: `load("models/skeleton_posterior.pkl")` now raises `PosteriorIncompatible`; `load_or_retrain` on the same path returns the `_sk1.9.1` model.

**Not done / limits.** Result rows still do not record which posterior file was used (only `source_path` on the loaded object); logging it per row is a small addition to the queue stamps. The M3 results
were produced with `--model models/skeleton_posterior_sk1.9.1.pkl` after the restart, so they are unaffected.

## WP-repro-2 - probability quality and interval estimates in the tables  (done in code; nothing run on real data)

**Correction to the review.** Brier was already in the M3 E7 table (`aggregate_quick.py` kept the column). It was missing only from the full aggregator `aggregate_m3.py`, and from the inference scores.

**What was wrong and is fixed**
 * `aggregate_quick.py` (the script behind the submitted M3 tables): `paired()` looked for CI keys named `ci_lo/ci_hi`, but `paired_summary` returns `ci_mean_lo/hi` and `ci_pct_lo/hi`, so the paired tables never had a CI.
   They now have `ci95_lo, ci95_hi` (t interval of the mean paired difference), `ci95_pct_lo/hi`, `ci_excludes_0` and `decision` (the plan's rule: paired t p < 0.05 and Wilcoxon p < 0.05 and the t interval excludes 0).
   Every mean ± SD table also gets a `<metric> CI95` column (t interval of the mean over the runs in the group; `n/a` for one run).
 * `pgm/inference.predictive_scores` now returns, besides `logpdf` (NLL), `rmse`, `cond_sd`: `coverage90`, `coverage50` (calibrated: 0.90 / 0.50), `z2` (mean squared standardized error, 1 = calibrated variance),
   `pit_ks` (KS distance of the PIT values from uniform) and `crps` (closed form for the Gaussian predictive). `validate_inference.py` (`evaluate`, `rolling_rows`) carries them through the query-B averages.
 * `aggregate_m3.py`: table T5 (inference) adds NLL with 95 % CI and the five calibration columns when the result file has them (files from before this change still aggregate); table T7 adds Brier with a bootstrap CI
   (`posterior_quality.py` now bootstraps Brier too).
 * Tests (`tests/test_inference_calibration.py`, 5): a correctly specified model has coverage 0.90 +- 0.012, z2 about 1, small PIT-KS; the same mean with half the variance is caught (coverage down, z2 > 1.8, worse NLL,
   identical RMSE) and double variance too; the closed-form CRPS equals numerical integration; `rolling_rows` carries the columns; the aggregators show Brier, CIs and coverage; `aggregate_quick`'s CI equals the textbook t interval.
 * Smoke test: E8 through the queue on fake PetShop data (1 scenario, 1 graph seed, 5 splits, 80 rows), then `aggregate_m3.py`: T5 shows NLL, its CI, coverage, z2, PIT-KS and CRPS per model.

**Not done.** E8 has not been run on the real data (that needs the PetShop files and the quiet pass: `python scripts/run_queue_parallel.py --group quiet ...` includes E8, about 0.3 core-hours by the M3 estimate).
Its Gaussian conditionals use the model's own variance; coverage on heavy-tailed latency data is the open question the real run answers. The aggregators were not re-run on the real M3 result files.

## WP1-lite + scalability (11 Nov, cut-down M4)
Added: `skeleton/wide.py` (d 25-200 training/calibration/test sets, disjoint seed ranges), `max_neg_per_graph` in `PosteriorConfig` (default None = M3 behaviour),
`scripts/train_posterior_wide.py`, `calibrate_floor_wide.py`, `posterior_by_d.py` (E14), `run_scalability.py` + `scalability_report.py` (E15-lite),
`run_all.sh` (the single entry point: main and quiet passes), `run_variant(..., floor=)` override. NOT done: E16-E19, queue wiring of E20 (PetShop rerun with the wide model).
Honest status: the wide posterior was NOT trained in the sandbox (the container restarted twice); the training and floor calibration are untested at full size.
Tested: tests/test_wide.py (seeds, subsampling, capped fit), smoke training at 10 % size, smoke calibration/E14 scripts at tiny size, single-run pilots of ABIC and SPOT at d = 50-200.
Pilot finding (single graphs, not a result): ABIC with the default start W1 ~ U(-0.5, 0.5) diverges immediately at d = 100 (acyclicity term overflows, 0 edges, h = 3154);
with the start scaled by sqrt(25/d) it runs. Hence the variants `abic` (default) and `abic_s` (scaled). SPOT runs at d = 100 and 200 with default settings. The optimizer's final
model-selection step is not covered by the time budget (about 20-40 s extra at d = 75), so wall time exceeds the budget slightly; `t_total` records the real time.

## Fix: PetShop root and the quiet pass of run_all.sh  (12 Nov)
Two bugs reported from a local run.
 * **`--group quiet`, `--experiments E6 E8` and therefore `run_all.sh quiet` always errored.** The queue's E6 safety rule was `set(experiments) != {"E6"}` ("E6 must run alone"), which also rejects E8 -
   the other member of the quiet group. The rule is now `not set(experiments) <= QUIET`: E6 may share a call with E8, not with main-pass experiments (the original intent). Introduced in M3 item 8;
   the M3 tests only ran E6 alone, so it was not caught (the M3 quiet pass was never run through `--group quiet`).
 * **"Set PETSHOP_ROOT" although it was set.** `pgm.petshop.default_root` accepted a path only if `os.path.isdir` held, and then said nothing about why it rejected one. New `resolve_root`: expands `~` and `$VARS`,
   makes relative paths absolute, accepts either the `dataset` folder or the cloned repository, and when it fails lists every candidate it tried with the reason (not set / does not exist / no `low_traffic/graph.csv`).
   `load_petshop(root=...)` uses it too. Typical real causes: `PETSHOP_ROOT=...` set without `export`, an unexpanded `~`, a path one level too high or too low.
 * `run_all.sh`: absolute paths and `cd` to its own folder (it can be started from anywhere), a preflight that checks the dataset, the jar and the scikit-learn version and stops with the reason before any work (exit 3 / 4),
   `export`s the resolved root, clamps quiet-pass workers to cpu_count / 2, runs E6 and E8 as separate queue calls, and appends every step's exit code to `results/run_all_steps.log`
   (`grep -v 'exit=0' results/run_all_steps.log` lists failures).
 * Tests: `tests/test_root_and_queue_rules.py` (4): root forms and error texts, `--group quiet` / `E6 E8` accepted and `E3 E6` / `--group all` refused, the preflight exit codes through the real script.
 * Not run: the full quiet pass (scalability, wide posterior) was NOT run end to end here; only the queue call and the options of the scripts it launches were checked (`--help`). If a later step of the quiet pass fails,
   `results/run_all_steps.log` says which and its own log in `results/` says why.
