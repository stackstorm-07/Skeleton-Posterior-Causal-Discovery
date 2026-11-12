# M3 code changes (running log)

Base = the M2 submission (`Code.zip`). Tests: `python tests/run_tests.py --fast` (needs `CAUSAL_CMD_JAR` for the Tetrad tests; they skip otherwise).

## Step 0 - baseline merge (done)
M2 code + the earlier M3 fixes: `pgm/inference.py` (+7 tests), RCA ancestor modes / correlation baseline / graph seeds,
`SkeletonPosterior.load_or_retrain`, `requirements.txt`, `scripts/validate_inference.py`, stage notes. 56 tests pass.

## Step 1 - structural Hamming distance (done)
* `pgm/metrics.py`: `shd_pag(est, true)` and `shd_flat(est, true, prefix)`.
  `shd_skeleton` (extra + missing adjacencies), `shd_pag` (pairs whose edge differs in any way; the usual PAG-SHD),
  `shd_marks` (skeleton errors + wrong end marks), plus `extra`, `missing`, `shd_pag_norm` (SHD / #true edges).
* `scripts/spot_vs_abic_synthetic.py::score` now adds `c_shd_*` columns, so every synthetic and PetShop run line
  (run_petshop.py reuses `score`) carries SHD from now on. Old result files simply lack these columns.
* `tests/test_stage1.py::test_shd_pag` (hand-checked numbers). 57 tests pass.
* Note: `third_party/Spot_original/learn/bin/` in `Code.zip` is empty. Put `causal-cmd-1.3.0-jar-with-dependencies.jar`
  there or set `CAUSAL_CMD_JAR`.

## Step 2 - probability quality of the skeleton posterior (done; deliberately small)
Decision: ONE headline metric, **log-loss** (a proper scoring rule: it punishes both wrong and over-confident probabilities),
with **ECE + reliability diagram** as its companion plot (same bins, no extra computation). Brier and interval coverage are NOT
reported in M3. `posterior_metrics` already computed log-loss (as `kl`), Brier and ECE for M2; nothing new had to be built for the numbers.
* `skeleton/posterior.py`: `posterior_metrics` now also returns `log_loss`; new `reliability_curve(P, skeleton)` and
  `pool_reliability(list_of_(p, y))` give the per-bin data for the reliability diagram (and the pooled ECE).
* `tests/test_posterior.py::test_log_loss_and_reliability`. 58 tests pass.
* Inference quality is reported as held-out NLL (log density), already produced by `scripts/validate_inference.py`.

## Step 3 - moving-block bootstrap, paired ABIC vs SPOT (done)
* `pgm/stats.py`: `block_bootstrap_indices`, `default_block_length` (n^(1/3)), `paired_summary(a, b)` -> mean/SD of the paired
  difference, t-CI of the mean, percentile interval of the differences, wins/losses/ties, paired-t and Wilcoxon p-values.
* `scripts/run_bootstrap.py` (new). Resamples the raw PetShop rows in blocks, then preprocesses, runs the posterior and the
  optimizer on that resample for EVERY variant with the same optimizer seed (= resample number b), so differences are paired.
  Resample b depends only on (--boot-seed, scenario, b): reproducible, independent of worker count and order.
  - run (example; use physical cores - 1 for --workers):
    `python scripts/run_bootstrap.py --scenarios low_traffic high_traffic temporal_traffic1 --B 20 --workers 7 --variants abic spot_cal --out results/boot_main.jsonl`
  - summary of a finished or partial file (also writes `<file>_paired.csv`): `python scripts/run_bootstrap.py --summarize results/boot_main.jsonl`
  - one JSON line per (scenario, b, variant); resumable; progress line with ETA; BLAS threads forced to 1 per job.
  - workers use the 'spawn' start method (a forked child deadlocked after OpenMP had run in the parent during the test suite).
  - columns: F1, SHD, edge counts, times, and the posterior's AUROC / log-loss / ECE on the resample.
* `tests/test_stats.py`: block structure, autocorrelation kept by blocks (an i.i.d. bootstrap destroys it), exact Wilcoxon minimum p
  for 6 pairs (0.03125), end-to-end run on fake PetShop with 2 workers, resume, reproducibility.
  Fast suite with the Tetrad jar available: 67 passed, 0 failed, 3 skipped (the skipped ones need the real PetShop data).
* Sandbox note: scikit-learn here is 1.9.1 and the shipped posterior pickle needs 1.8.0, so this environment retrained into
  `models/skeleton_posterior_sk1.9.1.pkl` (not in the zip). On your machine use scikit-learn 1.8.0 (requirements.txt) or let it retrain once (~4 min).
* NOT yet run on real PetShop data: the 20-resample experiment is yours to launch (about 2.7 core-hours; ~25 min with 7 workers).

## Step 4 - robustness options for the synthetic benchmark (done; E1 / E2)
`scripts/spot_vs_abic_synthetic.py` and `skeleton/simulator.py`. All options change only what the observer sees; the truth
is always the clean graph. **Defaults are bit-identical to M2** (test `test_defaults_are_bit_identical_to_m2` compares against a
verbatim copy of the M2 generator), so the existing `results/syn_2c_*` files stay valid.
* `--density E` exactly E edges per node (M2 default: uniform 1.0-1.5). The graph changes by definition.
* `--ar RHO` AR(1) structural errors over the rows (`simulate_linear_gaussian(..., ar_rho)`). Same graph and same SEM parameters
  at every RHO for a given seed, so the AR curve is paired across levels.
* `--snr S` measurement noise, variance = var(column) / S, added to every column (`corrupt_data`). Own rng stream (20_000 + seed).
* `--missing P` P of the cells removed completely at random, then mean-imputed per column (`corrupt_data`). Own stream (30_000 + seed).
  Noise is applied before removal. The posterior and both learners see the same corrupted, z-scored matrix.
* `--design e1|e2` expands to the plan's job lists (`design_conditions`); `--graphs K` graphs per cell (cut rule: 6 for E1);
  `--list-jobs` prints the conditions and the job count and runs nothing.
  - e1: n in {200, 500, 1000, 2000} x 10 graphs x {abic, spot_cal} (+ spot at n = 1000) = 90 jobs.
  - e2: n = 1000, 8 graphs per cell, one factor at a time: density 1/2/3, SNR 10/3/1 (our choice of levels), AR 0.3/0.6, missing 5/20 %
    x {abic, spot_cal} = 160 jobs. The unperturbed cell (AR 0, no noise, no missing, M2 density) is the n = 1000 cell of e1 and is not repeated.
* Result lines now carry `density, snr, ar, missing, design`. The resume key is (seed, variant, d, n, data, density, snr, ar, missing),
  so one file can hold a whole sweep; M2 lines without the new fields resume as before (`test_resume_key_is_backward_compatible_with_m2_lines`).
* `tests/test_synthetic_options.py`: 10 tests (identity with M2, graph kept across AR/noise levels, achieved SNR, lag-1 autocorrelation,
  MCAR fraction and mean imputation, job counts of e1/e2, resume key, end-to-end run of five conditions with resume).
  Full fast suite here: 75 passed, 0 failed, 5 skipped (the skipped ones need real PetShop data).
* Run (sequential; the parallel pool is item 8, which will call these same conditions):
  `python scripts/spot_vs_abic_synthetic.py --design e1 --out results/syn_m3_e1.jsonl --tag E1`
  `python scripts/spot_vs_abic_synthetic.py --design e2 --out results/syn_m3_e2.jsonl --tag E2`
  Single condition, e.g. `--n 1000 --ar 0.3 --seeds 0 1 2 3 --variants abic spot_cal --out ...`. Set OMP/OPENBLAS/MKL_NUM_THREADS=1.
* Not run on the real design here (about 3.5 core-hours for E1 + E2 by the plan's estimate). Only a d = 8 smoke run of five
  conditions was executed, as part of the test.
* Caveats: at n = 200 and d = 8 one smoke graph shows a 0.52 -> 0.95 skeleton-F1 jump for SPOT; it is one graph, not a result.
  Measurement noise breaks the model's assumptions (errors-in-variables), so lower accuracy for both learners there is expected, not a bug.
  Density 3 is much denser than the posterior's training range (1.0-1.5 edges per node), so SPOT may suffer there.

## Step 5 - directed-only learner (E10) and FCI / RFCI baselines (E9) (done)
### E10: directed-only learner
* `DCDConfig.directed_only` (default False = M2 behaviour, tested bit-for-bit). When True, `Problem` pins the whole bidirected block of theta
  (box bounds (0, 0), `free = False`, so the gradient there is exactly 0 and the initial value is 0). Same loss, penalty, solver, thresholds, restarts.
  With W2 = 0 the ancestrality term vanishes and the learner returns a DAG; it is still scored through MAG -> PAG like every other learner.
* `SpotMask(directed_only=True)` (and `run_spot`, via `cfg.directed_only`) leaves the pinned coordinates out of its proposal counters, so
  `mask_summary` / `kept_fraction` describe the coordinates that can actually move. The random stream is unchanged.
* Variants: every variant has a twin `<variant>_dir` (`abic_dir`, `spot_cal_dir`, `spot_dir`, ...) in `spot_vs_abic_synthetic.ALL_VARIANTS`, so
  `run_petshop.py`, `run_bootstrap.py` and `run_rca.py` accept them with no further change. `run_variant` never mutates the caller's config.
  E10 on PetShop: `python scripts/run_petshop.py --seeds 0 1 2 --variants abic_dir spot_cal_dir --out results/ps_e10.jsonl`
  (compare with the abic / spot_cal rows of E3 for the same seeds).
* Interpretation caution: forcing W2 = 0 removes the model's way to explain confounding, so the directed edges must absorb it. If
  directed-only is NOT worse than the ADMG learners on arrowhead/tail, that supports "bidirected over-use explains the low scores";
  if it is worse, the ADMG is doing useful work. Neither reading is settled by the single smoke graph below.

### E9: FCI / RFCI
* `pgm/fci.py::run_tetrad(X, names, algorithm='fci'|'rfci', alpha, depth, timeout)` runs `causal-cmd` (Fisher-z, one thread) and returns a PAG
  over `names` (node names are sent to the jar as X1..Xd and mapped back). Failures and timeouts raise `TetradError`; NaN input is rejected.
  `t_wall` is the whole java subprocess (JVM start-up about 0.9 s included).
* `scripts/run_fci.py`: `--source petshop` (same data and preprocessing as `run_petshop.py`; one row per scenario x algorithm x alpha, no seeds
  because FCI/RFCI are deterministic) and `--source synthetic` (same graphs and data as `spot_vs_abic_synthetic.py`: same seeds, same
  `--density/--snr/--ar/--missing`, `--design e1|e2`). alpha in {0.01, 0.05} by default. Resumable; failed jobs are written with `failed: true` and retried.
  Scoring is `score_pag` (new, factored out of `score`): identical skeleton/arrowhead/tail F1, legacy columns and SHD as the learners.
  `python scripts/run_fci.py --source petshop --out results/fci_petshop.jsonl`
  `python scripts/run_fci.py --source synthetic --design e1 --out results/fci_syn_e1.jsonl`
* Tests: `tests/test_directed_and_fci.py` (8): pinned block and zero gradient, flag-off identity with M2, DAG output, mask counters, `_dir` variants,
  FCI/RFCI find a collider (A -> C <- B) with awkward node names, explicit errors incl. timeout, synthetic runner (scoring, resume, failed-retry,
  same truth as the ADMG benchmark), PetShop mode on fake data. Full fast suite: 83 passed, 0 failed, 5 skipped (need the real PetShop data).
* Not run on the real designs. Smoke results on ONE or TWO d = 25, n = 1000 z-scored graphs (not results; numbers in `/scratchpad` only):
  FCI/RFCI skeleton F1 0.84-0.92 in about 1.4 s per run; ABIC 0.585, SPOT-cal 0.719 on graph 0 in 57-66 s; directed-only twins 0.571 / 0.679.
  If this pattern survives E1/E3, FCI/RFCI is a strong baseline that the ADMG learners do not beat on skeleton accuracy in this setting, and the report
  should say so (the plan already allows this: "the text changes, not the data"). Known caveat: Tetrad's Fisher-z FCI scores much better on z-scored
  linear-Gaussian data than on PetShop's skewed, autocorrelated series, so the PetShop numbers are what matter.
* Arrowhead/tail columns of FCI use PAGs with circle marks; the true PAG also has circles, scored the same way as for the learners.

## Step 7 - posterior for scikit-learn 1.9.1 (done)
* `models/skeleton_posterior_sk1.9.1.pkl` is now IN the package (trained here with the M2 recipe and seeds: `scripts/train_posterior.py`,
  400 train / 60 held-out graphs, d 25-40, AR 0/0.3/0.6; training log `results/posterior_training_sk1.9.1.json`). The M2 pickle
  (`models/skeleton_posterior.pkl`, written with scikit-learn 1.8.0) is kept unchanged. `SkeletonPosterior.load_or_retrain("models/skeleton_posterior.pkl")`
  (what every script calls) loads the M2 file if it can, otherwise the `_sk<version>` file, otherwise retrains once and caches it.
  With scikit-learn 1.9.1 installed nothing retrains.
* `requirements.txt` now pins `scikit-learn==1.9.1` (the plan's environment); with 1.8.0 the M2 file loads, with any other version it retrains once.
* Held-out quality is the same as M2 (60 fresh simulated graphs; M2 -> sk1.9.1): AUROC 0.9889 -> 0.9888, AUPRC 0.9463 -> 0.9465,
  log-loss 0.0568 -> 0.0573, Brier 0.0151 -> 0.0152, ECE 0.0127 -> 0.0126, recall of true edges at p >= 0.05: 0.957 -> 0.956.
* Floor re-check (`results/floor_calibration_sk1.9.1.json`, same rule and the same 40 held-out graphs, seed 777): the chosen floor is still
  6e-3. It pins 1.00% of true edges (0.93% with the M2 model), i.e. exactly at the 1% rule, so `CALIBRATED_FLOOR` is left at 6e-3. The released 1e-3 pins 0.12%.
* `tests/test_posterior.py::test_shipped_posterior_loads_without_retraining`. Full fast suite below.
* **Consequence for the experiments (read before launching):** it is a different model, not the same file. The 24 PetShop runs of M2
  (`results/ps_2d_main_gauss_rank.jsonl`) used the M2 posterior. Do not merge them with M3 runs that use the 1.9.1 posterior inside one
  table: either re-run the M2 seeds under E3 (cheap, about 0.8 core-hours for everything), or run all of M3 with scikit-learn 1.8.0 and the M2 file.
  The plan's choice (1.9.1 everywhere, E3 re-run in full) is the clean one. The numbers in the M2 report tables stay valid as M2 results.
* Not checked here: how the new model scores on real PetShop (needs the data); E7 does that, with the model the experiments use.

## Step 8 - parallel queue `scripts/run_queue_parallel.py` (done), plus the pieces it needed
**Run it** (details and every rule in the script's docstring):
```
export PETSHOP_ROOT=<dataset>  CAUSAL_CMD_JAR=<jar>
python scripts/run_queue_parallel.py --group all --list --workers 7      # plan: jobs, done, todo, estimated core-hours (runs nothing)
python scripts/run_queue_parallel.py --group main  --workers 7           # E3 E4 E5 E1 E9 E10 E2 E12 E7 E13   (add --cut for the plan's cut rule)
python scripts/run_queue_parallel.py --group quiet --workers 3           # then, alone: E6 and E8
wc -l results/*.jsonl                                                    # progress; the same command after a crash simply continues
```
* One process pool ('spawn'), BLAS/OpenMP threads forced to 1 before numpy loads. Only the parent writes: a worker returns the rows of its job and the parent
  appends them with ONE `write()` (O_APPEND + fsync), so no races and no half-written jobs. Every row gets `job_id, exp, job_wall, queue_workers,
  queue_cores, timing_quiet`. Resume = skip `job_id`s already in the file. A job that raises goes to `results/queue_failures.jsonl` (traceback), is NOT marked
  done and is retried at the next start; the queue goes on. Ctrl-C keeps finished jobs. Progress line per job with an ETA from the observed time per job kind.
* E6 is refused unless it runs alone with `--workers <= cpu_count // 2` (`--force` overrides). `--group all` only works with `--list`. `timing_quiet`
  is True only for a quiet-group run with at most half the cores busy; filter on it before comparing times from E8.
* Job counts (default sizes, tested): E3 60, E4 48, E5 120, E1 40, E2 80, E6 48, E7 5, E8 4, E9 44 (4 PetShop + 40 synthetic graphs), E10 24, E12 12, E13 24.
  Estimated core-hours still to do for everything: about 15 (plan: 12); `--cut` gives about 12.5. The estimates are rough (M2 timings, no data here).
* Result files (in `results/`): ps_m3_e3/e4/e6/e10/e13.jsonl (same schema as `ps_2d_*`, so `summarize_2d.py` reads them), syn_m3_e1/e2.jsonl (same as `syn_2c_*`),
  boot_main.jsonl (`run_bootstrap.py --summarize` reads it), fci_petshop.jsonl, fci_syn_e1.jsonl, post_quality_m3.jsonl, inference_m3.jsonl, rca_m3.jsonl.
* Refactors so that the queue runs exactly the code the stand-alone scripts run (their behaviour and tests are unchanged): `spot_vs_abic_synthetic.run_graph_job`,
  `run_petshop.petshop_context / petshop_row`, `run_fci.fci_petshop_rows / fci_synthetic_rows`; `run_rca.get_graph` now writes its graph cache atomically
  (E12 and E8 may learn the same graph at the same time).
* NEW code that the plan listed as experiments but no earlier step covered:
  - **E7** `scripts/posterior_quality.py`: log-loss (headline), ECE, reliability bins, AUROC/AUPRC, share of true edges below the 1e-3 and 6e-3 floors, for 60 fresh
    simulated graphs (seed 4242) and for each PetShop scenario, for the raw posterior and for isotonic / Platt maps fitted on 40 SIMULATED graphs (seed 9001) only.
  - **E8** `validate_inference.rolling_rows`: 5 expanding-window, time-ordered splits (train on the first 50/60/70/80/90 % of the rows, test on the next 10 %, the last
    split on the final 10 %; transform fitted on the training rows only); models marginal, saturated, and ADMG/DAG from truth, ABIC, SPOT-cal; queries A (the sink given its
    true ancestors) and B (every service given all others); held-out log density and RMSE; `t_fit` and `t_graph_learn` for the error-vs-time plot. Graph seed 0 only.
    The "5 rolling splits" wording of the plan is open to interpretation; this is mine. 90 % interval coverage is not computed (dropped in item 2).
* Tests `tests/test_queue.py` (6): thread env, job counts of the plan, E6 safety rules, an end-to-end run of E3 E5 E9 E10 E7 E12 E1 on FAKE PetShop with two workers
  (rows, schema, meta, resume), the quiet group (E8 + E6 flags, time-ordered windows), a failing job (logged, not recorded, retried). Full fast suite: 90 passed, 0 failed, 5 skipped.
* **Not run on the real data, and not run for real size.** E1/E2/E3 ... have never been executed at the plan's sizes. The first thing to do on your machine:
  `--list`, then `--max-jobs 2` for E3 to see one real job finish and read its row, then the full pass.
* Things to know before launching: (1) E3 re-runs all 60 PetShop jobs with the sk1.9.1 posterior; the 24 M2 runs used the M2 posterior and are not mixed in.
  (2) E6's estimate (1.2 core-hours) is above the plan's 0.5 because every job runs to its 60 s / 120 s budget; it is a wall-clock budget, not a CPU one.
  (3) E12 and E8 share `results/rca_graphs/`; if E12 ran first, E8 reuses its graphs. (4) E11 (error analysis) is not a run: it reads these files.

## Step 6: `scripts/aggregate_m3.py` (plan item 6) and statistics helpers

```bash
python scripts/aggregate_m3.py --results-dir results --petshop-root "$PETSHOP_ROOT"   # -> results/aggregate/   (CAUSAL_CMD_JAR must be set: E11 loads the PetShop truth)
```
* Reads every result file (de-duplicated on the job key, so a re-run's duplicates count once; sha256 and row counts go into `manifest.json`) and writes, with nothing typed by hand:
  tables T0 completeness, T1 algorithm KPI, T1a/b synthetic (sample size, robustness), T2/T2b-g PetShop (per scenario, wins/losses/ties, bootstrap, equal budget, directed-only,
  preprocessing, FCI/RFCI), T3 ablation and all paired tests (with Holm-adjusted p), T4 ANOVA, T5 inference, T6/T6b RCA, T7 posterior quality, T8 error analysis (E11), T9 scorecard
  (as CSV, Markdown and LaTeX each; the Markdown of everything in `tables.md`), figures F1-F7 (PNG and PDF), `scorecard.md/csv`.
* Decision rule (the plan's, made stricter in one place): "supported" only if paired t-test p < 0.05 **and** Wilcoxon p < 0.05 **and** the 95 % CI of the paired difference excludes 0
  (t interval; percentile interval for bootstrap resamples). The Wilcoxon requirement is mine: it cannot reach 0.05 with fewer than 6 units, so such comparisons are never
  "supported". Remove it in `decide()` if you want the plan's wording only. No multiplicity correction in the decision; Holm-adjusted p-values are listed.
* Paired units: graph (synthetic), (scenario, seed) (PetShop optimizer seeds), (scenario, resample) (bootstrap), case (RCA). PetShop seed spread is optimizer variance only, so those rows are labelled descriptive; E5 is the data-variance test.
* The scorecard rules for H1-H5 are written in `build_scorecard` (read them before quoting a verdict): e.g. H4 counts "well calibrated" as ECE upper CI bound <= 0.02, a threshold I chose; H5's
  non-inferiority margin against the saturated Gaussian is -0.05 nats/row, also mine; H3 compares paired log optimizer time against ABIC. A hypothesis whose data are missing says "not run / incomplete".
* A missing or unfinished experiment is skipped; T0 lists planned jobs (the queue's default sizes), finished jobs, rows actually in the files, and logged failures. If you ran with `--cut` or other sizes, pass `--cut` or read `rows_in_files`.
* `pgm/stats.py`: `two_way_anova` (Type III, sum-to-zero), `bootstrap_mean_ci`, `bootstrap_paired_ci`, `welch_diff`, `holm`. `posterior_quality.py` now adds bootstrap CIs (over graphs for simulated, over rows for PetShop) and an over-confidence score.
* Tests `tests/test_aggregate.py` (4): decision rule, designed real/null effects, duplicates and missing files, stats helpers.
* **Verified only on fake data** (queue output on fake PetShop plus designed rows). Not run on real results: figure layout, column widths, and every verdict need a look once real files exist. On fake data the CIs of F1 can exceed 1 (few units); with real unit counts they are tighter.
* Bugs found in smoke testing and fixed: a key clash (`n`) in the paired-test rows, E11 swallowing a missing-jar error (now written to `manifest.json` notes), PAG edge access in E11.
