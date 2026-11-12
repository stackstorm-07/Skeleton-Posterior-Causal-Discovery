# Code – SPOT vs ABIC (CSE 516, team g6)

Everything in the report comes from result files written by `scripts/run_queue_parallel.py` (one JSON line per finished job) and is turned into tables and figures by `scripts/aggregate_m3.py`
(the full aggregator, with the hypothesis scorecard, a manifest and a completeness table) and `scripts/aggregate_quick.py` (the short script behind the M3 report's tables). No number in a table is typed by hand.
The code is under `src/`; run every command below from `Code/src`.

## 1. Layout

| Path | Contents |
|---|---|
| `pgm/` | graphs (ADMG, MAG, PAG, latent projection), linear-Gaussian ADMG model and RICF (`admg_model.py`, `ricf.py`), the shared differentiable optimizer (`dcd.py`), ABIC (`abic.py`), SPOT (`spot.py`), inference (`inference.py`), FCI/RFCI wrapper (`fci.py`), root-cause ranking (`rca.py`), PetShop loader (`petshop.py`), metrics and statistics (`metrics.py`, `stats.py`), run stamps (`envinfo.py`) |
| `skeleton/` | the skeleton posterior: simulator (`simulator.py`), Fisher-z features (`features.py`), gradient-boosting cascade (`posterior.py`), wide variant (`wide.py`) |
| `scripts/` | experiment runners, the parallel queue, aggregators, training and calibration scripts (section 6) |
| `models/` | trained posteriors: `skeleton_posterior_sk1.9.1.pkl` (use this one) and the M2 file `skeleton_posterior.pkl` (scikit-learn 1.8.0, refused by the strict loader) |
| `tests/` | unit and integration tests (`run_tests.py`) |
| `third_party/Spot_original/` | the original SPOT release (Apache-2.0); source of the Tetrad jar |
| `run_all.sh` | one entry point for every result (section 4) |
| `results/` | result files, logs, `aggregate/` tables and figures |
| `CHANGES_M3.md`, `CHANGES_M4.md`, `STAGE*_NOTES.md`, `ADMG_JUSTIFICATION.md` | design notes and change logs |

## 2. Environment

The submitted results were produced on an x86_64 Linux machine (Arch Linux, but any distribution with python 3.13 and Java >= 17 should work) with:

| Item | Version |
|---|---|
| Python | 3.14.10 (3.13 also works) |
| NumPy / SciPy | 2.5.4 / 1.18.1 |
| scikit-learn | **1.9.1, exact** (see section 8) |
| Java | 17 or newer, on the `PATH` |
| Tetrad `causal-cmd` | `causal-cmd-1.3.0-jar-with-dependencies.jar` |
| Machine | Intel Core Ultra 7 155H, 22 logical CPUs; 11 workers (main pass), 6 workers (quiet pass) |

`requirements.txt` lists the Python packages. The jar ships in the original SPOT repository (`microsoft/reliableai`, `causal-kit/Spot/learn/bin/`); put it in `third_party/Spot_original/learn/bin/` or point `CAUSAL_CMD_JAR` at it.
The PetShop dataset is described in `../Data/README.md`; set `PETSHOP_ROOT` to the folder that holds `low_traffic/`, `high_traffic/`, `temporal_traffic1/` and `temporal_traffic2/`.
Memory: 2-3 GB per worker is a safe budget. No GPU and no autodiff library are used.

```bash
cd Code/src
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export CAUSAL_CMD_JAR=/abs/path/causal-cmd-1.3.0-jar-with-dependencies.jar
export PETSHOP_ROOT=/abs/path/to/petshop-root-cause-analysis/dataset
java -version && python -c "import sklearn; print(sklearn.__version__)"          # expect 1.9.1
python -c "from pgm.petshop import load_petshop; ps = load_petshop('low_traffic'); print(len(ps.names), ps.data.shape)"   # loads the data and runs the jar
```

## 3. Check the installation

```bash
python tests/run_tests.py --fast        # no failures; the tests that need PetShop are skipped unless PETSHOP_ROOT is set
python scripts/gradcheck.py             # analytic vs finite-difference gradients
```

At the last full run of the fast suite: 109 passed, 0 failed, 5 skipped. Four tests added afterwards (`tests/test_root_and_queue_rules.py`) pass on their own; the full suite was not re-run after them.

## 4. Run everything

```bash
bash run_all.sh main  <physical cores - 1>     # pass 1: E1-E5, E7, E9, E10, E12, E13, then the tables (results/aggregate/, results/m3_report/)
bash run_all.sh quiet <cores / 2>              # pass 2, machine otherwise idle: E6, E8, scalability sweep, wide-posterior steps, figures
```

* Both passes are resumable: run the same command again after a crash or Ctrl-C. Finished jobs are skipped; failed jobs are retried and logged in `results/queue_failures.jsonl`. Each step's exit code is appended to `results/run_all_steps.log`.
* `run_all.sh` checks the PetShop root (exit code 3), the jar (exit code 4) and the scikit-learn version before it starts, and says why it stops.
* The quiet pass clamps workers to half the cores because E6 (equal wall-clock budget) is only meaningful on an idle machine. To skip the expensive equal-budget experiment: `QUIET_EXPS="E8" bash run_all.sh quiet 3`.
* Never start a second queue while one is running, and never run the quiet pass during the main pass.
* Plan without running anything: `python scripts/run_queue_parallel.py --group all --list` (jobs per experiment, how many are done, estimated core-hours). Smaller run: add `--cut`. Smoke test: `--max-jobs 2`.
* Watch a run: `grep -a "^\[" results/queue_main.log | tail -n 3` and `wc -l results/*.jsonl`. If `grep` says "binary file matches", use `grep -a`.

Without `run_all.sh`, the same two passes are:

```bash
nohup python scripts/run_queue_parallel.py --group main  --workers <physical cores - 1> > results/queue_main.log  2>&1 &
nohup python scripts/run_queue_parallel.py --group quiet --workers <cores / 2>          > results/queue_quiet.log 2>&1 &   # only after the main pass has ended
```

The default `--model` is `models/skeleton_posterior_sk<installed scikit-learn version>.pkl`; pass nothing unless you retrained.

## 5. Build the tables and figures

```bash
python scripts/aggregate_m3.py --results-dir results --petshop-root "$PETSHOP_ROOT"   # -> results/aggregate/{tables/, figs/, tables.md, scorecard.md, manifest.json}
python scripts/aggregate_quick.py --results results --out results/m3_report            # the M3-report tables (CSV + Markdown) and figures
python scripts/scalability_report.py                                                   # scalability table and figures (see section 7)
```

`aggregate_m3.py` skips any experiment whose file is missing and says so in `tables/T0_completeness.*`. `--petshop-root` enables the edge-level error analysis (E11, figure F7), which needs the graphs cached by E12 in `results/rca_graphs/`.
`manifest.json` lists every input file with its sha256 and row count, and states the decision rule used.

## 6. Experiments, result files and tables

| ID | Question | Result file (`results/`) | Aggregator output |
|---|---|---|---|
| E1 | synthetic sample-size sweep (d = 25, n = 200..2000), ABIC vs SPOT-cal | `syn_m3_e1.jsonl` | T1a, T3 paired tests, T4 ANOVA, F1 |
| E2 | synthetic robustness (density, SNR, AR(1), missing data) | `syn_m3_e2.jsonl` | T1b, F2 |
| E3 | PetShop main, 4 scenarios x 5 seeds x {ABIC, SPOT-cal, SPOT 1e-3} | `ps_m3_e3.jsonl` | T2, T2b, F3 |
| E4 | ablation: floor only, mask only, random, oracle posterior | `ps_m3_e4.jsonl` | T3 ablation, F4 |
| E5 | moving-block bootstrap of the data, ABIC vs SPOT-cal | `boot_main.jsonl` | T2c, F5 |
| E6 | equal wall-clock budget (60 s, 120 s) | `ps_m3_e6.jsonl` | T2d, F3 |
| E7 | skeleton-posterior probability quality (log-loss, Brier, ECE, reliability) | `post_quality_m3.jsonl` | T7, F6 |
| E8 | inference validation (ADMG vs DAG vs saturated vs marginal; NLL, coverage, CRPS) | `inference_m3.jsonl` | T5, F3 |
| E9 | FCI / RFCI baseline (Fisher-z, Tetrad) | `fci_petshop.jsonl`, `fci_syn_e1.jsonl` | T1, T2g |
| E10 | directed-only learner | `ps_m3_e10.jsonl` | T2e |
| E11 | edge-level error analysis (reads the others plus `rca_graphs/`) | none | T8, F7 |
| E12 | root-cause ranking | `rca_m3.jsonl` | T6, T6b |
| E13 | preprocessing sensitivity (raw_centered) | `ps_m3_e13.jsonl` | T2f |
| scalability | d = 25, 50, 100, 200, n = 1000, fixed budget per run | `scalability.jsonl` | `scalability_report.py` |

Other scripts: `train_posterior.py` (offline posterior training, about 4 minutes), `calibrate_floor.py` (hard-floor calibration), `check_pvalue_direction.py`, `petshop_data_check.py`, `time_abic.py`, `run_petshop.py`, `run_rca.py`, `validate_inference.py`, `posterior_quality.py`, `gradcheck.py`.
The M2 files (`ps_2d_*`, `syn_2c_*`) used an older posterior. Never pool them with later files; the aggregators read only the `*_m3`, `boot_main`, `fci_*` and `rca_m3` files.

Decision rule (fixed before the results): a difference is "supported" only if the paired t-test p < 0.05, the Wilcoxon signed-rank p < 0.05 and the 95% CI of the paired difference excludes 0; otherwise "not distinguishable from noise".
PetShop has one dataset per scenario, so PetShop seeds measure optimizer variance only; E5 (the block bootstrap) is the data-variance test.

## 7. Seeds and determinism

Seeds are the defaults of `run_queue_parallel.py`, and every result row stores its own `seed`:

| Experiment | Seeds / sizes |
|---|---|
| E3 | seeds 0-4, 4 scenarios |
| E4, E6, E10, E13 | seeds 0-2 |
| E5 | 20 resamples x 3 scenarios (high, low, temporal 1); 10 resamples with `--cut` |
| E1 | 10 graphs per n |
| E2 | 8 graphs per cell |
| E12 | graph seeds 0-2 |
| E8 | graph seed 0 |
| E7 | 60 simulated test graphs and 40 calibration graphs |
| scalability | graphs per d = 25, 50, 100, 200: 4, 3, 3, 2 (`--graphs`), n = 1000, 600 s per run (`--budget`); `--list` prints the plan |

Check what a finished run used: `python -c "import pandas as pd; d = pd.read_json('results/ps_m3_e3.jsonl', lines=True); print(sorted(d.seed.unique()))"`.
Accuracy columns (F1, SHD) are deterministic given the seed, the data and one BLAS thread per job (`test_deterministic_given_seed`); times are not. The queue forces `OMP_NUM_THREADS=1` and the equivalent variables itself.
Scalability: run `bash run_all.sh quiet 6` or `python scripts/run_scalability.py --variants abic abic_s spot_cal --workers 6`. Rows are written when a whole job finishes, so an interrupted job reruns from scratch. Rows with an `error` field are failed attempts, not results; drop them before analysis.

## 8. What the submitted run contains (and does not)

| Item | State |
|---|---|
| E1, E3, E4, E5, E6, E7, E8, E9, E10, E12, E13 | complete (`results/aggregate/tables/T0_completeness.csv`) |
| E2 | **79 of 80 jobs**; the main queue log stops at job 456 of 457 |
| E5 | 3 scenarios (high, low, temporal 1); temporal 2 is not included |
| E11 | not produced (the aggregator was run without `--petshop-root`) |
| Scalability | d = 25 (4 seeds), 50 (3), 100 (3) for `abic`, `abic_s`, `spot_cal`; **no d = 200**; the file also holds 20 error rows from a first attempt (a bug, since fixed) |
| Wide posterior, `spot_wide`, posterior-by-d | not part of the submitted results |

Provenance: rows written by the queue after the M4 changes carry `code_hash`, Python / NumPy / SciPy / scikit-learn versions, CPU model, peak memory, load and `timing_clean`. E6 and E8 carry code hash `6ea4a1d0223b` and scikit-learn 1.9.1; the scalability rows carry `8e632f632097`.
The main-pass files (E1-E5, E7, E9, E10, E12, E13) were written before those stamps existed and carry none. Run this to see what any file contains:

```bash
python - <<'EOF'
import glob, json, os
print(f"{'file':28s} {'rows':>6s} {'jobs':>5s}  code_hash         python  sklearn  timing_clean")
for f in sorted(glob.glob("results/*.jsonl")):
    rows = [json.loads(l) for l in open(f) if l.strip()]
    if not rows: continue
    g = lambda k: ",".join(sorted({str(r.get(k)) for r in rows}))
    tc = [r["timing_clean"] for r in rows if "timing_clean" in r]
    share = f"{sum(bool(x) for x in tc) / len(tc):.2f}" if tc else "n/a"
    print(f"{os.path.basename(f):28s} {len(rows):6d} {len({r.get('job_id') for r in rows}):5d}  {g('code_hash'):16s}  {g('py_version'):6s}  {g('sklearn_version'):7s}  {share:>12s}")
EOF
```

Timings: the main pass ran 11 workers at once, so its absolute times are inflated. Compare methods within a pass; for absolute times use E6 and E8, which ran on a quiet machine (43 of 48 E6 rows are `timing_clean`).
The main pass took about 6 hours (the planned estimate was 1.4 hours; the machine was also loaded by other processes for part of it). E6 took 5.6 minutes and E8 0.4 minutes.

## 9. Pitfalls we hit

* **A stale posterior pickle.** The M2 file `models/skeleton_posterior.pkl` (scikit-learn 1.8.0) unpickles under 1.9.1 with only a warning. `SkeletonPosterior.load` now refuses it and `load_or_retrain` uses, or creates, `skeleton_posterior_sk<version>.pkl` (about 3-4 minutes to retrain).
  If a log shows an `InconsistentVersionWarning`, stop: the run is using the wrong model.
* **Orphaned workers.** `pkill -f run_queue_parallel` does not stop the spawned worker processes. Two sets of workers halved the speed of one run. After launching, check that `uptime` shows a load near the worker count, and kill leftovers with `ps -eo pid,args | grep "[m]ultiprocessing.spawn"`.
* **Java and the jar.** Without both, every PetShop job fails at the PAG step with `causal-cmd jar not found`. Set `CAUSAL_CMD_JAR` before launching, with `export`, not a plain assignment.
* **PetShop root.** `run_all.sh` and `pgm.petshop.resolve_root` accept `PETSHOP_ROOT` pointing at the dataset folder or the repository root, expand `~` and relative paths, and list every candidate they tried when none is valid.
* **Plain ABIC at d >= 100.** With its default start the acyclicity term overflows (`RuntimeWarning` in `pgm/dcd.py`) and ABIC returns an empty graph in seconds. This is the reported behavior, not a fault of the run; `abic_s` (start scaled by sqrt(25/d)) is the fair baseline at large d.
* **A hung worker.** One scalability run hung at d = 200 and had to be killed. If a pool shows workers with near-zero CPU for a long time, kill them and rerun the same command; finished jobs are kept.

## 10. Based on

SPOT (Ma et al., KDD 2024, Apache-2.0), ABIC (Bhattacharya et al., 2021), PetShop (Hardt et al., CC-BY-4.0). Design notes per stage are in `STAGE*_NOTES.md`; changes for M3 and M4 are in `CHANGES_M3.md` and `CHANGES_M4.md`.
