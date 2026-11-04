#!/bin/sh
# Stage 2d PetShop queue.  Needs PETSHOP_ROOT (the dataset dir) and the Tetrad jar (CAUSAL_CMD_JAR or third_party/...).
# Single core, resumable (finished scenario/prep/seed/variant lines are skipped).  Rough cost at d~30, one core:
# ABIC ~150 s per run (2c timing), so: main ~2 h per preprocessing, ablations ~1.5 h, equal-budget ~50 min.
set -e
cd "$(dirname "$0")/.."
: "${PETSHOP_ROOT:?set PETSHOP_ROOT to the PetShop dataset directory}"
python -u scripts/petshop_data_check.py
PY="python -u scripts/run_petshop.py --restarts 1"
# 1. headline: Gaussian-rank + z-score, 5 optimizer seeds, no budget
$PY --prep gauss_rank   --seeds 0 1 2 3 4 --variants abic spot spot_cal --out results/ps_2d_main_gauss_rank.jsonl --tag main
# 2. ablations (does the gain come from the posterior?)
$PY --prep gauss_rank   --seeds 0 1 2   --variants spot_nofloor spot_floor_only spot_random spot_oracle --out results/ps_2d_ablations_gauss_rank.jsonl --tag ablation
# 3. equal wall-clock budget (120 s per run, posterior time charged to SPOT)
$PY --prep gauss_rank   --seeds 0 1 2   --variants abic spot_cal --budget 120 --out results/ps_2d_budget120_gauss_rank.jsonl --tag budget120
# 4. sensitivity: raw latency (centered, one global scale)
$PY --prep raw_centered --seeds 0 1 2 3 4 --variants abic spot spot_cal --out results/ps_2d_main_raw_centered.jsonl --tag main_raw
python scripts/summarize_2d.py results/ps_2d_*.jsonl --out results/ps_2d_summary.md
