#!/bin/sh
# Stage 2c synthetic evaluation queue (single core; ~1.5-2 h).  Resumable: finished (seed, variant) pairs are skipped.
cd "$(dirname "$0")/.."
PY="python -u scripts/spot_vs_abic_synthetic.py --d 25 --n 1000 --restarts 1"
$PY --data zscore --seeds 0 1 2 3 4 5 --variants abic spot spot_cal --out results/syn_2c_main_zscore.jsonl --tag main
$PY --data zscore --seeds 0 1 2 --variants spot_nofloor spot_floor_only spot_random spot_oracle spot_released --out results/syn_2c_ablations_zscore.jsonl --tag ablation
$PY --data raw --seeds 0 1 2 3 4 5 --variants abic spot spot_cal --out results/syn_2c_main_raw.jsonl --tag main_raw
$PY --data zscore --seeds 0 1 2 3 --variants abic spot_cal --budget 60 --out results/syn_2c_budget60_zscore.jsonl --tag budget60
