#!/usr/bin/env bash
# One entry point for every result in the report.   Run from Code/src:
#     CAUSAL_CMD_JAR=/abs/path/causal-cmd-1.3.0-jar-with-dependencies.jar PETSHOP_ROOT=/abs/path/dataset \
#         bash run_all.sh main  <workers>      # pass 1: any load allowed.  Experiments E1-E5, E7, E9, E10, E12, E13 (resumable), then the tables
#         bash run_all.sh quiet <workers>      # pass 2: machine otherwise IDLE (no browser etc.).  E6 + E8, scalability sweep, wide posterior, figures
# workers: main = physical cores - 1; quiet = about half the cores.  Every step is resumable; a failing step does not stop the next.
# To skip the expensive equal-budget experiment in the quiet pass:  QUIET_EXPS="E8" bash run_all.sh quiet 3
set -u
PASS=${1:?usage: bash run_all.sh main|quiet <workers>}
W=${2:-3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results
case "$PASS" in
main)
  python scripts/run_queue_parallel.py --group main --workers "$W" >> results/queue_main.log 2>&1
  python scripts/aggregate_m3.py --results-dir results --petshop-root "${PETSHOP_ROOT:-}" > results/aggregate.log 2>&1
  python scripts/aggregate_quick.py --results results --out results/m3_report >> results/aggregate.log 2>&1
  ;;
quiet)
  # inference validation (E8) and equal wall-clock budget (E6)
  python scripts/run_queue_parallel.py --experiments ${QUIET_EXPS:-E6 E8} --workers "$W" >> results/queue_quiet.log 2>&1
  # scalability sweep d = 25..200 (narrow posterior), fixed time budget per run
  python scripts/run_scalability.py --variants abic abic_s spot_cal --workers "$W" > results/scal_narrow.log 2>&1
  # wide posterior: train, calibrate floor, posterior quality by d, then its scalability rows
  python scripts/train_posterior_wide.py > results/train_wide.log 2>&1 \
   && python scripts/calibrate_floor_wide.py > results/floor_wide.log 2>&1 \
   && python scripts/posterior_by_d.py > results/posterior_by_d.log 2>&1 \
   && python scripts/run_scalability.py --variants spot_wide --workers "$W" > results/scal_wide.log 2>&1
  python scripts/scalability_report.py > results/scal_report.log 2>&1
  python scripts/posterior_by_d.py --summarize > results/pbd_summary.log 2>&1
  python scripts/aggregate_m3.py --results-dir results --petshop-root "${PETSHOP_ROOT:-}" > results/aggregate_quiet.log 2>&1
  ;;
*) echo "usage: bash run_all.sh main|quiet <workers>"; exit 2 ;;
esac
echo "$PASS pass finished $(date)" >> results/passes_done.txt
