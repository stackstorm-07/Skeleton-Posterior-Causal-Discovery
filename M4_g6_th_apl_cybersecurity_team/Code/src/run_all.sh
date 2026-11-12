#!/usr/bin/env bash
# One entry point for every result in the report.   Run from Code/src:
#     export CAUSAL_CMD_JAR=/abs/path/causal-cmd-1.3.0-jar-with-dependencies.jar     # (or put it in third_party/Spot_original/learn/bin/)
#     export PETSHOP_ROOT=/abs/path/to/petshop-root-cause-analysis/dataset            # `export`, not just a plain assignment
#     bash run_all.sh main  <workers>      # pass 1: any load allowed.  E1-E5, E7, E9, E10, E12, E13 (resumable), then the tables
#     bash run_all.sh quiet <workers>      # pass 2: machine otherwise IDLE.  E6 + E8, scalability sweep, wide posterior, figures
# workers: main = physical cores - 1; quiet = about half the cores (clamped to cpu_count / 2: E6 refuses more).
# Every step is resumable and a failing step does not stop the next one; each step's exit code is appended to results/run_all_steps.log.
# Skip the expensive equal-budget experiment in the quiet pass:  QUIET_EXPS="E8" bash run_all.sh quiet 3
set -u
# absolute paths first (relative paths and ~ are taken from the caller's directory), then work from this script's directory
[ -n "${PETSHOP_ROOT:-}" ] && PETSHOP_ROOT=$(realpath -m -- "${PETSHOP_ROOT/#\~/$HOME}") && export PETSHOP_ROOT
[ -n "${CAUSAL_CMD_JAR:-}" ] && CAUSAL_CMD_JAR=$(realpath -m -- "${CAUSAL_CMD_JAR/#\~/$HOME}") && export CAUSAL_CMD_JAR
cd "$(dirname "$(realpath "$0")")" || exit 1
PASS=${1:?usage: bash run_all.sh main|quiet <workers>}
W=${2:-3}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p results
STEPS=results/run_all_steps.log

# ---- preflight: fail early and say why (before hours of work)
ROOT_RESOLVED=$(python - <<'PY' 2>&1
import sys
try:
    from pgm.petshop import resolve_root
    print(resolve_root())
except Exception as e:
    print("ERROR " + str(e)); sys.exit(1)
PY
) || { echo "$ROOT_RESOLVED" >&2; echo "run_all.sh: PetShop dataset not found (see above)." >&2; exit 3; }
export PETSHOP_ROOT="$ROOT_RESOLVED"
python - <<'PY' || { echo "run_all.sh: the Tetrad jar is missing. export CAUSAL_CMD_JAR=/abs/path/causal-cmd-1.3.0-jar-with-dependencies.jar" >&2; exit 4; }
from pgm.graphs import find_tetrad_jar
print("jar:", find_tetrad_jar())
PY
echo "PetShop root: $PETSHOP_ROOT"
python -c "import sklearn; print('scikit-learn', sklearn.__version__)"

step() {   # step <label> <command...>: run, log the exit code, never abort the script
  local label=$1; shift
  echo "$(date '+%F %T') START $label" >> "$STEPS"
  "$@"; local rc=$?
  echo "$(date '+%F %T') END   $label exit=$rc" >> "$STEPS"
  return $rc
}

case "$PASS" in
main)
  step queue_main   bash -c "python scripts/run_queue_parallel.py --group main --workers $W >> results/queue_main.log 2>&1"
  step aggregate_m3 bash -c "python scripts/aggregate_m3.py --results-dir results --petshop-root '$PETSHOP_ROOT' > results/aggregate.log 2>&1"
  step aggregate_quick bash -c "python scripts/aggregate_quick.py --results results --out results/m3_report >> results/aggregate.log 2>&1"
  ;;
quiet)
  HALF=$(python -c "import os; print(max(1, (os.cpu_count() or 2) // 2))")
  if [ "$W" -gt "$HALF" ]; then echo "workers $W > cpu_count/2: using $HALF (E6 needs the machine at most half busy)"; W=$HALF; fi
  # E6 (equal wall-clock budget) and E8 (inference validation): one queue call per experiment, so each is judged on its own
  for E in ${QUIET_EXPS:-E6 E8}; do
    step "queue_$E" bash -c "python scripts/run_queue_parallel.py --experiments $E --workers $W >> results/queue_quiet.log 2>&1"
  done
  step scal_narrow  bash -c "python scripts/run_scalability.py --variants abic abic_s spot_cal --workers $W > results/scal_narrow.log 2>&1"
  step train_wide   bash -c "python scripts/train_posterior_wide.py > results/train_wide.log 2>&1" \
   && step floor_wide bash -c "python scripts/calibrate_floor_wide.py > results/floor_wide.log 2>&1" \
   && step post_by_d  bash -c "python scripts/posterior_by_d.py > results/posterior_by_d.log 2>&1" \
   && step scal_wide  bash -c "python scripts/run_scalability.py --variants spot_wide --workers $W > results/scal_wide.log 2>&1"
  step scal_report  bash -c "python scripts/scalability_report.py > results/scal_report.log 2>&1"
  step pbd_summary  bash -c "python scripts/posterior_by_d.py --summarize > results/pbd_summary.log 2>&1"
  step aggregate_m3 bash -c "python scripts/aggregate_m3.py --results-dir results --petshop-root '$PETSHOP_ROOT' > results/aggregate_quiet.log 2>&1"
  ;;
*) echo "usage: bash run_all.sh main|quiet <workers>"; exit 2 ;;
esac
echo "$PASS pass finished $(date)" >> results/passes_done.txt
echo "done. Exit codes per step: $STEPS   (grep -v 'exit=0' $STEPS shows failures)"
