#!/usr/bin/env bash
# Final before/after benchmark: full PMLB suite, sequential arms (fairness:
# each arm gets an otherwise-idle machine), then paired comparison.
set -uo pipefail

VENV="${VENV:-$HOME/venvs/taml-bench}"
PY="$VENV/bin/python"
BEFORE_TREE="${BEFORE_TREE:-/home/user/before-main}"
AFTER_TREE="${AFTER_TREE:-/home/user/TabularAML-refactored}"
RESULTS="${RESULTS:-/home/user/results}"
SEEDS="${SEEDS:-0 1 2}"
TIME_BUDGET="${TIME_BUDGET:-1200}"
N_WORKERS="${N_WORKERS:-2}"

export PMLB_CACHE_DIR="${PMLB_CACHE_DIR:-$HOME/.cache/pmlb}"

run_arm () {
  local tree="$1" out="$2" label="$3"
  echo "=== [$label] arm starting at $(date -u +%H:%M:%S) tree=$tree -> $out ==="
  cd "$tree"
  PYTHONPATH="$tree" "$PY" -m tabularaml.benchmarks.feature_gen.targeted \
    --suite pmlb --frameworks nofe tabularaml --seeds $SEEDS \
    --n-workers "$N_WORKERS" --time-budget "$TIME_BUDGET" --no-wandb \
    --results-dir "$out"
  echo "=== [$label] arm finished at $(date -u +%H:%M:%S) ==="
}

run_arm "$BEFORE_TREE" "$RESULTS/before-pmlb" before
run_arm "$AFTER_TREE"  "$RESULTS/after-pmlb"  after

cd "$AFTER_TREE"
"$PY" "$AFTER_TREE/scripts/compare_targeted_runs.py" \
  --before "$RESULTS/before-pmlb/master.csv" \
  --after  "$RESULTS/after-pmlb/master.csv" \
  --out    "$RESULTS/before_after_report"
echo "ALL DONE $(date -u +%H:%M:%S)"
