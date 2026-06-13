#!/usr/bin/env bash
# Self-healing supervisor: keep (re)launching the resume-safe benchmark until
# the paired comparison report exists. Survives idle-gap reaps of the inner run.
set -uo pipefail
REPORT=/home/user/results/before_after_report/summary.json
LOG=/home/user/results/final_benchmark_supervised.log

while [ ! -f "$REPORT" ]; do
  if ! pgrep -f "feature_gen.targeted" >/dev/null 2>&1 && ! pgrep -f "run_final_benchmark.sh" >/dev/null 2>&1; then
    echo "[supervisor $(date -u +%H:%M)] (re)launching benchmark"
    bash /home/user/TabularAML-refactored/scripts/run_final_benchmark.sh >> "$LOG" 2>&1 &
  fi
  sleep 60
done
echo "[supervisor $(date -u +%H:%M)] report present, exiting"
