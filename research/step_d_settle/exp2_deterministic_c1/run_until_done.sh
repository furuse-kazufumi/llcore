#!/usr/bin/env bash
# EXP2 chunked-resumable driver: re-invoke the experiment until all cells done.
# Each py invocation uses --budget=430 so a single process stays < 900s (G1):
# worst-case one big cell (~458s) starting near budget end -> ~888s < 900s.
set -u
cd "$(dirname "$0")/../../.." || exit 1   # -> repo root (llcore)
SCRIPT="research/step_d_settle/exp2_deterministic_c1/exp2_deterministic_c1.py"
RESULTS="research/step_d_settle/exp2_deterministic_c1/exp2_results.json"
for pass in $(seq 1 30); do
  echo "=== DRIVER pass $pass ==="
  py -3.11 "$SCRIPT" --budget=430 2>&1 | tail -6
  if [ -f "$RESULTS" ]; then
    echo "=== DRIVER: results.json present, all cells done after pass $pass ==="
    break
  fi
done
echo "=== DRIVER FINISHED ==="
