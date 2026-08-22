#!/usr/bin/env bash
# pull.sh -- bring cluster results back to the laptop, then collect + analyze.
#
#   ./experiments/capacity/pull.sh
#
# Pulls only the small artifacts: train.log and run_meta.json per run, plus phase-2 eval
# metrics. Checkpoints stay on the cluster (they are ~4-25 MB each x 18 and are not needed
# for the analysis; pull one explicitly if you want to evaluate it).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
HOST="${AD_HOST:-compute2}"
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0820_capacity}"

mkdir -p "$HERE/runs" "$HERE/results"
echo "=== pulling run logs from $HOST:$BASE/runs"
rsync -az --prune-empty-dirs \
  --include '*/' --include 'train.log' --include 'run_meta.json' --exclude '*' \
  "$HOST:$BASE/runs/" "$HERE/runs/"

echo "=== pulling phase-2 eval metrics (if any)"
rsync -az --ignore-missing-args "$HOST:$BASE/results/eval" "$HERE/results/" 2>/dev/null || \
  echo "    (none yet)"

echo "=== collect + analyze"
python3 "$HERE/collect.py"
python3 "$HERE/analyze.py"
