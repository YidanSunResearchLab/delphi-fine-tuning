#!/usr/bin/env bash
# run_local.sh -- the same sweep on a laptop (Apple MPS or CPU). One seed by default.
#
#   ./experiments/capacity/run_local.sh                   # 6 shapes x seed 42
#   SEEDS="42 43" ./experiments/capacity/run_local.sh      # more seeds
#   DEVICE=cpu ./experiments/capacity/run_local.sh
#
# This is NOT a cheaper approximation of the cluster run -- it is the SAME run. Identical
# configs, identical eval cadence, identical seed. Only the float backend differs (MPS/CPU
# vs CUDA), so results will not be bit-identical to the cluster but are the same experiment.
# Where both exist for a (shape, seed), agreement between them is a free robustness check.
#
# Requires Delphi-2M/data/nacc-dedup-s42 (a symlink is fine) to pass verify.py's fingerprint.
#
# Budget: EVAL DOMINATES. eval_interval=100 over 5000 iters = 50 evals x 200 eval_iters x 2
# splits = 20,000 forward passes against 5,000 training steps. Roughly 75 min for the largest
# shape on an M-series MPS, ~15 min for the smallest; ~5 h for all six.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PKG="$REPO/Delphi-2M"
DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-42}"
RUNS="${RUNS:-$HERE/runs}"
PY="${PY:-python3}"

# Reference first, then the shape under review, then the rest -- so an interrupted sweep
# still leaves the comparison that matters.
ORDER="${ORDER:-L12E120H12 L8E96H6 L12E120H6 L8E120H6 L6E96H6 L6E64H4}"

[ -e "$PKG/data/nacc-dedup-s42/train.bin" ] || {
  echo "FATAL: $PKG/data/nacc-dedup-s42/train.bin not found." >&2
  echo "  symlink or build the split first, then re-run verify.py" >&2; exit 1; }

"$PY" "$HERE/verify.py" >/dev/null || { echo "FATAL: preflight failed -- run verify.py" >&2; exit 1; }
echo "preflight ok | device=$DEVICE | seeds=$SEEDS"

SHAPE_Q='
import sys
from shapes import SHAPES, n_params
nl, nh, ne, _ = SHAPES[sys.argv[1]]
print(nl, nh, ne, n_params(nl, nh, ne))'

mkdir -p "$RUNS"
for SEED in $SEEDS; do
  for TAG in $ORDER; do
    CFG="$HERE/configs/cap_${TAG}.py"
    OUT="$RUNS/${TAG}_s${SEED}"
    if [ -f "$OUT/run_meta.json" ]; then echo "== skip $TAG s$SEED (already done)"; continue; fi
    mkdir -p "$OUT"
    read -r NL NH NE NP <<< "$(PYTHONPATH="$HERE" "$PY" -c "$SHAPE_Q" "$TAG")"
    echo "== $TAG seed $SEED  (${NL}L/${NH}H/${NE}d, ${NP} params)  -> $OUT"
    START=$(date +%s)
    ( cd "$PKG" && "$PY" -u training/train.py "$CFG" --device="$DEVICE" --seed="$SEED" \
        --out_dir="$OUT" 2>&1 ) | tee "$OUT/train.log" \
        | grep -E "^step |parameters|cohort filter" || true
    END=$(date +%s)
    cat > "$OUT/run_meta.json" <<JSON
{"run":"${TAG}_s${SEED}","tag":"$TAG","seed":$SEED,"n_layer":$NL,"n_head":$NH,"n_embd":$NE,
 "head_dim":$(( NE / NH )),"n_params":$NP,"wall_seconds":$((END-START)),
 "gpu":"local-$DEVICE","host":"$(hostname)","config":"$CFG"}
JSON
    echo "   done in $(( (END-START)/60 )) min"
  done
done
echo
echo "all local runs done -> $RUNS"
echo "next: $PY $HERE/collect.py && $PY $HERE/analyze.py"
