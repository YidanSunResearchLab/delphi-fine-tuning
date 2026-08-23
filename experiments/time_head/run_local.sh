#!/usr/bin/env bash
# run_local.sh -- the same 6 arms on a laptop (Apple MPS or CPU). One seed by default.
#
#   ./experiments/time_head/run_local.sh                    # 2 arms x seed 42
#   SEEDS="42 43 44" ./experiments/time_head/run_local.sh    # all three
#   DEVICE=cpu ./experiments/time_head/run_local.sh
#   SMOKE=200 ./experiments/time_head/run_local.sh           # 200 iters, ~2 min/arm, wiring check
#
# Same configs, same eval cadence, same seeds as the cluster; only the float backend differs.
#
# BUT: THE CONTROLS ARE CLUSTER RUNS. A local arm paired against a CUDA/CPU-cluster control is
# NOT a controlled comparison -- the pairing argument in arms.py rests on identical init and
# identical batches, which survives a backend change only approximately. Use this to check the
# wiring and to eyeball whether loss_dt moves at all; run the real thing with
# slurm/train_time_head.sbatch, or re-run the controls locally too:
#
#     ORDER="L12E120H12 L8E120H6" ../capacity/run_local.sh
#
# Budget: EVAL DOMINATES (eval_interval=100 over 5000 iters = 50 evals x 200 eval_iters x 2
# splits = 20,000 forward passes against 5,000 training steps). ~75 min for the 2.1M arm on an
# M-series MPS, ~50 min for the 1.41M one; ~2 h for both at one seed.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
PKG="$REPO/Delphi-2M"
DEVICE="${DEVICE:-mps}"
SEEDS="${SEEDS:-42}"
RUNS="${RUNS:-$HERE/runs}"
PY="${PY:-python3}"
ORDER="${ORDER:-TH_L8E120H6 TH_L12E120H12}"   # the recommended shape first

[ -e "$PKG/data/nacc-dedup-s42/train.bin" ] || {
  echo "FATAL: $PKG/data/nacc-dedup-s42/train.bin not found." >&2
  echo "  symlink or build the split first, then re-run verify.py" >&2; exit 1; }

"$PY" "$HERE/verify.py" >/dev/null || { echo "FATAL: preflight failed -- run verify.py" >&2; exit 1; }
echo "preflight ok | device=$DEVICE | seeds=$SEEDS"

ARM_Q='
import sys
from arms import shape, n_params, control
nl, nh, ne = shape(sys.argv[1])
print(nl, nh, ne, n_params(sys.argv[1]), control(sys.argv[1]))'

EXTRA=""
if [ -n "${SMOKE:-}" ]; then
  EXTRA="--max_iters=${SMOKE} --lr_decay_iters=${SMOKE} --warmup_iters=$(( SMOKE / 10 )) --eval_interval=$(( SMOKE / 2 ))"
  echo ">>> SMOKE MODE: $EXTRA  (results are NOT comparable to anything; wiring only)"
fi

mkdir -p "$RUNS"
for SEED in $SEEDS; do
  for ARM in $ORDER; do
    CFG="$HERE/configs/th_${ARM}.py"
    OUT="$RUNS/${ARM}_s${SEED}"
    [ -n "${SMOKE:-}" ] && OUT="${OUT}_SMOKE"
    if [ -f "$OUT/run_meta.json" ]; then echo "== skip $ARM s$SEED (already done)"; continue; fi
    mkdir -p "$OUT"
    read -r NL NH NE NP CTL <<< "$(PYTHONPATH="$HERE" "$PY" -c "$ARM_Q" "$ARM")"
    echo "== $ARM seed $SEED  (${NL}L/${NH}H/${NE}d + $(( NE + 1 ))-param head, ${NP} params)  -> $OUT"
    echo "   control: $REPO/experiments/capacity/runs/${CTL}_s${SEED}"
    START=$(date +%s)
    ( cd "$PKG" && "$PY" -u training/train.py "$CFG" --device="$DEVICE" --seed="$SEED" \
        --out_dir="$OUT" $EXTRA 2>&1 ) | tee "$OUT/train.log" \
        | grep -E "^step |parameters|cohort filter" || true
    END=$(date +%s)
    cat > "$OUT/run_meta.json" <<JSON
{"run":"${ARM}_s${SEED}","tag":"$ARM","control":"${CTL}_s${SEED}","time_head":true,
 "seed":$SEED,"n_layer":$NL,"n_head":$NH,"n_embd":$NE,"head_dim":$(( NE / NH )),
 "n_params":$NP,"wall_seconds":$((END-START)),"smoke":"${SMOKE:-}",
 "gpu":"local-$DEVICE","host":"$(hostname)","config":"$CFG"}
JSON
    echo "   done in $(( (END-START)/60 )) min"
  done
done
echo
echo "all local runs done -> $RUNS"
echo "next:"
echo "  $PY $REPO/experiments/capacity/decompose.py --runs $RUNS --out $HERE/results/decomposition.csv"
echo "  $PY $HERE/analyze.py"
