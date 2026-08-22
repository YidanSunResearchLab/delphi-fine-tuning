#!/bin/bash
# Capacity sweep -- one array task per (shape, seed) row of manifest.tsv.
#
#   sbatch --array=1-18%8 experiments/capacity/slurm/train_sweep.sbatch
#   sbatch --array=10     experiments/capacity/slurm/train_sweep.sbatch          # one arm
#   SMOKE=200 sbatch --array=10 experiments/capacity/slurm/train_sweep.sbatch    # 200-iter smoke
#   LONG=15000 sbatch -t 08:00:00 --array=1-3,10-18%8 \
#       experiments/capacity/slurm/train_sweep.sbatch    # schedule-matched follow-up
#
# ---------------------------------------------------------------------------------------
# WHY general-cpu AND NOT general-gpu  (measured 2026-08-20)
# ---------------------------------------------------------------------------------------
# The general-gpu partition does not schedule this account's jobs. Every submission sits at
# Reason=PartitionConfig / Priority indefinitely -- verified with a probe matrix in which the
# ONLY thing that varied was the partition:
#
#     general-gpu  --gres=gpu:1 -c 4  --mem=32G   -> stuck (PartitionConfig)
#     general-gpu  --gres=gpu:1 -c 8  --mem=32G   -> stuck   (the exact shape that ran here historically)
#     general-gpu  NO GPU      -c 4  --mem=32G    -> stuck   <-- rules out the GPU request itself
#     general-cpu  NO GPU      -c 4  --mem=32G    -> COMPLETED in 1 s
#
# So the blocker is the partition, not the resource request, not --exclude, not group
# membership (id -Gn contains compute2, which is in the partition's AllowGroups). Nothing in
# this repository can fix that; it needs an RIS admin. Until then the sweep runs on CPU.
#
# CPU IS NOT A COMPROMISE HERE. The model is 0.3-2.1M parameters. Measured on a general-cpu
# node for the LARGEST shape (12L/12H/120d):
#     8 cores  418 ms/iter        32 cores  170 ms/iter
#     16 cores 226 ms/iter        (a local M-series MPS laptop, for scale: 414 ms/iter)
# At 32 cores a full 5000-iter arm is ~14 min of training plus ~25 min of validation
# (eval dominates -- see below), so ~40 min. With 18 arms running concurrently on a
# partition holding 60 idle nodes, the whole sweep finishes in about the time of one arm.
#
# Numerics: device_type='cpu' means train.py uses nullcontext() instead of autocast, and
# CUDA's TF32 path is not involved, so these runs are plain fp32 -- if anything more precise
# than the CUDA equivalent. Every arm runs the same way, so the comparison is unaffected.
#
# EVAL COST: eval_interval=100 with eval_iters=200 over 2 splits = 400 forward passes per
# eval, 50 evals per run. That is ~64% of the wall time. It is the price of locating the
# val-loss bottom to +/-100 iters, which is the entire point of the sweep.
#
# WHY 16 CORES AND --array=...%8: the general-cpu-qos caps a user at cpu=128 (and mem=2T), so
# cores-per-job and concurrency trade off against each other. Aggregate throughput, using the
# measured per-core scaling above and a 128-core budget:
#     8 cores x 16 jobs = 38.3 iters/s     16 cores x 8 jobs = 35.4     32 cores x 4 jobs = 23.5
# 8 and 16 are within ~8% of each other on total wall time (~65 vs ~70 min for all 18 arms),
# but 16 halves the per-run latency (~52 min vs ~96), so the reference arm lands sooner and
# the 16-core numbers above are measured rather than extrapolated. Throttling the array to %8
# matches the quota, so slurm holds the rest instead of thrashing on QOSMaxCpuPerUserLimit.
#SBATCH --job-name=cap-sweep
#SBATCH --partition=general-cpu
#SBATCH --account=compute2-mdan
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cap_%A_%a.out
#SBATCH --error=logs/cap_%A_%a.err
# c2-node-011 is broken: jobs scheduled onto it die with signal 53 and write no log.
# (This node IS in general-cpu -- unlike the GPU partition, where the exclude was meaningless.)
#SBATCH --exclude=c2-node-011

set -euo pipefail

BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0820_capacity}"
CODE="$BASE/Delphi-2M"
EXP="$BASE/experiments/capacity"
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
DEVICE="${DEVICE:-cpu}"

[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ] || { echo "FATAL: no conda at $CONDA_ROOT" >&2; exit 1; }
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ad-projection
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

# ---------------------------------------------------------------- resolve this task
ROW=$(awk -v i="$SLURM_ARRAY_TASK_ID" 'NR>1 && $1==i {print; exit}' "$EXP/manifest.tsv")
[ -n "$ROW" ] || { echo "FATAL: no manifest row for array id $SLURM_ARRAY_TASK_ID" >&2; exit 1; }
read -r IDX TAG SEED NL NH NE NP <<< "$ROW"

RUN="${TAG}_s${SEED}"
OUT="$BASE/runs/$RUN"
CFG="$EXP/configs/cap_${TAG}.py"
[ -f "$CFG" ] || { echo "FATAL: config not found: $CFG" >&2; exit 1; }
mkdir -p "$OUT"

echo "=============================================================="
echo " capacity sweep task $IDX/$(( $(wc -l < "$EXP/manifest.tsv") - 1 ))"
echo "   shape    $TAG   (${NL}L / ${NH}H / ${NE}d, head_dim $(( NE / NH )))"
echo "   params   $NP"
echo "   seed     $SEED"
echo "   out_dir  $OUT"
echo "   host     $(hostname)   device=$DEVICE  threads=$OMP_NUM_THREADS"
echo "=============================================================="

# ---------------------------------------------------------------- preflight (cheap, per task)
cd "$CODE"
[ -e data/nacc-dedup-s42/train.bin ] || {
  echo "FATAL: data/nacc-dedup-s42/train.bin missing under $CODE" >&2; exit 1; }
python - <<'PY'
import os, sys
exp = {"train": 800110, "val": 114143, "test": 228057}
for s, e in exp.items():
    n = os.path.getsize(f"data/nacc-dedup-s42/{s}.bin") // 12
    if n != e:
        sys.exit(f"FATAL: {s}.bin has {n:,} events, expected {e:,} -- wrong dataset")
print("  dataset fingerprint ok (train 800,110 / val 114,143 / test 228,057)")
PY

# ---------------------------------------------------------------- run
EXTRA=""
if [ -n "${SMOKE:-}" ]; then
  EXTRA="--max_iters=${SMOKE} --lr_decay_iters=${SMOKE} --warmup_iters=$(( SMOKE / 10 )) --eval_interval=$(( SMOKE / 2 ))"
  OUT="${OUT}_SMOKE"; mkdir -p "$OUT"
  echo ">>> SMOKE MODE: $EXTRA"
fi
# LONG=<iters>: the schedule-matched follow-up.
#
# The 5000-iter sweep found the val-loss bottom moving monotonically LATER as capacity comes
# down -- to iter 4667 +/- 321 for the 0.31M arm, i.e. 93% of the way through the schedule.
# At that point "smaller is better" is confounded with "smaller has not finished improving":
# the 5000-iter cosine schedule was inherited from a config written for the 2.1M model, and a
# smaller model may simply need more of it. This mode re-runs selected arms with a schedule
# long enough that the bottom is interior rather than pinned against the end.
#
# max_iters AND lr_decay_iters move together, so the cosine still completes exactly at the
# end (leaving lr_decay_iters at 5000 would instead run 10000 iters at min_lr, which is a
# different and much less interesting experiment). warmup and eval_interval are unchanged,
# so best_val stays directly comparable to the 5000-iter arms.
if [ -n "${LONG:-}" ]; then
  EXTRA="--max_iters=${LONG} --lr_decay_iters=${LONG}"
  OUT="${OUT}_long${LONG}"; mkdir -p "$OUT"
  echo ">>> LONG SCHEDULE: $EXTRA  (warmup and eval_interval unchanged)"
fi

START=$(date +%s)
# wandb stays OFF: 18 runs would flood the dashboard, and collect.py reads the logs anyway.
# tee into the run dir so every run is self-contained -- collect.py never has to correlate
# a slurm %A_%a log filename back to a (shape, seed) pair.
python -u training/train.py "$CFG" --device="$DEVICE" --seed="$SEED" --out_dir="$OUT" $EXTRA 2>&1 \
  | tee "$OUT/train.log"
END=$(date +%s)

# ---------------------------------------------------------------- per-run metadata
cat > "$OUT/run_meta.json" <<JSON
{
  "run": "$RUN",
  "tag": "$TAG",
  "seed": $SEED,
  "n_layer": $NL,
  "n_head": $NH,
  "n_embd": $NE,
  "head_dim": $(( NE / NH )),
  "n_params": $NP,
  "array_job": "${SLURM_ARRAY_JOB_ID:-na}",
  "array_task": ${SLURM_ARRAY_TASK_ID:-0},
  "host": "$(hostname)",
  "device": "$DEVICE",
  "threads": ${OMP_NUM_THREADS},
  "wall_seconds": $(( END - START )),
  "smoke": "${SMOKE:-}",
  "config": "$CFG"
}
JSON

echo
echo "DONE  $RUN  in $(( (END - START) / 60 )) min  ->  $OUT"
