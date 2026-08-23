#!/bin/bash
# Time-head experiment, phase 1 -- one array task per (arm, seed) row of manifest.tsv.
#
#   sbatch --array=1-6 experiments/time_head/slurm/train_time_head.sbatch
#   sbatch --array=4   experiments/time_head/slurm/train_time_head.sbatch          # one arm
#   SMOKE=200 sbatch --array=4 experiments/time_head/slurm/train_time_head.sbatch  # 200-iter smoke
#
# THE CONTROLS ARE NOT IN THIS ARRAY. They are the capacity sweep's own runs at the same
# shape and seed (experiments/capacity/runs/<control>_s<seed>), reused exactly rather than
# re-run -- arms.py explains why that is sound, and verify.py checks the parts that could
# make it unsound. 6 tasks, not 12.
#
# Partition, cores and concurrency are inherited wholesale from the capacity sweep, which
# measured them (see experiments/capacity/slurm/train_sweep.sbatch for the numbers and for
# why general-gpu is unusable by this account). Nothing here changes the compute shape, so
# nothing here re-derives it. The one relevant fact: 16 cores, ~40 min per 5000-iter arm at
# the 2.1M shape, and eval is ~64% of that. 6 tasks fit inside the 128-core quota at once,
# so no %-throttle is needed.
#SBATCH --job-name=th-sweep
#SBATCH --partition=general-cpu
#SBATCH --account=compute2-mdan
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/th_%A_%a.out
#SBATCH --error=logs/th_%A_%a.err
# c2-node-011 is broken: jobs scheduled onto it die with signal 53 and write no log.
#SBATCH --exclude=c2-node-011

set -euo pipefail

BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0820_capacity}"
CODE="$BASE/Delphi-2M"
EXP="$BASE/experiments/time_head"
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
read -r IDX ARM CTL SEED NL NH NE NP <<< "$ROW"

RUN="${ARM}_s${SEED}"
OUT="$BASE/th_runs/$RUN"
CFG="$EXP/configs/th_${ARM}.py"
[ -f "$CFG" ] || { echo "FATAL: config not found: $CFG" >&2; exit 1; }
mkdir -p "$OUT"

CTLRUN="$BASE/runs/${CTL}_s${SEED}"
echo "=============================================================="
echo " time-head task $IDX/$(( $(wc -l < "$EXP/manifest.tsv") - 1 ))"
echo "   arm      $ARM   (${NL}L / ${NH}H / ${NE}d + a $(( NE + 1 ))-param time head)"
echo "   params   $NP"
echo "   seed     $SEED"
echo "   control  $CTLRUN  $( [ -f "$CTLRUN/train.log" ] && echo '(present)' || echo '(MISSING -- analysis will have nothing to pair against)' )"
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
# The head must actually exist in the model this env builds. Cheap here, and it fails the
# task in 2 s instead of producing a 40-minute run that silently trained the control again.
python - <<'PY'
import sys
sys.path.insert(0, ".")
from dataclasses import fields
from delphi.model import DelphiConfig
if "time_head" not in {f.name for f in fields(DelphiConfig)}:
    sys.exit("FATAL: this delphi/model.py has no time_head -- the tree on the cluster is "
             "older than experiments/time_head/. Sync Delphi-2M/ before submitting.")
print("  delphi.model supports time_head")
PY

# ---------------------------------------------------------------- run
EXTRA=""
if [ -n "${SMOKE:-}" ]; then
  EXTRA="--max_iters=${SMOKE} --lr_decay_iters=${SMOKE} --warmup_iters=$(( SMOKE / 10 )) --eval_interval=$(( SMOKE / 2 ))"
  OUT="${OUT}_SMOKE"; mkdir -p "$OUT"
  echo ">>> SMOKE MODE: $EXTRA"
fi

START=$(date +%s)
# wandb off; collect.py/analyze.py read the logs. tee into the run dir so each run is
# self-contained and nothing has to map a %A_%a filename back to an (arm, seed).
python -u training/train.py "$CFG" --device="$DEVICE" --seed="$SEED" --out_dir="$OUT" $EXTRA 2>&1 \
  | tee "$OUT/train.log"
END=$(date +%s)

# ---------------------------------------------------------------- per-run metadata
# Same schema as the capacity sweep's run_meta.json (collect.py and decompose.py read it),
# plus the two fields specific to this experiment.
cat > "$OUT/run_meta.json" <<JSON
{
  "run": "$RUN",
  "tag": "$ARM",
  "control": "${CTL}_s${SEED}",
  "time_head": true,
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
