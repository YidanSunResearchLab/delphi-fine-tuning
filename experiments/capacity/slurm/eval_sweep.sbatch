#!/bin/bash
# Phase 2 -- score each trained arm on the Figure-2 metrics (test split).
#
#   sbatch --array=1-6%3 experiments/capacity/slurm/eval_sweep.sbatch
#
# Array index here indexes SHAPES (not shape x seed): one eval per shape, using the seed
# given by EVAL_SEED (default 42). Scoring all 18 checkpoints would take ~7 CPU-hours of
# Monte-Carlo for a question -- "does the shape matter" -- that one seed already answers;
# widen it later if a shape turns out to be interesting.
#
# WHY EACH TASK GETS ITS OWN COPY OF THE CODE TREE:
# figure2_core.OUT_DIR is `os.path.join(HERE, "results", "figure2")` where HERE is derived
# from __file__ -- a module constant with no CLI override. Six concurrent tasks pointed at
# one tree would overwrite each other's metrics.json and figures (the _cache is keyed by
# checkpoint signature and would survive, but the outputs would not). Rather than patch the
# delivered pipeline just to run an experiment, each task copies the ~1 MB code tree to its
# own path, which makes HERE -- and therefore OUT_DIR -- unique for free. data/ is symlinked,
# never copied: no PHI moves.
#SBATCH --job-name=cap-eval
#SBATCH --partition=general-cpu
#SBATCH --account=compute2-mdan
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/capeval_%A_%a.out
#SBATCH --error=logs/capeval_%A_%a.err
#SBATCH --exclude=c2-node-011

set -euo pipefail

BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0820_capacity}"
EXP="$BASE/experiments/capacity"
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
EVAL_SEED="${EVAL_SEED:-42}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ad-projection
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1                       # each MC worker must stay single-threaded
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-32} - 2 ))

# distinct shapes, in manifest order
TAG=$(awk -F'\t' 'NR>1 {print $2}' "$EXP/manifest.tsv" | awk '!seen[$0]++' \
      | sed -n "${SLURM_ARRAY_TASK_ID}p")
[ -n "$TAG" ] || { echo "FATAL: no shape at index $SLURM_ARRAY_TASK_ID" >&2; exit 1; }

RUN="${TAG}_s${EVAL_SEED}"
CKPT="$BASE/runs/$RUN/ckpt.pt"
[ -f "$CKPT" ] || { echo "FATAL: no checkpoint at $CKPT -- did phase 1 finish for $RUN?" >&2; exit 1; }

TREE="$BASE/evaltrees/$RUN"
echo "=== eval $RUN"
echo "    ckpt  $CKPT"
echo "    tree  $TREE  (private results/ dir -- see header)"
rm -rf "$TREE"; mkdir -p "$TREE"
cp -r "$BASE/Delphi-2M" "$TREE/Delphi-2M"
rm -f "$TREE/Delphi-2M/data"
ln -s "$BASE/Delphi-2M/data" "$TREE/Delphi-2M/data"   # symlink-to-symlink, resolves fine
mkdir -p "$TREE/Delphi-2M/$(dirname "$RUN")"
cp "$CKPT" "$TREE/Delphi-2M/ckpt_under_test.pt"

cd "$TREE/Delphi-2M"
echo "    host  $(hostname) | workers $WORKERS"

START=$(date +%s)
# 1) Monte-Carlo trajectories + embeddings (the expensive part), cached by ckpt signature
python -u figure2/figure2_core.py --build --ckpt ckpt_under_test.pt \
       --dataset nacc-dedup-s42 --device cpu --workers "$WORKERS" "$@"
# 2) the cohort-matched panels -- the population train.py actually fitted on
python -u figure2/figure2_panels.py --cohort matched --ckpt ckpt_under_test.pt \
       --dataset nacc-dedup-s42 "$@"
END=$(date +%s)

# harvest just the numbers back into the run dir; the tree itself is disposable
DEST="$BASE/results/eval/$RUN"
mkdir -p "$DEST"
cp -f results/figure2/metrics_matched.json "$DEST/" 2>/dev/null || true
cp -f results/figure2/SUMMARY_matched.md   "$DEST/" 2>/dev/null || true
cp -f results/figure2/*_matched*.csv       "$DEST/" 2>/dev/null || true
cp -f results/figure2/figure2_combined_matched.png "$DEST/" 2>/dev/null || true
echo "{\"run\":\"$RUN\",\"tag\":\"$TAG\",\"eval_seed\":$EVAL_SEED,\"wall_seconds\":$((END-START))}" \
  > "$DEST/eval_meta.json"

echo "DONE  $RUN  in $(( (END-START)/60 )) min  ->  $DEST"
ls -la "$DEST"
