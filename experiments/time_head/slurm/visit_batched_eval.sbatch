#!/bin/bash
# Visit-batched generation, scored on the FULL Figure-2 pipeline so the timing bias is
# directly comparable with every existing number.
#   sbatch --array=1-2 experiments/time_head/slurm/visit_batched_eval.sbatch
#SBATCH --job-name=vb-eval
#SBATCH --partition=general-cpu
#SBATCH --account=compute2-mdan
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/vbeval_%A_%a.out
#SBATCH --error=logs/vbeval_%A_%a.err
#SBATCH --exclude=c2-node-011
set -euo pipefail
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0820_capacity}"
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate ad-projection
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-16} - 2 ))

# 1-3 = the control at each seed, 4-6 = the time-head arm at each seed. Same 3 seeds as
# every other comparison in this folder, so the visit-batched numbers pair against the
# delivered-sampler evals already in results/eval and results/th_eval.
case "${SLURM_ARRAY_TASK_ID:-1}" in
  1) RUN=L8E120H6_s42;    CKPT="$BASE/runs/$RUN/ckpt.pt" ;;
  2) RUN=L8E120H6_s43;    CKPT="$BASE/runs/$RUN/ckpt.pt" ;;
  3) RUN=L8E120H6_s44;    CKPT="$BASE/runs/$RUN/ckpt.pt" ;;
  4) RUN=TH_L8E120H6_s42; CKPT="$BASE/th_runs/$RUN/ckpt.pt" ;;
  5) RUN=TH_L8E120H6_s43; CKPT="$BASE/th_runs/$RUN/ckpt.pt" ;;
  6) RUN=TH_L8E120H6_s44; CKPT="$BASE/th_runs/$RUN/ckpt.pt" ;;
  *) echo "FATAL: bad array id" >&2; exit 1 ;;
esac
[ -f "$CKPT" ] || { echo "FATAL: no ckpt at $CKPT" >&2; exit 1; }

# the empirical k, written once by the helper below
SIZES="$BASE/results/time_head/visit_sizes.npy"
[ -f "$SIZES" ] || { echo "FATAL: $SIZES missing -- run make_visit_sizes.py first" >&2; exit 1; }
export DELPHI_VISIT_SIZES="$SIZES"

TREE="$BASE/vb_evaltrees/$RUN"
rm -rf "$TREE"; mkdir -p "$TREE"
cp -r "$BASE/Delphi-2M" "$TREE/Delphi-2M"
rm -f "$TREE/Delphi-2M/data"; ln -s "$BASE/Delphi-2M/data" "$TREE/Delphi-2M/data"
cp "$CKPT" "$TREE/Delphi-2M/ckpt_under_test.pt"
cd "$TREE/Delphi-2M"
echo "=== VISIT-BATCHED eval $RUN on $(hostname), workers $WORKERS"

START=$(date +%s)
python -u figure2/figure2_core.py --build --ckpt ckpt_under_test.pt \
       --dataset nacc-dedup-s42 --device cpu --workers "$WORKERS"
python -u figure2/figure2_panels.py --cohort matched --ckpt ckpt_under_test.pt \
       --dataset nacc-dedup-s42
END=$(date +%s)

DEST="$BASE/results/vb_eval/$RUN"
mkdir -p "$DEST"
# harvest the WHOLE figure, not just the metrics: the panels, the combined figure in both
# raster and vector, the summary, and the per-panel source data (Nature "source data" practice)
cp -f results/figure2/metrics_matched.json     "$DEST/" 2>/dev/null || true
cp -f results/figure2/SUMMARY_matched.md       "$DEST/" 2>/dev/null || true
cp -f results/figure2/*_matched*.csv           "$DEST/" 2>/dev/null || true
cp -f results/figure2/fig2?_matched.png        "$DEST/" 2>/dev/null || true
cp -f results/figure2/figure2_combined_matched.png "$DEST/" 2>/dev/null || true
cp -f results/figure2/figure2_combined_matched.pdf "$DEST/" 2>/dev/null || true
echo "{\"run\":\"$RUN\",\"visit_batched\":true,\"wall_seconds\":$((END-START))}" > "$DEST/eval_meta.json"
echo "DONE $RUN in $(( (END-START)/60 )) min -> $DEST"
