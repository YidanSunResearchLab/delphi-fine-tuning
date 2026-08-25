#!/bin/bash
#SBATCH --job-name=dt-ablation
#SBATCH --partition=general-cpu
#SBATCH --account=compute2-mdan
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dtabl_%j.out
#SBATCH --error=logs/dtabl_%j.err
#SBATCH --exclude=c2-node-011
set -euo pipefail
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0820_capacity}"
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate ad-projection
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd "$BASE/Delphi-2M"
mkdir -p "$BASE/results/time_head"
python -u "$BASE/experiments/time_head/dt_ablation.py" \
  --data-root "$BASE/Delphi-2M/data" --dataset nacc-dedup-s42 \
  --batches 30 --batch-size 128 \
  --out "$BASE/results/time_head/dt_ablation.csv"
echo DONE
