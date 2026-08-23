#!/bin/bash
# grad_balance -- one short CPU job, no training. Measures which of the two objectives
# actually supplies the gradient, and how much of loss_dt's gradient the t_min floor lets
# through, on the REAL split. See grad_balance.py's header for why the "loss_dt is 73% of
# the loss" figure cannot answer that.
#
#   sbatch experiments/time_head/slurm/grad_balance.sbatch
#
# ~20 batches x (1 forward + 2 backwards) per checkpoint on a 1.4-2.1M model: seconds, not
# minutes. 8 cores is plenty; the job exists rather than running on the login node.
#SBATCH --job-name=grad-bal
#SBATCH --partition=general-cpu
#SBATCH --account=compute2-mdan
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/gradbal_%j.out
#SBATCH --error=logs/gradbal_%j.err
#SBATCH --exclude=c2-node-011

set -euo pipefail

BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0820_capacity}"
EXP="$BASE/experiments/time_head"
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
# the recommended shape at all three seeds, plus the delivered architecture for reference
TAGS="${TAGS:-L8E120H6_s42,L8E120H6_s43,L8E120H6_s44,L12E120H12_s42}"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ad-projection
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$BASE/Delphi-2M"
[ -e data/nacc-dedup-s42/train.bin ] || { echo "FATAL: no train.bin under $PWD/data" >&2; exit 1; }

echo "host $(hostname) | threads $OMP_NUM_THREADS | tags $TAGS"
mkdir -p "$BASE/results/time_head"

python -u "$EXP/grad_balance.py" \
  --runs "$BASE/runs" --tags "$TAGS" \
  --data-root "$BASE/Delphi-2M/data" --dataset nacc-dedup-s42 \
  --batches 20 --batch-size 128 \
  --out "$BASE/results/time_head/grad_balance.csv"

echo "DONE"
