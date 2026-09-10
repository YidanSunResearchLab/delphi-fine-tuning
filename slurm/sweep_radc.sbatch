#!/bin/bash
# Capacity / regularisation sweep for the RADC model, scored by 5-fold CV.
#
#   sbatch slurm/sweep_radc.sbatch                     # full grid (15 cells x 5 folds)
#   sbatch slurm/sweep_radc.sbatch --shapes-only       # stage 1 only (5 cells x 5 folds)
#
#SBATCH --job-name=radc-sweep
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-mdan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/radc_sweep_%j.out
#SBATCH --error=logs/radc_sweep_%j.err
#SBATCH --exclude=c2-node-011
set -euo pipefail
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0910_radc}"
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate ad-projection
export MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1
cd "$BASE"; mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f data/radc-s42/train.bin ] || python -m radc_delphi.build_dataset --seed 42
python sweep.py --config configs/radc_base.py --device cuda "$@"
