#!/bin/bash
# 对照：冻住的 Delphi 隐状态 + 扁平逐年 hazard 头（无 signature 结构）。
#SBATCH --job-name=delphi-flat
#SBATCH --account=compute2-mdan
#SBATCH --partition=general-cpu
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/flat_%A_%a.out
#SBATCH --error=logs/flat_%A_%a.err
set -euo pipefail
cd /scratch2/fs1/mdan/huang.yi1/aladynoulli-rosmap
mkdir -p logs results
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "host=$(hostname) fold=$SLURM_ARRAY_TASK_ID start=$(date -Is)"
.venv/bin/python -u flat_head.py --fold "$SLURM_ARRAY_TASK_ID"
echo "end=$(date -Is)"
