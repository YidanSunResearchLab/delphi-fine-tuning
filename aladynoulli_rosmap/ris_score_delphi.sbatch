#!/bin/bash
# 用和 ALADYNOULLI 相同的 landmark 协议给 10 折的 Delphi 打分。
# 提交：  sbatch ris_score_delphi.sbatch
#SBATCH --job-name=delphi-score
#SBATCH --account=compute2-mdan
#SBATCH --partition=general-cpu
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/dscore_%A_%a.out
#SBATCH --error=logs/dscore_%A_%a.err

set -euo pipefail
cd /scratch2/fs1/mdan/huang.yi1/aladynoulli-rosmap
mkdir -p logs results
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "host=$(hostname) job=$SLURM_JOB_ID fold=$SLURM_ARRAY_TASK_ID start=$(date -Is)"
.venv/bin/python -u score_delphi.py --fold "$SLURM_ARRAY_TASK_ID" --n-mc 500
echo "end=$(date -Is)"
