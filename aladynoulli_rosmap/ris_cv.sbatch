#!/bin/bash
# K x 折 的完整网格：6 个 K 值 x 10 折 = 60 个任务。
# 每个任务 = 在 9 折上拟合 + 在留出的那 1 折上做 landmark 打分。
# 跑完用  pool_cv.py --k K  把 10 折池化成 n=4428 的 AUC 表。
# 提交：  sbatch ris_cv.sbatch
#SBATCH --job-name=aladyn-cv
#SBATCH --account=compute2-mdan
#SBATCH --partition=general-cpu
#SBATCH --array=0-59
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=logs/cv_%A_%a.out
#SBATCH --error=logs/cv_%A_%a.err

set -euo pipefail
cd /scratch2/fs1/mdan/huang.yi1/aladynoulli-rosmap
mkdir -p logs results

KS=(3 5 8 12 16 21)
K=${KS[$((SLURM_ARRAY_TASK_ID / 10))]}
FOLD=$((SLURM_ARRAY_TASK_ID % 10))
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID K=$K fold=$FOLD start=$(date -Is)"

.venv/bin/python fit.py --k "$K" --steps 12000 --lam-steps 1000 \
    --folds 10 --only-fold "$FOLD" --seed 0
.venv/bin/python evaluate.py --k "$K" --fold "$FOLD" --folds 10 --seed 0

echo "end=$(date -Is)"
