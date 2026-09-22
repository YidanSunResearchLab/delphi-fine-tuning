#!/bin/bash
# 接法一：Delphi 隐状态进 lambda 的 GP 先验均值。Delphi 冻住。
# 每折三步：抽隐状态 -> 拟合 -> 打分。
# 提交：  sbatch --export=ALL,K=21 ris_delphi_hybrid.sbatch
#SBATCH --job-name=aladyn-dh
#SBATCH --account=compute2-mdan
#SBATCH --partition=general-cpu
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dh_%A_%a.out
#SBATCH --error=logs/dh_%A_%a.err

set -euo pipefail
cd /scratch2/fs1/mdan/huang.yi1/aladynoulli-rosmap
mkdir -p logs results
K=${K:-21}
F=$SLURM_ARRAY_TASK_ID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "host=$(hostname) job=$SLURM_JOB_ID fold=$F K=$K start=$(date -Is)"
[ -f results/dstates_f${F}.npz ] || .venv/bin/python -u delphi_states.py --fold "$F"
.venv/bin/python -u fit.py --k "$K" --steps 12000 --lam-steps 1000 \
    --folds 10 --only-fold "$F" --seed 0 --use-delphi
.venv/bin/python -u evaluate.py --k "$K" --fold "$F" --folds 10 --seed 0 --use-delphi
echo "end=$(date -Is)"
