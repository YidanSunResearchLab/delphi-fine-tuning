#!/bin/bash
# 在和 ALADYNOULLI 完全相同的 10 折上训练 Delphi，用于头对头比较。
# 提交：  sbatch ris_delphi_folds.sbatch
#SBATCH --job-name=delphi-folds
#SBATCH --account=compute2-mdan
#SBATCH --partition=general-cpu
#SBATCH --array=0-9
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dfold_%A_%a.out
#SBATCH --error=logs/dfold_%A_%a.err

set -euo pipefail
ROOT=/scratch2/fs1/mdan/huang.yi1/aladynoulli-rosmap
cd $ROOT/_delphi
mkdir -p $ROOT/logs

F=$SLURM_ARRAY_TASK_ID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "host=$(hostname) job=$SLURM_JOB_ID fold=$F start=$(date -Is)"

# 配置整体沿用 config/train_delphi_rosmap.py，只换数据集和输出目录。
$ROOT/.venv/bin/python train.py config/train_delphi_rosmap.py \
    --device=cpu \
    --dataset=rosmap_f${F} \
    --out_dir=Delphi-fold${F}

echo "end=$(date -Is)"
