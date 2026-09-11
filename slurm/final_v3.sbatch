#!/bin/bash
# The v3 final run: train the swept architecture, audit it, then open the test split ONCE.
#   ARCH="--n_layer 4 --n_embd 64 --n_head 4 --dropout 0.1" sbatch slurm/final_v3.sbatch
#SBATCH --job-name=radc-v3-final
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-mdan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/v3_final_%j.out
#SBATCH --error=logs/v3_final_%j.err
#SBATCH --exclude=c2-node-011
set -euo pipefail
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0910_radc}"
source /storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
cd "$BASE"; mkdir -p logs results/figures
nvidia-smi --query-gpu=name --format=csv,noheader
OUT="${OUT:-out-radc-v3-final}"
echo "===== 1/5 TRAIN  (${ARCH:-config defaults})"
python train.py configs/radc_v3.py --device=cuda --out_dir "$OUT" ${ARCH:-}
echo; echo "===== 2/5 REGRESSION TESTS"
python tests/test_pipeline.py --data-dir data/radc-v3-s42 2>&1 | tail -3
echo; echo "===== 3/5 BASELINES (leakage ceiling only)"
python -m eval.baselines --split test 2>&1 | tail -30 || echo "[warn] baselines failed"
echo; echo "===== 4/5 LEAKAGE AUDIT"
python -m eval.leakage --ckpt "$OUT/ckpt.pt" --split test --n-mc 200 2>&1 | tail -25 || echo "[warn] leakage FAILED"
echo; echo "===== 5/5 ENDPOINTS ON TEST (opened once)"
python eval/evaluate.py --ckpt "$OUT/ckpt.pt" --split test --n-mc 200
echo "===== DONE -> $OUT/ckpt.pt"
