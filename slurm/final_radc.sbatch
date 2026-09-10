#!/bin/bash
# The final RADC run: train the swept architecture, then open the test split ONCE.
#
#   sbatch slurm/final_radc.sbatch --n_layer 4 --n_embd 64 --n_head 4 --dropout 0.2
#
# Everything after the script name is forwarded to train.py, which is how the architecture
# chosen by sweep.py gets in without editing a config file mid-experiment.
#
# THE TEST SPLIT IS OPENED HERE AND NOWHERE ELSE. Architecture, hysteresis widths and every
# hyperparameter must already be frozen by the cross-validated sweep before this runs. If you
# find yourself re-running this with a different --n_layer, you are tuning on test.
#
#SBATCH --job-name=radc-final
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-mdan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/radc_final_%j.out
#SBATCH --error=logs/radc_final_%j.err
#SBATCH --exclude=c2-node-011
set -euo pipefail
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0910_radc}"
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate ad-projection
export MKL_THREADING_LAYER=GNU PYTHONUNBUFFERED=1
cd "$BASE"; mkdir -p logs
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

OUT="${OUT:-out-radc-final}"
NMC="${NMC:-200}"

echo "===== 1/5  TRAIN  (out_dir=$OUT, extra: $*)"
python train.py configs/radc_base.py --device=cuda --out_dir "$OUT" "$@"

echo
echo "===== 2/5  REGRESSION TESTS"
python tests/test_pipeline.py

echo
echo "===== 3/5  BASELINES  (the bar the model has to clear)"
python eval/baselines.py --split test || echo "[warn] baselines failed"

echo
echo "===== 4/5  LEAKAGE AUDIT  (must pass before any number is believed)"
python eval/leakage.py --ckpt "$OUT/ckpt.pt" --split test || echo "[warn] leakage audit FAILED"

echo
echo "===== 5/5  ENDPOINTS ON THE TEST SPLIT  (opened once)"
python eval/evaluate.py --ckpt "$OUT/ckpt.pt" --split test --n-mc "$NMC"

echo
echo "===== DONE -> $OUT/ckpt.pt"
