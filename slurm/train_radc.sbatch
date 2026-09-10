#!/bin/bash
# Train the RADC Delphi model on WashU RIS compute2.
#
#   sbatch slurm/train_radc.sbatch                          # configs/radc_base.py
#   sbatch slurm/train_radc.sbatch configs/radc_small.py    # a different config
#   sbatch slurm/train_radc.sbatch configs/radc_base.py --max_iters 4000
#
# Submit from the checkout root so SLURM's relative logs/ path resolves. Override the checkout
# without editing this file:  AD_BASE=/my/checkout sbatch slurm/train_radc.sbatch
#
#SBATCH --job-name=radc-delphi
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-mdan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/radc_train_%j.out
#SBATCH --error=logs/radc_train_%j.err
# c2-node-011 was dead as of 2026-07-28 -- jobs scheduled onto it died with signal 53 and wrote
# no log at all, while Slurm kept reporting it idle. Drop this once it is confirmed fixed.
#SBATCH --exclude=c2-node-011

set -euo pipefail

BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0910_radc}"
# The conda env is shared with the older checkouts rather than duplicated -- it is ~6 GB.
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ] || {
  echo "FATAL: no conda at $CONDA_ROOT" >&2
  echo "  set AD_CONDA=/path/to/miniforge3 (the dir containing etc/profile.d/conda.sh)" >&2
  exit 1
}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ad-projection
export MKL_THREADING_LAYER=GNU
export PYTHONUNBUFFERED=1

cd "$BASE"
mkdir -p logs

CONFIG="${1:-configs/radc_base.py}"
shift || true

# wandb is OPTIONAL. Put WANDB_API_KEY in $BASE/.wandb.env (chmod 600) to enable it; absent,
# the run just proceeds with logging off instead of failing.
WANDB_ARG="--wandb disabled"
[ -f "$BASE/.wandb.env" ] && source "$BASE/.wandb.env"
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARG="--wandb online"
fi

echo "=== RADC Delphi training"
echo "    host    : $(hostname)"
echo "    base    : $BASE"
echo "    config  : $CONFIG"
echo "    extra   : ${*:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Build the dataset if it is not there yet. The tokenizer is deterministic and cheap (~1 min),
# and building on the node means the .bin is always in step with the tokenizer in this checkout
# rather than with whatever was rsync'd last.
if [ ! -f "$BASE/data/radc-s42/train.bin" ]; then
  echo "--- dataset not built; running the tokenizer"
  python -m radc_delphi.build_dataset --seed 42
fi

python train.py "$CONFIG" --device=cuda $WANDB_ARG "$@"

echo "=== DONE"
