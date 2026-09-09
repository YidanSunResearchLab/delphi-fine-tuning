#!/bin/bash
# Delphi-2M dedup(keep-transitions) + mask 0-21. Fills the 2x2 gap vs the baseline
# out-delphi2m-dedup-nomask-s42 (same nacc-dedup-s42 dedup data, same schedule) -- the ONLY change is
# ignore_tokens=0..21 (set in the config).
#SBATCH --job-name=ad-delphi2m-dedup-mask-s42
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-mdan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/delphi2m_mask_dedup_%j.out
#SBATCH --error=logs/delphi2m_mask_dedup_%j.err

set -euo pipefail

# Checkout root. Override without editing this file:  AD_BASE=/my/checkout sbatch <script>
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718}"
CODE="$BASE/Delphi-2M"

# conda root. Defaults to a miniforge3 INSIDE the checkout (the original layout); set
# AD_CONDA if yours lives elsewhere, e.g. AD_CONDA=/other/checkout/miniforge3
CONDA_ROOT="${AD_CONDA:-$BASE/miniforge3}"
[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ] || {
  echo "FATAL: no conda at $CONDA_ROOT" >&2
  echo "  set AD_CONDA=/path/to/miniforge3 (the dir containing etc/profile.d/conda.sh)" >&2
  exit 1
}
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ad-projection
export MKL_THREADING_LAYER=GNU

# wandb is OPTIONAL -- training does not need it. Enable by putting WANDB_API_KEY into
# $BASE/.wandb.env (chmod 600), or by exporting it before sbatch. Absent, the run just
# proceeds with logging off instead of failing.
WANDB_FLAGS=""
[ -f "$BASE/.wandb.env" ] && source "$BASE/.wandb.env"
if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_MODE=online
  WANDB_FLAGS="--wandb_log=True --wandb_run_name=delphi2m-dedup-mask-s42"
  echo "wandb: enabled"
else
  echo "wandb: disabled (no WANDB_API_KEY -- put one in $BASE/.wandb.env to enable)"
fi

cd "$CODE"
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# The config is the one thing that changes between training runs, so take it as an argument:
#   sbatch slurm/train_delphi2m_mask_dedup.sbatch                              # delivered model
#   COHORT=strict sbatch slurm/train_delphi2m_mask_dedup.sbatch config/train_cohort_sweep.py
CFG="${1:-config/train_delphi2m_mask_dedup.py}"
echo "config: $CFG"
python -u training/train.py "$CFG" --device=cuda $WANDB_FLAGS

echo "DONE. checkpoint -> the out_dir set in $CFG"
