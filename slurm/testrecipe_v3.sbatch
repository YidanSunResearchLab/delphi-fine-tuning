#!/bin/bash
# out-radc-testckpt's RECIPE, on the v3 tokenization -- so the two can be compared.
#
# WHY. out-radc-testckpt is the one checkpoint here whose rollouts still die (never-dies 2.8%
# against out-radc-final's 89.0%), so it is the only one whose simulated survival is usable.
# It cannot be compared with the delivered v3 model because THREE things differ at once: the
# tokenization (50 tokens, no `Dementia at entry` block, MMSE in four bins), the shape
# (3L/48d vs 4L/64d) and the schedule (800 iterations vs 6000). This run holds the
# tokenization at v3 and reproduces the other two, so the result differs from
# out-radc-v3-final in the RECIPE ALONE.
#
# max_iters IS the variable, and it is NOT the same as early-stopping the long run.
# experiments/death_collapse.py records why: lr_decay_iters tracks max_iters, so "iteration
# 800 of a 6000-step cosine" is a different optimiser state from "iteration 800 of an 800-step
# cosine" -- the second has annealed to the lr floor, the first is still near peak. Reading an
# intermediate checkpoint of the 6000-step run would therefore not reproduce this recipe.
#
# Expect a WORSE validation loss and worse AD discrimination. That is the trade being measured,
# not a defect of the run.
#SBATCH --job-name=ad-testrecipe-v3
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-mdan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/testrecipe_v3_%j.out
#SBATCH --error=logs/testrecipe_v3_%j.err
#SBATCH --exclude=c2-node-011

set -euo pipefail
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0910_radc}"
source /storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3/etc/profile.d/conda.sh
conda activate ad-projection
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE
cd "$BASE"
echo "host: $(hostname)"

# The overrides ARE the experiment; keeping them on the command line rather than in a derived
# config file means the diff from configs/radc_v3.py is visible in the log.
python train.py configs/radc_v3.py \
  --n_layer 3 --n_embd 48 --dropout 0.2 \
  --max_iters 800 --lr_decay_iters 800 --eval_interval 50 \
  --out_dir out-radc-v3-testrecipe \
  --device cuda
