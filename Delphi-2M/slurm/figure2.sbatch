#!/bin/bash
# Figure 2 (transitions / timing / individual trajectories / embeddings) on RIS compute2.
#
#   sbatch slurm/figure2.sbatch                 # full build + all four panels + combined
#   sbatch slurm/figure2.sbatch --limit 300     # quick smoke run on 300 patients
#
# The Monte-Carlo pass is CPU-multiprocess (one worker per core, torch single-threaded inside
# each), so this belongs on general-cpu, not the GPU queue. ~10-20 min on 32 cores.
#SBATCH --job-name=ad-figure2
#SBATCH --partition=general-cpu
#SBATCH --account=compute2-mdan
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/figure2_%j.out
#SBATCH --error=logs/figure2_%j.err
# c2-node-011 is broken (2026-07-28): every job scheduled onto it -- even `--wrap='hostname'` --
# dies instantly with signal 53 and writes no log at all. Slurm reports it idle, so it keeps
# attracting work. Drop this line once the node is fixed.
#SBATCH --exclude=c2-node-011

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
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1                      # each MC worker must stay single-threaded
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-32} - 2 ))

cd "$CODE"
echo "host: $(hostname) | workers=$WORKERS | extra args: ${*:-none}"

# ---------------------------------------------------------------- dependencies
python - <<'PYEOF'
import importlib.util, sys        # `import importlib` alone does NOT bind the .util submodule
missing = [m for m in ("torch", "numpy", "pandas", "sklearn", "matplotlib") if not importlib.util.find_spec(m)]
if missing:
    sys.exit(f"FATAL: missing required packages: {missing}")
print("deps ok" + ("" if importlib.util.find_spec("umap") else "  (umap-learn absent -> panel d will fail)"))
PYEOF

# ---------------------------------------------------------------------------- run
# Delegate to run_figure2.sh so the pipeline is defined in exactly one place. That script
# carries the checkpoint and dataset preflights; everything above is just SLURM plumbing.
#
#   sbatch slurm/figure2.sbatch                     # the delivered model
#   sbatch slurm/figure2.sbatch --limit 300         # smoke run
#   CKPT=out-delphi2m-strict-s42/ckpt.pt DATA=data/nacc-strict-s42 \
#       sbatch slurm/figure2.sbatch                 # a different model / cohort
#
# CKPT, DATA, DEVICE and EXPECT_EVENTS are inherited from the submitting environment.
WORKERS="$WORKERS" DEVICE="${DEVICE:-cpu}" ./figure2/run_figure2.sh "$@"
