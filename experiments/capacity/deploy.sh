#!/usr/bin/env bash
# deploy.sh -- push the CURRENT code + this experiment to RIS and preflight it there.
#
#   ./experiments/capacity/deploy.sh              # sync + preflight
#   ./experiments/capacity/deploy.sh --dry-run    # show what would transfer
#
# WHY A NEW DIRECTORY: the existing cluster checkout (AD0718) is the OLD flat layout --
# Delphi-2M/model.py, Delphi-2M/eval_ad.py -- from before the restructure into the
# delphi/ training/ data_prep/ figure2/ packages. Submitting this sweep against it would
# fail on import, and overwriting it would destroy the provenance of the delivered model.
# So we deploy alongside it, read-only against its data.
#
# NO PHI MOVES. data/ is a SYMLINK to the split that already lives on the cluster; the raw
# CSV and the .bin files are never transferred in either direction.
set -euo pipefail

LOCAL_REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HOST="${AD_HOST:-compute2}"
OLD="/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718"
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0820_capacity}"
DRY=""; [ "${1:-}" = "--dry-run" ] && DRY="--dry-run"

echo "=== deploy"
echo "    from  $LOCAL_REPO"
echo "    to    $HOST:$BASE"
echo "    data  symlink -> $OLD/Delphi-2M/data  (not copied)"
echo

ssh "$HOST" "mkdir -p '$BASE/logs' '$BASE/runs' '$BASE/results'"

# code only. Excludes cover everything regenerable, everything large, and all PHI.
rsync -az --delete $DRY \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '*.pt' --exclude '*.pth' --exclude '*.bin' --exclude '*.csv' \
  --exclude 'investigator_nacc*' --exclude '.DS_Store' \
  --exclude 'data/' --exclude 'out_ad/' --exclude 'out-*/' \
  --exclude 'runs/' --exclude 'logs/' --exclude 'results/figure2/_cache/' \
  --include 'Delphi-2M/data_prep/Vocabulary NACC.xlsx' \
  "$LOCAL_REPO/Delphi-2M" "$LOCAL_REPO/experiments" "$LOCAL_REPO/environment.yml" \
  "$HOST:$BASE/"

[ -n "$DRY" ] && { echo "(dry run -- stopping here)"; exit 0; }

# data/ symlink -> the split that is already on the cluster and already fingerprint-verified
ssh "$HOST" "set -e
  cd '$BASE/Delphi-2M'
  [ -L data ] || [ ! -e data ] || { echo 'FATAL: real data/ exists, refusing to replace' >&2; exit 1; }
  rm -f data
  ln -s '$OLD/Delphi-2M/data' data
  echo '    data -> ' \$(readlink data)
  ls data/"

echo
echo "=== remote preflight"
ssh "$HOST" "source '$OLD/miniforge3/etc/profile.d/conda.sh' && conda activate ad-projection && \
  cd '$BASE' && python experiments/capacity/verify.py --data-root '$BASE/Delphi-2M/data'"

echo
echo "=== deployed. next:"
echo "  ssh $HOST"
echo "  cd $BASE && sbatch --array=1-18%9 experiments/capacity/slurm/train_sweep.sbatch"
