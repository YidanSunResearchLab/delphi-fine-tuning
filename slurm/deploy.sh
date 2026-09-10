#!/usr/bin/env bash
# deploy.sh -- push the code (and ONLY the code) to the RIS checkout, then optionally submit.
#
#   ./slurm/deploy.sh                       # sync code
#   ./slurm/deploy.sh --submit               # sync, then sbatch the training job
#   ./slurm/deploy.sh --submit configs/radc_small.py
#   AD_BASE=/other/checkout ./slurm/deploy.sh
#
# WHAT IT REFUSES TO SEND. Everything under data/, every .bin, .pt, .xlsx and the pidmap. The
# raw RADC files and the projid->pid linkage live on the cluster already, uploaded once into
# a 0700 directory; they are not re-synced on every code push, and nothing derived from them
# ever travels back. If you need the data on a NEW cluster path, copy it deliberately -- do
# not relax this exclude list to do it by accident.
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${AD_HOST:-compute2}"
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0910_radc}"

SUBMIT=0
CONFIG="configs/radc_base.py"
while [ $# -gt 0 ]; do
  case "$1" in
    --submit) SUBMIT=1; shift ;;
    *) CONFIG="$1"; shift ;;
  esac
done

echo "=== syncing code -> $HOST:$BASE"
rsync -az --delete \
  --exclude '.git/' \
  --exclude 'data/' \
  --exclude 'logs/' \
  --exclude 'out-*/' \
  --exclude 'results/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.bin' \
  --exclude '*.pt' \
  --exclude '*.pth' \
  --exclude '*.xlsx' \
  --exclude '*.csv' \
  --exclude '.DS_Store' \
  --exclude '.wandb.env' \
  -e "ssh -o BatchMode=yes" \
  ./ "$HOST:$BASE/"

ssh -o BatchMode=yes "$HOST" "mkdir -p $BASE/logs && chmod +x $BASE/slurm/*.sbatch $BASE/slurm/*.sh 2>/dev/null; \
  echo '--- data on cluster ---'; ls -la $BASE/data/RADC/ 2>/dev/null | tail -6"

if [ "$SUBMIT" = "1" ]; then
  echo "=== submitting $CONFIG"
  ssh -o BatchMode=yes "$HOST" "cd $BASE && sbatch slurm/train_radc.sbatch $CONFIG"
fi
