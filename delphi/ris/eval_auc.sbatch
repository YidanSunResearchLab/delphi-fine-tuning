#!/bin/bash
# eval_auc.sbatch -- evaluate_auc_rosmap.py（上游 Delphi 的 AUC 口径）跑在 RIS compute2 上。
#
#   CKPT=Delphi-ROSMAP-gpubase/ckpt.pt  DS=rosmap          sbatch ris/eval_auc.sbatch
#   CKPT=Delphi-ROSMAP-nodedup/ckpt.pt  DS=rosmap_nodedup  sbatch ris/eval_auc.sbatch
#
# CKPT 和 DS 必须配对给。两份分词的 vocab_size 和 ignore_tokens 完全相同，所以配错不会崩，
# 只会给出一张安静的错表。输出落在 <ckpt 所在目录>/auc/。
# CPU 就够：整个评估是对 val 全量做**一次** forward，然后 DeLong 算方差。
#SBATCH --job-name=rosmap-auc
#SBATCH --partition=general-cpu
#SBATCH --account=compute2-mdan
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/auc_%j.out
#SBATCH --error=logs/auc_%j.err
#SBATCH --exclude=c2-node-011

set -euo pipefail
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0915_rosmap_fig2}"
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ad-projection
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE

cd "$BASE/delphi"
mkdir -p logs
CKPT="${CKPT:-Delphi-ROSMAP/ckpt.pt}"
DS="${DS:-rosmap}"
OUT="$(dirname "$CKPT")/auc"
echo "host: $(hostname) | ckpt=$CKPT | dataset=$DS | out=$OUT"
[ -e "$CKPT" ] || { echo "FATAL: 缺 $CKPT" >&2; exit 1; }
[ -e "data/$DS/train.bin" ] || { echo "FATAL: 缺 data/$DS/train.bin" >&2; exit 1; }

# offset=0.1 是上游叫 "no gap" 的设置；本数据是年度网格，同龄 token 被 offset 排除，所以它
# **实际已经是提前 1 年**，offset=365.25 对应提前 2 年（见 ../tokenization/README.md）。
for OFF in 0.1 365.25; do
  echo "=== offset=$OFF"
  python -u evaluate_auc_rosmap.py --ckpt "$CKPT" --data_dir "data/$DS" \
    --output_path "$OUT" --offset "$OFF" --device cpu "$@"
done
