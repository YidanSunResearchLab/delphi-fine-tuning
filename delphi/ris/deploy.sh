#!/usr/bin/env bash
# deploy.sh -- 把 delphi 的**训练**代码推到 RIS，然后可选提交训练。
#
#   ris/deploy.sh                                              # 只同步代码
#   CFG=config/train_delphi_rosmap_nodedup.py ris/deploy.sh --submit
#   ris/deploy.sh --data rosmap_nodedup                        # 额外上传一份受限数据（见下）
#
# 排除列表与理由照抄 ../figure2_eval/ris/deploy.sh：受限数据（.bin/.pt/.csv/.xlsx）**一次性
# 刻意**放到集群的 0700 目录，不随代码同步，衍生物不回传。要放新的数据集就显式 --data <name>，
# 不要靠放宽排除列表顺手完成。
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${AD_HOST:-compute2}"
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0915_rosmap_fig2}"

DATASETS=()
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --data) DATASETS+=("$2"); shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

ssh -o BatchMode=yes "$HOST" "mkdir -p '$BASE/delphi/logs'"

echo "=== 同步训练代码 -> $HOST:$BASE/delphi"
rsync -az \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' \
  -e "ssh -o BatchMode=yes" \
  train.py model.py utils.py configurator.py evaluate_auc.py evaluate_auc_rosmap.py \
  "$HOST:$BASE/delphi/"
rsync -az --delete --exclude '__pycache__/' -e "ssh -o BatchMode=yes" \
  config/ "$HOST:$BASE/delphi/config/"
rsync -az --exclude '__pycache__/' -e "ssh -o BatchMode=yes" \
  ris/ "$HOST:$BASE/delphi/ris/"
ssh -o BatchMode=yes "$HOST" "chmod +x $BASE/delphi/ris/*.sh 2>/dev/null || true"

for ds in ${DATASETS[@]+"${DATASETS[@]}"}; do
  echo "=== 上传受限数据 data/$ds （显式请求）"
  [ -d "data/$ds" ] || { echo "FATAL: 本地没有 data/$ds" >&2; exit 1; }
  ssh -o BatchMode=yes "$HOST" "mkdir -p '$BASE/delphi/data/$ds' && chmod 700 '$BASE/delphi/data/$ds'"
  # macOS 自带的 rsync 不认 --chmod=F600，所以权限在传完之后用 ssh 收紧（目录本身已是 0700）
  rsync -az -e "ssh -o BatchMode=yes" \
    "data/$ds/train.bin" "data/$ds/val.bin" "data/$ds/labels.csv" "data/$ds/meta.json" \
    "$HOST:$BASE/delphi/data/$ds/"
  ssh -o BatchMode=yes "$HOST" "chmod 600 '$BASE/delphi/data/$ds'/*"
done

ssh -o BatchMode=yes "$HOST" "cd $BASE/delphi && echo '--- 集群上的数据 ---' && ls -d data/*/ && \
  for d in data/*/; do echo \"  \$d \$(ls \$d | tr '\n' ' ')\"; done"

if [ "${ARGS[0]:-}" = "--submit" ]; then
  ssh -o BatchMode=yes "$HOST" "cd $BASE/delphi && CFG=${CFG:-config/train_delphi_rosmap.py} sbatch ris/train.sbatch"
fi
