#!/usr/bin/env bash
# deploy.sh -- 把 tokenizer 代码推到 RIS，可选一并上传原始 RADC 表格。
#
#   ris/deploy.sh                 # 只同步 build.py / spec.py / ris/
#   ris/deploy.sh --raw           # 外加上传三份原始表格（受限数据，显式请求才传）
#   ris/deploy.sh --raw --submit -- --out data_rosmap_fullvisit --dataset rosmap_fullvisit \
#                                    --nodedup all --no-global-dedup
#
# 为什么原始表格要单独一个开关：它们是**受限的参与者级数据**，去向是集群上
# $AD_BASE/raw/（0700，和已经在那里的派生 .bin 同一个安全目录）。放宽 rsync 的排除列表
# 顺手带上去 = 以后每次推代码都会同步受限数据，这不是同一件事。
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${AD_HOST:-compute2}"
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0915_rosmap_fig2}"
SRC="$(cd .. && pwd)"

RAW=0; SUBMIT=0; PASS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --raw) RAW=1; shift ;;
    --submit) SUBMIT=1; shift ;;
    --) shift; PASS=("$@"); break ;;
    *) shift ;;
  esac
done

ssh -o BatchMode=yes "$HOST" "mkdir -p '$BASE/tokenization/logs' '$BASE/raw'"
echo "=== 同步 tokenizer 代码"
rsync -az --exclude '__pycache__/' --exclude '.DS_Store' -e "ssh -o BatchMode=yes" \
  build.py spec.py README.md test_snapshot_first_occurrence.py "$HOST:$BASE/tokenization/"
rsync -az --exclude '__pycache__/' -e "ssh -o BatchMode=yes" ris/ "$HOST:$BASE/tokenization/ris/"
ssh -o BatchMode=yes "$HOST" "chmod +x $BASE/tokenization/ris/*.sh 2>/dev/null || true"

if [ "$RAW" = "1" ]; then
  echo "=== 上传原始 RADC 表格 -> $HOST:$BASE/raw/ （受限数据，显式请求）"
  ssh -o BatchMode=yes "$HOST" "chmod 700 '$BASE/raw'"
  rsync -az -e "ssh -o BatchMode=yes" \
    "$SRC/cross-sectional-data-gk.xlsx" "$SRC/longitudinal_data_gk.xlsx" \
    "$SRC/ROSMAP_clinical.csv" "$HOST:$BASE/raw/"
  ssh -o BatchMode=yes "$HOST" "chmod 600 '$BASE/raw'/*; ls -la '$BASE/raw'"
fi

if [ "$SUBMIT" = "1" ]; then
  ssh -o BatchMode=yes "$HOST" "cd $BASE/tokenization && sbatch ris/build.sbatch ${PASS[*]}"
fi
