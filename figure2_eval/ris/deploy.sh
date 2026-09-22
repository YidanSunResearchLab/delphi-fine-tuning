#!/usr/bin/env bash
# deploy.sh -- 只推代码到 RIS，然后可选提交。
#
#   ris/deploy.sh                 # 只同步代码
#   ris/deploy.sh --submit        # 同步后提交 figure2
#   JOB=pvo ris/deploy.sh --submit
#
# 排除列表照抄 delphi-fine-tuning 的 slurm/deploy.sh，理由也照抄：受限数据（.bin/.pt/.csv）
# **一次性刻意**放到集群的 0700 目录，不随代码推送同步，衍生物不回传。要把数据放到新路径，
# 单独做，不要靠放宽这里的排除列表顺手完成。见 ris/README.md。
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${AD_HOST:-compute2}"
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0915_rosmap_fig2}"

# rsync 不会递归创建多级目标目录（只建最后一级），首次部署会以 "No such file or directory" 失败。
ssh -o BatchMode=yes "$HOST" "mkdir -p '$BASE/figure2_eval' '$BASE/delphi'"

echo "=== 同步代码 -> $HOST:$BASE/figure2_eval"
rsync -az --delete \
  --exclude '.git/' --exclude 'data/' --exclude 'logs/' --exclude 'results/' \
  --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '*.bin' --exclude '*.pt' --exclude '*.pth' --exclude '*.xlsx' --exclude '*.csv' \
  --exclude '*.npy' --exclude '.DS_Store' \
  -e "ssh -o BatchMode=yes" ./ "$HOST:$BASE/figure2_eval/"

# engine 会 import ../delphi 的 model.py/utils.py，所以那两个也要（它们是代码，不是数据）
echo "=== 同步 delphi 的模型代码"
rsync -az -e "ssh -o BatchMode=yes" \
  ../delphi/model.py ../delphi/utils.py ../delphi/configurator.py \
  "$HOST:$BASE/delphi/"

ssh -o BatchMode=yes "$HOST" "mkdir -p $BASE/figure2_eval/logs $BASE/delphi/data/rosmap $BASE/delphi/Delphi-ROSMAP
  chmod +x $BASE/figure2_eval/ris/*.sh $BASE/figure2_eval/*.sh 2>/dev/null || true
  cd $BASE/figure2_eval && mkdir -p data
  for ds in \$(cd ../delphi/data && ls -d */ | tr -d /); do
    [ -e "data/\$ds" ] || ln -s ../../delphi/data/\$ds data/\$ds
  done
  echo '--- 集群上的数据 ---'
  ls -la $BASE/delphi/data/rosmap/ 2>/dev/null | tail -5
  ls -la $BASE/delphi/Delphi-ROSMAP/ckpt.pt 2>/dev/null || echo '  ckpt.pt 尚未上传'"

if [ "${1:-}" = "--submit" ]; then
  shift
  ssh -o BatchMode=yes "$HOST" "cd $BASE/figure2_eval && JOB=${JOB:-figure2} sbatch ris/figure2.sbatch $*"
fi
