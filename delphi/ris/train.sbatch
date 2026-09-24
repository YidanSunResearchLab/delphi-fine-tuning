#!/bin/bash
# train.sbatch -- 在 RIS compute2 的 GPU 上训练 Delphi-ROSMAP。
#
#   CFG=config/train_delphi_rosmap.py         sbatch ris/train.sbatch   # 交付版（认知量表去重）
#   CFG=config/train_delphi_rosmap_nodedup.py sbatch ris/train.sbatch   # 认知量表不去重
#
# general-gpu 排不上队时（实测优先级可能只有 1，START_TIME 估到两天后）改投可抢占分区：
#   sbatch -p general-preempt-gpu ris/train.sbatch
# 这个任务只要 ~15 分钟，被抢占的代价就是重投一次；但注意 always_save_checkpoint=False，
# 被抢占等于从头开始，**不要**用它跑长任务。
#
# 为什么用 GPU 而不是 compute2 的 general-cpu：模型只有 0.69M 参数，但 12,000 iter ×
# batch 128 × block 144 在 32 核 CPU 上实测量级是 ~5 s/iter（16 小时）。GPU 上是分钟级。
#
# 为什么 dtype 保持 float32（train.py 的默认值，这里不覆盖）：交付版 ckpt 是在 Mac/MPS 上以
# float32 训的。float32 + autocast(float32) 在 CUDA 上等于不降精度，且 torch 默认
# allow_tf32=False，所以两个平台走的是同一个数值 regime。**不要**为了快而改成 bfloat16：
# 那会让"去重 vs 不去重"的差异里混进一个精度变量。
#SBATCH --job-name=rosmap-train
#SBATCH --partition=general-gpu
#SBATCH --account=compute2-mdan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0915_rosmap_fig2}"
CODE="$BASE/delphi"
CFG="${CFG:-config/train_delphi_rosmap.py}"

CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ] || { echo "FATAL: $CONDA_ROOT 下没有 conda" >&2; exit 1; }
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ad-projection
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE

cd "$CODE"
mkdir -p logs
echo "host: $(hostname) | cfg=$CFG | extra: ${*:-none}"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

# 配置里的 dataset 决定读哪份 .bin；缺了就直接停，不要让 train.py 拿着空 memmap 往下跑。
DS=$(python - "$CFG" <<'PYEOF'
import re, sys
src = open(sys.argv[1]).read()
m = re.search(r"^dataset\s*=\s*['\"]([^'\"]+)['\"]", src, re.M)
print(m.group(1) if m else "")
PYEOF
)
[ -n "$DS" ] || { echo "FATAL: 从 $CFG 里读不出 dataset" >&2; exit 1; }
for f in "data/$DS/train.bin" "data/$DS/val.bin" "data/$DS/labels.csv"; do
  [ -e "$f" ] || { echo "FATAL: 缺 $f —— 数据没上传（受限数据要单独 scp，不进 deploy 的 rsync）" >&2; exit 1; }
done
echo "dataset=$DS  $(python -c "
import numpy as np
for s in ('train','val'):
    d = np.fromfile('data/$DS/%s.bin' % s, dtype=np.uint32).reshape(-1,3)
    print('%s %d 行/%d 人' % (s, len(d), len(np.unique(d[:,0]))), end='  ')")"

python -u train.py "$CFG" --device=cuda "$@"
