#!/bin/bash
# eval_auc_controls.sbatch -- figure 3 panel A/B：evaluate_auc_rosmap_controls.py 跑在 RIS 上。
#
#   OUT=results/fig3/A sbatch ris/eval_auc_controls.sbatch \
#       --run nodedup-pos-3w:42:Delphi-ROSMAP-nodedup-pos-3w/ckpt.pt:rosmap_nodedup_3w
#
# 所有参数原样透传给脚本（--run 可重复；--post_entry 1 / --ckpt_name ckpt_10000.pt 做敏感性分析）。
# 先跑等价性测试：外层循环必须逐位复现上游，否则图上的数就不能叫 "Delphi AUC"。
# CPU 就够：每个 run 一次 forward，地板是几十个小 logistic。
#SBATCH --job-name=rosmap-fig3auc
#SBATCH --partition=general-cpu
#SBATCH --account=compute2-mdan
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fig3auc_%j.out
#SBATCH --error=logs/fig3auc_%j.err
#SBATCH --exclude=c2-node-011

set -euo pipefail
BASE="${AD_BASE:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0915_rosmap_fig2}"
CONDA_ROOT="${AD_CONDA:-/storage3/fs1/mdan/Active/secure-ad-comorbidity/huang.yi1/AD0718/miniforge3}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate ad-projection
export MKL_THREADING_LAYER=GNU KMP_DUPLICATE_LIB_OK=TRUE

cd "$BASE/delphi"
mkdir -p logs
OUT="${OUT:?OUT=<输出目录> 必须给}"
echo "host: $(hostname) | out=$OUT | args: $*"

python -u test_auc_controls_equiv.py Delphi-ROSMAP-gpubase/ckpt.pt rosmap
python -u test_scope_mask.py
python -u evaluate_auc_rosmap_controls.py --out "$OUT" "$@"
